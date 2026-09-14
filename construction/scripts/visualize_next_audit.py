#!/usr/bin/env python3
"""Generate audit-first visualizations for NavBench3D-Next view bundles.

What it visualizes on-image (per routing candidate):
- point-mass walkable / blocked candidates
- embodied feasible / blocked candidates
- start / goal anchors
- target object mask (from object_index + object_index_map)
- path_world projection
- path_wp_ids_pointmass polyline
- path_wp_ids_embodied polyline

Usage example:
  conda run -n navbench3d python scripts/visualize_next_audit.py \
    --renders-dir /mnt/data/zhaoyang/navbench3d-data/renders_smoke_fact_v1 \
    --output-dir /mnt/data/zhaoyang/navbench3d-data/audit_next_smoke_fact_v1 \
    --scene-list /tmp/navbench_gtnext_smoke_scenes.txt --limit 20
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from object_index_utils import decode_object_index_array, load_object_index_mapping, max_pass_id_from_mapping
from overlay_render_utils import (
    clamp_marker_center,
    compute_label_layout,
    compute_leader_line_widths,
    compute_marker_style,
    measure_text,
)
from task_outputs_utils import load_scene_classification_views

VIEW_BUNDLE_FILENAME = "next_view_bundle_v3.json"
NON_BLOCKING_CATEGORIES = {"wall", "ceiling", "window", "door", "floor", "carpet", "rug", "mat"}
NAVIGATION_AXIS_ORDER = ("reference_axis", "geometry_axis", "embodiment_axis")
AFFORDANCE_AXIS_ORDER = ("clutter_axis", "boundary_axis", "embodiment_gap_axis")


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for fp in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        p = Path(fp)
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def load_font_cached(cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont], size: int):
    if size not in cache:
        cache[size] = load_font(size)
    return cache[size]


def require_dict(payload: object, *, context: str) -> dict:
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be a dict")
    return payload


def require_text(payload: dict, key: str, *, context: str) -> str:
    value = payload.get(key)
    if value is None:
        raise ValueError(f"{context} missing required field: {key}")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{context} missing required field: {key}")
    return text


def format_axis_summary(axis_payload: object, *, axis_order: tuple[str, ...], context: str) -> str:
    payload = require_dict(axis_payload, context=context)
    parts = []
    for axis_name in axis_order:
        value = payload.get(axis_name)
        if value is None:
            raise ValueError(f"{context} missing required field: {axis_name}")
        parts.append(f"{axis_name.replace('_axis', '')}={value}")
    return ", ".join(parts)


def format_validity_summary(validity_payload: object, *, context: str) -> str:
    payload = require_dict(validity_payload, context=context)
    failed_checks = payload.get("failed_checks")
    if failed_checks is None:
        raise ValueError(f"{context} missing required field: failed_checks")
    if not isinstance(failed_checks, list):
        raise ValueError(f"{context}.failed_checks must be a list")
    if failed_checks:
        return "ineligible:" + ",".join(str(item) for item in failed_checks)
    return "eligible"


def require_routing_path_list(candidate: dict, key: str) -> list:
    value = candidate.get(key)
    if not isinstance(value, list) or len(value) < 2:
        routing_id = candidate.get("routing_id", "routing_unknown")
        if key == "optimal_path_image_xy":
            raise ValueError(
                f"scene routing={routing_id} missing Blender optimal path image coordinates"
            )
        if key == "canonical_sparse_path_image_xy":
            raise ValueError(
                f"scene routing={routing_id} missing Blender canonical sparse path image coordinates"
            )
        raise ValueError(
            f"routing_id={routing_id} missing continuous-optimal routing field: {key}"
        )
    return list(value)


def project_point(world_xyz: list[float], cam_pos: list[float], cam_rot: list[float], fov_deg: float, res_x: int, res_y: int):
    rx, ry, rz = cam_rot
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rx_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    ry_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    cam_mat = rz_m @ ry_m @ rx_m

    wp = np.array(world_xyz, dtype=np.float32)
    cp = np.array(cam_pos, dtype=np.float32)
    cam_pt = cam_mat.T @ (wp - cp)
    if cam_pt[2] >= -1e-6:
        return None

    depth = -cam_pt[2]
    if depth < 0.05:
        return None

    vfov_half = math.radians(fov_deg) / 2.0
    aspect = float(res_x) / float(res_y)
    hfov_half = math.atan(math.tan(vfov_half) * aspect)
    half_h = depth * math.tan(vfov_half)
    half_w = depth * math.tan(hfov_half)
    if abs(cam_pt[0]) > half_w or abs(cam_pt[1]) > half_h:
        return None

    screen_x = cam_pt[0] / half_w
    screen_y = cam_pt[1] / half_h
    px = int(round((screen_x + 1.0) * 0.5 * res_x))
    py = int(round((1.0 - (screen_y + 1.0) * 0.5) * res_y))
    return [px, py]


def add_target_mask(img: Image.Image, view_dir: Path, target_id: int, alpha: int = 80) -> Image.Image:
    idx_path = view_dir / "object_index.png"
    map_path = view_dir.parent / "object_index_map.json"
    if not idx_path.exists() or not map_path.exists():
        return img

    obj_map = load_object_index_mapping(map_path)

    pass_ids = {int(k) for k, v in obj_map.items() if int(v.get("obj_id", -1)) == int(target_id)}
    if not pass_ids:
        return img

    idx_img = Image.open(idx_path)
    idx = decode_object_index_array(np.array(idx_img), max_pass_id=max_pass_id_from_mapping(obj_map))
    mask = np.isin(idx, list(pass_ids))
    if not mask.any():
        return img

    rgba = img.convert("RGBA")
    arr = np.array(rgba, dtype=np.uint8)
    arr[mask, 0] = np.clip(arr[mask, 0] * 0.35 + 255 * 0.65, 0, 255).astype(np.uint8)
    arr[mask, 1] = np.clip(arr[mask, 1] * 0.35 + 40 * 0.65, 0, 255).astype(np.uint8)
    arr[mask, 2] = np.clip(arr[mask, 2] * 0.35 + 40 * 0.65, 0, 255).astype(np.uint8)
    arr[mask, 3] = np.clip(arr[mask, 3] + alpha, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="RGBA")


def draw_candidates(
    draw: ImageDraw.ImageDraw,
    candidates: list[dict],
    font_cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont],
    image_size: tuple[int, int],
    base_radius: int = 7,
    base_font_size: int = 13,
):
    depth_values = [float(c.get("depth", 0.0)) for c in candidates if c.get("depth") is not None]
    min_depth = min(depth_values) if depth_values else 0.0
    max_depth = max(depth_values) if depth_values else 0.0
    image_width, image_height = image_size

    for idx, c in enumerate(candidates):
        xy = c.get("image_xy")
        if not isinstance(xy, list) or len(xy) != 2:
            continue
        style = compute_marker_style(
            depth=float(c.get("depth", min_depth)),
            min_depth=min_depth,
            max_depth=max_depth,
            base_radius=base_radius,
            base_font_size=base_font_size,
        )
        font = load_font_cached(font_cache, style.font_size)
        x, y = clamp_marker_center(
            x=int(xy[0]),
            y=int(xy[1]),
            image_width=image_width,
            image_height=image_height,
            radius=style.radius,
            bottom_lift_px=0,
        )
        pm = bool(c.get("pointmass_walkable", False))
        emb = bool(c.get("embodied_feasible", False))
        outer = (40, 220, 40) if pm else (240, 60, 60)
        inner = (60, 220, 220) if emb else (255, 160, 40)
        inner_radius = max(2, style.radius - 3)
        draw.ellipse((x - style.radius, y - style.radius, x + style.radius, y + style.radius), outline=outer, width=2)
        draw.ellipse((x - inner_radius, y - inner_radius, x + inner_radius, y + inner_radius), fill=inner)
        label = str(c.get("display_id", c.get("waypoint_id", "?")))
        text_w, text_h = measure_text(font, label)
        layout = compute_label_layout(
            center_x=x,
            center_y=y,
            text_w=text_w,
            text_h=text_h,
            image_width=image_width,
            image_height=image_height,
            radius=style.radius,
        )
        leader_outline_width, leader_inner_width = compute_leader_line_widths(radius=style.radius)
        tx, ty = layout.text_x, layout.text_y
        draw.line(
            ((layout.line_start_x, layout.line_start_y), (layout.line_end_x, layout.line_end_y)),
            fill=(0, 0, 0),
            width=leader_outline_width,
        )
        draw.line(
            ((layout.line_start_x, layout.line_start_y), (layout.line_end_x, layout.line_end_y)),
            fill=(255, 255, 255),
            width=leader_inner_width,
        )
        draw.text((tx + 1, ty + 1), label, fill=(0, 0, 0), font=font)
        draw.text((tx, ty), label, fill=(255, 255, 255), font=font)


def draw_polyline(draw: ImageDraw.ImageDraw, pts: list[list[int]], color: tuple[int, int, int], width: int = 3):
    if len(pts) >= 2:
        draw.line([tuple(p) for p in pts], fill=color, width=width)


def draw_polyline_with_breaks(
    draw: ImageDraw.ImageDraw,
    pts: list[list[int] | None],
    color: tuple[int, int, int],
    width: int = 3,
):
    run: list[list[int]] = []
    for pt in pts:
        if not isinstance(pt, list) or len(pt) != 2 or pt[0] is None or pt[1] is None:
            draw_polyline(draw, run, color=color, width=width)
            run = []
            continue
        run.append([int(pt[0]), int(pt[1])])
    draw_polyline(draw, run, color=color, width=width)


def _point_in_rect(pt: tuple[int, int], rect: tuple[int, int, int, int]) -> bool:
    x, y = pt
    x0, y0, x1, y1 = rect
    return x0 <= x <= x1 and y0 <= y <= y1


def _orientation(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> int:
    value = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
    if value == 0:
        return 0
    return 1 if value > 0 else 2


def _on_segment(a: tuple[int, int], b: tuple[int, int], c: tuple[int, int]) -> bool:
    return min(a[0], c[0]) <= b[0] <= max(a[0], c[0]) and min(a[1], c[1]) <= b[1] <= max(a[1], c[1])


def _segments_intersect(a1: tuple[int, int], a2: tuple[int, int], b1: tuple[int, int], b2: tuple[int, int]) -> bool:
    o1 = _orientation(a1, a2, b1)
    o2 = _orientation(a1, a2, b2)
    o3 = _orientation(b1, b2, a1)
    o4 = _orientation(b1, b2, a2)

    if o1 != o2 and o3 != o4:
        return True
    if o1 == 0 and _on_segment(a1, b1, a2):
        return True
    if o2 == 0 and _on_segment(a1, b2, a2):
        return True
    if o3 == 0 and _on_segment(b1, a1, b2):
        return True
    if o4 == 0 and _on_segment(b1, a2, b2):
        return True
    return False


def segment_intersects_rect(
    p0: tuple[int, int],
    p1: tuple[int, int],
    rect: tuple[int, int, int, int],
    pad_px: int = 0,
) -> bool:
    x0, y0, x1, y1 = rect
    rect = (x0 - pad_px, y0 - pad_px, x1 + pad_px, y1 + pad_px)
    if _point_in_rect(p0, rect) or _point_in_rect(p1, rect):
        return True
    rx0, ry0, rx1, ry1 = rect
    edges = [
        ((rx0, ry0), (rx1, ry0)),
        ((rx1, ry0), (rx1, ry1)),
        ((rx1, ry1), (rx0, ry1)),
        ((rx0, ry1), (rx0, ry0)),
    ]
    return any(_segments_intersect(p0, p1, e0, e1) for e0, e1 in edges)


def segment_hits_mask(
    p0: tuple[int, int],
    p1: tuple[int, int],
    mask: np.ndarray | None,
    pad_px: int = 0,
) -> bool:
    if mask is None:
        return False
    h, w = mask.shape[:2]
    x0, y0 = p0
    x1, y1 = p1
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for step in range(steps + 1):
        t = step / steps
        x = int(round(x0 + t * (x1 - x0)))
        y = int(round(y0 + t * (y1 - y0)))
        xx0 = max(0, x - pad_px)
        xx1 = min(w, x + pad_px + 1)
        yy0 = max(0, y - pad_px)
        yy1 = min(h, y + pad_px + 1)
        if xx0 >= xx1 or yy0 >= yy1:
            continue
        if bool(mask[yy0:yy1, xx0:xx1].any()):
            return True
    return False


def sparsify_polyline_for_clear_straight_segments(
    pts: list[list[int]],
    labels: list[int | str],
    obstacle_rects: list[tuple[int, int, int, int]] | None = None,
    obstacle_mask: np.ndarray | None = None,
    pad_px: int = 2,
) -> tuple[list[list[int]], list[int | str]]:
    if len(pts) <= 2:
        return pts, labels

    obstacle_rects = obstacle_rects or []

    def _segment_clear(a: list[int], b: list[int]) -> bool:
        pa = (int(a[0]), int(a[1]))
        pb = (int(b[0]), int(b[1]))
        if any(segment_intersects_rect(pa, pb, rect, pad_px=pad_px) for rect in obstacle_rects):
            return False
        if segment_hits_mask(pa, pb, obstacle_mask, pad_px=pad_px):
            return False
        return True

    kept_pts = [pts[0]]
    kept_labels = [labels[0]]
    current = 0
    while current < len(pts) - 1:
        chosen = current + 1
        for candidate in range(len(pts) - 1, current, -1):
            if _segment_clear(pts[current], pts[candidate]):
                chosen = candidate
                break
        kept_pts.append(pts[chosen])
        kept_labels.append(labels[chosen])
        current = chosen
    return kept_pts, kept_labels


def load_blocking_furniture_mask(view_dir: Path) -> np.ndarray | None:
    idx_path = view_dir / "object_index.png"
    map_path = view_dir.parent / "object_index_map.json"
    if not idx_path.exists() or not map_path.exists():
        return None
    obj_map = load_object_index_mapping(map_path)
    idx = decode_object_index_array(np.array(Image.open(idx_path)), max_pass_id=max_pass_id_from_mapping(obj_map))
    blocking_pass_ids = []
    for key, value in obj_map.items():
        pass_id = int(key)
        category = str(value.get("category", "unknown")).lower()
        if category in NON_BLOCKING_CATEGORIES:
            continue
        blocking_pass_ids.append(pass_id)
    if not blocking_pass_ids:
        return None
    return np.isin(idx, blocking_pass_ids)


def draw_path_ids(
    draw: ImageDraw.ImageDraw,
    path_ids: list[int],
    waypoint_by_id: dict[int, dict],
    color: tuple[int, int, int],
    font_cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont],
    image_size: tuple[int, int],
    obstacle_mask: np.ndarray | None = None,
    sparsify_for_display: bool = False,
    context: str = "routing_path",
    base_radius: int = 6,
    base_font_size: int = 13,
):
    pts = []
    labels = []
    seen = set()
    missing_ids: list[int] = []
    depths: list[float] = []
    for wp_id in path_ids:
        rec = waypoint_by_id.get(int(wp_id))
        if rec is None:
            missing_ids.append(int(wp_id))
            continue
        xy = rec.get("image_xy")
        if not isinstance(xy, list) or len(xy) != 2:
            missing_ids.append(int(wp_id))
            continue
        key = (int(wp_id), int(xy[0]), int(xy[1]))
        if key in seen:
            continue
        seen.add(key)
        pts.append([int(xy[0]), int(xy[1])])
        labels.append(int(wp_id))
        depths.append(float(rec.get("depth", 0.0)))
    if missing_ids:
        raise ValueError(f"{context} missing routing support waypoint records: {missing_ids}")
    if sparsify_for_display:
        pts, labels = sparsify_polyline_for_clear_straight_segments(pts, labels, obstacle_mask=obstacle_mask)
    min_depth = min(depths) if depths else 0.0
    max_depth = max(depths) if depths else 0.0
    image_width, image_height = image_size
    for i, (pt, wp_id) in enumerate(zip(pts, labels)):
        rec = waypoint_by_id.get(int(wp_id), {})
        style = compute_marker_style(
            depth=float(rec.get("depth", min_depth)),
            min_depth=min_depth,
            max_depth=max_depth,
            base_radius=base_radius,
            base_font_size=base_font_size,
        )
        font = load_font_cached(font_cache, style.font_size)
        x, y = clamp_marker_center(
            x=int(pt[0]),
            y=int(pt[1]),
            image_width=image_width,
            image_height=image_height,
            radius=style.radius,
            bottom_lift_px=0,
        )
        display_id = rec.get("display_id", wp_id)
        draw.ellipse((x - style.radius, y - style.radius, x + style.radius, y + style.radius), outline=color, width=2)
        label = f"{i}:{display_id}"
        text_w, text_h = measure_text(font, label)
        layout = compute_label_layout(
            center_x=x,
            center_y=y,
            text_w=text_w,
            text_h=text_h,
            image_width=image_width,
            image_height=image_height,
            radius=style.radius,
        )
        leader_outline_width, leader_inner_width = compute_leader_line_widths(radius=style.radius)
        tx, ty = layout.text_x, layout.text_y
        draw.line(
            ((layout.line_start_x, layout.line_start_y), (layout.line_end_x, layout.line_end_y)),
            fill=(0, 0, 0),
            width=leader_outline_width,
        )
        draw.line(
            ((layout.line_start_x, layout.line_start_y), (layout.line_end_x, layout.line_end_y)),
            fill=color,
            width=leader_inner_width,
        )
        draw.text((tx + 1, ty + 1), label, fill=(0, 0, 0), font=font)
        draw.text((tx, ty), label, fill=color, font=font)
    draw_polyline(draw, pts, color=color, width=4)


def normalize_routing_candidate(candidate: dict) -> dict:
    normalized = dict(candidate)
    normalized["path_world_raw"] = require_routing_path_list(candidate, "optimal_path_world")
    normalized["path_world"] = require_routing_path_list(candidate, "canonical_sparse_path_world")
    normalized["path_image_raw"] = require_routing_path_list(candidate, "optimal_path_image_xy")
    normalized["path_image"] = require_routing_path_list(candidate, "canonical_sparse_path_image_xy")
    normalized["path_wp_ids_pointmass"] = list(candidate.get("gt_path_waypoint_ids_pointmass") or [])
    normalized["path_wp_ids_embodied"] = list(candidate.get("gt_path_waypoint_ids_embodied") or [])
    normalized["dense_path_wp_ids_pointmass"] = list(candidate.get("debug_dense_waypoint_ids_pointmass") or [])
    normalized["dense_path_wp_ids_embodied"] = list(candidate.get("debug_dense_waypoint_ids_embodied") or [])
    return normalized


def build_view_summary_rows(routing_candidates: list[dict]) -> list[dict]:
    rows = []
    for idx, candidate in enumerate(routing_candidates):
        identity = candidate.get("instruction_identity", {}) or {}
        selection = candidate.get("instruction_selection", {}) or {}
        complexity = candidate.get("routing_complexity", {}) or {}
        tier = require_text(candidate, "navigation_tier", context=f"routing[{idx}]")
        rows.append({
            "routing_id": str(candidate.get("routing_id", "?")),
            "target_id": int(candidate.get("target_id", -1)),
            "target_category": candidate.get("target_category", "unknown"),
            "reference_family_id": identity.get("reference_family_id") or identity.get("semantic_group_id"),
            "visible_color_name": (
                identity.get("reference_color_name")
                or identity.get("visible_color_name")
                or identity.get("material_color_name")
            ),
            "target_rule": identity.get("target_rule"),
            "selection_status": selection.get("selection_status"),
            "cohort_size": int(selection.get("cohort_size", 0)),
            "distance_m": round(float(candidate.get("distance_m", 0.0)), 3),
            "detour_ratio_raw": round(float(complexity.get("detour_ratio_raw", 0.0)), 3),
            "dense_turn_count": int(complexity.get("dense_turn_count", 0)),
            "sparse_turn_count": int(complexity.get("sparse_turn_count", 0)),
            "tier_family": "navigation",
            "tier": tier,
            "axis_summary": format_axis_summary(
                candidate.get("navigation_axes"),
                axis_order=NAVIGATION_AXIS_ORDER,
                context=f"routing[{idx}].navigation_axes",
            ),
            "validity_summary": format_validity_summary(
                candidate.get("navigation_validity"),
                context=f"routing[{idx}].navigation_validity",
            ),
        })
    rows.sort(key=lambda row: (row["distance_m"], row["routing_id"]))
    return rows


def resolve_routing_panel_candidates(bundle: dict, classification_view: dict | None = None) -> list[dict]:
    del classification_view
    return [dict(candidate) for candidate in bundle.get("classification_candidates", [])]


def build_view_summary_metadata_lines(bundle: dict, classification_view: dict | None = None) -> list[str]:
    source = classification_view if classification_view is not None else bundle
    context = "classification_view" if classification_view is not None else "bundle"
    affordance_tier = require_text(source, "affordance_tier", context=context)
    return [
        f"View affordance tier={affordance_tier}",
        "View affordance axes: "
        + format_axis_summary(
            source.get("affordance_axes"),
            axis_order=AFFORDANCE_AXIS_ORDER,
            context=f"{context}.affordance_axes",
        ),
        "View affordance validity: "
        + format_validity_summary(
            source.get("affordance_validity"),
            context=f"{context}.affordance_validity",
        ),
        "Routing family in this panel: navigation",
    ]


def build_routing_metadata_lines(routing: dict) -> list[str]:
    tier = require_text(routing, "navigation_tier", context="routing")
    return [
        f"Tier family=navigation tier={tier}",
        "Navigation axes: "
        + format_axis_summary(
            routing.get("navigation_axes"),
            axis_order=NAVIGATION_AXIS_ORDER,
            context="routing.navigation_axes",
        ),
        "Navigation validity: "
        + format_validity_summary(
            routing.get("navigation_validity"),
            context="routing.navigation_validity",
        ),
    ]


def build_routing_legend_lines() -> list[str]:
    return [
        "Legend:",
        "Candidate outer ring: Green=pointmass walkable, Red=pointmass blocked",
        "Candidate inner dot: Cyan=embodied feasible, Orange=embodied blocked",
        "Routing panels draw the full render-bundle routing support grid, not A1/B1 shared_ab_candidates",
        "Panel A: optimal dense GT in image/world projection + dense visible support ids",
        "Panel B: stored pointmass sparse GT waypoint ids (no audit-time re-sparsify)",
        "Panel C: stored embodied sparse GT waypoint ids (no audit-time re-sparsify)",
        "Tier semantics: routing candidates use Navigation Tier; view header uses Affordance Tier",
    ]


def render_view_summary(
    scene_id: str,
    view_id: int,
    view_dir: Path,
    bundle: dict,
    out_path: Path,
    classification_view: dict | None = None,
) -> bool:
    rgb_path = view_dir / "rgb.png"
    if not rgb_path.exists():
        return False
    routing = [normalize_routing_candidate(candidate) for candidate in bundle.get("routing_candidates", [])]
    if not routing:
        return False

    base = Image.open(rgb_path).convert("RGBA")
    draw = ImageDraw.Draw(base)
    font = load_font(16)
    small = load_font(13)
    palette = [
        (255, 80, 80), (80, 180, 255), (80, 220, 120), (255, 210, 80), (220, 120, 255),
        (255, 140, 90), (120, 255, 230), (255, 255, 120), (180, 180, 255), (255, 160, 200),
    ]
    w, h = base.size
    obstacle_mask = load_blocking_furniture_mask(view_dir)
    rows = build_view_summary_rows(routing)

    for idx, candidate in enumerate(routing):
        color = palette[idx % len(palette)]
        base = add_target_mask(base, view_dir, int(candidate.get("target_id", -1)), alpha=50)
        draw = ImageDraw.Draw(base)
        pts = []
        for point in candidate.get("path_image_raw") or candidate.get("path_image"):
            if isinstance(point, list) and len(point) == 2 and point[0] is not None and point[1] is not None:
                pts.append([int(point[0]), int(point[1])])
        if len(pts) < 2:
            raise ValueError(
                f"scene={scene_id} view={view_id} routing={candidate.get('routing_id')} missing Blender optimal path image coordinates"
            )
        if len(pts) >= 2:
            draw_polyline(draw, pts, color=color, width=4)
            sx, sy = pts[0]
            gx, gy = pts[-1]
            draw.ellipse((sx - 5, sy - 5, sx + 5, sy + 5), outline=(255, 255, 255), width=2)
            draw.ellipse((gx - 5, gy - 5, gx + 5, gy + 5), outline=color, width=3)
        rid = str(candidate.get("routing_id", f"r{idx+1}"))
        label_xy = pts[-1] if pts else [20 + idx * 14, 20 + idx * 14]
        draw.text((label_xy[0] + 6, label_xy[1] + 6), rid, fill=color, font=font)

    panel_h = 220 + 18 * len(rows)
    canvas = Image.new("RGBA", (w * 2, max(h, panel_h)), (20, 20, 20, 255))
    canvas.paste(base, (0, 0))
    text = ImageDraw.Draw(canvas)
    text.text((w + 16, 16), f"Scene={scene_id} View={view_id} Routing Summary", fill=(255, 255, 255), font=font)
    text.text((w + 16, 40), "All valid routing targets in this view", fill=(200, 200, 200), font=small)
    y = 62
    for line in build_view_summary_metadata_lines(bundle, classification_view=classification_view):
        text.text((w + 16, y), line, fill=(200, 200, 200), font=small)
        y += 16
    y += 4
    for idx, row in enumerate(rows):
        color = palette[idx % len(palette)]
        line = (
            f"{row['routing_id']}  tgt={row['target_id']} {row['target_category']}  "
            f"id={row['reference_family_id']}/{row['visible_color_name']}  "
            f"dist={row['distance_m']:.2f}m  detour={row['detour_ratio_raw']:.2f}  "
            f"turns={row['dense_turn_count']}/{row['sparse_turn_count']}  "
            f"tier={row['tier']}  axes={row['axis_summary']}  "
            f"validity={row['validity_summary']}  status={row['selection_status']}"
        )
        text.text((w + 16, y), line, fill=color, font=small)
        y += 18

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out_path)
    return True


def render_one_candidate(
    scene_id: str,
    view_id: int,
    view_dir: Path,
    bundle: dict,
    routing: dict,
    out_path: Path,
    classification_view: dict | None = None,
):
    routing = normalize_routing_candidate(routing)
    rgb_path = view_dir / "rgb.png"
    if not rgb_path.exists():
        return False

    base = Image.open(rgb_path).convert("RGBA")
    base = add_target_mask(base, view_dir, int(routing.get("target_id", -1)))

    w, h = base.size
    panel_w = w
    top_panel_h = 250
    canvas = Image.new("RGBA", (panel_w * 3, h + top_panel_h), (20, 20, 20, 255))

    font = load_font(16)
    small = load_font(13)
    font_cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {
        16: font,
        13: small,
    }
    obstacle_mask = load_blocking_furniture_mask(view_dir)

    # Panel A: candidates + raw dense GT + dense visible support ids + start/goal
    p1 = base.copy()
    d1 = ImageDraw.Draw(p1)
    candidates = resolve_routing_panel_candidates(bundle, classification_view=classification_view)
    draw_candidates(d1, candidates, font_cache, base.size, base_radius=6, base_font_size=12)

    path_world_img = list(routing.get("path_image_raw") or routing.get("path_image") or [])
    if len(path_world_img) < 2:
        raise ValueError(
            f"scene={scene_id} view={view_id} routing={routing.get('routing_id')} missing Blender optimal path image coordinates"
        )
    dense_point_ids = [int(x) for x in routing.get("dense_path_wp_ids_pointmass", [])]
    dense_embodied_ids = [int(x) for x in routing.get("dense_path_wp_ids_embodied", [])]

    waypoint_by_id = {int(c.get("waypoint_id", -1)): c for c in candidates}
    if dense_point_ids:
        draw_path_ids(
            d1,
            dense_point_ids,
            waypoint_by_id,
            (40, 255, 40),
            font_cache,
            base.size,
            obstacle_mask=None,
            sparsify_for_display=False,
            context=f"scene={scene_id} view={view_id} routing={routing.get('routing_id')} dense_pointmass_path",
            base_radius=5,
            base_font_size=11,
        )
    if dense_embodied_ids:
        draw_path_ids(
            d1,
            dense_embodied_ids,
            waypoint_by_id,
            (70, 220, 255),
            font_cache,
            base.size,
            obstacle_mask=None,
            sparsify_for_display=False,
            context=f"scene={scene_id} view={view_id} routing={routing.get('routing_id')} dense_embodied_path",
            base_radius=5,
            base_font_size=11,
        )
    draw_polyline_with_breaks(d1, path_world_img, color=(20, 20, 20), width=6)
    draw_polyline_with_breaks(d1, path_world_img, color=(255, 255, 255), width=2)

    start_wp = int(routing.get("start_anchor", {}).get("waypoint_id", -1))
    goal_wp = int(routing.get("goal_anchor", {}).get("waypoint_id", -1))
    for wp_id, label, color in ((start_wp, "START", (255, 255, 0)), (goal_wp, "GOAL", (255, 0, 255))):
        rec = waypoint_by_id.get(wp_id)
        if rec and isinstance(rec.get("image_xy"), list):
            x, y = clamp_marker_center(
                x=int(rec["image_xy"][0]),
                y=int(rec["image_xy"][1]),
                image_width=base.size[0],
                image_height=base.size[1],
                radius=10,
                bottom_lift_px=18,
            )
            display_id = rec.get("display_id", wp_id)
            d1.ellipse((x - 10, y - 10, x + 10, y + 10), outline=color, width=3)
            start_goal_label = f"{label}:{display_id}"
            text_w, text_h = measure_text(font, start_goal_label)
            layout = compute_label_layout(
                center_x=x,
                center_y=y,
                text_w=text_w,
                text_h=text_h,
                image_width=base.size[0],
                image_height=base.size[1],
                radius=10,
            )
            leader_outline_width, leader_inner_width = compute_leader_line_widths(radius=10)
            tx, ty = layout.text_x, layout.text_y
            d1.line(
                ((layout.line_start_x, layout.line_start_y), (layout.line_end_x, layout.line_end_y)),
                fill=(0, 0, 0),
                width=leader_outline_width,
            )
            d1.line(
                ((layout.line_start_x, layout.line_start_y), (layout.line_end_x, layout.line_end_y)),
                fill=color,
                width=leader_inner_width,
            )
            d1.text((tx + 1, ty + 1), start_goal_label, fill=(0, 0, 0), font=font)
            d1.text((tx, ty), start_goal_label, fill=color, font=font)

    # Panel B: stored pointmass sparse GT path ids
    p2 = base.copy()
    d2 = ImageDraw.Draw(p2)
    draw_candidates(d2, candidates, font_cache, base.size, base_radius=6, base_font_size=12)
    draw_path_ids(
        d2,
        [int(x) for x in routing.get("path_wp_ids_pointmass", [])],
        waypoint_by_id,
        (40, 255, 40),
        font_cache,
        base.size,
        obstacle_mask=obstacle_mask,
        context=f"scene={scene_id} view={view_id} routing={routing.get('routing_id')} sparse_pointmass_path",
        base_radius=5,
        base_font_size=11,
    )

    # Panel C: stored embodied sparse GT path ids
    p3 = base.copy()
    d3 = ImageDraw.Draw(p3)
    draw_candidates(d3, candidates, font_cache, base.size, base_radius=6, base_font_size=12)
    draw_path_ids(
        d3,
        [int(x) for x in routing.get("path_wp_ids_embodied", [])],
        waypoint_by_id,
        (70, 220, 255),
        font_cache,
        base.size,
        obstacle_mask=obstacle_mask,
        context=f"scene={scene_id} view={view_id} routing={routing.get('routing_id')} sparse_embodied_path",
        base_radius=5,
        base_font_size=11,
    )

    canvas.paste(p1, (0, top_panel_h))
    canvas.paste(p2, (panel_w, top_panel_h))
    canvas.paste(p3, (panel_w * 2, top_panel_h))

    dc = ImageDraw.Draw(canvas)
    header = (
        f"Scene={scene_id}  View={view_id}  Routing={routing.get('routing_id','?')}  "
        f"Target={routing.get('target_id','?')} ({routing.get('target_category','?')})  "
        f"Dist={float(routing.get('distance_m',0.0)):.2f}m"
    )
    dc.text((16, 14), header, fill=(255, 255, 255), font=font)
    inv = routing.get("invariants", {})
    inv_txt = f"Invariants: start={inv.get('start_in_forward_sector')} goal={inv.get('goal_in_target_zone')} embodied={inv.get('embodied_collision_free')}"
    dc.text((16, 42), inv_txt, fill=(220, 220, 220), font=small)
    raw_len = float(routing.get("optimal_length_m", 0.0))
    smooth_len = float(routing.get("canonical_sparse_length_m", raw_len))
    gain = max(0.0, smooth_len - raw_len)
    gain_txt = f"Path length: optimal={raw_len:.2f}m canonical_sparse={smooth_len:.2f}m gap={gain:.2f}m"
    dc.text((16, 58), gain_txt, fill=(220, 220, 220), font=small)
    y = 74
    for line in build_routing_metadata_lines(routing):
        dc.text((16, y), line, fill=(220, 220, 220), font=small)
        y += 16
    for line in build_routing_legend_lines():
        dc.text((16, y), line, fill=(190, 190, 190), font=small)
        y += 16

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out_path)
    return True


def load_scene_ids(renders_dir: Path, scene_list: Path | None, limit: int | None):
    if scene_list and scene_list.exists():
        scene_ids = [s.strip() for s in scene_list.read_text().splitlines() if s.strip()]
    else:
        scene_ids = sorted([p.name for p in renders_dir.iterdir() if p.is_dir()])
    if limit is not None and limit > 0:
        scene_ids = scene_ids[:limit]
    return scene_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--renders-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-outputs-dir", default=None)
    parser.add_argument("--scene-list", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--all-routing", action="store_true", help="Render all routing candidates per view (default: only first)")
    args = parser.parse_args()

    renders_dir = Path(args.renders_dir)
    out_dir = Path(args.output_dir)
    task_outputs_dir = Path(args.task_outputs_dir) if args.task_outputs_dir else None
    scene_list = Path(args.scene_list) if args.scene_list else None
    scene_ids = load_scene_ids(renders_dir, scene_list, args.limit)

    rows = []
    generated = 0

    for scene_id in scene_ids:
        scene_dir = renders_dir / scene_id
        cam_path = scene_dir / "cameras.json"
        if not cam_path.exists():
            continue
        cameras = json.load(open(cam_path))
        if not cameras:
            continue
        classification_views_by_id, using_task_outputs, task_outputs_path = load_scene_classification_views(
            scene_id=scene_id,
            scene_dir=scene_dir,
            task_outputs_root=task_outputs_dir,
        )
        for cam in cameras:
            view_id = int(cam.get("view_id", 0))
            view_dir = scene_dir / f"view_{view_id}"
            bundle_path = view_dir / VIEW_BUNDLE_FILENAME
            if not bundle_path.exists():
                continue
            bundle = json.load(open(bundle_path))
            classification_view = classification_views_by_id.get(view_id)
            if using_task_outputs and classification_view is None:
                raise ValueError(
                    f"scene={scene_id} view={view_id} missing protocolized classification view in {task_outputs_path}"
                )
            routing = list(bundle.get("routing_candidates", []))
            if not routing:
                continue
            summary_path = out_dir / scene_id / f"view_{view_id}_summary.png"
            if render_view_summary(
                scene_id,
                view_id,
                view_dir,
                bundle,
                summary_path,
                classification_view=classification_view,
            ):
                rows.append({
                    "scene_id": scene_id,
                    "view_id": view_id,
                    "routing_id": "__summary__",
                    "target_id": -1,
                    "target_category": "__summary__",
                    "image": str(summary_path),
                })
            use = routing if args.all_routing else [routing[0]]
            for rc in use:
                rid = str(rc.get("routing_id", "routing_unknown"))
                out_path = out_dir / scene_id / f"view_{view_id}_{rid}.png"
                ok = render_one_candidate(
                    scene_id,
                    view_id,
                    view_dir,
                    bundle,
                    rc,
                    out_path,
                    classification_view=classification_view,
                )
                if ok:
                    generated += 1
                    rows.append({
                        "scene_id": scene_id,
                        "view_id": view_id,
                        "routing_id": rid,
                        "target_id": int(rc.get("target_id", -1)),
                        "target_category": rc.get("target_category", "unknown"),
                        "image": str(out_path),
                    })

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "audit_index.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scene_id", "view_id", "routing_id", "target_id", "target_category", "image"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated audit images: {generated}")
    print(f"Index: {csv_path}")


if __name__ == "__main__":
    main()
