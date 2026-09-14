#!/usr/bin/env python3
"""
Build Next GT for A1/A2/B1/B2/C from rendered overlays + visible waypoints.

Inputs (per scene/view):
  - renders/<scene>/cameras.json (target_candidates, camera pose)
  - renders/<scene>/view_<id>/visible_waypoints_{a1,a2,b1,b2}.json
  - renders/<scene>/waypoint_graph_pointmass.json
  - renders/<scene>/waypoint_graph_embodied.json
  - scenes/<scene>/layout.json (target positions)

Outputs (jsonl):
  - gt_next_a1.jsonl, gt_next_b1.jsonl, gt_next_a2.jsonl, gt_next_b2.jsonl, gt_next_c.jsonl
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


RULES = [
    ("nearest", 0),
    ("farthest", 1),
    ("leftmost", 2),
]

DEFAULT_TASK_OUTPUTS_FILENAME = "next_task_outputs_v1.json"
VIEW_BUNDLE_FILENAME = "next_view_bundle_v3.json"
VIEW_BUNDLE_SCHEMA = "next_view_bundle_v3"


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_scene_list(scene_list: Path | None, scenes_dir: Path, limit: int | None) -> list[str]:
    if scene_list and scene_list.exists():
        lines = [s.strip() for s in scene_list.read_text().splitlines() if s.strip()]
        return lines[:limit] if limit else lines
    scene_ids = sorted([p.name for p in scenes_dir.iterdir() if p.is_dir()])
    return scene_ids[:limit] if limit else scene_ids


def rotation_matrix_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    rx_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float32)
    ry_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float32)
    rz_m = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float32)
    return rz_m @ ry_m @ rx_m


def project_point(world_xyz: np.ndarray,
                  cam_pos: np.ndarray,
                  cam_rot: np.ndarray,
                  fov_deg: float,
                  res_x: int,
                  res_y: int) -> dict | None:
    cam_mat = rotation_matrix_xyz(cam_rot[0], cam_rot[1], cam_rot[2])
    cam_pt = cam_mat.T @ (world_xyz - cam_pos)
    if cam_pt[2] >= -1e-6:
        return None
    depth = -cam_pt[2]
    if depth < 0.05:
        return None

    vfov_half = math.radians(fov_deg) / 2
    aspect = res_x / res_y
    hfov_half = math.atan(math.tan(vfov_half) * aspect)

    half_h = depth * math.tan(vfov_half)
    half_w = depth * math.tan(hfov_half)
    if abs(cam_pt[0]) > half_w or abs(cam_pt[1]) > half_h:
        return None

    screen_x = cam_pt[0] / half_w
    screen_y = cam_pt[1] / half_h
    px = (screen_x + 1.0) * 0.5 * res_x
    py = (1.0 - (screen_y + 1.0) * 0.5) * res_y
    return {
        "depth": float(depth),
        "screen_xy_norm": [float(screen_x), float(screen_y)],
        "image_xy": [int(round(px)), int(round(py))],
    }


def lookup_target_screen_x(camera: dict, target_id: int) -> float | None:
    for visible in camera.get("visible_objects", []):
        if int(visible.get("id", -1)) != int(target_id):
            continue
        screen_x = visible.get("screen_x")
        if screen_x is None:
            return None
        return float(screen_x)
    for target in camera.get("target_candidates", []):
        if int(target.get("target_id", -1)) != int(target_id):
            continue
        screen_x = target.get("screen_x")
        if screen_x is not None:
            return float(screen_x)
        screen_xy_norm = target.get("screen_xy_norm")
        if isinstance(screen_xy_norm, list) and screen_xy_norm:
            return float(screen_xy_norm[0])
    return None


def build_adjacency(edges: Iterable[dict], allowed_ids: set[int]) -> dict[int, list[tuple[int, float]]]:
    adj: dict[int, list[tuple[int, float]]] = {nid: [] for nid in allowed_ids}
    for e in edges:
        u = int(e.get("source"))
        v = int(e.get("target"))
        if u not in allowed_ids or v not in allowed_ids:
            continue
        w = float(e.get("distance", 1.0))
        adj[u].append((v, w))
        adj[v].append((u, w))
    return adj


def dijkstra(adj: dict[int, list[tuple[int, float]]], start: int, goal: int) -> list[int] | None:
    if start not in adj or goal not in adj:
        return None
    heap = [(0.0, start)]
    dist = {start: 0.0}
    prev: dict[int, int] = {}
    while heap:
        d, u = heapq.heappop(heap)
        if u == goal:
            break
        if d > dist.get(u, float("inf")):
            continue
        for v, w in adj.get(u, []):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))
    if goal not in dist:
        return None
    # Reconstruct
    path = [goal]
    while path[-1] != start:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def nearest_waypoint_id(visible: list[dict], pos_xy: tuple[float, float]) -> int | None:
    if not visible:
        return None
    best_id = None
    best_d = float("inf")
    for wp in visible:
        wx, wy = wp["world_xyz"][0], wp["world_xyz"][1]
        d = math.hypot(wx - pos_xy[0], wy - pos_xy[1])
        if d < best_d:
            best_d = d
            best_id = int(wp["waypoint_id"])
    return best_id


def densify_path_xy(path_world: list[list[float]], step_m: float = 0.25) -> list[tuple[float, float]]:
    if not path_world:
        return []
    pts = []
    for i in range(len(path_world) - 1):
        p0 = path_world[i]
        p1 = path_world[i + 1]
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        seg = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(math.ceil(seg / max(step_m, 1e-3))))
        for s in range(n):
            t = s / n
            pts.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    last = path_world[-1]
    pts.append((float(last[0]), float(last[1])))
    return pts


def map_render_path_to_visible_ids(
    visible: list[dict],
    path_world: list[list[float]] | None,
    sample_step_m: float = 0.25,
    max_match_dist_m: float = 0.9,
) -> tuple[list[int], list[int]]:
    if not visible or not path_world:
        return [], []

    nodes = []
    for wp in visible:
        nodes.append((
            int(wp["waypoint_id"]),
            int(wp["display_id"]),
            float(wp["world_xyz"][0]),
            float(wp["world_xyz"][1]),
        ))

    dense_pts = densify_path_xy(path_world, step_m=sample_step_m)
    path_wp_ids: list[int] = []
    path_disp_ids: list[int] = []

    def _append_if_new(wp_id: int, disp_id: int) -> None:
        if not path_wp_ids or path_wp_ids[-1] != wp_id:
            path_wp_ids.append(wp_id)
            path_disp_ids.append(disp_id)

    for x, y in dense_pts:
        best = None
        best_d = float("inf")
        for wp_id, disp_id, wx, wy in nodes:
            d = math.hypot(wx - x, wy - y)
            if d < best_d:
                best_d = d
                best = (wp_id, disp_id)
        if best is not None and best_d <= max_match_dist_m:
            _append_if_new(best[0], best[1])

    endpoint_pairs = []
    for endpoint in (path_world[0], path_world[-1]):
        ex, ey = float(endpoint[0]), float(endpoint[1])
        best = None
        best_d = float("inf")
        for wp_id, disp_id, wx, wy in nodes:
            d = math.hypot(wx - ex, wy - ey)
            if d < best_d:
                best_d = d
                best = (wp_id, disp_id)
        if best is not None:
            endpoint_pairs.append(best)

    if endpoint_pairs:
        _append_if_new(endpoint_pairs[0][0], endpoint_pairs[0][1])
    if len(endpoint_pairs) > 1:
        _append_if_new(endpoint_pairs[1][0], endpoint_pairs[1][1])

    if len(path_wp_ids) < 2 and len(endpoint_pairs) > 1:
        gx, gy = float(path_world[-1][0]), float(path_world[-1][1])
        ranked = sorted(nodes, key=lambda n: math.hypot(n[2] - gx, n[3] - gy))
        for wp_id, disp_id, _wx, _wy in ranked:
            if wp_id != path_wp_ids[0]:
                _append_if_new(wp_id, disp_id)
                break

    dedup_wp_ids: list[int] = []
    dedup_disp_ids: list[int] = []
    seen_wp_ids: set[int] = set()
    for wp_id, disp_id in zip(path_wp_ids, path_disp_ids):
        if wp_id in seen_wp_ids:
            continue
        seen_wp_ids.add(wp_id)
        dedup_wp_ids.append(wp_id)
        dedup_disp_ids.append(disp_id)

    return dedup_wp_ids, dedup_disp_ids


def resolve_visible_path_ids(
    candidate: dict,
    visible: list[dict],
    *,
    path_key: str,
) -> tuple[list[int], list[int]]:
    if not visible:
        return [], []

    disp_to_wp = {
        int(wp["display_id"]): int(wp["waypoint_id"])
        for wp in visible
        if "display_id" in wp and "waypoint_id" in wp
    }
    wp_to_disp = {wp_id: disp_id for disp_id, wp_id in disp_to_wp.items()}

    def _dedupe(wp_ids: list[int], disp_ids: list[int]) -> tuple[list[int], list[int]]:
        out_wp_ids: list[int] = []
        out_disp_ids: list[int] = []
        seen_wp_ids: set[int] = set()
        for wp_id, disp_id in zip(wp_ids, disp_ids):
            if wp_id in seen_wp_ids:
                continue
            seen_wp_ids.add(wp_id)
            out_wp_ids.append(wp_id)
            out_disp_ids.append(disp_id)
        return out_wp_ids, out_disp_ids

    explicit_disp_ids = [
        int(x)
        for x in candidate.get(f"gt_path_display_ids_{path_key}", [])
        if int(x) in disp_to_wp
    ]
    if explicit_disp_ids:
        explicit_wp_ids = [disp_to_wp[disp_id] for disp_id in explicit_disp_ids]
        explicit_wp_ids, explicit_disp_ids = _dedupe(explicit_wp_ids, explicit_disp_ids)
        if len(explicit_wp_ids) >= 2:
            return explicit_wp_ids, explicit_disp_ids

    canonical_wp_ids = [
        int(x)
        for x in candidate.get(f"gt_path_waypoint_ids_{path_key}", [])
        if int(x) >= 0
    ]
    if canonical_wp_ids:
        missing_wp_ids = [wp_id for wp_id in canonical_wp_ids if wp_id not in wp_to_disp]
        if missing_wp_ids:
            raise ValueError(
                f"candidate target={candidate.get('target_id')} path_key={path_key} missing visible "
                f"canonical waypoint ids: {missing_wp_ids}"
            )
        canonical_disp_ids = [wp_to_disp[wp_id] for wp_id in canonical_wp_ids]
        canonical_wp_ids, canonical_disp_ids = _dedupe(canonical_wp_ids, canonical_disp_ids)
        if len(canonical_wp_ids) >= 2:
            return canonical_wp_ids, canonical_disp_ids

    path_world = candidate.get("gt_path_keypoints_world") or candidate.get("debug_dense_path_world") or []
    if path_world:
        mapped_wp_ids, mapped_disp_ids = map_render_path_to_visible_ids(visible, path_world)
        if len(mapped_wp_ids) >= 2:
            return mapped_wp_ids, mapped_disp_ids
        raise ValueError(
            f"candidate target={candidate.get('target_id')} path_key={path_key} failed to map "
            "gt_path_keypoints_world onto visible display IDs"
        )

    legacy_wp_ids = candidate.get(f"path_wp_ids_{path_key}", [])
    if legacy_wp_ids:
        raise ValueError(
            f"candidate target={candidate.get('target_id')} path_key={path_key} "
            "legacy waypoint fallback is disabled; regenerate the v3 render bundle"
        )

    raise ValueError(
        f"candidate target={candidate.get('target_id')} path_key={path_key} is missing "
        "both explicit display IDs and path keypoints in the v3 bundle"
    )


def select_target_by_rule_unique(cands: list[dict], rule: str, used_ids: set[int]) -> dict | None:
    if not cands:
        return None
    if rule == "nearest":
        ordered = sorted(cands, key=lambda c: c["distance"])
    elif rule == "farthest":
        ordered = sorted(cands, key=lambda c: -c["distance"])
    elif rule == "leftmost":
        ordered = [c for c in cands if c.get("screen_x") is not None]
        ordered = sorted(ordered, key=lambda c: c["screen_x"])
    else:
        return None
    for c in ordered:
        if int(c["target_id"]) not in used_ids:
            return c
    return None


def select_goal_candidates(
    visible: list[dict],
    goal_xy: tuple[float, float],
    max_dist_m: float = 0.6,
    max_count: int | None = None,
) -> tuple[list[int], list[int]]:
    """Return goal waypoint_ids and display_ids within a radius around goal_xy."""
    if not visible:
        return [], []
    items = []
    for wp in visible:
        wx, wy = wp["world_xyz"][0], wp["world_xyz"][1]
        d = math.hypot(wx - goal_xy[0], wy - goal_xy[1])
        items.append((d, int(wp["waypoint_id"]), int(wp["display_id"])))
    items.sort(key=lambda x: x[0])
    within = [it for it in items if it[0] <= max_dist_m]
    chosen = within if within else items[:1]
    if max_count is not None:
        chosen = chosen[:max_count]
    goal_wp_ids = [it[1] for it in chosen]
    goal_disp_ids = [it[2] for it in chosen]
    return goal_wp_ids, goal_disp_ids


def distance_to_oriented_bbox_2d(wx: float, wy: float, bbox: list[float]) -> float:
    """Distance from point to oriented bbox surface in 2D."""
    if len(bbox) < 5:
        return float("inf")
    cx, cy = bbox[0], bbox[1]
    sx, sy = bbox[3], bbox[4]
    rot = bbox[6] if len(bbox) > 6 else 0.0
    dx = wx - cx
    dy = wy - cy
    cosr = math.cos(-rot)
    sinr = math.sin(-rot)
    lx = dx * cosr - dy * sinr
    ly = dx * sinr + dy * cosr
    hx = sx / 2.0
    hy = sy / 2.0
    ddx = max(abs(lx) - hx, 0.0)
    ddy = max(abs(ly) - hy, 0.0)
    return math.hypot(ddx, ddy)


def select_goal_candidates_from_bbox(
    visible: list[dict],
    bbox: list[float],
    radius_m: float = 0.6,
    max_count: int | None = None,
) -> tuple[list[int], list[int]]:
    if not visible:
        return [], []
    items = []
    for wp in visible:
        wx, wy = wp["world_xyz"][0], wp["world_xyz"][1]
        d = distance_to_oriented_bbox_2d(wx, wy, bbox)
        items.append((d, int(wp["waypoint_id"]), int(wp["display_id"])))
    items.sort(key=lambda x: x[0])
    within = [it for it in items if it[0] <= radius_m]
    chosen = within if within else items[:1]
    if max_count is not None:
        chosen = chosen[:max_count]
    return [it[1] for it in chosen], [it[2] for it in chosen]


def target_bbox_image_metrics(
    bbox: list[float],
    cam_pos: np.ndarray,
    cam_rot: np.ndarray,
    fov_deg: float,
    res_x: int,
    res_y: int,
) -> dict | None:
    if len(bbox) < 9:
        return None
    corners = []
    cx, cy, cz, sx, sy, sz, rx, ry, rz = bbox[:9]
    rot = rotation_matrix_xyz(rx, ry, rz)
    center = np.array([cx, cy, cz], dtype=np.float32)
    cam_mat = rotation_matrix_xyz(cam_rot[0], cam_rot[1], cam_rot[2])
    vfov_half = math.radians(fov_deg) / 2
    aspect = res_x / res_y
    hfov_half = math.atan(math.tan(vfov_half) * aspect)

    for dx in (-sx * 0.5, sx * 0.5):
        for dy in (-sy * 0.5, sy * 0.5):
            for dz in (-sz * 0.5, sz * 0.5):
                local = np.array([dx, dy, dz], dtype=np.float32)
                world_xyz = rot @ local + center
                cam_pt = cam_mat.T @ (world_xyz - cam_pos)
                if cam_pt[2] >= -1e-6:
                    continue
                depth = -cam_pt[2]
                if depth < 0.05:
                    continue
                half_h = depth * math.tan(vfov_half)
                half_w = depth * math.tan(hfov_half)
                screen_x = cam_pt[0] / half_w
                screen_y = cam_pt[1] / half_h
                px = int(round((screen_x + 1.0) * 0.5 * res_x))
                py = int(round((1.0 - (screen_y + 1.0) * 0.5) * res_y))
                corners.append((px, py))
    if not corners:
        return None
    xs = [pt[0] for pt in corners]
    ys = [pt[1] for pt in corners]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    clipped_min_x = max(0, min_x)
    clipped_max_x = min(res_x, max_x)
    clipped_min_y = max(0, min_y)
    clipped_max_y = min(res_y, max_y)
    return {
        "min_x": min_x,
        "max_x": max_x,
        "min_y": min_y,
        "max_y": max_y,
        "clipped_min_x": clipped_min_x,
        "clipped_max_x": clipped_max_x,
        "clipped_min_y": clipped_min_y,
        "clipped_max_y": clipped_max_y,
        "width_px": max_x - min_x,
        "height_px": max_y - min_y,
        "area_px": max(0, clipped_max_x - clipped_min_x) * max(0, clipped_max_y - clipped_min_y),
    }


def target_is_recognizable(
    bbox: list[float],
    cam_pos: np.ndarray,
    cam_rot: np.ndarray,
    fov_deg: float,
    res_x: int,
    res_y: int,
    min_bbox_area_px: float,
    border_margin_px: int,
) -> bool:
    metrics = target_bbox_image_metrics(bbox, cam_pos, cam_rot, fov_deg, res_x, res_y)
    if metrics is None:
        return False
    if metrics["area_px"] < min_bbox_area_px:
        return False
    if 0 <= metrics["min_x"] < border_margin_px or 0 <= metrics["min_y"] < border_margin_px:
        return False
    if res_x >= metrics["max_x"] > (res_x - border_margin_px):
        return False
    if res_y >= metrics["max_y"] > (res_y - border_margin_px):
        return False
    return True


def select_goal_candidates_for_target(
    visible: list[dict],
    bbox: list[float],
    path_world: list[list[float]] | None,
    radius_m: float = 0.6,
    max_count: int | None = None,
) -> tuple[list[int], list[int]]:
    goal_xy = None
    if isinstance(path_world, list) and path_world:
        last = path_world[-1]
        if isinstance(last, (list, tuple)) and len(last) >= 2:
            goal_xy = (float(last[0]), float(last[1]))
    if goal_xy is not None:
        return select_goal_candidates(visible, goal_xy, max_dist_m=radius_m, max_count=max_count)
    return select_goal_candidates_from_bbox(visible, bbox, radius_m=radius_m, max_count=max_count)


def _normalized_optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def targets_match_semantic_goal(predicted: dict, accepted: dict) -> bool:
    pred_group = _normalized_optional_text(predicted.get("semantic_group_id"))
    acc_group = _normalized_optional_text(accepted.get("semantic_group_id"))
    if pred_group and acc_group:
        if pred_group != acc_group:
            return False
        acc_color = _normalized_optional_text(accepted.get("target_color_name"))
        if acc_color is None:
            return True
        pred_color = _normalized_optional_text(predicted.get("target_color_name"))
        return pred_color == acc_color
    try:
        if int(predicted.get("target_id", -1)) != int(accepted.get("target_id", -1)):
            return False
        acc_color = _normalized_optional_text(accepted.get("target_color_name"))
        if acc_color is None:
            return True
        pred_color = _normalized_optional_text(predicted.get("target_color_name"))
        return pred_color == acc_color
    except Exception:
        return False


def select_targets_for_instruction_rule(
    cands: list[dict],
    *,
    accepted: dict,
    rule: str,
    distance_tie_eps_m: float = 1e-3,
    screen_x_tie_eps: float = 1e-4,
) -> list[dict]:
    cohort = [c for c in cands if targets_match_semantic_goal(c, accepted)]
    if not cohort:
        raise ValueError(
            f"no instruction cohort found for target={accepted.get('target_id')} "
            f"semantic_group={accepted.get('semantic_group_id')}"
        )

    if rule == "nearest":
        best = min(float(c["distance"]) for c in cohort)
        winners = [c for c in cohort if abs(float(c["distance"]) - best) <= distance_tie_eps_m]
    elif rule == "farthest":
        best = max(float(c["distance"]) for c in cohort)
        winners = [c for c in cohort if abs(float(c["distance"]) - best) <= distance_tie_eps_m]
    elif rule == "leftmost":
        leftmost = [c for c in cohort if c.get("screen_x") is not None]
        if not leftmost:
            raise ValueError(
                f"leftmost rule cannot be resolved for target={accepted.get('target_id')} without screen_x"
            )
        best = min(float(c["screen_x"]) for c in leftmost)
        winners = [c for c in leftmost if abs(float(c["screen_x"]) - best) <= screen_x_tie_eps]
    else:
        raise ValueError(f"unsupported target rule: {rule}")

    deduped: list[dict] = []
    seen_target_ids: set[int] = set()
    for candidate in winners:
        target_id = int(candidate.get("target_id", -1))
        if target_id in seen_target_ids:
            continue
        seen_target_ids.add(target_id)
        deduped.append(candidate)

    if len(deduped) > 1:
        raise ValueError(
            f"ambiguous target selection for rule={rule} semantic_group={accepted.get('semantic_group_id')} "
            f"color={accepted.get('target_color_name')}"
        )

    return deduped


def select_instruction_targets_for_view(
    cands: list[dict],
    *,
    rule: str,
    distance_tie_eps_m: float = 1e-3,
) -> list[dict]:
    selected: list[dict] = []
    seen_keys: set[tuple[str, str | None]] = set()
    for candidate in sorted(cands, key=lambda c: (float(c["distance"]), int(c.get("target_id", -1)))):
        semantic_group = (
            _normalized_optional_text(candidate.get("semantic_group_id"))
            or _normalized_optional_text(candidate.get("canonical_category"))
            or _normalized_optional_text(candidate.get("category"))
            or "unknown"
        )
        color_name = _normalized_optional_text(candidate.get("target_color_name"))
        key = (semantic_group, color_name)
        if key in seen_keys:
            continue
        winners = select_targets_for_instruction_rule(
            cands,
            accepted=candidate,
            rule=rule,
            distance_tie_eps_m=distance_tie_eps_m,
        )
        selected.extend(winners)
        seen_keys.add(key)
    selected.sort(key=lambda c: (float(c["distance"]), int(c.get("target_id", -1))))
    return selected


def get_task_outputs_filename(config: dict | None = None) -> str:
    return (config or {}).get("data", {}).get("task_outputs_filename", DEFAULT_TASK_OUTPUTS_FILENAME)


def resolve_task_outputs_path(
    render_scene_dir: Path,
    task_outputs_root: Path | None,
    scene_id: str,
    config: dict | None = None,
) -> Path:
    filename = get_task_outputs_filename(config)
    if task_outputs_root is not None:
        return task_outputs_root / scene_id / filename
    return render_scene_dir / filename


def load_routing_candidates_for_view(
    render_scene_dir: Path,
    task_outputs_path: Path,
    cameras: list[dict],
    view_id: int,
    prefer_materialized: bool = True,
) -> list[dict]:
    bundle = load_view_bundle_for_view(render_scene_dir, view_id)
    facts_hash = bundle.get("facts_hash")
    visible_screen_x = {}
    for visible in bundle.get("visible_objects", []):
        try:
            visible_screen_x[int(visible.get("id", -1))] = float(visible.get("screen_x"))
        except Exception:
            continue
    loaded = []
    for candidate in bundle.get("routing_candidates", []):
        normalized = dict(candidate)
        if facts_hash is not None:
            normalized["facts_hash"] = facts_hash
        target_id = int(normalized.get("target_id", -1))
        if target_id in visible_screen_x and normalized.get("screen_x") is None:
            normalized["screen_x"] = visible_screen_x[target_id]
        loaded.append(normalized)
    return loaded


def load_view_bundle_for_view(render_scene_dir: Path, view_id: int) -> dict:
    bundle_path = render_scene_dir / f"view_{view_id}" / VIEW_BUNDLE_FILENAME
    if not bundle_path.exists():
        raise FileNotFoundError(f"missing required {VIEW_BUNDLE_FILENAME} for view {view_id}")
    with open(bundle_path) as f:
        bundle = json.load(f)
    if bundle.get("schema_version") != VIEW_BUNDLE_SCHEMA:
        raise ValueError(
            f"{VIEW_BUNDLE_FILENAME} for view {view_id} must declare schema_version={VIEW_BUNDLE_SCHEMA}"
        )
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Next GT (A1/A2/B1/B2/C).")
    parser.add_argument("--config", default="configs/internscene_local.yaml")
    parser.add_argument("--scenes-dir", required=True)
    parser.add_argument("--renders-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-outputs-dir", default=None)
    parser.add_argument("--scene-list", default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    render_cfg = config.get("render", {})
    task_outputs_cfg = config.get("task_outputs", {})
    prefer_materialized = bool(task_outputs_cfg.get("prefer_materialized", True))
    res_x, res_y = render_cfg.get("resolution", [1024, 1024])
    fov = float(render_cfg.get("fov", 120))

    scenes_dir = Path(args.scenes_dir)
    renders_dir = Path(args.renders_dir)
    output_dir = Path(args.output_dir)
    task_outputs_root = Path(args.task_outputs_dir) if args.task_outputs_dir else None
    output_dir.mkdir(parents=True, exist_ok=True)

    scene_list = Path(args.scene_list) if args.scene_list else None
    scene_ids = load_scene_list(scene_list, scenes_dir, args.limit)

    out_a1 = (output_dir / "gt_next_a1.jsonl").open("w")
    out_b1 = (output_dir / "gt_next_b1.jsonl").open("w")
    out_a2 = (output_dir / "gt_next_a2.jsonl").open("w")
    out_b2 = (output_dir / "gt_next_b2.jsonl").open("w")
    out_c = (output_dir / "gt_next_c.jsonl").open("w")

    counts = {"a1": 0, "b1": 0, "a2": 0, "b2": 0, "c": 0}

    for scene_id in scene_ids:
        scene_path = scenes_dir / scene_id
        render_scene_dir = renders_dir / scene_id
        if not render_scene_dir.exists():
            continue

        layout_path = scene_path / "layout.json"
        cameras_path = render_scene_dir / "cameras.json"
        if not layout_path.exists() or not cameras_path.exists():
            continue

        with open(layout_path) as f:
            layout = json.load(f)
        with open(cameras_path) as f:
            cameras = json.load(f)
        if not cameras:
            continue
        task_outputs_path = resolve_task_outputs_path(
            render_scene_dir=render_scene_dir,
            task_outputs_root=task_outputs_root,
            scene_id=scene_id,
            config=config,
        )

        obj_map = {}
        for obj in layout:
            if "id" in obj:
                obj_map[int(obj["id"])] = obj

        for cam in cameras:
            view_id = cam.get("view_id", 0)
            view_dir = render_scene_dir / f"view_{view_id}"
            if not view_dir.exists():
                continue

            vp_a1 = view_dir / "visible_waypoints_a1.json"
            vp_b1 = view_dir / "visible_waypoints_b1.json"
            vp_a2 = view_dir / "visible_waypoints_a2.json"
            vp_b2 = view_dir / "visible_waypoints_b2.json"

            if not (vp_a1.exists() and vp_b1.exists() and vp_a2.exists() and vp_b2.exists()):
                continue

            bundle = load_view_bundle_for_view(render_scene_dir, int(view_id))
            facts_hash = bundle.get("facts_hash")
            affordance_validity = bundle.get("affordance_validity")
            affordance_axes = bundle.get("affordance_axes")
            affordance_tier = bundle.get("affordance_tier")
            if not isinstance(affordance_validity, dict):
                raise ValueError(f"scene={scene_id} view={view_id} missing affordance_validity in {VIEW_BUNDLE_FILENAME}")
            if not isinstance(affordance_axes, dict):
                raise ValueError(f"scene={scene_id} view={view_id} missing affordance_axes in {VIEW_BUNDLE_FILENAME}")
            if affordance_tier is None:
                raise ValueError(f"scene={scene_id} view={view_id} missing affordance_tier in {VIEW_BUNDLE_FILENAME}")
            if not bool(affordance_validity.get("eligible", False)):
                raise ValueError(
                    f"scene={scene_id} view={view_id} has ineligible affordance payload: "
                    f"{affordance_validity.get('failed_checks', [])}"
                )

            with open(vp_a1) as f:
                visible_a1 = json.load(f).get("visible_waypoints", [])
            with open(vp_b1) as f:
                visible_b1 = json.load(f).get("visible_waypoints", [])
            with open(vp_a2) as f:
                visible_a2 = json.load(f).get("visible_waypoints", [])
            with open(vp_b2) as f:
                visible_b2 = json.load(f).get("visible_waypoints", [])

            if not visible_a1 or not visible_b1 or not visible_a2 or not visible_b2:
                continue

            # A1/B1 classification GT
            walkable_a1 = [int(wp["display_id"]) for wp in visible_a1 if wp.get("pointmass_walkable")]
            walkable_b1 = [int(wp["display_id"]) for wp in visible_b1 if wp.get("embodied_feasible")]
            all_a1 = [int(wp["display_id"]) for wp in visible_a1]
            all_b1 = [int(wp["display_id"]) for wp in visible_b1]

            rec_a1 = {
                "question_id": f"{scene_id}_v{view_id}_a1",
                "source": "internscene",
                "task": "a1",
                "scene_id": scene_id,
                "view_id": view_id,
                "image_path": str(view_dir / "rgb_overlay_a1.png"),
                "visible_waypoints_path": str(vp_a1),
                "all_ids": all_a1,
                "walkable_ids": walkable_a1,
                "tier_family": "affordance",
                "tier": affordance_tier,
                "affordance_axes": affordance_axes,
                "affordance_validity": affordance_validity,
                "facts_hash": facts_hash,
            }
            rec_b1 = {
                "question_id": f"{scene_id}_v{view_id}_b1",
                "source": "internscene",
                "task": "b1",
                "scene_id": scene_id,
                "view_id": view_id,
                "image_path": str(view_dir / "rgb_overlay_b1.png"),
                "visible_waypoints_path": str(vp_b1),
                "all_ids": all_b1,
                "walkable_ids": walkable_b1,
                "robot_diameter_m": 0.6,
                "tier_family": "affordance",
                "tier": affordance_tier,
                "affordance_axes": affordance_axes,
                "affordance_validity": affordance_validity,
                "facts_hash": facts_hash,
            }
            out_a1.write(json.dumps(rec_a1, ensure_ascii=True) + "\n")
            out_b1.write(json.dumps(rec_b1, ensure_ascii=True) + "\n")
            counts["a1"] += 1
            counts["b1"] += 1

            # Routing GT (root-cause aligned):
            # Use render-time validated path_world directly and map it to
            # visible display IDs, instead of re-planning on a sparse visible
            # waypoint subgraph.
            cam_pos = cam.get("position", [0.0, 0.0, 0.0])
            cam_rot = cam.get("rotation", [0.0, 0.0, 0.0])

            # Build target candidates
            cands = []
            routing_candidates = load_routing_candidates_for_view(
                render_scene_dir,
                task_outputs_path,
                cameras,
                view_id,
                prefer_materialized=prefer_materialized,
            )
            for t in routing_candidates:
                tid = int(t.get("target_id", -1))
                obj = obj_map.get(tid)
                if not obj:
                    continue
                debug_dense_path = t.get("debug_dense_path_world") or []
                path_wp_ids_pointmass = [int(x) for x in t.get("gt_path_waypoint_ids_pointmass", [])]
                path_wp_ids_embodied = [int(x) for x in t.get("gt_path_waypoint_ids_embodied", [])]
                path_keypoints_world = t.get("gt_path_keypoints_world") or []
                if not path_wp_ids_pointmass or not path_wp_ids_embodied or not path_keypoints_world:
                    raise ValueError(
                        f"routing candidate for scene={scene_id} view={view_id} target={tid} is missing v3 sparse path fields"
                    )
                invariants = t.get("invariants", {})
                if invariants:
                    if not bool(invariants.get("start_in_forward_sector", False)):
                        continue
                    if not bool(invariants.get("goal_in_target_zone", False)):
                        continue
                    if not bool(invariants.get("embodied_collision_free", False)):
                        continue
                bbox = obj.get("bbox", [])
                pos = np.array([bbox[0], bbox[1], bbox[2]], dtype=np.float32)
                dist = float(t.get("distance", t.get("distance_m", math.hypot(bbox[0] - cam_pos[0], bbox[1] - cam_pos[1]))))
                screen_x = lookup_target_screen_x(cam, tid)
                cands.append({
                    "routing_id": t.get("routing_id"),
                    "target_id": tid,
                    "category": obj.get("category", t.get("category", t.get("target_category", "unknown"))),
                    "canonical_category": t.get("canonical_category", obj.get("category", t.get("target_category", "unknown"))),
                    "semantic_group_id": t.get("semantic_group_id", t.get("canonical_category", obj.get("category", "unknown"))),
                    "target_color_name": t.get("material_color_name"),
                    "distance": dist,
                    "world_xyz": [float(pos[0]), float(pos[1]), float(pos[2])],
                    "screen_x": screen_x,
                    "debug_dense_path_world": debug_dense_path,
                    "gt_path_keypoints_world": path_keypoints_world,
                    "bbox": bbox,
                    "path_wp_ids_pointmass": path_wp_ids_pointmass,
                    "path_wp_ids_embodied": path_wp_ids_embodied,
                    "instruction_identity": t.get("instruction_identity"),
                    "instruction_selection": t.get("instruction_selection"),
                    "routing_complexity": t.get("routing_complexity"),
                    "navigation_validity": t.get("navigation_validity"),
                    "navigation_axes": t.get("navigation_axes"),
                    "navigation_tier": t.get("navigation_tier"),
                    "facts_hash": t.get("facts_hash"),
                })
            if not cands:
                continue

            target_rule = "nearest"
            selected_targets = select_instruction_targets_for_view(cands, rule=target_rule)
            for tgt in selected_targets:
                instruction_targets = select_targets_for_instruction_rule(
                    cands,
                    accepted=tgt,
                    rule=target_rule,
                )
                if len(instruction_targets) != 1:
                    raise ValueError(
                        f"scene={scene_id} view={view_id} target={tgt['target_id']} "
                        f"resolved to {len(instruction_targets)} instruction winners"
                    )
                tgt = instruction_targets[0]

                canonical_a2, disp_path_a2 = resolve_visible_path_ids(tgt, visible_a2, path_key="pointmass")
                canonical_b2, disp_path_b2 = resolve_visible_path_ids(tgt, visible_b2, path_key="embodied")
                if len(canonical_a2) < 2 or len(canonical_b2) < 2:
                    raise ValueError(
                        f"scene={scene_id} view={view_id} target={tgt['target_id']} missing displayed sparse path projection"
                    )
                if len(disp_path_a2) < 2 or len(disp_path_b2) < 2:
                    continue

                goal_id_a2 = int(disp_path_a2[-1])
                goal_id_b2 = int(disp_path_b2[-1])
                goal_ids_a2 = {goal_id_a2}
                goal_ids_b2 = {goal_id_b2}
                for candidate in instruction_targets:
                    _alt_wp_a2, alt_a2 = resolve_visible_path_ids(candidate, visible_a2, path_key="pointmass")
                    _alt_wp_b2, alt_b2 = resolve_visible_path_ids(candidate, visible_b2, path_key="embodied")
                    if len(alt_a2) >= 2:
                        goal_ids_a2.add(int(alt_a2[-1]))
                    if len(alt_b2) >= 2:
                        goal_ids_b2.add(int(alt_b2[-1]))

                routing_id = str(tgt.get("routing_id", f"routing_v{view_id}_t{int(tgt['target_id']):03d}"))
                qstem = f"{scene_id}_v{view_id}_{routing_id}"
                routing_complexity = tgt.get("routing_complexity")
                navigation_validity = tgt.get("navigation_validity")
                navigation_axes = tgt.get("navigation_axes")
                navigation_tier = tgt.get("navigation_tier")
                if not isinstance(navigation_validity, dict):
                    raise ValueError(
                        f"scene={scene_id} view={view_id} target={tgt['target_id']} missing navigation_validity"
                    )
                if not isinstance(navigation_axes, dict):
                    raise ValueError(
                        f"scene={scene_id} view={view_id} target={tgt['target_id']} missing navigation_axes"
                    )
                if navigation_tier is None:
                    raise ValueError(
                        f"scene={scene_id} view={view_id} target={tgt['target_id']} missing navigation_tier"
                    )
                if not bool(navigation_validity.get("eligible", False)):
                    raise ValueError(
                        f"scene={scene_id} view={view_id} target={tgt['target_id']} has ineligible navigation payload: "
                        f"{navigation_validity.get('failed_checks', [])}"
                    )
                rec_common = {
                    "source": "internscene",
                    "tier_family": "navigation",
                    "tier": navigation_tier,
                    "target_rule": target_rule,
                    "target_rule_fallback": False,
                    "target_rule_degenerate": False,
                    "scene_id": scene_id,
                    "view_id": view_id,
                    "routing_id": routing_id,
                    "target_id": tgt["target_id"],
                    "target_category": tgt["category"],
                    "canonical_category": tgt["canonical_category"],
                    "semantic_group_id": tgt["semantic_group_id"],
                    "target_color_name": tgt["target_color_name"],
                    "target_distance_m": round(tgt["distance"], 3),
                    "instruction_identity": tgt.get("instruction_identity"),
                    "instruction_selection": tgt.get("instruction_selection"),
                    "routing_complexity": routing_complexity,
                    "navigation_axes": navigation_axes,
                    "navigation_validity": navigation_validity,
                    "facts_hash": tgt.get("facts_hash"),
                }

                rec_a2 = {
                    "question_id": f"{qstem}_a2",
                    "task": "a2",
                    "image_path": str(view_dir / "rgb_overlay_a2.png"),
                    "visible_waypoints_path": str(vp_a2),
                    "start_id": int(disp_path_a2[0]),
                    "goal_id": goal_id_a2,
                    "goal_ids": sorted(goal_ids_a2),
                    "path_ids": disp_path_a2,
                    **rec_common,
                }
                rec_b2 = {
                    "question_id": f"{qstem}_b2",
                    "task": "b2",
                    "image_path": str(view_dir / "rgb_overlay_b2.png"),
                    "visible_waypoints_path": str(vp_b2),
                    "start_id": int(disp_path_b2[0]),
                    "goal_id": goal_id_b2,
                    "goal_ids": sorted(goal_ids_b2),
                    "path_ids": disp_path_b2,
                    "robot_diameter_m": 0.6,
                    **rec_common,
                }

                rec_c = {
                    "question_id": f"{qstem}_c",
                    "source": "internscene",
                    "task": "c",
                    "scene_id": scene_id,
                    "view_id": view_id,
                    "routing_id": routing_id,
                    "image_path": str(view_dir / "rgb_overlay_c.png"),
                    "visible_waypoints_path": str(view_dir / "visible_waypoints_c.json"),
                    "start_id": int(disp_path_a2[0]),
                    "goal_id": goal_id_a2,
                    "path_ids": disp_path_a2,
                    "target_id": tgt["target_id"],
                    "target_category": tgt["category"],
                    "canonical_category": tgt["canonical_category"],
                    "semantic_group_id": tgt["semantic_group_id"],
                    "target_color_name": tgt["target_color_name"],
                    "tier_family": "navigation",
                    "tier": navigation_tier,
                    "instruction_identity": tgt.get("instruction_identity"),
                    "instruction_selection": tgt.get("instruction_selection"),
                    "routing_complexity": routing_complexity,
                    "navigation_axes": navigation_axes,
                    "navigation_validity": navigation_validity,
                    "facts_hash": tgt.get("facts_hash"),
                }

                out_a2.write(json.dumps(rec_a2, ensure_ascii=True) + "\n")
                out_b2.write(json.dumps(rec_b2, ensure_ascii=True) + "\n")
                out_c.write(json.dumps(rec_c, ensure_ascii=True) + "\n")
                counts["a2"] += 1
                counts["b2"] += 1
                counts["c"] += 1

    out_a1.close()
    out_b1.close()
    out_a2.close()
    out_b2.close()
    out_c.close()

    stats_path = output_dir / "gt_next_stats.json"
    with open(stats_path, "w") as f:
        json.dump(counts, f, indent=2)

    print("GT Next generation complete.")
    for k, v in counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
