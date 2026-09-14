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
import hashlib
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

from scene_nav_facts import REQUIRED_CONTINUOUS_OPTIMAL_ROUTING_FIELDS
from h_view_utils import build_h_view, write_h_view_sidecar
from prompt_truth_contract import (
    FIXED_TARGET_CONTRACT_VERSION,
    derive_fixed_gt_identity_hash_from_identity,
)
from tier_policy import classify_navigation_tier, classify_reference_axis


RULES = [
    ("nearest", 0),
    ("farthest", 1),
    ("leftmost", 2),
]
DEFAULT_RENDER_WIDTH_PX = 1024

DEFAULT_TASK_OUTPUTS_FILENAME = "next_task_outputs_v1.json"
VIEW_BUNDLE_FILENAME = "next_view_bundle_v3.json"
VIEW_BUNDLE_SCHEMA = "next_view_bundle_v3"
TRUTH_REPAIR_PROTOCOL_PATH = SCRIPT_DIR.parent / "configs" / "truth_repair_protocol_v1.json"
CUE_REPAIR_PROTOCOL_PATH = SCRIPT_DIR.parent / "configs" / "cue_repair_protocol_v1.json"
HVIEW_CONFUSABLE_UNIVERSE_PATH = SCRIPT_DIR.parent / "configs" / "hview_confusable_universe_v1.json"
SCHEMA_DIR = SCRIPT_DIR.parent / "schemas"
SCHEMA_BUNDLE_FILENAMES = (
    "cue_search_trace_v1.schema.json",
    "h_view_confusable_universe_v1.schema.json",
    "prompt_resolution_trace_v1.schema.json",
    "resolved_cue_package_v1.schema.json",
    "scene_relation_graph_v1.schema.json",
    "view_relation_projection_v1.schema.json",
)
TRUTH_REPAIR_PROTOCOL_REQUIRED_KEYS = (
    "left_right_margin_px",
    "ordinal_margin_px",
    "front_back_margin_m",
    "near_far_margin_m",
    "max_cues",
    "max_relation_cues",
    "max_prompt_delta_sentences",
    "max_prompt_delta_tokens",
    "max_cue_tokens_per_cue",
    "max_attribute_slots",
    "max_relation_slots",
    "counts",
    "ratio_tolerance",
    "a2b2_recovery",
    "c_benchmark_freeze",
    "train_val_qc",
    "global_resplit",
)
TRUTH_REPAIR_PROTOCOL_RUNTIME_KEYS = (
    "left_right_margin_px",
    "ordinal_margin_px",
    "front_back_margin_m",
    "near_far_margin_m",
    "max_cues",
    "max_relation_cues",
    "max_prompt_delta_sentences",
    "max_prompt_delta_tokens",
    "max_cue_tokens_per_cue",
    "max_attribute_slots",
    "max_relation_slots",
    "ratio_tolerance",
    "a2b2_recovery",
    "c_benchmark_freeze",
    "train_val_qc",
    "global_resplit",
)
_TRUTH_REPAIR_PROTOCOL_CACHE: dict[Path, dict] = {}
_CUE_REPAIR_PROTOCOL_CACHE: dict[Path, dict] = {}
_HVIEW_CONFUSABLE_UNIVERSE_CACHE: dict[Path, dict] = {}
_SCHEMA_BUNDLE_HASH_CACHE: str | None = None


def _stable_json_dumps(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _compute_hash(payload: object) -> str:
    return hashlib.sha256(_stable_json_dumps(payload).encode("utf-8")).hexdigest()


def load_truth_repair_protocol(path: Path | None = None) -> dict:
    protocol_path = (path or TRUTH_REPAIR_PROTOCOL_PATH).resolve()
    with open(protocol_path) as f:
        protocol = json.load(f)
    missing = [key for key in TRUTH_REPAIR_PROTOCOL_REQUIRED_KEYS if key not in protocol]
    if missing:
        raise ValueError(
            f"truth-repair protocol missing required keys in {protocol_path}: {missing}"
        )
    counts = protocol.get("counts") or {}
    benchmark_counts = counts.get("benchmark") or {}
    for key in ("total", "minimum"):
        if key not in benchmark_counts:
            raise ValueError(
                f"truth-repair protocol missing counts.benchmark.{key} in {protocol_path}"
            )
    per_task_counts = counts.get("per_task") or {}
    for task in ("a1", "b1", "a2", "b2", "c"):
        task_counts = per_task_counts.get(task) or {}
        for key in ("total", "minimum"):
            if key not in task_counts:
                raise ValueError(
                    f"truth-repair protocol missing counts.per_task.{task}.{key} in {protocol_path}"
                )
    global_resplit = protocol.get("global_resplit") or {}
    for key in (
        "enabled",
        "selection_order",
        "scene_isolation_mode",
        "benchmark_target_questions",
        "benchmark_units",
        "immutability_fields",
        "selection_objective",
        "reporting_required",
        "full_corpus_preflight_required",
    ):
        if key not in global_resplit:
            raise ValueError(
                f"truth-repair protocol missing global_resplit.{key} in {protocol_path}"
            )
    if global_resplit.get("selection_order") != ["benchmark", "val", "train"]:
        raise ValueError(
            f"truth-repair protocol must freeze global_resplit.selection_order=['benchmark', 'val', 'train'] in {protocol_path}"
        )
    benchmark_target_questions = global_resplit.get("benchmark_target_questions") or {}
    for key in ("min", "max"):
        if key not in benchmark_target_questions:
            raise ValueError(
                f"truth-repair protocol missing global_resplit.benchmark_target_questions.{key} in {protocol_path}"
            )
    benchmark_units = global_resplit.get("benchmark_units") or {}
    expected_units = {
        "a1": "view_bundle",
        "b1": "view_bundle",
        "a2": "route_bundle",
        "b2": "route_bundle",
        "c": "route_bundle",
    }
    if benchmark_units != expected_units:
        raise ValueError(
            f"truth-repair protocol has unexpected global_resplit.benchmark_units in {protocol_path}: {benchmark_units}"
        )
    immutability_fields = global_resplit.get("immutability_fields") or {}
    for key in ("view_bundle", "route_bundle", "c_route_bundle"):
        value = immutability_fields.get(key)
        if not isinstance(value, list) or not value:
            raise ValueError(
                f"truth-repair protocol missing non-empty global_resplit.immutability_fields.{key} in {protocol_path}"
            )
    return protocol


def load_cue_repair_protocol(path: Path | None = None) -> dict:
    protocol_path = (path or CUE_REPAIR_PROTOCOL_PATH).resolve()
    with open(protocol_path) as f:
        protocol = json.load(f)
    required_keys = (
        "schema_version",
        "cue_priority_order",
        "trusted_relation_types",
        "disabled_relation_types",
    )
    missing = [key for key in required_keys if key not in protocol]
    if missing:
        raise ValueError(f"cue-repair protocol missing required keys in {protocol_path}: {missing}")
    return protocol


def load_hview_confusable_universe_protocol(path: Path | None = None) -> dict:
    protocol_path = (path or HVIEW_CONFUSABLE_UNIVERSE_PATH).resolve()
    with open(protocol_path) as f:
        protocol = json.load(f)
    required_keys = (
        "schema_version",
        "same_type_modes",
        "supported_only_uniqueness_witness",
        "audit_only_inputs",
    )
    missing = [key for key in required_keys if key not in protocol]
    if missing:
        raise ValueError(
            f"hview confusable universe protocol missing required keys in {protocol_path}: {missing}"
        )
    return protocol


def get_truth_repair_protocol(path: Path | None = None) -> dict:
    protocol_path = (path or TRUTH_REPAIR_PROTOCOL_PATH).resolve()
    cached = _TRUTH_REPAIR_PROTOCOL_CACHE.get(protocol_path)
    if cached is not None:
        return cached
    protocol = load_truth_repair_protocol(protocol_path)
    _TRUTH_REPAIR_PROTOCOL_CACHE[protocol_path] = protocol
    return protocol


def get_cue_repair_protocol(path: Path | None = None) -> dict:
    protocol_path = (path or CUE_REPAIR_PROTOCOL_PATH).resolve()
    cached = _CUE_REPAIR_PROTOCOL_CACHE.get(protocol_path)
    if cached is not None:
        return cached
    protocol = load_cue_repair_protocol(protocol_path)
    _CUE_REPAIR_PROTOCOL_CACHE[protocol_path] = protocol
    return protocol


def get_hview_confusable_universe_protocol(path: Path | None = None) -> dict:
    protocol_path = (path or HVIEW_CONFUSABLE_UNIVERSE_PATH).resolve()
    cached = _HVIEW_CONFUSABLE_UNIVERSE_CACHE.get(protocol_path)
    if cached is not None:
        return cached
    protocol = load_hview_confusable_universe_protocol(protocol_path)
    _HVIEW_CONFUSABLE_UNIVERSE_CACHE[protocol_path] = protocol
    return protocol


def get_schema_bundle_hash() -> str:
    global _SCHEMA_BUNDLE_HASH_CACHE
    if _SCHEMA_BUNDLE_HASH_CACHE is not None:
        return _SCHEMA_BUNDLE_HASH_CACHE
    payload = {}
    for filename in SCHEMA_BUNDLE_FILENAMES:
        with open(SCHEMA_DIR / filename) as f:
            payload[filename] = json.load(f)
    _SCHEMA_BUNDLE_HASH_CACHE = _compute_hash(payload)
    return _SCHEMA_BUNDLE_HASH_CACHE


def truth_repair_protocol_runtime_snapshot(protocol: dict | None = None) -> dict:
    source = protocol or get_truth_repair_protocol()
    return {key: source[key] for key in TRUTH_REPAIR_PROTOCOL_RUNTIME_KEYS}


def require_continuous_optimal_routing_candidate(candidate: dict) -> None:
    missing = []
    for key in REQUIRED_CONTINUOUS_OPTIMAL_ROUTING_FIELDS:
        value = candidate.get(key)
        if value is None:
            missing.append(key)
            continue
        if key.endswith("_world") and (not isinstance(value, list) or len(value) < 2):
            missing.append(key)
            continue
        if key.endswith("_m") and float(value) <= 0.0:
            missing.append(key)
            continue
        if key.startswith("gt_path_waypoint_ids_") and (not isinstance(value, list) or len(value) < 2):
            missing.append(key)
            continue
    if missing:
        routing_id = candidate.get("routing_id", "routing_unknown")
        raise ValueError(
            f"routing_id={routing_id} missing continuous-optimal routing facts: {missing}"
        )


def require_rule_winner_instruction_selection(candidate: dict) -> dict:
    selection = candidate.get("instruction_selection")
    routing_id = candidate.get("routing_id", "routing_unknown")
    if not isinstance(selection, dict):
        raise ValueError(f"routing_id={routing_id} missing required instruction_selection")
    if "selection_status" not in selection or "is_rule_winner" not in selection:
        raise ValueError(f"routing_id={routing_id} instruction_selection missing rule-winner fields")
    if not isinstance(selection.get("is_rule_winner"), bool):
        raise ValueError(f"routing_id={routing_id} instruction_selection is_rule_winner must be bool")
    return selection


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

    raise ValueError(
        f"candidate target={candidate.get('target_id')} path_key={path_key} is missing "
        "canonical sparse waypoint ids; legacy waypoint fallback is disabled"
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


def infer_source_group_id(scene_id: str | None) -> str | None:
    scene_text = _normalized_optional_text(scene_id)
    if scene_text is None:
        return None
    parts = [part for part in str(scene_id).split("__") if part]
    if len(parts) < 2:
        return str(scene_id)
    if parts[0] == "matterport3d" and len(parts) >= 3:
        return "__".join(parts[:2])
    return str(scene_id)


def build_h_view_sidecar_relpath(
    *,
    scene_id: str | None,
    view_id: int | None,
    target_id: int,
) -> str:
    if scene_id is not None and view_id is not None:
        return f"sidecars/h_view/{scene_id}/view_{int(view_id)}.json"
    return f"sidecars/h_view/target_{int(target_id)}.json"


def derive_human_visible_legality(
    *,
    human_visible_support: str,
    human_visible_uniqueness: str,
    prompt_truth_tier: str,
    prompt_answer_leakage: bool,
) -> str:
    if prompt_answer_leakage:
        return "illegal"
    if human_visible_support == "supported" and human_visible_uniqueness == "unique":
        return "legal"
    if prompt_truth_tier == "reject" or human_visible_uniqueness == "ambiguous":
        return "illegal"
    return "uncertain"


def _normalized_cue_color(value: object) -> str | None:
    color = _normalized_optional_text(value)
    if color in (None, "unknown", "none", "n/a", "na", "null"):
        return None
    return color


def truth_repair_category_key(candidate: dict) -> str:
    return (
        _normalized_optional_text(candidate.get("canonical_category"))
        or _normalized_optional_text(candidate.get("category"))
        or "unknown"
    )


def _optional_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _lookup_candidate_metric(candidate: dict, keys: tuple[str, ...]) -> float | None:
    for key in keys:
        value = _optional_float(candidate.get(key))
        if value is not None:
            return value
    return None


def _lookup_relation_anchor_id(candidate: dict) -> str | None:
    for key in ("relation_anchor_id", "relation_anchor_target_id", "anchor_target_id"):
        value = candidate.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def _lookup_h_view_screen_x(obj: dict) -> float | None:
    anchor = obj.get("screen_box_or_anchor")
    if isinstance(anchor, dict) and anchor.get("x") is not None:
        return float(anchor["x"])
    if obj.get("screen_x") is not None:
        return float(obj["screen_x"])
    return None


def _screen_margin_ndc(left_right_margin_px: float, *, render_width_px: int = DEFAULT_RENDER_WIDTH_PX) -> float:
    return 2.0 * float(left_right_margin_px) / float(render_width_px)


def _derive_extreme_relation_side(
    *,
    target_id: int,
    supported_visible_ids: list[int],
    supported_visible_objects: list[dict],
    left_right_margin_px: float,
    front_back_margin_m: float,
    near_far_margin_m: float,
    depth_map: dict[int, float],
    distance_map: dict[int, float],
) -> str | None:
    if len(supported_visible_ids) <= 1:
        return None

    threshold_ndc = _screen_margin_ndc(left_right_margin_px)
    screen_x_by_id: dict[int, float] = {}
    for obj in supported_visible_objects:
        object_id = int(obj.get("object_id", -1))
        if object_id not in supported_visible_ids:
            continue
        screen_x = _lookup_h_view_screen_x(obj)
        if screen_x is not None:
            screen_x_by_id[object_id] = screen_x
    if len(screen_x_by_id) == len(supported_visible_ids):
        ordered = sorted(screen_x_by_id.items(), key=lambda item: (item[1], item[0]))
        if target_id == ordered[0][0] and len(ordered) >= 2 and (ordered[1][1] - ordered[0][1]) > threshold_ndc:
            return "left"
        if target_id == ordered[-1][0] and len(ordered) >= 2 and (ordered[-1][1] - ordered[-2][1]) > threshold_ndc:
            return "right"

    depth_by_id = {
        candidate_id: depth
        for candidate_id, depth in depth_map.items()
        if candidate_id in supported_visible_ids
    }
    if len(depth_by_id) == len(supported_visible_ids):
        ordered = sorted(depth_by_id.items(), key=lambda item: (item[1], item[0]))
        if target_id == ordered[0][0] and len(ordered) >= 2 and (ordered[1][1] - ordered[0][1]) > front_back_margin_m:
            return "front"
        if target_id == ordered[-1][0] and len(ordered) >= 2 and (ordered[-1][1] - ordered[-2][1]) > front_back_margin_m:
            return "back"

    distance_by_id = {
        candidate_id: distance
        for candidate_id, distance in distance_map.items()
        if candidate_id in supported_visible_ids
    }
    if len(distance_by_id) == len(supported_visible_ids):
        ordered = sorted(distance_by_id.items(), key=lambda item: (item[1], item[0]))
        if target_id == ordered[0][0] and len(ordered) >= 2 and (ordered[1][1] - ordered[0][1]) > near_far_margin_m:
            return "front"
        if target_id == ordered[-1][0] and len(ordered) >= 2 and (ordered[-1][1] - ordered[-2][1]) > near_far_margin_m:
            return "back"
    return None


def _directional_relation_side_allowed(side: str | None, cue_protocol: dict) -> bool:
    normalized_side = _normalized_optional_text(side)
    if normalized_side is None:
        return False
    relation_type_by_side = {
        "left": "left_of",
        "right": "right_of",
        "front": "in_front_of",
        "back": "behind",
    }
    relation_type = relation_type_by_side.get(normalized_side)
    if relation_type is None:
        return False
    disabled_relation_types = {
        _normalized_optional_text(value)
        for value in list(cue_protocol.get("disabled_relation_types") or [])
        if _normalized_optional_text(value) is not None
    }
    return relation_type not in disabled_relation_types


def build_prompt_aligned_navigation_reference_facts(
    *,
    original_reference_facts: dict | None,
    truth_repair_metadata: dict,
) -> dict:
    reference_facts = dict(original_reference_facts or {})
    prompt_repair_status = _normalized_optional_text(truth_repair_metadata.get("prompt_repair_status"))
    cue_family = _normalized_optional_text(truth_repair_metadata.get("cue_family"))
    cue_type = _normalized_optional_text(truth_repair_metadata.get("cue_type"))
    ambiguity_count = int(truth_repair_metadata.get("ambiguity_count_H_view") or 0)

    resolution_mode = _normalized_optional_text(reference_facts.get("reference_resolution_mode")) or "category_only"
    if prompt_repair_status == "repaired":
        if cue_family == "relation" or cue_type in {"side", "near", "on", "inside", "under"}:
            resolution_mode = "relation_only"
        elif cue_family in {"distance", "strict_distance"} or cue_type == "distance_rank":
            resolution_mode = "distance_only"
        elif cue_type == "color":
            resolution_mode = "color_only"

    reference_facts["reference_resolution_mode"] = resolution_mode
    reference_facts["instruction_cohort_size"] = max(ambiguity_count, 1)
    reference_facts["same_semantic_group_count_total"] = max(ambiguity_count, 1)
    reference_facts["same_semantic_group_same_color_count"] = int(
        truth_repair_metadata.get("human_visible_same_color_count") or 0
    )
    reference_facts["same_semantic_group_diff_color_count"] = int(
        truth_repair_metadata.get("human_visible_diff_color_count") or 0
    )
    reference_facts["reference_top2_gap_m"] = float(
        truth_repair_metadata.get("reference_top2_gap_m")
        or reference_facts.get("reference_top2_gap_m")
        or 999.0
    )
    return reference_facts


def build_truth_repair_metadata(
    target: dict,
    candidates: list[dict],
    *,
    bundle: dict,
    source_row: dict | None = None,
    h_view: dict | None = None,
) -> dict:
    protocol = get_truth_repair_protocol()
    cue_protocol = get_cue_repair_protocol()
    hview_protocol = get_hview_confusable_universe_protocol()
    same_type_key = truth_repair_category_key(target)
    target_id = int(target["target_id"])
    h_view = h_view or build_h_view(
        bundle,
        routing_rows=bundle.get("routing_candidates", []),
        proxy_group_map={},
        same_type_mode="canonical_category",
    )
    same_type_objects = [
        obj for obj in h_view.get("h_view_objects", []) if obj.get("same_type_key") == same_type_key
    ]
    target_h_view = next(
        (obj for obj in same_type_objects if int(obj.get("object_id", -1)) == target_id),
        None,
    )
    supported_visible_objects = [
        obj
        for obj in same_type_objects
        if obj.get("support_status") == "supported" and obj.get("visibility_status") == "visible"
    ]
    supported_visible_ids = [int(obj["object_id"]) for obj in supported_visible_objects]
    ambiguity_count = len({int(obj["object_id"]) for obj in supported_visible_objects})

    target_supported_visible = (
        target_h_view is not None
        and target_h_view.get("support_status") == "supported"
        and target_h_view.get("visibility_status") == "visible"
    )
    has_non_supported_same_type = any(
        obj.get("support_status") != "supported" or obj.get("visibility_status") != "visible"
        for obj in same_type_objects
    )
    target_support_evidence_sources = list(target_h_view.get("evidence_sources") or []) if target_h_view is not None else []
    competitor_resolution_decisions = []
    for obj in same_type_objects:
        object_id = int(obj.get("object_id", -1))
        if object_id == target_id:
            continue
        is_supported_visible = (
            obj.get("support_status") == "supported"
            and obj.get("visibility_status") == "visible"
        )
        competitor_resolution_decisions.append(
            {
                "object_id": object_id,
                "support_status": obj.get("support_status"),
                "visibility_status": obj.get("visibility_status"),
                "same_type_key": obj.get("same_type_key"),
                "evidence_sources": list(obj.get("evidence_sources") or []),
                "resolution": "kept_supported_visible" if is_supported_visible else "kept_unresolved",
            }
        )
    if not target_supported_visible:
        support_failure_reason = "target_not_supported_visible"
    elif has_non_supported_same_type:
        support_failure_reason = "competitor_not_supported_visible"
    else:
        support_failure_reason = "none"
    support_visibility_resolution_status = (
        "resolved_supported_visible"
        if target_supported_visible and not has_non_supported_same_type
        else "unresolved"
    )
    support_visibility_fail_closed = support_visibility_resolution_status == "unresolved"
    candidate_by_target_id = {
        int(candidate["target_id"]): candidate
        for candidate in candidates
        if truth_repair_category_key(candidate) == same_type_key
    }

    protocol_snapshot = truth_repair_protocol_runtime_snapshot(protocol)

    distance_map: dict[int, float] = {}
    depth_map: dict[int, float] = {}
    anchor_map: dict[int, str] = {}
    relation_side_map: dict[int, str] = {}
    for candidate_id, candidate in candidate_by_target_id.items():
        distance = _lookup_candidate_metric(candidate, ("distance", "distance_m", "target_distance_m"))
        if distance is not None:
            distance_map[candidate_id] = distance
        depth = _lookup_candidate_metric(candidate, ("depth", "depth_m", "camera_depth_m", "target_depth_m"))
        if depth is not None:
            depth_map[candidate_id] = depth
        anchor_id = _lookup_relation_anchor_id(candidate)
        if anchor_id is not None:
            anchor_map[candidate_id] = anchor_id
        relation_side = _normalized_optional_text(candidate.get("relation_side"))
        if relation_side in {"left", "right", "front", "back"}:
            relation_side_map[candidate_id] = relation_side

    target_distance = _lookup_candidate_metric(target, ("distance", "distance_m", "target_distance_m"))
    if target_distance is None:
        target_distance = distance_map.get(target_id)
    target_depth = _lookup_candidate_metric(target, ("depth", "depth_m", "camera_depth_m", "target_depth_m"))
    if target_depth is None:
        target_depth = depth_map.get(target_id)
    target_anchor_id = _lookup_relation_anchor_id(target) or anchor_map.get(target_id)
    target_relation_side = _normalized_optional_text(target.get("relation_side")) or relation_side_map.get(target_id)
    resolved_relation_side = target_relation_side
    why_not_relation = "not_applicable"

    near_far_tie = False
    front_back_tie = False
    if target_supported_visible and not has_non_supported_same_type and ambiguity_count > 1:
        near_far_margin_m = float(protocol_snapshot["near_far_margin_m"])
        front_back_margin_m = float(protocol_snapshot["front_back_margin_m"])
        if target_distance is not None:
            near_far_tie = any(
                candidate_id != target_id
                and candidate_id in supported_visible_ids
                and candidate_distance is not None
                and abs(candidate_distance - target_distance) <= near_far_margin_m
                for candidate_id, candidate_distance in distance_map.items()
            )
        if target_depth is not None:
            front_back_tie = any(
                candidate_id != target_id
                and candidate_id in supported_visible_ids
                and candidate_depth is not None
                and abs(candidate_depth - target_depth) <= front_back_margin_m
                for candidate_id, candidate_depth in depth_map.items()
            )

    cue_type = "none"
    cue_value = None
    prompt_repair_status = "unresolved"
    prompt_truth_tier = "reject"
    has_unique_color = False
    has_unique_relation_anchor = False
    has_unique_spatial = False
    same_color_count = 0
    diff_color_count = 0
    spatial_ready = bool(
        target_supported_visible and not has_non_supported_same_type and not near_far_tie and not front_back_tie
    )

    if spatial_ready:
        if ambiguity_count == 1:
            prompt_repair_status = "not_needed"
            prompt_truth_tier = "benchmark_grade"
            has_unique_spatial = True
        elif ambiguity_count > 1:
            has_spatial_measurement = bool(
                (target_distance is not None or target_depth is not None)
                and any(
                    candidate_id != target_id and (
                        (target_distance is not None and candidate_id in distance_map)
                        or (target_depth is not None and candidate_id in depth_map)
                    )
                    for candidate_id in supported_visible_ids
                )
            )
            has_unique_spatial = has_spatial_measurement
            relation_resolution_ready = all(
                candidate_id in candidate_by_target_id for candidate_id in supported_visible_ids
            )
            derived_relation_side = None
            if relation_resolution_ready:
                derived_relation_side = _derive_extreme_relation_side(
                    target_id=target_id,
                    supported_visible_ids=supported_visible_ids,
                    supported_visible_objects=supported_visible_objects,
                    left_right_margin_px=float(protocol_snapshot["left_right_margin_px"]),
                    front_back_margin_m=float(protocol_snapshot["front_back_margin_m"]),
                    near_far_margin_m=float(protocol_snapshot["near_far_margin_m"]),
                    depth_map=depth_map,
                    distance_map=distance_map,
                )
            if relation_resolution_ready and target_anchor_id is not None:
                anchor_target_ids = [
                    candidate_id
                    for candidate_id in supported_visible_ids
                    if anchor_map.get(candidate_id) == target_anchor_id
                ]
                has_unique_relation_anchor = (
                    len(anchor_target_ids) == 1
                    and int(anchor_target_ids[0]) == target_id
                    and _directional_relation_side_allowed(target_relation_side, cue_protocol)
                )
                if has_unique_relation_anchor:
                    resolved_relation_side = target_relation_side
                elif target_relation_side in {"left", "right", "front", "back"}:
                    why_not_relation = "directional_relations_disabled_in_v1"
            if (
                not has_unique_relation_anchor
                and derived_relation_side in {"left", "right", "front", "back"}
                and _directional_relation_side_allowed(derived_relation_side, cue_protocol)
            ):
                has_unique_relation_anchor = True
                resolved_relation_side = derived_relation_side
            elif (
                not has_unique_relation_anchor
                and derived_relation_side in {"left", "right", "front", "back"}
            ):
                why_not_relation = "directional_relations_disabled_in_v1"
            target_color = _normalized_cue_color(target.get("target_color_name"))
            color_target_ids: list[int] = []
            color_resolution_ready = target_color is not None
            same_color_count = 0
            diff_color_count = 0
            for obj in supported_visible_objects:
                candidate = candidate_by_target_id.get(int(obj["object_id"]))
                if candidate is None:
                    color_resolution_ready = False
                    break
                candidate_color = _normalized_cue_color(candidate.get("target_color_name"))
                if candidate_color is None:
                    color_resolution_ready = False
                    break
                if candidate_color == target_color:
                    color_target_ids.append(int(candidate["target_id"]))
                    if int(candidate["target_id"]) != target_id:
                        same_color_count += 1
                elif int(candidate["target_id"]) != target_id:
                    diff_color_count += 1
            if color_resolution_ready and len(color_target_ids) == 1 and color_target_ids[0] == target_id:
                has_unique_color = True
        else:
            same_color_count = 0
            diff_color_count = 0

    if target_distance is not None and ambiguity_count > 1:
        competitor_deltas = [
            abs(float(candidate_distance) - float(target_distance))
            for candidate_id, candidate_distance in distance_map.items()
            if candidate_id != target_id and candidate_id in supported_visible_ids
        ]
        reference_top2_gap_m = min(competitor_deltas) if competitor_deltas else 999.0
    else:
        reference_top2_gap_m = 999.0

    if not spatial_ready:
        same_color_count = 0
        diff_color_count = 0

    if has_unique_relation_anchor:
        cue_type = "side"
        cue_value = resolved_relation_side
        prompt_repair_status = "repaired"
        prompt_truth_tier = "benchmark_grade"
    elif why_not_relation == "not_applicable" and target_relation_side in {"left", "right", "front", "back"}:
        why_not_relation = "directional_relations_disabled_in_v1"
    elif has_unique_color:
        cue_type = "color"
        cue_value = target_color
        prompt_repair_status = "repaired"
        prompt_truth_tier = "benchmark_grade"

    task_family = (
        _normalized_optional_text((source_row or {}).get("task_family"))
        or _normalized_optional_text((source_row or {}).get("tier_family"))
        or _normalized_optional_text(target.get("task_family"))
        or "navigation"
    )
    goal_anchor_id = (
        (source_row or {}).get("goal_anchor_id")
        if (source_row or {}).get("goal_anchor_id") is not None
        else (
            (source_row or {}).get("goal_id")
            if (source_row or {}).get("goal_id") is not None
            else (target.get("goal_anchor_id") if target.get("goal_anchor_id") is not None else target.get("goal_id"))
        )
    )
    if goal_anchor_id is not None:
        try:
            goal_anchor_id = int(goal_anchor_id)
        except (TypeError, ValueError):
            goal_anchor_id = None
    routing_id_for_path = (
        _normalized_optional_text((source_row or {}).get("gt_path_id"))
        or _normalized_optional_text((source_row or {}).get("routing_id"))
        or _normalized_optional_text(target.get("gt_path_id"))
        or _normalized_optional_text(target.get("routing_id"))
    )
    route_semantics = (
        _normalized_optional_text((source_row or {}).get("route_semantics"))
        or _normalized_optional_text(target.get("route_semantics"))
        or ("embodied" if _normalized_optional_text((source_row or {}).get("task")) == "b2" else "pointmass")
    )
    if routing_id_for_path is None:
        routing_id_for_path = f"target_{target_id}"
    gt_path_id = (
        routing_id_for_path
        if "::" in routing_id_for_path
        else f"{routing_id_for_path}::{route_semantics}"
    )

    if target_h_view is not None and target_h_view.get("support_status") == "contradicted":
        human_visible_support = "contradicted"
    elif target_supported_visible and not has_non_supported_same_type:
        human_visible_support = "supported"
    else:
        human_visible_support = "uncertain"

    if (
        not target_supported_visible
        or has_non_supported_same_type
        or near_far_tie
        or front_back_tie
    ):
        human_visible_uniqueness = "uncertain"
    elif ambiguity_count <= 1 or has_unique_color or has_unique_relation_anchor:
        human_visible_uniqueness = "unique"
    else:
        human_visible_uniqueness = "ambiguous"

    fingerprint_leakage = bool(
        has_unique_color
        and has_unique_spatial
        and has_unique_relation_anchor
    )
    leakage_text = _normalized_optional_text(cue_value)
    prompt_answer_leakage = bool(
        leakage_text
        and (
            leakage_text.startswith("[")
            or leakage_text.endswith("]")
            or "path" in leakage_text
            or "display id" in leakage_text
        )
    )
    if fingerprint_leakage or prompt_answer_leakage:
        cue_type = "none"
        cue_value = None
        prompt_repair_status = "unresolved"
        prompt_truth_tier = "reject"
        human_visible_support = "uncertain"
        human_visible_uniqueness = "uncertain"
        has_unique_color = False
        has_unique_spatial = False
        has_unique_relation_anchor = False

    scene_id = (source_row or {}).get("scene_id")
    view_id = (source_row or {}).get("view_id")
    routing_id = (
        _normalized_optional_text((source_row or {}).get("routing_id"))
        or _normalized_optional_text(target.get("routing_id"))
    )
    human_visible_legality = derive_human_visible_legality(
        human_visible_support=human_visible_support,
        human_visible_uniqueness=human_visible_uniqueness,
        prompt_truth_tier=prompt_truth_tier,
        prompt_answer_leakage=prompt_answer_leakage,
    )
    goal_anchor_value = goal_anchor_id
    gt_path_value = gt_path_id
    source_group_id = infer_source_group_id(scene_id)
    human_visible_sidecar_ref = build_h_view_sidecar_relpath(
        scene_id=str(scene_id) if scene_id is not None else None,
        view_id=int(view_id) if view_id is not None else None,
        target_id=target_id,
    )
    human_visible_same_type_object_ids = sorted(
        {
            int(obj.get("object_id", -1))
            for obj in same_type_objects
            if int(obj.get("object_id", -1)) >= 0
        }
    )
    supported_visible_same_type_ids = sorted(
        {
            int(obj.get("object_id", -1))
            for obj in supported_visible_objects
            if int(obj.get("object_id", -1)) >= 0
        }
    )
    uncertain_same_type_ids = sorted(
        {
            int(obj.get("object_id", -1))
            for obj in same_type_objects
            if int(obj.get("object_id", -1)) >= 0
            and not (
                obj.get("support_status") == "supported"
                and obj.get("visibility_status") == "visible"
            )
        }
    )
    unique_by_relation = bool(has_unique_relation_anchor and prompt_repair_status == "repaired")
    unique_by_attribute = False
    unique_by_color = bool(has_unique_color and prompt_repair_status == "repaired")
    unique_by_distance = bool(
        _normalized_optional_text(cue_type) == "distance_rank"
        and prompt_repair_status == "repaired"
    )
    natural_unique_under_hview = bool(
        target_supported_visible
        and not has_non_supported_same_type
        and ambiguity_count <= 1
        and prompt_repair_status == "not_needed"
        and prompt_truth_tier != "reject"
    )
    fixed_target_supported_under_hview = bool(target_supported_visible)
    prompt_unique_under_hview = bool(
        human_visible_support == "supported"
        and human_visible_uniqueness == "unique"
        and prompt_truth_tier != "reject"
        and not prompt_answer_leakage
    )
    cue_family = "none"
    normalized_cue_type = _normalized_optional_text(cue_type)
    if unique_by_relation or normalized_cue_type == "side":
        cue_family = "relation"
    elif unique_by_attribute:
        cue_family = "attribute"
    elif unique_by_color or normalized_cue_type == "color":
        cue_family = "color"
    elif unique_by_distance or normalized_cue_type == "distance_rank":
        cue_family = "distance"

    if prompt_unique_under_hview:
        promptability_status = "natural_unique" if natural_unique_under_hview else "cue_unique"
    elif not fixed_target_supported_under_hview:
        promptability_status = "target_not_supported"
    elif human_visible_uniqueness == "ambiguous":
        promptability_status = "ambiguous"
    elif human_visible_uniqueness == "uncertain":
        promptability_status = "uncertain"
    else:
        promptability_status = "not_promptable"

    promptability_resolution_trace = {
        "cue_family": cue_family,
        "cue_type": cue_type,
        "cue_value": cue_value,
        "prompt_repair_status": prompt_repair_status,
        "prompt_truth_tier": prompt_truth_tier,
        "human_visible_support": human_visible_support,
        "human_visible_uniqueness": human_visible_uniqueness,
        "ambiguity_count_H_view": int(ambiguity_count),
        "support_failure_reason": support_failure_reason,
        "near_far_tie": bool(near_far_tie),
        "front_back_tie": bool(front_back_tie),
        "why_not_relation": why_not_relation,
    }

    immutable_identity = {
        "scene_id": scene_id,
        "view_id": int(view_id) if view_id is not None else None,
        "routing_id": routing_id,
        "target_id": target_id,
        "goal_anchor_id": int(goal_anchor_value) if goal_anchor_value is not None else None,
        "gt_path_id": gt_path_value,
    }

    return {
        "task_family": task_family,
        "goal_anchor_id": goal_anchor_id,
        "gt_path_id": gt_path_id,
        "fixed_target_contract_version": FIXED_TARGET_CONTRACT_VERSION,
        "fixed_gt_identity_hash": derive_fixed_gt_identity_hash_from_identity(immutable_identity),
        "source_group_id": source_group_id,
        "cue_repair_protocol_version": cue_protocol["schema_version"],
        "cue_repair_protocol_hash": _compute_hash(cue_protocol),
        "hview_confusable_universe_version": hview_protocol["schema_version"],
        "hview_confusable_universe_hash": _compute_hash(hview_protocol),
        "schema_bundle_hash": get_schema_bundle_hash(),
        "human_visible_protocol_version": "hview_uview_v2",
        "human_visible_legality": human_visible_legality,
        "human_visible_sidecar_ref": human_visible_sidecar_ref,
        "benchmark_unit_type": "route_bundle",
        "immutable_identity": immutable_identity,
        "human_visible_support": human_visible_support,
        "human_visible_uniqueness": human_visible_uniqueness,
        "has_unique_color": has_unique_color,
        "has_unique_spatial": has_unique_spatial,
        "has_unique_relation_anchor": has_unique_relation_anchor,
        "human_visible_same_type_object_ids": human_visible_same_type_object_ids,
        "supported_visible_same_type_ids": supported_visible_same_type_ids,
        "uncertain_same_type_ids": uncertain_same_type_ids,
        "human_visible_same_color_count": int(same_color_count),
        "human_visible_diff_color_count": int(diff_color_count),
        "reference_top2_gap_m": float(reference_top2_gap_m),
        "fixed_target_supported_under_hview": fixed_target_supported_under_hview,
        "natural_unique_under_hview": natural_unique_under_hview,
        "unique_by_relation": unique_by_relation,
        "unique_by_attribute": unique_by_attribute,
        "unique_by_color": unique_by_color,
        "unique_by_distance": unique_by_distance,
        "prompt_unique_under_hview": prompt_unique_under_hview,
        "promptability_status": promptability_status,
        "promptability_resolution_trace": promptability_resolution_trace,
        "cue_family": cue_family,
        "fixed_target_prompt_cue_bundle": {
            "cue_family": cue_family,
            "cue_type": cue_type,
            "cue_value": cue_value,
        },
        "fingerprint_leakage": fingerprint_leakage,
        "prompt_answer_leakage": prompt_answer_leakage,
        "target_lock_id": target_id,
        "target_lock_category": target["canonical_category"],
        "same_type_key": same_type_key,
        "ambiguity_count_H_view": int(ambiguity_count),
        "target_supported_visible": bool(target_supported_visible),
        "same_type_supported_visible_count": int(ambiguity_count),
        "has_non_supported_same_type": bool(has_non_supported_same_type),
        "support_failure_reason": support_failure_reason,
        "support_visibility_resolution_status": support_visibility_resolution_status,
        "target_support_evidence_sources": target_support_evidence_sources,
        "competitor_resolution_decisions": competitor_resolution_decisions,
        "support_visibility_fail_closed": support_visibility_fail_closed,
        "cue_type": cue_type,
        "cue_value": cue_value,
        "prompt_repair_status": prompt_repair_status,
        "prompt_truth_tier": prompt_truth_tier,
        "target_lock_preserved": True,
        "left_right_margin_px": protocol_snapshot["left_right_margin_px"],
        "ordinal_margin_px": protocol_snapshot["ordinal_margin_px"],
        "front_back_margin_m": protocol_snapshot["front_back_margin_m"],
        "near_far_margin_m": protocol_snapshot["near_far_margin_m"],
        "max_cues": protocol_snapshot["max_cues"],
        "max_relation_cues": protocol_snapshot["max_relation_cues"],
        "max_prompt_delta_sentences": protocol_snapshot["max_prompt_delta_sentences"],
        "max_prompt_delta_tokens": protocol_snapshot["max_prompt_delta_tokens"],
        "max_cue_tokens_per_cue": protocol_snapshot["max_cue_tokens_per_cue"],
        "max_attribute_slots": protocol_snapshot["max_attribute_slots"],
        "max_relation_slots": protocol_snapshot["max_relation_slots"],
        "truth_repair_protocol": protocol_snapshot,
        "truth_repair_protocol_snapshot": protocol_snapshot,
    }


def apply_resolved_prompt_package(
    rec: dict,
    *,
    resolved_cue_package: dict,
    prompt_resolution_trace: dict,
) -> dict:
    repaired = dict(rec)
    cue_family = _normalized_optional_text(resolved_cue_package.get("cue_family")) or "drop"
    cue_type = _normalized_optional_text(resolved_cue_package.get("cue_type")) or "none"
    cue_value = resolved_cue_package.get("cue_value")
    anchor_category = _normalized_optional_text(resolved_cue_package.get("anchor_category"))
    cue_match_count = resolved_cue_package.get("cue_match_count")
    residual_cue_type = _normalized_optional_text(resolved_cue_package.get("residual_cue_type"))
    residual_cue_value = resolved_cue_package.get("residual_cue_value")
    minimal_residual_cue = bool(resolved_cue_package.get("minimal_residual_cue"))
    if cue_match_count is not None:
        cue_match_count = int(cue_match_count)
    resolver_result = _normalized_optional_text(prompt_resolution_trace.get("resolver_result")) or "drop"
    prompt_pass = resolver_result == "pass"

    if cue_family == "natural_unique":
        prompt_repair_status = "not_needed" if prompt_pass else "unresolved"
        cue_type = "none"
        cue_value = None
        promptability_status = "natural_unique" if prompt_pass else "not_promptable"
    else:
        prompt_repair_status = "repaired" if prompt_pass else "unresolved"
        promptability_status = "cue_unique" if prompt_pass else "not_promptable"

    fixed_target_prompt_cue_bundle = {
        "cue_family": cue_family,
        "cue_type": cue_type,
        "cue_value": cue_value,
        "anchor_category": anchor_category,
        "cue_match_count": cue_match_count,
        "residual_cue_type": residual_cue_type,
        "residual_cue_value": residual_cue_value,
        "minimal_residual_cue": minimal_residual_cue,
    }

    repaired.update(
        {
            "cue_family": cue_family,
            "cue_type": cue_type,
            "cue_value": cue_value,
            "residual_cue_type": residual_cue_type,
            "residual_cue_value": residual_cue_value,
            "minimal_residual_cue": minimal_residual_cue,
            "fixed_target_prompt_cue_bundle": fixed_target_prompt_cue_bundle,
            "promptability_resolution_trace": dict(prompt_resolution_trace),
            "prompt_unique_under_hview": bool(prompt_pass),
            "prompt_repair_status": prompt_repair_status,
            "prompt_truth_tier": "benchmark_grade" if prompt_pass else "reject",
            "promptability_status": promptability_status,
            "target_lock_preserved": True,
        }
    )

    if _normalized_optional_text(repaired.get("task")) == "c":
        repaired["route_source_task"] = "b2"
        repaired["route_semantics"] = "embodied"
        implicit_resolution_mode = _normalized_optional_text(
            prompt_resolution_trace.get("implicit_resolution_mode")
        ) or _normalized_optional_text(repaired.get("implicit_resolution_mode"))
        if implicit_resolution_mode is not None:
            repaired["implicit_resolution_mode"] = implicit_resolution_mode

    return repaired


def targets_match_semantic_goal(predicted: dict, accepted: dict) -> bool:
    pred_group = (
        _normalized_optional_text(predicted.get("reference_family_id"))
        or _normalized_optional_text(predicted.get("semantic_group_id"))
    )
    acc_group = (
        _normalized_optional_text(accepted.get("reference_family_id"))
        or _normalized_optional_text(accepted.get("semantic_group_id"))
    )
    if pred_group and acc_group:
        if pred_group != acc_group:
            return False
        acc_color = (
            _normalized_optional_text(accepted.get("reference_color_name"))
            or _normalized_optional_text(accepted.get("visible_color_name"))
            or _normalized_optional_text(accepted.get("target_color_name"))
        )
        if acc_color is None:
            return True
        pred_color = (
            _normalized_optional_text(predicted.get("reference_color_name"))
            or _normalized_optional_text(predicted.get("visible_color_name"))
            or _normalized_optional_text(predicted.get("target_color_name"))
        )
        return pred_color == acc_color
    try:
        if int(predicted.get("target_id", -1)) != int(accepted.get("target_id", -1)):
            return False
        acc_color = (
            _normalized_optional_text(accepted.get("reference_color_name"))
            or _normalized_optional_text(accepted.get("visible_color_name"))
            or _normalized_optional_text(accepted.get("target_color_name"))
        )
        if acc_color is None:
            return True
        pred_color = (
            _normalized_optional_text(predicted.get("reference_color_name"))
            or _normalized_optional_text(predicted.get("visible_color_name"))
            or _normalized_optional_text(predicted.get("target_color_name"))
        )
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


def resolve_task_view_dir(
    render_scene_dir: Path,
    task_outputs_root: Path | None,
    scene_id: str,
    view_id: int,
) -> Path:
    if task_outputs_root is not None:
        return task_outputs_root / scene_id / f"view_{int(view_id)}"
    return render_scene_dir / f"view_{int(view_id)}"


def load_required_visible_waypoints(
    view_dir: Path,
    scene_id: str,
    view_id: int,
    tasks: tuple[str, ...] = ("a1", "b1", "a2", "b2", "c"),
) -> dict[str, list[dict]]:
    if not view_dir.exists():
        raise FileNotFoundError(
            f"scene={scene_id} view={view_id} missing task output view directory: {view_dir}"
        )

    loaded: dict[str, list[dict]] = {}
    for task in tasks:
        path = view_dir / f"visible_waypoints_{task}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"scene={scene_id} view={view_id} missing visible_waypoints_{task}.json: {path}"
            )
        with open(path) as f:
            visible = json.load(f).get("visible_waypoints", [])
        if not visible:
            raise ValueError(
                f"scene={scene_id} view={view_id} has empty visible_waypoints payload for task={task}: {path}"
            )
        loaded[task] = visible
    return loaded


def load_task_outputs_payload(task_outputs_path: Path) -> dict:
    if not task_outputs_path.exists():
        raise FileNotFoundError(f"missing required task outputs payload: {task_outputs_path}")
    with open(task_outputs_path) as f:
        payload = json.load(f)
    classification = payload.get("classification")
    if not isinstance(classification, dict):
        raise ValueError(f"task outputs missing classification payload: {task_outputs_path}")
    views = classification.get("views")
    if not isinstance(views, list):
        raise ValueError(f"task outputs missing classification.views list: {task_outputs_path}")
    return payload


def index_classification_views(task_outputs_payload: dict) -> dict[int, dict]:
    indexed: dict[int, dict] = {}
    classification = task_outputs_payload.get("classification") or {}
    for view in classification.get("views", []) or []:
        view_id = int(view.get("view_id", -1))
        if view_id >= 0:
            indexed[view_id] = dict(view)
    return indexed


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
        require_continuous_optimal_routing_candidate(normalized)
        selection = require_rule_winner_instruction_selection(normalized)
        if normalized.get("pruned_reason") not in (None, ""):
            continue
        if str(selection.get("selection_status")) != "unique_winner":
            continue
        if not bool(selection.get("is_rule_winner", False)):
            continue
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
        task_outputs_payload = load_task_outputs_payload(task_outputs_path)
        classification_views_by_id = index_classification_views(task_outputs_payload)

        obj_map = {}
        for obj in layout:
            if "id" in obj:
                obj_map[int(obj["id"])] = obj

        for cam in cameras:
            view_id = cam.get("view_id", 0)
            view_dir = resolve_task_view_dir(
                render_scene_dir=render_scene_dir,
                task_outputs_root=task_outputs_root,
                scene_id=scene_id,
                view_id=int(view_id),
            )

            vp_a1 = view_dir / "visible_waypoints_a1.json"
            vp_b1 = view_dir / "visible_waypoints_b1.json"
            vp_a2 = view_dir / "visible_waypoints_a2.json"
            vp_b2 = view_dir / "visible_waypoints_b2.json"
            vp_c = view_dir / "visible_waypoints_c.json"

            bundle = load_view_bundle_for_view(render_scene_dir, int(view_id))
            h_view = build_h_view(
                bundle,
                routing_rows=bundle.get("routing_candidates", []),
                proxy_group_map={},
                same_type_mode="canonical_category",
            )
            h_view_relpath = build_h_view_sidecar_relpath(
                scene_id=scene_id,
                view_id=int(view_id),
                target_id=-1,
            )
            write_h_view_sidecar(output_dir / h_view_relpath, h_view)
            facts_hash = task_outputs_payload.get("facts_hash") or bundle.get("facts_hash")
            classification_view = classification_views_by_id.get(int(view_id))
            if not isinstance(classification_view, dict):
                raise ValueError(
                    f"scene={scene_id} view={view_id} missing protocolized classification view in {task_outputs_path}"
                )
            affordance_validity = classification_view.get("affordance_validity")
            affordance_axes = classification_view.get("affordance_axes")
            affordance_tier = classification_view.get("affordance_tier")
            if not isinstance(affordance_validity, dict):
                raise ValueError(
                    f"scene={scene_id} view={view_id} missing affordance_validity in {task_outputs_path}"
                )
            if not isinstance(affordance_axes, dict):
                raise ValueError(
                    f"scene={scene_id} view={view_id} missing affordance_axes in {task_outputs_path}"
                )
            if affordance_tier is None:
                raise ValueError(
                    f"scene={scene_id} view={view_id} missing affordance_tier in {task_outputs_path}"
                )
            if not bool(affordance_validity.get("eligible", False)):
                raise ValueError(
                    f"scene={scene_id} view={view_id} has ineligible affordance payload: "
                    f"{affordance_validity.get('failed_checks', [])}"
                )

            visible_payloads = load_required_visible_waypoints(
                view_dir=view_dir,
                scene_id=scene_id,
                view_id=int(view_id),
            )
            visible_a1 = visible_payloads["a1"]
            visible_b1 = visible_payloads["b1"]
            visible_a2 = visible_payloads["a2"]
            visible_b2 = visible_payloads["b2"]
            visible_c = visible_payloads["c"]

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
                require_continuous_optimal_routing_candidate(t)
                path_wp_ids_pointmass = [int(x) for x in t.get("gt_path_waypoint_ids_pointmass", [])]
                path_wp_ids_embodied = [int(x) for x in t.get("gt_path_waypoint_ids_embodied", [])]
                optimal_path_world = t.get("optimal_path_world") or []
                canonical_sparse_path_world = t.get("canonical_sparse_path_world") or []
                optimal_path_world_embodied = t.get("optimal_path_world_embodied") or []
                canonical_sparse_path_world_embodied = t.get("canonical_sparse_path_world_embodied") or []
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
                    "reference_family_id": t.get("reference_family_id", t.get("semantic_group_id", t.get("canonical_category", obj.get("category", "unknown")))),
                    "reference_color_name": t.get("reference_color_name"),
                    "visible_color_name": t.get("visible_color_name"),
                    "target_color_name": (
                        t.get("reference_color_name")
                        or t.get("visible_color_name")
                        or t.get("target_color_name")
                        or t.get("material_color_name")
                    ),
                    "distance": dist,
                    "world_xyz": [float(pos[0]), float(pos[1]), float(pos[2])],
                    "screen_x": screen_x,
                    "optimal_path_world": optimal_path_world,
                    "optimal_length_m": float(t.get("optimal_length_m", 0.0)),
                    "canonical_sparse_path_world": canonical_sparse_path_world,
                    "canonical_sparse_length_m": float(t.get("canonical_sparse_length_m", 0.0)),
                    "optimal_path_world_embodied": optimal_path_world_embodied,
                    "optimal_length_m_embodied": float(t.get("optimal_length_m_embodied", 0.0)),
                    "canonical_sparse_path_world_embodied": canonical_sparse_path_world_embodied,
                    "canonical_sparse_length_m_embodied": float(t.get("canonical_sparse_length_m_embodied", 0.0)),
                    "bbox": bbox,
                    "gt_path_waypoint_ids_pointmass": path_wp_ids_pointmass,
                    "gt_path_waypoint_ids_embodied": path_wp_ids_embodied,
                    "instruction_identity": t.get("instruction_identity"),
                    "instruction_selection": t.get("instruction_selection"),
                    "routing_complexity": t.get("routing_complexity"),
                    "navigation_reference_facts": t.get("navigation_reference_facts"),
                    "navigation_geometry_facts": t.get("navigation_geometry_facts"),
                    "navigation_embodiment_facts": t.get("navigation_embodiment_facts"),
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
                navigation_axes = dict(tgt.get("navigation_axes") or {})
                navigation_tier = tgt.get("navigation_tier")
                navigation_reference_facts = dict(tgt.get("navigation_reference_facts") or {})
                navigation_geometry_facts = tgt.get("navigation_geometry_facts")
                navigation_embodiment_facts = tgt.get("navigation_embodiment_facts")
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
                truth_repair_metadata = build_truth_repair_metadata(
                    tgt,
                    cands,
                    bundle=bundle,
                    source_row={
                        "scene_id": scene_id,
                        "view_id": view_id,
                        "routing_id": routing_id,
                        "goal_anchor_id": goal_id_a2,
                        "gt_path_id": f"{routing_id}::pointmass",
                    },
                    h_view=h_view,
                )
                navigation_reference_facts = build_prompt_aligned_navigation_reference_facts(
                    original_reference_facts=navigation_reference_facts,
                    truth_repair_metadata=truth_repair_metadata,
                )
                navigation_axes["reference_axis"] = classify_reference_axis(navigation_reference_facts)
                navigation_tier = classify_navigation_tier(
                    reference_axis=navigation_axes["reference_axis"],
                    geometry_axis=str(navigation_axes.get("geometry_axis") or "low"),
                    embodiment_axis=str(navigation_axes.get("embodiment_axis") or "medium"),
                )
                fixed_gt_source_refs_common = {
                    "cameras_path": str(cameras_path),
                    "view_bundle_path": str(view_dir / VIEW_BUNDLE_FILENAME),
                    "task_outputs_path": str(task_outputs_path),
                }
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
                    "reference_family_id": tgt.get("reference_family_id"),
                    "target_color_name": tgt["target_color_name"],
                    "target_distance_m": round(tgt["distance"], 3),
                    "instruction_identity": tgt.get("instruction_identity"),
                    "instruction_selection": tgt.get("instruction_selection"),
                    "routing_complexity": routing_complexity,
                    "navigation_reference_facts": navigation_reference_facts,
                    "navigation_geometry_facts": navigation_geometry_facts,
                    "navigation_embodiment_facts": navigation_embodiment_facts,
                    "navigation_axes": navigation_axes,
                    "navigation_validity": navigation_validity,
                    "facts_hash": tgt.get("facts_hash"),
                    **truth_repair_metadata,
                }

                rec_a2 = {
                    **rec_common,
                    "question_id": f"{qstem}_a2",
                    "task": "a2",
                    "image_path": str(view_dir / "rgb_overlay_a2.png"),
                    "visible_waypoints_path": str(vp_a2),
                    "start_id": int(disp_path_a2[0]),
                    "goal_id": goal_id_a2,
                    "goal_ids": sorted(goal_ids_a2),
                    "path_ids": disp_path_a2,
                    "optimal_length_m": round(float(tgt.get("optimal_length_m", 0.0)), 3),
                    "canonical_sparse_length_m": round(float(tgt.get("canonical_sparse_length_m", 0.0)), 3),
                    "fixed_gt_source_refs": {
                        **fixed_gt_source_refs_common,
                        "visible_waypoints_path": str(vp_a2),
                    },
                }
                rec_b2 = {
                    **rec_common,
                    "question_id": f"{qstem}_b2",
                    "task": "b2",
                    "image_path": str(view_dir / "rgb_overlay_b2.png"),
                    "visible_waypoints_path": str(vp_b2),
                    "start_id": int(disp_path_b2[0]),
                    "goal_id": goal_id_b2,
                    "goal_ids": sorted(goal_ids_b2),
                    "path_ids": disp_path_b2,
                    "robot_diameter_m": 0.6,
                    "optimal_length_m": round(float(tgt.get("optimal_length_m_embodied", 0.0)), 3),
                    "canonical_sparse_length_m": round(float(tgt.get("canonical_sparse_length_m_embodied", 0.0)), 3),
                    "fixed_gt_source_refs": {
                        **fixed_gt_source_refs_common,
                        "visible_waypoints_path": str(vp_b2),
                    },
                }

                rec_c = {
                    **rec_common,
                    "question_id": f"{qstem}_c",
                    "task": "c",
                    "image_path": str(view_dir / "rgb_overlay_c.png"),
                    "visible_waypoints_path": str(vp_c),
                    "start_id": int(disp_path_a2[0]),
                    "goal_id": goal_id_a2,
                    "goal_ids": sorted(goal_ids_a2),
                    "path_ids": disp_path_a2,
                    "optimal_length_m": round(float(tgt.get("optimal_length_m", 0.0)), 3),
                    "canonical_sparse_length_m": round(float(tgt.get("canonical_sparse_length_m", 0.0)), 3),
                    "fixed_gt_source_refs": {
                        **fixed_gt_source_refs_common,
                        "visible_waypoints_path": str(vp_c),
                    },
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
