#!/usr/bin/env python3
"""
Build multi-task outputs for A1/A2/B1/B2/C:
  - waypoint_core.json / waypoint_graph.json
  - per-view visible_waypoints_{a1,a2,b1,b2,c}.json
  - per-view rgb_overlay_{a1,a2,b1,b2,c}.png
  - per-target path_target_<id>.png (red path + highlighted target)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont, ImageFilter

from build_navigation_gt import build_occupancy_grid
from object_index_utils import decode_object_index_array
from scene_nav_facts import SCENE_NAV_FACTS_SCHEMA, load_scene_nav_facts


DEPTH_MAX_METERS = 20.0  # must match scripts/render_scenes.py
DEFAULT_OBSERVATION_FILENAME = "next_observation_v1.json"
DEFAULT_TASK_OUTPUTS_FILENAME = "next_task_outputs_v1.json"
VIEW_BUNDLE_FILENAME = "next_view_bundle_v3.json"
VIEW_BUNDLE_SCHEMA = "next_view_bundle_v3"
DEFAULT_DENSE_SPACING_M = 0.4
DEFAULT_SPARSE_COUNT = 50
DEFAULT_MIN_PIXEL_DIST = 22
DEFAULT_MAX_VISIBLE = 200

TASK_COLORS = {
    "a1": (255, 80, 80),
    "b1": (80, 200, 80),
    "a2": (80, 140, 255),
    "b2": (255, 220, 80),
    "c": (180, 180, 180),
}


def get_observation_filename(config: dict | None = None) -> str:
    return (config or {}).get("data", {}).get("observation_pack_filename", DEFAULT_OBSERVATION_FILENAME)


def get_task_outputs_filename(config: dict | None = None) -> str:
    return (config or {}).get("data", {}).get("task_outputs_filename", DEFAULT_TASK_OUTPUTS_FILENAME)


def load_view_bundle_v3(scene_out: Path, view_id: int) -> dict:
    bundle_path = scene_out / f"view_{view_id}" / VIEW_BUNDLE_FILENAME
    if not bundle_path.exists():
        raise FileNotFoundError(f"missing required {VIEW_BUNDLE_FILENAME} for view {view_id}")
    with open(bundle_path) as f:
        bundle = json.load(f)
    if bundle.get("schema_version") != VIEW_BUNDLE_SCHEMA:
        raise ValueError(
            f"{VIEW_BUNDLE_FILENAME} for view {view_id} must declare schema_version={VIEW_BUNDLE_SCHEMA}"
        )
    return bundle


def validate_render_scene_inputs(render_scene_dir: Path, cameras: list[dict]) -> None:
    for cam in cameras:
        view_id = int(cam.get("view_id", 0))
        view_dir = render_scene_dir / f"view_{view_id}"
        rgb_path = view_dir / "rgb.png"
        bundle_path = view_dir / VIEW_BUNDLE_FILENAME
        if not rgb_path.exists():
            raise FileNotFoundError(f"missing required rgb.png for view_{view_id}: {rgb_path}")
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"missing required {VIEW_BUNDLE_FILENAME} for view_{view_id}: {bundle_path}"
            )


def build_next_task_outputs(
    scene_id: str,
    cameras: list[dict],
    scene_out: Path,
    observation_pack_present: bool,
    observation_pack_filename: str = DEFAULT_OBSERVATION_FILENAME,
) -> dict:
    scene_facts = load_scene_nav_facts(scene_out)
    view_facts_by_id = {
        int(view.get("view_id", -1)): view
        for view in scene_facts.get("views", [])
    }
    routing_views = []
    classification_views = []
    for cam in cameras:
        view_id = int(cam.get("view_id", 0))
        bundle = load_view_bundle_v3(scene_out, view_id)
        view_facts = view_facts_by_id.get(view_id, {})
        routing_candidates = []
        for idx, rc in enumerate(bundle.get("routing_candidates", []), start=1):
            target_id = int(rc.get("target_id", -1))
            if target_id < 0:
                continue
            routing_candidates.append({
                "candidate_id": str(rc.get("routing_id", f"routing_v{view_id}_c{idx:03d}")),
                "display_id": idx,
                "target_id": target_id,
                "target_category": rc.get("target_category", "unknown"),
                "canonical_category": rc.get("canonical_category", rc.get("target_category", "unknown")),
                "semantic_group_id": rc.get("semantic_group_id", rc.get("canonical_category", rc.get("target_category", "unknown"))),
                "material_color_name": rc.get("material_color_name"),
                "distance_m": round(float(rc.get("distance_m", 0.0)), 3),
                "gt_path_display_ids_pointmass": list(rc.get("gt_path_display_ids_pointmass", [])),
                "gt_path_display_ids_embodied": list(rc.get("gt_path_display_ids_embodied", [])),
                "gt_path_waypoint_ids_pointmass": list(rc.get("gt_path_waypoint_ids_pointmass", [])),
                "gt_path_waypoint_ids_embodied": list(rc.get("gt_path_waypoint_ids_embodied", [])),
                "gt_path_keypoints_world": list(rc.get("gt_path_keypoints_world", [])),
                "debug_dense_path_world": list(rc.get("debug_dense_path_world", [])),
                "instruction_identity": dict(rc.get("instruction_identity", {})),
                "instruction_selection": dict(rc.get("instruction_selection", {})),
                "routing_complexity": dict(rc.get("routing_complexity", {})),
            })
        routing_views.append({
            "view_id": view_id,
            "candidate_count": len(routing_candidates),
            "routing_candidates": routing_candidates,
        })
        shared_ab_candidates = []
        for idx, candidate in enumerate(bundle.get("classification_candidates", []), start=1):
            shared_ab_candidates.append({
                "candidate_id": f"classification_v{view_id}_c{idx:03d}",
                "display_id": int(candidate.get("display_id", idx)),
                "waypoint_id": int(candidate.get("waypoint_id", -1)),
                "world_xyz": list(candidate.get("world_xyz", [])),
                "image_xy": list(candidate.get("image_xy", [])),
                "screen_xy_norm": list(candidate.get("screen_xy_norm", [])),
                "depth": float(candidate.get("depth", 0.0)),
                "pointmass_walkable": bool(candidate.get("pointmass_walkable", False)),
                "embodied_feasible": bool(candidate.get("embodied_feasible", False)),
                "candidate_source": candidate.get("candidate_source", "unknown"),
            })
        classification_views.append({
            "view_id": view_id,
            "candidate_count": len(shared_ab_candidates),
            "affordance_validity": dict(view_facts.get("affordance_validity", {})),
            "affordance_axes": dict(view_facts.get("affordance_axes", {})),
            "affordance_tier": view_facts.get("affordance_tier"),
            "affordance_complexity": dict(view_facts.get("affordance_complexity", {})),
            "shared_ab_candidates": shared_ab_candidates,
        })

    return {
        "schema_version": "next_task_outputs_v2",
        "facts_schema": SCENE_NAV_FACTS_SCHEMA,
        "facts_hash": scene_facts.get("facts_hash"),
        "scene_id": scene_id,
        "observation_pack": {
            "present": observation_pack_present,
            "filename": observation_pack_filename,
        },
        "task_families": {
            "routing": "materialized_v3",
            "classification": "materialized_v3",
        },
        "routing": {
            "source": VIEW_BUNDLE_FILENAME,
            "views": routing_views,
        },
        "classification": {
            "source": VIEW_BUNDLE_FILENAME,
            "views": classification_views,
        },
    }


def write_next_task_outputs(
    scene_out: Path,
    scene_id: str,
    cameras: list[dict],
    config: dict | None = None,
) -> Path:
    task_outputs = build_next_task_outputs(
        scene_id=scene_id,
        cameras=cameras,
        scene_out=scene_out,
        observation_pack_present=(scene_out / get_observation_filename(config)).exists(),
        observation_pack_filename=get_observation_filename(config),
    )
    out_path = scene_out / get_task_outputs_filename(config)
    with open(out_path, "w") as f:
        json.dump(task_outputs, f, indent=2)
    return out_path




def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_sampling_params(
    config: dict,
    *,
    dense_spacing: float | None,
    sparse_count: int | None,
    min_pixel_dist: int | None,
    max_visible: int | None,
) -> dict[str, float | int]:
    task_cfg = config.get("task_outputs", {})
    resolved_dense_spacing = (
        float(dense_spacing)
        if dense_spacing is not None
        else float(task_cfg.get("dense_spacing_m", DEFAULT_DENSE_SPACING_M))
    )
    resolved_sparse_count = (
        int(sparse_count)
        if sparse_count is not None
        else int(task_cfg.get("sparse_count", DEFAULT_SPARSE_COUNT))
    )
    resolved_min_pixel_dist = (
        int(min_pixel_dist)
        if min_pixel_dist is not None
        else int(task_cfg.get("min_pixel_dist", DEFAULT_MIN_PIXEL_DIST))
    )
    resolved_max_visible = (
        int(max_visible)
        if max_visible is not None
        else int(task_cfg.get("max_visible", DEFAULT_MAX_VISIBLE))
    )
    return {
        "dense_spacing": resolved_dense_spacing,
        "sparse_count": resolved_sparse_count,
        "min_pixel_dist": resolved_min_pixel_dist,
        "max_visible": resolved_max_visible,
    }


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


def load_depth_image(depth_path: Path) -> np.ndarray | None:
    if not depth_path.exists():
        return None
    img = Image.open(depth_path)
    depth_raw = np.array(img).astype(np.float32)
    if depth_raw.max() <= 1.0:
        return depth_raw * DEPTH_MAX_METERS
    return depth_raw / 65535.0 * DEPTH_MAX_METERS


def load_object_index_image(object_index_path: Path, max_pass_id: int | None = None) -> np.ndarray | None:
    if not object_index_path.exists():
        return None
    try:
        arr = np.array(Image.open(object_index_path))
        return decode_object_index_array(arr, max_pass_id=max_pass_id)
    except Exception:
        return None


def load_object_index_pass_ids(scene_render_dir: Path) -> set[int]:
    mapping_path = scene_render_dir / "object_index_map.json"
    if not mapping_path.exists():
        return set()
    try:
        mapping = json.load(open(mapping_path))
    except Exception:
        return set()
    pass_ids: set[int] = set()
    for key in mapping.keys():
        try:
            pass_ids.add(int(key))
        except Exception:
            continue
    return pass_ids


def is_occluded(depth_map: np.ndarray | None,
                px: int,
                py: int,
                depth_m: float,
                eps: float,
                window_radius: int = 1,
                footprint_radius_px: int = 0) -> bool:
    if depth_map is None:
        return False
    h, w = depth_map.shape[:2]
    if px < 0 or py < 0 or px >= w or py >= h:
        return True

    r = max(int(window_radius), int(footprint_radius_px), 0)
    x0 = max(0, px - r)
    x1 = min(w, px + r + 1)
    y0 = max(0, py - r)
    y1 = min(h, py + r + 1)
    patch = depth_map[y0:y1, x0:x1]
    valid_mask = np.isfinite(patch) & (patch > 0.0)
    if footprint_radius_px > 0:
        yy, xx = np.ogrid[y0:y1, x0:x1]
        circle = (xx - px) ** 2 + (yy - py) ** 2 <= footprint_radius_px ** 2
        valid_mask &= circle
    valid = patch[valid_mask]
    if valid.size == 0:
        return True
    observed_nearest = float(np.min(valid))
    return (observed_nearest + eps) < depth_m


def projected_radius_px(depth_m: float,
                        radius_m: float,
                        fov_deg: float,
                        res_y: int,
                        min_px: int = 0,
                        max_px: int = 12) -> int:
    if radius_m <= 1e-6 or depth_m <= 1e-6:
        return 0
    half_h = depth_m * math.tan(math.radians(fov_deg) / 2.0)
    if half_h <= 1e-6:
        return 0
    px_per_meter = res_y / (2.0 * half_h)
    radius_px = int(round(radius_m * px_per_meter))
    return max(min_px, min(radius_px, max_px))


def screen_norm_to_image_xy(screen_x: float,
                            screen_y: float,
                            res_x: int,
                            res_y: int) -> list[int]:
    px = int(round((screen_x + 1.0) * 0.5 * res_x))
    py = int(round((1.0 - (screen_y + 1.0) * 0.5) * res_y))
    return [px, py]


def image_xy_to_screen_norm(px: int,
                            py: int,
                            res_x: int,
                            res_y: int) -> list[float]:
    screen_x = (float(px) / float(res_x)) * 2.0 - 1.0
    screen_y = ((1.0 - float(py) / float(res_y)) * 2.0) - 1.0
    return [screen_x, screen_y]


def oriented_bbox_corners(bbox: list[float]) -> list[np.ndarray]:
    if len(bbox) < 9:
        return []
    cx, cy, cz, sx, sy, sz, rx, ry, rz = bbox[:9]
    rot = rotation_matrix_xyz(rx, ry, rz)
    corners = []
    for dx in (-sx * 0.5, sx * 0.5):
        for dy in (-sy * 0.5, sy * 0.5):
            for dz in (-sz * 0.5, sz * 0.5):
                local = np.array([dx, dy, dz], dtype=np.float32)
                world = rot @ local + np.array([cx, cy, cz], dtype=np.float32)
                corners.append(world)
    return corners


def sample_waypoints(grid, spacing_m: float) -> list[tuple[int, int]]:
    stride = max(1, int(round(spacing_m / grid.resolution)))
    sampled = []
    for gy in range(0, grid.height, stride):
        for gx in range(0, grid.width, stride):
            sampled.append((gx, gy))
    return sampled


def sample_interior_nonwalkable_cells(grid,
                                      spacing_m: float,
                                      skip_cells: set[tuple[int, int]] | None = None,
                                      floor_neighbor_radius: int = 1,
                                      min_occupied_neighbors: int = 2) -> list[tuple[int, int]]:
    """Sample occupied cells that are clearly inside the floor mask.

    We use these as extra A1/B1 negatives so blocked points come from obstacle
    footprints inside the room instead of from outside-floor or near-wall cells.
    """
    stride = max(1, int(round(spacing_m / grid.resolution)))
    floor_mask = getattr(grid, "floor_mask", None)
    skip_cells = skip_cells or set()
    sampled = []

    for gy in range(0, grid.height, stride):
        for gx in range(0, grid.width, stride):
            if (gx, gy) in skip_cells:
                continue
            if floor_mask is not None and not bool(floor_mask[gy, gx]):
                continue
            if grid.is_free(gx, gy):
                continue

            if floor_mask is not None and floor_neighbor_radius > 0:
                floor_ok = True
                for dy in range(-floor_neighbor_radius, floor_neighbor_radius + 1):
                    for dx in range(-floor_neighbor_radius, floor_neighbor_radius + 1):
                        nx, ny = gx + dx, gy + dy
                        if not (0 <= nx < grid.width and 0 <= ny < grid.height):
                            floor_ok = False
                            break
                        if not bool(floor_mask[ny, nx]):
                            floor_ok = False
                            break
                    if not floor_ok:
                        break
                if not floor_ok:
                    continue

            occupied_neighbors = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = gx + dx, gy + dy
                    if not (0 <= nx < grid.width and 0 <= ny < grid.height):
                        continue
                    if floor_mask is not None and not bool(floor_mask[ny, nx]):
                        continue
                    if not grid.is_free(nx, ny):
                        occupied_neighbors += 1
            if occupied_neighbors >= min_occupied_neighbors:
                sampled.append((gx, gy))

    return sampled


def project_visible_waypoints(waypoints: list[dict],
                              depth_map: np.ndarray | None,
                              cam_pos: np.ndarray,
                              cam_rot: np.ndarray,
                              fov: float,
                              res_x: int,
                              res_y: int,
                              occlusion_eps: float,
                              occlusion_window: int,
                              image_clearance_m: float,
                              image_clearance_max_px: int) -> list[dict]:
    projected = []
    for wp in waypoints:
        world_xyz = np.array(wp["world_xyz"], dtype=np.float32)
        proj = project_point(world_xyz, cam_pos, cam_rot, fov, res_x, res_y)
        if proj is None:
            continue
        footprint_radius_px = projected_radius_px(
            proj["depth"],
            image_clearance_m,
            fov,
            res_y,
            max_px=image_clearance_max_px,
        )
        if is_occluded(depth_map, proj["image_xy"][0], proj["image_xy"][1],
                       proj["depth"], occlusion_eps,
                       window_radius=occlusion_window,
                       footprint_radius_px=footprint_radius_px):
            continue
        proj.update({
            "waypoint_id": wp["waypoint_id"],
            "world_xyz": wp["world_xyz"],
            "pointmass_walkable": wp["pointmass_walkable"],
            "embodied_feasible": wp["embodied_feasible"],
            "is_occluded": False,
        })
        for key in ("grid_xy", "inside_floor_mask", "clearance_point_m",
                    "clearance_embodied_m", "candidate_source"):
            if key in wp:
                proj[key] = wp[key]
        projected.append(proj)
    return projected


def build_routing_gt_keypoint_candidates(existing_points: list[dict],
                                         routing_candidates: list[dict],
                                         source_candidates: list[dict],
                                         grid_point,
                                         grid_embodied,
                                         cam_pos: np.ndarray,
                                         cam_rot: np.ndarray,
                                         fov: float,
                                         res_x: int,
                                         res_y: int,
                                         depth_map: np.ndarray | None,
                                         object_index_img: np.ndarray | None,
                                         object_index_pass_ids: set[int],
                                         occlusion_eps: float,
                                         occlusion_window: int,
                                         image_clearance_m: float,
                                         image_clearance_max_px: int,
                                         next_waypoint_id: int,
                                         merge_world_dist_m: float = 0.15) -> tuple[list[dict], int]:
    injected: list[dict] = []
    existing_world_xy = [
        (float(point["world_xyz"][0]), float(point["world_xyz"][1]))
        for point in existing_points
        if isinstance(point.get("world_xyz"), list) and len(point["world_xyz"]) >= 2
    ]

    def _is_near_existing(wx: float, wy: float) -> bool:
        for ex, ey in existing_world_xy:
            if math.hypot(wx - ex, wy - ey) <= merge_world_dist_m:
                return True
        return False

    source_lookup = {
        int(point.get("waypoint_id", -1)): point
        for point in source_candidates
        if int(point.get("waypoint_id", -1)) >= 0
    }

    def _in_grid(gx: int, gy: int, grid) -> bool:
        return 0 <= gx < int(getattr(grid, "width", 0)) and 0 <= gy < int(getattr(grid, "height", 0))

    def _build_candidate(point: list[float],
                         image_xy: list[int] | None,
                         routing_id: str,
                         keypoint_idx: int,
                         source_waypoint_id: int | None) -> dict | None:
        wx, wy, wz = float(point[0]), float(point[1]), float(point[2])
        if _is_near_existing(wx, wy):
            return None

        source_point = None
        if source_waypoint_id is not None and int(source_waypoint_id) >= 0:
            source_point = source_lookup.get(int(source_waypoint_id))
        if source_point is not None:
            return {
                "waypoint_id": int(next_waypoint_id),
                "world_xyz": list(source_point.get("world_xyz", [round(wx, 3), round(wy, 3), round(wz, 3)])),
                "grid_xy": list(source_point.get("grid_xy", list(grid_point.world_to_grid(wx, wy)))),
                "image_xy": [int(source_point["image_xy"][0]), int(source_point["image_xy"][1])],
                "screen_xy_norm": list(source_point.get("screen_xy_norm", image_xy_to_screen_norm(
                    int(source_point["image_xy"][0]), int(source_point["image_xy"][1]), res_x, res_y,
                ))),
                "depth": round(float(source_point.get("depth", 0.0)), 4),
                "pointmass_walkable": bool(source_point.get("pointmass_walkable", False)),
                "embodied_feasible": bool(source_point.get("embodied_feasible", False)),
                "is_occluded": False,
                "raycast_visible": bool(source_point.get("raycast_visible", True)),
                "candidate_source": "routing_gt_keypoint",
                "routing_id": routing_id,
                "path_keypoint_index": int(keypoint_idx),
                "source_waypoint_id": int(source_waypoint_id),
            }

        if image_xy is None or not isinstance(image_xy, list) or len(image_xy) != 2:
            raise ValueError(
                f"routing_id={routing_id} keypoint_idx={keypoint_idx} missing Blender gt_path_keypoints_image_xy"
            )
        px, py = int(image_xy[0]), int(image_xy[1])
        if px < 0 or py < 0 or px >= res_x or py >= res_y:
            raise ValueError(
                f"routing_id={routing_id} keypoint_idx={keypoint_idx} has out-of-frame Blender projection {image_xy}"
            )
        gx, gy = grid_point.world_to_grid(wx, wy)
        if not _in_grid(int(gx), int(gy), grid_point) or not _in_grid(int(gx), int(gy), grid_embodied):
            raise ValueError(
                f"routing_id={routing_id} keypoint_idx={keypoint_idx} sparse keypoint is outside occupancy grid bounds"
            )
        point_ok = bool(grid_point.is_free(int(gx), int(gy)))
        emb_ok = bool(grid_embodied.is_free(int(gx), int(gy)))
        if not point_ok and not emb_ok:
            raise ValueError(
                f"routing_id={routing_id} keypoint_idx={keypoint_idx} sparse keypoint is not walkable in occupancy grids"
            )

        return {
            "waypoint_id": int(next_waypoint_id),
            "world_xyz": [round(wx, 3), round(wy, 3), round(wz, 3)],
            "grid_xy": [int(gx), int(gy)],
            "image_xy": [px, py],
            "screen_xy_norm": image_xy_to_screen_norm(px, py, res_x, res_y),
            "depth": 0.0,
            "pointmass_walkable": point_ok,
            "embodied_feasible": emb_ok,
            "is_occluded": False,
            "raycast_visible": True,
            "candidate_source": "routing_gt_keypoint",
            "routing_id": routing_id,
            "path_keypoint_index": int(keypoint_idx),
        }

    for routing in routing_candidates:
        path_world = routing.get("gt_path_keypoints_world") or []
        path_image = routing.get("gt_path_keypoints_image_xy") or []
        path_point_ids = routing.get("gt_path_waypoint_ids_pointmass") or []
        path_embodied_ids = routing.get("gt_path_waypoint_ids_embodied") or []
        routing_id = str(routing.get("routing_id", "routing_unknown"))
        if len(path_image) < len(path_world):
            raise ValueError(
                f"routing_id={routing_id} missing Blender gt_path_keypoints_image_xy for sparse path"
            )
        for keypoint_idx, point in enumerate(path_world):
            if not isinstance(point, list) or len(point) < 3:
                continue
            if _is_near_existing(float(point[0]), float(point[1])):
                continue
            image_xy = path_image[keypoint_idx]
            source_waypoint_id = None
            if keypoint_idx < len(path_embodied_ids):
                source_waypoint_id = int(path_embodied_ids[keypoint_idx])
            elif keypoint_idx < len(path_point_ids):
                source_waypoint_id = int(path_point_ids[keypoint_idx])
            candidate = _build_candidate(point, image_xy, routing_id, keypoint_idx, source_waypoint_id)
            if candidate is None:
                continue

            candidate["waypoint_id"] = int(next_waypoint_id)
            injected.append(candidate)
            existing_world_xy.append((float(candidate["world_xyz"][0]), float(candidate["world_xyz"][1])))
            next_waypoint_id += 1

    return injected, next_waypoint_id


def build_visible_object_surface_negatives(visible_objects: list[dict],
                                           layout_by_id: dict[int, dict],
                                           cam_pos: np.ndarray,
                                           cam_rot: np.ndarray,
                                           fov: float,
                                           res_x: int,
                                           res_y: int,
                                           next_waypoint_id: int,
                                           min_object_area_px: float = 26.0 * 26.0,
                                           max_points_per_object: int = 3) -> list[dict]:
    out = []
    floor_cover_categories = {"carpet", "rug", "mat"}
    for vo in visible_objects:
        if not bool(vo.get("raycast_verified", False)):
            continue
        obj_id = int(vo.get("id", -1))
        obj = layout_by_id.get(obj_id)
        if not obj:
            continue
        bbox = obj.get("bbox", [])
        if len(bbox) < 9:
            continue

        category = str(vo.get("category", obj.get("category", "object"))).lower()
        if category in floor_cover_categories:
            continue

        sx, sy, sz = float(bbox[3]), float(bbox[4]), float(bbox[5])
        bottom_z = float(bbox[2]) - sz * 0.5
        footprint_area = sx * sy

        # Skip tiny or clearly floating objects; these rarely function as
        # meaningful navigation blockers in image-space.
        if footprint_area < 0.05 and max(sx, sy) < 0.3:
            continue
        if bottom_z > 0.35:
            continue

        projected_corners = []
        for corner in oriented_bbox_corners(bbox):
            proj = project_point(corner, cam_pos, cam_rot, fov, res_x, res_y)
            if proj is not None:
                projected_corners.append(proj["image_xy"])

        base_xy = screen_norm_to_image_xy(
            float(vo.get("screen_x", 0.0)),
            float(vo.get("screen_y", 0.0)),
            res_x,
            res_y,
        )
        if projected_corners:
            xs = [xy[0] for xy in projected_corners] + [base_xy[0]]
            ys = [xy[1] for xy in projected_corners] + [base_xy[1]]
            x0, x1 = max(0, min(xs)), min(res_x - 1, max(xs))
            y0, y1 = max(0, min(ys)), min(res_y - 1, max(ys))
        else:
            pad = 18
            x0, x1 = max(0, base_xy[0] - pad), min(res_x - 1, base_xy[0] + pad)
            y0, y1 = max(0, base_xy[1] - pad), min(res_y - 1, base_xy[1] + pad)

        width = max(1, x1 - x0)
        height = max(1, y1 - y0)
        area = float(width * height)
        sample_pts = [(int(round((x0 + x1) * 0.5)), int(round((y0 + y1) * 0.5)))]

        if max_points_per_object > 1 and area >= min_object_area_px:
            if width >= height:
                sample_pts.append((int(round(x0 + width * 0.3)), int(round((y0 + y1) * 0.5))))
                if max_points_per_object > 2 and width >= 80:
                    sample_pts.append((int(round(x0 + width * 0.7)), int(round((y0 + y1) * 0.5))))
            else:
                sample_pts.append((int(round((x0 + x1) * 0.5)), int(round(y0 + height * 0.3))))
                if max_points_per_object > 2 and height >= 80:
                    sample_pts.append((int(round((x0 + x1) * 0.5)), int(round(y0 + height * 0.7))))

        seen_xy = set()
        for px, py in sample_pts:
            px = int(max(0, min(res_x - 1, px)))
            py = int(max(0, min(res_y - 1, py)))
            if (px, py) in seen_xy:
                continue
            seen_xy.add((px, py))
            out.append({
                "waypoint_id": next_waypoint_id,
                "world_xyz": [round(float(vo["position"][0]), 3),
                              round(float(vo["position"][1]), 3),
                              round(float(vo["position"][2]), 3)],
                "image_xy": [px, py],
                "screen_xy_norm": image_xy_to_screen_norm(px, py, res_x, res_y),
                "depth": float(vo.get("depth", 0.0)),
                "pointmass_walkable": False,
                "embodied_feasible": False,
                "is_occluded": False,
                "candidate_source": "visible_object_surface",
                "object_id": obj_id,
                "object_category": category,
            })
            next_waypoint_id += 1
    return out


def build_projected_obstacle_rects(visible_objects: list[dict],
                                   layout_by_id: dict[int, dict],
                                   cam_pos: np.ndarray,
                                   cam_rot: np.ndarray,
                                   fov: float,
                                   res_x: int,
                                   res_y: int) -> list[tuple[int, int, int, int]]:
    rects: list[tuple[int, int, int, int]] = []
    floor_cover_categories = {"carpet", "rug", "mat"}
    structural_categories = {"wall", "ceiling", "window", "door", "floor"}

    for vo in visible_objects:
        if not bool(vo.get("raycast_verified", False)):
            continue
        obj_id = int(vo.get("id", -1))
        obj = layout_by_id.get(obj_id)
        if not obj:
            continue
        bbox = obj.get("bbox", [])
        if len(bbox) < 9:
            continue

        category = str(vo.get("category", obj.get("category", "object"))).lower()
        if category in floor_cover_categories or category in structural_categories:
            continue

        bottom_z = float(bbox[2]) - float(bbox[5]) * 0.5
        top_z = float(bbox[2]) + float(bbox[5]) * 0.5
        if not (bottom_z < 1.4 and top_z > 0.10):
            continue

        projected = []
        for corner in oriented_bbox_corners(bbox):
            proj = project_point(corner, cam_pos, cam_rot, fov, res_x, res_y)
            if proj is not None:
                projected.append(proj["image_xy"])
        if len(projected) < 3:
            continue

        xs = [xy[0] for xy in projected]
        ys = [xy[1] for xy in projected]
        x0 = max(0, min(xs))
        x1 = min(res_x - 1, max(xs))
        y0 = max(0, min(ys))
        y1 = min(res_y - 1, max(ys))
        if x1 <= x0 or y1 <= y0:
            continue
        rects.append((x0, y0, x1, y1))

    return rects


def filter_points_by_obstacle_rects(points: list[dict],
                                    obstacle_rects: list[tuple[int, int, int, int]],
                                    pad_px: int = 2) -> list[dict]:
    if not obstacle_rects:
        return points
    kept = []
    for point in points:
        x, y = point["image_xy"]
        blocked = False
        for x0, y0, x1, y1 in obstacle_rects:
            if (x0 - pad_px) <= x <= (x1 + pad_px) and (y0 - pad_px) <= y <= (y1 + pad_px):
                blocked = True
                break
        if not blocked:
            kept.append(point)
    return kept


def build_graph_edges(nodes: dict[tuple[int, int], int],
                      grid,
                      stride: int) -> list[dict]:
    edges = []
    neighbor_offsets = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]
    for (gx, gy), src_id in nodes.items():
        wx0, wy0 = grid.grid_to_world(gx, gy)
        for dx, dy in neighbor_offsets:
            ng = (gx + dx * stride, gy + dy * stride)
            tgt_id = nodes.get(ng)
            if tgt_id is None or tgt_id <= src_id:
                continue
            wx1, wy1 = grid.grid_to_world(ng[0], ng[1])
            traversable = grid.is_path_traversable([(wx0, wy0), (wx1, wy1)])
            if not traversable["traversable"]:
                continue
            dist = math.hypot(wx1 - wx0, wy1 - wy0)
            edges.append({"source": src_id, "target": tgt_id, "distance": round(dist, 3)})
    return edges


def draw_overlay(rgb_path: Path,
                 out_path: Path,
                 waypoints: list[dict],
                 radius: int,
                 font_size: int,
                 color: tuple[int, int, int] = (255, 215, 0)) -> None:
    img = Image.open(rgb_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = None
    for fpath in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(fpath).exists():
            font = ImageFont.truetype(fpath, font_size)
            break
    if font is None:
        font = ImageFont.load_default()

    for wp in waypoints:
        x, y = wp["image_xy"]
        left = x - radius
        top = y - radius
        right = x + radius
        bottom = y + radius
        draw.ellipse((left, top, right, bottom), fill=color, outline=(0, 0, 0), width=2)

        label = str(wp.get("display_id", wp["waypoint_id"]))
        text_pos = (x + radius + 2, y - radius - 2)
        draw.text((text_pos[0] + 1, text_pos[1] + 1), label, fill=(0, 0, 0), font=font)
        draw.text(text_pos, label, fill=(255, 255, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def draw_path_image(rgb_path: Path,
                    object_index_path: Path,
                    out_path: Path,
                    path_image_xy: list[list[int] | None],
                    pass_index: int,
                    path_width_px: int = 6) -> None:
    base = Image.open(rgb_path).convert("RGB")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    if object_index_path.exists():
        idx_img = Image.open(object_index_path)
        idx_arr = np.array(idx_img)
        if idx_arr.dtype != np.uint16:
            idx_arr = idx_arr.astype(np.uint16)
        mask = idx_arr == pass_index
        if mask.any():
            arr = np.array(base).astype(np.float32)
            highlight = np.array([255, 0, 0], dtype=np.float32)
            arr[mask] = arr[mask] * 0.2 + highlight * 0.8
            base = Image.fromarray(arr.astype(np.uint8)).convert("RGBA")
            mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
            glow_mask = mask_img.filter(ImageFilter.GaussianBlur(radius=6))
            glow_layer = Image.new("RGBA", base.size, (255, 0, 0, 90))
            base = Image.alpha_composite(
                base,
                Image.composite(glow_layer, Image.new("RGBA", base.size, (0, 0, 0, 0)), glow_mask),
            )
        else:
            base = base.convert("RGBA")
    else:
        base = base.convert("RGBA")

    pts = []
    for pt in path_image_xy:
        if not isinstance(pt, list) or len(pt) != 2:
            continue
        pts.append((int(pt[0]), int(pt[1])))
    if len(pts) < 2:
        raise ValueError(f"path_target visualization requires Blender path_image_xy with at least 2 points: {out_path}")
    draw.line(pts, fill=(255, 0, 0, 180), width=max(1, int(path_width_px)))

    out = Image.alpha_composite(base, overlay)
    out.save(out_path)


def filter_by_min_dist(points: list[dict], min_dist: int) -> list[dict]:
    if min_dist <= 0:
        return points
    kept = []
    min_dist2 = min_dist * min_dist
    for p in points:
        x, y = p["image_xy"]
        ok = True
        for q in kept:
            dx = x - q["image_xy"][0]
            dy = y - q["image_xy"][1]
            if (dx * dx + dy * dy) < min_dist2:
                ok = False
                break
        if ok:
            kept.append(p)
    return kept


def balanced_sample(points: list[dict], key: str, max_count: int, rng: random.Random) -> list[dict]:
    """Sample a mixed overlay while preferring non-walkable >= walkable.

    The previous implementation could return:
      1. zero walkable points whenever non-walkable >= target
      2. at most 2 * len(non) total points when negatives were scarce

    After floor-only waypoint sampling removed many invalid outside-floor
    negatives, that behavior made A1/B1 overlays collapse to a handful of
    points even when many valid walkable candidates were visible.
    """
    walk = [p for p in points if p[key]]
    non = [p for p in points if not p[key]]
    rng.shuffle(walk)
    rng.shuffle(non)
    target = max_count if max_count is not None else len(points)

    if target <= 0:
        return []

    if not walk:
        return non[:target]
    if not non:
        return walk[:target]

    # Start from a balanced core so both classes are represented when possible.
    walk_take = min(len(walk), max(1, target // 2), len(non))
    non_take = min(len(non), target - walk_take)
    if non_take < walk_take:
        walk_take = non_take

    selected = []
    selected.extend(non[:non_take])
    selected.extend(walk[:walk_take])

    remaining = target - len(selected)
    if remaining > 0:
        # Fill remaining slots, preferring extra negatives first, then walkables.
        selected.extend(non[non_take:non_take + remaining])
        remaining = target - len(selected)
    if remaining > 0:
        selected.extend(walk[walk_take:walk_take + remaining])

    return selected


def rebalance_by_key(points: list[dict], key: str, rng: random.Random) -> list[dict]:
    """Ensure non-walkable >= walkable for the given key by downsampling walkable."""
    if not points:
        return points
    pos = [p for p in points if p.get(key)]
    neg = [p for p in points if not p.get(key)]
    if not neg:
        # Cannot enforce balance; keep original to avoid empty overlays.
        return points
    if len(pos) <= len(neg):
        return points
    rng.shuffle(pos)
    keep_pos = pos[:len(neg)]
    return keep_pos + neg


def source_priority(point: dict) -> tuple[int, float]:
    source = point.get("candidate_source", "base")
    if source == "visible_object_surface":
        priority = 0
    elif point.get("pointmass_walkable") and not point.get("embodied_feasible"):
        priority = 1
    elif source == "base":
        priority = 2
    else:
        priority = 3
    return (priority, float(point.get("depth", 1e9)))


def collect_required_routing_waypoint_ids(routing_candidates: list[dict]) -> set[int]:
    required: set[int] = set()
    for routing in routing_candidates or []:
        for key in ("gt_path_waypoint_ids_pointmass", "gt_path_waypoint_ids_embodied"):
            for waypoint_id in routing.get(key, []) or []:
                try:
                    waypoint_id = int(waypoint_id)
                except Exception:
                    continue
                if waypoint_id >= 0:
                    required.add(waypoint_id)
        for anchor_key in ("start_anchor", "goal_anchor"):
            anchor = routing.get(anchor_key) or {}
            try:
                waypoint_id = int(anchor.get("waypoint_id", -1))
            except Exception:
                waypoint_id = -1
            if waypoint_id >= 0:
                required.add(waypoint_id)
    return required


def _pixel_distance_sq(a: dict, b: dict) -> int:
    ax, ay = a["image_xy"]
    bx, by = b["image_xy"]
    dx = int(ax) - int(bx)
    dy = int(ay) - int(by)
    return dx * dx + dy * dy


def select_routing_visible_candidates(
    candidates: list[dict],
    routing_candidates: list[dict],
    *,
    max_visible: int,
    min_pixel_dist: int,
) -> list[dict]:
    free_candidates = [
        dict(candidate)
        for candidate in candidates or []
        if bool(candidate.get("pointmass_walkable", False))
        and isinstance(candidate.get("image_xy"), list)
        and len(candidate["image_xy"]) == 2
    ]
    free_candidates.sort(key=lambda item: float(item.get("depth", 1e9)))
    required_ids = collect_required_routing_waypoint_ids(routing_candidates)

    selected: list[dict] = []
    selected_waypoint_ids: set[int] = set()
    for candidate in free_candidates:
        waypoint_id = int(candidate.get("waypoint_id", -1))
        if waypoint_id not in required_ids:
            continue
        if waypoint_id in selected_waypoint_ids:
            continue
        selected.append(candidate)
        selected_waypoint_ids.add(waypoint_id)

    min_dist_sq = int(min_pixel_dist) * int(min_pixel_dist)
    for candidate in free_candidates:
        waypoint_id = int(candidate.get("waypoint_id", -1))
        if waypoint_id in selected_waypoint_ids:
            continue
        if max_visible > 0 and len(selected) >= max_visible:
            break
        if min_pixel_dist > 0 and any(_pixel_distance_sq(candidate, other) < min_dist_sq for other in selected):
            continue
        selected.append(candidate)
        selected_waypoint_ids.add(waypoint_id)

    selected.sort(key=lambda item: float(item.get("depth", 1e9)))
    return selected[:max_visible] if max_visible > 0 else selected


def select_classification_points(points: list[dict],
                                 key: str,
                                 max_count: int,
                                 rng: random.Random,
                                 min_surface_keep: int = 4,
                                 min_embodied_gap_keep: int = 4) -> list[dict]:
    if not points:
        return []

    target = max_count if max_count is not None else len(points)
    if target <= 0:
        return []

    surface_neg = [p for p in points if p.get("candidate_source") == "visible_object_surface"]
    reserved = []
    if surface_neg:
        rng.shuffle(surface_neg)
        keep_n = min(len(surface_neg), max(1, min(min_surface_keep, target // 2 if target > 1 else 1)))
        reserved = surface_neg[:keep_n]

    gap_ground = [
        p for p in points
        if p.get("candidate_source", "base") == "base"
        and p.get("pointmass_walkable")
        and not p.get("embodied_feasible")
    ]
    if gap_ground:
        rng.shuffle(gap_ground)
        remaining_budget = max(0, target - len(reserved))
        keep_n = min(len(gap_ground), min(min_embodied_gap_keep, remaining_budget))
        reserved.extend(gap_ground[:keep_n])

    reserved_ids = {id(p) for p in reserved}
    remainder = [p for p in points if id(p) not in reserved_ids]
    sampled = balanced_sample(remainder, key, max(0, target - len(reserved)), rng)
    return reserved + sampled


def load_scene_list(scene_list: Path | None,
                    scenes_dir: Path,
                    limit: int | None) -> list[str]:
    if scene_list and scene_list.exists():
        lines = [s.strip() for s in scene_list.read_text().splitlines() if s.strip()]
        return lines[:limit] if limit else lines
    scene_ids = sorted([p.name for p in scenes_dir.iterdir() if p.is_dir()])
    return scene_ids[:limit] if limit else scene_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multi-task overlays and path images.")
    parser.add_argument("--config", required=True, help="Config YAML")
    parser.add_argument("--scenes-dir", required=True, help="Path to scenes/")
    parser.add_argument("--renders-dir", required=True, help="Path to renders/")
    parser.add_argument("--output-dir", default=None, help="Output root (default: renders-dir)")
    parser.add_argument("--scene-list", default=None, help="Optional scene list file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of scenes")
    parser.add_argument("--dense-spacing", type=float, default=None, help="Dense waypoint spacing (m)")
    parser.add_argument("--sparse-count", type=int, default=None, help="Sparse sample count for A1/B1")
    parser.add_argument("--min-pixel-dist", type=int, default=None, help="Min pixel distance between overlays")
    parser.add_argument("--occlusion-eps", type=float, default=0.05, help="Occlusion epsilon (meters)")
    parser.add_argument("--occlusion-window", type=int, default=1, help="Depth neighborhood radius in pixels")
    parser.add_argument("--image-clearance-m", type=float, default=None,
                        help="Optional projected clearance radius used to reject edge-overlapping points")
    parser.add_argument("--image-clearance-max-px", type=int, default=12,
                        help="Cap the projected clearance radius in pixels")
    parser.add_argument("--max-visible", type=int, default=None, help="Cap visible waypoints per view")
    parser.add_argument("--font-size", type=int, default=18, help="Font size for overlay labels")
    parser.add_argument("--path-width-m", type=float, default=0.6, help="Path corridor width in meters")
    args = parser.parse_args()

    config = load_config(args.config)
    scenes_dir = Path(args.scenes_dir)
    renders_dir = Path(args.renders_dir)
    output_dir = Path(args.output_dir) if args.output_dir else renders_dir
    scene_list = Path(args.scene_list) if args.scene_list else None

    render_cfg = config["render"]
    res_x, res_y = render_cfg["resolution"]
    fov = float(render_cfg["fov"])
    nav_cfg = config["navigation"]
    image_clearance_m = args.image_clearance_m
    if image_clearance_m is None:
        image_clearance_m = max(float(nav_cfg.get("passable_margin", 0.0)), 0.08)
    task_cfg = config.get("task_outputs", {})
    a1_min_positive = int(task_cfg.get("a1_min_positive", 12))
    sampling = resolve_sampling_params(
        config,
        dense_spacing=args.dense_spacing,
        sparse_count=args.sparse_count,
        min_pixel_dist=args.min_pixel_dist,
        max_visible=args.max_visible,
    )
    dense_spacing = float(sampling["dense_spacing"])
    sparse_count = int(sampling["sparse_count"])
    min_pixel_dist = int(sampling["min_pixel_dist"])
    max_visible = int(sampling["max_visible"])

    scene_ids = load_scene_list(scene_list, scenes_dir, args.limit)
    if not scene_ids:
        raise RuntimeError("No scenes found to process")

    for scene_id in scene_ids:
        scene_path = scenes_dir / scene_id
        render_scene_dir = renders_dir / scene_id
        if not render_scene_dir.exists():
            print(f"[skip] {scene_id}: renders missing")
            continue
        cameras_path = render_scene_dir / "cameras.json"
        if not cameras_path.exists():
            print(f"[skip] {scene_id}: cameras.json missing")
            continue

        with open(scene_path / "layout.json") as f:
            layout = json.load(f)
        layout_by_id = {int(obj["id"]): obj for obj in layout if "id" in obj}
        with open(cameras_path) as f:
            cameras = json.load(f)
        if not cameras:
            print(f"[skip] {scene_id}: empty cameras")
            continue
        validate_render_scene_inputs(render_scene_dir, cameras)
        object_index_pass_ids = load_object_index_pass_ids(render_scene_dir)

        # Build occupancy grids
        cfg_point = json.loads(json.dumps(config))
        cfg_point["navigation"]["agent_radius"] = 0.0
        cfg_point["navigation"]["passable_margin"] = 0.0
        grid_point = build_occupancy_grid(layout, cfg_point, scene_path=scene_path)
        grid_embodied = build_occupancy_grid(layout, config, scene_path=scene_path)

        spacing = dense_spacing
        stride = max(1, int(round(spacing / grid_point.resolution)))
        sampled_cells = sample_waypoints(grid_point, spacing)
        if getattr(grid_point, "floor_mask", None) is not None:
            sampled_cells = [
                (gx, gy) for (gx, gy) in sampled_cells
                if bool(grid_point.floor_mask[gy, gx])
            ]

        dist_point = grid_point.get_distance_map()
        dist_embodied = grid_embodied.get_distance_map()

        waypoints = []
        nodes = {}
        for idx, (gx, gy) in enumerate(sampled_cells):
            wx, wy = grid_point.grid_to_world(gx, gy)
            wp = {
                "waypoint_id": idx,
                "world_xyz": [round(wx, 3), round(wy, 3), 0.0],
                "grid_xy": [int(gx), int(gy)],
                "inside_floor_mask": True,
                "pointmass_walkable": bool(grid_point.is_free(gx, gy)),
                "embodied_feasible": bool(grid_embodied.is_free(gx, gy)),
                "clearance_point_m": round(float(dist_point[gy, gx]), 3),
                "clearance_embodied_m": round(float(dist_embodied[gy, gx]), 3),
            }
            waypoints.append(wp)
            nodes[(gx, gy)] = idx

        next_extra_id = len(waypoints)

        edges_pointmass = build_graph_edges(nodes, grid_point, stride)
        edges_embodied = build_graph_edges(nodes, grid_embodied, stride)

        scene_out = output_dir / scene_id
        scene_out.mkdir(parents=True, exist_ok=True)
        if scene_out != render_scene_dir:
            dst_cameras = scene_out / "cameras.json"
            if not dst_cameras.exists():
                shutil.copy2(cameras_path, dst_cameras)
        with open(scene_out / "waypoint_core.json", "w") as f:
            json.dump({
                "scene_id": scene_id,
                "grid_resolution": grid_point.resolution,
                "waypoint_spacing": spacing,
                "agent_radius": config["navigation"]["agent_radius"],
                "waypoints": waypoints,
            }, f, indent=2)
        # Backward-compatible default (embodied)
        with open(scene_out / "waypoint_graph.json", "w") as f:
            json.dump({
                "scene_id": scene_id,
                "graph_type": "embodied",
                "grid_resolution": grid_point.resolution,
                "waypoint_spacing": spacing,
                "agent_radius": config["navigation"]["agent_radius"],
                "waypoints": waypoints,
                "edges": edges_embodied,
            }, f, indent=2)

        # Explicit graph variants for next GT/VQA
        with open(scene_out / "waypoint_graph_pointmass.json", "w") as f:
            json.dump({
                "scene_id": scene_id,
                "graph_type": "pointmass",
                "grid_resolution": grid_point.resolution,
                "waypoint_spacing": spacing,
                "agent_radius": 0.0,
                "waypoints": waypoints,
                "edges": edges_pointmass,
            }, f, indent=2)
        with open(scene_out / "waypoint_graph_embodied.json", "w") as f:
            json.dump({
                "scene_id": scene_id,
                "graph_type": "embodied",
                "grid_resolution": grid_point.resolution,
                "waypoint_spacing": spacing,
                "agent_radius": config["navigation"]["agent_radius"],
                "waypoints": waypoints,
                "edges": edges_embodied,
            }, f, indent=2)

        write_next_task_outputs(scene_out, scene_id, cameras, config)

        for cam in cameras:
            view_id = cam.get("view_id", 0)
            view_in_dir = render_scene_dir / f"view_{view_id}"
            view_out_dir = scene_out / f"view_{view_id}"
            view_out_dir.mkdir(parents=True, exist_ok=True)
            view_bundle = load_view_bundle_v3(render_scene_dir, int(view_id))
            rgb_path = view_in_dir / "rgb.png"
            depth_path = view_in_dir / "depth.png"
            obj_idx_path = view_in_dir / "object_index.png"

            if not rgb_path.exists():
                continue

            depth_map = load_depth_image(depth_path)
            object_index_img = load_object_index_image(
                obj_idx_path,
                max_pass_id=max(object_index_pass_ids) if object_index_pass_ids else None,
            )
            cam_pos = np.array(cam["position"], dtype=np.float32)
            cam_rot = np.array(cam["rotation"], dtype=np.float32)

            blender_candidates = []
            for candidate in view_bundle.get("classification_candidates", []):
                image_xy = candidate.get("image_xy")
                if not isinstance(image_xy, list) or len(image_xy) != 2:
                    continue
                if image_xy[0] is None or image_xy[1] is None:
                    continue
                depth = float(candidate.get("depth", 0.0))
                if depth <= 0.0:
                    continue
                if not bool(candidate.get("raycast_visible", True)):
                    continue
                blender_candidates.append({
                    "waypoint_id": candidate.get("waypoint_id"),
                    "world_xyz": candidate.get("world_xyz"),
                    "grid_xy": candidate.get("grid_xy"),
                    "image_xy": [int(image_xy[0]), int(image_xy[1])],
                    "screen_xy_norm": candidate.get("screen_xy_norm"),
                    "depth": depth,
                    "pointmass_walkable": bool(candidate.get("pointmass_walkable", False)),
                    "embodied_feasible": bool(candidate.get("embodied_feasible", False)),
                    "is_occluded": False,
                    "raycast_visible": True,
                    "candidate_source": candidate.get("candidate_source", "blender_grid"),
                })

            if not blender_candidates:
                raise ValueError(
                    f"scene={scene_id} view={view_id} has no valid Blender classification_candidates after visibility checks"
                )
            projected = sorted(blender_candidates, key=lambda p: p["depth"])

            # A1/B1 share the same candidate set:
            # 1) ground points that are point-mass walkable
            # 2) obstacle-surface negatives
            # This lets B1 stay strictly harder than A1 because some ground
            # points remain visible but flip from walkable->blocked under the
            # embodied feasibility label.
            rng_ab = random.Random(f"{scene_id}:{view_id}:ab")
            projected_ab = list(projected)
            projected_ab = sorted(projected_ab, key=source_priority)
            projected_ab = filter_by_min_dist(projected_ab, min_pixel_dist)
            if len(projected_ab) > max_visible:
                projected_ab = projected_ab[:max_visible]
            displayed_ab = select_classification_points(
                projected_ab, "pointmass_walkable", sparse_count, rng_ab
            )
            if not displayed_ab:
                displayed_ab = projected_ab[:sparse_count] if sparse_count else projected_ab
            displayed_ab = rebalance_by_key(displayed_ab, "pointmass_walkable", rng_ab)

            if blender_candidates and a1_min_positive > 0:
                def _key(point: dict):
                    return (
                        point.get("waypoint_id"),
                        tuple(point.get("image_xy", [])),
                        point.get("candidate_source", "base"),
                    )

                selected_keys = {_key(point) for point in displayed_ab}
                selected_pos = [point for point in displayed_ab if point.get("pointmass_walkable")]
                pos_pool = [point for point in projected_ab if point.get("pointmass_walkable")]
                neg_pool = [point for point in projected_ab if not point.get("pointmass_walkable")]

                pos_need = max(0, min(a1_min_positive, len(pos_pool)) - len(selected_pos))
                if pos_need > 0:
                    for point in pos_pool:
                        key = _key(point)
                        if key in selected_keys:
                            continue
                        displayed_ab.append(point)
                        selected_keys.add(key)
                        pos_need -= 1
                        if pos_need <= 0:
                            break

                selected_pos = [point for point in displayed_ab if point.get("pointmass_walkable")]
                selected_neg = [point for point in displayed_ab if not point.get("pointmass_walkable")]
                neg_need = len(selected_pos) - len(selected_neg)
                if neg_need > 0:
                    for point in neg_pool:
                        key = _key(point)
                        if key in selected_keys:
                            continue
                        displayed_ab.append(point)
                        selected_keys.add(key)
                        neg_need -= 1
                        if neg_need <= 0:
                            break

                selected_pos = [point for point in displayed_ab if point.get("pointmass_walkable")]
                selected_neg = [point for point in displayed_ab if not point.get("pointmass_walkable")]
                if len(selected_neg) < len(selected_pos):
                    keep_pos = len(selected_neg)
                    kept = []
                    pos_kept = 0
                    for point in sorted(displayed_ab, key=lambda item: float(item.get("depth", 1e9))):
                        if point.get("pointmass_walkable"):
                            if pos_kept >= keep_pos:
                                continue
                            pos_kept += 1
                        kept.append(point)
                    displayed_ab = kept

            displayed_ab.sort(key=lambda w: w["depth"])
            displayed_ab = [dict(wp) for wp in displayed_ab]
            for i, wp in enumerate(displayed_ab, start=1):
                wp["display_id"] = i

            # A2/B2/C: only walkable points
            projected_free = [p for p in projected if p["pointmass_walkable"]]
            projected_free = sorted(projected_free, key=lambda w: w["depth"])
            a2_min_dist = min_pixel_dist
            a2_min_dist = max(a2_min_dist, int(round(min_pixel_dist * 1.35)))
            displayed_free = select_routing_visible_candidates(
                projected_free,
                list(view_bundle.get("routing_candidates", [])),
                max_visible=max_visible,
                min_pixel_dist=a2_min_dist,
            )
            displayed_free.sort(key=lambda w: w["depth"])
            for i, wp in enumerate(displayed_free, start=1):
                wp["display_id"] = i

            for task in ("a1", "b1"):
                out_meta = {
                    "view_id": view_id,
                    "image_path": f"rgb_overlay_{task}.png",
                    "visible_waypoints": displayed_ab,
                }
                with open(view_out_dir / f"visible_waypoints_{task}.json", "w") as f:
                    json.dump(out_meta, f, indent=2)

                draw_overlay(
                    rgb_path,
                    view_out_dir / f"rgb_overlay_{task}.png",
                    displayed_ab,
                    radius=10,
                    font_size=args.font_size,
                )

            for task in ("a2", "b2", "c"):
                out_meta = {
                    "view_id": view_id,
                    "image_path": f"rgb_overlay_{task}.png",
                    "visible_waypoints": displayed_free,
                }
                with open(view_out_dir / f"visible_waypoints_{task}.json", "w") as f:
                    json.dump(out_meta, f, indent=2)

                draw_overlay(
                    rgb_path,
                    view_out_dir / f"rgb_overlay_{task}.png",
                    displayed_free,
                    radius=10,
                    font_size=args.font_size,
                    color=TASK_COLORS[task],
                )

            # Debug visualizations for manual verification
            from visualize_next_results import (
                render_ab_semantics_debug,
                render_routing_semantics_debug,
                routing_debug_filename,
            )

            task_outputs_path = scene_out / get_task_outputs_filename(config)
            render_ab_semantics_debug(
                view_out_dir,
                view_out_dir / "ab_semantics_debug.png",
                task_outputs_path=task_outputs_path,
            )
            if not (view_out_dir / "ab_semantics_debug.png").exists():
                raise RuntimeError(
                    f"ab semantics debug visualization was not generated for {scene_id} view {view_id}"
                )

            # Routing debug requires GT records; build a minimal per-view unit list here.
            routing_units = []
            try:
                from build_navigation_gt_next import group_routing_records_by_target_rule
            except Exception:
                group_routing_records_by_target_rule = None

            # Build minimal routing-unit records from available camera target candidates
            # (Full GT alignment happens in build_navigation_gt_next.py.)
            a2_records = []
            b2_records = []
            for tgt in cam.get("target_candidates", []):
                target_id = int(tgt.get("target_id", -1))
                if target_id < 0:
                    continue
                target_rule = str(tgt.get("target_rule", "nearest"))
                unit_base = {
                    "target_id": target_id,
                    "target_rule": target_rule,
                    "target_category": tgt.get("category", "unknown"),
                    "target_bbox": tgt.get("bbox"),
                }
                a2_records.append({"task": "a2", **unit_base})
                b2_records.append({"task": "b2", **unit_base})
            if group_routing_records_by_target_rule is not None:
                routing_units = group_routing_records_by_target_rule(a2_records, b2_records)

            if routing_units:
                single_unit = len(routing_units) == 1
                for unit in routing_units:
                    filename = routing_debug_filename(
                        unit.get("target_id", -1), unit.get("target_rule", "unknown"), single_unit
                    )
                    render_routing_semantics_debug(
                        view_out_dir,
                        view_out_dir / filename,
                        unit,
                    )
                expected = (
                    [routing_debug_filename(
                        routing_units[0].get("target_id", -1),
                        routing_units[0].get("target_rule", "unknown"),
                        True,
                    )]
                    if single_unit
                    else [
                        routing_debug_filename(
                            unit.get("target_id", -1), unit.get("target_rule", "unknown"), False
                        )
                        for unit in routing_units
                    ]
                )
                for name in expected:
                    if not (view_out_dir / name).exists():
                        raise RuntimeError(
                            f"routing debug visualization was not generated for {scene_id} view {view_id}"
                        )

            # Path images per target
            routing_by_target_id = {
                int(candidate.get("target_id", -1)): candidate
                for candidate in view_bundle.get("routing_candidates", [])
                if int(candidate.get("target_id", -1)) >= 0
            }
            for tgt in cam.get("target_candidates", []):
                pass_index = int(tgt.get("pass_index", -1))
                target_id = int(tgt.get("target_id", -1))
                if pass_index < 0 or target_id < 0:
                    continue
                routing = routing_by_target_id.get(target_id)
                if routing is None:
                    raise ValueError(
                        f"scene={scene_id} view={view_id} target={target_id} missing routing candidate for path image"
                    )
                path_image_xy = list(routing.get("debug_dense_path_image_xy") or routing.get("gt_path_keypoints_image_xy") or [])
                out_path = view_out_dir / f"path_target_{tgt['target_id']}.png"
                draw_path_image(
                    rgb_path,
                    obj_idx_path,
                    out_path,
                    path_image_xy=path_image_xy,
                    pass_index=pass_index,
                    path_width_px=max(2, int(round(args.path_width_m * 12))),
                )

        print(f"[done] {scene_id}")


if __name__ == "__main__":
    main()
