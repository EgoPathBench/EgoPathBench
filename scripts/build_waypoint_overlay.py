#!/usr/bin/env python3
"""
Build waypoint graph + per-view visible waypoints and overlay images.

Outputs per scene:
  - waypoint_graph.json
  - sample_filtering.json
  - view_*/visible_waypoints.json
  - view_*/rgb_overlay.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from build_navigation_gt import build_occupancy_grid


DEPTH_MAX_METERS = 20.0  # must match scripts/render_scenes.py


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def rotation_matrix_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    """Blender-style XYZ Euler -> rotation matrix (local->world)."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    rx_m = np.array([[1, 0, 0],
                     [0, cx, -sx],
                     [0, sx, cx]], dtype=np.float32)
    ry_m = np.array([[cy, 0, sy],
                     [0, 1, 0],
                     [-sy, 0, cy]], dtype=np.float32)
    rz_m = np.array([[cz, -sz, 0],
                     [sz, cz, 0],
                     [0, 0, 1]], dtype=np.float32)
    return rz_m @ ry_m @ rx_m


def project_point(world_xyz: np.ndarray,
                  cam_pos: np.ndarray,
                  cam_rot: np.ndarray,
                  fov_deg: float,
                  res_x: int,
                  res_y: int) -> dict | None:
    """Project world point to image plane. Returns dict or None if outside."""
    cam_mat = rotation_matrix_xyz(cam_rot[0], cam_rot[1], cam_rot[2])
    cam_pt = cam_mat.T @ (world_xyz - cam_pos)

    if cam_pt[2] >= -1e-6:  # behind camera (camera looks along -Z)
        return None

    depth = -cam_pt[2]
    if depth < 0.05:
        return None

    vfov_half = math.radians(fov_deg) / 2
    aspect = res_x / res_y
    hfov_half = math.atan(math.tan(vfov_half) * aspect)

    half_h = depth * math.tan(vfov_half)
    half_w = depth * math.tan(hfov_half)
    if half_h < 1e-6 or half_w < 1e-6:
        return None

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


def is_occluded(depth_map: np.ndarray | None,
                px: int,
                py: int,
                depth_m: float,
                eps: float) -> bool:
    if depth_map is None:
        return False
    h, w = depth_map.shape[:2]
    if px < 0 or py < 0 or px >= w or py >= h:
        return True
    observed = float(depth_map[py, px])
    if observed <= 0.0:
        return True
    return (observed + eps) < depth_m


def sample_waypoints(grid, spacing_m: float) -> list[tuple[int, int]]:
    stride = max(1, int(round(spacing_m / grid.resolution)))
    sampled = []
    for gy in range(0, grid.height, stride):
        for gx in range(0, grid.width, stride):
            if grid.is_free(gx, gy):
                sampled.append((gx, gy))
    return sampled


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
            if tgt_id is None:
                continue
            if tgt_id <= src_id:
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
                 radius: int) -> None:
    img = Image.open(rgb_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for wp in waypoints:
        x, y = wp["image_xy"]
        left = x - radius
        top = y - radius
        right = x + radius
        bottom = y + radius
        draw.ellipse((left, top, right, bottom), fill=(255, 215, 0), outline=(0, 0, 0), width=2)

        label = str(wp["waypoint_id"])
        text_pos = (x + radius + 2, y - radius - 2)
        # simple shadow for readability
        draw.text((text_pos[0] + 1, text_pos[1] + 1), label, fill=(0, 0, 0), font=font)
        draw.text(text_pos, label, fill=(255, 255, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def filter_by_pixel_distance(points: list[dict],
                             min_dist_px: int,
                             max_keep: int) -> list[dict]:
    kept = []
    min_dist_sq = min_dist_px * min_dist_px
    for p in points:
        px, py = p["image_xy"]
        ok = True
        for k in kept:
            kx, ky = k["image_xy"]
            if (px - kx) ** 2 + (py - ky) ** 2 < min_dist_sq:
                ok = False
                break
        if ok:
            kept.append(p)
        if len(kept) >= max_keep:
            break
    return kept


def load_scene_list(scene_list: Path | None,
                    scenes_dir: Path,
                    limit: int | None) -> list[str]:
    if scene_list and scene_list.exists():
        lines = [s.strip() for s in scene_list.read_text().splitlines() if s.strip()]
        if limit:
            return lines[:limit]
        return lines
    scene_ids = sorted([p.name for p in scenes_dir.iterdir() if p.is_dir()])
    if limit:
        scene_ids = scene_ids[:limit]
    return scene_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Build waypoint overlays + visible waypoint metadata.")
    parser.add_argument("--config", required=True, help="Config YAML")
    parser.add_argument("--scenes-dir", required=True, help="Path to scenes/")
    parser.add_argument("--renders-dir", required=True, help="Path to renders/")
    parser.add_argument("--output-dir", default=None, help="Output root (default: renders-dir)")
    parser.add_argument("--scene-list", default=None, help="Optional scene list file")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of scenes")
    parser.add_argument("--waypoint-spacing", type=float, default=0.6, help="Waypoint spacing (meters)")
    parser.add_argument("--min-depth", type=float, default=0.3, help="Min depth to keep waypoint")
    parser.add_argument("--max-waypoints-per-view", type=int, default=80, help="Cap per view")
    parser.add_argument("--min-pixel-dist", type=int, default=18, help="Min pixel distance between overlays")
    parser.add_argument("--occlusion-eps", type=float, default=0.15, help="Occlusion epsilon (meters)")
    parser.add_argument("--min-visible", type=int, default=12, help="Min visible waypoints to keep view")
    parser.add_argument("--skip-existing", action="store_true", help="Skip if overlay already exists")
    args = parser.parse_args()

    config = load_config(args.config)
    scenes_dir = Path(args.scenes_dir)
    renders_dir = Path(args.renders_dir)
    output_dir = Path(args.output_dir) if args.output_dir else renders_dir
    scene_list = Path(args.scene_list) if args.scene_list else None

    render_cfg = config["render"]
    res_x, res_y = render_cfg["resolution"]
    fov = float(render_cfg["fov"])

    scene_ids = load_scene_list(scene_list, scenes_dir, args.limit)
    if not scene_ids:
        raise RuntimeError("No scenes found to process")

    for scene_id in scene_ids:
        scene_path = scenes_dir / scene_id
        if not scene_path.exists():
            print(f"[skip] {scene_id}: scene dir missing")
            continue

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

        with open(cameras_path) as f:
            cameras = json.load(f)

        if not cameras:
            print(f"[skip] {scene_id}: empty cameras")
            continue

        # Build occupancy grids
        cfg_point = json.loads(json.dumps(config))
        cfg_point["navigation"]["agent_radius"] = 0.0
        grid_point = build_occupancy_grid(layout, cfg_point, scene_path=scene_path)
        grid_embodied = build_occupancy_grid(layout, config, scene_path=scene_path)

        spacing = args.waypoint_spacing
        stride = max(1, int(round(spacing / grid_point.resolution)))
        sampled_cells = sample_waypoints(grid_point, spacing)
        if getattr(grid_point, "floor_mask", None) is not None:
            sampled_cells = [
                (gx, gy) for (gx, gy) in sampled_cells
                if bool(grid_point.floor_mask[gy, gx])
            ]

        # Precompute distance maps
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

        edges = build_graph_edges(nodes, grid_embodied, stride)

        scene_out = output_dir / scene_id
        scene_out.mkdir(parents=True, exist_ok=True)

        waypoint_graph = {
            "scene_id": scene_id,
            "grid_resolution": grid_point.resolution,
            "waypoint_spacing": spacing,
            "agent_radius": config["navigation"]["agent_radius"],
            "waypoints": waypoints,
            "edges": edges,
        }
        with open(scene_out / "waypoint_graph.json", "w") as f:
            json.dump(waypoint_graph, f, indent=2)

        sample_report = {
            "scene_id": scene_id,
            "num_waypoints": len(waypoints),
            "views": [],
        }

        for cam in cameras:
            view_id = cam.get("view_id", 0)
            view_dir = scene_out / f"view_{view_id}"
            rgb_path = view_dir / "rgb.png"
            depth_path = view_dir / "depth.png"

            if not rgb_path.exists():
                sample_report["views"].append({
                    "view_id": view_id,
                    "status": "skip",
                    "reason": "missing_rgb",
                })
                continue

            overlay_path = view_dir / "rgb_overlay.png"
            if args.skip_existing and overlay_path.exists():
                sample_report["views"].append({
                    "view_id": view_id,
                    "status": "skip",
                    "reason": "overlay_exists",
                })
                continue

            depth_map = load_depth_image(depth_path)

            cam_pos = np.array(cam["position"], dtype=np.float32)
            cam_rot = np.array(cam["rotation"], dtype=np.float32)

            projected = []
            for wp in waypoints:
                world_xyz = np.array(wp["world_xyz"], dtype=np.float32)
                proj = project_point(world_xyz, cam_pos, cam_rot, fov, res_x, res_y)
                if proj is None:
                    continue
                if proj["depth"] < args.min_depth:
                    continue
                occluded = is_occluded(depth_map, proj["image_xy"][0], proj["image_xy"][1],
                                       proj["depth"], args.occlusion_eps)
                if occluded:
                    continue

                proj.update({
                    "waypoint_id": wp["waypoint_id"],
                    "world_xyz": wp["world_xyz"],
                    "pointmass_walkable": wp["pointmass_walkable"],
                    "embodied_feasible": wp["embodied_feasible"],
                    "is_occluded": False,
                })
                projected.append(proj)

            projected.sort(key=lambda x: x["depth"])
            filtered = filter_by_pixel_distance(
                projected,
                args.min_pixel_dist,
                args.max_waypoints_per_view,
            )

            status = "keep" if len(filtered) >= args.min_visible else "drop"
            reason = "" if status == "keep" else "too_few_waypoints"

            visible_meta = {
                "view_id": view_id,
                "image_path": "rgb_overlay.png",
                "visible_waypoints": filtered,
            }
            with open(view_dir / "visible_waypoints.json", "w") as f:
                json.dump(visible_meta, f, indent=2)

            draw_overlay(rgb_path, overlay_path, filtered, radius=6)

            sample_report["views"].append({
                "view_id": view_id,
                "status": status,
                "num_visible_waypoints": len(filtered),
                "num_projected": len(projected),
                "reason": reason,
            })

        with open(scene_out / "sample_filtering.json", "w") as f:
            json.dump(sample_report, f, indent=2)

        print(f"[done] {scene_id}: waypoints={len(waypoints)}, views={len(cameras)}")


if __name__ == "__main__":
    main()
