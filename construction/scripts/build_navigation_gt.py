"""
NavBench3D - Step 4: Navigation Ground Truth Construction.

Builds occupancy grids from scene layouts and computes:
  GT-1: Optimal navigation path with turn-by-turn directions (for training)
  GT-2: Full traversability map (for benchmark path validation)

Usage:
    python scripts/build_navigation_gt.py --config configs/default.yaml --scenes-dir data/scenes --renders-dir data/renders
"""

import json
import math
import argparse
import heapq
import importlib
import os
import sys
from pathlib import Path
from typing import Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
VENDORED_PY_SITE = REPO_ROOT / ".vendor" / "py312"


def _sanitize_mesh_python_path() -> None:
    """Prefer repo-vendored / conda packages over conflicting user-site wheels."""
    if os.environ.get("NAVBENCH_ALLOW_USER_SITE") == "1":
        return

    user_site = Path.home() / ".local" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    sanitized = []
    for entry in sys.path:
        if not entry:
            sanitized.append(entry)
            continue
        try:
            resolved = Path(entry).expanduser().resolve()
        except Exception:
            sanitized.append(entry)
            continue
        if resolved == user_site:
            continue
        sanitized.append(entry)
    sys.path[:] = sanitized

    if VENDORED_PY_SITE.exists():
        vendor_str = str(VENDORED_PY_SITE)
        if vendor_str not in sys.path:
            sys.path.insert(0, vendor_str)


_sanitize_mesh_python_path()

import numpy as np
import yaml
from PIL import Image

_TRIMESH_IMPORT_ERROR: Exception | None = None
_TRIMESH_WARNING_EMITTED = False


def optional_package_site_candidates(module_name: str) -> list[Path]:
    """Discover bridge site-packages directories that contain a module."""
    seen: set[Path] = set()
    candidates: list[Path] = []

    def _append(site_dir: Path) -> None:
        resolved = Path(site_dir).expanduser()
        if resolved in seen or not resolved.exists():
            return
        if not ((resolved / module_name).exists() or (resolved / f"{module_name}.py").exists()):
            return
        seen.add(resolved)
        candidates.append(resolved)

    explicit = os.environ.get("NAVBENCH_OPTIONAL_PY_SITES", "")
    for raw in explicit.split(os.pathsep):
        raw = raw.strip()
        if raw:
            _append(Path(raw))

    _append(VENDORED_PY_SITE)

    env_names: list[str] = []
    preferred_env = os.environ.get("NAVBENCH_OPTIONAL_ENV_NAME", "navbench3d").strip()
    if preferred_env:
        env_names.append(preferred_env)
    active_env = os.environ.get("CONDA_DEFAULT_ENV", "").strip()
    if active_env and active_env not in env_names:
        env_names.append(active_env)

    for env_name in env_names:
        for root in (
            Path.home() / ".conda" / "envs",
            Path("/usr/local/anaconda3/envs"),
            Path("/opt/conda/envs"),
        ):
            env_dir = root / env_name
            if not env_dir.exists():
                continue
            for site_dir in sorted(env_dir.glob("lib/python*/site-packages")):
                _append(site_dir)

    return candidates


def import_optional_module(module_name: str):
    """Import an optional module, retrying from discovered bridge sites."""
    last_error: Exception | None = None
    try:
        return importlib.import_module(module_name)
    except Exception as exc:  # pragma: no cover - depends on local env
        last_error = exc

    for site_dir in optional_package_site_candidates(module_name):
        site_str = str(site_dir)
        if site_str not in sys.path:
            sys.path.append(site_str)
        sys.modules.pop(module_name, None)
        importlib.invalidate_caches()
        try:
            return importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - depends on local env
            last_error = exc

    if module_name == "trimesh":
        global _TRIMESH_IMPORT_ERROR
        _TRIMESH_IMPORT_ERROR = last_error
    return None


trimesh = import_optional_module("trimesh")


NON_BLOCKING_FLOOR_COVER_CATEGORIES = {
    "carpet",
    "rug",
    "mat",
    "towel",
    "blanket",
}


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def navigation_category_blocks_movement(category: str | None) -> bool:
    return str(category or "").strip().lower() not in NON_BLOCKING_FLOOR_COVER_CATEGORIES


def _load_floor_mesh(scene_path: Path):
    """Load floor mesh from StructureMesh/floor.glb as a single trimesh object."""
    if trimesh is None:
        global _TRIMESH_WARNING_EMITTED
        if not _TRIMESH_WARNING_EMITTED:
            detail = f": {_TRIMESH_IMPORT_ERROR}" if _TRIMESH_IMPORT_ERROR is not None else ""
            print(f"  [warn] trimesh unavailable, mesh-based floor occupancy disabled{detail}")
            _TRIMESH_WARNING_EMITTED = True
        return None

    floor_glb = scene_path / "StructureMesh" / "floor.glb"
    if not floor_glb.exists():
        return None

    try:
        loaded = trimesh.load(str(floor_glb), process=False)
    except Exception:
        return None

    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    elif isinstance(loaded, trimesh.Scene):
        try:
            mesh = loaded.dump(concatenate=True)
        except Exception:
            return None
    else:
        return None

    if mesh is None or len(mesh.faces) == 0:
        return None
    # StructureMesh GLBs are Y-up; convert back to Z-up world convention used
    # by layout.json and all navigation code.
    to_z_up = trimesh.transformations.rotation_matrix(math.pi / 2.0, [1, 0, 0])
    mesh = mesh.copy()
    mesh.apply_transform(to_z_up)
    return mesh


def _infer_asset_base(scene_path: Path | None) -> Path | None:
    if scene_path is None:
        return None
    root = scene_path.parent.parent
    candidates = [
        root / "internscenes_raw" / "asset_library",
        root / "asset_library",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _resolve_model_path(model_uid: str, asset_base: Path | None) -> Path | None:
    if asset_base is None or not model_uid:
        return None

    if model_uid.startswith("partnet_mobility/"):
        glb = asset_base / model_uid / "whole.glb"
        if glb.exists():
            return glb

    glb = asset_base / f"{model_uid}.glb"
    if glb.exists():
        return glb

    glb = asset_base / model_uid / "whole.glb"
    if glb.exists():
        return glb
    return None


_ASSET_MESH_CACHE: dict[str, object | None] = {}


def _load_asset_mesh(model_uid: str, asset_base: Path | None):
    if trimesh is None:
        return None
    cache_key = f"{asset_base}:{model_uid}"
    if cache_key in _ASSET_MESH_CACHE:
        return _ASSET_MESH_CACHE[cache_key]

    glb_path = _resolve_model_path(model_uid, asset_base)
    if glb_path is None:
        _ASSET_MESH_CACHE[cache_key] = None
        return None

    try:
        loaded = trimesh.load(str(glb_path), process=False)
    except Exception:
        _ASSET_MESH_CACHE[cache_key] = None
        return None

    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    elif isinstance(loaded, trimesh.Scene):
        try:
            mesh = loaded.to_geometry()
        except Exception:
            try:
                mesh = loaded.dump(concatenate=True)
            except Exception:
                mesh = None
    else:
        mesh = None

    if mesh is None or len(mesh.faces) == 0:
        _ASSET_MESH_CACHE[cache_key] = None
        return None

    _ASSET_MESH_CACHE[cache_key] = mesh
    return mesh


def _scale_matrix_for_instance(mesh_extents: np.ndarray, instance: dict) -> np.ndarray:
    target_size = np.asarray(instance["bbox"][3:6], dtype=np.float64)
    safe_extents = np.where(np.abs(mesh_extents) < 1e-6, 1.0, mesh_extents)
    category = instance.get("category", "")

    if category == "carpet":
        scale_factors = target_size / safe_extents
        if target_size[2] / max(target_size[0], 1e-6) > 150 or target_size[2] / max(target_size[1], 1e-6) > 150:
            if target_size[2] / max(target_size[0], 1e-6) > target_size[2] / max(target_size[1], 1e-6):
                rot = trimesh.transformations.rotation_matrix(0.5 * math.pi, [0, 1, 0])
                target_size = np.array([target_size[2], target_size[0], target_size[1]], dtype=np.float64)
                scale_factors = target_size / safe_extents
                scale_mat = np.diag([scale_factors[0], scale_factors[1], scale_factors[2] / 100.0, 1.0])
                return scale_mat @ rot
            rot = trimesh.transformations.rotation_matrix(0.5 * math.pi, [1, 0, 0])
            target_size = np.array([target_size[0], target_size[2], target_size[1]], dtype=np.float64)
            scale_factors = target_size / safe_extents
            scale_mat = np.diag([scale_factors[0], scale_factors[1], scale_factors[2] / 100.0, 1.0])
            return scale_mat @ rot
        return np.diag([scale_factors[0], scale_factors[1], scale_factors[2] / 100.0, 1.0])

    if category == "clothes":
        scale = target_size / safe_extents
        min_scale = float(np.min(scale))
        return np.diag([min_scale, min_scale, min_scale, 1.0])

    scale = target_size / safe_extents
    return np.diag([scale[0], scale[1], scale[2], 1.0])


def _transform_asset_mesh(mesh, instance: dict):
    mesh_t = mesh.copy()
    mesh_extents = np.asarray(mesh_t.bounding_box.extents, dtype=np.float64)
    transform = _scale_matrix_for_instance(mesh_extents, instance)

    euler = instance["bbox"][6:9]
    rot = trimesh.transformations.euler_matrix(
        float(euler[0]), float(euler[1]), float(euler[2]), axes="rzxy"
    )
    transform = rot @ transform
    transform[:3, 3] = np.asarray(instance["bbox"][0:3], dtype=np.float64)
    mesh_t.apply_transform(transform)
    return mesh_t


def _points_in_triangle(xs: np.ndarray, ys: np.ndarray, tri_xy: np.ndarray) -> np.ndarray:
    """Vectorized 2D point-in-triangle test (inclusive)."""
    ax, ay = tri_xy[0]
    bx, by = tri_xy[1]
    cx, cy = tri_xy[2]

    v0x = cx - ax
    v0y = cy - ay
    v1x = bx - ax
    v1y = by - ay
    denom = v0x * v1y - v1x * v0y
    if abs(float(denom)) < 1e-12:
        return np.zeros_like(xs, dtype=bool)

    v2x = xs - ax
    v2y = ys - ay
    inv = 1.0 / denom
    u = (v2x * v1y - v1x * v2y) * inv
    v = (v0x * v2y - v2x * v0y) * inv
    eps = 1e-8
    return (u >= -eps) & (v >= -eps) & ((u + v) <= (1.0 + eps))


def _rasterize_floor_mask(mesh, grid: "OccupancyGrid") -> np.ndarray | None:
    """Rasterize walkable floor triangles onto the occupancy grid."""
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if verts.size == 0 or faces.size == 0:
        return None

    normals = np.asarray(mesh.face_normals, dtype=np.float64)
    if normals.shape[0] == faces.shape[0]:
        keep = np.abs(normals[:, 2]) >= 0.7
        if np.any(keep):
            faces = faces[keep]
    if faces.size == 0:
        return None

    mask = np.zeros((grid.height, grid.width), dtype=bool)
    for face in faces:
        tri = verts[face, :2]
        min_x = float(np.min(tri[:, 0]))
        max_x = float(np.max(tri[:, 0]))
        min_y = float(np.min(tri[:, 1]))
        max_y = float(np.max(tri[:, 1]))

        gx1, gy1 = grid.world_to_grid(min_x, min_y)
        gx2, gy2 = grid.world_to_grid(max_x, max_y)
        if gx2 < gx1:
            gx1, gx2 = gx2, gx1
        if gy2 < gy1:
            gy1, gy2 = gy2, gy1
        if gx1 > grid.width - 1 or gy1 > grid.height - 1:
            continue

        xs = grid.min_x + (np.arange(gx1, gx2 + 1) + 0.5) * grid.resolution
        ys = grid.min_y + (np.arange(gy1, gy2 + 1) + 0.5) * grid.resolution
        xx, yy = np.meshgrid(xs, ys)
        inside = _points_in_triangle(xx, yy, tri)
        if np.any(inside):
            mask[gy1:gy2 + 1, gx1:gx2 + 1] |= inside

    return mask if np.any(mask) else None


def _rasterize_obstacle_mesh(mesh, grid: "OccupancyGrid",
                             z_min: float,
                             z_max: float) -> np.ndarray | None:
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if verts.size == 0 or faces.size == 0:
        return None

    mask = np.zeros((grid.height, grid.width), dtype=bool)
    tri_verts = verts[faces]
    tri_min_z = tri_verts[:, :, 2].min(axis=1)
    tri_max_z = tri_verts[:, :, 2].max(axis=1)
    keep = (tri_max_z > z_min) & (tri_min_z < z_max)
    if not np.any(keep):
        return None

    for tri in tri_verts[keep]:
        tri_xy = tri[:, :2]
        min_x = float(np.min(tri_xy[:, 0]))
        max_x = float(np.max(tri_xy[:, 0]))
        min_y = float(np.min(tri_xy[:, 1]))
        max_y = float(np.max(tri_xy[:, 1]))

        gx1, gy1 = grid.world_to_grid(min_x, min_y)
        gx2, gy2 = grid.world_to_grid(max_x, max_y)
        if gx2 < gx1:
            gx1, gx2 = gx2, gx1
        if gy2 < gy1:
            gy1, gy2 = gy2, gy1

        xs = grid.min_x + (np.arange(gx1, gx2 + 1) + 0.5) * grid.resolution
        ys = grid.min_y + (np.arange(gy1, gy2 + 1) + 0.5) * grid.resolution
        xx, yy = np.meshgrid(xs, ys)
        inside = _points_in_triangle(xx, yy, tri_xy)
        if np.any(inside):
            mask[gy1:gy2 + 1, gx1:gx2 + 1] |= inside

    return mask if np.any(mask) else None


def _inflate_obstacle_mask(mask: np.ndarray | None, resolution: float, radius_m: float) -> np.ndarray | None:
    """Inflate a rasterized obstacle mask by agent radius in world meters."""
    if mask is None:
        return None

    mask_bool = mask.astype(bool, copy=False)
    if not np.any(mask_bool) or radius_m <= 1e-6:
        return mask_bool

    ys, xs = np.where(mask_bool)
    pad_cells = max(1, int(math.ceil(float(radius_m) / float(resolution))))
    y0 = max(0, int(np.min(ys)) - pad_cells)
    y1 = min(mask_bool.shape[0], int(np.max(ys)) + pad_cells + 1)
    x0 = max(0, int(np.min(xs)) - pad_cells)
    x1 = min(mask_bool.shape[1], int(np.max(xs)) + pad_cells + 1)

    from scipy.ndimage import distance_transform_edt

    cropped = mask_bool[y0:y1, x0:x1]
    dist = distance_transform_edt(~cropped) * float(resolution)
    inflated_crop = dist <= float(radius_m) + 1e-9

    inflated = mask_bool.copy()
    inflated[y0:y1, x0:x1] = inflated_crop
    return inflated


def _build_floor_mask(scene_path: Path, grid: "OccupancyGrid", shrink_m: float) -> np.ndarray | None:
    """Build floor-constrained free-space mask from StructureMesh/floor.glb."""
    mesh = _load_floor_mesh(scene_path)
    if mesh is None:
        return None

    mask = _rasterize_floor_mask(mesh, grid)
    if mask is None:
        return None

    if shrink_m <= 1e-6:
        return mask

    from scipy.ndimage import distance_transform_edt
    dist = distance_transform_edt(mask.astype(np.uint8)) * grid.resolution
    shrunk = dist >= float(shrink_m)
    if np.any(shrunk):
        return shrunk
    return mask


class OccupancyGrid:
    """2D occupancy grid for navigation planning."""

    def __init__(self, min_x: float, min_y: float, max_x: float, max_y: float,
                 resolution: float, agent_radius: float):
        self.min_x = min_x
        self.min_y = min_y
        self.resolution = resolution
        self.agent_radius = agent_radius
        self.floor_mask: np.ndarray | None = None

        self.width = int(math.ceil((max_x - min_x) / resolution))
        self.height = int(math.ceil((max_y - min_y) / resolution))

        # 0 = free, 1 = occupied
        self.grid = np.zeros((self.height, self.width), dtype=np.uint8)

    def world_to_grid(self, x: float, y: float) -> tuple[int, int]:
        gx = int((x - self.min_x) / self.resolution)
        gy = int((y - self.min_y) / self.resolution)
        return max(0, min(gx, self.width - 1)), max(0, min(gy, self.height - 1))

    def grid_to_world(self, gx: int, gy: int) -> tuple[float, float]:
        x = self.min_x + (gx + 0.5) * self.resolution
        y = self.min_y + (gy + 0.5) * self.resolution
        return x, y

    def mark_obstacle(self, x: float, y: float, size_x: float, size_y: float,
                      rotation: float = 0.0):
        """Rasterize an oriented obstacle footprint onto the grid.

        The older AABB approximation over-blocked cells around rotated objects,
        which made some visually open floor points look non-walkable.
        """
        margin = self.agent_radius
        half_x = size_x / 2.0 + margin
        half_y = size_y / 2.0 + margin

        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)

        corners = []
        for dx in (-half_x, half_x):
            for dy in (-half_y, half_y):
                wx = x + dx * cos_r - dy * sin_r
                wy = y + dx * sin_r + dy * cos_r
                corners.append((wx, wy))

        min_x = min(p[0] for p in corners)
        max_x = max(p[0] for p in corners)
        min_y = min(p[1] for p in corners)
        max_y = max(p[1] for p in corners)

        gx1, gy1 = self.world_to_grid(min_x, min_y)
        gx2, gy2 = self.world_to_grid(max_x, max_y)
        if gx2 < gx1:
            gx1, gx2 = gx2, gx1
        if gy2 < gy1:
            gy1, gy2 = gy2, gy1

        xs = self.min_x + (np.arange(gx1, gx2 + 1) + 0.5) * self.resolution
        ys = self.min_y + (np.arange(gy1, gy2 + 1) + 0.5) * self.resolution
        xx, yy = np.meshgrid(xs, ys)

        dx = xx - x
        dy = yy - y
        local_x = cos_r * dx + sin_r * dy
        local_y = -sin_r * dx + cos_r * dy
        inside = (np.abs(local_x) <= half_x) & (np.abs(local_y) <= half_y)
        if np.any(inside):
            self.grid[gy1:gy2 + 1, gx1:gx2 + 1] |= inside.astype(np.uint8)

    def is_free(self, gx: int, gy: int) -> bool:
        if 0 <= gx < self.width and 0 <= gy < self.height:
            return self.grid[gy, gx] == 0
        return False

    def distance_to_nearest_obstacle(self, gx: int, gy: int) -> float:
        """Compute distance from grid cell to nearest obstacle."""
        from scipy.ndimage import distance_transform_edt
        if not hasattr(self, "_dist_map"):
            self._dist_map = distance_transform_edt(1 - self.grid) * self.resolution
        if 0 <= gx < self.width and 0 <= gy < self.height:
            return float(self._dist_map[gy, gx])
        return 0.0

    def get_distance_map(self) -> np.ndarray:
        """Get full distance-to-nearest-obstacle map (meters). Lazy-computed."""
        from scipy.ndimage import distance_transform_edt
        if not hasattr(self, "_dist_map"):
            self._dist_map = distance_transform_edt(1 - self.grid) * self.resolution
        return self._dist_map

    def is_path_traversable(self, waypoints_world: list[tuple[float, float]],
                            step_size: float = None) -> dict:
        """Check if a path defined by world-coordinate waypoints is fully traversable.

        Interpolates between waypoints at grid resolution and checks every
        intermediate cell is free.

        Returns:
            dict with keys:
              - traversable (bool): True if the entire path is collision-free
              - collision_point (list|None): first collision world coord, or None
              - collision_segment (int|None): segment index where collision occurs
              - traversable_fraction (float): fraction of path that is collision-free
              - path_length (float): total path length in meters
              - min_clearance (float): minimum distance to obstacle along path
        """
        if step_size is None:
            step_size = self.resolution * 0.5  # sub-cell accuracy

        total_steps = 0
        free_steps = 0
        collision_point = None
        collision_segment = None
        min_clearance = float("inf")
        path_length = 0.0

        dist_map = self.get_distance_map()

        for seg_idx in range(len(waypoints_world) - 1):
            x0, y0 = waypoints_world[seg_idx][:2]
            x1, y1 = waypoints_world[seg_idx + 1][:2]
            seg_len = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            path_length += seg_len
            n_steps = max(1, int(math.ceil(seg_len / step_size)))

            for s in range(n_steps + 1):
                t = s / n_steps
                wx = x0 + t * (x1 - x0)
                wy = y0 + t * (y1 - y0)

                gx, gy = self.world_to_grid(wx, wy)
                total_steps += 1

                if self.is_free(gx, gy):
                    free_steps += 1
                    clearance = float(dist_map[gy, gx]) if (
                        0 <= gy < self.height and 0 <= gx < self.width) else 0.0
                    min_clearance = min(min_clearance, clearance)
                else:
                    if collision_point is None:
                        collision_point = [wx, wy]
                        collision_segment = seg_idx

        traversable_fraction = free_steps / max(total_steps, 1)
        return {
            "traversable": collision_point is None,
            "collision_point": collision_point,
            "collision_segment": collision_segment,
            "traversable_fraction": round(traversable_fraction, 4),
            "path_length": round(path_length, 3),
            "min_clearance": round(min_clearance, 3) if min_clearance < float("inf") else 0.0,
        }


def build_occupancy_grid(layout: list[dict], config: dict,
                         scene_path: str | Path | None = None) -> OccupancyGrid:
    """Build occupancy grid from scene layout."""
    nav_cfg = config["navigation"]
    resolution = nav_cfg["grid_resolution"]
    agent_radius = float(nav_cfg["agent_radius"])
    passable_margin = float(nav_cfg.get("passable_margin", 0.0))
    effective_radius = agent_radius + passable_margin

    # Compute scene bounds
    positions = []
    for obj in layout:
        bbox = obj.get("bbox", [])
        if len(bbox) >= 6:
            x, y = bbox[0], bbox[1]
            sx, sy = bbox[3], bbox[4]
            positions.append((x - sx, y - sy))
            positions.append((x + sx, y + sy))

    if not positions:
        raise ValueError("No positioned objects in layout")

    floor_mesh_bounds = None
    scene_path_obj = Path(scene_path) if scene_path is not None else None
    if scene_path_obj is not None:
        floor_mesh = _load_floor_mesh(scene_path_obj)
        if floor_mesh is not None:
            floor_mesh_bounds = floor_mesh.bounds

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    if floor_mesh_bounds is not None:
        xs.extend([float(floor_mesh_bounds[0][0]), float(floor_mesh_bounds[1][0])])
        ys.extend([float(floor_mesh_bounds[0][1]), float(floor_mesh_bounds[1][1])])
    margin = 2.0  # Extra margin around scene

    grid = OccupancyGrid(
        min(xs) - margin, min(ys) - margin,
        max(xs) + margin, max(ys) + margin,
        resolution, effective_radius,
    )

    # If structure floor is available, constrain free space to floor region.
    if scene_path_obj is not None:
        floor_mask = _build_floor_mask(scene_path_obj, grid, shrink_m=effective_radius)
        grid.floor_mask = floor_mask
        if floor_mask is not None:
            grid.grid[:, :] = 1
            grid.grid[floor_mask] = 0
        else:
            print(f"  [warn] {scene_path_obj.name}: floor mask unavailable, fallback to bbox-only occupancy")

    asset_base = _infer_asset_base(scene_path_obj)

    # Mark obstacles
    for obj in layout:
        bbox = obj.get("bbox", [])
        if len(bbox) >= 9:
            if not navigation_category_blocks_movement(obj.get("category")):
                continue
            x, y = bbox[0], bbox[1]
            sx, sy = bbox[3], bbox[4]
            rot = bbox[6] if len(bbox) > 6 else 0.0

            height = bbox[5] if len(bbox) > 5 else 1.0
            center_z = bbox[2] if len(bbox) > 2 else 0.0
            obj_bottom = center_z - height / 2.0
            obj_top = center_z + height / 2.0

            # Match render-time occupancy convention: only geometry intersecting
            # robot body volume [0.10m, 1.4m] blocks movement.
            if obj_bottom < 1.4 and obj_top > 0.10:
                used_mesh = False
                model_uid = obj.get("model_uid", "")
                asset_mesh = _load_asset_mesh(model_uid, asset_base) if model_uid else None
                if asset_mesh is not None:
                    try:
                        world_mesh = _transform_asset_mesh(asset_mesh, obj)
                        mesh_mask = _rasterize_obstacle_mesh(world_mesh, grid, z_min=0.10, z_max=1.4)
                        if mesh_mask is not None:
                            mesh_mask = _inflate_obstacle_mask(mesh_mask, grid.resolution, effective_radius)
                            grid.grid[mesh_mask] = 1
                            used_mesh = True
                    except Exception:
                        used_mesh = False
                if not used_mesh:
                    grid.mark_obstacle(x, y, sx, sy, rot)

    return grid


def astar(grid: OccupancyGrid, start: tuple[int, int], goal: tuple[int, int]) -> Optional[list[tuple[int, int]]]:
    """A* pathfinding on the occupancy grid."""
    if not grid.is_free(*start) or not grid.is_free(*goal):
        return None

    def heuristic(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}

    # 8-connected grid
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return list(reversed(path))

        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            if not grid.is_free(*neighbor):
                continue

            # Diagonal movement costs more
            move_cost = math.sqrt(dx * dx + dy * dy)
            tentative_g = g_score[current] + move_cost

            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))

    return None  # No path found


def _line_of_sight(grid: OccupancyGrid, x0: float, y0: float,
                   x1: float, y1: float) -> bool:
    """Check if a straight line between two world points is collision-free.

    Uses Bresenham-like stepping at sub-cell resolution.
    """
    step_size = grid.resolution * 0.5
    dx = x1 - x0
    dy = y1 - y0
    dist = math.sqrt(dx * dx + dy * dy)
    n_steps = max(1, int(math.ceil(dist / step_size)))

    for s in range(n_steps + 1):
        t = s / n_steps
        wx = x0 + t * dx
        wy = y0 + t * dy
        gx, gy = grid.world_to_grid(wx, wy)
        if not grid.is_free(gx, gy):
            return False
    return True


def simplify_path_obstacle_aware(
    path: list[tuple[float, float, float]],
    grid: OccupancyGrid,
    tolerance: float,
) -> list[tuple[float, float, float]]:
    """Simplify path while guaranteeing no segment passes through obstacles.

    Greedy visibility-based simplification:
      From current waypoint, find the farthest reachable point via
      straight line-of-sight. This guarantees the simplified path is
      fully traversable since every straight-line segment is collision-free.
    """
    if len(path) < 3:
        return path

    simplified = [path[0]]
    i = 0

    while i < len(path) - 1:
        # Try to reach as far ahead as possible via straight line
        best_j = i + 1  # worst case: keep the next point
        for j in range(len(path) - 1, i + 1, -1):
            if _line_of_sight(grid,
                              path[i][0], path[i][1],
                              path[j][0], path[j][1]):
                best_j = j
                break
        simplified.append(path[best_j])
        i = best_j

    return simplified


def _rdp_simplify(path: list[tuple[float, float, float]],
                  tolerance: float) -> list[tuple[float, float, float]]:
    """Ramer-Douglas-Peucker simplification (geometry only)."""
    if len(path) < 3:
        return path

    def point_line_distance(point, start, end):
        if start == end:
            return math.sqrt(sum((a - b) ** 2 for a, b in zip(point, start)))

        line_len_sq = sum((a - b) ** 2 for a, b in zip(end, start))
        t = max(0, min(1, sum((p - s) * (e - s) for p, s, e in zip(point, start, end)) / line_len_sq))
        projection = tuple(s + t * (e - s) for s, e in zip(start, end))
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(point, projection)))

    # Find farthest point
    max_dist = 0
    max_idx = 0
    for i in range(1, len(path) - 1):
        dist = point_line_distance(path[i], path[0], path[-1])
        if dist > max_dist:
            max_dist = dist
            max_idx = i

    if max_dist > tolerance:
        left = _rdp_simplify(path[:max_idx + 1], tolerance)
        right = _rdp_simplify(path[max_idx:], tolerance)
        return left[:-1] + right
    else:
        return [path[0], path[-1]]


def world_to_robot_frame(
    waypoints: list[tuple[float, float, float]],
    robot_pos: tuple[float, float, float],
    robot_yaw: float,
) -> list[dict]:
    """Convert world-coordinate waypoints to robot-local coordinate frame.

    Robot frame convention:
      - Origin: robot position (camera position projected to floor)
      - +X axis: robot's right
      - +Y axis: robot's forward (heading direction)
      - Z: height (always 0 for floor-level navigation)

    Args:
        waypoints: list of (x, y, z) in world coordinates
        robot_pos: robot's world position (x, y, z)
        robot_yaw: robot's heading angle in radians (0 = +X world, pi/2 = +Y world)

    Returns:
        list of dicts with:
          - local_xy: [forward, right] in robot frame (meters)
          - world_xy: [x, y] in world frame (meters)
          - distance: straight-line distance from robot to this waypoint
          - bearing_deg: angle from robot's forward to waypoint (+ = right, - = left)
          - cumulative_distance: path distance from robot to this waypoint along the path
    """
    cos_yaw = math.cos(robot_yaw)
    sin_yaw = math.sin(robot_yaw)

    results = []
    cumulative_dist = 0.0

    for i, wp in enumerate(waypoints):
        # Vector from robot to waypoint in world frame
        dx = wp[0] - robot_pos[0]
        dy = wp[1] - robot_pos[1]

        # Rotate into robot frame
        # Robot forward = [cos(yaw), sin(yaw)] in world
        # Robot right   = [sin(yaw), -cos(yaw)] in world (for right-hand rule with Z up)
        # local_forward = dx * cos(yaw) + dy * sin(yaw)
        # local_right   = dx * sin(yaw) - dy * cos(yaw)
        local_forward = dx * cos_yaw + dy * sin_yaw
        local_right = dx * sin_yaw - dy * cos_yaw

        straight_dist = math.sqrt(dx * dx + dy * dy)

        # Bearing: angle from forward axis (+ = right, - = left)
        bearing = math.atan2(local_right, local_forward) if straight_dist > 0.01 else 0.0

        # Cumulative path distance
        if i > 0:
            seg_dx = wp[0] - waypoints[i - 1][0]
            seg_dy = wp[1] - waypoints[i - 1][1]
            cumulative_dist += math.sqrt(seg_dx * seg_dx + seg_dy * seg_dy)

        results.append({
            "local_xy": [round(local_forward, 3), round(local_right, 3)],
            "world_xy": [round(wp[0], 3), round(wp[1], 3)],
            "distance": round(straight_dist, 3),
            "bearing_deg": round(math.degrees(bearing), 1),
            "cumulative_distance": round(cumulative_dist, 3),
        })

    return results


def path_to_directions(path: list[tuple[float, float, float]],
                       initial_yaw: float = None) -> list[dict]:
    """Convert a waypoint path into turn-by-turn navigation directions.

    Returns a list of direction steps with:
      - action: "forward" | "turn_left" | "turn_right"
      - distance (meters, for forward)
      - angle (degrees, for turns)
    """
    if len(path) < 2:
        return []

    directions = []
    # Compute segment headings (radians, 0=+X, pi/2=+Y)
    headings = []
    for i in range(len(path) - 1):
        dx = path[i + 1][0] - path[i][0]
        dy = path[i + 1][1] - path[i][1]
        headings.append(math.atan2(dy, dx))

    # Initial orientation: facing first segment direction, or use provided yaw
    current_heading = initial_yaw if initial_yaw is not None else headings[0]

    for i, heading in enumerate(headings):
        # Compute turn needed
        turn_angle = heading - current_heading
        # Normalize to [-pi, pi]
        turn_angle = (turn_angle + math.pi) % (2 * math.pi) - math.pi
        turn_deg = math.degrees(turn_angle)

        if abs(turn_deg) > 5.0:  # Threshold for meaningful turn
            directions.append({
                "action": "turn_left" if turn_deg > 0 else "turn_right",
                "angle": round(abs(turn_deg), 1),
            })

        # Forward distance
        dist = math.sqrt(
            (path[i + 1][0] - path[i][0]) ** 2 +
            (path[i + 1][1] - path[i][1]) ** 2
        )
        if dist > 0.05:  # Threshold for meaningful movement
            directions.append({
                "action": "forward",
                "distance": round(dist, 2),
            })
        current_heading = heading

    return directions


def compute_navigation_gt(
    layout: list[dict],
    camera_info: dict,
    target_obj: dict,
    config: dict,
    grid: OccupancyGrid = None,
) -> Optional[dict]:
    """
    Compute navigation ground truth for a single (camera, target) pair.

    Returns dict with:
      GT-1: optimal_path + directions (for training)
      GT-2: path validation result using full traversability grid
    """
    nav_cfg = config["navigation"]

    # Build occupancy grid (reuse if passed in)
    if grid is None:
        grid = build_occupancy_grid(layout, config)

    # Start point: camera position
    cam_pos = camera_info["position"]
    start_grid = grid.world_to_grid(cam_pos[0], cam_pos[1])

    # End point: target object position
    target_bbox = target_obj.get("bbox", [])
    if len(target_bbox) < 3:
        return None
    target_pos = (target_bbox[0], target_bbox[1])
    goal_grid = grid.world_to_grid(*target_pos)

    # Find nearest free cell if start/goal is occupied
    def find_nearest_free(gx, gy, max_search=50):
        for r in range(max_search):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) == r or abs(dy) == r:
                        nx, ny = gx + dx, gy + dy
                        if grid.is_free(nx, ny):
                            return (nx, ny)
        return None

    if not grid.is_free(*start_grid):
        start_grid = find_nearest_free(*start_grid)
        if start_grid is None:
            return None

    if not grid.is_free(*goal_grid):
        goal_grid = find_nearest_free(*goal_grid)
        if goal_grid is None:
            return None

    # Run A*
    grid_path = astar(grid, start_grid, goal_grid)
    if grid_path is None:
        return None

    # Convert to world coordinates (3D: use floor height z=0)
    floor_z = 0.0
    world_path = []
    for gx, gy in grid_path:
        wx, wy = grid.grid_to_world(gx, gy)
        world_path.append((wx, wy, floor_z))

    # Simplify path (obstacle-aware: ensures no segment clips through walls)
    simplified_path = simplify_path_obstacle_aware(
        world_path, grid, nav_cfg["path_simplification"]
    )

    # Compute path length
    path_length = 0.0
    for i in range(1, len(simplified_path)):
        path_length += math.sqrt(
            sum((a - b) ** 2 for a, b in zip(simplified_path[i], simplified_path[i-1]))
        )

    # GT-1: Turn-by-turn directions from the camera's look angle
    initial_yaw = camera_info.get("look_angle", None)
    directions = path_to_directions(simplified_path, initial_yaw)

    # GT-1b: Robot-frame waypoints (origin = robot position, +Y = forward)
    robot_pos = (cam_pos[0], cam_pos[1], 0.0)
    # Robot heading: look_angle is the yaw in world frame
    robot_yaw = initial_yaw if initial_yaw is not None else 0.0
    robot_frame_wps = world_to_robot_frame(simplified_path, robot_pos, robot_yaw)

    # GT-2: Validate the optimal path itself (should be 100% traversable)
    # This also serves as a sanity check
    validation = grid.is_path_traversable(simplified_path)

    # Compute per-waypoint clearance
    dist_map = grid.get_distance_map()
    waypoint_clearances = []
    for wp in simplified_path:
        gx, gy = grid.world_to_grid(wp[0], wp[1])
        clearance = float(dist_map[gy, gx]) if (
            0 <= gy < grid.height and 0 <= gx < grid.width) else 0.0
        waypoint_clearances.append(round(clearance, 3))

    return {
        # === GT-1: Training — optimal path + directions ===
        "optimal_path": [list(p) for p in simplified_path],
        "robot_frame_waypoints": robot_frame_wps,
        "directions": directions,
        "path_length": round(path_length, 3),
        "num_path_points": len(simplified_path),
        "waypoint_clearances": waypoint_clearances,

        # === GT-2: Benchmark — validation of this path ===
        "path_validation": validation,

        # === Robot / Metadata ===
        "start_position": list(cam_pos),
        "goal_position": list(simplified_path[-1]),  # actual reachable endpoint (nearest free cell to target)
        "start_yaw_rad": round(robot_yaw, 4),
        "agent_radius": grid.agent_radius,
        "target_position": [target_bbox[0], target_bbox[1], target_bbox[2]],
        "target_category": target_obj.get("category", "unknown"),
        "target_id": target_obj.get("id", -1),
        "grid_resolution": grid.resolution,
    }


def select_target_objects(
    layout: list[dict],
    camera_info: dict,
    config: dict,
) -> list[dict]:
    """Select target objects for navigation tasks.
    
    Targets are filtered to objects that are visible from the camera
    (present in camera_info['visible_objects'] with raycast_verified=True
    preferred, but all frustum-visible objects are eligible).
    This ensures VQA questions reference objects the VLM can actually see.
    """
    nav_cfg = config["navigation"]
    filter_cfg = config["filter"]
    num_targets = config["vqa"]["targets_per_scene"]

    cam_pos = camera_info["position"]

    # Build set of visible object IDs from render output
    visible_ids = set()
    verified_ids = set()
    for vo in camera_info.get("visible_objects", []):
        obj_id = vo.get("id")
        if obj_id is not None:
            visible_ids.add(obj_id)
            if vo.get("raycast_verified", False):
                verified_ids.add(obj_id)
    # Also include frustum_visible for broader coverage
    for fv in camera_info.get("frustum_visible", []):
        fv_id = fv.get("id")
        if fv_id is not None:
            visible_ids.add(fv_id)

    # Score objects by suitability as navigation targets
    candidates = []
    for obj in layout:
        bbox = obj.get("bbox", [])
        if len(bbox) < 6:
            continue

        obj_id = obj.get("id")

        # Skip objects not visible from this camera
        if obj_id not in visible_ids:
            continue

        # Distance from camera
        dist = math.sqrt((bbox[0] - cam_pos[0]) ** 2 + (bbox[1] - cam_pos[1]) ** 2)

        # Filter by distance range
        if dist < filter_cfg["min_nav_distance"] or dist > filter_cfg["max_nav_distance"]:
            continue

        # Prefer larger, more recognizable objects
        volume = bbox[3] * bbox[4] * bbox[5]
        category = obj.get("category", "unknown")

        # Score: prefer moderate distance + reasonable size + raycast verified
        score = 1.0 / (1.0 + abs(dist - 4.0))  # Prefer ~4m distance
        score *= min(volume, 2.0)  # Prefer larger objects, capped
        if obj_id in verified_ids:
            score *= 1.5  # Bonus for raycast-verified visibility

        candidates.append({
            "obj": obj,
            "score": score,
            "distance": dist,
            "raycast_verified": obj_id in verified_ids,
        })

    # Sort by score and select top-N diverse targets
    candidates.sort(key=lambda x: -x["score"])

    selected = []
    selected_categories = set()
    for c in candidates:
        cat = c["obj"].get("category", "unknown")
        # Prefer diverse categories
        if cat not in selected_categories or len(selected) < num_targets:
            selected.append(c["obj"])
            selected_categories.add(cat)
        if len(selected) >= num_targets:
            break

    return selected


def export_traversability_map(grid: OccupancyGrid, output_dir: Path, scene_id: str):
    """Export the full traversability map as GT-2 for benchmark path validation.

    Saves:
      - traversability_grid.png: 8-bit image (0=obstacle, 255=free)
      - traversability_meta.json: coordinate transform metadata
      - clearance_grid.png: 16-bit distance-to-obstacle map (mm units)

    External tools / evaluation scripts can load these to validate any
    arbitrary path a VLM generates:
      1. Load traversability_meta.json for world↔pixel coordinate mapping
      2. Convert VLM waypoints to pixel coords
      3. Check that all pixels along the path are free (value > 0 in traversability_grid.png)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- traversability_grid.png: binary free/occupied ---
    # grid.grid: 0=free, 1=occupied  →  image: 0=occupied, 255=free
    trav_img = ((1 - grid.grid) * 255).astype(np.uint8)
    Image.fromarray(trav_img).save(output_dir / "traversability_grid.png")

    # --- clearance_grid.png: distance to nearest obstacle in mm (16-bit) ---
    dist_map = grid.get_distance_map()
    # Convert meters → millimeters, cap at 65535mm = 65.5m
    clearance_mm = np.clip(dist_map * 1000, 0, 65535).astype(np.uint16)
    Image.fromarray(clearance_mm).save(output_dir / "clearance_grid.png")

    # --- traversability_meta.json: coordinate mapping ---
    meta = {
        "scene_id": scene_id,
        "grid_width": grid.width,
        "grid_height": grid.height,
        "resolution": grid.resolution,
        "agent_radius": grid.agent_radius,
        "obstacle_inflation": {
            "description": "Obstacles are Minkowski-inflated by agent_radius. "
                           "A pixel marked free means the robot CENTER can be there "
                           "without its body (radius=agent_radius) touching any obstacle. "
                           "When validating a path, only the center-line needs to be checked.",
            "agent_radius_m": grid.agent_radius,
            "agent_diameter_m": grid.agent_radius * 2,
            "agent_height_m": 1.4,
        },
        "origin": [grid.min_x, grid.min_y],  # world coords of pixel (0,0) corner
        "world_to_pixel": {
            "description": "pixel_x = (world_x - origin_x) / resolution; pixel_y = (world_y - origin_y) / resolution",
            "origin_x": grid.min_x,
            "origin_y": grid.min_y,
            "scale": 1.0 / grid.resolution,
        },
        "pixel_to_world": {
            "description": "world_x = origin_x + (pixel_x + 0.5) * resolution; world_y = origin_y + (pixel_y + 0.5) * resolution",
        },
        "free_area_m2": round(float(np.sum(1 - grid.grid)) * grid.resolution ** 2, 2),
        "occupied_area_m2": round(float(np.sum(grid.grid)) * grid.resolution ** 2, 2),
    }

    with open(output_dir / "traversability_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def process_scene(scene_dir: str, renders_dir: str, output_dir: str, config: dict):
    """Process one scene: compute navigation GT for all (camera, target) pairs.

    Outputs per scene:
      GT-1 (training): navigation_gt.json with optimal_path + directions
      GT-2 (benchmark): traversability_grid.png + clearance_grid.png + traversability_meta.json
    """
    scene_path = Path(scene_dir)
    renders_path = Path(renders_dir) / scene_path.name
    output_path = Path(output_dir) / scene_path.name
    output_path.mkdir(parents=True, exist_ok=True)

    # Load layout
    with open(scene_path / "layout.json") as f:
        layout = json.load(f)

    # Load camera info
    cameras_file = renders_path / "cameras.json"
    if not cameras_file.exists():
        print(f"  No cameras.json found for {scene_path.name}")
        return

    with open(cameras_file) as f:
        cameras = json.load(f)

    # Skip scenes with no valid cameras (0 views)
    if len(cameras) == 0:
        print(f"  Skipping {scene_path.name} (no valid camera views)")
        return

    # Build occupancy grid ONCE per scene (reused across all camera-target pairs)
    grid = build_occupancy_grid(layout, config, scene_path=scene_path)

    # === GT-2: Export full traversability map (per-scene, not per-pair) ===
    trav_meta = export_traversability_map(grid, output_path, scene_path.name)
    print(f"  Traversability map: {grid.width}x{grid.height} cells, "
          f"free={trav_meta['free_area_m2']}m², occupied={trav_meta['occupied_area_m2']}m²")

    # === GT-1: Optimal paths for each (camera, target) pair ===
    scene_gt = {
        "scene_id": scene_path.name,
        "traversability_grid_file": "traversability_grid.png",
        "clearance_grid_file": "clearance_grid.png",
        "traversability_meta_file": "traversability_meta.json",
        "navigation_pairs": [],
    }

    for cam_info in cameras:
        view_id = cam_info["view_id"]

        # Select target objects
        targets = select_target_objects(layout, cam_info, config)

        for target in targets:
            gt = compute_navigation_gt(layout, cam_info, target, config, grid=grid)
            if gt is None:
                continue

            gt["view_id"] = view_id
            scene_gt["navigation_pairs"].append(gt)

    print(f"  Generated {len(scene_gt['navigation_pairs'])} navigation pairs")

    with open(output_path / "navigation_gt.json", "w") as f:
        json.dump(scene_gt, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Build Navigation Ground Truth")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--scenes-dir", required=True)
    parser.add_argument("--renders-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--scene-id", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = args.output_dir or config["data"]["gt_paths_dir"]

    scenes_dir = Path(args.scenes_dir)
    if args.scene_id:
        scene_dirs = [scenes_dir / args.scene_id]
    else:
        scene_dirs = sorted([d for d in scenes_dir.iterdir() if d.is_dir() and (d / "layout.json").exists()])

    print(f"Processing {len(scene_dirs)} scenes for navigation GT")

    total_pairs = 0
    skipped = 0
    for i, scene_dir in enumerate(scene_dirs):
        # Skip if already processed
        gt_file = Path(output_dir) / scene_dir.name / "navigation_gt.json"
        if gt_file.exists():
            skipped += 1
            continue
        print(f"[{i+1}/{len(scene_dirs)}] {scene_dir.name}")
        try:
            process_scene(str(scene_dir), args.renders_dir, output_dir, config)
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nDone. Processed {len(scene_dirs) - skipped} scenes ({skipped} skipped).")


# ============================================================
# GT-2 Utility: Validate arbitrary paths from VLM predictions
# ============================================================

def validate_vlm_path(
    gt_dir: str,
    waypoints: list[list[float]],
    robot_frame: bool = False,
    start_position: list[float] = None,
    start_yaw_rad: float = None,
) -> dict:
    """Validate a VLM-predicted path against the traversability map.

    This function is the core of GT-2 usage. It loads the pre-exported
    traversability grid and checks if the VLM's predicted path is
    collision-free.

    The traversability grid is already Minkowski-inflated by agent_radius,
    so checking the center-line is sufficient — the robot's full body
    (radius=0.30m) is accounted for.

    Args:
        gt_dir: Path to scene's GT directory (containing traversability_grid.png
                and traversability_meta.json)
        waypoints: List of [x, y] or [x, y, z] waypoints. Either world coords
                   or robot-local coords (if robot_frame=True).
        robot_frame: If True, waypoints are in robot-local frame
                     [forward, right] and will be converted to world coords
                     using start_position and start_yaw_rad.
        start_position: Robot start position [x, y, z] in world coords.
                        Required if robot_frame=True.
        start_yaw_rad: Robot heading in radians. Required if robot_frame=True.

    Returns:
        dict with:
          - traversable (bool): entire path collision-free?
          - collision_point ([x,y]|None): first collision world coord
          - collision_segment (int|None): which segment collided
          - traversable_fraction (float): fraction of path that is free
          - path_length (float): total path length in meters
          - min_clearance (float): minimum clearance to obstacle (meters)
          - agent_radius (float): robot radius already accounted for

    Example usage in evaluation script::

        from build_navigation_gt import validate_vlm_path

        vlm_waypoints = [[1.2, 3.4], [2.1, 4.5], [3.0, 5.6]]  # from VLM output
        result = validate_vlm_path("/datadisk/NavBench3D/gt_paths/scene001", vlm_waypoints)
        if result["traversable"]:
            print("VLM path is valid!")
        else:
            print(f"Collision at {result['collision_point']}")
    """
    gt_path = Path(gt_dir)

    # Load traversability image
    trav_img = np.array(Image.open(gt_path / "traversability_grid.png"))
    # Load metadata
    with open(gt_path / "traversability_meta.json") as f:
        meta = json.load(f)

    resolution = meta["resolution"]
    agent_radius = meta.get("agent_radius", 0.30)
    origin_x = meta["world_to_pixel"]["origin_x"]
    origin_y = meta["world_to_pixel"]["origin_y"]
    height, width = trav_img.shape

    # Convert robot-frame waypoints to world coords if needed
    if robot_frame:
        if start_position is None or start_yaw_rad is None:
            raise ValueError("start_position and start_yaw_rad are required when robot_frame=True")
        cos_yaw = math.cos(start_yaw_rad)
        sin_yaw = math.sin(start_yaw_rad)
        world_waypoints = []
        for wp in waypoints:
            # wp = [forward, right] in robot frame
            local_fwd = wp[0]
            local_right = wp[1]
            # inverse rotation: world = robot_pos + R^T * local
            wx = start_position[0] + local_fwd * cos_yaw + local_right * sin_yaw
            wy = start_position[1] + local_fwd * sin_yaw - local_right * cos_yaw
            world_waypoints.append([wx, wy])
        waypoints = world_waypoints

    # Build a lightweight grid-like checker
    step_size = resolution * 0.5  # sub-cell accuracy

    total_steps = 0
    free_steps = 0
    collision_point = None
    collision_segment = None
    min_clearance = float("inf")
    path_length = 0.0

    # Optionally load clearance map
    clearance_file = gt_path / "clearance_grid.png"
    clearance_map = None
    if clearance_file.exists():
        clearance_map = np.array(Image.open(clearance_file)).astype(np.float32) / 1000.0  # mm→m

    for seg_idx in range(len(waypoints) - 1):
        x0, y0 = waypoints[seg_idx][0], waypoints[seg_idx][1]
        x1, y1 = waypoints[seg_idx + 1][0], waypoints[seg_idx + 1][1]
        seg_len = math.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
        path_length += seg_len
        n_steps = max(1, int(math.ceil(seg_len / step_size)))

        for s in range(n_steps + 1):
            t = s / n_steps
            wx = x0 + t * (x1 - x0)
            wy = y0 + t * (y1 - y0)

            # World to pixel
            px = int((wx - origin_x) / resolution)
            py = int((wy - origin_y) / resolution)

            # Out-of-bounds = collision (path goes outside known scene)
            if px < 0 or px >= width or py < 0 or py >= height:
                total_steps += 1
                if collision_point is None:
                    collision_point = [round(wx, 3), round(wy, 3)]
                    collision_segment = seg_idx
                continue

            total_steps += 1

            if trav_img[py, px] > 0:  # free
                free_steps += 1
                if clearance_map is not None:
                    min_clearance = min(min_clearance, float(clearance_map[py, px]))
            else:
                if collision_point is None:
                    collision_point = [round(wx, 3), round(wy, 3)]
                    collision_segment = seg_idx

    traversable_fraction = free_steps / max(total_steps, 1)
    return {
        "traversable": collision_point is None,
        "collision_point": collision_point,
        "collision_segment": collision_segment,
        "traversable_fraction": round(traversable_fraction, 4),
        "path_length": round(path_length, 3),
        "min_clearance": round(min_clearance, 3) if min_clearance < float("inf") else 0.0,
        "agent_radius": agent_radius,
    }


if __name__ == "__main__":
    main()
