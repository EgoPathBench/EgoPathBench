"""
NavBench3D - Step 2: Render scenes using Blender.

Renders RGB, depth, normal, semantic segmentation, and top-down views
for each filtered scene.

Usage:
    blender --background --python scripts/render_scenes.py -- --config configs/default.yaml --scenes-dir data/scenes

    Or without Blender (generates render commands):
    python scripts/render_scenes.py --config configs/default.yaml --scenes-dir data/scenes --dry-run
"""

import os
import sys
import json
import math
import argparse
import traceback
from collections import Counter
from typing import Iterable, Mapping
from pathlib import Path
import numpy as np

# Check if running inside Blender
from pathlib import Path

# Ensure local scripts are importable inside Blender.
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
EXTRA_PY_SITE = os.environ.get("NAVBENCH_PY_SITE")
if EXTRA_PY_SITE and EXTRA_PY_SITE not in sys.path:
    sys.path.insert(0, EXTRA_PY_SITE)

try:
    import bpy
    import mathutils
    from bpy_extras.object_utils import world_to_camera_view
    IN_BLENDER = True
except ImportError:
    world_to_camera_view = None
    IN_BLENDER = False

try:
    import yaml
except ImportError:
    yaml = None

try:
    from PIL import Image
except ImportError:
    Image = None

from object_index_utils import OBJECT_INDEX_PNG_MAX_VALUE, decode_object_index_array
from affordance_facts import build_affordance_fact_metadata as shared_build_affordance_fact_metadata
from scene_nav_facts import (
    build_scene_nav_facts,
    build_view_bundle_from_scene_nav_facts,
    save_scene_nav_facts,
)
from tier_policy import (
    classify_affordance_tier,
    classify_boundary_axis,
    classify_clutter_axis,
    classify_embodiment_axis,
    classify_embodiment_gap_axis,
    classify_geometry_axis,
    classify_navigation_tier,
    classify_reference_axis,
)

DEBUG = bool(os.environ.get("NB3D_DEBUG"))


def load_config(config_path: str) -> dict:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load config files.")
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_cycles_gpu_selection(
    env: Mapping[str, str] | None,
    gpu_count: int,
) -> list[int] | None:
    """Resolve explicit Cycles GPU selection from environment variables.

    When unset, callers should keep Blender's default "use all detected GPUs"
    behavior. This allows resume workers to pin each Blender process to a
    single device via ``NB3D_CYCLES_DEVICE_INDEX``.
    """
    if gpu_count <= 0:
        return None
    env_map = env or os.environ
    raw_index = str(env_map.get("NB3D_CYCLES_DEVICE_INDEX", "")).strip()
    if not raw_index:
        return None
    try:
        gpu_index = int(raw_index)
    except ValueError as exc:
        raise ValueError(
            f"NB3D_CYCLES_DEVICE_INDEX must be an integer, got {raw_index!r}"
        ) from exc
    if gpu_index < 0 or gpu_index >= gpu_count:
        raise ValueError(
            f"NB3D_CYCLES_DEVICE_INDEX={gpu_index} out of range for {gpu_count} GPUs"
        )
    return [gpu_index]


def setup_blender_scene(resolution: list[int], render_depth=True, render_normal=True):
    """Configure Blender render settings with all required passes."""
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 64
    scene.cycles.use_denoising = True
    scene.render.resolution_x = resolution[0]
    scene.render.resolution_y = resolution[1]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"

    # Enable GPU if available (prefer OPTIX/CUDA/HIP/ONEAPI/METAL in that order)
    prefs = bpy.context.preferences.addons.get("cycles")
    if prefs:
        cprefs = prefs.preferences
        chosen = None
        candidates = ["OPTIX", "CUDA", "HIP", "ONEAPI", "METAL"]
        for dev_type in candidates:
            try:
                cprefs.compute_device_type = dev_type
            except Exception:
                continue
            try:
                cprefs.get_devices()
            except Exception:
                pass
            devices = getattr(cprefs, "devices", [])
            gpus = [d for d in devices if getattr(d, "type", "") != "CPU"]
            if gpus:
                chosen = dev_type
                selected_gpu_indices = resolve_cycles_gpu_selection(
                    os.environ,
                    gpu_count=len(gpus),
                )
                for d in devices:
                    d.use = False
                if selected_gpu_indices is None:
                    for gpu in gpus:
                        gpu.use = True
                else:
                    for gpu_idx in selected_gpu_indices:
                        gpus[gpu_idx].use = True
                bpy.context.scene.cycles.device = "GPU"
                print(f"[cycles] Using {chosen} with devices: "
                      f"{[d.name for d in devices if d.use]}")
                break
        if chosen is None:
            bpy.context.scene.cycles.device = "CPU"
            print("[cycles] No GPU devices found; using CPU")

    # Enable render passes upfront
    vl = scene.view_layers["ViewLayer"]
    if render_depth:
        vl.use_pass_z = True
    if render_normal:
        vl.use_pass_normal = True
    vl.use_pass_object_index = True


def clear_scene():
    """Remove all objects and orphan data from the scene to prevent memory leaks."""
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=True)
    for collection in bpy.data.collections:
        bpy.data.collections.remove(collection)
    # Purge orphan data blocks to free memory
    for block_type in [bpy.data.meshes, bpy.data.materials, bpy.data.textures,
                        bpy.data.images, bpy.data.lights, bpy.data.cameras,
                        bpy.data.node_groups, bpy.data.actions]:
        for block in block_type:
            if block.users == 0:
                block_type.remove(block)


def resolve_model_path(model_uid: str, asset_base: str) -> str | None:
    """Resolve model_uid to actual .glb file path on disk.

    Mapping rules:
      partnet_mobility/ID       -> asset_library/partnet_mobility/partnet_mobility/ID/whole.glb
      hssd-models/objects/X/H   -> asset_library/hssd-models/objects/X/H.glb
      3D-FUTURE-model/UUID      -> asset_library/3D-FUTURE-model/UUID.glb
      gr100/NAME                -> asset_library/gr100/NAME.glb
      objaverse_old/ID          -> asset_library/objaverse_old/ID.glb
      objaverse/ID              -> asset_library/objaverse/ID.glb
      gen_assets/TYPE/ID        -> asset_library/gen_assets/TYPE/ID.glb
    """
    if not model_uid:
        return None

    # Special case: partnet_mobility uses whole.glb inside a sub-directory
    if model_uid.startswith("partnet_mobility/"):
        glb = os.path.join(asset_base, "partnet_mobility", model_uid, "whole.glb")
        if os.path.exists(glb):
            return glb

    # Default: asset_library/{model_uid}.glb
    glb = os.path.join(asset_base, model_uid + ".glb")
    if os.path.exists(glb):
        return glb

    # Fallback: asset_library/{model_uid}/whole.glb
    glb = os.path.join(asset_base, model_uid, "whole.glb")
    if os.path.exists(glb):
        return glb

    return None


def _first_material_base_color(mat) -> tuple[float, float, float] | None:
    if mat is None:
        return None
    try:
        if getattr(mat, "use_nodes", False) and getattr(mat, "node_tree", None) is not None:
            for node in mat.node_tree.nodes:
                if getattr(node, "type", "") == "BSDF_PRINCIPLED":
                    color = node.inputs["Base Color"].default_value
                    return (float(color[0]), float(color[1]), float(color[2]))
    except Exception:
        pass
    try:
        color = mat.diffuse_color
        return (float(color[0]), float(color[1]), float(color[2]))
    except Exception:
        return None


def extract_material_rgb_from_objects(
    object_names: set[str],
    default_rgb: tuple[float, float, float] | None = (0.5, 0.5, 0.5),
) -> tuple[float, float, float] | None:
    colors: list[tuple[float, float, float]] = []
    for obj_name in object_names:
        obj = bpy.data.objects.get(obj_name)
        if obj is None:
            continue
        if obj.type == "MESH":
            for mat in getattr(obj.data, "materials", []) or []:
                color = _first_material_base_color(mat)
                if color is not None:
                    colors.append(color)
                    break
        if not colors:
            try:
                color = obj.color
                colors.append((float(color[0]), float(color[1]), float(color[2])))
            except Exception:
                pass
    if not colors:
        if default_rgb is None:
            return None
        return (
            float(default_rgb[0]),
            float(default_rgb[1]),
            float(default_rgb[2]),
        )
    arr = np.array(colors, dtype=np.float32)
    mean = np.mean(arr, axis=0)
    return (float(mean[0]), float(mean[1]), float(mean[2]))


def load_scene(scene_dir: str, layout: list[dict], asset_base: str = None,
               object_index_map: dict[int, dict] | None = None):
    """Load 3D scene from StructureMesh and place objects according to layout.

    Args:
        scene_dir: Path to the filtered scene directory (with layout.json + StructureMesh symlink).
        layout: Parsed layout.json list of objects.
        asset_base: Path to asset_library/ root directory.
    """
    scene_path = Path(scene_dir)

    # Load structure mesh (walls, floor — skip ceiling to allow lighting)
    mesh_dir = scene_path / "StructureMesh"
    if mesh_dir.exists():
        for glb_name in ["wall.glb", "floor.glb"]:
            glb_path = mesh_dir / glb_name
            if glb_path.exists():
                bpy.ops.import_scene.gltf(filepath=str(glb_path))

    loaded_count = 0
    fallback_count = 0

    # Place objects according to layout.json
    for obj_info in layout:
        bbox = obj_info.get("bbox", [])
        if len(bbox) < 9:
            continue

        model_uid = obj_info.get("model_uid", "")
        category = obj_info.get("category", "unknown")
        obj_id = int(obj_info.get("id", 0))
        pass_index = obj_id + 1  # avoid 0 (background)

        # bbox format: [pos_x, pos_y, pos_z, size_x, size_y, size_z, rot_x, rot_y, rot_z]
        pos = (bbox[0], bbox[1], bbox[2])
        size = (bbox[3], bbox[4], bbox[5])
        rot = (bbox[6], bbox[7], bbox[8])

        # Try to load the actual 3D model
        glb_path = resolve_model_path(model_uid, asset_base) if asset_base else None

        if glb_path:
            # Import actual model
            try:
                before = set(bpy.data.objects.keys())
                bpy.ops.import_scene.gltf(filepath=glb_path)
                after = set(bpy.data.objects.keys())
                new_objs = after - before
            except Exception:
                new_objs = set()

            if new_objs:
                material_rgb = extract_material_rgb_from_objects(new_objs)
                # Parent all imported objects to an empty and transform
                parent = bpy.data.objects.new(f"{category}_{obj_info.get('id', 0)}", None)
                bpy.context.scene.collection.objects.link(parent)
                parent.location = pos
                parent.rotation_euler = rot

                for obj_name in new_objs:
                    obj = bpy.data.objects[obj_name]
                    obj.parent = parent
                    if obj.type == "MESH":
                        obj.pass_index = pass_index

                parent["category"] = category
                parent["model_uid"] = model_uid
                parent["obj_id"] = obj_id
                parent["material_color_rgb"] = material_rgb
                parent["material_color_name"] = quantize_material_color_name(material_rgb)
                obj_info["material_color_rgb"] = [round(v, 4) for v in material_rgb]
                obj_info["material_color_name"] = quantize_material_color_name(material_rgb)
                loaded_count += 1
                if object_index_map is not None:
                    object_index_map[pass_index] = {
                        "obj_id": obj_id,
                        "category": category,
                        "model_uid": model_uid,
                        "material_color_rgb": [round(v, 4) for v in material_rgb],
                        "material_color_name": quantize_material_color_name(material_rgb),
                    }
                continue

        # Fallback: create placeholder cube with correct dimensions and a basic material
        bpy.ops.mesh.primitive_cube_add(size=1, location=pos, rotation=rot)
        obj = bpy.context.active_object
        obj.scale = (size[0] / 2, size[1] / 2, size[2] / 2)
        obj.name = f"{category}_{obj_info.get('id', 0)}"
        obj["category"] = category
        obj["model_uid"] = model_uid
        obj["obj_id"] = obj_id
        obj.pass_index = pass_index
        fallback_rgb = (0.5, 0.5, 0.5)
        obj["material_color_rgb"] = fallback_rgb
        obj["material_color_name"] = quantize_material_color_name(fallback_rgb)
        obj_info["material_color_rgb"] = [round(v, 4) for v in fallback_rgb]
        obj_info["material_color_name"] = quantize_material_color_name(fallback_rgb)
        fallback_count += 1
        if object_index_map is not None:
            object_index_map[pass_index] = {
                "obj_id": obj_id,
                "category": category,
                "model_uid": model_uid,
                "material_color_rgb": [round(v, 4) for v in fallback_rgb],
                "material_color_name": quantize_material_color_name(fallback_rgb),
            }

    print(f"  Loaded {loaded_count} models, {fallback_count} fallback cubes")


def _parse_compose_index(name: str) -> int | None:
    """Parse leading numeric index from composed-GLB object names."""
    base = name.split(".")[0]
    if "_" not in base:
        return None
    prefix = base.split("_", 1)[0]
    if prefix.isdigit():
        return int(prefix)
    return None


def resolve_composed_glb(scene_id: str, composed_dir: str | Path | None) -> Path | None:
    if composed_dir is None:
        return None
    path = Path(composed_dir) / f"{scene_id}.glb"
    if not path.exists():
        raise FileNotFoundError(f"missing required pre-composed GLB: {path}")
    return path


def assign_pass_index_from_compose(layout: list[dict],
                                   object_index_map: dict[int, dict] | None = None) -> int:
    """Assign pass_index / obj_id to Blender objects imported from composed GLB."""
    index_to_info = {i: obj for i, obj in enumerate(layout)}
    index_to_object_names: dict[int, set[str]] = {}
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        idx = _parse_compose_index(obj.name)
        if idx is None:
            continue
        index_to_object_names.setdefault(idx, set()).add(obj.name)
    assigned = 0
    for idx, object_names in index_to_object_names.items():
        info = index_to_info.get(idx)
        if not info:
            continue
        obj_id = int(info.get("id", 0))
        category = info.get("category", "unknown")
        model_uid = info.get("model_uid", "")
        pass_index = obj_id + 1
        material_rgb = extract_material_rgb_from_objects(object_names, default_rgb=None)
        material_color_rgb = (
            [round(float(v), 4) for v in material_rgb]
            if material_rgb is not None
            else None
        )
        material_color_name = quantize_material_color_name(material_color_rgb)
        info["material_color_rgb"] = material_color_rgb
        info["material_color_name"] = material_color_name
        for obj_name in object_names:
            obj = bpy.data.objects.get(obj_name)
            if obj is None:
                continue
            obj["obj_id"] = obj_id
            obj["category"] = category
            obj["model_uid"] = model_uid
            if material_color_rgb is not None:
                obj["material_color_rgb"] = material_color_rgb
            obj["material_color_name"] = material_color_name
            obj.pass_index = pass_index
        if object_index_map is not None:
            object_index_map[pass_index] = {
                "obj_id": obj_id,
                "category": category,
                "model_uid": model_uid,
                "material_color_rgb": material_color_rgb,
                "material_color_name": material_color_name,
            }
        assigned += 1
    if assigned == 0:
        print("  WARNING: composed GLB imported but no objects were indexed")
    return assigned


def get_room_bounds_from_scene() -> tuple[float, float, float, float]:
    """Get room XY bounds from imported wall/floor mesh objects.

    Must be called after load_scene() so that wall.glb is already imported.
    Only uses structure meshes (wall/floor), excluding furniture objects
    which may have extreme local coordinates in their GLB files.
    Returns (min_x, min_y, max_x, max_y).
    """
    all_xs, all_ys = [], []
    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        # Skip furniture: loaded models are children of empties with obj_id,
        # fallback cubes have obj_id directly
        if "obj_id" in obj or (obj.parent and "obj_id" in obj.parent):
            continue
        for v in obj.data.vertices:
            world_co = obj.matrix_world @ v.co
            all_xs.append(world_co.x)
            all_ys.append(world_co.y)

    if not all_xs:
        return (-5.0, -5.0, 5.0, 5.0)
    return (min(all_xs), min(all_ys), max(all_xs), max(all_ys))


def _resolve_optional_config_path(path_value: str | None, config_path: str | None = None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute() and config_path:
        path = (Path(config_path).parent / path).resolve()
    else:
        path = path.expanduser().resolve()
    return path


def load_category_aliases(target_cfg: dict, config_path: str | None = None) -> dict[str, dict]:
    alias_path = _resolve_optional_config_path(target_cfg.get("category_aliases_file"), config_path)
    if alias_path is None or not alias_path.exists():
        return {}
    try:
        with open(alias_path) as f:
            raw = json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}
    aliases: dict[str, dict] = {}
    for key, value in raw.items():
        key_norm = str(key).strip().lower()
        canonical = str(value.get("canonical_category", key_norm)).strip().lower()
        group_id = str(value.get("semantic_group_id", canonical)).strip().lower()
        aliases[key_norm] = {
            "canonical_category": canonical,
            "semantic_group_id": group_id,
            "display_name": value.get("display_name", canonical),
        }
    return aliases


def canonicalize_category(category: str, alias_map: dict[str, dict]) -> dict[str, str]:
    raw = str(category or "unknown").strip().lower()
    meta = alias_map.get(raw, {})
    canonical = str(meta.get("canonical_category", raw or "unknown")).strip().lower()
    semantic_group = str(meta.get("semantic_group_id", canonical)).strip().lower()
    return {
        "raw_category": raw or "unknown",
        "canonical_category": canonical or "unknown",
        "semantic_group_id": semantic_group or canonical or "unknown",
        "display_name": str(meta.get("display_name", canonical or raw or "unknown")),
    }


def quantize_material_color_name(rgb: tuple[float, float, float] | list[float] | None) -> str:
    if not rgb or len(rgb) < 3:
        return "unknown"
    r = max(0.0, min(float(rgb[0]), 1.0))
    g = max(0.0, min(float(rgb[1]), 1.0))
    b = max(0.0, min(float(rgb[2]), 1.0))
    mx = max(r, g, b)
    mn = min(r, g, b)
    if mx < 0.12:
        return "black"
    if mx - mn < 0.08:
        if mx > 0.82:
            return "white"
        if mx > 0.55:
            return "gray"
        return "brown" if r > g + 0.04 else "gray"
    if r > 0.35 and g > 0.18 and b < 0.28 and r > g and g > b:
        return "brown"
    if r >= g and r >= b:
        return "red" if g < 0.55 else "yellow"
    if g >= r and g >= b:
        return "green"
    return "blue"


def _blender_image_to_array(image) -> np.ndarray:
    width, height = int(image.size[0]), int(image.size[1])
    if width <= 0 or height <= 0:
        raise ValueError("invalid Blender image size")
    pixels = np.asarray(image.pixels[:], dtype=np.float32)
    denom = max(width * height, 1)
    channels = int(pixels.size // denom)
    if channels <= 0 or pixels.size != width * height * channels:
        raise ValueError("invalid Blender image pixel buffer")
    return pixels.reshape(height, width, channels).copy()


def _load_image_array_via_blender(path: Path | str) -> np.ndarray:
    if not IN_BLENDER:
        raise RuntimeError("Blender image loading requires Blender runtime")
    image = bpy.data.images.load(str(path), check_existing=False)
    try:
        return _blender_image_to_array(image)
    finally:
        bpy.data.images.remove(image)


def load_render_rgb_array(path: Path | str) -> np.ndarray:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"missing RGB render: {image_path}")
    if Image is not None:
        return np.asarray(Image.open(image_path).convert("RGB"))
    rgb = _load_image_array_via_blender(image_path)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"RGB render has invalid shape: {rgb.shape}")
    return np.rint(np.clip(rgb[..., :3], 0.0, 1.0) * 255.0).astype(np.uint8)


def load_render_object_index_array(path: Path | str, *, max_pass_id: int | None) -> np.ndarray:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"missing object-index render: {image_path}")
    if Image is not None:
        raw = np.asarray(Image.open(image_path))
    else:
        raw = _load_image_array_via_blender(image_path)
        if raw.ndim == 3:
            raw = raw[..., 0]
        raw = np.rint(np.clip(raw.astype(np.float32), 0.0, 1.0) * OBJECT_INDEX_PNG_MAX_VALUE).astype(np.uint16)
    return decode_object_index_array(raw, max_pass_id=max_pass_id)


def load_render_depth_map(path: Path | str) -> np.ndarray:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"missing depth render: {image_path}")
    if Image is not None:
        raw = np.asarray(Image.open(image_path)).astype(np.float32)
    else:
        raw = _load_image_array_via_blender(image_path).astype(np.float32)
    if raw.ndim == 3:
        raw = raw[..., 0]
    if raw.size == 0:
        raise ValueError(f"depth render is empty: {image_path}")
    if float(raw.max()) <= 1.0:
        return raw * 20.0
    return raw / 65535.0 * 20.0


def sparsify_gt_path_keypoints(
    path_world: list[list[float]] | list[tuple[float, float, float]],
    turn_angle_deg: float = 20.0,
    min_segment_m: float = 0.6,
) -> list[list[float]]:
    if not path_world:
        return []
    if len(path_world) <= 2:
        return [[round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3)] for p in path_world]

    keep = [0]
    angle_thresh = math.radians(float(turn_angle_deg))
    last_keep = 0
    for idx in range(1, len(path_world) - 1):
        p0 = np.array(path_world[idx - 1][:2], dtype=np.float32)
        p1 = np.array(path_world[idx][:2], dtype=np.float32)
        p2 = np.array(path_world[idx + 1][:2], dtype=np.float32)
        v1 = p1 - p0
        v2 = p2 - p1
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cosang = max(-1.0, min(1.0, float(np.dot(v1, v2) / (n1 * n2))))
        angle = math.acos(cosang)
        dist_from_last = math.hypot(
            float(path_world[idx][0]) - float(path_world[last_keep][0]),
            float(path_world[idx][1]) - float(path_world[last_keep][1]),
        )
        if angle >= angle_thresh or dist_from_last >= float(min_segment_m):
            keep.append(idx)
            last_keep = idx
    if keep[-1] != len(path_world) - 1:
        keep.append(len(path_world) - 1)
    out = []
    for idx in keep:
        p = path_world[idx]
        out.append([round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3)])
    return out


def rects_to_mask(rects: list[tuple[int, int, int, int]], width: int, height: int) -> np.ndarray:
    mask = np.zeros((max(0, int(height)), max(0, int(width))), dtype=bool)
    for x0, y0, x1, y1 in rects:
        xx0 = max(0, min(int(x0), int(x1)))
        xx1 = min(mask.shape[1], max(int(x0), int(x1)) + 1)
        yy0 = max(0, min(int(y0), int(y1)))
        yy1 = min(mask.shape[0], max(int(y0), int(y1)) + 1)
        if xx0 < xx1 and yy0 < yy1:
            mask[yy0:yy1, xx0:xx1] = True
    return mask


def segment_hits_mask_2d(
    p0: list[int] | tuple[int, int],
    p1: list[int] | tuple[int, int],
    obstacle_mask: np.ndarray | None,
    pad_px: int = 2,
) -> bool:
    if obstacle_mask is None:
        return False
    h, w = obstacle_mask.shape[:2]
    x0, y0 = int(p0[0]), int(p0[1])
    x1, y1 = int(p1[0]), int(p1[1])
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for step in range(steps + 1):
        t = step / steps
        x = int(round(x0 + t * (x1 - x0)))
        y = int(round(y0 + t * (y1 - y0)))
        xx0 = max(0, x - int(pad_px))
        xx1 = min(w, x + int(pad_px) + 1)
        yy0 = max(0, y - int(pad_px))
        yy1 = min(h, y + int(pad_px) + 1)
        if xx0 >= xx1 or yy0 >= yy1:
            continue
        if bool(obstacle_mask[yy0:yy1, xx0:xx1].any()):
            return True
    return False


def projected_polyline_collision_free(
    projected_points: list[list[int] | None],
    obstacle_mask: np.ndarray | None,
    pad_px: int = 2,
    require_all_points_visible: bool = True,
) -> bool:
    if not projected_points:
        return False
    if require_all_points_visible and any(pt is None for pt in projected_points):
        return False
    pts = [pt for pt in projected_points if pt is not None]
    if len(pts) < 2:
        return False
    for p0, p1 in zip(pts, pts[1:]):
        if segment_hits_mask_2d(p0, p1, obstacle_mask, pad_px=pad_px):
            return False
    return True


def first_blocking_segment_index(
    projected_points: list[list[int] | None],
    obstacle_mask: np.ndarray | None,
    pad_px: int = 2,
    require_all_points_visible: bool = True,
) -> int | None:
    if not projected_points:
        return None
    if require_all_points_visible and any(pt is None for pt in projected_points):
        return -1
    pts = [pt for pt in projected_points if pt is not None]
    if len(pts) < 2:
        return -1
    for idx, (p0, p1) in enumerate(zip(pts, pts[1:])):
        if segment_hits_mask_2d(p0, p1, obstacle_mask, pad_px=pad_px):
            return int(idx)
    return None


def evaluate_dense_path_projection(
    projected_points: list[list[int] | None],
    obstacle_mask: np.ndarray | None,
    pad_px: int = 2,
) -> tuple[bool, int | None]:
    if not projected_points:
        return False, None
    if obstacle_mask is None:
        return True, None
    return (
        projected_polyline_collision_free(
            projected_points,
            obstacle_mask=obstacle_mask,
            pad_px=pad_px,
            require_all_points_visible=False,
        ),
        first_blocking_segment_index(
            projected_points,
            obstacle_mask=obstacle_mask,
            pad_px=pad_px,
            require_all_points_visible=False,
        ),
    )


def ensure_start_anchor_projection(
    projected_points: list[list[int] | None],
    start_anchor_image_xy: list[int] | tuple[int, int] | None,
) -> list[list[int] | None]:
    if not projected_points:
        return []
    if start_anchor_image_xy is None:
        return list(projected_points)
    anchored = []
    anchor_xy = [int(start_anchor_image_xy[0]), int(start_anchor_image_xy[1])]
    first_visible_idx = None
    for idx, pt in enumerate(projected_points):
        if pt is not None:
            first_visible_idx = idx
            break
    if first_visible_idx is None:
        return [list(anchor_xy)] + [None] * (len(projected_points) - 1)
    for idx, pt in enumerate(projected_points):
        if idx < first_visible_idx and pt is None:
            anchored.append(list(anchor_xy))
        else:
            anchored.append(None if pt is None else [int(pt[0]), int(pt[1])])
    return anchored


def include_fixed_start_candidate(
    candidates: list[dict],
    fixed_start_candidate: dict | None,
) -> list[dict]:
    enriched = [dict(cand) for cand in (candidates or [])]
    if fixed_start_candidate is None:
        return enriched
    existing_idx = next((
        idx for idx, cand in enumerate(enriched)
        if str(cand.get("candidate_source", "")) == "fixed_bottom_center_start"
    ), None)
    fixed_copy = dict(fixed_start_candidate)
    if existing_idx is not None:
        enriched[existing_idx] = fixed_copy
    else:
        next_waypoint_id = max(
            [int(c.get("waypoint_id", -1)) for c in enriched] + [-1]
        ) + 1
        fixed_copy["waypoint_id"] = int(next_waypoint_id)
        enriched.append(fixed_copy)
    enriched.sort(key=lambda c: (float(c.get("depth", 1e9)), int(c.get("waypoint_id", -1))))
    for idx, cand in enumerate(enriched, start=1):
        cand["display_id"] = int(idx)
    return enriched


def refine_sparse_path_indices_for_projection(
    projected_points: list[list[int] | None],
    keep_indices: list[int],
    obstacle_mask: np.ndarray | None,
    pad_px: int = 2,
    path_world: list[list[float]] | list[tuple[float, float, float]] | None = None,
    nav_grid=None,
) -> list[int]:
    if len(keep_indices) <= 1:
        return list(keep_indices)
    point_count = len(projected_points)
    if point_count <= 0:
        return []
    if any(int(idx) < 0 or int(idx) >= point_count for idx in keep_indices):
        return []

    def _world_segment_traversable(start_idx: int, end_idx: int) -> bool:
        if path_world is None or nav_grid is None or not hasattr(nav_grid, "is_path_traversable"):
            return True
        if int(start_idx) >= len(path_world) or int(end_idx) >= len(path_world):
            return False
        p0 = path_world[int(start_idx)]
        p1 = path_world[int(end_idx)]
        result = nav_grid.is_path_traversable([
            (float(p0[0]), float(p0[1])),
            (float(p1[0]), float(p1[1])),
        ])
        return bool(result.get("traversable", False))

    refined = [int(keep_indices[0])]
    for seg_start, seg_end in zip(keep_indices, keep_indices[1:]):
        current = int(seg_start)
        end = int(seg_end)
        while current < end:
            p0 = projected_points[current]
            if p0 is None:
                return []
            chosen = None
            for candidate in range(end, current, -1):
                p1 = projected_points[candidate]
                if p1 is None:
                    continue
                if obstacle_mask is not None and segment_hits_mask_2d(p0, p1, obstacle_mask, pad_px=pad_px):
                    continue
                if not _world_segment_traversable(current, candidate):
                    continue
                chosen = candidate
                break
            if chosen is None:
                return []
            if refined[-1] != int(chosen):
                refined.append(int(chosen))
            current = int(chosen)
    refined_points = [projected_points[idx] for idx in refined]
    if obstacle_mask is not None:
        if not projected_polyline_collision_free(
            refined_points,
            obstacle_mask=obstacle_mask,
            pad_px=pad_px,
            require_all_points_visible=True,
        ):
            return []
    for start_idx, end_idx in zip(refined, refined[1:]):
        if not _world_segment_traversable(start_idx, end_idx):
            return []
    return refined


def collect_candidate_path_projection(
    path_ids: list[int],
    candidate_lookup: dict[int, dict],
) -> tuple[list[int], list[list[int]]]:
    ordered_ids: list[int] = []
    projected: list[list[int]] = []
    for wp_id in path_ids:
        rec = candidate_lookup.get(int(wp_id))
        if rec is None:
            continue
        xy = rec.get("image_xy")
        if not isinstance(xy, list) or len(xy) != 2:
            continue
        if ordered_ids and ordered_ids[-1] == int(wp_id):
            continue
        ordered_ids.append(int(wp_id))
        projected.append([int(xy[0]), int(xy[1])])
    return ordered_ids, projected


def normalize_candidate_path_ids_for_projection(
    path_ids: list[int],
    candidate_lookup: dict[int, dict],
    obstacle_mask: np.ndarray | None,
    pad_px: int = 2,
) -> list[int]:
    ordered_ids, projected = collect_candidate_path_projection(path_ids, candidate_lookup)
    if len(ordered_ids) < 2:
        return []
    if obstacle_mask is None:
        return ordered_ids
    if not projected_polyline_collision_free(
        projected,
        obstacle_mask=obstacle_mask,
        pad_px=pad_px,
        require_all_points_visible=True,
    ):
        return []
    return ordered_ids


def sparsify_candidate_path_ids_for_projection(
    path_ids: list[int],
    candidate_lookup: dict[int, dict],
    obstacle_mask: np.ndarray | None,
    pad_px: int = 2,
) -> list[int]:
    ordered_ids, projected = collect_candidate_path_projection(path_ids, candidate_lookup)
    if len(ordered_ids) < 2:
        return []
    if obstacle_mask is None:
        return ordered_ids
    if len(ordered_ids) <= 2:
        return ordered_ids if projected_polyline_collision_free(
            projected,
            obstacle_mask=obstacle_mask,
            pad_px=pad_px,
            require_all_points_visible=True,
        ) else []
    keep_indices = refine_sparse_path_indices_for_projection(
        projected,
        keep_indices=[0, len(projected) - 1],
        obstacle_mask=obstacle_mask,
        pad_px=pad_px,
    )
    if not keep_indices:
        return []
    return [ordered_ids[idx] for idx in keep_indices]


def path_ids_projection_collision_free(
    path_ids: list[int],
    candidate_lookup: dict[int, dict],
    obstacle_mask: np.ndarray | None,
    pad_px: int = 2,
) -> bool:
    ordered_ids, pts = collect_candidate_path_projection(path_ids, candidate_lookup)
    if len(ordered_ids) < 2:
        return False
    if obstacle_mask is None:
        return True
    return projected_polyline_collision_free(
        pts,
        obstacle_mask=obstacle_mask,
        pad_px=pad_px,
        require_all_points_visible=True,
    )


def path_ids_world_collision_free(
    path_ids: list[int],
    candidate_lookup: dict[int, dict],
    nav_grid,
) -> bool:
    if not path_ids or len(path_ids) < 2:
        return False
    if nav_grid is None or not hasattr(nav_grid, "is_path_traversable"):
        return True

    world_pts: list[tuple[float, float]] = []
    seen: set[tuple[int, float, float]] = set()
    for wp_id in path_ids:
        rec = candidate_lookup.get(int(wp_id))
        if rec is None:
            continue
        world_xyz = rec.get("world_xyz")
        if not isinstance(world_xyz, list) or len(world_xyz) < 2:
            continue
        key = (int(wp_id), float(world_xyz[0]), float(world_xyz[1]))
        if key in seen:
            continue
        seen.add(key)
        world_pts.append((float(world_xyz[0]), float(world_xyz[1])))
    if len(world_pts) < 2:
        return False
    for p0, p1 in zip(world_pts, world_pts[1:]):
        result = nav_grid.is_path_traversable([p0, p1])
        if not bool(result.get("traversable", False)):
            return False
    return True


DEFAULT_PROJECTION_NON_BLOCKING_CATEGORIES = {
    "wall",
    "ceiling",
    "window",
    "door",
    "floor",
    "carpet",
    "rug",
    "mat",
    "blanket",
    "towel",
}


def build_blocking_object_mask(
    object_index_img,
    object_index_map: dict[int, dict] | None,
    excluded_obj_id: int | None = None,
    non_blocking_categories: set[str] | None = None,
):
    if object_index_img is None:
        return None
    if not object_index_map:
        return None
    category_allowlist = {
        str(cat).strip().lower() for cat in (
            non_blocking_categories or DEFAULT_PROJECTION_NON_BLOCKING_CATEGORIES
        )
    }
    blocking_pass_ids: list[int] = []
    for pass_id, meta in object_index_map.items():
        try:
            pixel_id = int(pass_id)
        except (TypeError, ValueError):
            continue
        obj_id = meta.get("obj_id")
        if excluded_obj_id is not None and obj_id is not None and int(obj_id) == int(excluded_obj_id):
            continue
        category = str(meta.get("category", "")).strip().lower()
        if category in category_allowlist:
            continue
        blocking_pass_ids.append(pixel_id)
    if not blocking_pass_ids:
        return None
    import numpy as _np
    return _np.isin(object_index_img, _np.asarray(blocking_pass_ids, dtype=_np.int32))


def collect_nonblocking_object_pass_ids(
    object_index_map: dict[int, dict] | None,
    non_blocking_categories: set[str] | None = None,
) -> set[int]:
    if not object_index_map:
        return set()
    category_allowlist = {
        str(cat).strip().lower() for cat in (
            non_blocking_categories or DEFAULT_PROJECTION_NON_BLOCKING_CATEGORIES
        )
    }
    pass_ids: set[int] = set()
    for pass_id, meta in object_index_map.items():
        try:
            pixel_id = int(pass_id)
        except (TypeError, ValueError):
            continue
        category = str(meta.get("category", "")).strip().lower()
        if category in category_allowlist:
            pass_ids.add(pixel_id)
    return pass_ids


def extract_visible_color_rgb(
    rgb_image,
    object_index_img,
    *,
    pass_index: int,
    min_pixels: int = 12,
) -> list[float] | None:
    if rgb_image is None or object_index_img is None:
        return None
    try:
        import numpy as _np

        rgb_arr = _np.asarray(rgb_image)
        if rgb_arr.ndim != 3 or rgb_arr.shape[2] < 3:
            return None
        mask = _np.asarray(object_index_img) == int(pass_index)
        if mask.ndim != 2 or not bool(mask.any()):
            return None
        pixels = rgb_arr[mask][:, :3]
        if int(pixels.shape[0]) < int(min_pixels):
            return None
        median = _np.median(pixels.astype(_np.float32), axis=0) / 255.0
        return [round(float(v), 4) for v in median.tolist()]
    except Exception:
        return None


def assign_visible_target_colors(
    targets: list[dict],
    *,
    rgb_image,
    object_index_img,
    min_pixels: int = 12,
) -> None:
    for target in targets or []:
        pass_index = int(target.get("pass_index", -1))
        visible_rgb = extract_visible_color_rgb(
            rgb_image,
            object_index_img,
            pass_index=pass_index,
            min_pixels=min_pixels,
        )
        target["visible_color_rgb"] = visible_rgb
        target["visible_color_name"] = quantize_material_color_name(visible_rgb)


def project_path_world_to_image(
    path_world: list[list[float]] | list[tuple[float, float, float]],
    cam_obj,
    vfov_half: float,
    hfov_half: float,
    res_x: int,
    res_y: int,
    scene=None,
) -> list[list[int] | None]:
    if not path_world:
        return []
    cam_inv = cam_obj.matrix_world.inverted()
    scene = scene or bpy.context.scene
    projected: list[list[int] | None] = []
    for pt in path_world:
        proj = project_world_point(
            cam_inv,
            mathutils.Vector((float(pt[0]), float(pt[1]), float(pt[2]))),
            vfov_half,
            hfov_half,
            res_x,
            res_y,
            cam_obj=cam_obj,
            scene=scene,
        )
        if proj is None:
            projected.append(None)
            continue
        px, py = proj["pixel_xy"]
        projected.append([int(px), int(py)])
    return projected


def _match_sparse_points_to_dense_indices(
    dense_path_world: list[list[float]] | list[tuple[float, float, float]],
    sparse_path_world: list[list[float]] | list[tuple[float, float, float]],
) -> list[int]:
    if not dense_path_world:
        return []
    if not sparse_path_world:
        return []

    matched: list[int] = []
    search_start = 0
    for sparse_pt in sparse_path_world:
        best_idx = None
        best_dist = float("inf")
        sx, sy, sz = float(sparse_pt[0]), float(sparse_pt[1]), float(sparse_pt[2])
        for idx in range(search_start, len(dense_path_world)):
            dense_pt = dense_path_world[idx]
            dx = float(dense_pt[0]) - sx
            dy = float(dense_pt[1]) - sy
            dz = float(dense_pt[2]) - sz
            dist = dx * dx + dy * dy + dz * dz
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
            if dist <= 1e-8:
                break
        if best_idx is None:
            continue
        if not matched or matched[-1] != int(best_idx):
            matched.append(int(best_idx))
        search_start = int(best_idx)
    if not matched:
        return []
    if matched[0] != 0:
        matched.insert(0, 0)
    last_idx = len(dense_path_world) - 1
    if matched[-1] != last_idx:
        matched.append(last_idx)
    return matched


def refine_routing_outputs_for_projection(
    dense_path_world: list[list[float]] | list[tuple[float, float, float]],
    gt_path_keypoints_world: list[list[float]] | list[tuple[float, float, float]],
    path_wp_point: list[int],
    path_wp_embodied: list[int],
    classification_candidates: list[dict],
    obstacle_mask: np.ndarray | None,
    projected_dense_path: list[list[int] | None] | None = None,
    pad_px: int = 2,
    nav_grid=None,
) -> tuple[list[list[float]], list[int], list[int]]:
    if not dense_path_world:
        return [], list(path_wp_point), list(path_wp_embodied)

    refined_keypoints = [
        [float(pt[0]), float(pt[1]), float(pt[2])] for pt in gt_path_keypoints_world
    ] if gt_path_keypoints_world else [
        [float(dense_path_world[0][0]), float(dense_path_world[0][1]), float(dense_path_world[0][2])],
        [float(dense_path_world[-1][0]), float(dense_path_world[-1][1]), float(dense_path_world[-1][2])],
    ]
    if obstacle_mask is not None and projected_dense_path:
        keep_indices = _match_sparse_points_to_dense_indices(dense_path_world, refined_keypoints)
        if not keep_indices:
            return [], [], []
        keep_indices = refine_sparse_path_indices_for_projection(
            projected_dense_path,
            keep_indices=keep_indices,
            obstacle_mask=obstacle_mask,
            pad_px=pad_px,
            path_world=dense_path_world,
            nav_grid=nav_grid,
        )
        if not keep_indices:
            return [], [], []
        refined_keypoints = [
            [
                float(dense_path_world[idx][0]),
                float(dense_path_world[idx][1]),
                float(dense_path_world[idx][2]),
            ]
            for idx in keep_indices
        ]

    candidate_lookup = {
        int(c.get("waypoint_id", -1)): c for c in classification_candidates
        if int(c.get("waypoint_id", -1)) >= 0
    }
    refined_point_ids = sparsify_candidate_path_ids_for_projection(
        path_wp_point,
        candidate_lookup,
        obstacle_mask=obstacle_mask,
        pad_px=pad_px,
    )
    refined_embodied_ids = sparsify_candidate_path_ids_for_projection(
        path_wp_embodied,
        candidate_lookup,
        obstacle_mask=obstacle_mask,
        pad_px=pad_px,
    )
    if obstacle_mask is not None:
        if (path_wp_point and not refined_point_ids) or (path_wp_embodied and not refined_embodied_ids):
            return [], [], []
    return refined_keypoints, refined_point_ids, refined_embodied_ids


def load_target_config(config: dict, config_path: str | None = None) -> tuple[dict, set[str], set[str], dict[str, dict]]:
    """Load target selection config and recommended categories."""
    target_cfg = config.get("targets", {})
    rec_categories: set[str] = set()
    large_categories: set[str] = set()

    rec_file = target_cfg.get("recommended_categories_file")
    if rec_file:
        rec_path = Path(rec_file)
        if not rec_path.is_absolute() and config_path:
            rec_path = (Path(config_path).parent / rec_path).resolve()
        else:
            rec_path = rec_path.expanduser().resolve()
        if rec_path.exists():
            try:
                with open(rec_path) as f:
                    data = json.load(f)
                rec_categories = set(data.get("recommended_categories", []))
                large_categories = set(data.get("large_categories", []))
            except (json.JSONDecodeError, IOError):
                rec_categories = set()
                large_categories = set()
    alias_map = load_category_aliases(target_cfg, config_path)
    return target_cfg, rec_categories, large_categories, alias_map


def _fov_params(fov_deg: float, res_x: int, res_y: int) -> tuple[float, float]:
    vfov_half = math.radians(fov_deg) / 2
    aspect = res_x / res_y
    hfov_half = math.atan(math.tan(vfov_half) * aspect)
    return vfov_half, hfov_half


def project_world_point(cam_inv, world_pt, vfov_half: float, hfov_half: float,
                        res_x: int, res_y: int,
                        cam_obj=None,
                        scene=None,
                        edge_tolerance_ndc: float = 1e-4):
    if not IN_BLENDER or world_to_camera_view is None:
        raise RuntimeError("project_world_point requires Blender native world_to_camera_view")
    scene = scene or bpy.context.scene
    cam_obj = cam_obj or scene.camera
    if cam_obj is None:
        return None
    ndc = world_to_camera_view(scene, cam_obj, world_pt)
    if ndc.z <= 0.0:
        return None
    ndc_x = float(ndc.x)
    ndc_y = float(ndc.y)
    tol = max(0.0, float(edge_tolerance_ndc))
    if ndc_x < -tol or ndc_x > 1.0 + tol or ndc_y < -tol or ndc_y > 1.0 + tol:
        return None
    ndc_x = min(1.0, max(0.0, ndc_x))
    ndc_y = min(1.0, max(0.0, ndc_y))
    depth = -((cam_obj.matrix_world.inverted() @ world_pt).z)
    if depth < 0.05:
        return None
    px = int(round(ndc_x * float(res_x)))
    py = int(round((1.0 - ndc_y) * float(res_y)))
    px = max(0, min(int(res_x) - 1, px))
    py = max(0, min(int(res_y) - 1, py))
    screen_x = ndc_x * 2.0 - 1.0
    screen_y = ndc_y * 2.0 - 1.0
    return {
        "depth": float(depth),
        "screen_xy": (float(screen_x), float(screen_y)),
        "pixel_xy": (px, py),
    }


def raycast_visible(scene, depsgraph, origin, target, tol: float = 0.05) -> bool:
    direction = target - origin
    dist = direction.length
    if dist < 0.05:
        return True
    direction = direction.normalized()
    hit, loc, _n, _i, _o, _m = scene.ray_cast(
        depsgraph, origin, direction, distance=dist + tol
    )
    if not hit:
        return True
    return (loc - origin).length >= dist - tol


def sample_ground_z(scene, depsgraph, x: float, y: float, default_z: float = 0.0) -> float:
    origin_down = mathutils.Vector((float(x), float(y), 6.0))
    dir_down = mathutils.Vector((0.0, 0.0, -1.0))
    hit, loc, _n, _i, _o, _m = scene.ray_cast(depsgraph, origin_down, dir_down, distance=20.0)
    if hit:
        return float(loc.z) + 0.02
    origin_up = mathutils.Vector((float(x), float(y), -3.0))
    dir_up = mathutils.Vector((0.0, 0.0, 1.0))
    hit2, loc2, _n2, _i2, _o2, _m2 = scene.ray_cast(depsgraph, origin_up, dir_up, distance=20.0)
    if hit2:
        return float(loc2.z) + 0.02
    return float(default_z)


def estimate_floor_z(scene, depsgraph, grid_point, floor_mask, stride: int) -> float:
    samples = []
    sample_stride = max(int(stride) * 4, 1)
    for gy in range(0, int(grid_point.height), sample_stride):
        for gx in range(0, int(grid_point.width), sample_stride):
            if floor_mask is not None and not bool(floor_mask[gy, gx]):
                continue
            if not bool(grid_point.is_free(gx, gy)):
                continue
            wx, wy = grid_point.grid_to_world(gx, gy)
            z = sample_ground_z(scene, depsgraph, float(wx), float(wy), default_z=0.0)
            if -5.0 < z < 5.0:
                samples.append(float(z))
    if not samples:
        return 0.0
    samples.sort()
    q = samples[:max(1, int(len(samples) * 0.35))]
    mid = len(q) // 2
    if len(q) % 2 == 1:
        return float(q[mid])
    return float((q[mid - 1] + q[mid]) * 0.5)


def compute_bottom_center_floor_anchor(cam_obj, scene, floor_z: float) -> list[float] | None:
    if not IN_BLENDER or cam_obj is None or scene is None:
        return None
    cam_data = getattr(cam_obj, "data", None)
    if cam_data is None or not hasattr(cam_data, "view_frame"):
        return None
    frame = list(cam_data.view_frame(scene=scene))
    if len(frame) < 4:
        return None

    bottom_two = sorted(frame, key=lambda corner: (float(corner.y), float(corner.x)))[:2]
    if len(bottom_two) < 2:
        return None
    bottom_left, bottom_right = sorted(bottom_two, key=lambda corner: float(corner.x))
    bottom_center_local = (bottom_left + bottom_right) * 0.5

    origin = cam_obj.matrix_world.translation.copy()
    bottom_center_world = cam_obj.matrix_world @ bottom_center_local
    direction = bottom_center_world - origin
    if abs(float(direction.z)) <= 1e-8:
        return None
    scale = (float(floor_z) - float(origin.z)) / float(direction.z)
    if scale <= 0.0:
        return None
    hit = origin + direction * scale
    return [float(hit.x), float(hit.y), float(floor_z)]


def lookup_point_clearance_m(grid_point, wx: float, wy: float) -> float:
    if grid_point is None or not hasattr(grid_point, "get_distance_map"):
        return 0.0
    gx, gy = grid_point.world_to_grid(float(wx), float(wy))
    dist_map = grid_point.get_distance_map()
    if dist_map is None or gy < 0 or gx < 0 or gy >= dist_map.shape[0] or gx >= dist_map.shape[1]:
        return 0.0
    return float(dist_map[gy, gx])


def lookup_grid_distance_m(grid, wx: float, wy: float) -> float:
    if grid is None or not hasattr(grid, "get_distance_map"):
        return 0.0
    gx, gy = grid.world_to_grid(float(wx), float(wy))
    dist_map = grid.get_distance_map()
    if dist_map is None or gy < 0 or gx < 0 or gy >= dist_map.shape[0] or gx >= dist_map.shape[1]:
        return 0.0
    return float(dist_map[gy, gx])


def build_fixed_bottom_center_start_candidate(
    anchor_world_xyz: list[float] | tuple[float, float, float] | None,
    cam_obj,
    depsgraph,
    scene,
    grid_point,
    grid_embodied,
    vfov_half: float,
    hfov_half: float,
    res_x: int,
    res_y: int,
) -> dict | None:
    if anchor_world_xyz is None or cam_obj is None or scene is None:
        return None
    if grid_point is None or grid_embodied is None:
        return None

    wx, wy, wz = float(anchor_world_xyz[0]), float(anchor_world_xyz[1]), float(anchor_world_xyz[2])
    gx, gy = grid_embodied.world_to_grid(wx, wy)
    if not bool(grid_embodied.is_free(gx, gy)):
        return None
    pgx, pgy = grid_point.world_to_grid(wx, wy)
    if not bool(grid_point.is_free(pgx, pgy)):
        return None
    wz = sample_ground_z(scene, depsgraph, float(wx), float(wy), default_z=float(wz))

    world_pt = mathutils.Vector((wx, wy, wz))
    proj = project_world_point(
        cam_obj.matrix_world.inverted(),
        world_pt,
        vfov_half,
        hfov_half,
        res_x,
        res_y,
        cam_obj=cam_obj,
        scene=scene,
    )
    if proj is None:
        return None
    if not raycast_visible(scene, depsgraph, cam_obj.matrix_world.translation.copy(), world_pt, tol=0.08):
        return None

    px, py = proj["pixel_xy"]
    point_clearance_m = lookup_point_clearance_m(grid_point, wx, wy)
    embodied_margin_m = lookup_grid_distance_m(grid_embodied, wx, wy)
    return {
        "waypoint_id": -1,
        "display_id": 0,
        "world_xyz": [round(wx, 3), round(wy, 3), round(wz, 3)],
        "grid_xy": [int(gx), int(gy)],
        "image_xy": [int(px), int(py)],
        "screen_xy_norm": [round(float(proj["screen_xy"][0]), 6), round(float(proj["screen_xy"][1]), 6)],
        "depth": round(float(proj["depth"]), 4),
        "pointmass_walkable": True,
        "embodied_feasible": True,
        "point_clearance_m": round(point_clearance_m, 3),
        "embodied_clearance_margin_m": round(embodied_margin_m, 3),
        "raycast_visible": True,
        "candidate_source": "fixed_bottom_center_start",
    }


def ensure_sparse_routing_keypoint_candidates(
    candidates: list[dict],
    keypoints_world: list[list[float]] | list[tuple[float, float, float]],
    keypoints_image_xy: list[list[int] | None],
    *,
    feasible_key: str,
    grid_point,
    grid_embodied,
    cam_obj,
    scene,
    vfov_half: float,
    hfov_half: float,
    res_x: int,
    res_y: int,
    merge_world_dist_m: float = 0.12,
    merge_pixel_dist_px: float = 6.0,
) -> list[int]:
    if not keypoints_world:
        return []
    if len(keypoints_image_xy) < len(keypoints_world):
        raise ValueError("sparse routing keypoints are missing Blender image coordinates")

    next_waypoint_id = max([int(c.get("waypoint_id", -1)) for c in candidates] + [-1]) + 1
    ensured_ids: list[int] = []

    def _candidate_score(candidate: dict, wx: float, wy: float, image_xy: list[int]) -> tuple[float, float, int]:
        c_world = candidate.get("world_xyz", [0.0, 0.0, 0.0])
        c_image = candidate.get("image_xy", [0, 0])
        world_dist = math.hypot(float(c_world[0]) - wx, float(c_world[1]) - wy)
        pixel_dist = math.hypot(float(c_image[0]) - float(image_xy[0]), float(c_image[1]) - float(image_xy[1]))
        return (
            world_dist,
            pixel_dist,
            int(candidate.get("waypoint_id", -1)),
        )

    for keypoint, image_xy in zip(keypoints_world, keypoints_image_xy):
        if not isinstance(image_xy, list) or len(image_xy) != 2:
            raise ValueError("sparse routing keypoints require Blender gt_path_keypoints_image_xy")
        wx, wy, wz = float(keypoint[0]), float(keypoint[1]), float(keypoint[2])
        matched = None
        for candidate in candidates:
            if not bool(candidate.get(feasible_key, False)):
                continue
            if not isinstance(candidate.get("image_xy"), list) or len(candidate["image_xy"]) != 2:
                continue
            score = _candidate_score(candidate, wx, wy, image_xy)
            if score[0] <= merge_world_dist_m or score[1] <= merge_pixel_dist_px:
                if matched is None or score < matched[0]:
                    matched = (score, candidate)
        if matched is not None:
            ensured_ids.append(int(matched[1]["waypoint_id"]))
            continue

        gx, gy = grid_point.world_to_grid(wx, wy)
        point_ok = bool(grid_point.is_free(gx, gy))
        embodied_ok = bool(grid_embodied.is_free(gx, gy))
        if feasible_key == "pointmass_walkable":
            if not point_ok:
                raise ValueError("sparse routing keypoint is not pointmass traversable")
        elif feasible_key == "embodied_feasible":
            if not embodied_ok:
                raise ValueError("sparse routing keypoint is not embodied traversable")
        else:
            raise ValueError(f"unsupported sparse routing feasible_key={feasible_key}")

        proj = project_world_point(
            cam_obj.matrix_world.inverted(),
            mathutils.Vector((wx, wy, wz)),
            vfov_half,
            hfov_half,
            res_x,
            res_y,
            cam_obj=cam_obj,
            scene=scene,
        )
        if proj is None:
            raise ValueError("sparse routing keypoint is missing Blender-native projection")

        point_clearance_m = lookup_point_clearance_m(grid_point, wx, wy)
        embodied_margin_m = lookup_grid_distance_m(grid_embodied, wx, wy)
        candidates.append({
            "waypoint_id": int(next_waypoint_id),
            "display_id": 0,
            "world_xyz": [round(wx, 3), round(wy, 3), round(wz, 3)],
            "grid_xy": [int(gx), int(gy)],
            "image_xy": [int(image_xy[0]), int(image_xy[1])],
            "screen_xy_norm": [round(float(proj["screen_xy"][0]), 6), round(float(proj["screen_xy"][1]), 6)],
            "depth": round(float(proj["depth"]), 4),
            "pointmass_walkable": point_ok,
            "embodied_feasible": embodied_ok,
            "point_clearance_m": round(point_clearance_m, 3),
            "embodied_clearance_margin_m": round(embodied_margin_m, 3),
            "raycast_visible": True,
            "candidate_source": "routing_sparse_keypoint",
        })
        ensured_ids.append(int(next_waypoint_id))
        next_waypoint_id += 1

    candidates.sort(key=lambda c: (float(c.get("depth", 1e9)), int(c.get("waypoint_id", -1))))
    for idx, candidate in enumerate(candidates, start=1):
        candidate["display_id"] = int(idx)
    return ensured_ids


def build_classification_candidates_for_view(
    cam_obj,
    depsgraph,
    grid_point,
    grid_embodied,
    fov_deg: float,
    res_x: int,
    res_y: int,
    spacing_m: float = 0.4,
    max_candidates: int = 220,
) -> list[dict]:
    if grid_point is None or grid_embodied is None:
        return []

    cam_inv = cam_obj.matrix_world.inverted()
    cam_origin = cam_obj.matrix_world.translation.copy()
    vfov_half, hfov_half = _fov_params(fov_deg, res_x, res_y)

    stride = max(1, int(round(spacing_m / float(grid_point.resolution))))
    floor_mask = getattr(grid_point, "floor_mask", None)
    scene = bpy.context.scene
    floor_z_ref = estimate_floor_z(scene, depsgraph, grid_point, floor_mask, stride)

    candidates = []
    waypoint_id = 0
    for gy in range(0, grid_point.height, stride):
        for gx in range(0, grid_point.width, stride):
            if floor_mask is not None and not bool(floor_mask[gy, gx]):
                continue

            wx, wy = grid_point.grid_to_world(gx, gy)
            wz = float(floor_z_ref)
            world_pt = mathutils.Vector((float(wx), float(wy), float(wz)))
            proj = project_world_point(cam_inv, world_pt, vfov_half, hfov_half, res_x, res_y, cam_obj=cam_obj, scene=scene)
            if proj is None:
                continue

            px, py = proj["pixel_xy"]
            if px < 0 or py < 0 or px >= res_x or py >= res_y:
                continue

            visible = raycast_visible(scene, depsgraph, cam_origin, world_pt, tol=0.08)
            if not visible:
                continue

            point_clearance_m = lookup_point_clearance_m(grid_point, wx, wy)
            embodied_margin_m = lookup_grid_distance_m(grid_embodied, wx, wy)
            candidates.append({
                "waypoint_id": waypoint_id,
                "display_id": 0,
                "world_xyz": [round(float(wx), 3), round(float(wy), 3), round(float(wz), 3)],
                "grid_xy": [int(gx), int(gy)],
                "image_xy": [int(px), int(py)],
                "screen_xy_norm": [round(float(proj["screen_xy"][0]), 6), round(float(proj["screen_xy"][1]), 6)],
                "depth": round(float(proj["depth"]), 4),
                "pointmass_walkable": bool(grid_point.is_free(gx, gy)),
                "embodied_feasible": bool(grid_embodied.is_free(gx, gy)),
                "point_clearance_m": round(point_clearance_m, 3),
                "embodied_clearance_margin_m": round(embodied_margin_m, 3),
                "raycast_visible": True,
                "candidate_source": "blender_grid",
            })
            waypoint_id += 1

    candidates.sort(key=lambda c: c["depth"])
    if max_candidates > 0:
        candidates = candidates[:max_candidates]
    for idx, cand in enumerate(candidates, start=1):
        cand["display_id"] = idx
    return candidates


def build_world_routing_candidates(
    grid_point,
    grid_embodied,
    spacing_m: float = 0.1,
    max_candidates: int = 0,
    floor_z: float = 0.0,
) -> list[dict]:
    if grid_point is None or grid_embodied is None:
        return []

    stride = max(1, int(round(spacing_m / float(grid_point.resolution))))
    floor_mask = getattr(grid_point, "floor_mask", None)
    candidates = []
    waypoint_id = 0
    for gy in range(0, grid_point.height, stride):
        for gx in range(0, grid_point.width, stride):
            if floor_mask is not None and not bool(floor_mask[gy, gx]):
                continue
            point_ok = bool(grid_point.is_free(gx, gy))
            embodied_ok = bool(grid_embodied.is_free(gx, gy))
            if not point_ok and not embodied_ok:
                continue
            wx, wy = grid_point.grid_to_world(gx, gy)
            candidates.append({
                "waypoint_id": int(waypoint_id),
                "display_id": 0,
                "world_xyz": [round(float(wx), 3), round(float(wy), 3), round(float(floor_z), 3)],
                "grid_xy": [int(gx), int(gy)],
                "pointmass_walkable": point_ok,
                "embodied_feasible": embodied_ok,
                "candidate_source": "world_grid",
            })
            waypoint_id += 1

    if max_candidates > 0:
        candidates = candidates[:max_candidates]
    for idx, cand in enumerate(candidates, start=1):
        cand["display_id"] = idx
    return candidates


def bbox_corners_world(bbox: list[float]) -> list:
    if len(bbox) < 6:
        return []
    cx, cy, cz = bbox[0], bbox[1], bbox[2]
    sx, sy, sz = bbox[3], bbox[4], bbox[5]
    rx = bbox[6] if len(bbox) > 6 else 0.0
    ry = bbox[7] if len(bbox) > 7 else 0.0
    rz = bbox[8] if len(bbox) > 8 else 0.0
    rot = mathutils.Euler((rx, ry, rz), "XYZ").to_matrix()
    corners = []
    for dx in (-sx / 2, sx / 2):
        for dy in (-sy / 2, sy / 2):
            for dz in (-sz / 2, sz / 2):
                local = mathutils.Vector((dx, dy, dz))
                world = mathutils.Vector((cx, cy, cz)) + rot @ local
                corners.append(world)
    return corners


def sample_bbox_surface_points(bbox: list[float], samples_per_face: int = 4,
                               max_samples: int = 64) -> list:
    if len(bbox) < 6:
        return []
    cx, cy, cz = bbox[0], bbox[1], bbox[2]
    sx, sy, sz = bbox[3], bbox[4], bbox[5]
    rx = bbox[6] if len(bbox) > 6 else 0.0
    ry = bbox[7] if len(bbox) > 7 else 0.0
    rz = bbox[8] if len(bbox) > 8 else 0.0
    rot = mathutils.Euler((rx, ry, rz), "XYZ").to_matrix()

    xs = [(-sx / 2 + i * sx / (samples_per_face - 1)) for i in range(samples_per_face)]
    ys = [(-sy / 2 + i * sy / (samples_per_face - 1)) for i in range(samples_per_face)]
    zs = [(-sz / 2 + i * sz / (samples_per_face - 1)) for i in range(samples_per_face)]

    points = []
    for x in xs:
        for y in ys:
            points.append(mathutils.Vector((x, y, -sz / 2)))
            points.append(mathutils.Vector((x, y, sz / 2)))
    for x in xs:
        for z in zs:
            points.append(mathutils.Vector((x, -sy / 2, z)))
            points.append(mathutils.Vector((x, sy / 2, z)))
    for y in ys:
        for z in zs:
            points.append(mathutils.Vector((-sx / 2, y, z)))
            points.append(mathutils.Vector((sx / 2, y, z)))

    world_points = []
    center = mathutils.Vector((cx, cy, cz))
    for p in points:
        world_points.append(center + rot @ p)

    if len(world_points) > max_samples:
        world_points = world_points[:max_samples]
    return world_points


def target_bbox_image_metrics(
    bbox: list[float],
    cam_obj,
    fov_deg: float,
    res_x: int,
    res_y: int,
) -> dict | None:
    if len(bbox) < 9:
        return None
    vfov_half, hfov_half = _fov_params(fov_deg, res_x, res_y)
    cam_inv = cam_obj.matrix_world.inverted()
    corners = bbox_corners_world(bbox)
    pts = []
    for corner in corners:
        proj = project_world_point(
            cam_inv,
            corner,
            vfov_half,
            hfov_half,
            res_x,
            res_y,
            cam_obj=cam_obj,
            scene=bpy.context.scene,
        )
        if proj is not None:
            pts.append(proj["pixel_xy"])
    if not pts:
        return None
    xs = [int(p[0]) for p in pts]
    ys = [int(p[1]) for p in pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    clipped_min_x = max(0, min_x)
    clipped_max_x = min(res_x - 1, max_x)
    clipped_min_y = max(0, min_y)
    clipped_max_y = min(res_y - 1, max_y)
    width_px = max(0, clipped_max_x - clipped_min_x)
    height_px = max(0, clipped_max_y - clipped_min_y)
    return {
        "min_x": int(min_x),
        "max_x": int(max_x),
        "min_y": int(min_y),
        "max_y": int(max_y),
        "clipped_min_x": int(clipped_min_x),
        "clipped_max_x": int(clipped_max_x),
        "clipped_min_y": int(clipped_min_y),
        "clipped_max_y": int(clipped_max_y),
        "width_px": int(width_px),
        "height_px": int(height_px),
        "short_side_px": int(min(width_px, height_px)),
        "area_px": int(width_px * height_px),
    }


def is_large_object(obj: dict, large_categories: set[str], large_size_m: float) -> bool:
    category = obj.get("category", "unknown")
    if category in large_categories:
        return True
    bbox = obj.get("bbox", [])
    if len(bbox) >= 6:
        return max(bbox[3], bbox[4], bbox[5]) >= large_size_m
    return False


def target_visible_small(cam_obj, obj: dict, vfov_half: float, hfov_half: float,
                         res_x: int, res_y: int, depsgraph) -> bool:
    corners = bbox_corners_world(obj.get("bbox", []))
    if not corners:
        return False
    cam_inv = cam_obj.matrix_world.inverted()
    origin = cam_obj.matrix_world.translation.copy()
    for pt in corners:
        proj = project_world_point(cam_inv, pt, vfov_half, hfov_half, res_x, res_y, cam_obj=cam_obj, scene=bpy.context.scene)
        if proj is None:
            return False
        if not raycast_visible(bpy.context.scene, depsgraph, origin, pt):
            return False
    return True


def target_visible_large(cam_obj, obj: dict, vfov_half: float, hfov_half: float,
                         res_x: int, res_y: int, depsgraph, ratio: float) -> tuple[bool, float]:
    bbox = obj.get("bbox", [])
    if len(bbox) < 6:
        return False, 0.0
    center = mathutils.Vector((bbox[0], bbox[1], bbox[2]))
    cam_inv = cam_obj.matrix_world.inverted()
    origin = cam_obj.matrix_world.translation.copy()
    proj_center = project_world_point(cam_inv, center, vfov_half, hfov_half, res_x, res_y, cam_obj=cam_obj, scene=bpy.context.scene)
    if proj_center is None:
        return False, 0.0
    if not raycast_visible(bpy.context.scene, depsgraph, origin, center):
        return False, 0.0

    points = sample_bbox_surface_points(bbox, samples_per_face=4, max_samples=64)
    if not points:
        return False, 0.0
    visible = 0
    for pt in points:
        proj = project_world_point(cam_inv, pt, vfov_half, hfov_half, res_x, res_y, cam_obj=cam_obj, scene=bpy.context.scene)
        if proj is None:
            continue
        if raycast_visible(bpy.context.scene, depsgraph, origin, pt):
            visible += 1
    vis_ratio = visible / max(len(points), 1)
    return vis_ratio >= ratio, vis_ratio


def sample_target_surface_visibility_ratio(
    cam_obj,
    obj: dict,
    vfov_half: float,
    hfov_half: float,
    res_x: int,
    res_y: int,
    depsgraph,
) -> float:
    points = sample_bbox_surface_points(obj.get("bbox", []), samples_per_face=4, max_samples=64)
    if not points:
        return 0.0
    cam_inv = cam_obj.matrix_world.inverted()
    origin = cam_obj.matrix_world.translation.copy()
    visible = 0
    total = 0
    for pt in points:
        proj = project_world_point(
            cam_inv,
            pt,
            vfov_half,
            hfov_half,
            res_x,
            res_y,
            cam_obj=cam_obj,
            scene=bpy.context.scene,
        )
        if proj is None:
            continue
        total += 1
        if raycast_visible(bpy.context.scene, depsgraph, origin, pt):
            visible += 1
    if total <= 0:
        return 0.0
    return float(visible) / float(total)


def target_surface_visibility_threshold(
    is_large: bool,
    min_visible_ratio: float,
    large_visibility_ratio: float,
) -> float:
    return float(large_visibility_ratio if is_large else min_visible_ratio)


def target_surface_visibility_ok(
    is_large: bool,
    vis_ratio: float,
    center_visible: bool,
    min_visible_ratio: float,
    large_visibility_ratio: float,
) -> bool:
    if not center_visible:
        return False
    threshold = target_surface_visibility_threshold(
        is_large=is_large,
        min_visible_ratio=min_visible_ratio,
        large_visibility_ratio=large_visibility_ratio,
    )
    return float(vis_ratio) >= threshold


def build_target_candidates_from_visible_objects(
    *,
    layout: list[dict],
    visible_objects: list[dict],
    cam_position: list[float] | tuple[float, float, float],
    cam_obj,
    depsgraph,
    fov_deg: float,
    res_x: int,
    res_y: int,
    min_dist: float,
    max_targets: int,
    prefilter_targets: int,
    min_area_px: float,
    min_short_side_px: float,
    min_visible_surface_ratio: float = 0.15,
    large_visibility_ratio: float = 0.70,
    rec_categories: set[str],
    category_aliases: dict,
    large_categories: set[str],
    large_size: float,
    bbox_metrics_provider=target_bbox_image_metrics,
    visibility_ratio_provider=sample_target_surface_visibility_ratio,
) -> tuple[list[dict], list[dict], dict]:
    targets: list[dict] = []
    rejections: list[dict] = []
    dbg_counts = {
        "total": 0,
        "dist": 0,
        "raycast": 0,
        "bbox": 0,
        "size": 0,
        "vis": 0,
        "visibility_ok": 0,
        "path": 0,
        "path_visible": 0,
    }
    if not visible_objects:
        return targets, rejections, dbg_counts

    visible_lookup = {
        int(item.get("id", -1)): item
        for item in visible_objects
        if int(item.get("id", -1)) >= 0
    }
    layout_lookup = {
        int(obj.get("id", -1)): obj
        for obj in layout
        if obj.get("model_uid", "") and len(obj.get("bbox", [])) >= 6
    }

    cam_x = float(cam_position[0])
    cam_y = float(cam_position[1])
    vfov_half, hfov_half = _fov_params(float(fov_deg), int(res_x), int(res_y))

    for obj_id, visible in visible_lookup.items():
        obj = layout_lookup.get(obj_id)
        if obj is None:
            continue
        dbg_counts["total"] += 1

        bbox = obj.get("bbox", [])
        dist = math.hypot(float(bbox[0]) - cam_x, float(bbox[1]) - cam_y)
        if dist < float(min_dist):
            continue
        dbg_counts["dist"] += 1

        if not bool(visible.get("raycast_verified", False)):
            rejections.append({"target_id": obj_id, "reason": "target_foreground_occluded"})
            continue
        dbg_counts["raycast"] += 1

        bbox_metrics = bbox_metrics_provider(
            bbox,
            cam_obj=cam_obj,
            fov_deg=fov_deg,
            res_x=res_x,
            res_y=res_y,
        )
        if bbox_metrics is None:
            rejections.append({"target_id": obj_id, "reason": "outside_view"})
            continue
        dbg_counts["bbox"] += 1
        if float(bbox_metrics.get("area_px", 0.0)) < float(min_area_px):
            rejections.append({"target_id": obj_id, "reason": "target_too_small"})
            continue
        if float(bbox_metrics.get("short_side_px", 0.0)) < float(min_short_side_px):
            rejections.append({"target_id": obj_id, "reason": "target_short_side_too_small"})
            continue
        dbg_counts["size"] += 1

        category_meta = canonicalize_category(obj.get("category", "unknown"), category_aliases)
        is_large = is_large_object(obj, large_categories, large_size)
        vis_ratio = float(visibility_ratio_provider(
            cam_obj,
            obj,
            vfov_half,
            hfov_half,
            res_x,
            res_y,
            depsgraph,
        ))
        dbg_counts["vis"] += 1
        if not target_surface_visibility_ok(
            is_large=is_large,
            vis_ratio=vis_ratio,
            center_visible=True,
            min_visible_ratio=min_visible_surface_ratio,
            large_visibility_ratio=large_visibility_ratio,
        ):
            rejections.append({
                "target_id": obj_id,
                "reason": "target_visibility_too_low",
                "visibility_ratio": round(vis_ratio, 3),
            })
            continue
        dbg_counts["visibility_ok"] += 1

        material_color_rgb = obj.get("material_color_rgb")
        if not isinstance(material_color_rgb, (list, tuple)) or len(material_color_rgb) < 3:
            material_color_rgb = None
        else:
            material_color_rgb = [round(float(v), 4) for v in material_color_rgb[:3]]
        material_color_name = quantize_material_color_name(material_color_rgb)
        distance_bonus = min(max(float(dist) - float(min_dist), 0.0), 10.0) * 0.35
        score = (
            (10.0 if category_meta["canonical_category"] in rec_categories else 0.0)
            + vis_ratio * 2.0
            + distance_bonus
        )
        targets.append({
            "target_id": obj_id,
            "pass_index": obj_id + 1,
            "category": obj.get("category", "unknown"),
            "canonical_category": category_meta["canonical_category"],
            "semantic_group_id": category_meta["semantic_group_id"],
            "reference_family_id": category_meta["semantic_group_id"],
            "material_color_rgb": material_color_rgb,
            "material_color_name": material_color_name,
            "visible_color_rgb": None,
            "visible_color_name": "unknown",
            "distance": round(dist, 3),
            "is_large": is_large,
            "visibility_ratio": round(vis_ratio, 3),
            "bbox_metrics": bbox_metrics,
            "score": round(score, 3),
        })

    family_groups: dict[str, list[dict]] = {}
    family_color_groups: dict[tuple[str, str], list[dict]] = {}
    for target in targets:
        family_id = str(target.get("reference_family_id", "unknown"))
        color_name = str(target.get("material_color_name", "unknown"))
        family_groups.setdefault(family_id, []).append(target)
        family_color_groups.setdefault((family_id, color_name), []).append(target)

    for target in targets:
        family_id = str(target.get("reference_family_id", "unknown"))
        color_name = str(target.get("material_color_name", "unknown"))
        same_family_total = len(family_groups.get(family_id, []))
        same_family_same_color = len(family_color_groups.get((family_id, color_name), []))
        same_family_diff_color = max(0, same_family_total - same_family_same_color)
        ambiguity_bonus = (
            min(max(0, same_family_same_color - 1), 2) * 3.0
            + min(same_family_diff_color, 2) * 2.0
        )
        target["score"] = round(float(target.get("score", 0.0)) + ambiguity_bonus, 3)
        target["same_family_visible_count"] = int(same_family_total)
        target["same_family_same_color_visible_count"] = max(0, same_family_same_color - 1)
        target["same_family_diff_color_visible_count"] = int(same_family_diff_color)

    targets.sort(key=lambda x: (-x["score"], float(x.get("distance", 1e9)), int(x.get("target_id", -1))))
    targets = targets[:max(int(prefilter_targets), int(max_targets))]
    return targets, rejections, dbg_counts


def view_candidate_meets_visibility_floor(
    candidate: dict,
    min_visible_objects: int = 3,
) -> bool:
    return int(candidate.get("n_vis", 0)) >= int(min_visible_objects)


def build_scored_camera_candidates(
    candidate_records: list[dict],
    clearance_provider,
) -> list[dict]:
    scored: list[dict] = []
    for candidate in candidate_records:
        pos = candidate.get("pos", (0.0, 0.0))
        cx, cy = float(pos[0]), float(pos[1])
        angle = float(candidate.get("angle", 0.0))
        clearance = clearance_provider(cx, cy, angle) or {}
        scored.append({
            "pos": (cx, cy),
            "angle": angle,
            "n_vis": int(candidate.get("n_vis", 0)),
            "path_score": float(candidate.get("path_score", 0.0)),
            "ids": set(candidate.get("ids", set())),
            "objs": list(candidate.get("objs", [])),
            "bdist": float(candidate.get("bdist", 0.0)),
            "target_id": candidate.get("target_id"),
            "camera_seed_source": candidate.get("camera_seed_source"),
            "candidate_pool_stage": candidate.get("candidate_pool_stage"),
            "fwd_clearance_ray": float(clearance.get("fwd_clearance_ray", 3.0)),
            "fwd_clearance_obj": float(clearance.get("fwd_clearance_obj", 3.0)),
        })
    return scored


def target_view_seed_score(candidate: dict) -> float:
    target_distance = float(candidate.get("target_distance_m", 0.0))
    same_family_total = max(0, int(candidate.get("same_family_visible_count", 0)))
    same_family_diff_color = max(0, int(candidate.get("same_family_diff_color_visible_count", 0)))
    visibility_ratio = max(0.0, min(1.0, float(candidate.get("visibility_ratio", 0.0))))
    is_recommended = bool(candidate.get("is_recommended", False))

    distance_bonus = max(0.0, target_distance - 2.0) * 1.25
    family_bonus = min(max(0, same_family_total - 1), 3) * 1.5
    color_bonus = min(same_family_diff_color, 2) * 2.0
    visibility_bonus = visibility_ratio * 1.5
    recommended_bonus = 1.5 if is_recommended else 0.0
    return distance_bonus + family_bonus + color_bonus + visibility_bonus + recommended_bonus


def resolve_view_seed_experiment_mode(target_cfg: dict | None) -> str:
    raw_mode = str((target_cfg or {}).get("view_seed_experiment_mode", "preselection_off")).strip().lower()
    aliases = {
        "baseline": "baseline",
        "default": "preselection_off",
        "preselection_off": "preselection_off",
        "disable_target_preselection": "preselection_off",
        "route_aware": "route_aware",
        "route-aware": "route_aware",
    }
    return aliases.get(raw_mode, "preselection_off")


def route_aware_view_seed_score(
    candidate: dict,
    *,
    min_dist: float,
    rec_categories: set[str] | None = None,
) -> float:
    rec_categories = rec_categories or set()
    pos = candidate.get("pos", (0.0, 0.0))
    cx, cy = float(pos[0]), float(pos[1])
    objs = list(candidate.get("objs", []) or [])
    if not objs:
        return 0.0

    family_counts: Counter[str] = Counter()
    family_color_sets: dict[str, set[str]] = {}
    eligible_count = 0
    far_count = 0
    recommended_count = 0
    mean_dist_sum = 0.0

    for obj in objs:
        ox = float(obj.get("x", 0.0))
        oy = float(obj.get("y", 0.0))
        dist = math.hypot(ox - cx, oy - cy)
        if dist < float(min_dist):
            continue
        eligible_count += 1
        mean_dist_sum += dist
        if dist >= float(min_dist) + 1.0:
            far_count += 1
        canonical_category = str(obj.get("canonical_category", obj.get("category", "unknown")))
        if canonical_category in rec_categories:
            recommended_count += 1
        family_id = str(obj.get("semantic_group_id", canonical_category))
        color_name = _normalized_instruction_text(obj.get("material_color_name")) or "unknown"
        family_counts[family_id] += 1
        family_color_sets.setdefault(family_id, set()).add(color_name)

    if eligible_count <= 0:
        return 0.0

    same_family_competition = sum(max(0, count - 1) for count in family_counts.values())
    diff_color_competition = sum(
        1
        for colors in family_color_sets.values()
        if len(colors - {"unknown"}) >= 2
    )
    multi_family_bonus = max(0, len(family_counts) - 1)
    mean_dist = mean_dist_sum / max(eligible_count, 1)

    return (
        eligible_count * 2.0
        + far_count * 1.5
        + same_family_competition * 2.5
        + diff_color_competition * 3.0
        + multi_family_bonus * 0.75
        + max(0.0, mean_dist - float(min_dist)) * 0.75
        + recommended_count * 0.8
    )


def select_scored_camera_views(
    scored: list[dict],
    num_views: int,
    *,
    min_visible_objects: int = 3,
    route_aware_weight: float = 0.0,
) -> list[dict]:
    if num_views <= 0 or not scored:
        return []

    selected: list[dict] = []
    covered: set[int] = set()
    remaining = list(range(len(scored)))

    for _ in range(int(num_views)):
        if not remaining:
            break
        best_score = -1.0
        best_ri = -1
        for ri, ci in enumerate(remaining):
            c = scored[ci]
            if not view_candidate_meets_visibility_floor(c, min_visible_objects=min_visible_objects):
                continue
            if float(c.get("fwd_clearance_ray", 0.0)) < 0.8:
                continue

            n_new = len(set(c.get("ids", set())) - covered)
            score = (
                n_new * 6.0
                + float(c.get("n_vis", 0.0)) * 0.3
                + float(route_aware_weight) * float(c.get("path_score", 0.0))
            )
            if float(c.get("bdist", 0.0)) < 1.0:
                score *= 1.2
            if float(c.get("fwd_clearance_ray", 0.0)) < 1.5:
                score *= 0.3
            if float(c.get("fwd_clearance_obj", 0.0)) < 0.5:
                score *= 0.2
            elif float(c.get("fwd_clearance_obj", 0.0)) < 1.0:
                score *= 0.5

            for prev in selected:
                d = math.hypot(
                    float(c.get("pos", (0.0, 0.0))[0]) - float(prev.get("pos", (0.0, 0.0))[0]),
                    float(c.get("pos", (0.0, 0.0))[1]) - float(prev.get("pos", (0.0, 0.0))[1]),
                )
                if d < 1.0:
                    score *= 0.15
                elif d < 1.8:
                    score *= 0.5

            if score > best_score:
                best_score = score
                best_ri = ri

        if best_ri < 0:
            break

        sel = scored[remaining[best_ri]]
        selected.append(sel)
        covered |= {
            int(value)
            for value in (sel.get("ids", set()) or set())
            if int(value) >= 0
        }
        remaining.pop(best_ri)

    return selected


def camera_selection_coverage_count(selected: list[dict]) -> int:
    covered: set[int] = set()
    for candidate in selected:
        covered.update(
            int(value)
            for value in (candidate.get("ids", set()) or set())
            if int(value) >= 0
        )
    return len(covered)


def select_target_view_seeds(
    seed_records: list[dict],
    num_views: int,
    *,
    min_pos_separation_m: float = 1.0,
    min_yaw_separation_deg: float = 20.0,
) -> list[dict]:
    if num_views <= 0 or not seed_records:
        return []

    selected: list[dict] = []
    covered_ids: set[int] = set()
    remaining = [dict(record) for record in seed_records]
    min_yaw_separation = math.radians(float(min_yaw_separation_deg))

    while remaining and len(selected) < int(num_views):
        best_idx = -1
        best_score = -1e18
        for idx, record in enumerate(remaining):
            record_ids = {
                int(value)
                for value in (record.get("ids", set()) or set())
                if int(value) >= 0
            }
            score = (
                float(record.get("path_score", 0.0)) * 8.0
                + len(record_ids - covered_ids) * 3.0
                + float(record.get("n_vis", 0.0)) * 0.3
            )
            pos = record.get("pos", (0.0, 0.0))
            angle = float(record.get("angle", 0.0))
            for prev in selected:
                prev_pos = prev.get("pos", (0.0, 0.0))
                dist = math.hypot(float(pos[0]) - float(prev_pos[0]), float(pos[1]) - float(prev_pos[1]))
                yaw_diff = abs((angle - float(prev.get("angle", 0.0)) + math.pi) % (2 * math.pi) - math.pi)
                if dist < float(min_pos_separation_m):
                    score -= 8.0
                    if yaw_diff < min_yaw_separation:
                        score -= 4.0
                elif dist < 1.8:
                    score -= 2.5
            if score > best_score:
                best_idx = idx
                best_score = score

        if best_idx < 0:
            break

        best = remaining.pop(best_idx)
        selected.append(best)
        covered_ids |= {
            int(value)
            for value in (best.get("ids", set()) or set())
            if int(value) >= 0
        }

    return selected


def forward_clear(cam_pos: tuple[float, float, float], cam_yaw: float,
                  layout: list[dict], radius: float, height: float) -> bool:
    fwd_cos, fwd_sin = math.cos(cam_yaw), math.sin(cam_yaw)
    for obj in layout:
        bbox = obj.get("bbox", [])
        if len(bbox) < 6:
            continue
        ox, oy, oz = bbox[0], bbox[1], bbox[2]
        sx, sy, sz = bbox[3], bbox[4], bbox[5]
        obj_bottom = oz - sz / 2
        obj_top = oz + sz / 2
        if obj_bottom >= height or obj_top < 0.10:
            continue
        dx = ox - cam_pos[0]
        dy = oy - cam_pos[1]
        # In front half-plane
        if dx * fwd_cos + dy * fwd_sin <= 0:
            continue
        # lateral distance to forward axis
        lat = abs(-dx * fwd_sin + dy * fwd_cos)
        if lat > radius:
            continue
        # radial distance within 1m
        if math.hypot(dx, dy) <= radius:
            return False
    return True


def find_nearest_free(grid, target_xy: tuple[float, float], max_radius_m: float = 1.0):
    gx, gy = grid.world_to_grid(target_xy[0], target_xy[1])
    if grid.is_free(gx, gy):
        return gx, gy
    max_r = int(math.ceil(max_radius_m / grid.resolution))
    for r in range(1, max_r + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nx, ny = gx + dx, gy + dy
                if grid.is_free(nx, ny):
                    return nx, ny
    return None


def build_nav_grid(layout: list[dict], config: dict):
    import numpy as np
    nav_cfg = config["navigation"]
    resolution = nav_cfg["grid_resolution"]
    agent_radius = nav_cfg["agent_radius"]

    positions = []
    for obj in layout:
        bbox = obj.get("bbox", [])
        if len(bbox) >= 6:
            x, y = bbox[0], bbox[1]
            sx, sy = bbox[3], bbox[4]
            positions.append((x - sx, y - sy))
            positions.append((x + sx, y + sy))

    if not positions:
        return None

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    margin = 2.0
    min_x, max_x = min(xs) - margin, max(xs) + margin
    min_y, max_y = min(ys) - margin, max(ys) + margin

    width = int(math.ceil((max_x - min_x) / resolution))
    height = int(math.ceil((max_y - min_y) / resolution))
    grid = np.zeros((height, width), dtype=np.uint8)

    def world_to_grid(x: float, y: float):
        gx = int((x - min_x) / resolution)
        gy = int((y - min_y) / resolution)
        gx = max(0, min(gx, width - 1))
        gy = max(0, min(gy, height - 1))
        return gx, gy

    def grid_to_world(gx: int, gy: int):
        x = min_x + (gx + 0.5) * resolution
        y = min_y + (gy + 0.5) * resolution
        return x, y

    # Mark obstacles
    for obj in layout:
        bbox = obj.get("bbox", [])
        if len(bbox) >= 9:
            x, y = bbox[0], bbox[1]
            sx, sy = bbox[3], bbox[4]
            rot = bbox[6] if len(bbox) > 6 else 0.0
            height_obj = bbox[5] if len(bbox) > 5 else 1.0
            base_z = bbox[2] if len(bbox) > 2 else 0.0
            if base_z < 1.5 and height_obj > 0.1:
                cos_r = abs(math.cos(rot))
                sin_r = abs(math.sin(rot))
                eff_sx = (sx * cos_r + sy * sin_r) / 2 + agent_radius
                eff_sy = (sx * sin_r + sy * cos_r) / 2 + agent_radius
                gx1, gy1 = world_to_grid(x - eff_sx, y - eff_sy)
                gx2, gy2 = world_to_grid(x + eff_sx, y + eff_sy)
                grid[gy1:gy2 + 1, gx1:gx2 + 1] = 1

    class SimpleGrid:
        def __init__(self):
            self.min_x = min_x
            self.min_y = min_y
            self.resolution = resolution
            self.width = width
            self.height = height
            self.grid = grid

        def world_to_grid(self, x: float, y: float):
            return world_to_grid(x, y)

        def grid_to_world(self, gx: int, gy: int):
            return grid_to_world(gx, gy)

        def is_free(self, gx: int, gy: int) -> bool:
            if 0 <= gx < self.width and 0 <= gy < self.height:
                return self.grid[gy, gx] == 0
            return False

    return SimpleGrid()


def astar_nav(grid, start: tuple[int, int], goal: tuple[int, int]):
    import heapq
    if not grid.is_free(*start) or not grid.is_free(*goal):
        return None
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def heuristic(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    while open_set:
        _f, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return list(reversed(path))
        for dx, dy in neighbors:
            nb = (current[0] + dx, current[1] + dy)
            if not grid.is_free(*nb):
                continue
            tentative = g_score[current] + heuristic(current, nb)
            if tentative < g_score.get(nb, 1e9):
                came_from[nb] = current
                g_score[nb] = tentative
                f = tentative + heuristic(nb, goal)
                heapq.heappush(open_set, (f, nb))
    return None


def compute_path_world(grid, start_xy: tuple[float, float], goal_xy: tuple[float, float]):
    if grid is None:
        return None
    start = grid.world_to_grid(start_xy[0], start_xy[1])
    goal = find_nearest_free(grid, goal_xy, max_radius_m=1.5)
    if goal is None:
        return None
    path = astar_nav(grid, start, goal)
    if path is None:
        return None
    world_path = [grid.grid_to_world(gx, gy) for gx, gy in path]
    return [(p[0], p[1], 0.0) for p in world_path]


def smooth_path_world_with_los(
    path_world: list[tuple[float, float, float]],
    nav_grid,
) -> list[tuple[float, float, float]]:
    if len(path_world) <= 2 or nav_grid is None or not hasattr(nav_grid, "is_path_traversable"):
        return path_world

    smoothed: list[tuple[float, float, float]] = [path_world[0]]
    i = 0
    n = len(path_world)
    while i < n - 1:
        best_j = i + 1
        for j in range(n - 1, i, -1):
            p0 = (float(path_world[i][0]), float(path_world[i][1]))
            p1 = (float(path_world[j][0]), float(path_world[j][1]))
            ok = nav_grid.is_path_traversable([p0, p1])
            if bool(ok.get("traversable", False)):
                best_j = j
                break
        smoothed.append(path_world[best_j])
        i = best_j
    return smoothed


def _normalize_path_world_points(
    path_world: list[tuple[float, float, float]] | list[list[float]] | None,
) -> list[list[float]]:
    normalized: list[list[float]] = []
    for pt in path_world or []:
        if pt is None or len(pt) < 2:
            continue
        z = float(pt[2]) if len(pt) > 2 else 0.0
        normalized.append([float(pt[0]), float(pt[1]), z])
    return normalized


def build_continuous_optimal_path(
    search_path_world: list[tuple[float, float, float]] | list[list[float]],
    nav_grid,
) -> list[list[float]]:
    normalized = _normalize_path_world_points(search_path_world)
    if len(normalized) <= 2:
        return normalized
    if nav_grid is None or not hasattr(nav_grid, "is_path_traversable"):
        return normalized

    current = [tuple(pt) for pt in normalized]
    while True:
        refined = smooth_path_world_with_los(current, nav_grid)
        if len(refined) == len(current) and all(
            all(abs(float(a) - float(b)) <= 1e-8 for a, b in zip(p0, p1))
            for p0, p1 in zip(refined, current)
        ):
            break
        current = refined
    return _normalize_path_world_points(current)


def validate_canonical_sparse_path(
    *,
    optimal_path_world: list[tuple[float, float, float]] | list[list[float]],
    canonical_sparse_path_world: list[tuple[float, float, float]] | list[list[float]],
    nav_grid,
    obstacle_mask: np.ndarray | None,
    projected_points: list[list[int] | None] | None,
    pad_px: int = 2,
    epsilon_length_m: float = 1e-3,
) -> None:
    optimal = _normalize_path_world_points(optimal_path_world)
    sparse = _normalize_path_world_points(canonical_sparse_path_world)
    if len(optimal) < 2:
        raise ValueError("optimal_path_world must contain at least two points")
    if len(sparse) < 2:
        raise ValueError("canonical_sparse_path_world must contain at least two points")

    def _segment_traversable(path_pts: list[list[float]]) -> bool:
        if nav_grid is None or not hasattr(nav_grid, "is_path_traversable"):
            return True
        for p0, p1 in zip(path_pts, path_pts[1:]):
            result = nav_grid.is_path_traversable([
                (float(p0[0]), float(p0[1])),
                (float(p1[0]), float(p1[1])),
            ])
            if not bool(result.get("traversable", False)):
                return False
        return True

    def _projection_safe(points: list[list[int] | None] | None) -> bool:
        if obstacle_mask is None or points is None:
            return True
        return projected_polyline_collision_free(
            points,
            obstacle_mask=obstacle_mask,
            pad_px=pad_px,
            require_all_points_visible=True,
        )

    if not _segment_traversable(sparse):
        raise ValueError("canonical sparse route contains non-traversable segment")

    sparse_len = path_length_m(sparse)
    optimal_len = path_length_m(optimal)
    if sparse_len - optimal_len > float(epsilon_length_m):
        raise ValueError("canonical sparse route exceeds optimal length tolerance")

    if not _projection_safe(projected_points):
        raise ValueError("canonical sparse route is not projection-safe")

    if len(sparse) <= 2:
        return

    for idx in range(1, len(sparse) - 1):
        candidate_sparse = sparse[:idx] + sparse[idx + 1:]
        candidate_projected = None
        if projected_points is not None and len(projected_points) == len(sparse):
            candidate_projected = list(projected_points[:idx]) + list(projected_points[idx + 1:])
        if not _segment_traversable(candidate_sparse):
            continue
        if not _projection_safe(candidate_projected):
            continue
        candidate_len = path_length_m(candidate_sparse)
        if candidate_len - optimal_len <= float(epsilon_length_m):
            raise ValueError("non-minimal sparse route")


def path_length_m(path_world: list[tuple[float, float, float]] | list[list[float]]) -> float:
    if not path_world or len(path_world) < 2:
        return 0.0
    total = 0.0
    for i in range(len(path_world) - 1):
        x0, y0 = float(path_world[i][0]), float(path_world[i][1])
        x1, y1 = float(path_world[i + 1][0]), float(path_world[i + 1][1])
        total += math.hypot(x1 - x0, y1 - y0)
    return float(total)


def find_shortest_candidate_path(
    nav_grid,
    start_candidates: list[dict],
    goal_candidates: list[dict],
    max_start_candidates: int = 18,
    max_goal_candidates: int = 18,
) -> list[tuple[float, float, float]] | None:
    if nav_grid is None or not start_candidates or not goal_candidates:
        return None

    ranked_starts = sorted(
        start_candidates,
        key=lambda cand: (
            float(cand.get("screen_xy_norm", [0.0, 0.0])[1]),
            abs(float(cand.get("screen_xy_norm", [0.0, 0.0])[0])),
            float(cand.get("depth", 1e9)),
        ),
    )[:max_start_candidates]
    ranked_goals = sorted(
        goal_candidates,
        key=lambda cand: (
            float(cand.get("goal_distance_m", 1e9)),
            float(cand.get("depth", 1e9)),
        ),
    )[:max_goal_candidates]

    best_path = None
    best_len = float("inf")
    for start in ranked_starts:
        sw = start.get("world_xyz", [0.0, 0.0, 0.0])
        start_xy = (float(sw[0]), float(sw[1]))
        for goal in ranked_goals:
            gw = goal.get("world_xyz", [0.0, 0.0, 0.0])
            goal_xy = (float(gw[0]), float(gw[1]))
            path = compute_path_world(nav_grid, start_xy, goal_xy)
            if not path:
                continue
            plen = path_length_m(path)
            if plen < best_len:
                best_len = plen
                best_path = path
    return best_path


def select_fixed_start_candidate(
    start_candidates: list[dict],
    anchor_world_xyz: list[float] | tuple[float, float, float] | None = None,
    max_anchor_snap_dist_m: float | None = None,
) -> dict | None:
    if not start_candidates:
        return None
    candidates = [cand for cand in start_candidates if bool(cand.get("embodied_feasible", False))]
    if not candidates:
        return None

    def _legacy_score(cand: dict) -> tuple[float, float, int]:
        return (
            (3.0 * abs(float(cand.get("screen_xy_norm", [0.0, 0.0])[0])))
            + abs(float(cand.get("screen_xy_norm", [0.0, 0.0])[1]) + 1.0),
            float(cand.get("depth", 1e9)),
            int(cand.get("waypoint_id", -1)),
        )

    if anchor_world_xyz is None:
        return min(candidates, key=_legacy_score)

    ax, ay = float(anchor_world_xyz[0]), float(anchor_world_xyz[1])

    def _anchored_score(cand: dict) -> tuple[float, float, float, float, int]:
        world_xyz = cand.get("world_xyz", [0.0, 0.0, 0.0])
        wx, wy = float(world_xyz[0]), float(world_xyz[1])
        return (
            math.hypot(wx - ax, wy - ay),
            abs(float(cand.get("screen_xy_norm", [0.0, 0.0])[0])),
            abs(float(cand.get("screen_xy_norm", [0.0, 0.0])[1]) + 1.0),
            float(cand.get("depth", 1e9)),
            int(cand.get("waypoint_id", -1)),
        )

    best = min(candidates, key=_anchored_score)
    if max_anchor_snap_dist_m is not None:
        best_score = _anchored_score(best)[0]
        if best_score > float(max_anchor_snap_dist_m):
            return None
    return best


def select_path_endpoint_anchor(
    endpoint_world: list[float] | tuple[float, float, float],
    primary_path_ids: list[int],
    fallback_path_ids: list[int],
    candidate_lookup: dict[int, dict],
) -> dict | None:
    if endpoint_world is None:
        return None
    ex, ey = float(endpoint_world[0]), float(endpoint_world[1])
    ordered_ids: list[int] = []
    for path_ids in (primary_path_ids, fallback_path_ids):
        for wp_id in path_ids:
            wp_id = int(wp_id)
            if wp_id < 0 or wp_id in ordered_ids:
                continue
            ordered_ids.append(wp_id)
    if not ordered_ids:
        return None

    def _candidate_score(wp_id: int) -> tuple[float, int, int]:
        rec = candidate_lookup.get(int(wp_id), {})
        world_xyz = rec.get("world_xyz", [0.0, 0.0, 0.0])
        wx, wy = float(world_xyz[0]), float(world_xyz[1])
        return (
            math.hypot(wx - ex, wy - ey),
            ordered_ids.index(int(wp_id)),
            int(wp_id),
        )

    best_id = min(ordered_ids, key=_candidate_score)
    return candidate_lookup.get(int(best_id))


def select_goal_routing_candidates(
    candidates: list[dict],
    bbox: list[float],
    ring_tolerance_m: float,
    feasible_key: str = "embodied_feasible",
) -> list[dict]:
    goal_candidates = []
    min_goal_dist = float("inf")
    for cand in candidates:
        if not bool(cand.get(feasible_key, False)):
            continue
        w = cand.get("world_xyz", [0.0, 0.0, 0.0])
        goal_dist = distance_to_oriented_bbox_2d(float(w[0]), float(w[1]), bbox)
        goal_copy = dict(cand)
        goal_copy["goal_distance_m"] = round(goal_dist, 4)
        goal_candidates.append(goal_copy)
        min_goal_dist = min(min_goal_dist, float(goal_dist))
    if not goal_candidates:
        return []
    goal_ring_threshold_m = float(min_goal_dist) + float(ring_tolerance_m)
    if feasible_key == "pointmass_walkable":
        nearest_embodied_goal_dist = min(
            (
                float(cand.get("goal_distance_m", 1e9))
                for cand in goal_candidates
                if bool(cand.get("embodied_feasible", False))
            ),
            default=None,
        )
        if nearest_embodied_goal_dist is not None:
            goal_ring_threshold_m = max(goal_ring_threshold_m, float(nearest_embodied_goal_dist))
    filtered = []
    for cand in goal_candidates:
        if float(cand.get("goal_distance_m", 1e9)) <= goal_ring_threshold_m + 1e-9:
            cand["goal_ring_min_distance_m"] = round(float(min_goal_dist), 4)
            cand["goal_ring_threshold_m"] = round(float(goal_ring_threshold_m), 4)
            filtered.append(cand)
    return filtered


def routing_endpoint_within_goal_radius(
    path_world: list[list[float]] | list[tuple[float, float, float]],
    bbox: list[float],
    goal_radius_m: float,
) -> bool:
    if not path_world:
        return False
    endpoint = path_world[-1]
    return distance_to_oriented_bbox_2d(
        float(endpoint[0]),
        float(endpoint[1]),
        bbox,
    ) <= float(goal_radius_m)


def _normalized_instruction_text(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text in {"unknown", "none", "null", "nan"}:
        return None
    return text


def _candidate_reference_family_id(candidate: dict) -> str:
    return (
        _normalized_instruction_text(candidate.get("reference_family_id"))
        or _normalized_instruction_text(candidate.get("semantic_group_id"))
        or _normalized_instruction_text(candidate.get("canonical_category"))
        or _normalized_instruction_text(candidate.get("target_category"))
        or "unknown"
    )


def _candidate_visible_color_name(candidate: dict) -> str | None:
    return (
        _normalized_instruction_text(candidate.get("visible_color_name"))
        or _normalized_instruction_text(candidate.get("target_color_name"))
        or _normalized_instruction_text(candidate.get("material_color_name"))
    )


def _candidate_reference_color_name(candidate: dict) -> str | None:
    return (
        _normalized_instruction_text(candidate.get("reference_color_name"))
        or _normalized_instruction_text(candidate.get("visible_color_name"))
        or _normalized_instruction_text(candidate.get("target_color_name"))
        or _normalized_instruction_text(candidate.get("material_color_name"))
    )


def _candidate_reference_color_source(candidate: dict) -> str | None:
    if _normalized_instruction_text(candidate.get("reference_color_name")) and _normalized_instruction_text(
        candidate.get("reference_color_source")
    ):
        return _normalized_instruction_text(candidate.get("reference_color_source"))
    if _normalized_instruction_text(candidate.get("visible_color_name")) or _normalized_instruction_text(
        candidate.get("target_color_name")
    ):
        return "visible"
    if _normalized_instruction_text(candidate.get("material_color_name")):
        return "material"
    return None


def _candidate_has_reference_support(candidate: dict) -> bool:
    axes = candidate.get("navigation_axes", {}) or {}
    if str(axes.get("reference_axis", "low")) != "low":
        return True
    reference_facts = candidate.get("navigation_reference_facts", {}) or {}
    return str(reference_facts.get("reference_resolution_mode", "category_only")) != "category_only"


def _candidate_has_geometry_support(candidate: dict) -> bool:
    axes = candidate.get("navigation_axes", {}) or {}
    if str(axes.get("geometry_axis", "low")) != "low":
        return True
    geometry_facts = candidate.get("navigation_geometry_facts", {}) or {}
    return str(geometry_facts.get("geometry_profile", "straight_short")) != "straight_short"


def routing_support_summary(candidates: list[dict]) -> dict[str, int]:
    reference_supported = 0
    geometry_supported = 0
    nontrivial_supported = 0
    for candidate in candidates or []:
        has_reference = _candidate_has_reference_support(candidate)
        has_geometry = _candidate_has_geometry_support(candidate)
        if has_reference:
            reference_supported += 1
        if has_geometry:
            geometry_supported += 1
        if has_reference or has_geometry:
            nontrivial_supported += 1
    return {
        "total": len(candidates or []),
        "reference_supported": int(reference_supported),
        "geometry_supported": int(geometry_supported),
        "nontrivial_supported": int(nontrivial_supported),
    }


def routing_view_has_support(
    candidates: list[dict],
    *,
    min_nontrivial_routes: int = 1,
    require_reference_support: bool = False,
    require_geometry_support: bool = False,
) -> bool:
    summary = routing_support_summary(candidates)
    if require_reference_support and summary["reference_supported"] <= 0:
        return False
    if require_geometry_support and summary["geometry_supported"] <= 0:
        return False
    return summary["nontrivial_supported"] >= int(min_nontrivial_routes)


def annotate_routing_instruction_metadata(
    candidates: list[dict],
    target_rule: str = "nearest",
    distance_tie_eps_m: float = 1e-3,
):
    annotated: list[dict] = [dict(candidate) for candidate in candidates]
    grouped: dict[tuple[str, str | None], list[dict]] = {}
    for candidate in annotated:
        reference_family_id = _candidate_reference_family_id(candidate)
        reference_color_name = _candidate_reference_color_name(candidate)
        key = (reference_family_id, reference_color_name)
        grouped.setdefault(key, []).append(candidate)

    rejections: list[dict] = []
    for (reference_family_id, reference_color_name), cohort in grouped.items():
        ordered = sorted(
            cohort,
            key=lambda c: (
                float(c.get("distance_m", c.get("distance", 1e9))),
                int(c.get("target_id", -1)),
            ),
        )
        cohort_ids = [int(c.get("target_id", -1)) for c in ordered]
        best_distance = float(ordered[0].get("distance_m", ordered[0].get("distance", 1e9)))
        winner_ids = [
            int(c.get("target_id", -1))
            for c in ordered
            if abs(float(c.get("distance_m", c.get("distance", 1e9))) - best_distance) <= distance_tie_eps_m
        ]
        ambiguous = len(winner_ids) > 1

        for rank, candidate in enumerate(ordered, start=1):
            target_id = int(candidate.get("target_id", -1))
            selection_status = "ambiguous_distance_tie"
            is_rule_winner = False
            if not ambiguous:
                if target_id == winner_ids[0]:
                    selection_status = "unique_winner"
                    is_rule_winner = True
                else:
                    selection_status = "farther_same_identity"

            raw_visible_color_name = _normalized_instruction_text(candidate.get("visible_color_name"))
            raw_material_color_name = _normalized_instruction_text(candidate.get("material_color_name"))
            candidate["instruction_identity"] = {
                "reference_family_id": reference_family_id,
                "semantic_group_id": reference_family_id,
                "visible_color_name": raw_visible_color_name,
                "material_color_name": raw_material_color_name,
                "reference_color_name": reference_color_name,
                "reference_color_source": _candidate_reference_color_source(candidate),
                "target_rule": target_rule,
            }
            candidate["instruction_selection"] = {
                "cohort_target_ids": cohort_ids,
                "cohort_size": len(cohort_ids),
                "winner_target_ids": winner_ids,
                "winner_count": len(winner_ids),
                "distance_rank": rank,
                "is_rule_winner": is_rule_winner,
                "selection_status": selection_status,
            }

            if ambiguous:
                rejections.append({
                    "target_id": target_id,
                    "reason": "instruction_identity_ambiguous",
                    "reference_family_id": reference_family_id,
                    "semantic_group_id": reference_family_id,
                    "visible_color_name": raw_visible_color_name,
                    "material_color_name": raw_material_color_name,
                    "reference_color_name": reference_color_name,
                    "reference_color_source": _candidate_reference_color_source(candidate),
                    "target_rule": target_rule,
                    "winner_target_ids": winner_ids,
                })

    return annotated, rejections


def _routing_axis_cell(candidate: dict) -> str:
    axes = candidate.get("navigation_axes", {}) or {}
    reference_axis = str(axes.get("reference_axis", "unknown"))
    geometry_axis = str(axes.get("geometry_axis", "unknown"))
    embodiment_axis = str(axes.get("embodiment_axis", "unknown"))
    return f"{reference_axis}/{geometry_axis}/{embodiment_axis}"


def summarize_camera_seed_funnel_audit(
    *,
    scene_id: str,
    camera_seed_audit: dict | None,
) -> dict:
    camera_seed_audit = camera_seed_audit or {}
    candidate_pool_counts = camera_seed_audit.get("candidate_pool_counts", {}) or {}
    target_seed_counts = camera_seed_audit.get("target_seed_counts", {}) or {}
    selection_counts = camera_seed_audit.get("selection_counts", {}) or {}
    selected_records = list(camera_seed_audit.get("selected_records", []) or [])
    source_counts = Counter(
        str(record.get("source", "unknown"))
        for record in selected_records
    )

    return {
        "record_type": "camera_seed_summary",
        "scene_id": str(scene_id),
        "experiment_mode": str(camera_seed_audit.get("experiment_mode", "baseline")),
        "rng_seed": camera_seed_audit.get("rng_seed"),
        "candidate_pool_stage": str(camera_seed_audit.get("candidate_pool_stage", "unknown")),
        "room_bounds_m": [
            float(v)
            for v in camera_seed_audit.get("room_bounds_m", [0.0, 0.0, 0.0, 0.0])
        ],
        "grid_resolution_m": float(camera_seed_audit.get("grid_resolution_m", 0.0)),
        "n_layout_objects": int(camera_seed_audit.get("n_layout_objects", 0)),
        "n_edge_candidates": int(candidate_pool_counts.get("edge_near", 0)),
        "n_relaxed_clearance_candidates": int(candidate_pool_counts.get("relaxed_clearance", 0)),
        "n_relaxed_minimal_candidates": int(candidate_pool_counts.get("relaxed_minimal", 0)),
        "n_candidate_xy_before_subsample": int(candidate_pool_counts.get("before_subsample", 0)),
        "n_candidate_xy_after_subsample": int(candidate_pool_counts.get("after_subsample", 0)),
        "max_candidate_budget": int(candidate_pool_counts.get("max_candidate_budget", 0)),
        "used_target_preselection": bool(camera_seed_audit.get("used_target_preselection", False)),
        "n_target_seed_attempted": int(target_seed_counts.get("attempted", 0)),
        "n_target_seed_accepted": int(target_seed_counts.get("accepted", 0)),
        "n_target_seed_selected": int(target_seed_counts.get("selected", 0)),
        "n_scored_candidates": int(selection_counts.get("scored_candidates", 0)),
        "n_selected_before_post_filters": int(selection_counts.get("selected_before_post_filters", 0)),
        "n_collision_filtered_out": int(selection_counts.get("collision_filtered_out", 0)),
        "n_nav_filtered_out": int(selection_counts.get("nav_filtered_out", 0)),
        "n_final_cameras": int(selection_counts.get("final_cameras", len(selected_records))),
        "selected_seed_source_counts": dict(sorted(source_counts.items())),
    }


def summarize_view_funnel_audit(
    *,
    scene_id: str,
    view_id: int,
    cam_info: dict,
    target_filter_counts: dict | None = None,
    targets: list[dict] | None = None,
    target_rejections: list[dict] | None = None,
    routing_candidates: list[dict] | None = None,
    routing_stage_counts: dict | None = None,
    support_summary: dict | None = None,
    camera_seed_record: dict | None = None,
    final_status: str,
) -> dict:
    target_filter_counts = target_filter_counts or {}
    targets = list(targets or [])
    target_rejections = list(target_rejections or [])
    routing_candidates = list(routing_candidates or [])
    routing_stage_counts = routing_stage_counts or {}
    support_summary = support_summary or {}
    camera_seed_record = camera_seed_record or {}

    rejection_reason_counts = Counter(str(item.get("reason", "unknown")) for item in target_rejections)
    axis_cells = sorted({
        _routing_axis_cell(candidate)
        for candidate in routing_candidates
        if candidate.get("navigation_axes")
    })
    instruction_winners = sum(
        1
        for candidate in routing_candidates
        if bool(((candidate.get("instruction_selection", {}) or {}).get("is_rule_winner", False)))
    )
    goal_zone_ok = int(routing_stage_counts.get("goal_zone_ok", 0))
    if goal_zone_ok <= 0:
        goal_zone_ok = sum(
            1
            for candidate in routing_candidates
            if bool(((candidate.get("invariants", {}) or {}).get("goal_in_target_zone", False)))
        )

    return {
        "record_type": "view_funnel",
        "scene_id": str(scene_id),
        "view_id": int(view_id),
        "camera_position": [float(v) for v in cam_info.get("position", [0.0, 0.0, 0.0])],
        "camera_look_angle": float(cam_info.get("look_angle", 0.0)),
        "final_status": str(final_status),
        "n_visible_objects": int(cam_info.get("visible_count", len(cam_info.get("visible_objects", [])))),
        "n_target_total": int(target_filter_counts.get("total", 0)),
        "n_dist_ok": int(target_filter_counts.get("dist", 0)),
        "n_raycast_ok": int(target_filter_counts.get("raycast", 0)),
        "n_bbox_ok": int(target_filter_counts.get("bbox", 0)),
        "n_size_ok": int(target_filter_counts.get("size", 0)),
        "n_visibility_ok": int(target_filter_counts.get("visibility_ok", 0)),
        "n_targets_selected": int(len(targets)),
        "n_pointmass_success": int(routing_stage_counts.get("pointmass_success", 0)),
        "n_embodied_success": int(routing_stage_counts.get("embodied_success", 0)),
        "n_goal_zone_ok": int(goal_zone_ok),
        "n_instruction_winners": int(instruction_winners),
        "n_budget_pruned": int(rejection_reason_counts.get("pruned_by_route_diversity_budget", 0)),
        "n_final_routes": int(len(routing_candidates)),
        "n_reference_supported": int(support_summary.get("reference_supported", 0)),
        "n_geometry_supported": int(support_summary.get("geometry_supported", 0)),
        "n_nontrivial_supported": int(support_summary.get("nontrivial_supported", 0)),
        "axis_cells_added": axis_cells,
        "selected_target_ids": [int(target.get("target_id", -1)) for target in targets],
        "final_target_ids": [int(candidate.get("target_id", -1)) for candidate in routing_candidates],
        "camera_seed_source": camera_seed_record.get("source"),
        "camera_seed_candidate_pool_stage": camera_seed_record.get("candidate_pool_stage"),
        "camera_seed_target_id": camera_seed_record.get("target_id"),
        "camera_seed_n_vis": int(camera_seed_record.get("n_vis", 0)),
        "camera_seed_path_score": float(camera_seed_record.get("path_score", 0.0)),
        "rejection_reason_counts": dict(sorted(rejection_reason_counts.items())),
    }


def write_scene_funnel_audit_records(
    funnel_audit_out: str | os.PathLike | None,
    *,
    scene_id: str,
    records: list[dict],
) -> None:
    if not funnel_audit_out:
        return
    out_dir = Path(funnel_audit_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{scene_id}.jsonl"
    with open(out_path, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def _turn_angles_deg(path_world: list[list[float]] | list[tuple[float, float, float]]) -> list[float]:
    angles: list[float] = []
    if len(path_world) < 3:
        return angles
    for idx in range(1, len(path_world) - 1):
        p0 = np.array(path_world[idx - 1][:2], dtype=np.float32)
        p1 = np.array(path_world[idx][:2], dtype=np.float32)
        p2 = np.array(path_world[idx + 1][:2], dtype=np.float32)
        v1 = p1 - p0
        v2 = p2 - p1
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cosang = max(-1.0, min(1.0, float(np.dot(v1, v2) / (n1 * n2))))
        angles.append(math.degrees(math.acos(cosang)))
    return angles


def _sample_path_xy(
    path_world: list[list[float]] | list[tuple[float, float, float]],
    *,
    step_m: float,
) -> list[tuple[float, float]]:
    if not path_world:
        return []
    if len(path_world) == 1:
        return [(float(path_world[0][0]), float(path_world[0][1]))]

    step = max(float(step_m), 1e-3)
    samples: list[tuple[float, float]] = []
    for idx in range(len(path_world) - 1):
        x0, y0 = float(path_world[idx][0]), float(path_world[idx][1])
        x1, y1 = float(path_world[idx + 1][0]), float(path_world[idx + 1][1])
        if idx == 0:
            samples.append((x0, y0))
        seg_len = math.hypot(x1 - x0, y1 - y0)
        num_steps = max(1, int(math.ceil(seg_len / step)))
        for step_idx in range(1, num_steps + 1):
            t = float(step_idx) / float(num_steps)
            samples.append((
                x0 + (x1 - x0) * t,
                y0 + (y1 - y0) * t,
            ))
    return samples


def _clearance_stats_along_path(
    path_world: list[list[float]] | list[tuple[float, float, float]],
    nav_grid,
    *,
    narrow_threshold_m: float = 0.2,
) -> dict[str, float]:
    if not path_world or nav_grid is None or not hasattr(nav_grid, "get_distance_map"):
        return {
            "min_clearance_m": 0.0,
            "mean_clearance_m": 0.0,
            "narrow_passage_fraction": 0.0,
            "start_zone_clearance_m": 0.0,
            "goal_zone_clearance_m": 0.0,
        }
    dist_map = nav_grid.get_distance_map()
    samples: list[float] = []
    sampled_points = _sample_path_xy(
        path_world,
        step_m=max(float(getattr(nav_grid, "resolution", 0.1)), 0.05),
    )
    for wx, wy in sampled_points:
        gx, gy = nav_grid.world_to_grid(float(wx), float(wy))
        if 0 <= gx < nav_grid.width and 0 <= gy < nav_grid.height:
            samples.append(float(dist_map[gy, gx]))
    if not samples:
        return {
            "min_clearance_m": 0.0,
            "mean_clearance_m": 0.0,
            "narrow_passage_fraction": 0.0,
            "start_zone_clearance_m": 0.0,
            "goal_zone_clearance_m": 0.0,
        }
    narrow_fraction = sum(value <= narrow_threshold_m for value in samples) / float(len(samples))
    return {
        "min_clearance_m": round(min(samples), 3),
        "mean_clearance_m": round(sum(samples) / float(len(samples)), 3),
        "narrow_passage_fraction": round(narrow_fraction, 3),
        "start_zone_clearance_m": round(samples[0], 3),
        "goal_zone_clearance_m": round(samples[-1], 3),
    }


def build_routing_complexity_metadata(
    *,
    dense_path_world: list[list[float]] | list[tuple[float, float, float]],
    sparse_path_world: list[list[float]] | list[tuple[float, float, float]],
    projected_dense_path: list[list[int] | None],
    nav_grid,
    bbox_metrics: dict | None,
    visibility_ratio: float,
    dense_path_projection_safe: bool,
    sparse_path_projection_safe: bool,
    start_in_bottom_band: bool,
) -> dict[str, float | int | bool]:
    raw_len = path_length_m(dense_path_world)
    sparse_len = path_length_m(sparse_path_world)
    if dense_path_world and len(dense_path_world) >= 2:
        start_goal_l2 = math.hypot(
            float(dense_path_world[-1][0]) - float(dense_path_world[0][0]),
            float(dense_path_world[-1][1]) - float(dense_path_world[0][1]),
        )
    else:
        start_goal_l2 = 0.0
    dense_angles = _turn_angles_deg(dense_path_world)
    sparse_angles = _turn_angles_deg(sparse_path_world)
    dense_turn_count = sum(angle >= 20.0 for angle in dense_angles)
    sparse_turn_count = sum(angle >= 20.0 for angle in sparse_angles)
    clearance_stats = _clearance_stats_along_path(dense_path_world, nav_grid)
    visible_points = sum(
        isinstance(point, list) and len(point) == 2 and point[0] is not None and point[1] is not None
        for point in projected_dense_path
    )
    path_visible_fraction = (
        float(visible_points) / float(len(projected_dense_path))
        if projected_dense_path
        else 0.0
    )
    return {
        "path_length_raw_m": round(raw_len, 3),
        "path_length_sparse_m": round(sparse_len, 3),
        "start_goal_l2_m": round(start_goal_l2, 3),
        "detour_ratio_raw": round(raw_len / max(start_goal_l2, 1e-6), 3) if raw_len > 0.0 else 0.0,
        "detour_ratio_sparse": round(sparse_len / max(start_goal_l2, 1e-6), 3) if sparse_len > 0.0 else 0.0,
        "path_smoothing_gain_m": round(max(0.0, raw_len - sparse_len), 3),
        "dense_turn_count": int(dense_turn_count),
        "sparse_turn_count": int(sparse_turn_count),
        "max_turn_angle_deg": round(max(dense_angles) if dense_angles else 0.0, 3),
        "mean_turn_angle_deg": round(sum(dense_angles) / len(dense_angles), 3) if dense_angles else 0.0,
        "num_segments_sparse": max(0, len(sparse_path_world) - 1),
        "num_dense_points": len(dense_path_world),
        "target_bbox_area_px": int((bbox_metrics or {}).get("area_px", 0)),
        "target_bbox_short_side_px": int((bbox_metrics or {}).get("short_side_px", 0)),
        "target_visible_surface_ratio": round(float(visibility_ratio), 3),
        "path_visible_point_fraction": round(path_visible_fraction, 3),
        "path_projection_safe": bool(sparse_path_projection_safe),
        "dense_path_projection_safe": bool(dense_path_projection_safe),
        "start_in_bottom_band": bool(start_in_bottom_band),
        **clearance_stats,
    }


def _path_overlap_ratio(
    path_a: list[list[float]] | list[tuple[float, float, float]],
    path_b: list[list[float]] | list[tuple[float, float, float]],
    *,
    cell_size_m: float = 0.25,
) -> float:
    def _cells(path):
        cells = set()
        for wx, wy in _sample_path_xy(path, step_m=max(float(cell_size_m) * 0.5, 0.05)):
            cells.add(
                (
                    int(round(float(wx) / cell_size_m)),
                    int(round(float(wy) / cell_size_m)),
                )
            )
        return cells

    cells_a = _cells(path_a)
    cells_b = _cells(path_b)
    if not cells_a or not cells_b:
        return 0.0
    return float(len(cells_a & cells_b)) / float(len(cells_a | cells_b))


def _sampled_path_world(
    path_world: list[list[float]] | list[tuple[float, float, float]],
    *,
    step_m: float = 0.05,
) -> list[list[float]]:
    sampled_xy = _sample_path_xy(path_world, step_m=max(float(step_m), 0.02))
    return [[float(x), float(y), 0.0] for x, y in sampled_xy]


def _path_prefix_outside_goal_zone(
    path_world: list[list[float]] | list[tuple[float, float, float]],
    *,
    bbox: list[float] | None,
    goal_radius_m: float,
    step_m: float = 0.05,
) -> list[list[float]]:
    sampled = _sampled_path_world(path_world, step_m=step_m)
    if len(sampled) <= 2 or not bbox or len(bbox) < 5:
        return sampled
    prefix: list[list[float]] = []
    for point in sampled:
        if distance_to_oriented_bbox_2d(float(point[0]), float(point[1]), bbox) <= float(goal_radius_m):
            break
        prefix.append(point)
    if len(prefix) >= 2:
        return prefix
    return sampled[:2]


def _turn_count_for_path(
    path_world: list[list[float]] | list[tuple[float, float, float]],
    *,
    threshold_deg: float = 20.0,
) -> int:
    return int(sum(angle >= float(threshold_deg) for angle in _turn_angles_deg(path_world)))


def merge_point_and_embodied_routing_complexity(
    *,
    point_complexity: dict[str, float | int | bool],
    embodied_complexity: dict[str, float | int | bool],
    point_dense_path_world: list[list[float]] | list[tuple[float, float, float]],
    embodied_dense_path_world: list[list[float]] | list[tuple[float, float, float]],
    target_bbox: list[float] | None = None,
    goal_distance_threshold_m: float = 0.0,
    embodied_nav_grid=None,
) -> dict[str, float | int | bool]:
    point_route_burden_path = _path_prefix_outside_goal_zone(
        point_dense_path_world,
        bbox=target_bbox,
        goal_radius_m=float(goal_distance_threshold_m),
    )
    embodied_route_burden_path = _path_prefix_outside_goal_zone(
        embodied_dense_path_world,
        bbox=target_bbox,
        goal_radius_m=float(goal_distance_threshold_m),
    )
    burden_clearance_stats = _clearance_stats_along_path(embodied_route_burden_path, embodied_nav_grid)
    route_burden_overlap = _path_overlap_ratio(point_route_burden_path, embodied_route_burden_path)
    route_burden_extra_length = max(
        0.0,
        path_length_m(embodied_route_burden_path) - path_length_m(point_route_burden_path),
    )
    route_burden_extra_decisions = max(
        0,
        _turn_count_for_path(embodied_route_burden_path) - _turn_count_for_path(point_route_burden_path),
    )
    merged = dict(point_complexity)
    merged.update({
        "point_path_projection_safe": bool(point_complexity.get("path_projection_safe", False)),
        "point_dense_path_projection_safe": bool(point_complexity.get("dense_path_projection_safe", False)),
        "embodied_path_length_raw_m": round(float(embodied_complexity.get("path_length_raw_m", 0.0)), 3),
        "embodied_path_length_sparse_m": round(float(embodied_complexity.get("path_length_sparse_m", 0.0)), 3),
        "embodied_start_goal_l2_m": round(float(embodied_complexity.get("start_goal_l2_m", 0.0)), 3),
        "embodied_detour_ratio_raw": round(float(embodied_complexity.get("detour_ratio_raw", 0.0)), 3),
        "embodied_detour_ratio_sparse": round(float(embodied_complexity.get("detour_ratio_sparse", 0.0)), 3),
        "embodied_path_smoothing_gain_m": round(float(embodied_complexity.get("path_smoothing_gain_m", 0.0)), 3),
        "embodied_dense_turn_count": int(embodied_complexity.get("dense_turn_count", 0)),
        "embodied_sparse_turn_count": int(embodied_complexity.get("sparse_turn_count", 0)),
        "embodied_max_turn_angle_deg": round(float(embodied_complexity.get("max_turn_angle_deg", 0.0)), 3),
        "embodied_mean_turn_angle_deg": round(float(embodied_complexity.get("mean_turn_angle_deg", 0.0)), 3),
        "embodied_num_segments_sparse": int(embodied_complexity.get("num_segments_sparse", 0)),
        "embodied_num_dense_points": int(embodied_complexity.get("num_dense_points", 0)),
        "embodied_path_visible_point_fraction": round(float(embodied_complexity.get("path_visible_point_fraction", 0.0)), 3),
        "embodied_path_projection_safe": bool(embodied_complexity.get("path_projection_safe", False)),
        "embodied_dense_path_projection_safe": bool(embodied_complexity.get("dense_path_projection_safe", False)),
        "embodied_min_clearance_m": round(float(embodied_complexity.get("min_clearance_m", 0.0)), 3),
        "embodied_mean_clearance_m": round(float(embodied_complexity.get("mean_clearance_m", 0.0)), 3),
        "embodied_narrow_passage_fraction": round(float(embodied_complexity.get("narrow_passage_fraction", 0.0)), 3),
        "embodied_start_zone_clearance_m": round(float(embodied_complexity.get("start_zone_clearance_m", 0.0)), 3),
        "embodied_goal_zone_clearance_m": round(float(embodied_complexity.get("goal_zone_clearance_m", 0.0)), 3),
        "path_overlap_point_vs_embodied": round(
            _path_overlap_ratio(point_dense_path_world, embodied_dense_path_world),
            3,
        ),
        "embodied_extra_length_m": round(
            max(
                0.0,
                float(embodied_complexity.get("path_length_raw_m", 0.0))
                - float(point_complexity.get("path_length_raw_m", 0.0)),
            ),
            3,
        ),
        "embodied_extra_decisions": max(
            0,
            int(embodied_complexity.get("num_segments_sparse", 0))
            - int(point_complexity.get("num_segments_sparse", 0)),
        ),
        "embodied_route_burden_overlap": round(route_burden_overlap, 3),
        "embodied_route_burden_extra_length_m": round(route_burden_extra_length, 3),
        "embodied_route_burden_extra_decisions": int(route_burden_extra_decisions),
        "embodied_route_burden_narrow_passage_fraction": round(
            float(burden_clearance_stats.get("narrow_passage_fraction", 0.0)),
            3,
        ),
        "terminal_goal_clearance_m": round(
            float(embodied_complexity.get("goal_zone_clearance_m", 0.0)),
            3,
        ),
    })
    merged["path_projection_safe"] = bool(
        point_complexity.get("path_projection_safe", False)
        and embodied_complexity.get("path_projection_safe", False)
    )
    merged["dense_path_projection_safe"] = bool(
        point_complexity.get("dense_path_projection_safe", False)
        and embodied_complexity.get("dense_path_projection_safe", False)
    )
    return merged


def enrich_routing_complexity_metadata(candidates: list[dict]) -> list[dict]:
    enriched: list[dict] = [dict(candidate) for candidate in candidates]
    total_visible = len(enriched)
    for idx, candidate in enumerate(enriched):
        identity = candidate.get("instruction_identity", {})
        selection = candidate.get("instruction_selection", {})
        semantic_group = identity.get("reference_family_id") or identity.get("semantic_group_id")
        color_name = identity.get("reference_color_name")
        same_group = [
            other for other in enriched
            if (
                (other.get("instruction_identity", {}) or {}).get("reference_family_id")
                or (other.get("instruction_identity", {}) or {}).get("semantic_group_id")
            ) == semantic_group
        ]
        same_group_same_color = [
            other for other in same_group
            if (
                (other.get("instruction_identity", {}) or {}).get("reference_color_name")
            ) == color_name
        ]
        cohort = sorted(
            same_group_same_color,
            key=lambda item: (
                float(item.get("distance_m", 1e9)),
                int(item.get("target_id", -1)),
            ),
        )
        family_cohort = sorted(
            same_group,
            key=lambda item: (
                float(item.get("distance_m", 1e9)),
                int(item.get("target_id", -1)),
            ),
        )
        distance_gap = 0.0
        family_distance_gap = 999.0
        current_target = int(candidate.get("target_id", -1))
        current_distance = float(candidate.get("distance_m", candidate.get("distance", 1e9)))
        for pos, other in enumerate(cohort):
            if int(other.get("target_id", -1)) != current_target:
                continue
            if pos + 1 < len(cohort):
                distance_gap = round(
                    float(cohort[pos + 1].get("distance_m", 0.0)) - float(other.get("distance_m", 0.0)),
                    3,
                )
            break
        family_gap_candidates = [
            abs(float(other.get("distance_m", other.get("distance", 1e9))) - current_distance)
            for other in family_cohort
            if int(other.get("target_id", -1)) != current_target
        ]
        if family_gap_candidates:
            family_distance_gap = round(min(family_gap_candidates), 3)
        routing_complexity = dict(candidate.get("routing_complexity", {}))
        routing_complexity.update({
            "num_visible_routing_candidates": total_visible,
            "instruction_cohort_size": int(selection.get("cohort_size", len(cohort))),
            "distance_rank_in_cohort": int(selection.get("distance_rank", 0)),
            "distance_gap_to_next_m": distance_gap,
            "family_distance_gap_to_next_m": family_distance_gap,
            "family_distance_gap_to_nearest_m": family_distance_gap,
            "is_unique_by_color_and_distance": bool(selection.get("selection_status") == "unique_winner"),
            "num_same_group_diff_color": max(0, len(same_group) - len(same_group_same_color)),
            "num_same_group_same_color": max(0, len(same_group_same_color) - 1),
            "synonym_competitor_count": max(0, len(same_group) - 1),
        })
        candidate["routing_complexity"] = routing_complexity
    return enriched


def _build_navigation_validity(candidate: dict) -> dict:
    routing_complexity = candidate.get("routing_complexity", {}) or {}
    invariants = candidate.get("invariants", {}) or {}
    check_results = {
        "start_in_bottom_band": bool(routing_complexity.get("start_in_bottom_band", False)),
        "path_projection_safe": bool(routing_complexity.get("path_projection_safe", False)),
        "dense_path_projection_safe": bool(routing_complexity.get("dense_path_projection_safe", False)),
        "start_in_forward_sector": bool(invariants.get("start_in_forward_sector", False)),
        "goal_in_target_zone": bool(invariants.get("goal_in_target_zone", False)),
        "embodied_collision_free": bool(invariants.get("embodied_collision_free", False)),
    }
    failed_checks = [key for key, passed in check_results.items() if not passed]
    return {
        "eligible": not failed_checks,
        "failed_checks": failed_checks,
        "check_results": check_results,
    }


def _build_navigation_reference_facts(candidate: dict) -> dict[str, float | int | str]:
    routing_complexity = candidate.get("routing_complexity", {}) or {}
    selection = candidate.get("instruction_selection", {}) or {}
    same_group_diff_color = int(routing_complexity.get("num_same_group_diff_color", 0))
    same_group_same_color = int(routing_complexity.get("num_same_group_same_color", 0))
    cohort_size = int(selection.get("cohort_size", routing_complexity.get("instruction_cohort_size", 1)))
    if same_group_same_color > 0 and same_group_diff_color > 0:
        resolution_mode = "color_and_distance"
    elif same_group_same_color > 0:
        resolution_mode = "distance_only"
    elif same_group_diff_color > 0:
        resolution_mode = "color_only"
    else:
        resolution_mode = "category_only"

    if same_group_same_color > 0 or cohort_size > 1:
        top2_gap_m = float(routing_complexity.get("distance_gap_to_next_m", 0.0))
    elif same_group_diff_color > 0:
        top2_gap_m = float(
            routing_complexity.get(
                "family_distance_gap_to_nearest_m",
                routing_complexity.get("family_distance_gap_to_next_m", 999.0),
            )
        )
    else:
        top2_gap_m = 999.0

    return {
        "reference_resolution_mode": resolution_mode,
        "same_semantic_group_count_total": 1 + same_group_diff_color + same_group_same_color,
        "same_semantic_group_same_color_count": same_group_same_color,
        "same_semantic_group_diff_color_count": same_group_diff_color,
        "reference_top2_gap_m": round(top2_gap_m, 3),
        "synonym_competitor_count": int(routing_complexity.get("synonym_competitor_count", 0)),
        "instruction_cohort_size": cohort_size,
    }


def _build_navigation_geometry_facts(candidate: dict) -> dict[str, float | int]:
    routing_complexity = candidate.get("routing_complexity", {}) or {}
    num_segments_sparse = int(routing_complexity.get("num_segments_sparse", 0))
    path_length = round(float(routing_complexity.get("path_length_raw_m", 0.0)), 3)
    detour_ratio = round(float(routing_complexity.get("detour_ratio_raw", 0.0)), 3)
    turn_count = int(routing_complexity.get("dense_turn_count", 0))
    decision_count = max(0, num_segments_sparse - 1)
    if turn_count >= 2 or decision_count >= 2 or detour_ratio > 1.20:
        geometry_profile = "multi_turn_or_detour"
    elif turn_count >= 1 or decision_count >= 1:
        geometry_profile = "single_turn"
    elif path_length > 4.5:
        geometry_profile = "straight_far"
    else:
        geometry_profile = "straight_short"
    return {
        "path_length_point_m": path_length,
        "detour_ratio_point": detour_ratio,
        "turn_count_point": turn_count,
        "decision_count_point": decision_count,
        "geometry_profile": geometry_profile,
    }


def _build_navigation_embodiment_facts(candidate: dict) -> dict[str, float | int]:
    routing_complexity = candidate.get("routing_complexity", {}) or {}
    num_segments_sparse = int(
        routing_complexity.get("embodied_num_segments_sparse", routing_complexity.get("num_segments_sparse", 0))
    )
    min_clearance = float(
        routing_complexity.get("embodied_min_clearance_m", routing_complexity.get("min_clearance_m", 0.0))
    )
    goal_clearance = float(
        routing_complexity.get("embodied_goal_zone_clearance_m", routing_complexity.get("goal_zone_clearance_m", 0.0))
    )
    raw_length = float(
        routing_complexity.get("embodied_path_length_raw_m", routing_complexity.get("path_length_raw_m", 0.0))
    )
    detour_ratio = float(
        routing_complexity.get("embodied_detour_ratio_raw", routing_complexity.get("detour_ratio_raw", 0.0))
    )
    route_burden_overlap = float(
        routing_complexity.get(
            "embodied_route_burden_overlap",
            routing_complexity.get("path_overlap_point_vs_embodied", 1.0),
        )
    )
    route_burden_extra_length = float(
        routing_complexity.get(
            "embodied_route_burden_extra_length_m",
            routing_complexity.get("embodied_extra_length_m", 0.0),
        )
    )
    route_burden_extra_decisions = int(
        routing_complexity.get(
            "embodied_route_burden_extra_decisions",
            routing_complexity.get("embodied_extra_decisions", 0),
        )
    )
    route_burden_narrow_fraction = float(
        routing_complexity.get(
            "embodied_route_burden_narrow_passage_fraction",
            routing_complexity.get(
                "embodied_narrow_passage_fraction",
                routing_complexity.get("narrow_passage_fraction", 0.0),
            ),
        )
    )
    terminal_goal_clearance = float(
        routing_complexity.get(
            "terminal_goal_clearance_m",
            routing_complexity.get("embodied_goal_zone_clearance_m", goal_clearance),
        )
    )
    terminal_goal_tightness_only = (
        terminal_goal_clearance < 0.05
        and route_burden_overlap >= 0.90
        and route_burden_narrow_fraction <= 0.12
        and route_burden_extra_length <= 0.30
        and route_burden_extra_decisions <= 0
    )
    return {
        "path_length_embodied_m": round(raw_length, 3),
        "detour_ratio_embodied": round(detour_ratio, 3),
        "turn_count_embodied": int(
            routing_complexity.get("embodied_dense_turn_count", routing_complexity.get("dense_turn_count", 0))
        ),
        "decision_count_embodied": max(0, num_segments_sparse - 1),
        "path_overlap_point_vs_embodied": round(float(routing_complexity.get("path_overlap_point_vs_embodied", 1.0)), 3),
        "embodied_extra_length_m": round(float(routing_complexity.get("embodied_extra_length_m", 0.0)), 3),
        "embodied_extra_decisions": int(routing_complexity.get("embodied_extra_decisions", 0)),
        "embodied_route_burden_overlap": round(route_burden_overlap, 3),
        "embodied_route_burden_extra_length_m": round(route_burden_extra_length, 3),
        "embodied_route_burden_extra_decisions": int(route_burden_extra_decisions),
        "embodied_clearance_margin_min_m": round(min_clearance, 3),
        "embodied_clearance_margin_goal_m": round(goal_clearance, 3),
        "narrow_passage_fraction": round(
            float(
                routing_complexity.get(
                    "embodied_narrow_passage_fraction",
                    routing_complexity.get("narrow_passage_fraction", 0.0),
                )
            ),
            3,
        ),
        "embodied_route_burden_narrow_fraction": round(route_burden_narrow_fraction, 3),
        "terminal_goal_clearance_m": round(terminal_goal_clearance, 3),
        "terminal_goal_tightness_only": bool(terminal_goal_tightness_only),
    }


def enrich_navigation_fact_metadata(candidates: list[dict]) -> list[dict]:
    enriched: list[dict] = [dict(candidate) for candidate in candidates]
    for candidate in enriched:
        navigation_validity = _build_navigation_validity(candidate)
        reference_facts = _build_navigation_reference_facts(candidate)
        geometry_facts = _build_navigation_geometry_facts(candidate)
        embodiment_facts = _build_navigation_embodiment_facts(candidate)
        reference_axis = classify_reference_axis(reference_facts)
        geometry_axis = classify_geometry_axis(geometry_facts)
        embodiment_axis = classify_embodiment_axis(embodiment_facts)
        candidate["navigation_validity"] = navigation_validity
        candidate["navigation_reference_facts"] = reference_facts
        candidate["navigation_geometry_facts"] = geometry_facts
        candidate["navigation_embodiment_facts"] = embodiment_facts
        candidate["navigation_axes"] = {
            "reference_axis": reference_axis,
            "geometry_axis": geometry_axis,
            "embodiment_axis": embodiment_axis,
        }
        candidate["navigation_tier"] = classify_navigation_tier(
            reference_axis=reference_axis,
            geometry_axis=geometry_axis,
            embodiment_axis=embodiment_axis,
        )
    return enriched


def _candidate_nn_distances_px(candidates: list[dict]) -> list[float]:
    points = []
    for candidate in candidates:
        image_xy = candidate.get("image_xy")
        if not isinstance(image_xy, list) or len(image_xy) != 2:
            continue
        points.append((float(image_xy[0]), float(image_xy[1])))
    if len(points) < 2:
        return []

    nn_distances = []
    for idx, (x0, y0) in enumerate(points):
        best = None
        for jdx, (x1, y1) in enumerate(points):
            if idx == jdx:
                continue
            dist = math.hypot(x1 - x0, y1 - y0)
            if best is None or dist < best:
                best = dist
        if best is not None:
            nn_distances.append(float(best))
    return nn_distances


def build_affordance_fact_metadata(candidates: list[dict]) -> dict[str, dict | str]:
    return shared_build_affordance_fact_metadata(candidates)


def select_diverse_routing_candidates(candidates: list[dict], max_candidates: int) -> list[dict]:
    if max_candidates <= 0 or len(candidates) <= max_candidates:
        return list(candidates)

    pool = [dict(candidate) for candidate in candidates]
    selected: list[dict] = []
    selected_ids: set[int] = set()
    while len(selected) < max_candidates and len(selected_ids) < len(pool):
        best = None
        best_score = -1e18
        selected_reference_profiles = {
            str(((other.get("navigation_reference_facts", {}) or {}).get("reference_resolution_mode", "category_only")))
            for other in selected
        }
        selected_geometry_profiles = {
            str(((other.get("navigation_geometry_facts", {}) or {}).get("geometry_profile", "straight_short")))
            for other in selected
        }
        selected_coverage_keys = {
            (
                ((other.get("instruction_identity", {}) or {}).get("reference_family_id")
                 or (other.get("instruction_identity", {}) or {}).get("semantic_group_id")),
                ((other.get("instruction_identity", {}) or {}).get("reference_color_name")),
                str(((other.get("navigation_reference_facts", {}) or {}).get("reference_resolution_mode", "category_only"))),
            )
            for other in selected
        }
        selected_support = routing_support_summary(selected)
        for candidate in pool:
            target_id = int(candidate.get("target_id", -1))
            if target_id in selected_ids:
                continue
            identity = candidate.get("instruction_identity", {}) or {}
            identity_key = (
                identity.get("reference_family_id") or identity.get("semantic_group_id"),
                identity.get("reference_color_name"),
            )
            reference_facts = candidate.get("navigation_reference_facts", {}) or {}
            geometry_facts = candidate.get("navigation_geometry_facts", {}) or {}
            reference_profile = str(reference_facts.get("reference_resolution_mode", "category_only"))
            geometry_profile = str(geometry_facts.get("geometry_profile", "straight_short"))
            coverage_key = (identity_key[0], identity_key[1], reference_profile)
            routing_complexity = candidate.get("routing_complexity", {}) or {}
            base_score = (
                float(routing_complexity.get("detour_ratio_raw", 1.0)) * 3.0
                + float(routing_complexity.get("dense_turn_count", 0.0)) * 0.8
                + float(routing_complexity.get("path_length_raw_m", 0.0)) * 0.2
            )
            identity_penalty = 0.0
            overlap_penalty = 0.0
            coverage_bonus = 0.0
            has_reference_support = _candidate_has_reference_support(candidate)
            has_geometry_support = _candidate_has_geometry_support(candidate)
            if selected:
                if any(
                    (
                        ((other.get("instruction_identity", {}) or {}).get("reference_family_id")
                         or (other.get("instruction_identity", {}) or {}).get("semantic_group_id")),
                        ((other.get("instruction_identity", {}) or {}).get("reference_color_name")),
                    ) == identity_key
                    for other in selected
                ):
                    identity_penalty = 2.0
                overlap_penalty = max(
                    _path_overlap_ratio(
                        candidate.get("debug_dense_path_world", []),
                        other.get("debug_dense_path_world", []),
                    )
                    for other in selected
                )
            if reference_profile not in selected_reference_profiles:
                coverage_bonus += 4.0
            if geometry_profile not in selected_geometry_profiles:
                coverage_bonus += 3.0
            if coverage_key not in selected_coverage_keys:
                coverage_bonus += 1.5
            if has_reference_support and selected_support["reference_supported"] <= 0:
                coverage_bonus += 6.0
            if has_geometry_support and selected_support["geometry_supported"] <= 0:
                coverage_bonus += 5.0
            if has_reference_support or has_geometry_support:
                coverage_bonus += 2.0
            else:
                coverage_bonus -= 1.5
            score = base_score + coverage_bonus - identity_penalty - overlap_penalty
            if score > best_score:
                best = candidate
                best_score = score
        if best is None:
            break
        selected_ids.add(int(best.get("target_id", -1)))
        selected.append(best)

    return selected


def distance_to_oriented_bbox_2d(px: float, py: float, bbox: list[float]) -> float:
    if len(bbox) < 5:
        return float("inf")
    cx, cy = float(bbox[0]), float(bbox[1])
    sx, sy = float(bbox[3]), float(bbox[4])
    rz = float(bbox[8]) if len(bbox) > 8 else (float(bbox[6]) if len(bbox) > 6 else 0.0)
    dx = px - cx
    dy = py - cy
    cosr = math.cos(-rz)
    sinr = math.sin(-rz)
    lx = dx * cosr - dy * sinr
    ly = dx * sinr + dy * cosr
    hx = sx / 2.0
    hy = sy / 2.0
    ddx = max(abs(lx) - hx, 0.0)
    ddy = max(abs(ly) - hy, 0.0)
    return math.hypot(ddx, ddy)


def map_path_world_to_candidate_ids(
    path_world: list[list[float]] | list[tuple[float, float, float]],
    candidates: list[dict],
    feasible_key: str,
    sample_step_m: float = 0.25,
    max_match_dist_m: float = 0.9,
) -> list[int]:
    if not path_world or not candidates:
        return []

    nodes = []
    for cand in candidates:
        if feasible_key and not bool(cand.get(feasible_key, False)):
            continue
        w = cand.get("world_xyz", [0.0, 0.0, 0.0])
        nodes.append((
            int(cand.get("waypoint_id", -1)),
            float(w[0]),
            float(w[1]),
        ))
    if not nodes:
        return []

    samples = sample_path_points([(float(p[0]), float(p[1]), float(p[2])) for p in path_world], sample_step_m)
    if not samples:
        samples = [(float(path_world[0][0]), float(path_world[0][1]), 0.0),
                   (float(path_world[-1][0]), float(path_world[-1][1]), 0.0)]

    ids: list[int] = []

    def _append_if_new(wp_id: int) -> None:
        if wp_id < 0:
            return
        if not ids or ids[-1] != wp_id:
            ids.append(wp_id)

    def _best_node(sx: float, sy: float):
        best_id = -1
        best_d = float("inf")
        for wp_id, wx, wy in nodes:
            d = math.hypot(wx - sx, wy - sy)
            if d < best_d:
                best_d = d
                best_id = wp_id
        return best_id, best_d

    for s in samples:
        sx, sy = float(s[0]), float(s[1])
        best_id, best_d = _best_node(sx, sy)
        if best_id >= 0 and best_d <= max_match_dist_m:
            _append_if_new(best_id)

    # Anchor both endpoints while preserving path order.
    start_x, start_y = float(path_world[0][0]), float(path_world[0][1])
    start_id, _ = _best_node(start_x, start_y)
    if start_id >= 0 and (not ids or ids[0] != start_id):
        ids.insert(0, start_id)

    end_x, end_y = float(path_world[-1][0]), float(path_world[-1][1])
    end_id, _ = _best_node(end_x, end_y)
    if end_id >= 0 and (not ids or ids[-1] != end_id):
        ids.append(end_id)

    return ids


def collect_dense_debug_waypoint_ids(
    path_world: list[list[float]] | list[tuple[float, float, float]],
    classification_candidates: list[dict],
    obstacle_mask,
    pad_px: int,
) -> tuple[list[int], list[int]]:
    """Map dense world path samples onto visible candidates for debug outputs."""
    waypoint_lookup = {
        int(candidate.get("waypoint_id", -1)): candidate
        for candidate in classification_candidates
    }
    point_ids = map_path_world_to_candidate_ids(
        path_world=path_world,
        candidates=classification_candidates,
        feasible_key="pointmass_walkable",
        sample_step_m=0.25,
        max_match_dist_m=0.9,
    )
    embodied_ids = map_path_world_to_candidate_ids(
        path_world=path_world,
        candidates=classification_candidates,
        feasible_key="embodied_feasible",
        sample_step_m=0.25,
        max_match_dist_m=0.9,
    )
    point_ids = normalize_candidate_path_ids_for_projection(
        point_ids,
        waypoint_lookup,
        obstacle_mask=obstacle_mask,
        pad_px=pad_px,
    )
    embodied_ids = normalize_candidate_path_ids_for_projection(
        embodied_ids,
        waypoint_lookup,
        obstacle_mask=obstacle_mask,
        pad_px=pad_px,
    )
    return point_ids, embodied_ids


def check_embodied_collision_free(path_world: list[list[float]], nav_grid) -> bool:
    if not path_world or nav_grid is None:
        return False
    if not hasattr(nav_grid, "is_path_traversable"):
        return True
    for i in range(len(path_world) - 1):
        p0 = (float(path_world[i][0]), float(path_world[i][1]))
        p1 = (float(path_world[i + 1][0]), float(path_world[i + 1][1]))
        ok = nav_grid.is_path_traversable([p0, p1])
        if not bool(ok.get("traversable", False)):
            return False
    return True


def apply_candidate_pixel_deobjectification(
    candidates: list[dict],
    object_index_img,
    object_index_pass_ids: set[int],
    nonblocking_pass_ids: set[int] | None = None,
    structural_pass_ids: set[int] | None = None,
    depth_map=None,
    depth_eps_m: float = 0.08,
    depth_window_radius_px: int = 2,
) -> int:
    import numpy as _np
    nonblocking_pass_ids = nonblocking_pass_ids or set()
    structural_pass_ids = structural_pass_ids or set()
    if object_index_img is None:
        return 0
    h, w = object_index_img.shape[:2]
    changed = 0
    for cand in candidates:
        xy = cand.get("image_xy")
        if not isinstance(xy, list) or len(xy) != 2:
            cand["on_object_pixel"] = False
            cand["depth_foreground_occluded"] = False
            continue
        px, py = int(xy[0]), int(xy[1])
        if not (0 <= px < w and 0 <= py < h):
            cand["on_object_pixel"] = False
            cand["on_structural_pixel"] = False
            continue
        pixel_id = int(object_index_img[py, px])
        on_object = pixel_id in object_index_pass_ids and pixel_id not in nonblocking_pass_ids
        on_structural = pixel_id in structural_pass_ids
        depth_occ = False
        cand_depth = float(cand.get("depth", 0.0))
        if depth_map is not None and cand_depth > 0.0:
            y0 = max(0, py - int(depth_window_radius_px))
            y1 = min(h, py + int(depth_window_radius_px) + 1)
            x0 = max(0, px - int(depth_window_radius_px))
            x1 = min(w, px + int(depth_window_radius_px) + 1)
            patch = depth_map[y0:y1, x0:x1]
            valid = _np.isfinite(patch) & (patch > 0.0)
            if _np.any(valid):
                nearest = float(_np.min(patch[valid]))
                depth_occ = nearest + float(depth_eps_m) < cand_depth
        cand["on_object_pixel"] = bool(on_object)
        cand["on_structural_pixel"] = bool(on_structural)
        cand["depth_foreground_occluded"] = bool(depth_occ)
        if (on_object or depth_occ) and (bool(cand.get("pointmass_walkable", False)) or bool(cand.get("embodied_feasible", False))):
            cand["pointmass_walkable"] = False
            cand["embodied_feasible"] = False
            changed += 1
    return changed


def infer_structural_pass_ids(object_index_img, object_index_pass_ids: set[int]) -> set[int]:
    if object_index_img is None:
        return set()
    import numpy as _np
    ids = {int(v) for v in _np.unique(object_index_img)}
    return {pid for pid in ids if pid > 0 and pid not in object_index_pass_ids}


def load_depth_map_metric(depth_path: str | None):
    if not depth_path:
        return None
    candidates = [Path(depth_path)]
    if depth_path.endswith(".png"):
        candidates.append(Path(depth_path.replace(".png", "_0001.png")))
    import numpy as _np
    for p in candidates:
        if not p.exists():
            continue
        try:
            arr = _np.array(Image.open(p)).astype(_np.float32)
            if arr.ndim == 3:
                arr = arr[..., 0]
            if arr.max() <= 1.0:
                return arr * 20.0
            return arr / 65535.0 * 20.0
        except Exception:
            continue
    return None


def force_include_pathworld_coverage_candidates(
    candidates: list[dict],
    target_candidates: list[dict],
    cam_obj,
    vfov_half: float,
    hfov_half: float,
    res_x: int,
    res_y: int,
    grid_point,
    grid_embodied,
    object_index_img,
    object_index_pass_ids: set[int],
    structural_pass_ids: set[int] | None = None,
    depth_map=None,
    depth_eps_m: float = 0.08,
    sample_step_m: float = 0.25,
    max_pixel_merge_dist: float = 8.0,
    max_world_merge_dist: float = 0.18,
) -> int:
    import numpy as _np
    if not candidates or not target_candidates or grid_point is None or grid_embodied is None:
        return 0

    next_waypoint_id = max(int(c.get("waypoint_id", -1)) for c in candidates) + 1
    cam_inv = cam_obj.matrix_world.inverted()

    def _exists_near(wx: float, wy: float, px: int, py: int) -> bool:
        for c in candidates:
            w = c.get("world_xyz", [0.0, 0.0, 0.0])
            cx, cy = float(w[0]), float(w[1])
            if math.hypot(cx - wx, cy - wy) <= max_world_merge_dist:
                return True
            cxy = c.get("image_xy")
            if isinstance(cxy, list) and len(cxy) == 2:
                if math.hypot(int(cxy[0]) - px, int(cxy[1]) - py) <= max_pixel_merge_dist:
                    return True
        return False

    structural_pass_ids = structural_pass_ids or set()
    added = 0
    h = object_index_img.shape[0] if object_index_img is not None else -1
    w = object_index_img.shape[1] if object_index_img is not None else -1

    for tgt in target_candidates:
        pw = tgt.get("path_world", [])
        if not pw:
            continue
        samples = sample_path_points([(float(p[0]), float(p[1]), float(p[2])) for p in pw], sample_step_m)
        if not samples:
            samples = [(float(pw[0][0]), float(pw[0][1]), float(pw[0][2]))]
        if len(pw) >= 2:
            samples.append((float(pw[-1][0]), float(pw[-1][1]), float(pw[-1][2])))

        for sx, sy, sz in samples:
            proj = project_world_point(
                cam_inv,
                mathutils.Vector((sx, sy, sz)),
                vfov_half,
                hfov_half,
                res_x,
                res_y,
                cam_obj=cam_obj,
                scene=bpy.context.scene,
            )
            if proj is None:
                continue
            px, py = int(proj["pixel_xy"][0]), int(proj["pixel_xy"][1])
            if not (0 <= px < res_x and 0 <= py < res_y):
                continue
            if object_index_img is not None and object_index_pass_ids and 0 <= py < h and 0 <= px < w:
                pixel_id = int(object_index_img[py, px])
                if pixel_id in object_index_pass_ids:
                    continue
            if depth_map is not None and object_index_img is not None and 0 <= py < h and 0 <= px < w and int(object_index_img[py, px]) > 0:
                cand_depth = float(proj.get("depth", 0.0))
                if cand_depth > 0.0:
                    y0 = max(0, py - 2)
                    y1 = min(res_y, py + 3)
                    x0 = max(0, px - 2)
                    x1 = min(res_x, px + 3)
                    patch = depth_map[y0:y1, x0:x1]
                    valid = _np.isfinite(patch) & (patch > 0.0)
                    if _np.any(valid):
                        nearest = float(_np.min(patch[valid]))
                        if nearest + float(depth_eps_m) < cand_depth:
                            continue
            if _exists_near(sx, sy, px, py):
                continue

            gx, gy = grid_point.world_to_grid(sx, sy)
            point_ok = bool(grid_point.is_free(gx, gy))
            emb_ok = bool(grid_embodied.is_free(gx, gy))
            if not point_ok and not emb_ok:
                continue

            candidates.append({
                "waypoint_id": int(next_waypoint_id),
                "display_id": 0,
                "world_xyz": [round(float(sx), 3), round(float(sy), 3), 0.0],
                "grid_xy": [int(gx), int(gy)],
                "image_xy": [int(px), int(py)],
                "screen_xy_norm": [round(float(proj["screen_xy"][0]), 6), round(float(proj["screen_xy"][1]), 6)],
                "depth": round(float(proj["depth"]), 4),
                "pointmass_walkable": point_ok,
                "embodied_feasible": emb_ok,
                "raycast_visible": True,
                "candidate_source": "path_world_forced",
                "is_path_coverage_point": True,
                "on_object_pixel": False,
            })
            next_waypoint_id += 1
            added += 1

    candidates.sort(key=lambda c: float(c.get("depth", 1e9)))
    for idx, cand in enumerate(candidates, start=1):
        cand["display_id"] = idx
    return added


def sample_path_points(path_world: list[tuple[float, float, float]], step: float) -> list:
    if len(path_world) < 2:
        return []
    samples = []
    for i in range(len(path_world) - 1):
        x0, y0, z0 = path_world[i]
        x1, y1, z1 = path_world[i + 1]
        seg_len = math.hypot(x1 - x0, y1 - y0)
        n = max(1, int(math.ceil(seg_len / step)))
        for s in range(n + 1):
            t = s / n
            samples.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0), z0 + t * (z1 - z0)))
    return samples


def path_centerline_visible(cam_obj, path_world: list[tuple[float, float, float]],
                            vfov_half: float, hfov_half: float,
                            res_x: int, res_y: int,
                            depsgraph, step: float, bottom_y: float) -> bool:
    if not path_world:
        return False
    cam_inv = cam_obj.matrix_world.inverted()
    origin = cam_obj.matrix_world.translation.copy()
    samples = sample_path_points(path_world, step)
    if not samples:
        return False
    # Find the first sample that is actually in view (skip points behind camera)
    first_idx = None
    for i, s in enumerate(samples):
        pt = mathutils.Vector(s)
        proj = project_world_point(cam_inv, pt, vfov_half, hfov_half, res_x, res_y, cam_obj=cam_obj, scene=bpy.context.scene)
        if proj is None:
            continue
        first_idx = i
        break

    if first_idx is None:
        if DEBUG:
            print("    [debug] path_vis fail: no_visible_start")
        return False

    has_bottom = False
    # Check all *visible* samples from the first in-view point onward
    for s in samples[first_idx:]:
        pt = mathutils.Vector(s)
        proj = project_world_point(cam_inv, pt, vfov_half, hfov_half, res_x, res_y, cam_obj=cam_obj, scene=bpy.context.scene)
        if proj is None:
            if DEBUG:
                print("    [debug] path_vis fail: sample_out_of_view")
            return False
        if proj["screen_xy"][1] <= bottom_y:
            has_bottom = True
        if not raycast_visible(bpy.context.scene, depsgraph, origin, pt):
            if DEBUG:
                print("    [debug] path_vis fail: sample_occluded")
            return False
    if not has_bottom:
        if DEBUG:
            print(f"    [debug] path_vis fail: no_bottom_sample "
                  f"bottom_y={bottom_y:.3f}")
        return False
    return True


# ---- Floor-clutter cleanup ------------------------------------------------
# Small objects sitting on the floor that block navigation but have no
# functional importance for a navigation benchmark.
REMOVABLE_CATEGORIES = {
    # Personal items / clutter
    "shoe", "shoes", "sneakers", "boots", "sandals", "slippers",
    "bag", "backpack", "suitcase", "purse", "briefcase",
    "hat", "cap", "helmet",
    "toy", "teddy bear", "stuffed animal", "doll",
    "bottle", "can", "cup", "mug", "glass",
    "book", "books", "magazine", "paper", "newspaper",
    "clothes", "clothing", "jacket", "coat", "shirt", "pants",
    "towel", "blanket", "rug", "mat", "carpet",
    "pillow", "cushion",
    "box", "carton", "package",
    "basket", "bin", "trash", "garbage", "waste",
    "case", "container", "crate", "bucket",
    "hamper", "laundry",
    "board",
    "cable", "cord", "wire",
    "plant", "flower", "pot",  # small floor plants
    "scale", "object", "item", "stuff", "misc",
    "umbrella", "broom", "mop",
    "toilet paper", "tissue",
    "stopcock", "dispenser",
    "teapot", "kettle",
}

# Categories that should NEVER be removed — functional furniture
KEEP_CATEGORIES = {
    "table", "desk", "counter", "countertop",
    "chair", "stool", "bench", "seat", "armchair",
    "couch", "sofa", "loveseat",
    "bed", "crib", "bunk",
    "cabinet", "dresser", "wardrobe", "closet", "shelf", "shelves",
    "stand", "nightstand", "bookshelf", "bookcase",
    "refrigerator", "fridge", "oven", "stove", "microwave",
    "sink", "bathtub", "shower", "toilet",
    "door", "window",
    "tv", "television", "monitor", "screen",
    "lamp", "light", "chandelier",  # keep light sources
    "radiator", "heater", "air conditioner",
    "washer", "dryer", "dishwasher",
    "piano", "fireplace",
    "computer",
}


def _quick_reachable_area(layout, room_bounds, resolution=0.1,
                          robot_radius=0.30, robot_height=1.4):
    """Quick check: max reachable area in the scene (no Blender needed)."""
    import numpy as np
    from collections import deque

    min_x, min_y, max_x, max_y = room_bounds
    w = max(1, int((max_x - min_x) / resolution))
    h = max(1, int((max_y - min_y) / resolution))
    grid = np.zeros((h, w), dtype=bool)

    wall_cells = max(1, int(robot_radius / resolution))
    grid[:wall_cells, :] = True
    grid[-wall_cells:, :] = True
    grid[:, :wall_cells] = True
    grid[:, -wall_cells:] = True

    for obj in layout:
        bbox = obj.get("bbox", [])
        if len(bbox) < 6:
            continue
        x, y, z = bbox[0], bbox[1], bbox[2]
        sx, sy, sz = bbox[3], bbox[4], bbox[5]
        obj_bottom = z - sz / 2
        obj_top = z + sz / 2
        if obj_bottom < robot_height and obj_top > 0.10:
            half_sx = sx / 2 + robot_radius
            half_sy = sy / 2 + robot_radius
            gx1 = max(0, int((x - half_sx - min_x) / resolution))
            gy1 = max(0, int((y - half_sy - min_y) / resolution))
            gx2 = min(w, int((x + half_sx - min_x) / resolution))
            gy2 = min(h, int((y + half_sy - min_y) / resolution))
            grid[gy1:gy2, gx1:gx2] = True

    free_indices = np.argwhere(~grid)
    if len(free_indices) == 0:
        return 0.0

    # Sample start points, find largest connected component
    rng = np.random.RandomState(42)
    n_samples = min(10, len(free_indices))
    sample_idx = rng.choice(len(free_indices), n_samples, replace=False)
    best_size = 0
    for idx in sample_idx:
        sr, sc = int(free_indices[idx, 0]), int(free_indices[idx, 1])
        if grid[sr, sc]:
            continue
        visited = {(sr, sc)}
        q = deque([(sr, sc)])
        while q:
            r, c = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and not grid[nr, nc]:
                    visited.add((nr, nc))
                    q.append((nr, nc))
        if len(visited) > best_size:
            best_size = len(visited)
    return best_size * resolution * resolution


def _is_removable_clutter(obj: dict, robot_height: float = 1.4) -> bool:
    """Check if an object is small floor clutter that can be safely removed."""
    bbox = obj.get("bbox", [])
    if len(bbox) < 6:
        return False

    cat = obj.get("category", "").lower().strip()
    x, y, z = bbox[0], bbox[1], bbox[2]
    sx, sy, sz = bbox[3], bbox[4], bbox[5]
    obj_bottom = z - sz / 2
    obj_top = z + sz / 2
    vol = sx * sy * sz
    footprint = sx * sy

    # Never remove things in the KEEP list
    for keep_cat in KEEP_CATEGORIES:
        if keep_cat in cat or cat in keep_cat:
            return False

    # Must be on or near the floor
    if obj_bottom > 0.20:
        return False

    # Must be small enough to be "clutter"
    # Height < 0.6m, footprint < 0.5m², volume < 0.2m³
    if obj_top > 0.6 or footprint > 0.5 or vol > 0.2:
        return False

    # Check if category is in the removable list
    for rem_cat in REMOVABLE_CATEGORIES:
        if rem_cat in cat or cat in rem_cat:
            return True

    # If it's very small (vol < 0.05m³) and on the floor, remove anyway
    if vol < 0.05 and footprint < 0.15:
        return True

    return False


def clean_floor_clutter(layout: list[dict], room_bounds, min_area=2.0):
    """Remove small floor clutter objects if the scene is too dense for navigation.

    Returns (cleaned_layout, removed_ids) — only removes objects if the
    original layout fails the reachable-area threshold.
    """
    # Quick check: does the scene already pass?
    orig_area = _quick_reachable_area(layout, room_bounds)
    if orig_area >= min_area:
        return layout, []  # No cleanup needed

    # Identify removable clutter, sorted by footprint (remove biggest blockers first)
    removable = []
    for i, obj in enumerate(layout):
        if _is_removable_clutter(obj):
            bbox = obj.get("bbox", [])
            footprint = bbox[3] * bbox[4] if len(bbox) >= 5 else 0
            removable.append((i, obj, footprint))
    removable.sort(key=lambda x: -x[2])  # largest footprint first

    if not removable:
        print(f"  Floor cleanup: no removable clutter found "
              f"(area={orig_area:.1f}m² < {min_area}m²)")
        return layout, []

    # Try removing objects one by one until we pass the threshold
    removed_indices = set()
    removed_objs = []
    for idx, obj, fp in removable:
        removed_indices.add(idx)
        test_layout = [o for i, o in enumerate(layout) if i not in removed_indices]
        new_area = _quick_reachable_area(test_layout, room_bounds)
        removed_objs.append(obj)
        cat = obj.get("category", "?")
        obj_id = obj.get("id", 0)
        print(f"    Removed {cat} (id={obj_id}, footprint={fp:.3f}m²) "
              f"-> area={new_area:.1f}m²")
        if new_area >= min_area:
            break

    cleaned_layout = [o for i, o in enumerate(layout) if i not in removed_indices]
    final_area = _quick_reachable_area(cleaned_layout, room_bounds)
    print(f"  Floor cleanup: removed {len(removed_objs)} objects, "
          f"area {orig_area:.1f} -> {final_area:.1f}m²")
    removed_ids = [o.get("id", 0) for o in removed_objs]
    return cleaned_layout, removed_ids


def remove_objects_from_blender(removed_ids: list[int]):
    """Delete Blender objects with matching obj_id custom property."""
    if not removed_ids:
        return
    id_set = set(removed_ids)
    to_delete = []
    for obj in bpy.data.objects:
        oid = obj.get("obj_id")
        if oid is not None and oid in id_set:
            to_delete.append(obj)
        elif obj.parent and obj.parent.get("obj_id") in id_set:
            to_delete.append(obj)
    # Delete objects
    for obj in to_delete:
        bpy.data.objects.remove(obj, do_unlink=True)
    if to_delete:
        print(f"  Deleted {len(to_delete)} Blender objects for floor cleanup")


def build_occupancy_for_camera(
    layout: list[dict],
    room_bounds: tuple[float, float, float, float],
    resolution: float = 0.1,
    robot_radius: float = 0.30,
    robot_height: float = 1.4,
):
    """Build a 2D occupancy grid marking walls and furniture as obstacles.

    Models the robot as a cylinder (diameter=60cm, height=1.4m) — slightly
    larger than actual (50cm, 1.3m) for safety margin.
    Inflates all obstacles by *robot_radius* so that any free cell in the
    grid is reachable by the robot's centre.

    Returns numpy bool array (True = occupied).
    """
    import numpy as np

    min_x, min_y, max_x, max_y = room_bounds
    w = max(1, int((max_x - min_x) / resolution))
    h = max(1, int((max_y - min_y) / resolution))
    grid = np.zeros((h, w), dtype=bool)

    # 1) Mark wall margins — cells within robot_radius of room boundary
    wall_cells = max(1, int(robot_radius / resolution))
    grid[:wall_cells, :] = True
    grid[-wall_cells:, :] = True
    grid[:, :wall_cells] = True
    grid[:, -wall_cells:] = True

    # 2) Mark furniture bounding boxes + robot_radius inflation
    for obj in layout:
        bbox = obj.get("bbox", [])
        if len(bbox) < 6:
            continue
        x, y, z = bbox[0], bbox[1], bbox[2]
        sx, sy, sz = bbox[3], bbox[4], bbox[5]

        # Only mark objects that block walking:
        #   obj_bottom < robot_height (1.4m) AND obj_top > 10cm
        obj_bottom = z - sz / 2
        obj_top = z + sz / 2
        if obj_bottom < robot_height and obj_top > 0.10:
            half_sx = sx / 2 + robot_radius
            half_sy = sy / 2 + robot_radius
            gx1 = max(0, int((x - half_sx - min_x) / resolution))
            gy1 = max(0, int((y - half_sy - min_y) / resolution))
            gx2 = min(w, int((x + half_sx - min_x) / resolution))
            gy2 = min(h, int((y + half_sy - min_y) / resolution))
            grid[gy1:gy2, gx1:gx2] = True

    return grid


def compute_camera_positions(layout: list[dict], config: dict,
                             nav_grid=None,
                             target_cfg: dict | None = None,
                             rec_categories: set[str] | None = None,
                             large_categories: set[str] | None = None,
                             category_aliases: dict | None = None,
                             camera_seed_audit: dict | None = None) -> list[dict]:
    """
    Compute camera positions near room edges, looking inward to maximise
    the number of visible objects.

    Algorithm
    ---------
    1. Build occupancy grid (walls + furniture).
    2. Find walkable cells that are *near room boundaries* (edge positions).
    3. For every candidate, sweep look-angles (every 10°) and count how
       many layout objects fall inside the horizontal view frustum.
    4. Greedy selection: pick the (position, angle) that adds the most
       *new* objects to the already-covered set, with tie-breaking by
       total visible count and an edge-proximity bonus.
    5. 20° downward pitch for floor + obstacle visibility (navigation).
    """
    import numpy as np
    from scipy.ndimage import distance_transform_edt

    render_cfg = config["render"]
    target_cfg = target_cfg or {}
    rec_categories = rec_categories or set()
    large_categories = large_categories or set()
    category_aliases = category_aliases or {}
    num_views = render_cfg["views_per_scene"]
    camera_height = render_cfg["camera_height"]
    experiment_mode = resolve_view_seed_experiment_mode(target_cfg)
    route_aware_weight = float(target_cfg.get("view_seed_route_aware_weight", 1.0))
    rng_seed_raw = target_cfg.get("view_select_rng_seed")
    rng_seed = None if rng_seed_raw in (None, "") else int(rng_seed_raw)
    rng = np.random.default_rng(rng_seed) if rng_seed is not None else None
    max_candidate_budget = int(target_cfg.get("view_select_candidates", 200))
    vfov = math.radians(render_cfg["fov"])  # config FOV is VFOV
    # Compute actual HFOV from VFOV + aspect ratio (sensor_fit = VERTICAL)
    res_w, res_h = render_cfg["resolution"]
    aspect = res_w / res_h
    hfov = 2 * math.atan(math.tan(vfov / 2) * aspect)
    print(f"  Camera FOV: VFOV={math.degrees(vfov):.1f}°, "
          f"HFOV={math.degrees(hfov):.1f}° (aspect={aspect:.2f})")
    resolution = 0.10  # occupancy-grid cell size (metres)

    # ---- room bounds from imported wall mesh ----
    room_bounds = get_room_bounds_from_scene()
    min_x, min_y, max_x, max_y = room_bounds

    if camera_seed_audit is not None:
        camera_seed_audit.clear()
        camera_seed_audit.update({
            "candidate_pool_stage": "edge_near",
            "room_bounds_m": [
                float(min_x),
                float(min_y),
                float(max_x),
                float(max_y),
            ],
            "grid_resolution_m": float(resolution),
            "n_layout_objects": int(len(layout)),
            "experiment_mode": experiment_mode,
            "rng_seed": rng_seed,
            "candidate_pool_counts": {
                "edge_near": 0,
                "relaxed_clearance": 0,
                "relaxed_minimal": 0,
                "before_subsample": 0,
                "after_subsample": 0,
                "max_candidate_budget": int(max_candidate_budget),
            },
            "target_seed_counts": {
                "attempted": 0,
                "accepted": 0,
                "selected": 0,
            },
            "selection_counts": {
                "scored_candidates": 0,
                "selected_before_post_filters": 0,
                "collision_filtered_out": 0,
                "nav_filtered_out": 0,
                "final_cameras": 0,
            },
            "used_target_preselection": False,
            "selected_records": [],
        })

    # ---- occupancy grid ----
    grid = build_occupancy_for_camera(layout, room_bounds, resolution)
    h, w = grid.shape

    # distance of each free cell to the nearest obstacle
    dist_map = distance_transform_edt(~grid) * resolution

    # distance of each cell to the nearest room-boundary edge
    iy = np.arange(h)[:, None]
    ix = np.arange(w)[None, :]
    boundary_dist = np.minimum(
        np.minimum(iy, h - 1 - iy),
        np.minimum(ix, w - 1 - ix),
    ).astype(float) * resolution

    # ---- candidate cells: walkable AND near room edges ----
    # Occupancy grid already inflated by robot_radius, so any free cell
    # guarantees the 50cm-wide robot fits.  Use small extra clearance (5cm)
    # to avoid numerical edge cases.
    min_clearance = 0.05  # extra buffer beyond robot-radius inflation
    edge_near = 2.0       # prefer cells within 2 m of boundary
    edge_mask = (dist_map > min_clearance) & (boundary_dist < edge_near)
    ys_idx, xs_idx = np.where(edge_mask)
    if camera_seed_audit is not None:
        camera_seed_audit["candidate_pool_counts"]["edge_near"] = int(len(ys_idx))

    # fall-backs (progressively relax constraints)
    if len(ys_idx) < 10:
        ys_idx, xs_idx = np.where(dist_map > min_clearance)
        if camera_seed_audit is not None:
            camera_seed_audit["candidate_pool_stage"] = "relaxed_clearance"
            camera_seed_audit["candidate_pool_counts"]["relaxed_clearance"] = int(len(ys_idx))
    if len(ys_idx) < 5:
        ys_idx, xs_idx = np.where(dist_map > 0.01)
        if camera_seed_audit is not None:
            camera_seed_audit["candidate_pool_stage"] = "relaxed_minimal"
            camera_seed_audit["candidate_pool_counts"]["relaxed_minimal"] = int(len(ys_idx))
    if len(ys_idx) == 0:
        cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
        if camera_seed_audit is not None:
            camera_seed_audit["candidate_pool_stage"] = "center_fallback_no_candidates"
            camera_seed_audit["selection_counts"]["final_cameras"] = 1
            camera_seed_audit["selected_records"] = [{
                "view_id": 0,
                "source": "center_fallback_no_candidates",
                "candidate_pool_stage": "center_fallback_no_candidates",
                "target_id": None,
                "n_vis": 0,
                "path_score": 0.0,
            }]
        return [{
            "position": [cx, cy, camera_height],
            "rotation": [math.pi / 2 - math.radians(8), 0, 0],
            "look_angle": 0.0,
            "view_id": 0,
            "frustum_visible": [],
        }]

    world_x = min_x + (xs_idx + 0.5) * resolution
    world_y = min_y + (ys_idx + 0.5) * resolution
    cand_xy = np.column_stack([world_x, world_y])       # (N, 2)
    cand_bdist = boundary_dist[ys_idx, xs_idx]
    if camera_seed_audit is not None:
        camera_seed_audit["candidate_pool_counts"]["before_subsample"] = int(len(cand_xy))

    # sub-sample if there are too many candidates
    if len(cand_xy) > max_candidate_budget:
        # bias towards edge positions
        w_edge = 1.0 / (cand_bdist + 0.1)
        probs = w_edge / w_edge.sum()
        if rng is None:
            idx_chosen = np.random.choice(
                len(cand_xy),
                max_candidate_budget,
                replace=False,
                p=probs,
            )
        else:
            idx_chosen = rng.choice(
                len(cand_xy),
                max_candidate_budget,
                replace=False,
                p=probs,
            )
        cand_xy = cand_xy[idx_chosen]
        cand_bdist = cand_bdist[idx_chosen]
    if camera_seed_audit is not None:
        camera_seed_audit["candidate_pool_counts"]["after_subsample"] = int(len(cand_xy))

    # ---- gather furniture XY positions ----
    obj_list = []
    for obj in layout:
        bbox = obj.get("bbox", [])
        if len(bbox) >= 6 and obj.get("model_uid", ""):
            category_meta = canonicalize_category(obj.get("category", "unknown"), category_aliases)
            material_color_name = (
                _normalized_instruction_text(obj.get("material_color_name"))
                or quantize_material_color_name(obj.get("material_color_rgb"))
            )
            obj_list.append({
                "obj": obj,
                "x": bbox[0], "y": bbox[1], "z": bbox[2],
                "sx": bbox[3], "sy": bbox[4], "sz": bbox[5],
                "category": obj.get("category", "unknown"),
                "canonical_category": category_meta["canonical_category"],
                "semantic_group_id": category_meta["semantic_group_id"],
                "material_color_name": material_color_name,
                "id": obj.get("id", 0),
                "model_uid": obj.get("model_uid", ""),
            })

    if not obj_list:
        cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2
        if camera_seed_audit is not None:
            camera_seed_audit["candidate_pool_stage"] = "center_fallback_no_objects"
            camera_seed_audit["selection_counts"]["final_cameras"] = 1
            camera_seed_audit["selected_records"] = [{
                "view_id": 0,
                "source": "center_fallback_no_objects",
                "candidate_pool_stage": "center_fallback_no_objects",
                "target_id": None,
                "n_vis": 0,
                "path_score": 0.0,
            }]
        return [{
            "position": [cx, cy, camera_height],
            "rotation": [math.pi / 2 - math.radians(8), 0, 0],
            "look_angle": 0.0,
            "view_id": 0,
            "frustum_visible": [],
        }]

    obj_xy = np.array([(o["x"], o["y"]) for o in obj_list])  # (M, 2)
    scene_family_counter = Counter(str(o.get("semantic_group_id", "unknown")) for o in obj_list)
    scene_family_color_counter = Counter(
        (
            str(o.get("semantic_group_id", "unknown")),
            _normalized_instruction_text(o.get("material_color_name")) or "unknown",
        )
        for o in obj_list
    )
    hfov_half = hfov / 2.0  # Use HFOV for horizontal frustum sweep

    # ---- Pre-compute 3D line-of-sight via Blender ray_cast ----
    # The composed scene is already loaded; cast rays from each candidate
    # position to each object centre to detect wall / furniture occlusion.
    depsgraph = bpy.context.evaluated_depsgraph_get()
    bpy.context.view_layer.update()

    # Temporary camera for path-aware scoring / target-driven selection
    n_angles = int(target_cfg.get("view_select_angles", 12))
    angles = np.linspace(0, 2 * math.pi, n_angles, endpoint=False)
    min_dist = float(target_cfg.get("min_target_distance", 2.0))
    large_ratio = float(target_cfg.get("large_visibility_ratio", 0.70))
    large_size = float(target_cfg.get("large_size_m", 1.0))
    min_visible_ratio = float(target_cfg.get("min_target_visible_surface_ratio", 0.15))

    tmp_cam = None
    tmp_cam_data = None
    if nav_grid is not None:
        tmp_cam_data = bpy.data.cameras.new("TmpSelectCam")
        tmp_cam_data.lens_unit = "FOV"
        tmp_cam_data.angle = math.radians(render_cfg["fov"])
        tmp_cam_data.sensor_fit = "VERTICAL"
        tmp_cam = bpy.data.objects.new("TmpSelectCam", tmp_cam_data)
        bpy.context.scene.collection.objects.link(tmp_cam)
        tmp_cam.location = (0.0, 0.0, camera_height)
        tmp_cam.rotation_euler = (math.pi / 2 - math.radians(20), 0.0, 0.0)

    # ---- Target-driven view sampling (priority) ----
    selected = None
    allow_target_preselection = experiment_mode == "baseline"
    if allow_target_preselection and nav_grid is not None and tmp_cam is not None and obj_list:
        target_limit = int(target_cfg.get("view_select_target_limit", 40))
        target_angles = int(target_cfg.get("view_select_target_angles", 12))
        target_seed_limit = int(target_cfg.get(
            "view_select_target_seed_limit",
            max(int(num_views) * 8, min(int(target_limit) * 2, 64)),
        ))
        # Default radii (meters): near to mid-range
        radii = target_cfg.get("view_select_target_radii")
        if radii is None:
            radii = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0]
        elif isinstance(radii, str):
            radii = [float(r.strip()) for r in radii.split(",") if r.strip()]

        dedup_pos = float(target_cfg.get("dedup_position_m", 1.0))
        dedup_yaw = math.radians(float(target_cfg.get("dedup_yaw_deg", 20.0)))

        # Sort targets: recommended categories first
        sorted_objs = sorted(
            obj_list,
            key=lambda o: (0 if o["category"] in rec_categories else 1, o["id"]),
        )
        target_views = []
        for obj in sorted_objs[:target_limit]:
            ox, oy = obj["x"], obj["y"]
            for r in radii:
                for k in range(target_angles):
                    if camera_seed_audit is not None:
                        camera_seed_audit["target_seed_counts"]["attempted"] += 1
                    theta = 2 * math.pi * k / target_angles
                    cx = ox + r * math.cos(theta)
                    cy = oy + r * math.sin(theta)

                    # Keep inside room bounds with margin
                    if (cx < min_x + 0.3 or cx > max_x - 0.3
                            or cy < min_y + 0.3 or cy > max_y - 0.3):
                        continue

                    gx, gy = nav_grid.world_to_grid(cx, cy)
                    if not nav_grid.is_free(gx, gy):
                        continue

                    yaw = math.atan2(oy - cy, ox - cx)
                    # Dedup vs existing selected views
                    too_similar = False
                    for prev in target_views:
                        dxp = cx - prev["pos"][0]
                        dyp = cy - prev["pos"][1]
                        if math.hypot(dxp, dyp) < dedup_pos:
                            yaw_diff = abs((yaw - prev["angle"] + math.pi) % (2 * math.pi) - math.pi)
                            if yaw_diff < dedup_yaw:
                                too_similar = True
                                break
                    if too_similar:
                        continue

                    if not forward_clear(
                        cam_pos=(cx, cy, camera_height),
                        cam_yaw=yaw,
                        layout=layout,
                        radius=float(target_cfg.get("forward_clear_radius_m", 1.0)),
                        height=float(target_cfg.get("forward_clear_height_m", 1.4)),
                    ):
                        continue

                    tmp_cam.location = (cx, cy, camera_height)
                    tmp_cam.rotation_euler = (math.pi / 2 - math.radians(20), 0.0, yaw - math.pi / 2)
                    bpy.context.view_layer.update()

                    obj_ref = obj["obj"]
                    bbox = obj_ref.get("bbox", [])
                    if len(bbox) < 6:
                        continue

                    dist = math.hypot(ox - cx, oy - cy)
                    if dist < min_dist:
                        continue

                    is_large = is_large_object(obj_ref, large_categories, large_size)
                    center_visible = raycast_visible(
                        bpy.context.scene,
                        depsgraph,
                        tmp_cam.matrix_world.translation.copy(),
                        mathutils.Vector((float(bbox[0]), float(bbox[1]), float(bbox[2]))),
                        tol=0.08,
                    )
                    vis_ratio = sample_target_surface_visibility_ratio(
                        tmp_cam,
                        obj_ref,
                        vfov / 2,
                        hfov / 2,
                        res_w,
                        res_h,
                        depsgraph,
                    )
                    if not target_surface_visibility_ok(
                        is_large=is_large,
                        vis_ratio=vis_ratio,
                        center_visible=center_visible,
                        min_visible_ratio=min_visible_ratio,
                        large_visibility_ratio=large_ratio,
                    ):
                        continue

                    family_id = str(obj.get("semantic_group_id", "unknown"))
                    color_name = _normalized_instruction_text(obj.get("material_color_name")) or "unknown"
                    same_family_total = int(scene_family_counter.get(family_id, 1))
                    same_family_same_color = int(scene_family_color_counter.get((family_id, color_name), 1))
                    same_family_diff_color = max(0, same_family_total - same_family_same_color)
                    is_recommended = obj.get("canonical_category") in rec_categories
                    target_views.append({
                        "target_id": int(obj["id"]),
                        "pos": (cx, cy),
                        "angle": yaw,
                        "n_vis": 1,
                        "path_score": round(target_view_seed_score({
                            "target_distance_m": float(dist),
                            "same_family_visible_count": same_family_total,
                            "same_family_diff_color_visible_count": same_family_diff_color,
                            "is_recommended": is_recommended,
                            "visibility_ratio": float(vis_ratio),
                        }), 3),
                        "ids": {obj["id"]},
                        "objs": [obj],
                        "bdist": 0.0,
                        "fwd_clearance_ray": 2.0,
                        "fwd_clearance_obj": 2.0,
                        "target_distance_m": round(float(dist), 3),
                        "same_family_visible_count": same_family_total,
                        "same_family_diff_color_visible_count": same_family_diff_color,
                        "visibility_ratio": round(float(vis_ratio), 3),
                        "is_recommended": is_recommended,
                        "camera_seed_source": "target_preselected",
                        "candidate_pool_stage": camera_seed_audit.get("candidate_pool_stage", "edge_near")
                        if camera_seed_audit is not None else "edge_near",
                    })
                    if camera_seed_audit is not None:
                        camera_seed_audit["target_seed_counts"]["accepted"] += 1
                    if len(target_views) >= target_seed_limit:
                        break
                if len(target_views) >= target_seed_limit:
                    break
            if len(target_views) >= target_seed_limit:
                break

        if target_views:
            selected = select_target_view_seeds(target_views, num_views)
            if camera_seed_audit is not None:
                camera_seed_audit["target_seed_counts"]["selected"] = int(len(selected))

    use_preselected = selected is not None
    if camera_seed_audit is not None:
        camera_seed_audit["used_target_preselection"] = bool(use_preselected)

    if selected is None:
        los_clear = np.ones((len(cand_xy), len(obj_list)), dtype=bool)
        for ci in range(len(cand_xy)):
            cx, cy = float(cand_xy[ci, 0]), float(cand_xy[ci, 1])
            origin = mathutils.Vector((cx, cy, camera_height))
            for oi, obj in enumerate(obj_list):
                target = mathutils.Vector((obj["x"], obj["y"], obj["z"]))
                direction = target - origin
                dist = direction.length
                if dist < 0.3:
                    continue  # too close, assume visible
                direction = direction.normalized()
                obj_radius = max(obj["sx"], obj["sy"], obj["sz"]) / 2

                hit, loc, _n, _i, _o, _m = bpy.context.scene.ray_cast(
                    depsgraph, origin, direction, distance=dist + 0.1)
                if hit:
                    hit_dist = (loc - origin).length
                    # Something blocks the path before reaching the object
                    if hit_dist < dist - obj_radius:
                        los_clear[ci, oi] = False

        print(f"  LOS pre-computation: {len(cand_xy)} candidates × "
              f"{len(obj_list)} objects, "
              f"{int(los_clear.sum())}/{los_clear.size} clear")

        candidate_records = []
        for ci in range(len(cand_xy)):
            cx, cy = float(cand_xy[ci, 0]), float(cand_xy[ci, 1])
            dx = obj_xy[:, 0] - cx
            dy = obj_xy[:, 1] - cy
            dists = np.sqrt(dx ** 2 + dy ** 2)
            angs = np.arctan2(dy, dx)
            los = los_clear[ci]  # per-object LOS for this candidate

            best_n = 0
            best_a = 0.0
            best_ids: set = set()
            best_objs: list = []

            for a in angles:
                adiff = np.abs((angs - a + math.pi) % (2 * math.pi) - math.pi)
                # Only count objects that are in FOV AND have clear line-of-sight
                in_fov = (adiff < hfov_half) & (dists > 0.5) & (dists < 12.0) & los
                n = int(np.sum(in_fov))
                if n > best_n:
                    best_n = n
                    best_a = float(a)
                    best_ids = {obj_list[j]["id"] for j in np.where(in_fov)[0]}
                    best_objs = [obj_list[j] for j in np.where(in_fov)[0]]

            candidate_records.append({
                "pos": (float(cx), float(cy)),
                "angle": best_a,
                "n_vis": best_n,
                "path_score": 0.0,
                "ids": best_ids,
                "objs": best_objs,
                "bdist": float(cand_bdist[ci]),
                "camera_seed_source": "grid_scored",
                "candidate_pool_stage": (
                    camera_seed_audit.get("candidate_pool_stage", "edge_near")
                    if camera_seed_audit is not None else "edge_near"
                ),
            })

        if experiment_mode == "route_aware":
            for candidate in candidate_records:
                candidate["path_score"] = round(route_aware_view_seed_score(
                    candidate,
                    min_dist=min_dist,
                    rec_categories=rec_categories,
                ), 3)

        def clearance_provider(cx: float, cy: float, best_a: float) -> dict:
            # Combine (a) Blender ray_cast for walls with (b) analytical
            # corridor sweep over layout objects. The robot is modelled as a
            # cylinder (diameter=60 cm, height=1.4 m).
            fwd_cos, fwd_sin = math.cos(best_a), math.sin(best_a)
            fwd_dir = mathutils.Vector((fwd_cos, fwd_sin, 0.0)).normalized()
            fwd_origin = mathutils.Vector((cx, cy, camera_height))
            fwd_hit, fwd_loc, _, _, _, _ = bpy.context.scene.ray_cast(
                depsgraph, fwd_origin, fwd_dir, distance=3.0)
            fwd_clearance_ray = float(
                (fwd_loc - fwd_origin).length) if fwd_hit else 3.0

            corridor_radius = 0.30
            corridor_height = 1.4
            fwd_clearance_obj = 3.0
            for obj in layout:
                bbox = obj.get("bbox", [])
                if len(bbox) < 6:
                    continue
                ox, oy, oz = bbox[0], bbox[1], bbox[2]
                osx, osy, osz = bbox[3], bbox[4], bbox[5]
                obj_bottom = oz - osz / 2
                obj_top = oz + osz / 2
                if obj_bottom >= corridor_height or obj_top < 0.10:
                    continue
                half_x, half_y = osx / 2, osy / 2
                corners = [
                    (ox - half_x - cx, oy - half_y - cy),
                    (ox + half_x - cx, oy - half_y - cy),
                    (ox - half_x - cx, oy + half_y - cy),
                    (ox + half_x - cx, oy + half_y - cy),
                ]
                fwd_projs = [dx * fwd_cos + dy * fwd_sin for dx, dy in corners]
                lat_projs = [-dx * fwd_sin + dy * fwd_cos for dx, dy in corners]
                fwd_min_o, fwd_max_o = min(fwd_projs), max(fwd_projs)
                lat_min_o, lat_max_o = min(lat_projs), max(lat_projs)
                if (fwd_max_o > 0.0
                        and lat_max_o > -corridor_radius
                        and lat_min_o < corridor_radius):
                    fwd_clearance_obj = min(fwd_clearance_obj, max(0.0, fwd_min_o))
            return {
                "fwd_clearance_ray": fwd_clearance_ray,
                "fwd_clearance_obj": fwd_clearance_obj,
            }

        scored = build_scored_camera_candidates(
            candidate_records,
            clearance_provider=clearance_provider,
        )
        if camera_seed_audit is not None:
            camera_seed_audit["selection_counts"]["scored_candidates"] = int(len(scored))

    # ---- greedy selection maximising coverage ----
    selected = selected or []
    if not use_preselected:
        selected = select_scored_camera_views(
            scored,
            num_views=num_views,
            min_visible_objects=3,
            route_aware_weight=(route_aware_weight if experiment_mode == "route_aware" else 0.0),
        )
    if camera_seed_audit is not None:
        camera_seed_audit["selection_counts"]["selected_before_post_filters"] = int(len(selected))

    # ---- Post-selection: precise collision check against layout ----
    # Occupancy grid has 10cm resolution; small objects may slip through.
    # Reject any camera position whose XY footprint overlaps with any
    # layout object.  Robot modelled as cylinder: diameter=60cm, height=1.4m
    # (slightly larger than actual 50cm/1.3m for safety margin).
    robot_r = 0.30
    robot_h = 1.4  # collision cylinder height
    filtered = []
    for sel in selected:
        cx, cy = sel["pos"]
        collision = False
        for obj in layout:
            bbox = obj.get("bbox", [])
            if len(bbox) < 6:
                continue
            ox, oy, oz = bbox[0], bbox[1], bbox[2]
            sx, sy, sz = bbox[3], bbox[4], bbox[5]
            obj_bottom = oz - sz / 2
            obj_top = oz + sz / 2
            # Skip objects fully above the robot cylinder or flat on the floor
            if obj_bottom >= robot_h or obj_top < 0.10:
                continue
            # Check XY overlap (robot circle vs object AABB)
            closest_x = max(ox - sx / 2, min(cx, ox + sx / 2))
            closest_y = max(oy - sy / 2, min(cy, oy + sy / 2))
            dist = math.sqrt((cx - closest_x) ** 2 + (cy - closest_y) ** 2)
            if dist < robot_r:
                collision = True
                cat = obj.get("category", "?")
                print(f"    Rejected pos ({cx:.2f},{cy:.2f}): "
                      f"collides with {cat} at ({ox:.2f},{oy:.2f})")
                break
        if not collision:
            filtered.append(sel)

    if len(filtered) < len(selected):
        print(f"  Collision filter: {len(selected)} -> {len(filtered)} views")
    if camera_seed_audit is not None:
        camera_seed_audit["selection_counts"]["collision_filtered_out"] = int(max(0, len(selected) - len(filtered)))
    selected = filtered

    # ---- Post-selection: navigability check ----
    # For each view, ensure at least one visible object ≥ MIN_NAV_DIST
    # away is *reachable* from the camera position through the occupancy
    # grid.  This guarantees the view has navigation value.
    #
    # Strategy: ONE flood-fill from camera → reachable set, then check
    # if any cell within GOAL_SEARCH_R of a distant visible object is
    # in the reachable set.  This avoids repeated BFS and correctly
    # handles disconnected occupancy grids.
    def _flood_fill(grid, start_r, start_c):
        """BFS flood-fill from start.  Returns set of reachable (r,c)."""
        from collections import deque
        h, w = grid.shape
        sr, sc = int(start_r), int(start_c)
        if not (0 <= sr < h and 0 <= sc < w and not grid[sr, sc]):
            return set()
        visited = {(sr, sc)}
        q = deque([(sr, sc)])
        while q:
            r, c = q.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and not grid[nr, nc]:
                    visited.add((nr, nc))
                    q.append((nr, nc))
        return visited

    # Minimum distance for an object to count as a "navigation target":
    # must be far enough that the robot actually needs to move.
    # Scale with room size — at least 1.0m, but no more than 2.0m.
    room_diag = math.hypot(max_x - min_x, max_y - min_y)
    MIN_NAV_DIST = max(1.0, min(2.0, room_diag * 0.15))
    # Search radius for reachable cell near object (cells).
    # Needs to be large enough to escape inflated zone around furniture.
    GOAL_SEARCH_R = 15  # 1.5m
    nav_filtered = []
    for sel in selected:
        cx, cy = sel["pos"]
        cam_r = int((cy - min_y) / resolution)
        cam_c = int((cx - min_x) / resolution)

        # One flood-fill from camera position
        reachable = _flood_fill(grid, cam_r, cam_c)
        reach_area = len(reachable) * resolution * resolution

        # Minimum reachable area: robot needs room to actually navigate
        MIN_REACHABLE_AREA = 2.0  # m²
        if reach_area < MIN_REACHABLE_AREA:
            print(f"    Rejected pos ({cx:.2f},{cy:.2f}): "
                  f"reachable area={reach_area:.1f}m² < {MIN_REACHABLE_AREA}m²")
            continue

        has_reachable_target = False
        n_far = 0
        for obj in sel.get("objs", []):
            ox = obj.get("x", 0)
            oy = obj.get("y", 0)
            obj_dist = math.hypot(ox - cx, oy - cy)
            if obj_dist < MIN_NAV_DIST:
                continue  # too close — not a navigation target
            n_far += 1
            # Check if any reachable cell is within GOAL_SEARCH_R of object
            obj_r = int((oy - min_y) / resolution)
            obj_c = int((ox - min_x) / resolution)
            for dr in range(-GOAL_SEARCH_R, GOAL_SEARCH_R + 1):
                for dc in range(-GOAL_SEARCH_R, GOAL_SEARCH_R + 1):
                    if (obj_r + dr, obj_c + dc) in reachable:
                        has_reachable_target = True
                        break
                if has_reachable_target:
                    break
            if has_reachable_target:
                break
        if has_reachable_target:
            nav_filtered.append(sel)
        else:
            print(f"    Rejected pos ({cx:.2f},{cy:.2f}): "
                  f"no reachable nav target "
                  f"({n_far} far objs, area={reach_area:.1f}m²)")

    if len(nav_filtered) < len(selected):
        print(f"  Navigability filter: {len(selected)} -> {len(nav_filtered)} views")
    if camera_seed_audit is not None:
        camera_seed_audit["selection_counts"]["nav_filtered_out"] = int(max(0, len(selected) - len(nav_filtered)))
    selected = nav_filtered

    if tmp_cam is not None:
        bpy.data.objects.remove(tmp_cam, do_unlink=True)
    if tmp_cam_data is not None:
        bpy.data.cameras.remove(tmp_cam_data)

    # ---- build camera dicts ----
    cameras = []
    selected_records: list[dict] = []
    for vi, sel in enumerate(selected):
        cx, cy = sel["pos"]
        look = sel["angle"]
        pitch = math.pi / 2 - math.radians(20)  # 20° downward tilt: floor visible from ~0.9m ahead
        selected_records.append({
            "view_id": int(vi),
            "source": sel.get("camera_seed_source"),
            "candidate_pool_stage": sel.get("candidate_pool_stage"),
            "target_id": sel.get("target_id"),
            "n_vis": int(sel.get("n_vis", 0)),
            "path_score": float(sel.get("path_score", 0.0)),
        })

        cameras.append({
            "position": [cx, cy, camera_height],
            "rotation": [pitch, 0, look - math.pi / 2],
            "look_angle": look,
            "view_id": vi,
            "frustum_visible": [
                {"id": o["id"], "category": o["category"],
                 "model_uid": o["model_uid"]}
                for o in sel["objs"]
            ],
        })
    if camera_seed_audit is not None:
        camera_seed_audit["selection_counts"]["final_cameras"] = int(len(cameras))
        camera_seed_audit["selected_records"] = selected_records

    total = len(obj_list)
    print(f"  Camera placement: {len(cameras)} views, "
          f"{camera_selection_coverage_count(selected)}/{total} objects in frustum")
    return cameras


def setup_camera(cam_info: dict, fov: float):
    """Set up camera with given position and rotation."""
    cam_data = bpy.data.cameras.new("Camera")
    cam_data.lens_unit = "FOV"
    cam_data.angle = math.radians(fov)  # This is VFOV
    cam_data.sensor_fit = 'VERTICAL'   # FOV controls vertical; horizontal auto-expands with aspect ratio
    cam_obj = bpy.data.objects.new("Camera", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    cam_obj.location = cam_info["position"]
    cam_obj.rotation_euler = cam_info["rotation"]
    return cam_obj


def _raycast_passthrough(scene, depsgraph, origin, direction, target_dist,
                         obj_radius, max_bounces=8):
    """Cast a ray through the scene, continuing past thin surfaces.

    Handles transparent/thin geometry (curtains, glass) by re-casting from
    just past each hit point.  Returns True if the ray eventually reaches
    within *obj_radius* of *target_dist* without exceeding *max_bounces*.
    """
    current = origin.copy()
    for _ in range(max_bounces):
        remaining = target_dist + obj_radius - (current - origin).length
        if remaining <= 0:
            return True  # passed beyond target

        hit, loc, _n, _i, _o, _m = scene.ray_cast(
            depsgraph, current, direction, distance=remaining + 0.1,
        )
        if not hit:
            return True  # no more geometry blocking

        d_from_origin = (loc - origin).length
        if abs(d_from_origin - target_dist) < obj_radius:
            return True  # reached the target
        if d_from_origin > target_dist + obj_radius:
            return True  # passed the target

        # Continue from 1 cm past the hit
        current = loc + direction * 0.01

    return False  # too many surfaces — likely a thick wall


def verify_object_visibility(cam_info: dict, layout: list[dict],
                             fov_deg: float) -> list[dict]:
    """Determine which layout objects are visible in the current camera.

    **Primary filter**: frustum test — project each object centre to camera
    space and reject if outside the field of view.

    **Supplementary check**: pass-through ray-cast — cast a ray from camera
    toward each in-frustum object.  If the first hit is *not* near the
    target, the ray continues past thin geometry (curtains, glass tables)
    for up to 8 bounces.  The result is stored as ``raycast_verified``
    (True/False).  Objects are included in the output regardless so that
    downstream VQA generation has a complete candidate set.

    Returns a list of dicts with id, category, model_uid, position, size,
    depth, normalised screen coordinates, and raycast_verified flag.
    """
    cam_obj = bpy.context.scene.camera
    if cam_obj is None:
        return []

    # Force Blender to update transforms — matrix_world is lazy-evaluated
    bpy.context.view_layer.update()

    cam_pos = cam_obj.matrix_world.translation.copy()
    cam_inv = cam_obj.matrix_world.inverted()

    fov_rad = math.radians(fov_deg)    # fov_deg is VFOV from config
    vfov_half = fov_rad / 2
    res_x = bpy.context.scene.render.resolution_x
    res_y = bpy.context.scene.render.resolution_y
    aspect = res_x / res_y
    # With sensor_fit=VERTICAL, HFOV = 2*atan(tan(VFOV/2)*aspect)
    hfov_half = math.atan(math.tan(vfov_half) * aspect)

    depsgraph = bpy.context.evaluated_depsgraph_get()

    visible: list[dict] = []
    for obj in layout:
        bbox = obj.get("bbox", [])
        if len(bbox) < 6 or not obj.get("model_uid", ""):
            continue
        ox, oy, oz = bbox[0], bbox[1], bbox[2]
        sx, sy, sz = bbox[3], bbox[4], bbox[5]

        world_pt = mathutils.Vector((ox, oy, oz))
        proj = project_world_point(
            cam_inv,
            world_pt,
            vfov_half,
            hfov_half,
            res_x,
            res_y,
            cam_obj=cam_obj,
            scene=bpy.context.scene,
        )
        if proj is None:
            continue
        depth = float(proj.get("depth", 0.0))

        # ---- pass-through ray-cast (supplementary) ----
        target = mathutils.Vector((ox, oy, oz))
        direction = (target - cam_pos)
        dist = direction.length
        if dist < 0.01:
            continue
        direction = direction.normalized()
        obj_max_r = math.sqrt(sx**2 + sy**2 + sz**2) / 2

        rc_ok = _raycast_passthrough(
            bpy.context.scene, depsgraph, cam_pos, direction,
            dist, obj_max_r, max_bounces=8,
        )

        proj_x, proj_y = proj["screen_xy"]
        visible.append({
            "id": obj.get("id", 0),
            "category": obj.get("category", "unknown"),
            "model_uid": obj.get("model_uid", ""),
            "position": [round(ox, 3), round(oy, 3), round(oz, 3)],
            "size": [round(sx, 3), round(sy, 3), round(sz, 3)],
            "depth": round(depth, 2),
            "screen_x": round(float(proj_x), 3),
            "screen_y": round(float(proj_y), 3),
            "raycast_verified": rc_ok,
        })

    return visible


def render_all_passes(rgb_path: str, depth_path: str = None,
                      normal_path: str = None, object_index_path: str = None,
                      object_index_max_pass_id: int | None = None):
    """Render RGB, depth, and normal in a single pass using compositor nodes."""
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree
    tree.links.clear()
    tree.nodes.clear()

    # Render Layers node
    rl = tree.nodes.new("CompositorNodeRLayers")
    rl.location = (0, 0)

    # RGB output via Composite node
    composite = tree.nodes.new("CompositorNodeComposite")
    composite.location = (600, 0)
    tree.links.new(rl.outputs["Image"], composite.inputs["Image"])

    # Set RGB output path
    scene.render.filepath = rgb_path

    # Depth output — fixed-range mapping so metric depth is recoverable
    # depth_metric = pixel_value * DEPTH_MAX_METERS
    DEPTH_MAX_METERS = 20.0  # max representable depth in metres
    if depth_path:
        depth_output = tree.nodes.new("CompositorNodeOutputFile")
        depth_output.location = (600, -200)
        depth_output.base_path = str(Path(depth_path).parent)
        depth_output.file_slots[0].path = Path(depth_path).stem + "_"
        depth_output.format.file_format = "PNG"
        depth_output.format.color_mode = "BW"
        depth_output.format.color_depth = "16"

        # Map Range: [0, DEPTH_MAX_METERS] → [0, 1]
        # Blender Depth output is in metres (distance from camera plane).
        # Using fixed bounds makes the mapping deterministic and recoverable.
        map_range = tree.nodes.new("CompositorNodeMapRange")
        map_range.location = (400, -200)
        map_range.inputs["From Min"].default_value = 0.0
        map_range.inputs["From Max"].default_value = DEPTH_MAX_METERS
        map_range.inputs["To Min"].default_value = 0.0
        map_range.inputs["To Max"].default_value = 1.0
        map_range.use_clamp = True
        tree.links.new(rl.outputs["Depth"], map_range.inputs["Value"])
        tree.links.new(map_range.outputs["Value"], depth_output.inputs[0])

    # Normal output
    if normal_path:
        normal_output = tree.nodes.new("CompositorNodeOutputFile")
        normal_output.location = (600, -400)
        normal_output.base_path = str(Path(normal_path).parent)
        normal_output.file_slots[0].path = Path(normal_path).stem + "_"
        normal_output.format.file_format = "PNG"
        normal_output.format.color_mode = "RGB"
        tree.links.new(rl.outputs["Normal"], normal_output.inputs[0])

    # Object index output
    if object_index_path:
        idx_output = tree.nodes.new("CompositorNodeOutputFile")
        idx_output.location = (600, -600)
        idx_output.base_path = str(Path(object_index_path).parent)
        idx_output.file_slots[0].path = Path(object_index_path).stem + "_"
        idx_output.format.file_format = "PNG"
        idx_output.format.color_mode = "BW"
        idx_output.format.color_depth = "16"
        idx_map_range = tree.nodes.new("CompositorNodeMapRange")
        idx_map_range.location = (400, -600)
        idx_map_range.inputs["From Min"].default_value = 0.0
        idx_map_range.inputs["From Max"].default_value = float(max(1, int(object_index_max_pass_id or 1)))
        idx_map_range.inputs["To Min"].default_value = 0.0
        idx_map_range.inputs["To Max"].default_value = 1.0
        idx_map_range.use_clamp = True
        tree.links.new(rl.outputs["IndexOB"], idx_map_range.inputs["Value"])
        tree.links.new(idx_map_range.outputs["Value"], idx_output.inputs[0])

    # Single render call for all passes
    bpy.ops.render.render(write_still=True)

    # Rename output files (Blender appends frame number)
    if depth_path:
        stem = Path(depth_path).stem
        parent = Path(depth_path).parent
        for f in parent.iterdir():
            if f.name.startswith(stem + "_") and f.suffix == ".png":
                f.rename(depth_path)
                break

    if normal_path:
        stem = Path(normal_path).stem
        parent = Path(normal_path).parent
        for f in parent.iterdir():
            if f.name.startswith(stem + "_") and f.suffix == ".png":
                f.rename(normal_path)
                break

    if object_index_path:
        stem = Path(object_index_path).stem
        parent = Path(object_index_path).parent
        for f in parent.iterdir():
            if f.name.startswith(stem + "_") and f.suffix == ".png":
                f.rename(object_index_path)
                break

    scene.use_nodes = False


def render_topdown(output_path: str, layout: list[dict], resolution: list[int]):
    """Render orthographic top-down view with object labels."""
    # Get scene bounds
    positions = []
    for obj in layout:
        bbox = obj.get("bbox", [])
        if len(bbox) >= 3:
            positions.append(bbox[:3])

    if not positions:
        return

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]

    center_x = (max(xs) + min(xs)) / 2
    center_y = (max(ys) + min(ys)) / 2
    extent = max(max(xs) - min(xs), max(ys) - min(ys)) + 4.0

    # Set up orthographic camera looking down
    cam_data = bpy.data.cameras.new("TopDownCam")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = extent
    cam_obj = bpy.data.objects.new("TopDownCam", cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)

    cam_obj.location = (center_x, center_y, 10.0)
    cam_obj.rotation_euler = (0, 0, 0)  # Looking straight down

    old_cam = bpy.context.scene.camera
    bpy.context.scene.camera = cam_obj

    # Adjust resolution
    old_res = (bpy.context.scene.render.resolution_x, bpy.context.scene.render.resolution_y)
    bpy.context.scene.render.resolution_x = resolution[0]
    bpy.context.scene.render.resolution_y = resolution[1]

    bpy.context.scene.render.filepath = output_path
    bpy.ops.render.render(write_still=True)

    # Restore
    bpy.context.scene.render.resolution_x = old_res[0]
    bpy.context.scene.render.resolution_y = old_res[1]
    bpy.context.scene.camera = old_cam
    bpy.data.objects.remove(cam_obj)
    bpy.data.cameras.remove(cam_data)


def render_scene(scene_dir: str, output_dir: str, config: dict,
                 asset_base: str = None, composed_dir: str = None,
                 config_path: str | None = None,
                 funnel_audit_out: str | None = None):
    """Render all views for a single scene."""
    scene_path = Path(scene_dir)
    output_path = Path(output_dir)
    scene_id = scene_path.name

    # Skip if already rendered (cameras.json exists and has views)
    render_cfg = config["render"]
    scene_output = output_path / scene_id
    cameras_json = scene_output / "cameras.json"
    if cameras_json.exists():
        try:
            with open(cameras_json) as f:
                existing_cams = json.load(f)
        except Exception:
            existing_cams = None
        existing_views = sum(1 for v in scene_output.iterdir()
                             if v.is_dir() and v.name.startswith("view_")
                             and (v / "rgb.png").exists())
        if isinstance(existing_cams, list):
            print(f"  Skipping (already rendered {existing_views} views)")
            return
        print("  Re-rendering (invalid cameras.json)")

    # Load layout
    with open(scene_path / "layout.json") as f:
        layout = json.load(f)

    target_cfg, rec_categories, large_categories, category_aliases = load_target_config(config, config_path)
    task_cfg = config.get("task_outputs", {})
    candidate_spacing_m = float(task_cfg.get("candidate_spacing_m", 0.4))
    max_classification_candidates = int(task_cfg.get("max_classification_candidates", 220))

    render_cfg = config["render"]

    # Set up Blender
    clear_scene()
    setup_blender_scene(render_cfg["resolution"],
                        render_depth=render_cfg.get("render_depth", True),
                        render_normal=render_cfg.get("render_normal", True))

    # When a composed directory is explicitly provided, missing GLBs are fatal.
    composed_glb = resolve_composed_glb(scene_id, composed_dir)

    # For target indexing, assign pass_index either from composed GLB names
    # or from per-object loading.
    object_index_map: dict[int, dict] = {}

    if composed_glb:
        print(f"  Loading pre-composed GLB: {composed_glb}")
        bpy.ops.import_scene.gltf(filepath=str(composed_glb))
        assign_pass_index_from_compose(layout, object_index_map=object_index_map)
    else:
        # Fallback: load individual models
        load_scene(scene_dir, layout, asset_base=asset_base, object_index_map=object_index_map)

    object_index_pass_ids = {int(k) for k in object_index_map.keys()}

    # Add lighting — indoor scene illumination
    # 1) Sun for general ambient (from an angle to simulate window light)
    bpy.ops.object.light_add(type="SUN", location=(0, 0, 5))
    sun = bpy.context.active_object
    sun.data.energy = 2.0
    sun.rotation_euler = (math.radians(45), math.radians(15), math.radians(30))

    # 2) Area lights inside the room for even illumination
    xs = [obj.get("bbox", [0])[0] for obj in layout if len(obj.get("bbox", [])) >= 3]
    ys = [obj.get("bbox", [0, 0])[1] for obj in layout if len(obj.get("bbox", [])) >= 3]
    zs = [obj.get("bbox", [0, 0, 0])[2] for obj in layout if len(obj.get("bbox", [])) >= 3]
    if xs:
        center_x = (max(xs) + min(xs)) / 2
        center_y = (max(ys) + min(ys)) / 2
        max_z = max(zs) if zs else 2.0
        # Ceiling-level area light
        bpy.ops.object.light_add(type="AREA", location=(center_x, center_y, max_z + 1.5))
        area = bpy.context.active_object
        area.data.energy = 200
        area.data.size = max(max(xs) - min(xs), max(ys) - min(ys)) * 0.8
        area.rotation_euler = (0, 0, 0)  # Pointing down

    # Set world background to slight gray for ambient fill
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    bpy.context.scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.05, 0.05, 0.05, 1.0)
        bg.inputs[1].default_value = 1.0

    # ---- Floor clutter cleanup (if scene too dense for navigation) ----
    room_bounds_for_cleanup = get_room_bounds_from_scene()
    layout, removed_ids = clean_floor_clutter(layout, room_bounds_for_cleanup)
    if removed_ids:
        remove_objects_from_blender(removed_ids)

    # Build navigation grid once per scene (embodied)
    nav_grid = None
    nav_grid_point = None
    try:
        from build_navigation_gt import build_occupancy_grid
        nav_grid = build_occupancy_grid(layout, config, scene_path=scene_path)
        cfg_point = json.loads(json.dumps(config))
        cfg_point["navigation"]["agent_radius"] = 0.0
        cfg_point["navigation"]["passable_margin"] = 0.0
        nav_grid_point = build_occupancy_grid(layout, cfg_point, scene_path=scene_path)
    except Exception as e:
        print(f"  WARNING: failed to build navigation grid via build_navigation_gt: {e}")
        nav_grid = build_nav_grid(layout, config)
        cfg_point = json.loads(json.dumps(config))
        cfg_point["navigation"]["agent_radius"] = 0.0
        cfg_point["navigation"]["passable_margin"] = 0.0
        nav_grid_point = build_nav_grid(layout, cfg_point)
        if nav_grid is None:
            print("  WARNING: fallback SimpleGrid failed; targets will be skipped")
    if DEBUG and nav_grid is not None:
        res = getattr(nav_grid, "resolution", None)
        print(f"  [debug] nav_grid={type(nav_grid).__name__} resolution={res}")

    # Compute camera positions (path-aware selection)
    camera_seed_audit: dict = {}
    cameras = compute_camera_positions(
        layout, config,
        nav_grid=nav_grid,
        target_cfg=target_cfg,
        rec_categories=rec_categories,
        large_categories=large_categories,
        category_aliases=category_aliases,
        camera_seed_audit=camera_seed_audit,
    )

    scene_output = output_path / scene_id
    scene_output.mkdir(parents=True, exist_ok=True)

    vfov_half, hfov_half = _fov_params(render_cfg["fov"],
                                       render_cfg["resolution"][0],
                                       render_cfg["resolution"][1])
    res_x, res_y = render_cfg["resolution"]

    # Dedup views by position/yaw
    dedup_pos = float(target_cfg.get("dedup_position_m", 1.0))
    dedup_yaw = math.radians(float(target_cfg.get("dedup_yaw_deg", 20.0)))
    accepted_views: list[dict] = []
    view_fact_records: list[dict] = []
    scene_funnel_audit_records: list[dict] = []
    scene_funnel_audit_records.append(summarize_camera_seed_funnel_audit(
        scene_id=scene_id,
        camera_seed_audit=camera_seed_audit,
    ))
    camera_seed_records_by_view_id = {
        int(record.get("view_id", -1)): record
        for record in camera_seed_audit.get("selected_records", [])
        if int(record.get("view_id", -1)) >= 0
    }

    for cam_info in cameras:
        view_id = cam_info["view_id"]
        camera_seed_record = camera_seed_records_by_view_id.get(int(view_id), {})
        view_dir = scene_output / f"view_{view_id}"
        view_dir.mkdir(exist_ok=True)
        target_filter_counts: dict = {}
        routing_stage_counts = {
            "pointmass_success": 0,
            "embodied_success": 0,
            "goal_zone_ok": 0,
        }
        audit_rejections: list[dict] = []
        view_bundle_routing: list[dict] = []

        # Dedup by camera position + yaw
        too_similar = False
        for prev in accepted_views:
            dx = cam_info["position"][0] - prev["position"][0]
            dy = cam_info["position"][1] - prev["position"][1]
            if math.hypot(dx, dy) < dedup_pos:
                yaw_diff = abs((cam_info["look_angle"] - prev["look_angle"] + math.pi) % (2 * math.pi) - math.pi)
                if yaw_diff < dedup_yaw:
                    too_similar = True
                    break
        if too_similar:
            scene_funnel_audit_records.append(summarize_view_funnel_audit(
                scene_id=scene_id,
                view_id=view_id,
                cam_info=cam_info,
                target_filter_counts=target_filter_counts,
                targets=[],
                target_rejections=[],
                routing_candidates=[],
                routing_stage_counts=routing_stage_counts,
                support_summary={},
                camera_seed_record=camera_seed_record,
                final_status="rejected_dedup_too_similar",
            ))
            continue

        # Set up camera
        cam_obj = setup_camera(cam_info, render_cfg["fov"])
        depsgraph = bpy.context.evaluated_depsgraph_get()
        bpy.context.view_layer.update()

        # Verify object visibility via frustum projection + ray-cast
        cam_info["visible_objects"] = verify_object_visibility(
            cam_info, layout, render_cfg["fov"],
        )
        cam_info["visible_count"] = len(cam_info["visible_objects"])
        print(f"    View {view_id}: {cam_info['visible_count']} objects visible "
              f"(ray-cast verified)")

        # Forward 1m half-cylinder clear check
        fwd_ok = forward_clear(
            cam_pos=tuple(cam_info["position"]),
            cam_yaw=cam_info["look_angle"],
            layout=layout,
            radius=float(target_cfg.get("forward_clear_radius_m", 1.0)),
            height=float(target_cfg.get("forward_clear_height_m", 1.4)),
        )
        if not fwd_ok:
            scene_funnel_audit_records.append(summarize_view_funnel_audit(
                scene_id=scene_id,
                view_id=view_id,
                cam_info=cam_info,
                target_filter_counts=target_filter_counts,
                targets=[],
                target_rejections=[],
                routing_candidates=[],
                routing_stage_counts=routing_stage_counts,
                support_summary={},
                camera_seed_record=camera_seed_record,
                final_status="rejected_forward_clear_blocked",
            ))
            bpy.data.objects.remove(cam_obj)
            continue

        # Target selection per view
        targets = []
        target_rejections = []
        if nav_grid is not None:
            min_dist = float(target_cfg.get("min_target_distance", 2.0))
            large_size = float(target_cfg.get("large_size_m", 1.0))
            max_targets = int(target_cfg.get("max_targets_per_view", 10))
            prefilter_targets = int(target_cfg.get("prefilter_targets_per_view", max_targets))
            min_area_px = float(target_cfg.get("min_target_area_px", 900.0))
            min_short_side_px = float(target_cfg.get("min_target_short_side_px", 18.0))
            min_visible_surface_ratio = float(target_cfg.get("min_target_visible_surface_ratio", 0.15))
            large_visibility_ratio = float(target_cfg.get("large_visibility_ratio", 0.70))
            targets, target_rejections, dbg_counts = build_target_candidates_from_visible_objects(
                layout=layout,
                visible_objects=cam_info.get("visible_objects", []),
                cam_position=cam_info["position"],
                cam_obj=cam_obj,
                depsgraph=depsgraph,
                fov_deg=render_cfg["fov"],
                res_x=res_x,
                res_y=res_y,
                min_dist=min_dist,
                max_targets=max_targets,
                prefilter_targets=prefilter_targets,
                min_area_px=min_area_px,
                min_short_side_px=min_short_side_px,
                min_visible_surface_ratio=min_visible_surface_ratio,
                large_visibility_ratio=large_visibility_ratio,
                rec_categories=rec_categories,
                category_aliases=category_aliases,
                large_categories=large_categories,
                large_size=large_size,
            )
            target_filter_counts = dict(dbg_counts)
            if DEBUG:
                print(f"    [debug] target_filter total={dbg_counts['total']} "
                      f"dist={dbg_counts['dist']} vis={dbg_counts['vis']} "
                      f"path={dbg_counts['path']} path_vis={dbg_counts['path_visible']} "
                      f"selected={len(targets)}")

        if not targets:
            scene_funnel_audit_records.append(summarize_view_funnel_audit(
                scene_id=scene_id,
                view_id=view_id,
                cam_info=cam_info,
                target_filter_counts=target_filter_counts,
                targets=targets,
                target_rejections=target_rejections,
                routing_candidates=[],
                routing_stage_counts=routing_stage_counts,
                support_summary={},
                camera_seed_record=camera_seed_record,
                final_status="rejected_no_targets",
            ))
            bpy.data.objects.remove(cam_obj)
            continue

        cam_info["target_candidates"] = targets
        cam_info["classification_candidates"] = build_classification_candidates_for_view(
            cam_obj,
            depsgraph,
            nav_grid_point,
            nav_grid,
            fov_deg=render_cfg["fov"],
            res_x=res_x,
            res_y=res_y,
            spacing_m=candidate_spacing_m,
            max_candidates=max_classification_candidates,
        )

        # Render all passes first so candidate semantics can use object-index pixels.
        depth_path = str(view_dir / "depth.png") if render_cfg.get("render_depth", True) else None
        normal_path = str(view_dir / "normal.png") if render_cfg.get("render_normal", True) else None
        object_index_path = str(view_dir / "object_index.png")
        render_all_passes(
            rgb_path=str(view_dir / "rgb.png"),
            depth_path=depth_path,
            normal_path=normal_path,
            object_index_path=object_index_path,
            object_index_max_pass_id=max(object_index_map) if object_index_map else None,
        )

        rgb_image = load_render_rgb_array(view_dir / "rgb.png")
        object_index_img = load_render_object_index_array(
            object_index_path,
            max_pass_id=max(object_index_map) if object_index_map else None,
        )
        depth_map = load_render_depth_map(depth_path) if depth_path else None

        assign_visible_target_colors(
            cam_info.get("target_candidates", []),
            rgb_image=rgb_image,
            object_index_img=object_index_img,
        )

        structural_pass_ids = infer_structural_pass_ids(object_index_img, object_index_pass_ids)
        nonblocking_object_pass_ids = collect_nonblocking_object_pass_ids(
            object_index_map,
            non_blocking_categories=set(target_cfg.get(
                "projection_non_blocking_categories",
                sorted(DEFAULT_PROJECTION_NON_BLOCKING_CATEGORIES),
            )),
        )
        dense_candidate_spacing_m = float(target_cfg.get("dense_candidate_spacing_m", 0.1))
        floor_mask = getattr(nav_grid_point, "floor_mask", None) if nav_grid_point is not None else None
        floor_stride = max(1, int(round(dense_candidate_spacing_m / float(nav_grid_point.resolution)))) if nav_grid_point is not None else 1
        floor_z_ref = estimate_floor_z(bpy.context.scene, depsgraph, nav_grid_point, floor_mask, floor_stride) if nav_grid_point is not None else 0.0
        dense_view_candidates = build_classification_candidates_for_view(
            cam_obj,
            depsgraph,
            nav_grid_point,
            nav_grid,
            fov_deg=render_cfg["fov"],
            res_x=res_x,
            res_y=res_y,
            spacing_m=dense_candidate_spacing_m,
            max_candidates=0,
        )
        dense_world_candidates = build_world_routing_candidates(
            nav_grid_point,
            nav_grid,
            spacing_m=dense_candidate_spacing_m,
            max_candidates=0,
            floor_z=floor_z_ref,
        )
        bottom_center_anchor_world = compute_bottom_center_floor_anchor(
            cam_obj,
            bpy.context.scene,
            floor_z_ref,
        )
        if DEBUG and bottom_center_anchor_world is not None and nav_grid is not None and nav_grid_point is not None:
            dbg_ax = float(bottom_center_anchor_world[0])
            dbg_ay = float(bottom_center_anchor_world[1])
            dbg_gx, dbg_gy = nav_grid.world_to_grid(dbg_ax, dbg_ay)
            dbg_pg = nav_grid_point.world_to_grid(dbg_ax, dbg_ay)
            dbg_proj = project_world_point(
                cam_obj.matrix_world.inverted(),
                mathutils.Vector((
                    float(bottom_center_anchor_world[0]),
                    float(bottom_center_anchor_world[1]),
                    float(bottom_center_anchor_world[2]),
                )),
                vfov_half,
                hfov_half,
                res_x,
                res_y,
                cam_obj=cam_obj,
                scene=bpy.context.scene,
            )
            dbg_visible = raycast_visible(
                bpy.context.scene,
                depsgraph,
                cam_obj.matrix_world.translation.copy(),
                mathutils.Vector((
                    float(bottom_center_anchor_world[0]),
                    float(bottom_center_anchor_world[1]),
                    float(bottom_center_anchor_world[2]),
                )),
                tol=0.08,
            )
            print(
                "    [debug] fixed_start_anchor "
                f"view_id={view_id} "
                f"world={[round(float(v), 3) for v in bottom_center_anchor_world]} "
                f"grid_embodied={[int(dbg_gx), int(dbg_gy)]} embodied_free={bool(nav_grid.is_free(dbg_gx, dbg_gy))} "
                f"grid_point={[int(dbg_pg[0]), int(dbg_pg[1])]} point_free={bool(nav_grid_point.is_free(int(dbg_pg[0]), int(dbg_pg[1])))} "
                f"projected={None if dbg_proj is None else dbg_proj['pixel_xy']} "
                f"raycast_visible={bool(dbg_visible)}"
            )
        fixed_bottom_center_start = build_fixed_bottom_center_start_candidate(
            bottom_center_anchor_world,
            cam_obj,
            depsgraph,
            bpy.context.scene,
            nav_grid_point,
            nav_grid,
            vfov_half,
            hfov_half,
            res_x,
            res_y,
        )

        classification_candidates = cam_info.get("classification_candidates", [])
        changed_cnt = apply_candidate_pixel_deobjectification(
            classification_candidates,
            object_index_img,
            object_index_pass_ids,
            nonblocking_pass_ids=nonblocking_object_pass_ids,
            structural_pass_ids=structural_pass_ids,
            depth_map=depth_map,
            depth_eps_m=0.08,
            depth_window_radius_px=2,
        )
        dense_changed_cnt = apply_candidate_pixel_deobjectification(
            dense_view_candidates,
            object_index_img,
            object_index_pass_ids,
            nonblocking_pass_ids=nonblocking_object_pass_ids,
            structural_pass_ids=structural_pass_ids,
            depth_map=depth_map,
            depth_eps_m=0.08,
            depth_window_radius_px=2,
        )
        cam_info["classification_candidates"] = classification_candidates
        if DEBUG and (changed_cnt > 0 or dense_changed_cnt > 0):
            print(
                f"    [debug] candidate_cleanup sparse_changed={changed_cnt} dense_changed={dense_changed_cnt}"
            )
        classification_candidates = include_fixed_start_candidate(
            classification_candidates,
            fixed_bottom_center_start,
        )
        cam_info["classification_candidates"] = classification_candidates

        # Canonical per-view fact bundle (single source of truth for GT/VQA).
        classification_candidates = cam_info.get("classification_candidates", [])
        goal_ring_tolerance_m = float(target_cfg.get(
            "goal_ring_tolerance_m",
            max(float(getattr(nav_grid, "resolution", dense_candidate_spacing_m)), dense_candidate_spacing_m, 0.05),
        ))
        for idx, tgt in enumerate(cam_info.get("target_candidates", []), start=1):
            target_obj = next((o for o in layout if int(o.get("id", -1)) == int(tgt.get("target_id", -1))), None)
            if target_obj is None:
                continue
            bbox = target_obj.get("bbox", [])
            fixed_start_candidate = dict(fixed_bottom_center_start) if fixed_bottom_center_start is not None else None
            projection_pad_px = int(target_cfg.get("projection_safe_pad_px", 2))
            obstacle_mask = build_blocking_object_mask(
                object_index_img,
                object_index_map,
                excluded_obj_id=int(tgt.get("target_id", -1)),
                non_blocking_categories=set(target_cfg.get(
                    "projection_non_blocking_categories",
                    sorted(DEFAULT_PROJECTION_NON_BLOCKING_CATEGORIES),
                )),
            )
            point_goal_candidates = select_goal_routing_candidates(
                dense_world_candidates,
                bbox,
                ring_tolerance_m=goal_ring_tolerance_m,
                feasible_key="pointmass_walkable",
            )
            embodied_goal_candidates = select_goal_routing_candidates(
                dense_world_candidates,
                bbox,
                ring_tolerance_m=goal_ring_tolerance_m,
                feasible_key="embodied_feasible",
            )

            if DEBUG:
                print(
                    "    [debug] dense_path_candidates "
                    f"target_id={int(tgt.get('target_id', -1))} "
                    f"starts={1 if fixed_start_candidate is not None else 0} "
                    f"goals_point={len(point_goal_candidates)} goals_embodied={len(embodied_goal_candidates)} "
                    f"dense_view_total={len(dense_view_candidates)} "
                    f"dense_world_total={len(dense_world_candidates)}"
                )

            def _materialize_routing_mode(
                *,
                mode_name: str,
                feasible_key: str,
                route_nav_grid,
                goal_candidates_mode: list[dict],
                candidate_pool: list[dict],
            ) -> tuple[dict | None, dict | None]:
                if not goal_candidates_mode:
                    return None, {"reason": f"no_goal_candidates_{mode_name}"}

                working_candidates = [dict(candidate) for candidate in candidate_pool]
                search_path_world = find_shortest_candidate_path(
                    route_nav_grid,
                    [fixed_start_candidate] if fixed_start_candidate is not None else [],
                    goal_candidates_mode,
                    max_start_candidates=int(target_cfg.get("dense_path_max_starts", 18)),
                    max_goal_candidates=int(target_cfg.get("dense_path_max_goals", 18)),
                )
                if not search_path_world:
                    return None, {"reason": f"no_dense_{mode_name}_path"}

                optimal_path_world_mode = build_continuous_optimal_path(search_path_world, route_nav_grid)
                if not optimal_path_world_mode or len(optimal_path_world_mode) < 2:
                    return None, {"reason": f"{mode_name}_dense_path_too_short"}

                canonical_sparse_path_world_mode = sparsify_gt_path_keypoints(
                    optimal_path_world_mode,
                    turn_angle_deg=float(target_cfg.get("gt_turn_angle_deg", 20.0)),
                    min_segment_m=float(target_cfg.get("gt_min_segment_m", 0.6)),
                )
                projected_dense_path_mode = project_path_world_to_image(
                    optimal_path_world_mode,
                    cam_obj,
                    vfov_half,
                    hfov_half,
                    res_x,
                    res_y,
                    scene=bpy.context.scene,
                )
                projected_dense_path_mode = ensure_start_anchor_projection(
                    projected_dense_path_mode,
                    fixed_start_candidate.get("image_xy") if fixed_start_candidate is not None else None,
                )
                dense_path_projection_safe_mode, dense_path_first_blocking_segment_mode = evaluate_dense_path_projection(
                    projected_dense_path_mode,
                    obstacle_mask=obstacle_mask,
                    pad_px=projection_pad_px,
                )
                canonical_sparse_path_world_mode, _, _ = refine_routing_outputs_for_projection(
                    dense_path_world=optimal_path_world_mode,
                    gt_path_keypoints_world=canonical_sparse_path_world_mode,
                    path_wp_point=[],
                    path_wp_embodied=[],
                    classification_candidates=working_candidates,
                    obstacle_mask=obstacle_mask,
                    projected_dense_path=projected_dense_path_mode,
                    pad_px=projection_pad_px,
                    nav_grid=route_nav_grid,
                )
                canonical_sparse_path_image_mode = project_path_world_to_image(
                    canonical_sparse_path_world_mode,
                    cam_obj,
                    vfov_half,
                    hfov_half,
                    res_x,
                    res_y,
                    scene=bpy.context.scene,
                )
                canonical_sparse_path_image_mode = ensure_start_anchor_projection(
                    canonical_sparse_path_image_mode,
                    fixed_start_candidate.get("image_xy") if fixed_start_candidate is not None else None,
                )
                keypoint_projection_safe_mode = projected_polyline_collision_free(
                    canonical_sparse_path_image_mode,
                    obstacle_mask=obstacle_mask,
                    pad_px=projection_pad_px,
                    require_all_points_visible=True,
                )
                if not canonical_sparse_path_world_mode or len(canonical_sparse_path_world_mode) < 2:
                    return None, {
                        "reason": f"{mode_name}_projection_keypoints_blocked",
                        "gt_path_keypoints_len": len(canonical_sparse_path_world_mode),
                    }
                if not keypoint_projection_safe_mode:
                    return None, {
                        "reason": f"{mode_name}_projection_keypoints_blocked",
                        "gt_path_keypoints_len": len(canonical_sparse_path_world_mode),
                    }

                try:
                    sparse_path_waypoint_ids_mode = ensure_sparse_routing_keypoint_candidates(
                        working_candidates,
                        canonical_sparse_path_world_mode,
                        canonical_sparse_path_image_mode,
                        feasible_key=feasible_key,
                        grid_point=nav_grid_point,
                        grid_embodied=nav_grid,
                        cam_obj=cam_obj,
                        scene=bpy.context.scene,
                        vfov_half=vfov_half,
                        hfov_half=hfov_half,
                        res_x=res_x,
                        res_y=res_y,
                    )
                except ValueError as exc:
                    return None, {
                        "reason": f"{mode_name}_sparse_keypoint_candidate_invalid",
                        "detail": str(exc),
                    }

                waypoint_lookup_mode = {
                    int(candidate.get("waypoint_id", -1)): candidate
                    for candidate in working_candidates
                }
                point_dense_ids_mode, embodied_dense_ids_mode = collect_dense_debug_waypoint_ids(
                    path_world=optimal_path_world_mode,
                    classification_candidates=working_candidates,
                    obstacle_mask=obstacle_mask,
                    pad_px=projection_pad_px,
                )
                debug_dense_waypoint_ids_mode = (
                    point_dense_ids_mode
                    if mode_name == "pointmass"
                    else embodied_dense_ids_mode
                )
                path_projection_safe_mode = path_ids_projection_collision_free(
                    sparse_path_waypoint_ids_mode,
                    waypoint_lookup_mode,
                    obstacle_mask=obstacle_mask,
                    pad_px=projection_pad_px,
                )
                world_safe_mode = path_ids_world_collision_free(
                    sparse_path_waypoint_ids_mode,
                    waypoint_lookup_mode,
                    nav_grid=route_nav_grid,
                )
                goal_distance_threshold_mode = float(
                    goal_candidates_mode[0].get("goal_ring_threshold_m", 0.0)
                )
                goal_in_target_zone_mode = routing_endpoint_within_goal_radius(
                    optimal_path_world_mode,
                    bbox,
                    goal_distance_threshold_mode,
                )
                continuous_collision_free_mode = check_embodied_collision_free(
                    optimal_path_world_mode,
                    route_nav_grid,
                )

                if not goal_in_target_zone_mode or not continuous_collision_free_mode:
                    return None, {
                        "reason": f"{mode_name}_invariant_failed",
                        "goal_in_target_zone": bool(goal_in_target_zone_mode),
                        "continuous_collision_free": bool(continuous_collision_free_mode),
                    }

                if len(sparse_path_waypoint_ids_mode) < 2:
                    return None, {
                        "reason": f"{mode_name}_path_projection_failed",
                        "path_wp_len": len(sparse_path_waypoint_ids_mode),
                        "path_wp_dense_len": len(debug_dense_waypoint_ids_mode),
                    }

                if not dense_path_projection_safe_mode:
                    if DEBUG:
                        visible_projected_points = sum(
                            1 for point in projected_dense_path_mode if point is not None
                        ) if projected_dense_path_mode else 0
                        print(
                            "    [debug] dense_projection_block "
                            f"view_id={view_id} target_id={int(tgt.get('target_id', -1))} "
                            f"mode={mode_name} "
                            f"first_blocking_segment={dense_path_first_blocking_segment_mode} "
                            f"anchor_xy={None if fixed_start_candidate is None else fixed_start_candidate.get('image_xy')} "
                            f"visible_projected_points={visible_projected_points} "
                            f"path_len={len(optimal_path_world_mode)} "
                            f"path_start={[round(float(v), 3) for v in optimal_path_world_mode[0]]} "
                            f"path_end={[round(float(v), 3) for v in optimal_path_world_mode[-1]]} "
                            f"proj_prefix={projected_dense_path_mode[:5] if projected_dense_path_mode else []} "
                            f"proj_start={None if not projected_dense_path_mode else projected_dense_path_mode[0]} "
                            f"proj_end={None if not projected_dense_path_mode else projected_dense_path_mode[-1]}"
                        )
                    return None, {"reason": f"{mode_name}_dense_projection_path_blocked"}

                if not (path_projection_safe_mode and world_safe_mode):
                    return None, {
                        "reason": f"{mode_name}_projection_path_blocked",
                        "path_projection_safe": bool(path_projection_safe_mode),
                        "world_safe": bool(world_safe_mode),
                        "path_wp_len": len(sparse_path_waypoint_ids_mode),
                        "path_wp_dense_len": len(debug_dense_waypoint_ids_mode),
                    }

                goal_anchor_wp_mode = waypoint_lookup_mode.get(int(sparse_path_waypoint_ids_mode[-1]), None)
                if goal_anchor_wp_mode is None:
                    return None, {"reason": f"{mode_name}_goal_anchor_missing"}

                sparse_goal_in_target_zone_mode = routing_endpoint_within_goal_radius(
                    canonical_sparse_path_world_mode,
                    bbox,
                    goal_distance_threshold_mode,
                )
                display_goal_in_target_zone_mode = distance_to_oriented_bbox_2d(
                    float(goal_anchor_wp_mode.get("world_xyz", [0.0, 0.0])[0]),
                    float(goal_anchor_wp_mode.get("world_xyz", [0.0, 0.0])[1]),
                    bbox,
                ) <= goal_distance_threshold_mode
                if not sparse_goal_in_target_zone_mode or not display_goal_in_target_zone_mode:
                    return None, {
                        "reason": f"{mode_name}_final_goal_outside_target_zone",
                        "sparse_goal_in_target_zone": bool(sparse_goal_in_target_zone_mode),
                        "display_goal_in_target_zone": bool(display_goal_in_target_zone_mode),
                    }

                try:
                    validate_canonical_sparse_path(
                        optimal_path_world=optimal_path_world_mode,
                        canonical_sparse_path_world=canonical_sparse_path_world_mode,
                        nav_grid=route_nav_grid,
                        obstacle_mask=obstacle_mask,
                        projected_points=canonical_sparse_path_image_mode,
                        pad_px=projection_pad_px,
                    )
                except ValueError as exc:
                    return None, {
                        "reason": f"{mode_name}_invalid_canonical_sparse_path",
                        "detail": str(exc),
                    }

                routing_complexity_mode = build_routing_complexity_metadata(
                    dense_path_world=optimal_path_world_mode,
                    sparse_path_world=canonical_sparse_path_world_mode,
                    projected_dense_path=projected_dense_path_mode,
                    nav_grid=route_nav_grid,
                    bbox_metrics=tgt.get("bbox_metrics"),
                    visibility_ratio=float(tgt.get("visibility_ratio", 0.0)),
                    dense_path_projection_safe=bool(dense_path_projection_safe_mode),
                    sparse_path_projection_safe=bool(
                        path_projection_safe_mode
                        and world_safe_mode
                        and keypoint_projection_safe_mode
                    ),
                    start_in_bottom_band=bool(fixed_start_candidate is not None),
                )

                return {
                    "classification_candidates": working_candidates,
                    "optimal_path_world": optimal_path_world_mode,
                    "optimal_path_image_xy": projected_dense_path_mode,
                    "canonical_sparse_path_world": canonical_sparse_path_world_mode,
                    "canonical_sparse_path_image_xy": canonical_sparse_path_image_mode,
                    "path_waypoint_ids": list(sparse_path_waypoint_ids_mode),
                    "debug_dense_waypoint_ids": list(debug_dense_waypoint_ids_mode),
                    "goal_distance_threshold_m": round(goal_distance_threshold_mode, 4),
                    "goal_in_target_zone": bool(goal_in_target_zone_mode),
                    "continuous_collision_free": bool(continuous_collision_free_mode),
                    "routing_complexity": routing_complexity_mode,
                }, None

            classification_candidates_before_target = [dict(candidate) for candidate in classification_candidates]
            target_candidate_pool = [dict(candidate) for candidate in classification_candidates]
            point_route, point_rejection = _materialize_routing_mode(
                mode_name="pointmass",
                feasible_key="pointmass_walkable",
                route_nav_grid=nav_grid_point,
                goal_candidates_mode=point_goal_candidates,
                candidate_pool=target_candidate_pool,
            )
            if point_rejection is not None:
                classification_candidates[:] = classification_candidates_before_target
                target_rejections.append({
                    "target_id": int(tgt.get("target_id", -1)),
                    **point_rejection,
                })
                continue
            routing_stage_counts["pointmass_success"] += 1

            target_candidate_pool = point_route.pop("classification_candidates")
            embodied_route, embodied_rejection = _materialize_routing_mode(
                mode_name="embodied",
                feasible_key="embodied_feasible",
                route_nav_grid=nav_grid,
                goal_candidates_mode=embodied_goal_candidates,
                candidate_pool=target_candidate_pool,
            )
            if embodied_rejection is not None:
                classification_candidates[:] = classification_candidates_before_target
                target_rejections.append({
                    "target_id": int(tgt.get("target_id", -1)),
                    **embodied_rejection,
                })
                continue
            routing_stage_counts["embodied_success"] += 1

            classification_candidates[:] = embodied_route.pop("classification_candidates")
            waypoint_lookup = {int(c.get("waypoint_id", -1)): c for c in classification_candidates}
            path_wp_point = list(point_route.get("path_waypoint_ids", []))
            path_wp_embodied = list(embodied_route.get("path_waypoint_ids", []))
            path_wp_point_dense = list(point_route.get("debug_dense_waypoint_ids", []))
            path_wp_embodied_dense = list(embodied_route.get("debug_dense_waypoint_ids", []))
            start_in_forward_sector = fixed_start_candidate is not None
            goal_in_target_zone = bool(point_route.get("goal_in_target_zone", False)) and bool(
                embodied_route.get("goal_in_target_zone", False)
            )
            embodied_collision_free = bool(embodied_route.get("continuous_collision_free", False))

            start_anchor_wp = (
                fixed_start_candidate
                or waypoint_lookup.get(int(path_wp_embodied[0]), None)
                or waypoint_lookup.get(int(path_wp_point[0]), None)
            )
            goal_anchor_wp = (
                waypoint_lookup.get(int(path_wp_embodied[-1]), None)
                or waypoint_lookup.get(int(path_wp_point[-1]), None)
            )
            if start_anchor_wp is None or goal_anchor_wp is None:
                classification_candidates[:] = classification_candidates_before_target
                target_rejections.append({
                    "target_id": int(tgt.get("target_id", -1)),
                    "reason": "anchor_missing",
                })
                continue

            routing_complexity = merge_point_and_embodied_routing_complexity(
                point_complexity=point_route.get("routing_complexity", {}),
                embodied_complexity=embodied_route.get("routing_complexity", {}),
                point_dense_path_world=point_route.get("optimal_path_world", []),
                embodied_dense_path_world=embodied_route.get("optimal_path_world", []),
                target_bbox=bbox,
                goal_distance_threshold_m=float(embodied_route.get("goal_distance_threshold_m", 0.0)),
                embodied_nav_grid=nav_grid,
            )

            routing_material_color_rgb = tgt.get("material_color_rgb")
            if not isinstance(routing_material_color_rgb, (list, tuple)) or len(routing_material_color_rgb) < 3:
                routing_material_color_rgb = None
            else:
                routing_material_color_rgb = [round(float(v), 4) for v in routing_material_color_rgb[:3]]

            view_bundle_routing.append({
                "routing_id": f"routing_v{view_id}_c{idx:03d}",
                "target_id": int(tgt.get("target_id", -1)),
                "target_category": tgt.get("category", "unknown"),
                "canonical_category": tgt.get("canonical_category", tgt.get("category", "unknown")),
                "semantic_group_id": tgt.get("semantic_group_id", tgt.get("canonical_category", tgt.get("category", "unknown"))),
                "reference_family_id": tgt.get("reference_family_id", tgt.get("semantic_group_id", tgt.get("canonical_category", tgt.get("category", "unknown")))),
                "material_color_rgb": routing_material_color_rgb,
                "material_color_name": quantize_material_color_name(routing_material_color_rgb),
                "visible_color_rgb": tgt.get("visible_color_rgb"),
                "visible_color_name": tgt.get("visible_color_name", "unknown"),
                "distance_m": float(tgt.get("distance", 0.0)),
                "path_length_raw_m": float(routing_complexity["path_length_raw_m"]),
                "path_length_smoothed_m": float(routing_complexity["path_length_sparse_m"]),
                "path_smoothing_gain_m": float(routing_complexity["path_smoothing_gain_m"]),
                "optimal_path_world": [
                    [round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3)]
                    for p in point_route.get("optimal_path_world", [])
                ],
                "optimal_path_image_xy": list(point_route.get("optimal_path_image_xy", [])),
                "optimal_length_m": round(path_length_m(point_route.get("optimal_path_world", [])), 3),
                "canonical_sparse_path_world": [
                    [round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3)]
                    for p in point_route.get("canonical_sparse_path_world", [])
                ],
                "canonical_sparse_path_image_xy": list(point_route.get("canonical_sparse_path_image_xy", [])),
                "canonical_sparse_length_m": round(path_length_m(point_route.get("canonical_sparse_path_world", [])), 3),
                "optimal_path_world_embodied": [
                    [round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3)]
                    for p in embodied_route.get("optimal_path_world", [])
                ],
                "optimal_path_image_xy_embodied": list(embodied_route.get("optimal_path_image_xy", [])),
                "optimal_length_m_embodied": round(path_length_m(embodied_route.get("optimal_path_world", [])), 3),
                "canonical_sparse_path_world_embodied": [
                    [round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3)]
                    for p in embodied_route.get("canonical_sparse_path_world", [])
                ],
                "canonical_sparse_path_image_xy_embodied": list(embodied_route.get("canonical_sparse_path_image_xy", [])),
                "canonical_sparse_length_m_embodied": round(
                    path_length_m(embodied_route.get("canonical_sparse_path_world", [])),
                    3,
                ),
                "gt_path_keypoints_world": list(point_route.get("canonical_sparse_path_world", [])),
                "gt_path_keypoints_image_xy": list(point_route.get("canonical_sparse_path_image_xy", [])),
                "debug_dense_path_world": [
                    [round(float(p[0]), 3), round(float(p[1]), 3), round(float(p[2]), 3)]
                    for p in point_route.get("optimal_path_world", [])
                ],
                "debug_dense_path_image_xy": list(point_route.get("optimal_path_image_xy", [])),
                "debug_dense_waypoint_ids_pointmass": path_wp_point_dense,
                "debug_dense_waypoint_ids_embodied": path_wp_embodied_dense,
                "gt_path_waypoint_ids_pointmass": path_wp_point,
                "gt_path_waypoint_ids_embodied": path_wp_embodied,
                "start_anchor": {
                    "waypoint_id": int(start_anchor_wp.get("waypoint_id", -1)),
                    "world_xy": [
                        float(start_anchor_wp.get("world_xyz", [0.0, 0.0])[0]),
                        float(start_anchor_wp.get("world_xyz", [0.0, 0.0])[1]),
                    ],
                },
                "goal_anchor": {
                    "waypoint_id": int(goal_anchor_wp.get("waypoint_id", -1)),
                    "world_xy": [
                        float(goal_anchor_wp.get("world_xyz", [0.0, 0.0])[0]),
                        float(goal_anchor_wp.get("world_xyz", [0.0, 0.0])[1]),
                    ],
                },
                "invariants": {
                    "start_in_forward_sector": bool(start_in_forward_sector),
                    "goal_in_target_zone": bool(goal_in_target_zone),
                    "embodied_collision_free": bool(embodied_collision_free),
                },
                "routing_complexity": routing_complexity,
            })
            if goal_in_target_zone:
                routing_stage_counts["goal_zone_ok"] += 1

        view_bundle_routing, instruction_rejections = annotate_routing_instruction_metadata(
            view_bundle_routing,
            target_rule="nearest",
            distance_tie_eps_m=float(target_cfg.get("instruction_distance_tie_eps_m", 1e-3)),
        )
        audit_rejections = list(target_rejections) + list(instruction_rejections)
        view_bundle_routing = enrich_routing_complexity_metadata(view_bundle_routing)
        view_bundle_routing = enrich_navigation_fact_metadata(view_bundle_routing)
        if not view_bundle_routing:
            if DEBUG and target_rejections:
                reason_counts = Counter(str(item.get("reason", "unknown")) for item in target_rejections)
                print(f"    [debug] routing_rejections={dict(reason_counts)}")
            scene_funnel_audit_records.append(summarize_view_funnel_audit(
                scene_id=scene_id,
                view_id=view_id,
                cam_info=cam_info,
                target_filter_counts=target_filter_counts,
                targets=targets,
                target_rejections=audit_rejections,
                routing_candidates=[],
                routing_stage_counts=routing_stage_counts,
                support_summary={},
                camera_seed_record=camera_seed_record,
                final_status="rejected_no_routing_candidates",
            ))
            bpy.data.objects.remove(cam_obj)
            continue
        min_nontrivial_routes = int(target_cfg.get("min_nontrivial_routes_per_view", 1))
        require_reference_support = bool(target_cfg.get("require_reference_support_in_view", False))
        require_geometry_support = bool(target_cfg.get("require_geometry_support_in_view", False))
        support_summary = routing_support_summary(view_bundle_routing)
        if not routing_view_has_support(
            view_bundle_routing,
            min_nontrivial_routes=min_nontrivial_routes,
            require_reference_support=require_reference_support,
            require_geometry_support=require_geometry_support,
        ):
            target_rejections.append({
                "reason": "view_only_trivial_routes",
                **support_summary,
            })
            audit_rejections.append({
                "reason": "view_only_trivial_routes",
                **support_summary,
            })
            scene_funnel_audit_records.append(summarize_view_funnel_audit(
                scene_id=scene_id,
                view_id=view_id,
                cam_info=cam_info,
                target_filter_counts=target_filter_counts,
                targets=targets,
                target_rejections=audit_rejections,
                routing_candidates=view_bundle_routing,
                routing_stage_counts=routing_stage_counts,
                support_summary=support_summary,
                camera_seed_record=camera_seed_record,
                final_status="rejected_view_only_trivial_routes",
            ))
            bpy.data.objects.remove(cam_obj)
            continue
        max_routing_candidates = int(target_cfg.get("max_targets_per_view", len(view_bundle_routing)))
        if len(view_bundle_routing) > max_routing_candidates:
            selected_routing = select_diverse_routing_candidates(view_bundle_routing, max_routing_candidates)
            selected_target_ids = {int(candidate.get("target_id", -1)) for candidate in selected_routing}
            for candidate in view_bundle_routing:
                target_id = int(candidate.get("target_id", -1))
                if target_id in selected_target_ids:
                    continue
                reference_facts = candidate.get("navigation_reference_facts", {}) or {}
                geometry_facts = candidate.get("navigation_geometry_facts", {}) or {}
                target_rejections.append({
                    "target_id": target_id,
                    "reason": "pruned_by_route_diversity_budget",
                    "reference_resolution_mode": reference_facts.get("reference_resolution_mode"),
                    "geometry_profile": geometry_facts.get("geometry_profile"),
                })
                audit_rejections.append({
                    "target_id": target_id,
                    "reason": "pruned_by_route_diversity_budget",
                    "reference_resolution_mode": reference_facts.get("reference_resolution_mode"),
                    "geometry_profile": geometry_facts.get("geometry_profile"),
                })
            view_bundle_routing = selected_routing
        if not routing_view_has_support(
            view_bundle_routing,
            min_nontrivial_routes=min_nontrivial_routes,
            require_reference_support=require_reference_support,
            require_geometry_support=require_geometry_support,
        ):
            target_rejections.append({
                "reason": "selected_routes_lack_required_support",
                **routing_support_summary(view_bundle_routing),
            })
            audit_rejections.append({
                "reason": "selected_routes_lack_required_support",
                **routing_support_summary(view_bundle_routing),
            })
            scene_funnel_audit_records.append(summarize_view_funnel_audit(
                scene_id=scene_id,
                view_id=view_id,
                cam_info=cam_info,
                target_filter_counts=target_filter_counts,
                targets=targets,
                target_rejections=audit_rejections,
                routing_candidates=view_bundle_routing,
                routing_stage_counts=routing_stage_counts,
                support_summary=routing_support_summary(view_bundle_routing),
                camera_seed_record=camera_seed_record,
                final_status="rejected_selected_routes_lack_required_support",
            ))
            bpy.data.objects.remove(cam_obj)
            continue
        ambiguous_ids = {int(item.get("target_id", -1)) for item in instruction_rejections}
        for candidate in view_bundle_routing:
            identity = candidate.get("instruction_identity", {}) or {}
            selection = candidate.get("instruction_selection", {}) or {}
            raw_visible_color_name = _normalized_instruction_text(candidate.get("visible_color_name")) or identity.get(
                "visible_color_name"
            )
            raw_material_color_name = _normalized_instruction_text(candidate.get("material_color_name")) or identity.get(
                "material_color_name"
            )
            candidate["reference_family_id"] = (
                candidate.get("reference_family_id")
                or identity.get("reference_family_id")
                or candidate.get("semantic_group_id")
                or candidate.get("canonical_category")
                or candidate.get("target_category")
            )
            candidate["visible_color_name"] = raw_visible_color_name
            candidate["material_color_name"] = raw_material_color_name or candidate.get("material_color_name")
            candidate["reference_color_name"] = (
                candidate.get("reference_color_name")
                or identity.get("reference_color_name")
                or raw_visible_color_name
                or raw_material_color_name
            )
            candidate["reference_color_source"] = (
                candidate.get("reference_color_source")
                or identity.get("reference_color_source")
                or ("visible" if raw_visible_color_name else None)
                or ("material" if raw_material_color_name else None)
            )
            candidate["selection_status"] = selection.get("selection_status")
            candidate["pruned_reason"] = None
            if int(candidate.get("target_id", -1)) in ambiguous_ids:
                candidate["pruned_reason"] = "instruction_identity_ambiguous"
            elif not bool(selection.get("is_rule_winner", False)):
                candidate["pruned_reason"] = "instruction_rule_not_selected"

        # Keep camera target_candidates aligned with canonical routing facts.
        routing_lookup = {int(rc["target_id"]): rc for rc in view_bundle_routing}
        final_targets = []
        for t in cam_info.get("target_candidates", []):
            target_id = int(t.get("target_id", -1))
            routing = routing_lookup.get(target_id)
            if routing is None:
                continue
            merged = dict(t)
            merged["path_world"] = list(routing.get("optimal_path_world", []))
            merged["optimal_path_world"] = list(routing.get("optimal_path_world", []))
            merged["canonical_sparse_path_world"] = list(routing.get("canonical_sparse_path_world", []))
            merged["optimal_path_world_embodied"] = list(routing.get("optimal_path_world_embodied", []))
            merged["canonical_sparse_path_world_embodied"] = list(routing.get("canonical_sparse_path_world_embodied", []))
            merged["gt_path_keypoints_world"] = list(routing.get("canonical_sparse_path_world", []))
            final_targets.append(merged)
        cam_info["target_candidates"] = final_targets

        # Save per-view visibility info
        with open(view_dir / "visible_objects.json", "w") as f:
            json.dump(cam_info["visible_objects"], f, indent=2)

        affordance_payload = build_affordance_fact_metadata(cam_info.get("classification_candidates", []))
        view_fact_records.append({
            "view_id": int(view_id),
            "camera": {
                "position": [float(v) for v in cam_info.get("position", [0.0, 0.0, 0.0])],
                "rotation": [float(v) for v in cam_info.get("rotation", [0.0, 0.0, 0.0])],
                "fov": float(render_cfg["fov"]),
                "resolution": [int(res_x), int(res_y)],
            },
            "visible_objects": list(cam_info.get("visible_objects", [])),
            "affordance_validity": dict(affordance_payload.get("affordance_validity", {})),
            "affordance_axes": dict(affordance_payload.get("affordance_axes", {})),
            "affordance_tier": affordance_payload.get("affordance_tier"),
            "affordance_complexity": dict(affordance_payload.get("affordance_complexity", {})),
            "classification_candidates": cam_info.get("classification_candidates", []),
            "routing_candidates": view_bundle_routing,
            "rejection_reasons": target_rejections,
        })

        accepted_views.append(cam_info)
        scene_funnel_audit_records.append(summarize_view_funnel_audit(
            scene_id=scene_id,
            view_id=view_id,
            cam_info=cam_info,
            target_filter_counts=target_filter_counts,
            targets=cam_info.get("target_candidates", []),
            target_rejections=audit_rejections,
            routing_candidates=view_bundle_routing,
            routing_stage_counts=routing_stage_counts,
            support_summary=routing_support_summary(view_bundle_routing),
            camera_seed_record=camera_seed_record,
            final_status="accepted",
        ))

        # Clean up camera
        bpy.data.objects.remove(cam_obj)

    # Save camera info (with visibility + target data)
    total_objects = len([o for o in layout if o.get("model_uid", "")])
    all_visible_ids = set()
    for cam in accepted_views:
        for vo in cam.get("visible_objects", []):
            all_visible_ids.add(vo["id"])
    print(f"  Total coverage: {len(all_visible_ids)}/{total_objects} objects "
          f"visible across all views")

    with open(scene_output / "cameras.json", "w") as f:
        json.dump(accepted_views, f, indent=2)

    scene_facts = build_scene_nav_facts(scene_id=scene_id, views=view_fact_records)
    save_scene_nav_facts(scene_output, scene_facts)
    for view in scene_facts.get("views", []):
        view_id = int(view.get("view_id", 0))
        view_dir = scene_output / f"view_{view_id}"
        view_dir.mkdir(exist_ok=True)
        view_bundle = build_view_bundle_from_scene_nav_facts(scene_facts, view_id=view_id)
        with open(view_dir / "next_view_bundle_v3.json", "w") as f:
            json.dump(view_bundle, f, indent=2)

    if object_index_map:
        with open(scene_output / "object_index_map.json", "w") as f:
            json.dump(object_index_map, f, indent=2)

    # Save cleanup info if objects were removed
    if removed_ids:
        with open(scene_output / "floor_cleanup.json", "w") as f:
            json.dump({
                "removed_obj_ids": removed_ids,
                "n_removed": len(removed_ids),
            }, f, indent=2)

    # Render top-down view
    if render_cfg["render_topdown"]:
        render_topdown(
            str(scene_output / "topdown.png"),
            layout,
            render_cfg["topdown_resolution"],
        )

    write_scene_funnel_audit_records(
        funnel_audit_out,
        scene_id=scene_id,
        records=scene_funnel_audit_records,
    )


def main_blender():
    """Entry point when running inside Blender."""
    # Parse args after '--'
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--scenes-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--asset-base", default=None, help="Path to asset_library/ directory")
    parser.add_argument("--composed-dir", default=None, help="Path to pre-composed GLB directory")
    parser.add_argument("--scene-id", default=None, help="Render single scene")
    parser.add_argument("--extra-pythonpath", default=None,
                        help="Extra sys.path entries (':' or ',' separated)")
    parser.add_argument("--funnel-audit-out", default=None,
                        help="Optional output directory for per-view funnel audit JSONL files")
    args = parser.parse_args(argv)

    if args.extra_pythonpath:
        for part in args.extra_pythonpath.replace(",", ":").split(":"):
            p = part.strip()
            if p and p not in sys.path:
                sys.path.insert(0, p)

    config = load_config(args.config)

    output_dir = args.output_dir or config["data"]["renders_dir"]
    scenes_dir = Path(args.scenes_dir)

    # Resolve asset_base: CLI arg > config-derived > None
    asset_base = args.asset_base
    if not asset_base:
        candidate = Path(config["data"]["internscenes_dir"]) / "asset_library"
        if candidate.exists():
            asset_base = str(candidate)

    if args.scene_id:
        scene_dirs = [scenes_dir / args.scene_id]
    else:
        scene_dirs = sorted([d for d in scenes_dir.iterdir() if d.is_dir() and (d / "layout.json").exists()])

    print(f"Rendering {len(scene_dirs)} scenes")
    if asset_base:
        print(f"Asset library: {asset_base}")

    failed_scene_ids: list[str] = []
    for i, scene_dir in enumerate(scene_dirs):
        print(f"[{i+1}/{len(scene_dirs)}] Rendering {scene_dir.name}")
        try:
            render_scene(str(scene_dir), output_dir, config,
                         asset_base=asset_base, composed_dir=args.composed_dir,
                         config_path=args.config,
                         funnel_audit_out=args.funnel_audit_out)
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            failed_scene_ids.append(scene_dir.name)

    if failed_scene_ids:
        raise RuntimeError(
            "render_scenes.py failed for scene_ids="
            + ",".join(str(scene_id) for scene_id in failed_scene_ids)
        )


def main_standalone():
    """Entry point when running without Blender (generates commands)."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--scenes-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()

    config = load_config(args.config)
    scenes_dir = Path(args.scenes_dir)
    scene_dirs = sorted([d for d in scenes_dir.iterdir() if d.is_dir() and (d / "layout.json").exists()])

    print(f"Found {len(scene_dirs)} scenes to render")

    if args.dry_run:
        print("\nGenerated render commands:")
        for i in range(0, len(scene_dirs), args.batch_size):
            batch = scene_dirs[i:i + args.batch_size]
            for sd in batch:
                cmd = f"blender --background --python scripts/render_scenes.py -- --config {args.config} --scenes-dir {args.scenes_dir} --scene-id {sd.name}"
                print(cmd)
    else:
        print("Run with --dry-run to generate Blender commands, or run directly with Blender:")
        print(f"  blender --background --python scripts/render_scenes.py -- --config {args.config} --scenes-dir {args.scenes_dir}")


if IN_BLENDER:
    main_blender()
else:
    if __name__ == "__main__":
        main_standalone()
