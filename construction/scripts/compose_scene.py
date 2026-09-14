"""
NavBench3D - Scene Composition (InternScenes official logic).

Composes a scene from layout.json + individual asset GLBs into a single
GLB file using the exact same transformation pipeline as InternScenes.

Usage:
    python scripts/compose_scene.py \
        --scene-dir /datadisk/NavBench3D/scenes/3rscan__095821fb-... \
        --asset-base /datadisk/NavBench3D/internscenes_raw/asset_library \
        --output /datadisk/NavBench3D/composed/scene.glb
"""

import os
import sys
import json
import argparse
import tempfile
import struct
import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix, euler_matrix
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import multiprocessing


def _is_color_export_shape_error(exc: Exception) -> bool:
    message = str(exc)
    return "cannot reshape array of size" in message and "shape (4)" in message


def _rgba_from_main_color(material) -> np.ndarray:
    main_color = getattr(material, "main_color", None)
    if main_color is None:
        return np.array([102, 102, 102, 255], dtype=np.uint8)
    rgba = np.asarray(main_color, dtype=np.uint8).reshape(-1)
    if rgba.size >= 4:
        return rgba[:4]
    if rgba.size == 3:
        return np.append(rgba, np.array([255], dtype=np.uint8))
    if rgba.size == 1:
        return np.array([rgba[0], rgba[0], rgba[0], 255], dtype=np.uint8)
    return np.array([102, 102, 102, 255], dtype=np.uint8)


def _normalize_color_array_rgba(values) -> np.ndarray | None:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.uint8)
    if arr.ndim == 2 and arr.shape[1] == 4:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 3:
        alpha = np.full((arr.shape[0], 1), 255, dtype=np.uint8)
        return np.concatenate([arr, alpha], axis=1)
    if arr.ndim == 1 and arr.size % 4 == 0:
        return arr.reshape((-1, 4))
    if arr.ndim == 1 and arr.size % 3 == 0:
        rgb = arr.reshape((-1, 3))
        alpha = np.full((rgb.shape[0], 1), 255, dtype=np.uint8)
        return np.concatenate([rgb, alpha], axis=1)
    return None


def sanitize_geometry_visual_for_export(geom) -> bool:
    visual = getattr(geom, "visual", None)
    if visual is None:
        return False

    if isinstance(visual, trimesh.visual.color.ColorVisuals):
        vertex_colors = _normalize_color_array_rgba(getattr(visual, "vertex_colors", None))
        if vertex_colors is not None:
            geom.visual = trimesh.visual.color.ColorVisuals(mesh=geom, vertex_colors=vertex_colors)
            return True
        face_colors = _normalize_color_array_rgba(getattr(visual, "face_colors", None))
        if face_colors is not None:
            geom.visual = trimesh.visual.color.ColorVisuals(mesh=geom, face_colors=face_colors)
            return True
        return False

    if isinstance(visual, trimesh.visual.texture.TextureVisuals):
        color_visual = None
        try:
            candidate = visual.to_color()
            vertex_colors = _normalize_color_array_rgba(getattr(candidate, "vertex_colors", None))
            if vertex_colors is not None:
                color_visual = trimesh.visual.color.ColorVisuals(mesh=geom, vertex_colors=vertex_colors)
            else:
                face_colors = _normalize_color_array_rgba(getattr(candidate, "face_colors", None))
                if face_colors is not None:
                    color_visual = trimesh.visual.color.ColorVisuals(mesh=geom, face_colors=face_colors)
        except Exception:
            color_visual = None

        if color_visual is None:
            rgba = _rgba_from_main_color(getattr(visual, "material", None))
            color_visual = trimesh.visual.color.ColorVisuals(
                mesh=geom,
                vertex_colors=np.tile(rgba, (len(geom.vertices), 1)),
            )

        geom.visual = color_visual
        return True

    return False


def _geometry_export_has_color_shape_error(name: str, geom) -> bool:
    probe_scene = trimesh.Scene()
    probe_scene.add_geometry(geom, geom_name=name)
    fd, probe_path = tempfile.mkstemp(suffix=".glb")
    os.close(fd)
    try:
        trimesh.exchange.export.export_mesh(probe_scene, probe_path)
        return False
    except Exception as exc:
        if _is_color_export_shape_error(exc):
            return True
        raise
    finally:
        if os.path.exists(probe_path):
            os.remove(probe_path)


_GLB_MAGIC = b"glTF"
_GLB_JSON_CHUNK_TYPE = 0x4E4F534A
_GLB_BIN_CHUNK_TYPE = 0x004E4942
_GLTF_TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
    "MAT2": 4,
    "MAT3": 9,
    "MAT4": 16,
}
_GLTF_COMPONENT_DTYPES = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}


def _load_glb_document_and_bin(path: str | Path) -> tuple[dict, bytes | None]:
    data = Path(path).read_bytes()
    if len(data) < 12:
        raise ValueError(f"GLB too short: {path}")
    magic, version, length = struct.unpack_from("<4sII", data, 0)
    if magic != _GLB_MAGIC:
        raise ValueError(f"Invalid GLB magic in {path}: {magic!r}")
    if version != 2:
        raise ValueError(f"Unsupported GLB version in {path}: {version}")

    offset = 12
    document = None
    bin_chunk = None
    while offset < length:
        chunk_length, chunk_type = struct.unpack_from("<II", data, offset)
        offset += 8
        chunk = data[offset: offset + chunk_length]
        offset += chunk_length
        if chunk_type == _GLB_JSON_CHUNK_TYPE:
            document = json.loads(chunk.decode("utf-8"))
        elif chunk_type == _GLB_BIN_CHUNK_TYPE:
            bin_chunk = chunk
    if document is None:
        raise ValueError(f"GLB missing JSON chunk: {path}")
    return document, bin_chunk


def _decode_glb_accessor_array(document: dict, bin_chunk: bytes | None, accessor_idx: int) -> np.ndarray:
    accessor = document["accessors"][accessor_idx]
    component_dtype = _GLTF_COMPONENT_DTYPES[int(accessor["componentType"])]
    component_count = _GLTF_TYPE_COMPONENTS[str(accessor["type"])]
    count = int(accessor["count"])
    buffer_view_idx = accessor.get("bufferView")
    if buffer_view_idx is None:
        shape = (count, component_count) if component_count > 1 else (count,)
        return np.zeros(shape, dtype=component_dtype)
    if bin_chunk is None:
        raise ValueError("Accessor references a bufferView but GLB has no BIN chunk")

    buffer_view = document["bufferViews"][buffer_view_idx]
    byte_offset = int(buffer_view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    default_stride = np.dtype(component_dtype).itemsize * component_count
    byte_stride = int(buffer_view.get("byteStride", default_stride))

    if byte_stride == default_stride:
        flat = np.frombuffer(
            bin_chunk,
            dtype=component_dtype,
            count=count * component_count,
            offset=byte_offset,
        )
        if component_count == 1:
            return flat.reshape((count,))
        return flat.reshape((count, component_count))

    output = np.empty((count, component_count), dtype=component_dtype)
    for row_idx in range(count):
        start = byte_offset + row_idx * byte_stride
        output[row_idx] = np.frombuffer(
            bin_chunk,
            dtype=component_dtype,
            count=component_count,
            offset=start,
        )
    if component_count == 1:
        return output.reshape((count,))
    return output


def _find_blender_import_incompatible_mesh_names(document: dict, bin_chunk: bytes | None) -> list[str]:
    accessors = document.get("accessors", [])
    meshes = document.get("meshes", [])
    bad_mesh_names: list[str] = []

    for mesh_idx, mesh in enumerate(meshes):
        mesh_name = str(mesh.get("name") or f"mesh_{mesh_idx}")
        for primitive in mesh.get("primitives", []):
            attributes = primitive.get("attributes", {})
            if "COLOR_0" not in attributes or "POSITION" not in attributes:
                continue
            position_count = int(accessors[int(attributes["POSITION"])]["count"])
            color_count = int(accessors[int(attributes["COLOR_0"])]["count"])
            invalid = color_count != position_count

            indices_idx = primitive.get("indices")
            if indices_idx is not None:
                indices = _decode_glb_accessor_array(document, bin_chunk, int(indices_idx)).reshape(-1)
                if indices.size:
                    max_index = int(indices.max())
                    invalid = invalid or max_index >= position_count or max_index >= color_count

            if invalid:
                bad_mesh_names.append(mesh_name)
                break

    return bad_mesh_names


def find_blender_import_incompatible_mesh_names_in_glb(path: str | Path) -> list[str]:
    document, bin_chunk = _load_glb_document_and_bin(path)
    return _find_blender_import_incompatible_mesh_names(document, bin_chunk)


def existing_composed_glb_is_blender_compatible(path: str | Path) -> bool:
    try:
        return not find_blender_import_incompatible_mesh_names_in_glb(path)
    except Exception:
        return False


def export_scene_glb(scene, output_path: str | Path) -> list[str]:
    sanitized_total: list[str] = []
    try:
        trimesh.exchange.export.export_mesh(scene, output_path)
    except Exception as exc:
        if not _is_color_export_shape_error(exc):
            raise

        geometry_names = list(scene.geometry.keys())
        if len(geometry_names) == 1:
            bad_names = geometry_names
        else:
            bad_names = [
                name for name, geom in scene.geometry.items()
                if _geometry_export_has_color_shape_error(name, geom)
            ]

        if not bad_names:
            raise ValueError("GLB export hit COLOR_0 shape error but no offending geometry was found")

        sanitized: list[str] = []
        for name in bad_names:
            if sanitize_geometry_visual_for_export(scene.geometry[name]):
                sanitized.append(name)

        if not sanitized:
            raise ValueError("GLB export hit COLOR_0 shape error and no geometry could be sanitized")

        trimesh.exchange.export.export_mesh(scene, output_path)
        sanitized_total.extend(sanitized)

    incompatible_meshes = find_blender_import_incompatible_mesh_names_in_glb(output_path)
    if incompatible_meshes:
        sanitized_import: list[str] = []
        for name in incompatible_meshes:
            geom = scene.geometry.get(name)
            if geom is None:
                continue
            if sanitize_geometry_visual_for_export(geom):
                sanitized_import.append(name)

        if not sanitized_import:
            raise ValueError(
                "GLB export produced Blender-incompatible COLOR_0 primitives but no geometry could be sanitized: "
                + ", ".join(incompatible_meshes)
            )

        trimesh.exchange.export.export_mesh(scene, output_path)
        sanitized_total.extend(name for name in sanitized_import if name not in sanitized_total)

        remaining = find_blender_import_incompatible_mesh_names_in_glb(output_path)
        if remaining:
            raise ValueError(
                "GLB export remains Blender-incompatible after sanitization: "
                + ", ".join(remaining)
            )

    return sanitized_total


class AssetMeshLoader:
    """Loads and canonicalizes asset meshes (from InternScenes official code)."""

    def __init__(self, asset_dir: str):
        self.asset_dir = asset_dir
        uid_angle_path = os.path.join(asset_dir, "uid_2_angle.json")
        uid_cate_path = os.path.join(asset_dir, "uid_2_origin_cate.json")

        self.obja_uid_2_rotation = {}
        if os.path.exists(uid_angle_path):
            with open(uid_angle_path) as f:
                self.obja_uid_2_rotation = json.load(f)

        self.pm_uid_2_origin_cate = {}
        if os.path.exists(uid_cate_path):
            with open(uid_cate_path) as f:
                self.pm_uid_2_origin_cate = json.load(f)

    def get_mesh_path(self, uid: str) -> str | None:
        if uid.startswith("partnet_mobility"):
            p = os.path.join(self.asset_dir, uid, "whole.glb")
        else:
            p = os.path.join(self.asset_dir, uid + ".glb")
        return p if os.path.exists(p) else None

    def load_init_rotation(self, uid: str) -> np.ndarray:
        """Get the canonical rotation for each asset library type."""
        if uid.startswith("objaverse/"):
            short_uid = uid.split("objaverse/")[-1]
            rot_deg = self.obja_uid_2_rotation.get(short_uid, 0)
            rot_rad = rot_deg / 180.0 * np.pi
            return (rotation_matrix(rot_rad, [0, 0, 1])
                    @ rotation_matrix(0.5 * np.pi, [0, 0, 1])
                    @ rotation_matrix(0.5 * np.pi, [1, 0, 0]))
        elif uid.startswith("objaverse_old/"):
            return (rotation_matrix(0.5 * np.pi, [0, 0, 1])
                    @ rotation_matrix(0.5 * np.pi, [1, 0, 0]))
        elif uid.startswith("partnet_mobility"):
            transform = (rotation_matrix(np.pi, [0, 0, 1])
                         @ rotation_matrix(0.5 * np.pi, [1, 0, 0]))
            pm_cate = self.pm_uid_2_origin_cate.get(uid, "")
            if pm_cate in ["Pen", "Remote", "Phone"]:
                transform = (rotation_matrix(np.pi / 2, [0, 1, 0])
                             @ rotation_matrix(np.pi, [0, 0, 1])
                             @ transform)
            return transform
        else:
            # 3D-FUTURE-model, hssd-models, gen_assets, gr100
            return (rotation_matrix(0.5 * np.pi, [0, 0, 1])
                    @ rotation_matrix(0.5 * np.pi, [1, 0, 0]))

    def load_canonical_mesh(self, uid: str, use_texture: bool = False):
        """Load mesh in canonical orientation: Z-up, facing X-axis, centered at origin."""
        mesh_path = self.get_mesh_path(uid)
        if mesh_path is None:
            return None

        try:
            if use_texture:
                mesh = trimesh.load(mesh_path)
            else:
                mesh = trimesh.load(mesh_path, force="mesh")
        except Exception as e:
            print(f"  Failed to load {uid}: {e}")
            return None

        transform = self.load_init_rotation(uid)
        centroid = mesh.bounding_box.centroid
        mesh.apply_translation(-centroid)
        mesh.apply_transform(transform)
        return mesh


def get_scale_transform(mesh_size: np.ndarray, instance: dict) -> np.ndarray:
    """Compute scale matrix to fit mesh to target bbox size."""
    category = instance.get("category", "")
    target_size = np.array(instance["bbox"][3:6])

    if category == "carpet":
        scale_factors = target_size / mesh_size
        if target_size[2] / target_size[0] > 150 or target_size[2] / target_size[1] > 150:
            if target_size[2] / target_size[0] > target_size[2] / target_size[1]:
                rot = rotation_matrix(0.5 * np.pi, [0, 1, 0])
                target_size = np.array([target_size[2], target_size[0], target_size[1]])
                scale_factors = target_size / mesh_size
                scale_mat = np.diag([scale_factors[0], scale_factors[1], scale_factors[2] / 100.0, 1])
                return scale_mat @ rot
            else:
                rot = rotation_matrix(0.5 * np.pi, [1, 0, 0])
                target_size = np.array([target_size[0], target_size[2], target_size[1]])
                scale_factors = target_size / mesh_size
                scale_mat = np.diag([scale_factors[0], scale_factors[1], scale_factors[2] / 100.0, 1])
                return scale_mat @ rot
        else:
            return np.diag([scale_factors[0], scale_factors[1], scale_factors[2] / 100.0, 1])
    elif category == "clothes":
        scale = target_size / mesh_size
        min_scale = min(scale)
        return np.diag([min_scale, min_scale, min_scale, 1])
    else:
        scale = target_size / mesh_size
        return np.diag([scale[0], scale[1], scale[2], 1])


def compose_scene(scene_dir: str, asset_loader: AssetMeshLoader,
                  use_texture: bool = True, skip_ceiling: bool = True) -> trimesh.Scene:
    """Compose a full scene using the official InternScenes transformation pipeline."""
    layout_path = os.path.join(scene_dir, "layout.json")
    with open(layout_path) as f:
        instance_infos = json.load(f)

    scene = trimesh.Scene()
    lock = threading.Lock()

    def process_instance(index: int):
        instance = instance_infos[index]
        model_uid = instance.get("model_uid", "")
        if not model_uid:
            return

        mesh = asset_loader.load_canonical_mesh(model_uid, use_texture=use_texture)
        if mesh is None:
            return

        bbox = instance["bbox"]
        mesh_size = mesh.bounding_box.extents

        # 1. Scale to target size
        transform_final = get_scale_transform(mesh_size, instance)

        # 2. Rotation from bbox Euler angles (rzxy order)
        euler_angles = np.array(bbox[6:9])
        rot = euler_matrix(euler_angles[0], euler_angles[1], euler_angles[2], axes='rzxy')
        transform_final = rot @ transform_final

        # 3. Translation to center position
        center = np.array(bbox[0:3])
        transform_final[:3, 3] = center

        # 4. Final Y-up conversion for GLB export
        yup_rot = rotation_matrix(-np.pi / 2, [1, 0, 0])
        transform_final = yup_rot @ transform_final

        geometry_name = f"{index}_{instance.get('category', 'obj')}@{model_uid}"

        with lock:
            scene.graph.update(frame_to=geometry_name, matrix=transform_final)
            if isinstance(mesh, trimesh.Scene):
                for geom_name, mesh_part in mesh.geometry.items():
                    nodes = mesh.graph.geometry_nodes.get(geom_name, [])
                    for i, node_name in enumerate(nodes):
                        internal_transform, _ = mesh.graph.get(node_name)
                        scene.add_geometry(
                            mesh_part,
                            geom_name=f"{geometry_name}_{geom_name}_{i}",
                            transform=internal_transform,
                            parent_node_name=geometry_name,
                        )
                else:
                    if not nodes:
                        scene.add_geometry(
                            mesh_part,
                            geom_name=f"{geometry_name}_{geom_name}",
                            parent_node_name=geometry_name,
                        )
            else:
                scene.add_geometry(
                    mesh,
                    geom_name=geometry_name + "_geom",
                    parent_node_name=geometry_name,
                )

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_instance, i) for i in range(len(instance_infos))]
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"  Error: {e}")

    # Add structure meshes (wall, floor, optionally ceiling) — no transform needed
    mesh_dir = os.path.join(scene_dir, "StructureMesh")
    if os.path.islink(mesh_dir):
        mesh_dir = os.readlink(mesh_dir)
        if not os.path.isabs(mesh_dir):
            mesh_dir = os.path.join(scene_dir, mesh_dir)

    for name in ["wall.glb", "floor.glb"] + ([] if skip_ceiling else ["ceiling.glb"]):
        glb_path = os.path.join(mesh_dir, name)
        if os.path.exists(glb_path):
            try:
                struct_mesh = trimesh.load(glb_path)
                scene.add_geometry(struct_mesh, geom_name=name.replace(".glb", ""))
            except Exception as e:
                print(f"  Failed to load {name}: {e}")

    return scene


def _compose_worker(args_tuple):
    """Top-level worker function for multiprocessing (must be picklable)."""
    idx, total, scene_dir, asset_base, use_texture, skip_ceiling, out_dir = args_tuple
    sd = Path(scene_dir)
    out_glb = Path(out_dir) / f"{sd.name}.glb"
    # Double-check in case another worker finished it
    if out_glb.exists() and existing_composed_glb_is_blender_compatible(out_glb):
        return (sd.name, True, "skipped")

    worker_loader = AssetMeshLoader(asset_base)
    try:
        scene = compose_scene(
            str(sd), worker_loader,
            use_texture=use_texture,
            skip_ceiling=skip_ceiling,
        )
        # Atomic write: temp file then rename
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".glb", dir=str(out_dir))
        os.close(tmp_fd)
        sanitized = export_scene_glb(scene, tmp_path)
        os.rename(tmp_path, str(out_glb))
        if sanitized:
            print(f"[{idx+1}/{total}] {sd.name} → SANITIZED {len(sanitized)} bad geometries", flush=True)
        print(f"[{idx+1}/{total}] {sd.name} → OK", flush=True)
        return (sd.name, True, "ok")
    except Exception as e:
        print(f"[{idx+1}/{total}] {sd.name} → ERROR: {e}", flush=True)
        return (sd.name, False, str(e))


def _batch_scene_recompose_status(args_tuple):
    scene_dir, output_dir = args_tuple
    sd = Path(scene_dir)
    out_glb = Path(output_dir) / f"{sd.name}.glb"
    if not out_glb.exists():
        return (scene_dir, True, False)
    incompatible = not existing_composed_glb_is_blender_compatible(out_glb)
    return (scene_dir, incompatible, incompatible)


def main():
    parser = argparse.ArgumentParser(description="Compose InternScenes into single GLB")
    parser.add_argument("--scene-dir", default=None, help="Path to single scene directory with layout.json")
    parser.add_argument("--scenes-dir", default=None, help="Path to directory containing multiple scene directories (batch mode)")
    parser.add_argument("--asset-base", required=True, help="Path to asset_library/")
    parser.add_argument("--output", default=None, help="Output GLB path (single mode) or directory (batch mode)")
    parser.add_argument("--output-dir", default=None, help="Output directory for batch mode")
    parser.add_argument("--no-texture", action="store_true", help="Skip textures for faster composition")
    parser.add_argument("--keep-ceiling", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel scene compositions")
    args = parser.parse_args()

    loader = AssetMeshLoader(args.asset_base)

    if args.scenes_dir:
        # Batch mode
        scenes_dir = Path(args.scenes_dir)
        output_dir = Path(args.output_dir or args.output or "/datadisk/NavBench3D/composed")
        output_dir.mkdir(parents=True, exist_ok=True)

        scene_dirs = sorted([
            d for d in scenes_dir.iterdir()
            if d.is_dir() and (d / "layout.json").exists()
        ])

        scan_workers = min(max(args.workers, 1), len(scene_dirs)) if scene_dirs else 1
        scan_items = [(str(sd), str(output_dir)) for sd in scene_dirs]
        if scan_workers <= 1:
            scan_results = [_batch_scene_recompose_status(item) for item in scan_items]
        else:
            print(f"Validating existing composed GLBs with {scan_workers} worker processes")
            with ProcessPoolExecutor(max_workers=scan_workers) as pool:
                scan_results = list(pool.map(_batch_scene_recompose_status, scan_items))

        # Re-compose missing or Blender-incompatible existing GLBs.
        todo = []
        skipped_compatible = 0
        invalid_existing = 0
        for scene_dir, needs_recompose, incompatible_existing in scan_results:
            if needs_recompose:
                todo.append(Path(scene_dir))
                invalid_existing += int(incompatible_existing)
            else:
                skipped_compatible += 1

        print(
            f"Found {len(scene_dirs)} scenes, {len(todo)} need composition "
            f"({invalid_existing} invalid existing, {skipped_compatible} reusable)"
        )

        workers = min(args.workers, len(todo)) if todo else 1
        work_items = [
            (i, len(todo), str(sd), args.asset_base, not args.no_texture,
             not args.keep_ceiling, str(output_dir))
            for i, sd in enumerate(todo)
        ]

        if workers <= 1:
            results = [_compose_worker(item) for item in work_items]
        else:
            print(f"Using {workers} worker processes")
            with ProcessPoolExecutor(max_workers=workers) as pool:
                results = list(pool.map(_compose_worker, work_items))

        failed = [(name, err) for name, ok, err in results if not ok]
        succeeded = sum(1 for _, ok, _ in results if ok)
        print(f"\nDone: {succeeded} succeeded, {len(failed)} failed")
        if failed:
            for name, err in failed:
                print(f"  FAILED: {name}: {err}")
    else:
        # Single scene mode
        if not args.scene_dir:
            parser.error("Either --scene-dir or --scenes-dir is required")
        output = args.output
        if not output:
            output = f"/datadisk/NavBench3D/composed/{Path(args.scene_dir).name}.glb"

        print(f"Composing scene: {args.scene_dir}")
        scene = compose_scene(
            args.scene_dir,
            loader,
            use_texture=not args.no_texture,
            skip_ceiling=not args.keep_ceiling,
        )

        os.makedirs(os.path.dirname(output), exist_ok=True)
        export_scene_glb(scene, output)
        print(f"Saved composed scene to {output}")


if __name__ == "__main__":
    main()
