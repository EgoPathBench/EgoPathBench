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
import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix, euler_matrix
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
import multiprocessing


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
    if out_glb.exists():
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
        trimesh.exchange.export.export_mesh(scene, tmp_path)
        os.rename(tmp_path, str(out_glb))
        print(f"[{idx+1}/{total}] {sd.name} → OK", flush=True)
        return (sd.name, True, "ok")
    except Exception as e:
        print(f"[{idx+1}/{total}] {sd.name} → ERROR: {e}", flush=True)
        return (sd.name, False, str(e))


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

        # Skip already composed
        todo = []
        for sd in scene_dirs:
            out_glb = output_dir / f"{sd.name}.glb"
            if not out_glb.exists():
                todo.append(sd)

        print(f"Found {len(scene_dirs)} scenes, {len(todo)} need composition")

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
        trimesh.exchange.export.export_mesh(scene, output)
        print(f"Saved composed scene to {output}")


if __name__ == "__main__":
    main()
