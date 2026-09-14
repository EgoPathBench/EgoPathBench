"""
Diagnostic: Compare NavBench3D compose_scene.py vs InternScenes official compose logic.
Checks each object's final transform and renders top-down views for visual comparison.
"""

import os
import sys
import json
import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix, euler_matrix
from pathlib import Path

# ─── Paths ───
SCENE_DIR = "/datadisk/NavBench3D/scenes/3rscan__095821fb-e2c2-2de1-94df-20f2cb423bcb"
ASSET_BASE = "/datadisk/NavBench3D/internscenes_raw/asset_library"
OUTPUT_DIR = "/datadisk/NavBench3D/debug_compose"

# ─── Load canonical rotation (shared by both) ───
uid_2_angle = {}
uid_2_cate = {}
angle_path = os.path.join(ASSET_BASE, "uid_2_angle.json")
cate_path = os.path.join(ASSET_BASE, "uid_2_origin_cate.json")
if os.path.exists(angle_path):
    uid_2_angle = json.load(open(angle_path))
if os.path.exists(cate_path):
    uid_2_cate = json.load(open(cate_path))


def get_init_rotation(uid):
    if uid.startswith("objaverse/"):
        short = uid.split("objaverse/")[-1]
        rot_rad = uid_2_angle.get(short, 0) / 180.0 * np.pi
        return (rotation_matrix(rot_rad, [0, 0, 1])
                @ rotation_matrix(0.5 * np.pi, [0, 0, 1])
                @ rotation_matrix(0.5 * np.pi, [1, 0, 0]))
    elif uid.startswith("objaverse_old/"):
        return (rotation_matrix(0.5 * np.pi, [0, 0, 1])
                @ rotation_matrix(0.5 * np.pi, [1, 0, 0]))
    elif uid.startswith("partnet_mobility"):
        t = (rotation_matrix(np.pi, [0, 0, 1])
             @ rotation_matrix(0.5 * np.pi, [1, 0, 0]))
        pm_cate = uid_2_cate.get(uid, "")
        if pm_cate in ["Pen", "Remote", "Phone"]:
            t = (rotation_matrix(np.pi / 2, [0, 1, 0])
                 @ rotation_matrix(np.pi, [0, 0, 1]) @ t)
        return t
    else:
        return (rotation_matrix(0.5 * np.pi, [0, 0, 1])
                @ rotation_matrix(0.5 * np.pi, [1, 0, 0]))


def get_mesh_path(uid):
    if uid.startswith("partnet_mobility"):
        p = os.path.join(ASSET_BASE, uid, "whole.glb")
    else:
        p = os.path.join(ASSET_BASE, uid + ".glb")
    return p if os.path.exists(p) else None


def load_canonical(uid):
    path = get_mesh_path(uid)
    if not path:
        return None
    mesh = trimesh.load(path, force="mesh")
    centroid = mesh.bounding_box.centroid
    mesh.apply_translation(-centroid)
    mesh.apply_transform(get_init_rotation(uid))
    return mesh


def compute_official_transform(instance):
    """Exactly replicates InternScenes official compose_scenes.py logic."""
    uid = instance["model_uid"]
    if not uid:
        return None, None

    mesh = load_canonical(uid)
    if mesh is None:
        return None, None

    bbox = instance["bbox"]
    mesh_size = mesh.bounding_box.extents
    target_size = np.array(bbox[3:6])
    category = instance.get("category", "")

    # Scale (simplified — non-carpet/non-clothes)
    if category not in ["carpet", "clothes"]:
        scale = target_size / mesh_size
        scale_mat = np.diag([scale[0], scale[1], scale[2], 1])
    elif category == "carpet":
        scale_factors = target_size / mesh_size
        if target_size[2] / target_size[0] > 150 or target_size[2] / target_size[1] > 150:
            if target_size[2] / target_size[0] > target_size[2] / target_size[1]:
                rot = rotation_matrix(0.5 * np.pi, [0, 1, 0])
                target_size = np.array([target_size[2], target_size[0], target_size[1]])
                scale_factors = target_size / mesh_size
                scale_mat = np.diag([scale_factors[0], scale_factors[1], scale_factors[2] / 100.0, 1]) @ rot
            else:
                rot = rotation_matrix(0.5 * np.pi, [1, 0, 0])
                target_size = np.array([target_size[0], target_size[2], target_size[1]])
                scale_factors = target_size / mesh_size
                scale_mat = np.diag([scale_factors[0], scale_factors[1], scale_factors[2] / 100.0, 1]) @ rot
        else:
            scale_mat = np.diag([scale_factors[0], scale_factors[1], scale_factors[2] / 100.0, 1])
    else:  # clothes
        scale = target_size / mesh_size
        ms = min(scale)
        scale_mat = np.diag([ms, ms, ms, 1])

    # --- Official logic (exactly as in compose_scenes.py) ---
    transform_final = np.eye(4)

    # scale
    transform_final = scale_mat @ transform_final

    # rotation
    euler_angles = np.array(bbox[6:9])
    rot = euler_matrix(euler_angles[0], euler_angles[1], euler_angles[2], axes='rzxy')
    transform_final = rot @ transform_final

    # translation
    center = np.array(bbox[0:3])
    transform_final[:3, 3] = center

    # Y-up conversion
    yup = rotation_matrix(-np.pi / 2, [1, 0, 0])
    transform_final = yup @ transform_final

    return mesh, transform_final


def extract_position_and_forward(transform):
    """Extract position and forward direction from a 4x4 transform."""
    pos = transform[:3, 3]
    # The canonical mesh faces X-axis, so forward is the X column of the rotation
    forward = transform[:3, 0]
    forward = forward / np.linalg.norm(forward)
    return pos, forward


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(os.path.join(SCENE_DIR, "layout.json")) as f:
        instances = json.load(f)

    print(f"Scene has {len(instances)} objects")
    print("=" * 80)

    scene_zup = trimesh.Scene()  # Z-up scene for top-down view (without Y-up conversion)

    for i, inst in enumerate(instances):
        uid = inst.get("model_uid", "")
        cat = inst.get("category", "")
        if not uid:
            continue

        mesh, transform = compute_official_transform(inst)
        if mesh is None:
            print(f"  [{i}] {cat} ({uid}): MISSING")
            continue

        pos, fwd = extract_position_and_forward(transform)
        bbox = inst["bbox"]
        euler_angles = np.array(bbox[6:9])

        print(f"  [{i}] {cat}")
        print(f"       uid:       {uid}")
        print(f"       bbox center (layout): [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}]")
        print(f"       bbox size:   [{bbox[3]:.3f}, {bbox[4]:.3f}, {bbox[5]:.3f}]")
        print(f"       euler (rzxy): [{euler_angles[0]:.4f}, {euler_angles[1]:.4f}, {euler_angles[2]:.4f}]")
        print(f"       transform pos (after Y-up): [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        print(f"       forward dir (after Y-up):   [{fwd[0]:.3f}, {fwd[1]:.3f}, {fwd[2]:.3f}]")

        # Also compute Z-up transform (without the Y-up conversion) to check
        mesh_size = mesh.bounding_box.extents
        target_size = np.array(bbox[3:6])
        scale = target_size / mesh_size
        scale_mat = np.diag([scale[0], scale[1], scale[2], 1])

        t_zup = np.eye(4)
        t_zup = scale_mat @ t_zup
        rot = euler_matrix(euler_angles[0], euler_angles[1], euler_angles[2], axes='rzxy')
        t_zup = rot @ t_zup
        t_zup[:3, 3] = np.array(bbox[0:3])

        pos_z, fwd_z = extract_position_and_forward(t_zup)
        print(f"       Z-up pos:     [{pos_z[0]:.3f}, {pos_z[1]:.3f}, {pos_z[2]:.3f}]")
        print(f"       Z-up forward: [{fwd_z[0]:.3f}, {fwd_z[1]:.3f}, {fwd_z[2]:.3f}]")
        fwd_angle_deg = np.degrees(np.arctan2(fwd_z[1], fwd_z[0]))
        print(f"       Z-up forward angle: {fwd_angle_deg:.1f}°")
        print()

        # Add to Z-up scene for visualization
        scene_zup.add_geometry(mesh.copy(), transform=t_zup, geom_name=f"{i}_{cat}")

    # Load structure meshes for Z-up scene
    mesh_dir = os.path.join(SCENE_DIR, "StructureMesh")
    if os.path.islink(mesh_dir):
        target = os.readlink(mesh_dir)
        if not os.path.isabs(target):
            mesh_dir = os.path.join(SCENE_DIR, target)

    # Structure meshes are in Y-up (GLB convention), need Rx(90) to go to Z-up
    rx90 = rotation_matrix(np.pi / 2, [1, 0, 0])
    for name in ["wall.glb", "floor.glb"]:
        glb_path = os.path.join(mesh_dir, name)
        if os.path.exists(glb_path):
            try:
                struct = trimesh.load(glb_path, force="mesh")
                # Check bounds to understand coordinate system
                print(f"  Structure {name} bounds: {struct.bounds}")
                # Try without conversion first
                scene_zup.add_geometry(struct.copy(), geom_name=name.replace(".glb", ""))
            except Exception as e:
                print(f"  Failed to load {name}: {e}")

    # Check overall scene bounds
    print(f"\nZ-up scene bounds: {scene_zup.bounds}")

    # Save top-down PNG
    try:
        # Get scene bounds in XY plane
        bounds = scene_zup.bounds
        center_xy = (bounds[0][:2] + bounds[1][:2]) / 2
        extent_xy = bounds[1][:2] - bounds[0][:2]
        max_extent = max(extent_xy) * 1.2

        # Create a top-down camera
        from trimesh.scene.cameras import Camera
        cam = Camera(resolution=[1024, 1024], fov=[90, 90])

        # Render top-down view
        png = scene_zup.save_image(resolution=[1024, 1024])
        if png:
            out_path = os.path.join(OUTPUT_DIR, "topdown_zup.png")
            with open(out_path, 'wb') as f:
                f.write(png)
            print(f"\nSaved top-down view to {out_path}")
    except Exception as e:
        print(f"\nCould not render top-down view: {e}")

    # Export Z-up scene as GLB for inspection
    out_glb = os.path.join(OUTPUT_DIR, "scene_zup_debug.glb")
    trimesh.exchange.export.export_mesh(scene_zup, out_glb)
    print(f"Saved Z-up debug scene to {out_glb}")

    # Also export with Y-up conversion (the normal pipeline)
    scene_yup = trimesh.Scene()
    for i, inst in enumerate(instances):
        uid = inst.get("model_uid", "")
        if not uid:
            continue
        mesh, transform = compute_official_transform(inst)
        if mesh is None:
            continue
        cat = inst.get("category", "")
        scene_yup.add_geometry(mesh, transform=transform, geom_name=f"{i}_{cat}")

    # Add structure meshes (already Y-up from GLB, no transform needed)
    for name in ["wall.glb", "floor.glb"]:
        glb_path = os.path.join(mesh_dir, name)
        if os.path.exists(glb_path):
            try:
                struct = trimesh.load(glb_path)
                scene_yup.add_geometry(struct, geom_name=name.replace(".glb", "_struct"))
            except Exception as e:
                pass

    out_yup = os.path.join(OUTPUT_DIR, "scene_yup_composed.glb")
    trimesh.exchange.export.export_mesh(scene_yup, out_yup)
    print(f"Saved Y-up composed scene to {out_yup}")

    # ── Key diagnostic: check if structure mesh coordinate system matches object coordinate system ──
    print("\n" + "=" * 80)
    print("COORDINATE SYSTEM DIAGNOSTIC")
    print("=" * 80)

    # Load floor mesh to check its coordinate system
    floor_path = os.path.join(mesh_dir, "floor.glb")
    if os.path.exists(floor_path):
        floor = trimesh.load(floor_path, force="mesh")
        print(f"Floor mesh bounds (raw from GLB): {floor.bounds}")
        print(f"  X range: [{floor.bounds[0][0]:.3f}, {floor.bounds[1][0]:.3f}]")
        print(f"  Y range: [{floor.bounds[0][1]:.3f}, {floor.bounds[1][1]:.3f}]")
        print(f"  Z range: [{floor.bounds[0][2]:.3f}, {floor.bounds[1][2]:.3f}]")

        # Check which axis is ~0 (flat floor)
        extents = floor.bounds[1] - floor.bounds[0]
        min_axis = np.argmin(extents)
        axis_names = ["X", "Y", "Z"]
        print(f"  Thinnest axis: {axis_names[min_axis]} (extent={extents[min_axis]:.4f})")
        print(f"  → Floor is in {'Y-up (GLB standard)' if min_axis == 1 else 'Z-up' if min_axis == 2 else 'X-up (?)'} coordinate system")

    # Check wall mesh
    wall_path = os.path.join(mesh_dir, "wall.glb")
    if os.path.exists(wall_path):
        wall = trimesh.load(wall_path, force="mesh")
        print(f"\nWall mesh bounds (raw from GLB): {wall.bounds}")
        print(f"  X range: [{wall.bounds[0][0]:.3f}, {wall.bounds[1][0]:.3f}]")
        print(f"  Y range: [{wall.bounds[0][1]:.3f}, {wall.bounds[1][1]:.3f}]")
        print(f"  Z range: [{wall.bounds[0][2]:.3f}, {wall.bounds[1][2]:.3f}]")

        extents = wall.bounds[1] - wall.bounds[0]
        # Wall should be tall in one axis
        max_axis = np.argmax(extents)
        print(f"  Tallest axis: {axis_names[max_axis]} (extent={extents[max_axis]:.4f})")
        print(f"  → Walls are tall in {axis_names[max_axis]} direction")

    print("\n" + "=" * 80)
    print("CRITICAL CHECK: Object Z-up positions vs floor plane")
    print("=" * 80)

    # In Z-up space (without Y-up conversion), objects should sit ON the floor
    # The bbox center Z should be at roughly half the object height
    for i, inst in enumerate(instances):
        uid = inst.get("model_uid", "")
        cat = inst.get("category", "")
        if not uid:
            continue
        bbox = inst["bbox"]
        center_z = bbox[2]  # Z in layout space
        half_h = bbox[5] / 2  # half height in layout space
        floor_z = center_z - half_h
        print(f"  [{i}] {cat}: center_z={center_z:.3f}, half_h={half_h:.3f}, floor_z={floor_z:.3f}")

    print("\nDone!")


if __name__ == "__main__":
    main()
