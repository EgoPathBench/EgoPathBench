"""
AB Comparison: Run InternScenes official SceneComposer vs NavBench3D compose_scene.py
on the same test scene, compare per-object transforms.
Also generate trimesh top-down wireframe renders for visual comparison.
"""

import os
import sys
import json
import numpy as np
import trimesh
from trimesh.transformations import rotation_matrix, euler_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow, Rectangle
from matplotlib.transforms import Affine2D

SCENE_NAME = "3rscan__095821fb-e2c2-2de1-94df-20f2cb423bcb"
SCENE_DIR = f"/datadisk/NavBench3D/scenes/{SCENE_NAME}"
ASSET_BASE = "/datadisk/NavBench3D/internscenes_raw/asset_library"
OUTPUT_DIR = "/datadisk/NavBench3D/debug_compose"

# ─── Add InternScenes to path ───
sys.path.insert(0, "/home/v-yangzhao4/projects/InternScenes/InternScenes/InternScenes_Real2Sim")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_canonical_mesh_official(uid, asset_dir):
    """Replicates official AssetMeshLoader exactly."""
    uid_2_angle = json.load(open(os.path.join(asset_dir, "uid_2_angle.json")))
    uid_2_cate = json.load(open(os.path.join(asset_dir, "uid_2_origin_cate.json")))

    # Get mesh path
    if uid.startswith("partnet_mobility"):
        mesh_path = os.path.join(asset_dir, uid, "whole.glb")
    elif uid.startswith("objaverse/"):
        mesh_path = os.path.join(asset_dir, uid + ".glb")
    elif uid.startswith("objaverse_old/"):
        mesh_path = os.path.join(asset_dir, uid + ".glb")
    elif uid.startswith("3D-FUTURE-model"):
        mesh_path = os.path.join(asset_dir, uid + ".glb")
    elif uid.startswith("hssd-models"):
        mesh_path = os.path.join(asset_dir, uid + ".glb")
    elif uid.startswith("gen_assets"):
        mesh_path = os.path.join(asset_dir, uid + ".glb")
    elif uid.startswith("gr100"):
        mesh_path = os.path.join(asset_dir, uid + ".glb")
    else:
        return None

    if not os.path.exists(mesh_path):
        print(f"  MISSING: {mesh_path}")
        return None

    mesh = trimesh.load(mesh_path, force="mesh")

    # Get init rotation
    if uid.startswith("objaverse/"):
        short = uid.split("objaverse/")[-1]
        rot_rad = uid_2_angle.get(short, 0) / 180.0 * np.pi
        transform = (rotation_matrix(rot_rad, [0, 0, 1])
                     @ rotation_matrix(0.5 * np.pi, [0, 0, 1])
                     @ rotation_matrix(0.5 * np.pi, [1, 0, 0]))
    elif uid.startswith("objaverse_old/"):
        transform = (rotation_matrix(0.5 * np.pi, [0, 0, 1])
                     @ rotation_matrix(0.5 * np.pi, [1, 0, 0]))
    elif uid.startswith("partnet_mobility"):
        transform = (rotation_matrix(np.pi, [0, 0, 1])
                     @ rotation_matrix(0.5 * np.pi, [1, 0, 0]))
        pm_cate = uid_2_cate.get(uid, "")
        if pm_cate in ["Pen", "Remote", "Phone"]:
            transform = (rotation_matrix(np.pi / 2, [0, 1, 0])
                         @ rotation_matrix(np.pi, [0, 0, 1]) @ transform)
    else:
        transform = (rotation_matrix(0.5 * np.pi, [0, 0, 1])
                     @ rotation_matrix(0.5 * np.pi, [1, 0, 0]))

    mesh.apply_translation(-mesh.bounding_box.centroid)
    mesh.apply_transform(transform)
    return mesh


def compute_transform_official(instance, asset_dir):
    """Exact replication of official compose_scenes.py logic."""
    uid = instance["model_uid"]
    if not uid:
        return None, None

    mesh = load_canonical_mesh_official(uid, asset_dir)
    if mesh is None:
        return None, None

    bbox = instance["bbox"]
    mesh_size = mesh.bounding_box.extents
    target_size = np.array(bbox[3:6])
    category = instance.get("category", "")

    # Scale
    if category not in ["carpet", "clothes"]:
        scale = target_size / mesh_size
        scale_mat = np.diag([scale[0], scale[1], scale[2], 1])
    else:
        scale = target_size / mesh_size
        scale_mat = np.diag([scale[0], scale[1], scale[2], 1])

    # Official: transform_final = scale @ eye
    transform_final = scale_mat.copy()

    # Rotation
    euler_angles = np.array(bbox[6:9])
    rot = euler_matrix(euler_angles[0], euler_angles[1], euler_angles[2], axes='rzxy')
    transform_final = rot @ transform_final

    # Translation
    center = np.array(bbox[0:3])
    transform_final[:3, 3] = center

    # Y-up conversion
    yup = rotation_matrix(-np.pi / 2, [1, 0, 0])
    transform_final = yup @ transform_final

    return mesh, transform_final


def get_zup_transform(instance, asset_dir):
    """Same as official but WITHOUT the final Y-up conversion (for Z-up visualization)."""
    uid = instance["model_uid"]
    if not uid:
        return None, None

    mesh = load_canonical_mesh_official(uid, asset_dir)
    if mesh is None:
        return None, None

    bbox = instance["bbox"]
    mesh_size = mesh.bounding_box.extents
    target_size = np.array(bbox[3:6])
    category = instance.get("category", "")

    if category not in ["carpet", "clothes"]:
        scale = target_size / mesh_size
        scale_mat = np.diag([scale[0], scale[1], scale[2], 1])
    else:
        scale = target_size / mesh_size
        scale_mat = np.diag([scale[0], scale[1], scale[2], 1])

    transform_final = scale_mat.copy()
    euler_angles = np.array(bbox[6:9])
    rot = euler_matrix(euler_angles[0], euler_angles[1], euler_angles[2], axes='rzxy')
    transform_final = rot @ transform_final
    center = np.array(bbox[0:3])
    transform_final[:3, 3] = center

    return mesh, transform_final


def render_topdown_matplotlib(instances, asset_dir, output_path, title="Top-Down View"):
    """Render a top-down (XY plane, Z-up) view using matplotlib."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Load and draw structure meshes
    mesh_dir = os.path.join(SCENE_DIR, "StructureMesh")
    if os.path.islink(mesh_dir):
        target = os.readlink(mesh_dir)
        if not os.path.isabs(target):
            mesh_dir = os.path.join(SCENE_DIR, target)

    # Structure meshes are in Y-up, convert to Z-up for our plot
    for name, color in [("floor.glb", "#f0f0f0"), ("wall.glb", "#808080")]:
        glb_path = os.path.join(mesh_dir, name)
        if os.path.exists(glb_path):
            struct = trimesh.load(glb_path, force="mesh")
            # Y-up → Z-up: swap Y and Z
            verts = struct.vertices.copy()
            # In Y-up GLB: Y is up, so for top-down XZ view
            # Plot X vs Z (the floor plane)
            ax.triplot(verts[:, 0], verts[:, 2], struct.faces, color=color, alpha=0.3, linewidth=0.3)

    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for i, inst in enumerate(instances):
        uid = inst.get("model_uid", "")
        cat = inst.get("category", "")
        if not uid:
            continue

        mesh, t_zup = get_zup_transform(inst, asset_dir)
        if mesh is None:
            continue

        bbox = inst["bbox"]
        cx, cy, cz = bbox[0], bbox[1], bbox[2]
        sx, sy, sz = bbox[3], bbox[4], bbox[5]

        # Forward direction in Z-up space
        fwd = t_zup[:3, 0]  # X column = forward
        fwd_xy = fwd[:2]
        fwd_norm = np.linalg.norm(fwd_xy)
        if fwd_norm > 1e-6:
            fwd_xy = fwd_xy / fwd_norm

        # Draw rotated bounding box
        euler_angles = np.array(bbox[6:9])
        # First euler angle is rotation about Z (rzxy order)
        angle_deg = np.degrees(euler_angles[0])

        color = colors[i % 10]

        # Draw bbox as rotated rectangle
        rect = Rectangle((-sx / 2, -sy / 2), sx, sy,
                          linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.3)
        t = Affine2D().rotate_deg(angle_deg).translate(cx, cy) + ax.transData
        rect.set_transform(t)
        ax.add_patch(rect)

        # Draw forward arrow
        arrow_len = max(sx, sy) * 0.4
        ax.annotate('', xy=(cx + fwd_xy[0] * arrow_len, cy + fwd_xy[1] * arrow_len),
                     xytext=(cx, cy),
                     arrowprops=dict(arrowstyle='->', color=color, lw=2))

        # Label
        ax.text(cx, cy, f"{i}:{cat}", fontsize=7, ha='center', va='center',
                color='black', fontweight='bold')

    ax.set_aspect('equal')
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    with open(os.path.join(SCENE_DIR, "layout.json")) as f:
        instances = json.load(f)

    print(f"Scene: {SCENE_NAME}")
    print(f"Objects: {len(instances)}")
    print()

    # ─── 1. Render top-down view with Z-up transforms ───
    render_topdown_matplotlib(
        instances, ASSET_BASE,
        os.path.join(OUTPUT_DIR, "topdown_zup_official.png"),
        title=f"Z-up Top-Down (Official Logic) — {SCENE_NAME}"
    )

    # ─── 2. Check GLB round-trip ───
    print("\n" + "=" * 80)
    print("GLB ROUND-TRIP TEST")
    print("=" * 80)
    print("Testing if trimesh GLB export adds extra root rotation...")

    # Create a simple test: place a box at origin facing X
    test_scene = trimesh.Scene()
    box = trimesh.creation.box(extents=[2, 1, 0.5])
    # Place at (1, 0, 0) with no rotation
    t = np.eye(4)
    t[:3, 3] = [1, 0, 0]
    # Apply Y-up conversion
    yup = rotation_matrix(-np.pi / 2, [1, 0, 0])
    t_yup = yup @ t
    test_scene.add_geometry(box, transform=t_yup, geom_name="test_box")

    # Export to GLB
    test_glb = os.path.join(OUTPUT_DIR, "test_roundtrip.glb")
    trimesh.exchange.export.export_mesh(test_scene, test_glb)

    # Re-import
    reimported = trimesh.load(test_glb)
    if isinstance(reimported, trimesh.Scene):
        for name, geom in reimported.geometry.items():
            transform, _ = reimported.graph.get(name)
            pos = transform[:3, 3]
            fwd = transform[:3, 0]
            print(f"  Reimported '{name}': pos={pos}, fwd_x={fwd}")

        # Check if there's a root transform
        if hasattr(reimported.graph, 'base_frame'):
            print(f"  Base frame: {reimported.graph.base_frame}")
        
        # Get the world-space bounds
        print(f"  Reimported scene bounds: {reimported.bounds}")
    else:
        print(f"  Reimported as single mesh, bounds: {reimported.bounds}")

    # ─── 3. Key insight: check what Blender would see ───
    print("\n" + "=" * 80)
    print("BLENDER IMPORT SIMULATION")
    print("=" * 80)
    print("""
When Blender imports a GLB file:
1. Blender's GLTF importer reads Y-up transforms
2. It converts Y-up to Z-up (Blender convention)
3. This is equivalent to Rx(+90°) on the root

The compose code applies Rx(-90°) to convert Z-up → Y-up for GLB.
Blender then applies Rx(+90°) to convert back Y-up → Z-up.
Net effect: identity. Objects should appear at their original Z-up positions.

BUT: Does trimesh's GLB exporter ALSO add a Y-up conversion?
If so, the chain would be:
  Z-up data → Rx(-90°) [manual] → Rx(-90°) [trimesh] → GLB (now double-rotated)
  → Rx(+90°) [Blender] → still has one extra Rx(-90°)!
""")

    # ─── 4. Test trimesh GLB export behavior ───
    # Create Z-up scene (no Y-up conversion), export to GLB, see what happens
    test2 = trimesh.Scene()
    box2 = trimesh.creation.box(extents=[2, 1, 0.5])
    t2 = np.eye(4)
    t2[:3, 3] = [1, 2, 0.25]  # On ground at (1,2), height 0.25
    test2.add_geometry(box2, transform=t2, geom_name="box_zup")

    test2_glb = os.path.join(OUTPUT_DIR, "test_zup_no_convert.glb")
    trimesh.exchange.export.export_mesh(test2, test2_glb)

    reimp2 = trimesh.load(test2_glb)
    print("Test: Z-up box at (1, 2, 0.25) exported to GLB and reimported:")
    print(f"  Reimported bounds: {reimp2.bounds}")

    # Also test with explicit Y-up conversion
    test3 = trimesh.Scene()
    box3 = trimesh.creation.box(extents=[2, 1, 0.5])
    t3 = np.eye(4)
    t3[:3, 3] = [1, 2, 0.25]
    yup = rotation_matrix(-np.pi / 2, [1, 0, 0])
    t3_yup = yup @ t3
    test3.add_geometry(box3, transform=t3_yup, geom_name="box_yup")

    test3_glb = os.path.join(OUTPUT_DIR, "test_with_yup_convert.glb")
    trimesh.exchange.export.export_mesh(test3, test3_glb)

    reimp3 = trimesh.load(test3_glb)
    print(f"\nTest: Same box with Rx(-90°) conversion, exported and reimported:")
    print(f"  Reimported bounds: {reimp3.bounds}")
    print(f"  Expected if no extra conversion: box at (1, 0.25, -2), size (2, 0.5, 1)")
    print(f"  Expected if extra Rx(-90°) added by trimesh: box at (1, 2, -0.25), size (2, 1, 0.5)")

    # ─── 5. Definitive test: export GLB, reimport, check position ───
    print("\n" + "=" * 80)
    print("DEFINITIVE GLB ROUND-TRIP TEST")
    print("=" * 80)
    scene_composed = trimesh.Scene()
    for i, inst in enumerate(instances):
        uid = inst.get("model_uid", "")
        if not uid:
            continue
        mesh, transform = compute_transform_official(inst, ASSET_BASE)
        if mesh is None:
            continue
        cat = inst.get("category", "")
        scene_composed.add_geometry(mesh, transform=transform, geom_name=f"{i}_{cat}")

    # Add structure meshes
    for name in ["wall.glb", "floor.glb"]:
        glb_path = os.path.join(mesh_dir if 'mesh_dir' in dir() else os.path.join(SCENE_DIR, "StructureMesh"), name)
        if os.path.exists(glb_path):
            struct = trimesh.load(glb_path)
            scene_composed.add_geometry(struct, geom_name=name.replace(".glb", "_s"))

    composed_glb = os.path.join(OUTPUT_DIR, "scene_composed_official.glb")
    trimesh.exchange.export.export_mesh(scene_composed, composed_glb)

    # Reimport and check bounds
    reimported_scene = trimesh.load(composed_glb)
    print(f"Composed scene bounds: {scene_composed.bounds}")
    print(f"Reimported scene bounds: {reimported_scene.bounds}")

    if not np.allclose(scene_composed.bounds, reimported_scene.bounds, atol=0.01):
        print("  ⚠️  BOUNDS MISMATCH! GLB round-trip is NOT preserving geometry!")
        print(f"  Diff: {reimported_scene.bounds - scene_composed.bounds}")
    else:
        print("  ✓ GLB round-trip preserves geometry correctly")


if __name__ == "__main__":
    main()
