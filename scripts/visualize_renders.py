"""
Visualize rendered scenes: create a grid of topdown + RGB + depth + normal views
for each scene, and a summary sheet of all scenes.

Usage:
    conda run -n navbench3d python scripts/visualize_renders.py \
        --renders-dir /datadisk/NavBench3D/renders \
        --scenes-dir /datadisk/NavBench3D/scenes \
        --output-dir /datadisk/NavBench3D/render_viz \
        [--scene-ids sid1 sid2 ...]
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np


def load_image(path: str) -> np.ndarray | None:
    if not os.path.isfile(path):
        return None
    return mpimg.imread(path)


def load_image_multi(*paths: str) -> np.ndarray | None:
    """Try multiple paths, return the first one that exists."""
    for p in paths:
        img = load_image(p)
        if img is not None:
            return img
    return None


def visualize_single_scene(scene_id: str, renders_dir: str, scenes_dir: str, output_dir: str):
    """Create a visualization for a single scene showing topdown, RGB, depth, normal views."""
    render_path = Path(renders_dir) / scene_id
    if not render_path.is_dir():
        print(f"  No renders for {scene_id}")
        return None

    # Load scene info
    layout_path = Path(scenes_dir) / scene_id / "layout.json"
    n_objects = 0
    if layout_path.is_file():
        layout = json.load(open(layout_path))
        n_objects = len([o for o in layout if o.get("model_uid")])

    # Load camera / visibility info
    cameras_path = render_path / "cameras.json"
    cameras = []
    if cameras_path.is_file():
        cameras = json.load(open(cameras_path))

    # Find views
    view_dirs = sorted([d for d in render_path.iterdir() if d.is_dir() and d.name.startswith("view_")])
    n_views = len(view_dirs)

    # Layout: row 0 = topdown (spanning), row 1-3 = views (rgb, depth, normal)
    # Or simpler: columns = views, rows = [rgb, depth, normal], plus topdown on side

    # Create figure: topdown on left, views as columns on right
    n_cols = max(n_views, 1) + 1  # +1 for topdown column
    n_rows = 3  # rgb, depth, normal

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(f"{scene_id}\n{n_objects} objects, {n_views} views", fontsize=14, fontweight="bold")

    # Compute total visibility coverage
    all_vis_ids = set()
    for cam in cameras:
        for vo in cam.get("visible_objects", []):
            all_vis_ids.add(vo["id"])
    coverage_str = f" | Coverage: {len(all_vis_ids)}/{n_objects}" if cameras else ""
    fig.suptitle(f"{scene_id}\n{n_objects} objects, {n_views} views{coverage_str}",
                 fontsize=14, fontweight="bold")

    # Column 0: topdown (merged into first cell)
    topdown = load_image(str(render_path / "topdown.png"))
    for r in range(n_rows):
        axes[r, 0].axis("off")
    if topdown is not None:
        axes[0, 0].imshow(topdown)
        axes[0, 0].set_title("Top-down", fontsize=10)
    else:
        axes[0, 0].text(0.5, 0.5, "No topdown", ha="center", va="center", transform=axes[0, 0].transAxes)

    # Columns 1+: each view
    row_labels = ["RGB", "Depth", "Normal"]
    for vi, vdir in enumerate(view_dirs):
        col = vi + 1

        rgb = load_image(str(vdir / "rgb.png"))
        # Try both naming conventions
        depth = load_image_multi(str(vdir / "depth.png"), str(vdir / "depth_0001.png"))
        normal = load_image_multi(str(vdir / "normal.png"), str(vdir / "normal_0001.png"))

        images = [rgb, depth, normal]
        for r, (img, label) in enumerate(zip(images, row_labels)):
            if img is not None:
                axes[r, col].imshow(img)
            else:
                axes[r, col].text(0.5, 0.5, "Missing", ha="center", va="center",
                                  transform=axes[r, col].transAxes)
            axes[r, col].axis("off")
            if vi == 0:
                axes[r, 0].set_ylabel(label, fontsize=12, rotation=0, labelpad=60, va="center")
            if r == 0:
                # Add visibility info to title
                vis_info = ""
                if vi < len(cameras):
                    cam = cameras[vi]
                    vo = cam.get("visible_objects", [])
                    rv = sum(1 for v in vo if v.get("raycast_verified", False))
                    cats = sorted(set(v["category"] for v in vo))
                    vis_info = f"\n{len(vo)} vis ({rv} rc) | {', '.join(cats[:4])}"
                    if len(cats) > 4:
                        vis_info += "..."
                axes[r, col].set_title(f"View {vi}{vis_info}", fontsize=8)

    # Hide unused columns
    for col in range(n_views + 1, n_cols):
        for r in range(n_rows):
            axes[r, col].axis("off")

    plt.tight_layout()
    out_path = Path(output_dir) / f"{scene_id}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")
    return str(out_path)


def create_summary_sheet(scene_ids: list[str], renders_dir: str, scenes_dir: str, output_dir: str):
    """Create a summary sheet showing topdown + view_0 RGB for all scenes."""
    n_scenes = len(scene_ids)
    n_cols = min(5, n_scenes)
    n_rows = (n_scenes + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows * 2, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_rows * 2 == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    fig.suptitle(f"NavBench3D Render Summary ({n_scenes} scenes)", fontsize=16, fontweight="bold")

    for i, sid in enumerate(scene_ids):
        col = i % n_cols
        row_base = (i // n_cols) * 2

        render_path = Path(renders_dir) / sid

        # Load scene info
        layout_path = Path(scenes_dir) / sid / "layout.json"
        n_objects = 0
        if layout_path.is_file():
            layout = json.load(open(layout_path))
            n_objects = len([o for o in layout if o.get("model_uid")])

        # Load visibility info
        cameras_path = render_path / "cameras.json"
        total_vis = 0
        if cameras_path.is_file():
            cams_data = json.load(open(cameras_path))
            vis_ids = set()
            for cam in cams_data:
                for vo in cam.get("visible_objects", []):
                    vis_ids.add(vo["id"])
            total_vis = len(vis_ids)

        # Topdown (row 0)
        topdown = load_image(str(render_path / "topdown.png"))
        if topdown is not None:
            axes[row_base, col].imshow(topdown)
        axes[row_base, col].axis("off")

        # Shorten scene_id for display
        short_id = sid.split("__")[-1][:12] + "..."
        dataset = sid.split("__")[0]
        vis_str = f"\nVis: {total_vis}/{n_objects}" if total_vis else ""
        axes[row_base, col].set_title(f"{dataset}\n{short_id}\n{n_objects} objs{vis_str}", fontsize=9)

        # View 0 RGB (row 1)
        rgb = load_image(str(render_path / "view_0" / "rgb.png"))
        if rgb is not None:
            axes[row_base + 1, col].imshow(rgb)
        axes[row_base + 1, col].axis("off")

    # Hide unused cells
    for i in range(n_scenes, n_rows * n_cols):
        col = i % n_cols
        row_base = (i // n_cols) * 2
        axes[row_base, col].axis("off")
        axes[row_base + 1, col].axis("off")

    plt.tight_layout()
    out_path = Path(output_dir) / "summary_sheet.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Summary sheet saved: {out_path}")
    return str(out_path)


def pixel_quality_report(scene_ids: list[str], renders_dir: str):
    """Generate pixel-level quality stats for rendered images."""
    print("\n=== Render Quality Report ===")
    print(f"{'Scene':<55} {'Views':>5} {'RGB%':>6} {'Depth%':>7} {'Normal%':>8}")
    print("-" * 85)

    for sid in scene_ids:
        render_path = Path(renders_dir) / sid
        view_dirs = sorted([d for d in render_path.iterdir() if d.is_dir() and d.name.startswith("view_")])

        rgb_pcts, depth_pcts, normal_pcts = [], [], []

        for vdir in view_dirs:
            rgb = load_image(str(vdir / "rgb.png"))
            depth = load_image_multi(str(vdir / "depth.png"), str(vdir / "depth_0001.png"))
            normal = load_image_multi(str(vdir / "normal.png"), str(vdir / "normal_0001.png"))

            if rgb is not None:
                non_black = np.mean(np.any(rgb[..., :3] > 0.01, axis=-1)) * 100
                rgb_pcts.append(non_black)
            if depth is not None:
                non_zero = np.mean(depth > 0.001) * 100
                depth_pcts.append(non_zero)
            if normal is not None:
                non_black = np.mean(np.any(normal[..., :3] > 0.01, axis=-1)) * 100
                normal_pcts.append(non_black)

        def avg(lst):
            return f"{sum(lst)/len(lst):.1f}" if lst else "N/A"

        print(f"{sid:<55} {len(view_dirs):>5} {avg(rgb_pcts):>6} {avg(depth_pcts):>7} {avg(normal_pcts):>8}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--renders-dir", required=True)
    parser.add_argument("--scenes-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scene-ids", nargs="*", default=None,
                        help="Specific scene IDs to visualize. If not provided, uses all rendered scenes.")
    args = parser.parse_args()

    if args.scene_ids:
        scene_ids = args.scene_ids
    else:
        scene_ids = sorted(os.listdir(args.renders_dir))

    print(f"Visualizing {len(scene_ids)} scenes...")

    # Per-scene detail views
    for sid in scene_ids:
        print(f"\nProcessing {sid}...")
        visualize_single_scene(sid, args.renders_dir, args.scenes_dir, args.output_dir)

    # Summary sheet
    create_summary_sheet(scene_ids, args.renders_dir, args.scenes_dir, args.output_dir)

    # Quality report
    pixel_quality_report(scene_ids, args.renders_dir)


if __name__ == "__main__":
    main()
