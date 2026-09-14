"""
NavBench3D - Step 1: Download InternScenes data and filter suitable scenes.

Usage:
    python scripts/download_and_filter.py --config configs/default.yaml
"""

import os
import json
import argparse
import math
from pathlib import Path
from typing import Optional

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def compute_floor_area(structure_mesh_dir: str) -> Optional[float]:
    """
    Estimate floor area from StructureMesh data.
    Attempts to parse the wall.glb bounding box or falls back to layout-based estimation.
    """
    # For InternScenes, we estimate area from layout.json bounding boxes
    # This is a fallback; more accurate would be to parse the glb mesh
    return None  # Will be computed from layout


def estimate_area_from_layout(layout: dict) -> float:
    """Estimate navigable floor area from object bounding boxes in layout.
    
    Uses the convex hull area of object XY positions as a more accurate
    estimate, with a small margin for walls beyond the outermost objects.
    """
    if not layout:
        return 0.0

    all_positions = []
    for obj in layout:
        bbox = obj.get("bbox", [])
        if len(bbox) >= 6:
            x, y = bbox[0], bbox[1]
            sx, sy = bbox[3], bbox[4]
            # Add object corners for better area coverage
            half_sx, half_sy = sx / 2, sy / 2
            all_positions.extend([
                (x - half_sx, y - half_sy),
                (x + half_sx, y - half_sy),
                (x - half_sx, y + half_sy),
                (x + half_sx, y + half_sy),
            ])
        elif len(bbox) >= 3:
            all_positions.append((bbox[0], bbox[1]))

    if len(all_positions) < 3:
        if len(all_positions) == 2:
            xs = [p[0] for p in all_positions]
            ys = [p[1] for p in all_positions]
            return (max(xs) - min(xs) + 0.5) * (max(ys) - min(ys) + 0.5)
        return 0.0

    xs = [p[0] for p in all_positions]
    ys = [p[1] for p in all_positions]

    # Use bounding box of object extents + 0.5m wall margin
    width = max(xs) - min(xs) + 0.5
    depth = max(ys) - min(ys) + 0.5
    return width * depth


def filter_scene(layout: dict, config: dict) -> tuple[bool, str]:
    """
    Check if a scene meets the filtering criteria.
    Returns (passes, reason).
    """
    filter_cfg = config["filter"]

    if not layout or not isinstance(layout, list):
        return False, "invalid_layout"

    num_objects = len(layout)
    if num_objects < filter_cfg["min_objects"]:
        return False, f"too_few_objects ({num_objects})"
    if num_objects > filter_cfg["max_objects"]:
        return False, f"too_many_objects ({num_objects})"

    # Estimate area
    area = estimate_area_from_layout(layout)
    if area < filter_cfg["min_area"]:
        return False, f"area_too_small ({area:.1f}m²)"
    if area > filter_cfg["max_area"]:
        return False, f"area_too_large ({area:.1f}m²)"

    # Check if there are navigable pairs (objects far enough apart)
    positions = []
    categories = []
    for obj in layout:
        bbox = obj.get("bbox", [])
        if len(bbox) >= 3:
            positions.append((bbox[0], bbox[1], bbox[2]))
            categories.append(obj.get("category", "unknown"))

    if len(positions) < 2:
        return False, "insufficient_positioned_objects"

    # Check for reasonable navigation distances
    has_valid_pair = False
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(positions[i], positions[j])))
            if filter_cfg["min_nav_distance"] <= dist <= filter_cfg["max_nav_distance"]:
                has_valid_pair = True
                break
        if has_valid_pair:
            break

    if not has_valid_pair:
        return False, "no_valid_navigation_pairs"

    return True, "passed"


def check_assets_available(layout: list, asset_base: str) -> tuple[bool, list[str]]:
    """Check if all asset models referenced by a scene are available on disk.
    
    Checks actual file existence, not just library directory.
    """
    missing = []
    for obj in layout:
        uid = obj.get("model_uid", "")
        if not uid:
            continue

        # Determine actual file path
        if uid.startswith("partnet_mobility"):
            path = os.path.join(asset_base, uid, "whole.glb")
        else:
            path = os.path.join(asset_base, uid + ".glb")

        if not os.path.exists(path):
            missing.append(uid)
    return len(missing) == 0, missing


def process_scenes(raw_dir: str, output_dir: str, config: dict):
    """Process all scenes from InternScenes and filter suitable ones.

    Handles the actual InternScenes directory structure:
      Layout_info/{dataset_name}/{scene_id}/layout.json   (Real2Sim)
      InternScenes_Gen/Layout_info/{room_type}/{scene_id}/layout.json  (Generated)
    """
    raw_path = Path(raw_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    asset_base = raw_path / "asset_library"

    stats = {"total": 0, "passed": 0, "failed": 0, "reasons": {},
             "by_source": {}}

    # Optional: recommended target categories (not mandatory for filtering)
    rec_categories = set()
    target_cfg = config.get("targets", {})
    rec_file = target_cfg.get("recommended_categories_file")
    if rec_file:
        rec_path = Path(rec_file).expanduser().resolve()
        if rec_path.exists():
            try:
                with open(rec_path) as f:
                    data = json.load(f)
                rec_categories = set(data.get("recommended_categories", []))
            except (json.JSONDecodeError, IOError):
                rec_categories = set()

    # Collect all scene directories with two-level structure
    scene_entries = []  # list of (source_label, dataset_name, scene_dir)

    # 1) Root Layout_info — Real2Sim scenes
    root_layout = raw_path / "Layout_info"
    if root_layout.exists():
        for dataset_dir in sorted(root_layout.iterdir()):
            if not dataset_dir.is_dir():
                continue
            dataset_name = dataset_dir.name
            for scene_dir in sorted(dataset_dir.iterdir()):
                if scene_dir.is_dir() and (scene_dir / "layout.json").exists():
                    scene_entries.append(("real2sim", dataset_name, scene_dir))

    # 2) InternScenes_Gen/Layout_info — Generated scenes
    gen_layout = raw_path / "InternScenes_Gen" / "Layout_info"
    if gen_layout.exists():
        for room_dir in sorted(gen_layout.iterdir()):
            if not room_dir.is_dir():
                continue
            room_type = room_dir.name
            for scene_dir in sorted(room_dir.iterdir()):
                if scene_dir.is_dir() and (scene_dir / "layout.json").exists():
                    scene_entries.append(("generated", room_type, scene_dir))

    print(f"Found {len(scene_entries)} total scenes "
          f"(Real2Sim: {sum(1 for s in scene_entries if s[0]=='real2sim')}, "
          f"Generated: {sum(1 for s in scene_entries if s[0]=='generated')})")

    for source, dataset_name, scene_dir in scene_entries:
        stats["total"] += 1
        scene_id = scene_dir.name
        # Use dataset_name/scene_id as unique key to avoid collisions
        qualified_id = f"{dataset_name}__{scene_id}"

        layout_file = scene_dir / "layout.json"
        try:
            with open(layout_file) as f:
                layout = json.load(f)
        except (json.JSONDecodeError, IOError):
            stats["failed"] += 1
            stats["reasons"]["json_error"] = stats["reasons"].get("json_error", 0) + 1
            continue

        # Check asset availability first
        if asset_base.exists():
            assets_ok, missing = check_assets_available(layout, str(asset_base))
            if not assets_ok:
                stats["failed"] += 1
                stats["reasons"]["missing_assets"] = stats["reasons"].get("missing_assets", 0) + 1
                continue

        passes, reason = filter_scene(layout, config)

        if passes:
            stats["passed"] += 1
            stats["by_source"][source] = stats["by_source"].get(source, 0) + 1

            scene_output = output_path / qualified_id
            scene_output.mkdir(parents=True, exist_ok=True)

            # Save filtered layout
            with open(scene_output / "layout.json", "w") as f:
                json.dump(layout, f, indent=2)

            # Symlink StructureMesh if exists
            mesh_dir = scene_dir / "StructureMesh"
            mesh_link = scene_output / "StructureMesh"
            if mesh_dir.exists() and not mesh_link.exists():
                mesh_link.symlink_to(mesh_dir.resolve())

            # Save scene metadata
            recommended_target_count = 0
            if rec_categories:
                for obj in layout:
                    if obj.get("category", "unknown") in rec_categories:
                        recommended_target_count += 1

            metadata = {
                "scene_id": scene_id,
                "qualified_id": qualified_id,
                "source": source,
                "dataset": dataset_name,
                "num_objects": len(layout),
                "estimated_area": estimate_area_from_layout(layout),
                "categories": list(set(obj.get("category", "unknown") for obj in layout)),
                "asset_libraries": list(set(
                    obj.get("model_uid", "").split("/")[0]
                    for obj in layout if obj.get("model_uid")
                )),
                "recommended_target_count": recommended_target_count,
            }
            with open(scene_output / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
        else:
            stats["failed"] += 1
            stats["reasons"][reason.split(" ")[0]] = stats["reasons"].get(reason.split(" ")[0], 0) + 1

        if stats["total"] % 500 == 0:
            print(f"  Processed {stats['total']} scenes, {stats['passed']} passed")

    # Save filtering stats
    print(f"\n=== Filtering Summary ===")
    print(f"Total scenes: {stats['total']}")
    print(f"Passed: {stats['passed']}")
    print(f"Failed: {stats['failed']}")
    if stats["by_source"]:
        print(f"Passed by source:")
        for src, cnt in stats["by_source"].items():
            print(f"  {src}: {cnt}")
    print(f"Rejection reasons:")
    for reason, count in sorted(stats["reasons"].items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")

    with open(output_path / "filter_stats.json", "w") as f:
        json.dump(stats, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Download and filter InternScenes data")
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--raw-dir", default=None, help="Override raw data directory")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--download", action="store_true", help="Download InternScenes from HuggingFace")
    args = parser.parse_args()

    config = load_config(args.config)

    raw_dir = args.raw_dir or config["data"]["internscenes_dir"]
    output_dir = args.output_dir or config["data"]["scenes_dir"]

    if args.download:
        print("Downloading InternScenes from HuggingFace...")
        print("Dataset: InternRobotics/InternScenes")
        print(f"Target directory: {raw_dir}")
        os.makedirs(raw_dir, exist_ok=True)
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="InternRobotics/InternScenes",
                repo_type="dataset",
                local_dir=raw_dir,
            )
            print("Download complete.")
        except ImportError:
            print("Please install huggingface_hub: pip install huggingface_hub")
            print("Then run: huggingface-cli download InternRobotics/InternScenes --repo-type dataset --local-dir", raw_dir)
            return

    if not Path(raw_dir).exists():
        print(f"Raw data directory not found: {raw_dir}")
        print("Use --download flag to download from HuggingFace, or specify --raw-dir")
        return

    print(f"Filtering scenes from {raw_dir} -> {output_dir}")
    process_scenes(raw_dir, output_dir, config)


if __name__ == "__main__":
    main()
