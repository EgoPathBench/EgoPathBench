"""
NavBench3D - Step 5: VQA Question Set Construction.

Generates multi-level VQA questions for each (scene, view, target) triplet.
4 difficulty levels with varying auxiliary information.

Usage:
    python scripts/build_vqa_questions.py --config configs/default.yaml \
        --scenes-dir data/scenes --renders-dir data/renders \
        --gt-dir data/gt_paths --output-dir data/vqa_questions
"""

import json
import argparse
import base64
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def encode_image_base64(image_path: str) -> str:
    """Encode image to base64 for VLM API input."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


SYSTEM_PROMPT = """You are a navigation planning assistant. You are standing in an indoor room and can see the scene from your current viewpoint. Your task is to plan a navigation path from your current position to a target object.

Output your answer as a JSON array of 3D waypoint coordinates [[x1,y1,z1], [x2,y2,z2], ...] representing the path from your current position to the target. The coordinates are in meters in the scene's world coordinate system. The z-coordinate for floor-level navigation should be approximately 0.

Requirements:
- The path should avoid obstacles (furniture, walls)
- The path should be reasonably efficient (not unnecessarily long)
- Include at least 2 waypoints (start vicinity and target vicinity)
- Output ONLY the JSON array, no other text"""


def build_level1_prompt(nav_pair: dict, layout: list[dict]) -> dict:
    """
    Level 1 (Easy): Photo + floor plan + JSON layout + target 3D position.
    VLM gets maximum information.
    """
    target_cat = nav_pair["target_category"]
    target_pos = nav_pair["target_position"]
    start_pos = nav_pair["start_position"]

    # Simplify layout for prompt (remove unnecessary fields)
    simplified_layout = []
    for obj in layout:
        bbox = obj.get("bbox", [])
        if len(bbox) >= 6:
            simplified_layout.append({
                "category": obj.get("category", "unknown"),
                "position": [round(bbox[0], 2), round(bbox[1], 2), round(bbox[2], 2)],
                "size": [round(bbox[3], 2), round(bbox[4], 2), round(bbox[5], 2)],
            })

    user_prompt = f"""I am currently at position [{start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.2f}] in this indoor scene.

I need to navigate to the {target_cat} located at position [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}].

Here is the floor plan of the room (attached as the second image).

Here is the complete layout of objects in the scene:
```json
{json.dumps(simplified_layout, indent=2)}
```

Please plan a collision-free navigation path from my current position to the target {target_cat}. Output the path as a JSON array of 3D coordinates."""

    return {
        "system": SYSTEM_PROMPT,
        "user": user_prompt,
        "images": ["current_view", "topdown"],  # Placeholder for actual image paths
        "level": 1,
        "level_name": "easy",
    }


def build_level2_prompt(nav_pair: dict) -> dict:
    """
    Level 2 (Medium): Photo + floor plan + target 3D position.
    No JSON layout - VLM must understand obstacles from floor plan.
    """
    target_cat = nav_pair["target_category"]
    target_pos = nav_pair["target_position"]
    start_pos = nav_pair["start_position"]

    user_prompt = f"""I am currently at position [{start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.2f}] in this indoor scene.

I need to navigate to the {target_cat} located at position [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}].

The first image shows my current view. The second image is a top-down floor plan of the room showing the layout of furniture and obstacles.

Please plan a collision-free navigation path from my current position to the target {target_cat}. Output the path as a JSON array of 3D coordinates."""

    return {
        "system": SYSTEM_PROMPT,
        "user": user_prompt,
        "images": ["current_view", "topdown"],
        "level": 2,
        "level_name": "medium",
    }


def build_level3_prompt(nav_pair: dict) -> dict:
    """
    Level 3 (Hard): Photo + target 3D position only.
    No floor plan, no layout - VLM must infer scene structure from the photo.
    """
    target_cat = nav_pair["target_category"]
    target_pos = nav_pair["target_position"]
    start_pos = nav_pair["start_position"]

    user_prompt = f"""I am currently at position [{start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.2f}] in this indoor scene.

I need to navigate to the {target_cat} located at position [{target_pos[0]:.2f}, {target_pos[1]:.2f}, {target_pos[2]:.2f}].

The image shows my current view of the room. Based on what you can see and your understanding of indoor layouts, plan a collision-free navigation path from my current position to the target.

Output the path as a JSON array of 3D coordinates."""

    return {
        "system": SYSTEM_PROMPT,
        "user": user_prompt,
        "images": ["current_view"],
        "level": 3,
        "level_name": "hard",
    }


def build_level4_prompt(nav_pair: dict) -> dict:
    """
    Level 4 (Expert): Photo + target description only (no coordinates).
    VLM must identify the target, estimate its position, and plan a path.
    """
    target_cat = nav_pair["target_category"]
    start_pos = nav_pair["start_position"]

    user_prompt = f"""I am currently at position [{start_pos[0]:.2f}, {start_pos[1]:.2f}, {start_pos[2]:.2f}] in this indoor scene.

I need to navigate to the {target_cat} in this room. The image shows my current view.

Based on what you can see, locate the {target_cat} (or infer where it might be), and plan a collision-free navigation path from my current position to it.

Output the path as a JSON array of 3D coordinates."""

    return {
        "system": SYSTEM_PROMPT,
        "user": user_prompt,
        "images": ["current_view"],
        "level": 4,
        "level_name": "expert",
    }


def build_vqa_for_scene(
    scene_dir: str,
    renders_dir: str,
    ai_renders_dir: str,
    gt_dir: str,
    output_dir: str,
    config: dict,
):
    """Build VQA questions for a single scene."""
    scene_path = Path(scene_dir)
    scene_id = scene_path.name
    output_path = Path(output_dir) / scene_id
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    with open(scene_path / "layout.json") as f:
        layout = json.load(f)

    gt_file = Path(gt_dir) / scene_id / "navigation_gt.json"
    if not gt_file.exists():
        print(f"  No navigation GT for {scene_id}")
        return 0

    with open(gt_file) as f:
        gt_data = json.load(f)

    levels_to_generate = config["vqa"]["levels"]
    questions = []

    for nav_pair in gt_data.get("navigation_pairs", []):
        view_id = nav_pair["view_id"]

        # Determine image paths (prefer AI-rendered if available)
        ai_view_dir = Path(ai_renders_dir) / scene_id / f"view_{view_id}"
        render_view_dir = Path(renders_dir) / scene_id / f"view_{view_id}"

        if (ai_view_dir / "rgb_photorealistic.png").exists():
            current_view_path = str(ai_view_dir / "rgb_photorealistic.png")
        elif (render_view_dir / "rgb.png").exists():
            current_view_path = str(render_view_dir / "rgb.png")
        else:
            continue

        topdown_path = None
        for d in [Path(ai_renders_dir) / scene_id, Path(renders_dir) / scene_id]:
            if (d / "topdown.png").exists():
                topdown_path = str(d / "topdown.png")
                break

        # Generate questions for each level
        for level in levels_to_generate:
            if level == 1:
                prompt_data = build_level1_prompt(nav_pair, layout)
            elif level == 2:
                prompt_data = build_level2_prompt(nav_pair)
            elif level == 3:
                prompt_data = build_level3_prompt(nav_pair)
            elif level == 4:
                prompt_data = build_level4_prompt(nav_pair)
            else:
                continue

            # Resolve image paths
            resolved_images = []
            for img_ref in prompt_data["images"]:
                if img_ref == "current_view":
                    resolved_images.append(current_view_path)
                elif img_ref == "topdown" and topdown_path:
                    resolved_images.append(topdown_path)

            question = {
                "question_id": f"{scene_id}_v{view_id}_t{nav_pair['target_id']}_l{level}",
                "source": "internscene",
                "scene_id": scene_id,
                "view_id": view_id,
                "level": level,
                "level_name": prompt_data["level_name"],
                "system_prompt": prompt_data["system"],
                "user_prompt": prompt_data["user"],
                "images": resolved_images,
                "ground_truth": {
                    "optimal_path": nav_pair["optimal_path"],
                    "path_length": nav_pair["path_length"],
                    "num_path_points": nav_pair.get("num_path_points", len(nav_pair["optimal_path"])),
                    "target_position": nav_pair["target_position"],
                    "goal_position": nav_pair.get("goal_position", nav_pair["optimal_path"][-1]),
                    "target_category": nav_pair["target_category"],
                    "target_id": nav_pair.get("target_id", -1),
                    "start_position": nav_pair["start_position"],
                    "start_yaw_rad": nav_pair.get("start_yaw_rad", 0.0),
                    "agent_radius": nav_pair.get("agent_radius", 0.30),
                    "grid_resolution": nav_pair.get("grid_resolution", 0.05),
                    "path_validation": nav_pair.get("path_validation", {}),
                    "robot_frame_waypoints": nav_pair.get("robot_frame_waypoints", []),
                    "waypoint_clearances": nav_pair.get("waypoint_clearances", []),
                },
            }
            questions.append(question)

    # Save questions
    with open(output_path / "questions.json", "w") as f:
        json.dump(questions, f, indent=2)

    return len(questions)


def main():
    parser = argparse.ArgumentParser(description="Build VQA Question Set")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--scenes-dir", required=True)
    parser.add_argument("--renders-dir", required=True)
    parser.add_argument("--ai-renders-dir", default=None)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--scene-id", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = args.output_dir or config["data"]["vqa_dir"]
    ai_renders_dir = args.ai_renders_dir or config["data"]["ai_renders_dir"]

    scenes_dir = Path(args.scenes_dir)
    if args.scene_id:
        scene_dirs = [scenes_dir / args.scene_id]
    else:
        scene_dirs = sorted([d for d in scenes_dir.iterdir() if d.is_dir() and (d / "layout.json").exists()])

    print(f"Generating VQA questions for {len(scene_dirs)} scenes")

    total_questions = 0
    level_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    skipped = 0

    for i, scene_dir in enumerate(scene_dirs):
        # Skip if already processed
        q_file = Path(output_dir) / scene_dir.name / "questions.json"
        if q_file.exists():
            skipped += 1
            continue
        print(f"[{i+1}/{len(scene_dirs)}] {scene_dir.name}")
        try:
            n = build_vqa_for_scene(
                str(scene_dir), args.renders_dir, ai_renders_dir,
                args.gt_dir, output_dir, config,
            )
            total_questions += n
            # Approximate level counts
            per_level = n // len(config["vqa"]["levels"]) if config["vqa"]["levels"] else 0
            for lvl in config["vqa"]["levels"]:
                level_counts[lvl] += per_level
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    print(f"\n=== VQA Generation Summary ===")
    print(f"Total questions: {total_questions}")
    for lvl, count in sorted(level_counts.items()):
        names = {1: "Easy", 2: "Medium", 3: "Hard", 4: "Expert"}
        print(f"  Level {lvl} ({names.get(lvl, '?')}): ~{count}")

    # Save merged dataset file
    merged = []
    output_path = Path(output_dir)
    for qf in output_path.rglob("questions.json"):
        with open(qf) as f:
            merged.extend(json.load(f))

    with open(output_path / "all_questions.json", "w") as f:
        json.dump(merged, f, indent=2)
    print(f"\nMerged dataset saved to {output_path / 'all_questions.json'}")


if __name__ == "__main__":
    main()
