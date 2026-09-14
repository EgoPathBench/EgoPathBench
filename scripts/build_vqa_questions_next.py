#!/usr/bin/env python3
"""
Build Next VQA manifests from gt_next_*.jsonl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SYSTEM_A1 = "You are a navigation perception assistant."
SYSTEM_A2 = "You are a navigation planning assistant."

RULE_TEXT = {
    "nearest": "nearest",
    "farthest": "farthest",
    "leftmost": "leftmost",
}


def require_key(rec: dict, key: str, *, context: str) -> object:
    if key not in rec or rec[key] is None:
        raise ValueError(f"{context} missing required field: {key}")
    return rec[key]


def validate_record_for_task(rec: dict, task: str) -> None:
    common = [
        "question_id",
        "source",
        "task",
        "scene_id",
        "view_id",
        "image_path",
        "visible_waypoints_path",
    ]
    for key in common:
        require_key(rec, key, context=f"{task} record")

    if task in ("a1", "b1"):
        for key in ("all_ids", "walkable_ids"):
            require_key(rec, key, context=f"{task} record")
    elif task in ("a2", "b2"):
        for key in ("routing_id", "start_id", "goal_id", "goal_ids", "path_ids", "target_rule"):
            require_key(rec, key, context=f"{task} record")


def build_target_phrase(rec: dict) -> str:
    category = rec.get("canonical_category") or rec.get("target_category")
    if not category:
        raise ValueError("routing prompt missing target category")
    color = rec.get("target_color_name")
    if color:
        return f"{color} {category}"
    return str(category)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def build_a1_prompt(rec: dict, embodied: bool = False) -> dict:
    sys = SYSTEM_A1
    if embodied:
        user = (
            "The image shows numbered candidate points. "
            "The robot has diameter 0.6m. "
            "Output a JSON array of display IDs that are walkable for the robot."
        )
    else:
        user = (
            "The image shows numbered candidate points. "
            "Output a JSON array of display IDs that are walkable."
        )
    return {"system": sys, "user": user}


def build_a2_prompt(rec: dict, embodied: bool = False) -> dict:
    sys = SYSTEM_A2
    rule = RULE_TEXT.get(rec.get("target_rule", "nearest"), "nearest")
    target_phrase = build_target_phrase(rec)
    start_id = require_key(rec, "start_id", context="routing prompt")
    if embodied:
        user = (
            f"The image shows numbered candidate points. The robot has diameter 0.6m. "
            f"Start at display ID {start_id}. Navigate to the {rule} {target_phrase}. "
            "Output a JSON array of display IDs representing a valid path."
        )
    else:
        user = (
            f"The image shows numbered candidate points. "
            f"Start at display ID {start_id}. Navigate to the {rule} {target_phrase}. "
            "Output a JSON array of display IDs representing a valid path."
        )
    return {"system": sys, "user": user}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Next VQA from gt_next jsonl files.")
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = ["a1", "b1", "a2", "b2"]
    for task in tasks:
        gt_path = gt_dir / f"gt_next_{task}.jsonl"
        records = load_jsonl(gt_path)
        if not records:
            continue

        out_path = out_dir / f"vqa_next_{task}.jsonl"
        with open(out_path, "w") as f:
            for rec in records:
                validate_record_for_task(rec, task)
                if task in ("a1", "b1"):
                    prompt = build_a1_prompt(rec, embodied=(task == "b1"))
                    answer = rec["walkable_ids"]
                else:
                    prompt = build_a2_prompt(rec, embodied=(task == "b2"))
                    answer = rec["path_ids"]

                vqa = {
                    "question_id": rec["question_id"],
                    "source": rec["source"],
                    "task": rec["task"],
                    "tier_family": rec.get("tier_family"),
                    "tier": rec.get("tier"),
                    "scene_id": rec["scene_id"],
                    "view_id": rec["view_id"],
                    "routing_id": rec.get("routing_id"),
                    "image_path": rec["image_path"],
                    "visible_waypoints_path": rec["visible_waypoints_path"],
                    "system_prompt": prompt["system"],
                    "user_prompt": prompt["user"],
                    "ground_truth": {
                        "answer": answer,
                        "all_ids": rec.get("all_ids"),
                        "walkable_ids": rec.get("walkable_ids"),
                        "start_id": rec.get("start_id"),
                        "goal_id": rec.get("goal_id"),
                        "goal_ids": rec.get("goal_ids"),
                        "target_rule": rec.get("target_rule"),
                        "target_category": rec.get("target_category"),
                        "canonical_category": rec.get("canonical_category"),
                        "semantic_group_id": rec.get("semantic_group_id"),
                        "target_color_name": rec.get("target_color_name"),
                        "target_id": rec.get("target_id"),
                        "target_distance_m": rec.get("target_distance_m"),
                        "instruction_identity": rec.get("instruction_identity"),
                        "instruction_selection": rec.get("instruction_selection"),
                        "routing_complexity": rec.get("routing_complexity"),
                        "navigation_axes": rec.get("navigation_axes"),
                        "navigation_validity": rec.get("navigation_validity"),
                        "affordance_axes": rec.get("affordance_axes"),
                        "affordance_validity": rec.get("affordance_validity"),
                        "facts_hash": rec.get("facts_hash"),
                    },
                }
                f.write(json.dumps(vqa, ensure_ascii=True) + "\n")

        print(f"Wrote {out_path}")

    # C task placeholder passthrough
    c_path = gt_dir / "gt_next_c.jsonl"
    if c_path.exists():
        out_c = out_dir / "vqa_next_c.jsonl"
        with open(out_c, "w") as f:
            for rec in load_jsonl(c_path):
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")
        print(f"Wrote {out_c}")


if __name__ == "__main__":
    main()
