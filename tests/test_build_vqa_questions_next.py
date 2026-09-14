from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_vqa_questions_next.py"
spec = spec_from_file_location("build_vqa_questions_next", MODULE_PATH)
mod = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_build_a2_prompt_uses_canonical_category_and_color_disambiguator():
    prompt = mod.build_a2_prompt(
        {
            "target_rule": "nearest",
            "target_category": "dresser",
            "canonical_category": "cabinet",
            "target_color_name": "brown",
            "start_id": 4,
        },
        embodied=False,
    )

    assert "Start at display ID 4." in prompt["user"]
    assert "nearest brown cabinet" in prompt["user"]
    assert "dresser" not in prompt["user"]


def test_build_a1_prompt_uses_display_ids_not_waypoint_ids():
    prompt = mod.build_a1_prompt({}, embodied=False)

    assert "display IDs" in prompt["user"]
    assert "waypoint IDs" not in prompt["user"]
    assert "candidate points" in prompt["user"]


def test_build_a2_prompt_uses_display_ids_not_waypoint_ids():
    prompt = mod.build_a2_prompt(
        {
            "target_rule": "nearest",
            "canonical_category": "cabinet",
            "start_id": 4,
        },
        embodied=True,
    )

    assert "Start at display ID 4." in prompt["user"]
    assert "display IDs" in prompt["user"]
    assert "waypoint IDs" not in prompt["user"]


def test_build_a2_prompt_requires_start_id():
    with pytest.raises(ValueError, match="start_id"):
        mod.build_a2_prompt(
            {
                "target_rule": "nearest",
                "canonical_category": "cabinet",
            },
            embodied=False,
        )


def test_vqa_builder_passthroughs_facts_hash(tmp_path):
    gt_dir = tmp_path / "gt"
    out_dir = tmp_path / "vqa"
    gt_dir.mkdir()
    out_dir.mkdir()

    (gt_dir / "gt_next_a2.jsonl").write_text(
        json.dumps(
            {
                "question_id": "demo_v0_routing_v0_c001_a2",
                "source": "internscene",
                "task": "a2",
                "scene_id": "demo",
                "view_id": 0,
                "routing_id": "routing_v0_c001",
                "image_path": "rgb_overlay_a2.png",
                "visible_waypoints_path": "visible_waypoints_a2.json",
                "start_id": 1,
                "goal_id": 3,
                "goal_ids": [3],
                "path_ids": [1, 2, 3],
                "target_rule": "nearest",
                "canonical_category": "cabinet",
                "facts_hash": "hash-demo",
            }
        )
        + "\n"
    )

    mod.main = mod.main
    import sys

    argv = sys.argv
    try:
        sys.argv = [
            "build_vqa_questions_next.py",
            "--gt-dir",
            str(gt_dir),
            "--output-dir",
            str(out_dir),
        ]
        mod.main()
    finally:
        sys.argv = argv

    lines = (out_dir / "vqa_next_a2.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["ground_truth"]["facts_hash"] == "hash-demo"


def test_vqa_builder_passthroughs_tier_family_and_axes(tmp_path):
    gt_dir = tmp_path / "gt"
    out_dir = tmp_path / "vqa"
    gt_dir.mkdir()
    out_dir.mkdir()

    (gt_dir / "gt_next_b1.jsonl").write_text(
        json.dumps(
            {
                "question_id": "demo_v0_b1",
                "source": "internscene",
                "task": "b1",
                "tier_family": "affordance",
                "tier": "hard",
                "scene_id": "demo",
                "view_id": 0,
                "image_path": "rgb_overlay_b1.png",
                "visible_waypoints_path": "visible_waypoints_b1.json",
                "all_ids": [1, 2],
                "walkable_ids": [1],
                "affordance_axes": {
                    "clutter_axis": "medium",
                    "boundary_axis": "low",
                    "embodiment_gap_axis": "medium",
                },
                "affordance_validity": {
                    "eligible": True,
                    "failed_checks": [],
                    "check_results": {"has_candidates": True},
                },
                "facts_hash": "hash-b1",
            }
        )
        + "\n"
    )

    import sys

    argv = sys.argv
    try:
        sys.argv = [
            "build_vqa_questions_next.py",
            "--gt-dir",
            str(gt_dir),
            "--output-dir",
            str(out_dir),
        ]
        mod.main()
    finally:
        sys.argv = argv

    rec = json.loads((out_dir / "vqa_next_b1.jsonl").read_text().strip())
    assert rec["tier_family"] == "affordance"
    assert rec["tier"] == "hard"
    assert rec["ground_truth"]["affordance_axes"]["clutter_axis"] == "medium"
    assert rec["ground_truth"]["affordance_validity"]["eligible"] is True
