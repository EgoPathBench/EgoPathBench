from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_next.py"
spec = spec_from_file_location("evaluate_next", MODULE_PATH)
mod = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def _write_visible_waypoints(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "visible_waypoints": [
                    {"display_id": 1, "waypoint_id": 11, "world_xyz": [0.0, 0.0, 0.0]},
                    {"display_id": 2, "waypoint_id": 12, "world_xyz": [1.0, 0.0, 0.0]},
                    {"display_id": 3, "waypoint_id": 13, "world_xyz": [2.0, 0.0, 0.0]},
                ]
            }
        )
    )


def _write_graph(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "edges": [
                    {"source": 11, "target": 12, "distance": 1.0},
                    {"source": 12, "target": 13, "distance": 1.0},
                    {"source": 11, "target": 13, "distance": 1.2},
                ]
            }
        )
    )


def _write_direct_pairs(path: Path, adjacency: dict[str, list[dict]]) -> None:
    path.write_text(json.dumps({"direct_neighbors": adjacency}))


def test_eval_routing_uses_optimal_length_m_as_spl_denominator(tmp_path):
    scene_dir = tmp_path / "demo_scene"
    view_dir = scene_dir / "view_0"
    view_dir.mkdir(parents=True)
    visible_path = view_dir / "visible_waypoints_a2.json"
    _write_visible_waypoints(visible_path)
    _write_graph(scene_dir / "waypoint_graph_pointmass.json")

    vqa = [
        {
            "question_id": "demo_q",
            "task": "a2",
            "image_path": str(view_dir / "rgb_overlay_a2.png"),
            "visible_waypoints_path": str(visible_path),
            "ground_truth": {
                "start_id": 1,
                "goal_id": 3,
                "goal_ids": [3],
                "optimal_length_m": 1.6,
            },
        }
    ]
    preds = {"demo_q": {"output": [1, 2, 3]}}

    metrics = mod.eval_routing(vqa, preds, embodied=False)

    assert metrics["success"] == 1
    assert metrics["spl"] == 0.8
    assert metrics["spl_all"] == 0.8


def test_eval_routing_treats_recorded_failed_prediction_as_no_output(tmp_path):
    scene_dir = tmp_path / "demo_scene"
    view_dir = scene_dir / "view_0"
    view_dir.mkdir(parents=True)
    visible_path = view_dir / "visible_waypoints_a2.json"
    _write_visible_waypoints(visible_path)
    _write_graph(scene_dir / "waypoint_graph_pointmass.json")

    vqa = [
        {
            "question_id": "demo_q",
            "task": "a2",
            "image_path": str(view_dir / "rgb_overlay_a2.png"),
            "visible_waypoints_path": str(visible_path),
            "ground_truth": {
                "start_id": 1,
                "goal_id": 3,
                "goal_ids": [3],
                "optimal_length_m": 1.6,
            },
        }
    ]
    preds = {
        "demo_q": {
            "output": [1, 2, 3],
            "success": False,
            "error": "benchmark output contains IDs outside candidate ID space: [0]",
        }
    }

    metrics = mod.eval_routing(vqa, preds, embodied=False)

    assert metrics["valid_id_rate"] == 0.0
    assert metrics["success_rate"] == 0.0
    assert metrics["spl_all"] == 0.0


def test_eval_routing_rejects_missing_optimal_length_m(tmp_path):
    scene_dir = tmp_path / "demo_scene"
    view_dir = scene_dir / "view_0"
    view_dir.mkdir(parents=True)
    visible_path = view_dir / "visible_waypoints_a2.json"
    _write_visible_waypoints(visible_path)
    _write_graph(scene_dir / "waypoint_graph_pointmass.json")

    vqa = [
        {
            "question_id": "demo_q",
            "task": "a2",
            "image_path": str(view_dir / "rgb_overlay_a2.png"),
            "visible_waypoints_path": str(visible_path),
            "ground_truth": {
                "start_id": 1,
                "goal_id": 3,
                "goal_ids": [3],
            },
        }
    ]
    preds = {"demo_q": {"output": [1, 2, 3]}}

    with pytest.raises(ValueError, match="missing optimal_length_m"):
        mod.eval_routing(vqa, preds, embodied=False)


def test_c_task_uses_routing_evaluation_semantics(tmp_path):
    scene_dir = tmp_path / "demo_scene"
    view_dir = scene_dir / "view_0"
    view_dir.mkdir(parents=True)
    visible_path = view_dir / "visible_waypoints_c.json"
    _write_visible_waypoints(visible_path)
    _write_graph(scene_dir / "waypoint_graph_embodied.json")

    vqa_path = tmp_path / "vqa_next_c.jsonl"
    gt_path = tmp_path / "gt_next_c.jsonl"
    pred_path = tmp_path / "preds_c.jsonl"
    out_path = tmp_path / "metrics_c.json"

    vqa_row = {
        "question_id": "demo_c",
        "task": "c",
        "image_path": str(view_dir / "rgb_overlay_c.png"),
        "visible_waypoints_path": str(visible_path),
        "ground_truth": {
            "start_id": 1,
            "goal_id": 3,
            "goal_ids": [3],
            "route_semantics": "embodied",
            "route_source_task": "b2",
        },
    }
    vqa_path.write_text(json.dumps(vqa_row) + "\n")
    gt_path.write_text(json.dumps({"question_id": "demo_c", "optimal_length_m": 1.6}) + "\n")
    pred_path.write_text(json.dumps({"question_id": "demo_c", "output": [1, 2, 3]}) + "\n")

    mod.main.__globals__["argparse"].ArgumentParser
    import sys

    old_argv = sys.argv
    try:
        sys.argv = [
            "evaluate_next.py",
            "--vqa",
            str(vqa_path),
            "--predictions",
            str(pred_path),
            "--gt",
            str(gt_path),
            "--output",
            str(out_path),
        ]
        mod.main()
    finally:
        sys.argv = old_argv

    payload = json.loads(out_path.read_text())
    assert payload["task"] == "c"
    assert payload["metrics"]["success"] == 1
    assert payload["metrics"]["success_rate"] == 1.0
    assert payload["metrics"]["spl"] == 0.8


def test_build_display_to_graph_waypoint_repairs_unmapped_visible_waypoints():
    visible = [
        {
            "display_id": 1,
            "waypoint_id": 81,
            "world_xyz": [0.305, 2.082, 0.02],
            "grid_xy": [119, 181],
        }
    ]
    graph_waypoints = [
        {"waypoint_id": 257, "world_xyz": [0.319, 2.125, 0.0], "grid_xy": [119, 182]},
        {"waypoint_id": 160, "world_xyz": [-0.381, 0.375, 0.0], "grid_xy": [105, 147]},
    ]

    mapping = mod.build_display_to_graph_waypoint(visible, graph_waypoints, {257, 160})

    assert mapping[1] == 257


def test_eval_routing_direct_pairs_require_adjacent_legal_segments(tmp_path):
    scene_dir = tmp_path / "demo_scene"
    view_dir = scene_dir / "view_0"
    view_dir.mkdir(parents=True)
    visible_path = view_dir / "visible_waypoints_a2.json"
    _write_visible_waypoints(visible_path)
    direct_pairs_path = tmp_path / "direct_pairs.json"
    _write_direct_pairs(
        direct_pairs_path,
        {
            "1": [{"display_id": 2, "distance_m": 1.0}],
            "2": [{"display_id": 1, "distance_m": 1.0}, {"display_id": 3, "distance_m": 1.0}],
            "3": [{"display_id": 2, "distance_m": 1.0}],
        },
    )

    vqa = [
        {
            "question_id": "demo_q",
            "task": "a2",
            "image_path": str(view_dir / "rgb_overlay_a2.png"),
            "visible_waypoints_path": str(visible_path),
            "ground_truth": {
                "start_id": 1,
                "goal_id": 3,
                "goal_ids": [3],
                "acceptable_goal_ids": [3],
                "optimal_length_m": 2.0,
                "direct_pairs_ref": str(direct_pairs_path),
            },
        }
    ]
    preds = {"demo_q": {"output": [1, 3]}}

    metrics = mod.eval_routing(vqa, preds, embodied=False)

    assert metrics["valid_id_rate"] == 1.0
    assert metrics["valid_path_rate"] == 0.0
    assert metrics["success_rate"] == 0.0


def test_eval_routing_uses_acceptable_goal_ids_for_success(tmp_path):
    scene_dir = tmp_path / "demo_scene"
    view_dir = scene_dir / "view_0"
    view_dir.mkdir(parents=True)
    visible_path = view_dir / "visible_waypoints_a2.json"
    _write_visible_waypoints(visible_path)
    direct_pairs_path = tmp_path / "direct_pairs.json"
    _write_direct_pairs(
        direct_pairs_path,
        {
            "1": [{"display_id": 2, "distance_m": 1.0}],
            "2": [{"display_id": 1, "distance_m": 1.0}, {"display_id": 3, "distance_m": 1.0}],
            "3": [{"display_id": 2, "distance_m": 1.0}],
        },
    )

    vqa = [
        {
            "question_id": "demo_q",
            "task": "a2",
            "image_path": str(view_dir / "rgb_overlay_a2.png"),
            "visible_waypoints_path": str(visible_path),
            "ground_truth": {
                "start_id": 1,
                "goal_id": 2,
                "goal_ids": [2],
                "acceptable_goal_ids": [2, 3],
                "optimal_length_m": 2.0,
                "direct_pairs_ref": str(direct_pairs_path),
            },
        }
    ]
    preds = {"demo_q": {"output": [1, 2, 3]}}

    metrics = mod.eval_routing(vqa, preds, embodied=False)

    assert metrics["success"] == 1
    assert metrics["success_rate"] == 1.0


def test_eval_routing_rejects_single_goal_node_prediction(tmp_path):
    scene_dir = tmp_path / "demo_scene"
    view_dir = scene_dir / "view_0"
    view_dir.mkdir(parents=True)
    visible_path = view_dir / "visible_waypoints_a2.json"
    _write_visible_waypoints(visible_path)
    direct_pairs_path = tmp_path / "direct_pairs.json"
    _write_direct_pairs(
        direct_pairs_path,
        {
            "1": [{"display_id": 2, "distance_m": 1.0}],
            "2": [{"display_id": 1, "distance_m": 1.0}, {"display_id": 3, "distance_m": 1.0}],
            "3": [{"display_id": 2, "distance_m": 1.0}],
        },
    )

    vqa = [
        {
            "question_id": "demo_q",
            "task": "a2",
            "image_path": str(view_dir / "rgb_overlay_a2.png"),
            "visible_waypoints_path": str(visible_path),
            "ground_truth": {
                "start_id": 1,
                "goal_id": 3,
                "goal_ids": [3],
                "acceptable_goal_ids": [3],
                "optimal_length_m": 2.0,
                "direct_pairs_ref": str(direct_pairs_path),
            },
        }
    ]
    preds = {"demo_q": {"output": [3]}}

    metrics = mod.eval_routing(vqa, preds, embodied=False)

    assert metrics["valid_id_rate"] == 1.0
    assert metrics["valid_path_rate"] == 0.0
    assert metrics["success_rate"] == 0.0
    assert metrics["spl"] == 0.0


def test_eval_routing_requires_prediction_to_start_from_start_id(tmp_path):
    scene_dir = tmp_path / "demo_scene"
    view_dir = scene_dir / "view_0"
    view_dir.mkdir(parents=True)
    visible_path = view_dir / "visible_waypoints_a2.json"
    _write_visible_waypoints(visible_path)
    direct_pairs_path = tmp_path / "direct_pairs.json"
    _write_direct_pairs(
        direct_pairs_path,
        {
            "1": [{"display_id": 2, "distance_m": 1.0}],
            "2": [{"display_id": 1, "distance_m": 1.0}, {"display_id": 3, "distance_m": 1.0}],
            "3": [{"display_id": 2, "distance_m": 1.0}],
        },
    )

    vqa = [
        {
            "question_id": "demo_q",
            "task": "a2",
            "image_path": str(view_dir / "rgb_overlay_a2.png"),
            "visible_waypoints_path": str(visible_path),
            "ground_truth": {
                "start_id": 1,
                "goal_id": 3,
                "goal_ids": [3],
                "acceptable_goal_ids": [3],
                "optimal_length_m": 2.0,
                "direct_pairs_ref": str(direct_pairs_path),
            },
        }
    ]
    preds = {"demo_q": {"output": [2, 3]}}

    metrics = mod.eval_routing(vqa, preds, embodied=False)

    assert metrics["valid_id_rate"] == 1.0
    assert metrics["valid_path_rate"] == 0.0
    assert metrics["success_rate"] == 0.0
    assert metrics["spl"] == 0.0


def test_eval_routing_rejects_legal_path_to_wrong_goal(tmp_path):
    scene_dir = tmp_path / "demo_scene"
    view_dir = scene_dir / "view_0"
    view_dir.mkdir(parents=True)
    visible_path = view_dir / "visible_waypoints_a2.json"
    _write_visible_waypoints(visible_path)
    direct_pairs_path = tmp_path / "direct_pairs.json"
    _write_direct_pairs(
        direct_pairs_path,
        {
            "1": [{"display_id": 2, "distance_m": 1.0}],
            "2": [{"display_id": 1, "distance_m": 1.0}, {"display_id": 3, "distance_m": 1.0}],
            "3": [{"display_id": 2, "distance_m": 1.0}],
        },
    )

    vqa = [
        {
            "question_id": "demo_q",
            "task": "a2",
            "image_path": str(view_dir / "rgb_overlay_a2.png"),
            "visible_waypoints_path": str(visible_path),
            "ground_truth": {
                "start_id": 1,
                "goal_id": 3,
                "goal_ids": [3],
                "acceptable_goal_ids": [3],
                "optimal_length_m": 2.0,
                "direct_pairs_ref": str(direct_pairs_path),
            },
        }
    ]
    preds = {"demo_q": {"output": [1, 2]}}

    metrics = mod.eval_routing(vqa, preds, embodied=False)

    assert metrics["valid_id_rate"] == 1.0
    assert metrics["valid_path_rate"] == 1.0
    assert metrics["success_rate"] == 0.0
    assert metrics["spl"] == 0.0


def test_eval_routing_direct_pairs_require_acceptable_goal_ids(tmp_path):
    scene_dir = tmp_path / "demo_scene"
    view_dir = scene_dir / "view_0"
    view_dir.mkdir(parents=True)
    visible_path = view_dir / "visible_waypoints_a2.json"
    _write_visible_waypoints(visible_path)
    direct_pairs_path = tmp_path / "direct_pairs.json"
    _write_direct_pairs(
        direct_pairs_path,
        {
            "1": [{"display_id": 2, "distance_m": 1.0}],
            "2": [{"display_id": 1, "distance_m": 1.0}, {"display_id": 3, "distance_m": 1.0}],
            "3": [{"display_id": 2, "distance_m": 1.0}],
        },
    )

    vqa = [
        {
            "question_id": "demo_q",
            "task": "a2",
            "image_path": str(view_dir / "rgb_overlay_a2.png"),
            "visible_waypoints_path": str(visible_path),
            "ground_truth": {
                "start_id": 1,
                "goal_id": 3,
                "goal_ids": [3],
                "goal_ring_threshold_m": 0.25,
                "optimal_length_m": 2.0,
                "direct_pairs_ref": str(direct_pairs_path),
            },
        }
    ]
    preds = {"demo_q": {"output": [1, 2, 3]}}

    with pytest.raises(ValueError, match="missing acceptable_goal_ids"):
        mod.eval_routing(vqa, preds, embodied=False)


def test_eval_routing_requires_resolvable_direct_pairs_ref(tmp_path):
    scene_dir = tmp_path / "demo_scene"
    view_dir = scene_dir / "view_0"
    view_dir.mkdir(parents=True)
    visible_path = view_dir / "visible_waypoints_a2.json"
    _write_visible_waypoints(visible_path)

    vqa = [
        {
            "question_id": "demo_q",
            "task": "a2",
            "image_path": str(view_dir / "rgb_overlay_a2.png"),
            "visible_waypoints_path": str(visible_path),
            "ground_truth": {
                "start_id": 1,
                "goal_id": 3,
                "goal_ids": [3],
                "acceptable_goal_ids": [3],
                "optimal_length_m": 2.0,
                "direct_pairs_ref": "sidecars/a2/demo_scene/missing.json",
            },
            "__row_source_dir": str(tmp_path / "benchmark" / "vqa"),
        }
    ]
    preds = {"demo_q": {"output": [1, 2, 3]}}

    with pytest.raises(ValueError, match="cannot resolve direct_pairs_ref"):
        mod.eval_routing(vqa, preds, embodied=False)


def test_eval_routing_spl_uses_predicted_direct_segment_length(tmp_path):
    scene_dir = tmp_path / "demo_scene"
    view_dir = scene_dir / "view_0"
    view_dir.mkdir(parents=True)
    visible_path = view_dir / "visible_waypoints_a2.json"
    _write_visible_waypoints(visible_path)
    direct_pairs_path = tmp_path / "direct_pairs.json"
    _write_direct_pairs(
        direct_pairs_path,
        {
            "1": [{"display_id": 2, "distance_m": 1.0}],
            "2": [{"display_id": 1, "distance_m": 1.0}, {"display_id": 3, "distance_m": 3.0}],
            "3": [{"display_id": 2, "distance_m": 3.0}],
        },
    )

    vqa = [
        {
            "question_id": "demo_q",
            "task": "a2",
            "image_path": str(view_dir / "rgb_overlay_a2.png"),
            "visible_waypoints_path": str(visible_path),
            "ground_truth": {
                "start_id": 1,
                "goal_id": 3,
                "acceptable_goal_ids": [3],
                "optimal_length_m": 2.0,
                "direct_pairs_ref": str(direct_pairs_path),
            },
        }
    ]
    preds = {"demo_q": {"output": [1, 2, 3]}}

    metrics = mod.eval_routing(vqa, preds, embodied=False)

    assert metrics["success"] == 1
    assert metrics["spl"] == 0.5
    assert metrics["spl_all"] == 0.5
