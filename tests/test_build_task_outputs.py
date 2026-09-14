from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import sys
import pytest
import yaml
import numpy as np
from PIL import Image


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

MODULE_PATH = SCRIPTS_DIR / "build_task_outputs.py"
spec = spec_from_file_location("build_task_outputs", MODULE_PATH)
mod = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_build_next_task_outputs_requires_v3_bundle(tmp_path):
    scene_out = tmp_path / "demo_scene"
    scene_out.mkdir()
    (scene_out / "scene_nav_facts_v1.json").write_text(
        json.dumps(
            {
                "schema_version": "scene_nav_facts_v1",
                "scene_id": "demo_scene",
                "facts_hash": "hash-demo",
                "views": [],
            }
        )
    )

    cameras = [{"view_id": 0, "target_candidates": [], "classification_candidates": []}]

    try:
        mod.build_next_task_outputs(
            scene_id="demo_scene",
            cameras=cameras,
            scene_out=scene_out,
            observation_pack_present=False,
        )
    except FileNotFoundError as exc:
        assert "next_view_bundle_v3.json" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError when v3 bundle is missing")


def test_build_next_task_outputs_requires_scene_nav_facts(tmp_path):
    scene_out = tmp_path / "demo_scene"
    view_dir = scene_out / "view_0"
    view_dir.mkdir(parents=True)
    (view_dir / "next_view_bundle_v3.json").write_text(
        json.dumps({"schema_version": "next_view_bundle_v3", "classification_candidates": [], "routing_candidates": []})
    )

    with pytest.raises(FileNotFoundError, match="scene_nav_facts_v1.json"):
        mod.build_next_task_outputs(
            scene_id="demo_scene",
            cameras=[{"view_id": 0}],
            scene_out=scene_out,
            observation_pack_present=False,
        )


def test_build_next_task_outputs_materializes_v3_routing_and_classification(tmp_path):
    scene_out = tmp_path / "demo_scene"
    view_dir = scene_out / "view_0"
    view_dir.mkdir(parents=True)

    (scene_out / "scene_nav_facts_v1.json").write_text(
        json.dumps(
            {
                "schema_version": "scene_nav_facts_v1",
                "scene_id": "demo_scene",
                "facts_hash": "hash-demo",
                "views": [
                    {
                        "view_id": 0,
                        "camera": {
                            "position": [0.0, 0.0, 1.6],
                            "rotation": [0.0, 0.0, 0.0],
                            "fov": 120.0,
                            "resolution": [1024, 1024],
                        },
                        "visible_objects": [],
                        "affordance_validity": {
                            "eligible": True,
                            "failed_checks": [],
                            "check_results": {"has_candidates": True},
                        },
                        "affordance_axes": {
                            "clutter_axis": "medium",
                            "boundary_axis": "low",
                            "embodiment_gap_axis": "medium",
                        },
                        "affordance_tier": "hard",
                        "affordance_complexity": {
                            "visible_candidate_count": 18,
                            "point_embodied_disagreement_fraction": 0.22,
                        },
                        "classification_candidates": [],
                        "routing_candidates": [],
                        "rejection_reasons": [],
                    }
                ],
            }
        )
    )

    bundle = {
        "schema_version": "next_view_bundle_v3",
        "facts_schema": "scene_nav_facts_v1",
        "facts_hash": "hash-demo",
        "scene_id": "demo_scene",
        "view_id": 0,
        "classification_candidates": [
            {
                "waypoint_id": 101,
                "display_id": 1,
                "world_xyz": [1.0, 2.0, 0.0],
                "image_xy": [100, 200],
                "screen_xy_norm": [0.1, -0.2],
                "depth": 4.0,
                "pointmass_walkable": True,
                "embodied_feasible": False,
                "candidate_source": "dense_visible",
            },
            {
                "waypoint_id": 102,
                "display_id": 2,
                "world_xyz": [2.0, 3.0, 0.0],
                "image_xy": [300, 400],
                "screen_xy_norm": [0.3, -0.4],
                "depth": 5.0,
                "pointmass_walkable": False,
                "embodied_feasible": False,
                "candidate_source": "visible_object_surface",
            },
        ],
        "routing_candidates": [
            {
                "routing_id": "routing_v0_c001",
                "target_id": 7,
                "target_category": "dresser",
                "canonical_category": "cabinet",
                "semantic_group_id": "cabinet_like",
                "material_color_name": "brown",
                "distance_m": 3.2,
                "gt_path_display_ids_pointmass": [1, 4, 6],
                "gt_path_display_ids_embodied": [1, 5, 6],
                "gt_path_keypoints_world": [[0.0, 0.0, 0.0], [1.5, 0.2, 0.0], [3.0, 0.0, 0.0]],
                "debug_dense_path_world": [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
                "instruction_identity": {
                    "semantic_group_id": "cabinet_like",
                    "material_color_name": "brown",
                    "target_rule": "nearest",
                },
                "instruction_selection": {
                    "selection_status": "unique_winner",
                    "is_rule_winner": True,
                    "winner_target_ids": [7],
                    "cohort_size": 1,
                },
                "routing_complexity": {
                    "path_length_raw_m": 3.0,
                    "path_length_sparse_m": 3.014,
                    "start_goal_l2_m": 3.0,
                    "detour_ratio_raw": 1.0,
                    "detour_ratio_sparse": 1.005,
                    "dense_turn_count": 0,
                    "sparse_turn_count": 1,
                },
            }
        ],
    }
    (view_dir / "next_view_bundle_v3.json").write_text(json.dumps(bundle))

    task_outputs = mod.build_next_task_outputs(
        scene_id="demo_scene",
        cameras=[{"view_id": 0}],
        scene_out=scene_out,
        observation_pack_present=True,
        observation_pack_filename="next_observation_v1.json",
    )

    assert task_outputs["schema_version"] == "next_task_outputs_v2"
    assert task_outputs["facts_schema"] == "scene_nav_facts_v1"
    assert task_outputs["facts_hash"] == "hash-demo"
    assert task_outputs["task_families"]["routing"] == "materialized_v3"
    assert task_outputs["task_families"]["classification"] == "materialized_v3"
    assert task_outputs["observation_pack"]["present"] is True

    routing_view = task_outputs["routing"]["views"][0]
    candidate = routing_view["routing_candidates"][0]
    assert candidate["candidate_id"] == "routing_v0_c001"
    assert candidate["display_id"] == 1
    assert candidate["target_id"] == 7
    assert candidate["target_category"] == "dresser"
    assert candidate["canonical_category"] == "cabinet"
    assert candidate["semantic_group_id"] == "cabinet_like"
    assert candidate["material_color_name"] == "brown"
    assert candidate["gt_path_display_ids_pointmass"] == [1, 4, 6]
    assert candidate["gt_path_display_ids_embodied"] == [1, 5, 6]
    assert candidate["gt_path_keypoints_world"] == [[0.0, 0.0, 0.0], [1.5, 0.2, 0.0], [3.0, 0.0, 0.0]]
    assert candidate["debug_dense_path_world"] == [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
    assert candidate["instruction_identity"]["target_rule"] == "nearest"
    assert candidate["instruction_selection"]["selection_status"] == "unique_winner"
    assert candidate["routing_complexity"]["detour_ratio_raw"] == 1.0
    assert candidate["routing_complexity"]["sparse_turn_count"] == 1

    classification_view = task_outputs["classification"]["views"][0]
    assert classification_view["view_id"] == 0
    assert classification_view["candidate_count"] == 2
    assert classification_view["affordance_tier"] == "hard"
    assert classification_view["affordance_axes"]["clutter_axis"] == "medium"
    assert classification_view["affordance_validity"]["eligible"] is True
    assert [cand["display_id"] for cand in classification_view["shared_ab_candidates"]] == [1, 2]
    assert classification_view["shared_ab_candidates"][0]["candidate_id"] == "classification_v0_c001"
    assert classification_view["shared_ab_candidates"][1]["candidate_source"] == "visible_object_surface"


def test_build_next_task_outputs_rejects_legacy_bundle_version(tmp_path):
    scene_out = tmp_path / "demo_scene"
    view_dir = scene_out / "view_0"
    view_dir.mkdir(parents=True)
    (scene_out / "scene_nav_facts_v1.json").write_text(
        json.dumps(
            {
                "schema_version": "scene_nav_facts_v1",
                "scene_id": "demo_scene",
                "facts_hash": "hash-demo",
                "views": [],
            }
        )
    )
    (view_dir / "next_view_bundle_v3.json").write_text(json.dumps({"schema_version": "next_view_bundle_v2"}))

    try:
        mod.build_next_task_outputs(
            scene_id="demo_scene",
            cameras=[{"view_id": 0}],
            scene_out=scene_out,
            observation_pack_present=False,
        )
    except ValueError as exc:
        assert "next_view_bundle_v3" in str(exc)
    else:
        raise AssertionError("expected ValueError for legacy bundle schema")


def test_validate_render_scene_inputs_requires_rgb_and_v3_bundle(tmp_path):
    render_scene_dir = tmp_path / "demo_scene"
    view_dir = render_scene_dir / "view_5"
    view_dir.mkdir(parents=True)

    cameras = [{"view_id": 5}]

    try:
        mod.validate_render_scene_inputs(render_scene_dir, cameras)
    except FileNotFoundError as exc:
        assert "rgb.png" in str(exc)
        assert "view_5" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError when rgb is missing")

    (view_dir / "rgb.png").write_bytes(b"fake")

    try:
        mod.validate_render_scene_inputs(render_scene_dir, cameras)
    except FileNotFoundError as exc:
        assert "next_view_bundle_v3.json" in str(exc)
        assert "view_5" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError when v3 bundle is missing")


def test_resolve_sampling_params_prefers_cli_over_config():
    config = {
        "task_outputs": {
            "dense_spacing_m": 0.5,
            "sparse_count": 60,
            "min_pixel_dist": 18,
            "max_visible": 240,
        }
    }

    params = mod.resolve_sampling_params(
        config,
        dense_spacing=0.3,
        sparse_count=72,
        min_pixel_dist=16,
        max_visible=260,
    )

    assert params == {
        "dense_spacing": 0.3,
        "sparse_count": 72,
        "min_pixel_dist": 16,
        "max_visible": 260,
    }


def test_resolve_sampling_params_falls_back_to_config_then_defaults():
    config = {
        "task_outputs": {
            "sparse_count": 60,
            "min_pixel_dist": 18,
        }
    }

    params = mod.resolve_sampling_params(
        config,
        dense_spacing=None,
        sparse_count=None,
        min_pixel_dist=None,
        max_visible=None,
    )

    assert params == {
        "dense_spacing": 0.4,
        "sparse_count": 60,
        "min_pixel_dist": 18,
        "max_visible": 200,
    }


def test_load_object_index_image_decodes_normalized_png_pass_ids(tmp_path):
    object_index_path = tmp_path / "object_index.png"
    arr = np.array([[0, 32768, 65535]], dtype=np.uint16)
    Image.fromarray(arr, mode="I;16").save(object_index_path)

    decoded = mod.load_object_index_image(object_index_path, max_pass_id=32)

    assert decoded.dtype == np.int32
    assert decoded.tolist() == [[0, 16, 32]]


def test_task_output_configs_keep_visible_waypoints_dense_enough():
    repo_root = Path(__file__).resolve().parents[1]
    config_paths = [
        repo_root / "configs" / "default.yaml",
        repo_root / "configs" / "internscene_local.yaml",
    ]

    for config_path in config_paths:
        config = yaml.safe_load(config_path.read_text())
        task_cfg = config["task_outputs"]
        assert float(task_cfg["dense_spacing_m"]) <= 0.35
        assert int(task_cfg["sparse_count"]) >= 80
        assert int(task_cfg["min_pixel_dist"]) <= 16
        assert int(task_cfg["max_visible"]) >= 260
        assert int(config["targets"]["max_targets_per_view"]) >= 10


def test_build_routing_gt_keypoint_candidates_adds_missing_sparse_goal_point():
    class GridStub:
        resolution = 0.1
        width = 256
        height = 256

        def world_to_grid(self, x, y):
            return int(round(x * 10)) + 128, int(round(y * 10)) + 128

        def is_free(self, gx, gy):
            return True

    existing = [
        {
            "waypoint_id": 24,
            "world_xyz": [-1.665, -0.61, 0.0],
            "grid_xy": [111, 122],
            "image_xy": [1019, 784],
            "screen_xy_norm": [0.99, -0.53],
            "depth": 1.36,
            "pointmass_walkable": True,
            "embodied_feasible": True,
            "candidate_source": "blender_grid",
        }
    ]
    routing_candidates = [
        {
            "routing_id": "routing_v6_c003",
            "gt_path_keypoints_world": [
                [-1.565097255531172, -0.6095283222395462, 0.0],
                [0.03490274446882857, -0.8095283222395455, 0.0],
            ],
            "gt_path_keypoints_image_xy": [
                [975, 770],
                [547, 605],
            ],
        }
    ]

    injected, next_waypoint_id = mod.build_routing_gt_keypoint_candidates(
        existing_points=existing,
        routing_candidates=routing_candidates,
        source_candidates=existing,
        grid_point=GridStub(),
        grid_embodied=GridStub(),
        cam_pos=np.array([0.0, 0.0, 1.6], dtype=np.float32),
        cam_rot=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        fov=120.0,
        res_x=1024,
        res_y=1024,
        depth_map=np.full((1024, 1024), 20.0, dtype=np.float32),
        object_index_img=None,
        object_index_pass_ids=set(),
        occlusion_eps=0.15,
        occlusion_window=2,
        image_clearance_m=0.15,
        image_clearance_max_px=18,
        next_waypoint_id=1000,
    )

    assert len(injected) == 1
    assert injected[0]["waypoint_id"] == 1000
    assert injected[0]["image_xy"] == [547, 605]
    assert injected[0]["candidate_source"] == "routing_gt_keypoint"
    assert injected[0]["pointmass_walkable"] is True
    assert injected[0]["embodied_feasible"] is True
    assert next_waypoint_id == 1001


def test_build_routing_gt_keypoint_candidates_reuses_bundle_waypoint_metadata_without_rechecking_occlusion():
    class GridStub:
        resolution = 0.1
        width = 256
        height = 256

        def world_to_grid(self, x, y):
            return int(round(x * 10)) + 128, int(round(y * 10)) + 128

        def is_free(self, gx, gy):
            return True

    existing = [
        {
            "waypoint_id": 24,
            "world_xyz": [0.0, 0.0, 0.0],
            "grid_xy": [128, 128],
            "image_xy": [10, 40],
            "screen_xy_norm": [0.0, 0.0],
            "depth": 1.6,
            "pointmass_walkable": True,
            "embodied_feasible": True,
            "candidate_source": "blender_grid",
        }
    ]
    routing_candidates = [
        {
            "routing_id": "routing_v0_c001",
            "gt_path_waypoint_ids_pointmass": [24, 36],
            "gt_path_waypoint_ids_embodied": [24, 36],
            "gt_path_keypoints_world": [
                [0.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            "gt_path_keypoints_image_xy": [
                [10, 40],
                [50, 10],
            ],
        }
    ]
    object_index = np.zeros((64, 64), dtype=np.int32)
    object_index[10, 50] = 39
    source_candidates = existing + [
        {
            "waypoint_id": 36,
            "world_xyz": [2.0, 0.0, 0.0],
            "grid_xy": [148, 128],
            "image_xy": [50, 10],
            "screen_xy_norm": [0.56, 0.69],
            "depth": 3.2,
            "pointmass_walkable": True,
            "embodied_feasible": True,
            "candidate_source": "path_world_forced",
        }
    ]

    injected, next_waypoint_id = mod.build_routing_gt_keypoint_candidates(
        existing_points=existing,
        routing_candidates=routing_candidates,
        source_candidates=source_candidates,
        grid_point=GridStub(),
        grid_embodied=GridStub(),
        cam_pos=np.array([0.0, 0.0, 1.6], dtype=np.float32),
        cam_rot=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        fov=120.0,
        res_x=64,
        res_y=64,
        depth_map=np.full((64, 64), 20.0, dtype=np.float32),
        object_index_img=object_index,
        object_index_pass_ids={39},
        occlusion_eps=0.15,
        occlusion_window=2,
        image_clearance_m=0.15,
        image_clearance_max_px=8,
        next_waypoint_id=1000,
    )

    assert len(injected) == 1
    assert injected[0]["waypoint_id"] == 1000
    assert injected[0]["image_xy"] == [50, 10]
    assert injected[0]["screen_xy_norm"] == [0.56, 0.69]
    assert injected[0]["depth"] == 3.2
    assert injected[0]["candidate_source"] == "routing_gt_keypoint"
    assert next_waypoint_id == 1001


def test_build_routing_gt_keypoint_candidates_requires_blender_sparse_keypoint_image_xy():
    class GridStub:
        resolution = 0.1
        width = 256
        height = 256

        def world_to_grid(self, x, y):
            return int(round(x * 10)) + 128, int(round(y * 10)) + 128

        def is_free(self, gx, gy):
            return True

    with pytest.raises(ValueError, match="missing Blender gt_path_keypoints_image_xy"):
        mod.build_routing_gt_keypoint_candidates(
            existing_points=[],
            routing_candidates=[
                {
                    "routing_id": "routing_v1_c001",
                    "gt_path_keypoints_world": [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                    ],
                    "gt_path_keypoints_image_xy": [
                        [12, 20],
                    ],
                }
            ],
            source_candidates=[],
            grid_point=GridStub(),
            grid_embodied=GridStub(),
            cam_pos=np.array([0.0, 0.0, 1.6], dtype=np.float32),
            cam_rot=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            fov=120.0,
            res_x=64,
            res_y=64,
            depth_map=np.full((64, 64), 20.0, dtype=np.float32),
            object_index_img=None,
            object_index_pass_ids=set(),
            occlusion_eps=0.15,
            occlusion_window=2,
            image_clearance_m=0.15,
            image_clearance_max_px=8,
            next_waypoint_id=1000,
        )
