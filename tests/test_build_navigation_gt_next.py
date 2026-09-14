from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import json
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "build_navigation_gt_next.py"
spec = spec_from_file_location("build_navigation_gt_next", MODULE_PATH)
mod = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_select_goal_candidates_for_target_prefers_path_endpoint():
    visible = [
        {"waypoint_id": 10, "display_id": 1, "world_xyz": [0.1, 0.1, 0.0]},
        {"waypoint_id": 20, "display_id": 2, "world_xyz": [4.9, 5.1, 0.0]},
        {"waypoint_id": 21, "display_id": 3, "world_xyz": [5.2, 5.0, 0.0]},
    ]
    bbox = [0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    path_world = [[0.0, 0.0, 0.0], [5.0, 5.0, 0.0]]

    goal_wp_ids, goal_disp_ids = mod.select_goal_candidates_for_target(
        visible,
        bbox=bbox,
        path_world=path_world,
        radius_m=0.6,
        max_count=None,
    )

    assert goal_wp_ids == [20, 21]
    assert goal_disp_ids == [2, 3]


def test_load_routing_candidates_for_view_requires_v3_bundle(tmp_path):
    render_scene_dir = tmp_path / "renders" / "demo_scene"
    render_scene_dir.mkdir(parents=True)

    try:
        mod.load_routing_candidates_for_view(
            render_scene_dir=render_scene_dir,
            task_outputs_path=tmp_path / "next_task_outputs_v1.json",
            cameras=[{"view_id": 2, "target_candidates": []}],
            view_id=2,
        )
    except FileNotFoundError as exc:
        assert "next_view_bundle_v3.json" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError when v3 bundle is missing")


def test_load_routing_candidates_for_view_reads_v3_bundle_without_fallback(tmp_path):
    render_scene_dir = tmp_path / "renders" / "demo_scene"
    view_dir = render_scene_dir / "view_2"
    view_dir.mkdir(parents=True)
    (view_dir / "next_view_bundle_v3.json").write_text(
        json.dumps(
            {
                "schema_version": "next_view_bundle_v3",
                "routing_candidates": [
                    {
                        "routing_id": "routing_v2_c001",
                        "target_id": 9,
                        "target_category": "lamp",
                        "canonical_category": "lamp",
                        "semantic_group_id": "lamp",
                        "material_color_name": "white",
                        "distance_m": 4.5,
                        "gt_path_display_ids_pointmass": [1, 2, 3],
                        "gt_path_display_ids_embodied": [1, 2, 3],
                        "gt_path_keypoints_world": [[0.0, 0.0, 0.0], [1.0, 2.0, 0.0]],
                        "debug_dense_path_world": [[0.0, 0.0, 0.0], [1.0, 2.0, 0.0]],
                    }
                ],
            }
        )
    )

    loaded = mod.load_routing_candidates_for_view(
        render_scene_dir=render_scene_dir,
        task_outputs_path=tmp_path / "next_task_outputs_v1.json",
        cameras=[
            {
                "view_id": 2,
                "target_candidates": [
                    {"target_id": 1, "category": "chair", "distance": 3.0, "path_world": [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]}
                ],
            }
        ],
        view_id=2,
    )

    assert loaded == [
        {
            "routing_id": "routing_v2_c001",
            "target_id": 9,
            "target_category": "lamp",
            "canonical_category": "lamp",
            "semantic_group_id": "lamp",
            "material_color_name": "white",
            "distance_m": 4.5,
            "gt_path_display_ids_pointmass": [1, 2, 3],
            "gt_path_display_ids_embodied": [1, 2, 3],
            "gt_path_keypoints_world": [[0.0, 0.0, 0.0], [1.0, 2.0, 0.0]],
            "debug_dense_path_world": [[0.0, 0.0, 0.0], [1.0, 2.0, 0.0]],
        }
    ]


def test_load_routing_candidates_for_view_rejects_legacy_bundle_schema(tmp_path):
    render_scene_dir = tmp_path / "renders" / "demo_scene"
    view_dir = render_scene_dir / "view_2"
    view_dir.mkdir(parents=True)
    (view_dir / "next_view_bundle_v3.json").write_text(json.dumps({"schema_version": "next_view_bundle_v2"}))

    try:
        mod.load_routing_candidates_for_view(
            render_scene_dir=render_scene_dir,
            task_outputs_path=tmp_path / "next_task_outputs_v1.json",
            cameras=[{"view_id": 2, "target_candidates": []}],
            view_id=2,
        )
    except ValueError as exc:
        assert "next_view_bundle_v3" in str(exc)
    else:
        raise AssertionError("expected ValueError for legacy bundle schema")


def test_targets_match_when_semantic_group_matches():
    assert mod.targets_match_semantic_goal(
        predicted={"semantic_group_id": "cabinet_like", "target_id": 100, "target_color_name": "brown"},
        accepted={"semantic_group_id": "cabinet_like", "target_id": 200, "target_color_name": "brown"},
    )
    assert not mod.targets_match_semantic_goal(
        predicted={"semantic_group_id": "chair", "target_id": 100, "target_color_name": "brown"},
        accepted={"semantic_group_id": "cabinet_like", "target_id": 200, "target_color_name": "brown"},
    )
    assert not mod.targets_match_semantic_goal(
        predicted={"semantic_group_id": "cabinet_like", "target_id": 100, "target_color_name": "white"},
        accepted={"semantic_group_id": "cabinet_like", "target_id": 200, "target_color_name": "brown"},
    )


def test_map_render_path_to_visible_ids_uses_world_path_not_foreign_waypoint_namespace():
    visible = [
        {"waypoint_id": 6, "display_id": 1, "world_xyz": [-0.312, -0.278, 0.0]},
        {"waypoint_id": 11, "display_id": 2, "world_xyz": [-0.312, 0.122, 0.0]},
        {"waypoint_id": 5, "display_id": 3, "world_xyz": [-0.712, -0.278, 0.0]},
    ]
    path_world = [[-0.512, -0.078, 0.0], [-1.012, -0.078, 0.0]]

    path_wp_ids, path_disp_ids = mod.map_render_path_to_visible_ids(
        visible,
        path_world,
        sample_step_m=0.1,
        max_match_dist_m=0.5,
    )

    assert path_wp_ids == [6, 5]
    assert path_disp_ids == [1, 3]


def test_resolve_visible_path_ids_prefers_world_keypoints_when_bundle_waypoint_ids_are_foreign():
    visible = [
        {"waypoint_id": 6, "display_id": 1, "world_xyz": [-0.312, -0.278, 0.0]},
        {"waypoint_id": 11, "display_id": 2, "world_xyz": [-0.312, 0.122, 0.0]},
        {"waypoint_id": 5, "display_id": 3, "world_xyz": [-0.712, -0.278, 0.0]},
    ]
    candidate = {
        "gt_path_display_ids_pointmass": [],
        "gt_path_display_ids_embodied": [],
        "path_wp_ids_pointmass": [10, 9, 10, 9],
        "path_wp_ids_embodied": [10, 4, 10, 4],
        "gt_path_keypoints_world": [[-0.512, -0.078, 0.0], [-1.012, -0.078, 0.0]],
    }

    pointmass_wp_ids, pointmass_disp_ids = mod.resolve_visible_path_ids(
        candidate,
        visible,
        path_key="pointmass",
    )
    embodied_wp_ids, embodied_disp_ids = mod.resolve_visible_path_ids(
        candidate,
        visible,
        path_key="embodied",
    )

    assert pointmass_wp_ids == [6, 5]
    assert pointmass_disp_ids == [1, 3]
    assert embodied_wp_ids == [6, 5]
    assert embodied_disp_ids == [1, 3]


def test_resolve_visible_path_ids_uses_canonical_gt_waypoint_ids_before_world_remap():
    visible = [
        {"waypoint_id": 24, "display_id": 7, "world_xyz": [0.0, -1.0, 0.0]},
        {"waypoint_id": 36, "display_id": 8, "world_xyz": [1.0, -0.3, 0.0]},
    ]
    candidate = {
        "gt_path_display_ids_pointmass": [],
        "gt_path_waypoint_ids_pointmass": [24, 36],
        "gt_path_keypoints_world": [],
    }

    wp_ids, disp_ids = mod.resolve_visible_path_ids(
        candidate,
        visible,
        path_key="pointmass",
    )

    assert wp_ids == [24, 36]
    assert disp_ids == [7, 8]


def test_resolve_visible_path_ids_rejects_legacy_waypoint_fallback_without_v3_projection():
    visible = [
        {"waypoint_id": 6, "display_id": 1, "world_xyz": [-0.312, -0.278, 0.0]},
        {"waypoint_id": 11, "display_id": 2, "world_xyz": [-0.312, 0.122, 0.0]},
    ]
    candidate = {
        "target_id": 123,
        "gt_path_display_ids_pointmass": [],
        "gt_path_keypoints_world": [],
        "path_wp_ids_pointmass": [6, 11],
    }

    with pytest.raises(ValueError, match="legacy waypoint fallback"):
        mod.resolve_visible_path_ids(
            candidate,
            visible,
            path_key="pointmass",
        )


def test_select_targets_for_instruction_rule_uses_synonym_group_color_and_nearest():
    cands = [
        {
            "target_id": 10,
            "category": "dresser",
            "canonical_category": "cabinet",
            "semantic_group_id": "cabinet_like",
            "target_color_name": "brown",
            "distance": 1.2,
        },
        {
            "target_id": 11,
            "category": "cabinet",
            "canonical_category": "cabinet",
            "semantic_group_id": "cabinet_like",
            "target_color_name": "brown",
            "distance": 2.4,
        },
        {
            "target_id": 12,
            "category": "cabinet",
            "canonical_category": "cabinet",
            "semantic_group_id": "cabinet_like",
            "target_color_name": "white",
            "distance": 0.8,
        },
    ]

    winners = mod.select_targets_for_instruction_rule(
        cands,
        accepted=cands[0],
        rule="nearest",
    )

    assert [c["target_id"] for c in winners] == [10]


def test_select_targets_for_instruction_rule_rejects_distance_ties_after_color_filter():
    cands = [
        {
            "target_id": 10,
            "category": "dresser",
            "canonical_category": "cabinet",
            "semantic_group_id": "cabinet_like",
            "target_color_name": "brown",
            "distance": 1.2,
        },
        {
            "target_id": 11,
            "category": "cabinet",
            "canonical_category": "cabinet",
            "semantic_group_id": "cabinet_like",
            "target_color_name": "brown",
            "distance": 1.2004,
        },
    ]

    with pytest.raises(ValueError, match="ambiguous target selection"):
        mod.select_targets_for_instruction_rule(
            cands,
            accepted=cands[0],
            rule="nearest",
            distance_tie_eps_m=0.001,
        )


def test_select_instruction_targets_for_view_materializes_multiple_unique_identities():
    cands = [
        {
            "routing_id": "routing_v2_c001",
            "target_id": 10,
            "category": "dresser",
            "canonical_category": "cabinet",
            "semantic_group_id": "cabinet_like",
            "target_color_name": "brown",
            "distance": 1.2,
        },
        {
            "routing_id": "routing_v2_c002",
            "target_id": 11,
            "category": "cabinet",
            "canonical_category": "cabinet",
            "semantic_group_id": "cabinet_like",
            "target_color_name": "brown",
            "distance": 2.4,
        },
        {
            "routing_id": "routing_v2_c003",
            "target_id": 12,
            "category": "cabinet",
            "canonical_category": "cabinet",
            "semantic_group_id": "cabinet_like",
            "target_color_name": "white",
            "distance": 0.8,
        },
    ]

    selected = mod.select_instruction_targets_for_view(cands, rule="nearest")

    assert [c["target_id"] for c in selected] == [12, 10]
    assert [c["routing_id"] for c in selected] == ["routing_v2_c003", "routing_v2_c001"]


def test_lookup_target_screen_x_uses_camera_visible_objects():
    camera = {
        "visible_objects": [
            {"id": 3, "screen_x": 0.42},
            {"id": 7, "screen_x": -0.33},
        ]
    }

    assert mod.lookup_target_screen_x(camera, 7) == -0.33
    assert mod.lookup_target_screen_x(camera, 99) is None


def test_main_materializes_family_specific_tiers_from_fact_layer(tmp_path):
    scenes_dir = tmp_path / "scenes"
    renders_dir = tmp_path / "renders"
    out_dir = tmp_path / "gt"
    config_path = tmp_path / "config.yaml"
    scene_id = "demo_scene"
    scene_dir = scenes_dir / scene_id
    render_dir = renders_dir / scene_id
    view_dir = render_dir / "view_0"
    scene_dir.mkdir(parents=True)
    view_dir.mkdir(parents=True)
    out_dir.mkdir()

    config_path.write_text(
        "render:\n"
        "  resolution: [1024, 1024]\n"
        "  fov: 120\n"
        "task_outputs:\n"
        "  prefer_materialized: true\n"
    )

    (scene_dir / "layout.json").write_text(
        json.dumps(
            [
                {
                    "id": 7,
                    "bbox": [0.0, 3.0, 0.5, 0.8, 0.8, 1.0, 0.0, 0.0, 0.0],
                    "category": "chair",
                }
            ]
        )
    )
    (render_dir / "cameras.json").write_text(
        json.dumps(
            [
                {
                    "view_id": 0,
                    "position": [0.0, 0.0, 1.6],
                    "rotation": [0.0, 0.0, 0.0],
                    "visible_objects": [{"id": 7, "screen_x": 0.1}],
                }
            ]
        )
    )
    for name, visible_waypoints in {
        "a1": [
            {"display_id": 1, "pointmass_walkable": True},
            {"display_id": 2, "pointmass_walkable": False},
        ],
        "b1": [
            {"display_id": 1, "embodied_feasible": True},
            {"display_id": 2, "embodied_feasible": False},
        ],
        "a2": [
            {"waypoint_id": 11, "display_id": 1, "world_xyz": [0.0, 0.0, 0.0]},
            {"waypoint_id": 12, "display_id": 2, "world_xyz": [1.0, 0.0, 0.0]},
        ],
        "b2": [
            {"waypoint_id": 11, "display_id": 1, "world_xyz": [0.0, 0.0, 0.0]},
            {"waypoint_id": 12, "display_id": 2, "world_xyz": [1.0, 0.0, 0.0]},
        ],
    }.items():
        (view_dir / f"visible_waypoints_{name}.json").write_text(
            json.dumps({"view_id": 0, "visible_waypoints": visible_waypoints})
        )

    (view_dir / "next_view_bundle_v3.json").write_text(
        json.dumps(
            {
                "schema_version": "next_view_bundle_v3",
                "facts_hash": "hash-demo",
                "visible_objects": [{"id": 7, "screen_x": 0.1}],
                "affordance_validity": {
                    "eligible": True,
                    "failed_checks": [],
                    "check_results": {"has_candidates": True},
                },
                "affordance_axes": {
                    "clutter_axis": "medium",
                    "boundary_axis": "low",
                    "embodiment_gap_axis": "low",
                },
                "affordance_tier": "medium",
                "classification_candidates": [],
                "routing_candidates": [
                    {
                        "routing_id": "routing_v0_c001",
                        "target_id": 7,
                        "target_category": "chair",
                        "canonical_category": "chair",
                        "semantic_group_id": "chair",
                        "material_color_name": "gray",
                        "distance_m": 3.0,
                        "gt_path_waypoint_ids_pointmass": [11, 12],
                        "gt_path_waypoint_ids_embodied": [11, 12],
                        "gt_path_keypoints_world": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                        "debug_dense_path_world": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                        "instruction_identity": {
                            "semantic_group_id": "chair",
                            "material_color_name": "gray",
                            "target_rule": "nearest",
                        },
                        "instruction_selection": {
                            "selection_status": "unique_winner",
                            "is_rule_winner": True,
                            "winner_target_ids": [7],
                            "cohort_size": 1,
                        },
                        "routing_complexity": {
                            "path_length_raw_m": 1.0,
                            "path_length_sparse_m": 1.0,
                            "start_goal_l2_m": 1.0,
                            "detour_ratio_raw": 1.0,
                            "detour_ratio_sparse": 1.0,
                            "dense_turn_count": 0,
                            "sparse_turn_count": 0,
                            "narrow_passage_fraction": 0.0,
                            "instruction_cohort_size": 1,
                        },
                        "navigation_validity": {
                            "eligible": True,
                            "failed_checks": [],
                            "check_results": {"path_projection_safe": True},
                        },
                        "navigation_axes": {
                            "reference_axis": "low",
                            "geometry_axis": "medium",
                            "embodiment_axis": "low",
                        },
                        "navigation_tier": "medium",
                        "invariants": {
                            "start_in_forward_sector": True,
                            "goal_in_target_zone": True,
                            "embodied_collision_free": True,
                        },
                    }
                ],
            }
        )
    )

    import sys

    argv = sys.argv
    try:
        sys.argv = [
            "build_navigation_gt_next.py",
            "--config",
            str(config_path),
            "--scenes-dir",
            str(scenes_dir),
            "--renders-dir",
            str(renders_dir),
            "--output-dir",
            str(out_dir),
        ]
        mod.main()
    finally:
        sys.argv = argv

    rec_a1 = json.loads((out_dir / "gt_next_a1.jsonl").read_text().strip())
    rec_a2 = json.loads((out_dir / "gt_next_a2.jsonl").read_text().strip())
    rec_c = json.loads((out_dir / "gt_next_c.jsonl").read_text().strip())

    assert rec_a1["tier_family"] == "affordance"
    assert rec_a1["tier"] is not None
    assert rec_a2["tier_family"] == "navigation"
    assert rec_a2["tier"] == "medium"
    assert rec_a2["facts_hash"] == "hash-demo"
    assert rec_c["tier_family"] == "navigation"
    assert rec_c["tier"] == "medium"
