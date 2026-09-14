from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "render_scenes.py"
spec = spec_from_file_location("render_scenes", MODULE_PATH)
mod = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_canonicalize_category_uses_alias_group():
    alias_map = {
        "dresser": {
            "canonical_category": "cabinet",
            "semantic_group_id": "cabinet_like",
        }
    }

    result = mod.canonicalize_category("dresser", alias_map)

    assert result["raw_category"] == "dresser"
    assert result["canonical_category"] == "cabinet"
    assert result["semantic_group_id"] == "cabinet_like"


def test_quantize_material_color_name_returns_stable_bucket():
    assert mod.quantize_material_color_name((0.42, 0.28, 0.16)) == "brown"
    assert mod.quantize_material_color_name((0.82, 0.82, 0.79)) == "gray"


def test_path_keypoint_sparsification_keeps_turns_and_endpoints():
    path = [
        [0.0, 0.0, 0.0],
        [0.5, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.5, 0.0],
        [1.0, 1.0, 0.0],
    ]

    sparse = mod.sparsify_gt_path_keypoints(path, turn_angle_deg=20.0, min_segment_m=0.6)

    assert sparse == [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
    ]


def test_small_target_visibility_uses_relaxed_surface_ratio_gate():
    assert mod.target_surface_visibility_ok(
        is_large=False,
        vis_ratio=0.2,
        center_visible=True,
        min_visible_ratio=0.15,
        large_visibility_ratio=0.7,
    )


def test_large_target_visibility_keeps_stricter_surface_ratio_gate():
    assert not mod.target_surface_visibility_ok(
        is_large=True,
        vis_ratio=0.2,
        center_visible=True,
        min_visible_ratio=0.15,
        large_visibility_ratio=0.7,
    )


def test_view_candidate_eligibility_no_longer_depends_on_path_score():
    assert mod.view_candidate_meets_visibility_floor(
        {"n_vis": 4, "path_score": -1},
        min_visible_objects=3,
    )


def test_view_candidate_eligibility_still_rejects_too_few_visible_objects():
    assert not mod.view_candidate_meets_visibility_floor(
        {"n_vis": 2, "path_score": 99},
        min_visible_objects=3,
    )


def test_build_scored_camera_candidates_keeps_every_candidate_record():
    candidate_records = [
        {
            "pos": (0.0, 0.0),
            "angle": 0.0,
            "n_vis": 4,
            "ids": {1, 2, 3, 4},
            "objs": [{"id": 1}],
            "bdist": 0.2,
        },
        {
            "pos": (1.0, 1.0),
            "angle": 1.57,
            "n_vis": 1,
            "ids": {5},
            "objs": [{"id": 5}],
            "bdist": 1.1,
        },
    ]

    def clearance_provider(cx, cy, angle):
        return {
            "fwd_clearance_ray": round(1.0 + cx, 3),
            "fwd_clearance_obj": round(2.0 + cy, 3),
        }

    scored = mod.build_scored_camera_candidates(
        candidate_records,
        clearance_provider=clearance_provider,
    )

    assert len(scored) == 2
    assert scored[0]["pos"] == (0.0, 0.0)
    assert scored[1]["pos"] == (1.0, 1.0)
    assert scored[0]["fwd_clearance_ray"] == 1.0
    assert scored[1]["fwd_clearance_obj"] == 3.0


def test_build_fixed_bottom_center_start_candidate_keeps_original_anchor_world(monkeypatch):
    class _FakeMatrix:
        def __init__(self):
            self.translation = SimpleNamespace(copy=lambda: SimpleNamespace())

        def inverted(self):
            return self

    class _FakeGrid:
        width = 100
        height = 100

        def world_to_grid(self, _x, _y):
            return 2, 3

        def is_free(self, _gx, _gy):
            return True

        def grid_to_world(self, _gx, _gy):
            return 10.0, 20.0

    def _fake_vector(xyz):
        return SimpleNamespace(x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))

    def _fake_project(_cam_inv, world_pt, *_args, **_kwargs):
        if abs(float(world_pt.x) - 1.25) < 1e-6 and abs(float(world_pt.y) - 2.75) < 1e-6:
            return {
                "pixel_xy": (512, 900),
                "screen_xy": (0.0, -0.8),
                "depth": 2.0,
            }
        return None

    def _fake_raycast(_scene, _depsgraph, _origin, world_pt, tol=0.08):
        assert tol == 0.08
        return abs(float(world_pt.x) - 1.25) < 1e-6 and abs(float(world_pt.y) - 2.75) < 1e-6

    monkeypatch.setattr(mod, "mathutils", SimpleNamespace(Vector=_fake_vector), raising=False)
    monkeypatch.setattr(mod, "project_world_point", _fake_project)
    monkeypatch.setattr(mod, "raycast_visible", _fake_raycast)
    monkeypatch.setattr(mod, "sample_ground_z", lambda *_args, **_kwargs: 0.02)

    candidate = mod.build_fixed_bottom_center_start_candidate(
        anchor_world_xyz=[1.25, 2.75, 0.0],
        cam_obj=SimpleNamespace(matrix_world=_FakeMatrix()),
        depsgraph=object(),
        scene=object(),
        grid_point=_FakeGrid(),
        grid_embodied=_FakeGrid(),
        vfov_half=1.0,
        hfov_half=1.0,
        res_x=1024,
        res_y=1024,
    )

    assert candidate is not None
    assert candidate["world_xyz"] == [1.25, 2.75, 0.02]
    assert candidate["grid_xy"] == [2, 3]


def test_build_target_candidates_from_visible_objects_keeps_verified_visible_target():
    layout = [
        {
            "id": 7,
            "bbox": [0.0, 4.0, 0.5, 0.8, 0.8, 1.0, 0.0, 0.0, 0.0],
            "category": "chair",
            "model_uid": "chair_a",
            "material_color_rgb": [0.3, 0.2, 0.1],
            "material_color_name": "brown",
        }
    ]

    targets, rejections, dbg_counts = mod.build_target_candidates_from_visible_objects(
        layout=layout,
        visible_objects=[{"id": 7, "raycast_verified": True}],
        cam_position=[0.0, 0.0, 1.25],
        cam_obj=object(),
        depsgraph=object(),
        fov_deg=120.0,
        res_x=1024,
        res_y=1024,
        min_dist=2.0,
        max_targets=10,
        prefilter_targets=10,
        min_area_px=900.0,
        min_short_side_px=18.0,
        rec_categories=set(),
        category_aliases={},
        large_categories=set(),
        large_size=1.0,
        bbox_metrics_provider=lambda *_args, **_kwargs: {
            "area_px": 2400,
            "short_side_px": 48,
        },
        visibility_ratio_provider=lambda *_args, **_kwargs: 0.0,
    )

    assert [target["target_id"] for target in targets] == [7]
    assert targets[0]["visibility_ratio"] == 0.0
    assert not rejections
    assert dbg_counts["vis"] == 1


def test_goal_routing_candidates_use_nearest_embodied_ring_not_fixed_radius():
    bbox = [0.0, 0.0, 0.0, 0.6, 0.6, 1.0, 0.0, 0.0, 0.0]
    candidates = [
        {
            "waypoint_id": 1,
            "world_xyz": [0.35, 0.0, 0.0],
            "embodied_feasible": True,
        },
        {
            "waypoint_id": 2,
            "world_xyz": [0.4, 0.0, 0.0],
            "embodied_feasible": True,
        },
        {
            "waypoint_id": 3,
            "world_xyz": [0.8, 0.0, 0.0],
            "embodied_feasible": True,
        },
    ]

    goals = mod.select_goal_routing_candidates(candidates, bbox, ring_tolerance_m=0.06)

    assert [g["waypoint_id"] for g in goals] == [1, 2]
    assert goals[0]["goal_distance_m"] == 0.05
    assert goals[1]["goal_distance_m"] == 0.1
    assert all(g["goal_ring_threshold_m"] == 0.11 for g in goals)


def test_select_fixed_start_candidate_prefers_bottom_center_world_anchor():
    candidates = [
        {
            "waypoint_id": 1,
            "world_xyz": [0.0, -0.55, 0.0],
            "screen_xy_norm": [0.0, -0.55],
            "depth": 1.8,
            "embodied_feasible": True,
        },
        {
            "waypoint_id": 2,
            "world_xyz": [0.08, -0.96, 0.0],
            "screen_xy_norm": [0.08, -0.96],
            "depth": 2.0,
            "embodied_feasible": True,
        },
        {
            "waypoint_id": 3,
            "world_xyz": [-0.02, -0.91, 0.0],
            "screen_xy_norm": [-0.02, -0.91],
            "depth": 1.2,
            "embodied_feasible": True,
        },
    ]

    selected = mod.select_fixed_start_candidate(
        candidates,
        anchor_world_xyz=[0.0, -0.9, 0.0],
        max_anchor_snap_dist_m=0.2,
    )

    assert selected is not None
    assert selected["waypoint_id"] == 3


def test_select_fixed_start_candidate_rejects_anchor_when_no_close_visible_embodied_point():
    candidates = [
        {
            "waypoint_id": 1,
            "world_xyz": [0.9, -0.9, 0.0],
            "screen_xy_norm": [0.1, -0.95],
            "depth": 1.0,
            "embodied_feasible": True,
        }
    ]

    selected = mod.select_fixed_start_candidate(
        candidates,
        anchor_world_xyz=[0.0, -0.9, 0.0],
        max_anchor_snap_dist_m=0.2,
    )

    assert selected is None


def test_project_world_point_keeps_borderline_bottom_edge_projection(monkeypatch):
    class _FakeMatrix:
        def inverted(self):
            return self

        def __matmul__(self, other):
            return SimpleNamespace(z=-1.0)

    cam_obj = SimpleNamespace(matrix_world=_FakeMatrix())
    scene = SimpleNamespace(camera=cam_obj)

    monkeypatch.setattr(mod, "IN_BLENDER", True)
    monkeypatch.setattr(
        mod,
        "world_to_camera_view",
        lambda _scene, _cam, _world_pt: SimpleNamespace(x=0.5, y=1.00005, z=1.0),
    )

    proj = mod.project_world_point(
        cam_inv=cam_obj.matrix_world.inverted(),
        world_pt=SimpleNamespace(),
        vfov_half=1.0,
        hfov_half=1.0,
        res_x=1024,
        res_y=1024,
        cam_obj=cam_obj,
        scene=scene,
    )

    assert proj is not None
    assert proj["pixel_xy"] == (512, 0)


def test_project_world_point_still_rejects_clearly_outside_view(monkeypatch):
    class _FakeMatrix:
        def inverted(self):
            return self

        def __matmul__(self, other):
            return SimpleNamespace(z=-1.0)

    cam_obj = SimpleNamespace(matrix_world=_FakeMatrix())
    scene = SimpleNamespace(camera=cam_obj)

    monkeypatch.setattr(mod, "IN_BLENDER", True)
    monkeypatch.setattr(
        mod,
        "world_to_camera_view",
        lambda _scene, _cam, _world_pt: SimpleNamespace(x=0.5, y=1.05, z=1.0),
    )

    proj = mod.project_world_point(
        cam_inv=cam_obj.matrix_world.inverted(),
        world_pt=SimpleNamespace(),
        vfov_half=1.0,
        hfov_half=1.0,
        res_x=1024,
        res_y=1024,
        cam_obj=cam_obj,
        scene=scene,
    )

    assert proj is None


def test_ensure_start_anchor_projection_replaces_leading_offscreen_prefix():
    projected = [None, None, [520, 980], [540, 900]]

    anchored = mod.ensure_start_anchor_projection(projected, [512, 1023])

    assert anchored == [[512, 1023], [512, 1023], [520, 980], [540, 900]]


def test_ensure_start_anchor_projection_preserves_length_when_all_points_offscreen():
    projected = [None, None, None]

    anchored = mod.ensure_start_anchor_projection(projected, [512, 1023])

    assert anchored == [[512, 1023], None, None]


def test_include_fixed_start_candidate_injects_visible_bottom_center_start():
    candidates = [
        {
            "waypoint_id": 3,
            "display_id": 1,
            "world_xyz": [0.2, 0.1, 0.0],
            "image_xy": [530, 900],
            "depth": 2.0,
            "candidate_source": "blender_grid",
        }
    ]
    fixed_start = {
        "waypoint_id": -1,
        "display_id": 0,
        "world_xyz": [0.0, 0.0, 0.0],
        "image_xy": [512, 1023],
        "depth": 1.0,
        "pointmass_walkable": True,
        "embodied_feasible": True,
        "candidate_source": "fixed_bottom_center_start",
    }

    enriched = mod.include_fixed_start_candidate(candidates, fixed_start)

    start = next(c for c in enriched if c["candidate_source"] == "fixed_bottom_center_start")
    assert start["waypoint_id"] >= 0
    assert start["image_xy"] == [512, 1023]
    assert start["display_id"] == 1


def test_evaluate_dense_path_projection_allows_missing_obstacle_mask():
    projected = [[512, 1023], None, [540, 900]]

    safe, first_block = mod.evaluate_dense_path_projection(
        projected,
        obstacle_mask=None,
        pad_px=2,
    )

    assert safe is True
    assert first_block is None


def test_evaluate_dense_path_projection_detects_blocking_segment():
    projected = [[0, 0], [8, 0], [8, 8], [20, 8]]
    obstacle_mask = mod.rects_to_mask([(7, 2, 9, 6)], width=24, height=12)

    safe, first_block = mod.evaluate_dense_path_projection(
        projected,
        obstacle_mask=obstacle_mask,
        pad_px=0,
    )

    assert safe is False
    assert first_block == 1


def test_refine_sparse_path_indices_returns_empty_when_keep_index_exceeds_projection_length():
    refined = mod.refine_sparse_path_indices_for_projection(
        projected_points=[[512, 1023], None, None],
        keep_indices=[0, 4],
        obstacle_mask=mod.rects_to_mask([(0, 0, 1, 1)], width=4, height=4),
        pad_px=0,
    )

    assert refined == []


def test_routing_endpoint_within_goal_radius_uses_final_sparse_endpoint():
    bbox = [0.0, 0.0, 0.0, 0.6, 0.6, 0.6, 0.0, 0.0, 0.0]

    assert mod.routing_endpoint_within_goal_radius([[0.0, 0.0, 0.0], [0.7, 0.0, 0.0]], bbox, 1.0)
    assert not mod.routing_endpoint_within_goal_radius([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], bbox, 1.0)


def test_select_path_endpoint_anchor_uses_spatial_endpoint_not_path_order():
    candidate_lookup = {
        13: {"waypoint_id": 13, "world_xyz": [0.335, -1.01, 0.0], "candidate_source": "blender_grid"},
        48: {"waypoint_id": 48, "world_xyz": [-0.065, 1.49, 0.0], "candidate_source": "path_world_forced"},
        52: {"waypoint_id": 52, "world_xyz": [-0.196, -1.135, 0.0], "candidate_source": "path_world_forced"},
    }

    selected = mod.select_path_endpoint_anchor(
        endpoint_world=[-0.065, 1.49, 0.0],
        primary_path_ids=[13, 52],
        fallback_path_ids=[48, 52],
        candidate_lookup=candidate_lookup,
    )

    assert selected is not None
    assert selected["waypoint_id"] == 48


def test_annotate_routing_instruction_metadata_uses_synonym_group_color_and_distance():
    candidates = [
        {
            "target_id": 10,
            "target_category": "dresser",
            "canonical_category": "cabinet",
            "semantic_group_id": "cabinet_like",
            "material_color_name": "brown",
            "distance_m": 1.2,
        },
        {
            "target_id": 11,
            "target_category": "cabinet",
            "canonical_category": "cabinet",
            "semantic_group_id": "cabinet_like",
            "material_color_name": "brown",
            "distance_m": 2.4,
        },
        {
            "target_id": 12,
            "target_category": "cabinet",
            "canonical_category": "cabinet",
            "semantic_group_id": "cabinet_like",
            "material_color_name": "white",
            "distance_m": 0.8,
        },
    ]

    annotated, rejections = mod.annotate_routing_instruction_metadata(
        candidates,
        target_rule="nearest",
        distance_tie_eps_m=0.001,
    )

    by_id = {int(c["target_id"]): c for c in annotated}
    assert rejections == []
    assert by_id[10]["instruction_selection"]["selection_status"] == "unique_winner"
    assert by_id[10]["instruction_selection"]["is_rule_winner"] is True
    assert by_id[10]["instruction_selection"]["winner_target_ids"] == [10]
    assert by_id[11]["instruction_selection"]["selection_status"] == "farther_same_identity"
    assert by_id[11]["instruction_selection"]["is_rule_winner"] is False
    assert by_id[11]["instruction_selection"]["winner_target_ids"] == [10]
    assert by_id[12]["instruction_selection"]["selection_status"] == "unique_winner"
    assert by_id[12]["instruction_selection"]["is_rule_winner"] is True
    assert by_id[12]["instruction_selection"]["winner_target_ids"] == [12]


def test_annotate_routing_instruction_metadata_rejects_distance_ties():
    candidates = [
        {
            "target_id": 10,
            "target_category": "dresser",
            "canonical_category": "cabinet",
            "semantic_group_id": "cabinet_like",
            "material_color_name": "brown",
            "distance_m": 1.2,
        },
        {
            "target_id": 11,
            "target_category": "cabinet",
            "canonical_category": "cabinet",
            "semantic_group_id": "cabinet_like",
            "material_color_name": "brown",
            "distance_m": 1.2004,
        },
    ]

    annotated, rejections = mod.annotate_routing_instruction_metadata(
        candidates,
        target_rule="nearest",
        distance_tie_eps_m=0.001,
    )

    by_id = {int(c["target_id"]): c for c in annotated}
    assert by_id[10]["instruction_selection"]["selection_status"] == "ambiguous_distance_tie"
    assert by_id[11]["instruction_selection"]["selection_status"] == "ambiguous_distance_tie"
    assert by_id[10]["instruction_selection"]["is_rule_winner"] is False
    assert by_id[11]["instruction_selection"]["is_rule_winner"] is False
    assert rejections == [
        {
            "target_id": 10,
            "reason": "instruction_identity_ambiguous",
            "semantic_group_id": "cabinet_like",
            "material_color_name": "brown",
            "target_rule": "nearest",
            "winner_target_ids": [10, 11],
        },
        {
            "target_id": 11,
            "reason": "instruction_identity_ambiguous",
            "semantic_group_id": "cabinet_like",
            "material_color_name": "brown",
            "target_rule": "nearest",
            "winner_target_ids": [10, 11],
        },
    ]


def test_select_diverse_routing_candidates_prefers_distinct_identities_and_paths():
    candidates = [
        {
            "routing_id": "routing_v0_c001",
            "target_id": 10,
            "instruction_identity": {"semantic_group_id": "cabinet_like", "material_color_name": "brown"},
            "instruction_selection": {"selection_status": "unique_winner", "is_rule_winner": True},
            "routing_complexity": {"detour_ratio_raw": 1.05, "dense_turn_count": 0},
            "debug_dense_path_world": [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        },
        {
            "routing_id": "routing_v0_c002",
            "target_id": 11,
            "instruction_identity": {"semantic_group_id": "cabinet_like", "material_color_name": "brown"},
            "instruction_selection": {"selection_status": "unique_winner", "is_rule_winner": True},
            "routing_complexity": {"detour_ratio_raw": 1.02, "dense_turn_count": 0},
            "debug_dense_path_world": [[0.0, 0.0, 0.0], [2.1, 0.0, 0.0]],
        },
        {
            "routing_id": "routing_v0_c003",
            "target_id": 12,
            "instruction_identity": {"semantic_group_id": "chair_like", "material_color_name": "blue"},
            "instruction_selection": {"selection_status": "unique_winner", "is_rule_winner": True},
            "routing_complexity": {"detour_ratio_raw": 1.45, "dense_turn_count": 2},
            "debug_dense_path_world": [[0.0, 0.0, 0.0], [0.5, 1.0, 0.0], [2.0, 1.0, 0.0]],
        },
    ]

    selected = mod.select_diverse_routing_candidates(candidates, max_candidates=2)

    assert [c["target_id"] for c in selected] == [12, 10]


def test_enrich_navigation_fact_metadata_adds_validity_axes_and_tier():
    func = getattr(mod, "enrich_navigation_fact_metadata", None)
    assert callable(func), "render_scenes.enrich_navigation_fact_metadata is missing"

    candidates = [
        {
            "routing_id": "routing_v0_c001",
            "target_id": 10,
            "instruction_identity": {"semantic_group_id": "cabinet_like", "material_color_name": "brown"},
            "instruction_selection": {
                "selection_status": "unique_winner",
                "is_rule_winner": True,
                "cohort_size": 2,
                "distance_rank": 1,
            },
            "routing_complexity": {
                "path_length_raw_m": 4.5,
                "path_length_sparse_m": 4.0,
                "start_goal_l2_m": 4.0,
                "detour_ratio_raw": 1.125,
                "detour_ratio_sparse": 1.0,
                "path_smoothing_gain_m": 0.5,
                "dense_turn_count": 2,
                "sparse_turn_count": 2,
                "num_segments_sparse": 3,
                "min_clearance_m": 0.55,
                "mean_clearance_m": 0.55,
                "goal_zone_clearance_m": 0.55,
                "start_zone_clearance_m": 0.6,
                "narrow_passage_fraction": 0.08,
                "path_projection_safe": True,
                "dense_path_projection_safe": True,
                "start_in_bottom_band": True,
                "instruction_cohort_size": 2,
                "distance_gap_to_next_m": 0.55,
                "num_same_group_diff_color": 0,
                "num_same_group_same_color": 1,
            },
            "invariants": {
                "start_in_forward_sector": True,
                "goal_in_target_zone": True,
                "embodied_collision_free": True,
            },
        }
    ]

    enriched = func(candidates)

    candidate = enriched[0]
    assert candidate["navigation_validity"]["eligible"] is True
    assert candidate["navigation_validity"]["failed_checks"] == []
    assert candidate["navigation_axes"] == {
        "reference_axis": "medium",
        "geometry_axis": "medium",
        "embodiment_axis": "low",
    }
    assert candidate["navigation_tier"] == "hard"
    assert candidate["navigation_reference_facts"]["reference_resolution_mode"] == "distance_only"
    assert candidate["navigation_geometry_facts"]["decision_count_point"] == 2
    assert candidate["navigation_embodiment_facts"]["embodied_extra_length_m"] == 0.0
    assert candidate["navigation_embodiment_facts"]["path_overlap_point_vs_embodied"] == 1.0


def test_collect_dense_debug_waypoint_ids_maps_before_projection_normalization():
    classification_candidates = [
        {
            "waypoint_id": 1,
            "world_xyz": [0.0, 0.0, 0.0],
            "image_xy": [0, 0],
            "pointmass_walkable": True,
            "embodied_feasible": True,
        },
        {
            "waypoint_id": 2,
            "world_xyz": [0.4, 0.0, 0.0],
            "image_xy": [8, 0],
            "pointmass_walkable": True,
            "embodied_feasible": True,
        },
        {
            "waypoint_id": 3,
            "world_xyz": [0.8, 0.0, 0.0],
            "image_xy": [16, 0],
            "pointmass_walkable": True,
            "embodied_feasible": True,
        },
    ]

    point_ids, embodied_ids = mod.collect_dense_debug_waypoint_ids(
        path_world=[[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [0.8, 0.0, 0.0]],
        classification_candidates=classification_candidates,
        obstacle_mask=mod.rects_to_mask([], width=20, height=4),
        pad_px=0,
    )

    assert point_ids == [1, 2, 3]
    assert embodied_ids == [1, 2, 3]


def test_refine_path_indices_keeps_projection_safe_split_point():
    projected = [[0, 0], [8, 0], [8, 8], [20, 8]]
    mask = mod.rects_to_mask([(9, 0, 15, 5)], width=24, height=12)

    refined = mod.refine_sparse_path_indices_for_projection(
        projected,
        keep_indices=[0, 3],
        obstacle_mask=mask,
        pad_px=0,
    )

    assert refined == [0, 2, 3]


def test_refine_path_indices_requires_world_traversable_kept_segments():
    class NavGridStub:
        def is_path_traversable(self, pts):
            p0, p1 = pts
            blocked = (
                abs(float(p0[0]) - 0.0) < 1e-6
                and abs(float(p0[1]) - 0.0) < 1e-6
                and abs(float(p1[0]) - 3.0) < 1e-6
                and abs(float(p1[1]) - 0.0) < 1e-6
            )
            return {"traversable": not blocked}

    projected = [[0, 0], [8, 0], [16, 0], [24, 0]]
    world = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]

    refined = mod.refine_sparse_path_indices_for_projection(
        projected_points=projected,
        keep_indices=[0, 3],
        obstacle_mask=mod.rects_to_mask([], width=30, height=4),
        pad_px=0,
        path_world=world,
        nav_grid=NavGridStub(),
    )

    assert refined == [0, 2, 3]


def test_sparsify_candidate_path_ids_keeps_projection_safe_waypoints():
    candidate_lookup = {
        100: {"image_xy": [0, 0]},
        101: {"image_xy": [8, 0]},
        102: {"image_xy": [8, 8]},
        103: {"image_xy": [20, 8]},
    }
    mask = mod.rects_to_mask([(9, 0, 15, 5)], width=24, height=12)

    refined = mod.sparsify_candidate_path_ids_for_projection(
        [100, 101, 102, 103],
        candidate_lookup,
        obstacle_mask=mask,
        pad_px=0,
    )

    assert refined == [100, 102, 103]


def test_apply_candidate_pixel_deobjectification_keeps_nonblocking_object_pixels_walkable():
    candidates = [
        {
            "image_xy": [1, 1],
            "depth": 2.0,
            "pointmass_walkable": True,
            "embodied_feasible": True,
        }
    ]

    changed = mod.apply_candidate_pixel_deobjectification(
        candidates,
        object_index_img=mod.np.array([[0, 0, 0], [0, 5, 0], [0, 0, 0]], dtype=mod.np.int32),
        object_index_pass_ids={5},
        nonblocking_pass_ids={5},
        depth_map=None,
        depth_eps_m=0.08,
        depth_window_radius_px=1,
    )

    assert changed == 0
    assert candidates[0]["pointmass_walkable"] is True
    assert candidates[0]["embodied_feasible"] is True


def test_refine_path_indices_without_safe_visible_bridge_returns_empty():
    projected = [[0, 0], None, None, [19, 0]]
    mask = mod.rects_to_mask([(6, 0, 14, 2)], width=20, height=5)

    refined = mod.refine_sparse_path_indices_for_projection(
        projected,
        keep_indices=[0, 3],
        obstacle_mask=mask,
        pad_px=0,
    )

    assert refined == []


def test_sparsify_candidate_path_ids_without_safe_bridge_returns_empty():
    candidate_lookup = {
        10: {"image_xy": [0, 0]},
        11: {"image_xy": [5, 0]},
        12: {"image_xy": [19, 0]},
    }
    mask = mod.rects_to_mask([(6, 0, 14, 2)], width=20, height=5)

    refined = mod.sparsify_candidate_path_ids_for_projection(
        [10, 11, 12],
        candidate_lookup,
        obstacle_mask=mask,
        pad_px=0,
    )

    assert refined == []


def test_refine_routing_outputs_for_projection_updates_bundle_payloads():
    dense_path = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 1.0, 0.0],
    ]
    projected_dense_path = [[0, 0], [8, 0], [8, 8], [20, 8]]
    obstacle_mask = mod.rects_to_mask([(9, 0, 15, 5)], width=24, height=12)
    classification_candidates = [
        {"waypoint_id": 100, "image_xy": [0, 0]},
        {"waypoint_id": 101, "image_xy": [8, 0]},
        {"waypoint_id": 102, "image_xy": [8, 8]},
        {"waypoint_id": 103, "image_xy": [20, 8]},
        {"waypoint_id": 200, "image_xy": [0, 0]},
        {"waypoint_id": 201, "image_xy": [8, 0]},
        {"waypoint_id": 202, "image_xy": [8, 8]},
        {"waypoint_id": 203, "image_xy": [20, 8]},
    ]

    keypoints, point_ids, embodied_ids = mod.refine_routing_outputs_for_projection(
        dense_path_world=dense_path,
        gt_path_keypoints_world=[dense_path[0], dense_path[-1]],
        path_wp_point=[100, 101, 102, 103],
        path_wp_embodied=[200, 201, 202, 203],
        classification_candidates=classification_candidates,
        obstacle_mask=obstacle_mask,
        projected_dense_path=projected_dense_path,
        pad_px=0,
    )

    assert keypoints == [dense_path[0], dense_path[2], dense_path[3]]
    assert point_ids == [100, 102, 103]
    assert embodied_ids == [200, 202, 203]


def test_refine_routing_outputs_for_projection_rejects_projection_unsafe_path():
    dense_path = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [3.0, 0.0, 0.0],
    ]
    projected_dense_path = [[0, 0], None, None, [19, 0]]
    obstacle_mask = mod.rects_to_mask([(6, 0, 14, 2)], width=20, height=5)
    classification_candidates = [
        {"waypoint_id": 10, "image_xy": [0, 0]},
        {"waypoint_id": 11, "image_xy": [5, 0]},
        {"waypoint_id": 12, "image_xy": [19, 0]},
        {"waypoint_id": 20, "image_xy": [0, 0]},
        {"waypoint_id": 21, "image_xy": [5, 0]},
        {"waypoint_id": 22, "image_xy": [19, 0]},
    ]

    keypoints, point_ids, embodied_ids = mod.refine_routing_outputs_for_projection(
        dense_path_world=dense_path,
        gt_path_keypoints_world=[dense_path[0], dense_path[-1]],
        path_wp_point=[10, 11, 12],
        path_wp_embodied=[20, 21, 22],
        classification_candidates=classification_candidates,
        obstacle_mask=obstacle_mask,
        projected_dense_path=projected_dense_path,
        pad_px=0,
    )

    assert keypoints == []
    assert point_ids == []
    assert embodied_ids == []


def test_path_ids_projection_collision_free_rejects_blocked_direct_line():
    candidate_lookup = {
        45: {"image_xy": [331, 981]},
        31: {"image_xy": [604, 514]},
    }
    mask = mod.rects_to_mask([(430, 640, 520, 760)], width=1024, height=1024)

    assert not mod.path_ids_projection_collision_free(
        [45, 31],
        candidate_lookup,
        obstacle_mask=mask,
        pad_px=2,
    )
