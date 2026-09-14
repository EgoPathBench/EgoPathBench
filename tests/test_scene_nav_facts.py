from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "scene_nav_facts.py"
spec = spec_from_file_location("scene_nav_facts", MODULE_PATH)
mod = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_build_scene_nav_facts_assigns_canonical_candidate_ids_and_hash():
    facts = mod.build_scene_nav_facts(
        scene_id="demo_scene",
        views=[
            {
                "view_id": 0,
                "camera": {
                    "position": [0.0, 0.0, 1.6],
                    "rotation": [0.0, 0.0, 0.0],
                    "fov": 120.0,
                    "resolution": [1024, 1024],
                },
                "visible_objects": [
                    {"id": 7, "category": "cabinet", "screen_x": 0.1},
                ],
                "classification_candidates": [
                    {
                        "waypoint_id": 24,
                        "display_id": 1,
                        "world_xyz": [0.0, -1.0, 0.0],
                        "image_xy": [512, 1023],
                        "screen_xy_norm": [0.0, -1.0],
                        "depth": 1.0,
                        "pointmass_walkable": True,
                        "embodied_feasible": True,
                        "candidate_source": "fixed_bottom_center_start",
                    },
                    {
                        "waypoint_id": 36,
                        "display_id": 2,
                        "world_xyz": [1.0, -0.3, 0.0],
                        "image_xy": [700, 700],
                        "screen_xy_norm": [0.3, -0.4],
                        "depth": 2.0,
                        "pointmass_walkable": True,
                        "embodied_feasible": True,
                        "candidate_source": "routing_sparse_keypoint",
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
                        "distance_m": 2.4,
                        "gt_path_waypoint_ids_pointmass": [24, 36],
                        "gt_path_waypoint_ids_embodied": [24, 36],
                        "gt_path_keypoints_world": [
                            [0.0, -1.0, 0.0],
                            [1.0, -0.3, 0.0],
                        ],
                        "gt_path_keypoints_image_xy": [
                            [512, 1023],
                            [700, 700],
                        ],
                        "start_anchor": {"waypoint_id": 24},
                        "goal_anchor": {"waypoint_id": 36},
                    }
                ],
                "rejection_reasons": [],
            }
        ],
    )

    assert facts["schema_version"] == mod.SCENE_NAV_FACTS_SCHEMA
    assert isinstance(facts["facts_hash"], str) and facts["facts_hash"]
    view = facts["views"][0]
    assert view["classification_candidates"][0]["candidate_id"] == "classification_v0_c001"
    assert view["classification_candidates"][1]["candidate_id"] == "classification_v0_c002"
    assert view["routing_candidates"][0]["start_candidate_id"] == "classification_v0_c001"
    assert view["routing_candidates"][0]["goal_candidate_id"] == "classification_v0_c002"
    assert view["routing_candidates"][0]["gt_path_candidate_ids_pointmass"] == [
        "classification_v0_c001",
        "classification_v0_c002",
    ]


def test_build_view_bundle_from_scene_nav_facts_preserves_hash_and_visibility():
    facts = mod.build_scene_nav_facts(
        scene_id="demo_scene",
        views=[
            {
                "view_id": 3,
                "camera": {
                    "position": [0.0, 0.0, 1.6],
                    "rotation": [0.0, 0.0, 0.0],
                    "fov": 120.0,
                    "resolution": [1024, 1024],
                },
                "visible_objects": [{"id": 9, "category": "lamp", "screen_x": -0.2}],
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
                    "visible_candidate_count": 12,
                    "point_embodied_disagreement_fraction": 0.2,
                },
                "classification_candidates": [
                    {
                        "waypoint_id": 10,
                        "display_id": 1,
                        "world_xyz": [0.0, 0.0, 0.0],
                        "image_xy": [512, 900],
                        "screen_xy_norm": [0.0, -0.8],
                        "depth": 1.0,
                        "pointmass_walkable": True,
                        "embodied_feasible": True,
                        "candidate_source": "blender_grid",
                    }
                ],
                "routing_candidates": [],
                "rejection_reasons": [],
            }
        ],
    )

    bundle = mod.build_view_bundle_from_scene_nav_facts(facts, view_id=3)

    assert bundle["schema_version"] == "next_view_bundle_v3"
    assert bundle["facts_schema"] == mod.SCENE_NAV_FACTS_SCHEMA
    assert bundle["facts_hash"] == facts["facts_hash"]
    assert bundle["visible_objects"] == [{"id": 9, "category": "lamp", "screen_x": -0.2}]
    assert bundle["affordance_tier"] == "hard"
    assert bundle["affordance_axes"]["clutter_axis"] == "medium"


def test_build_scene_nav_facts_preserves_navigation_fact_payloads():
    facts = mod.build_scene_nav_facts(
        scene_id="demo_scene",
        views=[
            {
                "view_id": 1,
                "camera": {"position": [0.0, 0.0, 1.6]},
                "visible_objects": [],
                "classification_candidates": [
                    {
                        "waypoint_id": 5,
                        "display_id": 1,
                        "world_xyz": [0.0, 0.0, 0.0],
                        "image_xy": [100, 200],
                        "screen_xy_norm": [0.0, -0.8],
                        "depth": 1.0,
                        "pointmass_walkable": True,
                        "embodied_feasible": True,
                        "candidate_source": "blender_grid",
                    }
                ],
                "routing_candidates": [
                    {
                        "routing_id": "routing_v1_c001",
                        "target_id": 3,
                        "target_category": "chair",
                        "canonical_category": "chair",
                        "semantic_group_id": "chair",
                        "material_color_name": "gray",
                        "distance_m": 2.3,
                        "gt_path_waypoint_ids_pointmass": [5],
                        "gt_path_waypoint_ids_embodied": [5],
                        "start_anchor": {"waypoint_id": 5},
                        "goal_anchor": {"waypoint_id": 5},
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
                    }
                ],
                "rejection_reasons": [],
            }
        ],
    )

    routing = facts["views"][0]["routing_candidates"][0]
    assert routing["navigation_validity"]["eligible"] is True
    assert routing["navigation_axes"]["geometry_axis"] == "medium"
    assert routing["navigation_tier"] == "medium"
