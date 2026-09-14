from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "tier_policy.py"


def _load_module():
    assert MODULE_PATH.exists(), "scripts/tier_policy.py is missing"
    spec = spec_from_file_location("tier_policy", MODULE_PATH)
    mod = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_navigation_tier_composition_rules():
    mod = _load_module()

    assert mod.classify_navigation_tier(
        reference_axis="low",
        geometry_axis="low",
        embodiment_axis="low",
    ) == "easy"
    assert mod.classify_navigation_tier(
        reference_axis="medium",
        geometry_axis="low",
        embodiment_axis="low",
    ) == "medium"
    assert mod.classify_navigation_tier(
        reference_axis="medium",
        geometry_axis="medium",
        embodiment_axis="low",
    ) == "hard"
    assert mod.classify_navigation_tier(
        reference_axis="high",
        geometry_axis="low",
        embodiment_axis="low",
    ) == "hard"


def test_affordance_tier_composition_rules():
    mod = _load_module()

    assert mod.classify_affordance_tier(
        clutter_axis="low",
        boundary_axis="low",
        embodiment_gap_axis="low",
    ) == "easy"
    assert mod.classify_affordance_tier(
        clutter_axis="medium",
        boundary_axis="low",
        embodiment_gap_axis="low",
    ) == "medium"
    assert mod.classify_affordance_tier(
        clutter_axis="low",
        boundary_axis="low",
        embodiment_gap_axis="high",
    ) == "hard"


def test_navigation_reference_axis_thresholds():
    mod = _load_module()

    assert mod.classify_reference_axis(
        {
            "reference_resolution_mode": "category_only",
            "reference_top2_gap_m": 1.2,
            "instruction_cohort_size": 1,
            "same_semantic_group_same_color_count": 0,
        }
    ) == "low"
    assert mod.classify_reference_axis(
        {
            "reference_resolution_mode": "distance_only",
            "reference_top2_gap_m": 0.6,
            "instruction_cohort_size": 2,
            "same_semantic_group_same_color_count": 1,
        }
    ) == "medium"
    assert mod.classify_reference_axis(
        {
            "reference_resolution_mode": "color_and_distance",
            "reference_top2_gap_m": 0.2,
            "instruction_cohort_size": 3,
            "same_semantic_group_same_color_count": 2,
        }
    ) == "high"


def test_navigation_geometry_and_embodiment_thresholds():
    mod = _load_module()

    assert mod.classify_geometry_axis(
        {
            "path_length_point_m": 2.5,
            "detour_ratio_point": 1.05,
            "turn_count_point": 1,
            "decision_count_point": 1,
        }
    ) == "low"
    assert mod.classify_geometry_axis(
        {
            "path_length_point_m": 5.5,
            "detour_ratio_point": 1.2,
            "turn_count_point": 2,
            "decision_count_point": 2,
        }
    ) == "medium"
    assert mod.classify_geometry_axis(
        {
            "path_length_point_m": 7.2,
            "detour_ratio_point": 1.4,
            "turn_count_point": 4,
            "decision_count_point": 3,
        }
    ) == "high"

    assert mod.classify_embodiment_axis(
        {
            "embodied_clearance_margin_min_m": 0.25,
            "embodied_clearance_margin_goal_m": 0.25,
            "narrow_passage_fraction": 0.05,
            "embodied_extra_length_m": 0.2,
            "embodied_extra_decisions": 0,
            "path_overlap_point_vs_embodied": 0.95,
        }
    ) == "low"
    assert mod.classify_embodiment_axis(
        {
            "embodied_clearance_margin_min_m": 0.08,
            "embodied_clearance_margin_goal_m": 0.1,
            "narrow_passage_fraction": 0.2,
            "embodied_extra_length_m": 0.8,
            "embodied_extra_decisions": 1,
            "path_overlap_point_vs_embodied": 0.7,
        }
    ) == "medium"
    assert mod.classify_embodiment_axis(
        {
            "embodied_clearance_margin_min_m": 0.01,
            "embodied_clearance_margin_goal_m": 0.03,
            "narrow_passage_fraction": 0.5,
            "embodied_extra_length_m": 1.5,
            "embodied_extra_decisions": 2,
            "path_overlap_point_vs_embodied": 0.4,
        }
    ) == "high"


def test_affordance_axis_thresholds():
    mod = _load_module()

    assert mod.classify_clutter_axis(
        {
            "visible_candidate_count": 20,
            "candidate_nn_p25_px": 50,
            "label_overlap_fraction": 0.01,
        }
    ) == "low"
    assert mod.classify_clutter_axis(
        {
            "visible_candidate_count": 45,
            "candidate_nn_p25_px": 32,
            "label_overlap_fraction": 0.1,
        }
    ) == "medium"
    assert mod.classify_clutter_axis(
        {
            "visible_candidate_count": 80,
            "candidate_nn_p25_px": 24,
            "label_overlap_fraction": 0.2,
        }
    ) == "high"

    assert mod.classify_boundary_axis(
        {
            "near_point_boundary_fraction": 0.05,
            "near_embodied_boundary_fraction": 0.08,
        }
    ) == "low"
    assert mod.classify_boundary_axis(
        {
            "near_point_boundary_fraction": 0.2,
            "near_embodied_boundary_fraction": 0.2,
        }
    ) == "medium"
    assert mod.classify_boundary_axis(
        {
            "near_point_boundary_fraction": 0.3,
            "near_embodied_boundary_fraction": 0.1,
        }
    ) == "high"

    assert mod.classify_embodiment_gap_axis(
        {
            "point_embodied_disagreement_fraction": 0.05,
            "embodied_only_hard_negative_fraction": 0.05,
        }
    ) == "low"
    assert mod.classify_embodiment_gap_axis(
        {
            "point_embodied_disagreement_fraction": 0.2,
            "embodied_only_hard_negative_fraction": 0.1,
        }
    ) == "medium"
    assert mod.classify_embodiment_gap_axis(
        {
            "point_embodied_disagreement_fraction": 0.35,
            "embodied_only_hard_negative_fraction": 0.25,
        }
    ) == "high"
