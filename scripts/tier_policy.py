#!/usr/bin/env python3
"""
Shared absolute tier policy for NavBench3D-Next task families.
"""

from __future__ import annotations


AXIS_LEVELS = ("low", "medium", "high")
TIER_ORDER = ("easy", "medium", "hard")

LEGACY_ROUTING_REQUIRED_FIELDS = (
    "detour_ratio_raw",
    "dense_turn_count",
    "path_length_raw_m",
    "narrow_passage_fraction",
    "instruction_cohort_size",
)


def _require_dict(payload: dict | None, *, context: str) -> dict:
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be a dict")
    return payload


def _require_float(payload: dict, key: str, *, context: str) -> float:
    if key not in payload:
        raise ValueError(f"missing required {context} field: {key}")
    return float(payload[key])


def _require_text(payload: dict, key: str, *, context: str) -> str:
    if key not in payload or payload[key] is None:
        raise ValueError(f"missing required {context} field: {key}")
    value = str(payload[key]).strip().lower()
    if not value:
        raise ValueError(f"missing required {context} field: {key}")
    return value


def _validate_axis_level(value: str, *, axis_name: str) -> str:
    axis = str(value).strip().lower()
    if axis not in AXIS_LEVELS:
        raise ValueError(f"{axis_name} must be one of {AXIS_LEVELS}, got {value!r}")
    return axis


def compose_tier_from_axis_levels(*axis_levels: str) -> str:
    normalized = [_validate_axis_level(axis, axis_name="axis level") for axis in axis_levels]
    high_count = sum(axis == "high" for axis in normalized)
    medium_count = sum(axis == "medium" for axis in normalized)
    if high_count > 0 or medium_count >= 2:
        return "hard"
    if medium_count == 1:
        return "medium"
    return "easy"


def classify_navigation_tier(*, reference_axis: str, geometry_axis: str, embodiment_axis: str) -> str:
    return compose_tier_from_axis_levels(reference_axis, geometry_axis, embodiment_axis)


def classify_affordance_tier(*, clutter_axis: str, boundary_axis: str, embodiment_gap_axis: str) -> str:
    return compose_tier_from_axis_levels(clutter_axis, boundary_axis, embodiment_gap_axis)


def classify_reference_axis(reference_facts: dict | None) -> str:
    payload = _require_dict(reference_facts, context="reference_facts")
    resolution_mode = _require_text(payload, "reference_resolution_mode", context="reference_facts")
    top2_gap = _require_float(payload, "reference_top2_gap_m", context="reference_facts")
    cohort_size = int(_require_float(payload, "instruction_cohort_size", context="reference_facts"))
    same_color_count = int(_require_float(payload, "same_semantic_group_same_color_count", context="reference_facts"))

    if (
        resolution_mode == "color_and_distance"
        or top2_gap < 0.40
        or cohort_size >= 3
        or same_color_count >= 2
    ):
        return "high"

    if resolution_mode == "category_only":
        return "low"

    if resolution_mode in {"distance_only", "color_only"} and top2_gap >= 0.80 and cohort_size <= 1:
        return "low"

    return "medium"


def classify_geometry_axis(geometry_facts: dict | None) -> str:
    payload = _require_dict(geometry_facts, context="geometry_facts")
    path_length = _require_float(payload, "path_length_point_m", context="geometry_facts")
    detour_ratio = _require_float(payload, "detour_ratio_point", context="geometry_facts")
    turn_count = int(_require_float(payload, "turn_count_point", context="geometry_facts"))
    decision_count = int(_require_float(payload, "decision_count_point", context="geometry_facts"))

    if (
        path_length > 6.0
        or detour_ratio > 1.35
        or turn_count >= 4
        or decision_count >= 3
    ):
        return "high"

    if (
        path_length <= 3.0
        and detour_ratio <= 1.10
        and turn_count <= 1
        and decision_count <= 1
    ):
        return "low"

    return "medium"


def classify_embodiment_axis(embodiment_facts: dict | None) -> str:
    payload = _require_dict(embodiment_facts, context="embodiment_facts")
    clearance_margin_min = _require_float(payload, "embodied_clearance_margin_min_m", context="embodiment_facts")
    clearance_margin_goal = _require_float(payload, "embodied_clearance_margin_goal_m", context="embodiment_facts")
    narrow_fraction = _require_float(payload, "narrow_passage_fraction", context="embodiment_facts")
    extra_length = _require_float(payload, "embodied_extra_length_m", context="embodiment_facts")
    extra_decisions = int(_require_float(payload, "embodied_extra_decisions", context="embodiment_facts"))
    overlap = _require_float(payload, "path_overlap_point_vs_embodied", context="embodiment_facts")

    if (
        clearance_margin_min < 0.05
        or clearance_margin_goal < 0.05
        or narrow_fraction > 0.35
        or extra_length > 1.00
        or extra_decisions >= 2
        or overlap < 0.50
    ):
        return "high"

    if (
        clearance_margin_min >= 0.20
        and narrow_fraction <= 0.10
        and extra_length <= 0.30
        and extra_decisions <= 0
        and overlap >= 0.85
    ):
        return "low"

    return "medium"


def classify_clutter_axis(clutter_facts: dict | None) -> str:
    payload = _require_dict(clutter_facts, context="clutter_facts")
    candidate_count = int(_require_float(payload, "visible_candidate_count", context="clutter_facts"))
    nn_p25_px = _require_float(payload, "candidate_nn_p25_px", context="clutter_facts")
    overlap_fraction = _require_float(payload, "label_overlap_fraction", context="clutter_facts")

    if candidate_count > 65 or nn_p25_px < 28 or overlap_fraction > 0.15:
        return "high"

    if candidate_count <= 35 and nn_p25_px >= 42 and overlap_fraction <= 0.05:
        return "low"

    return "medium"


def classify_boundary_axis(boundary_facts: dict | None) -> str:
    payload = _require_dict(boundary_facts, context="boundary_facts")
    near_point = _require_float(payload, "near_point_boundary_fraction", context="boundary_facts")
    near_embodied = _require_float(payload, "near_embodied_boundary_fraction", context="boundary_facts")

    if near_point > 0.25 or near_embodied > 0.25:
        return "high"

    if near_point <= 0.10 and near_embodied <= 0.10:
        return "low"

    return "medium"


def classify_embodiment_gap_axis(gap_facts: dict | None) -> str:
    payload = _require_dict(gap_facts, context="gap_facts")
    disagreement_fraction = _require_float(payload, "point_embodied_disagreement_fraction", context="gap_facts")
    hard_negative_fraction = _require_float(payload, "embodied_only_hard_negative_fraction", context="gap_facts")

    if disagreement_fraction > 0.30 or hard_negative_fraction > 0.20:
        return "high"

    if disagreement_fraction <= 0.10:
        return "low"

    return "medium"


def classify_legacy_routing_tier(routing_complexity: dict | None) -> str:
    payload = _require_dict(routing_complexity, context="routing_complexity")
    detour_ratio_raw = _require_float(payload, "detour_ratio_raw", context="routing_complexity")
    dense_turn_count = int(_require_float(payload, "dense_turn_count", context="routing_complexity"))
    path_length_raw_m = _require_float(payload, "path_length_raw_m", context="routing_complexity")
    narrow_passage_fraction = _require_float(payload, "narrow_passage_fraction", context="routing_complexity")
    instruction_cohort_size = int(_require_float(payload, "instruction_cohort_size", context="routing_complexity"))

    if (
        detour_ratio_raw <= 1.10
        and dense_turn_count <= 1
        and path_length_raw_m <= 4.0
        and narrow_passage_fraction <= 0.10
        and instruction_cohort_size <= 1
    ):
        return "easy"

    if (
        detour_ratio_raw <= 1.35
        and dense_turn_count <= 2
        and path_length_raw_m <= 7.0
        and narrow_passage_fraction <= 0.25
        and instruction_cohort_size <= 2
    ):
        return "medium"

    return "hard"
