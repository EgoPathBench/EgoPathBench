#!/usr/bin/env python3
"""
Conservative cue synthesis for fixed-GT prompt repair.

Current scope:
- prefers natural uniqueness
- then trusted relation
- then weaker but auditable fallbacks
- keeps rejected candidates and reasons explicit
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


CUE_FAMILY_PRIORITY = {
    "natural_unique": 0,
    "relation": 1,
    "relation + anchor_repair": 2,
    "relation + attribute": 3,
    "attribute": 4,
    "relation + color": 5,
    "color": 6,
    "strict_distance": 7,
    "drop": 999,
}
STRICT_DISTANCE_MIN_GAP_M = 0.30
NEAR_MIN_MARGIN = 0.15
NEAR_MAX_DISTANCE = 0.45
NEAR_CLUSTER_PAD = 0.12
ROOT_DIR = Path(__file__).resolve().parents[1]
C_PROXY_GROUPS_PATH = ROOT_DIR / "configs" / "c_intent_family_proxy_groups_v1.json"
C_PROMPT_CATALOG_PATH = ROOT_DIR / "configs" / "c_prompt_catalog_v1.json"
C_ONTOLOGY_PATH = ROOT_DIR / "configs" / "c_main_vnext_candidate_ontology.json"
GENERIC_ANCHOR_CATEGORIES = frozenset(
    {"object", "thing", "item", "furniture", "appliance", "fixture", "light"}
)
RELATION_TYPE_PRIORITY = {
    "under": 0,
    "on": 0,
    "inside": 0,
    "near": 1,
}
_C_PROXY_GROUPS_CACHE: dict | None = None
_C_PROMPT_CATALOG_CACHE: dict | None = None
_C_ONTOLOGY_CACHE: dict | None = None
_C_SLOT_SEMANTIC_OVERRIDES = {
    ("cooking_appliance", "bake_food"): {"oven"},
    ("cooking_appliance", "heat_up_food"): {"microwave"},
    ("cooking_appliance", "cook_on_stovetop"): {"stove"},
    ("cooking_appliance", "cook_on_stove"): {"stove"},
    ("cooking_appliance", "boil_water"): {"kettle", "teapot", "pot", "stove", "microwave"},
    ("cooking_appliance", "make_tea"): {"kettle", "teapot"},
    ("cooking_appliance", "simmer_food"): {"pot", "stove"},
    ("cooking_appliance", "cook_in_pot"): {"pot", "stove"},
}
DEFAULT_C_RESIDUAL_CUES = ("none", "distance", "side", "ordinal")
SIDE_MIN_GAP_X = 0.08
ORDINAL_MIN_GAP_X = 0.08


def _load_json(path: Path) -> dict:
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _normalized(text: object) -> str | None:
    if text is None:
        return None
    value = str(text).strip()
    return value or None


def _normalized_lower(text: object) -> str | None:
    value = _normalized(text)
    return value.lower() if value is not None else None


def _supported_visible_objects(hview: dict) -> list[dict]:
    return [
        obj
        for obj in list(hview.get("objects") or [])
        if _normalized_lower(obj.get("support_status")) == "supported"
        and _normalized_lower(obj.get("visibility_status")) == "visible"
    ]


def _objects_by_id(hview: dict) -> dict[int, dict]:
    objects: dict[int, dict] = {}
    for obj in _supported_visible_objects(hview):
        try:
            objects[int(obj["object_id"])] = obj
        except (KeyError, TypeError, ValueError):
            continue
    return objects


def _screen_xy(obj: dict) -> tuple[float | None, float | None]:
    screen = obj.get("screen_box_or_anchor") or {}
    try:
        return float(screen.get("x")), float(screen.get("y"))
    except (TypeError, ValueError):
        return None, None


def _euclidean_distance(obj_a: dict, obj_b: dict) -> float | None:
    ax, ay = _screen_xy(obj_a)
    bx, by = _screen_xy(obj_b)
    if None in (ax, ay, bx, by):
        return None
    return math.hypot(ax - bx, ay - by)


def _screen_x(obj: dict) -> float | None:
    x, _ = _screen_xy(obj)
    return x


def _distances_to_anchor(candidates: list[dict], anchors: list[dict]) -> dict[int, float]:
    out: dict[int, float] = {}
    for candidate in candidates:
        candidate_id = int(candidate["object_id"])
        distances = [
            distance
            for distance in (_euclidean_distance(candidate, anchor) for anchor in anchors)
            if distance is not None
        ]
        if distances:
            out[candidate_id] = min(distances)
    return out


def _load_c_proxy_groups() -> dict[str, list[str]]:
    global _C_PROXY_GROUPS_CACHE
    if _C_PROXY_GROUPS_CACHE is None:
        payload = _load_json(C_PROXY_GROUPS_PATH)
        _C_PROXY_GROUPS_CACHE = {
            _normalized_lower(key) or "": [
                category
                for category in (_normalized_lower(item) for item in list(value or []))
                if category is not None
            ]
            for key, value in payload.items()
        }
    return _C_PROXY_GROUPS_CACHE


def _load_c_prompt_catalog() -> dict:
    global _C_PROMPT_CATALOG_CACHE
    if _C_PROMPT_CATALOG_CACHE is None:
        _C_PROMPT_CATALOG_CACHE = _load_json(C_PROMPT_CATALOG_PATH)
    return _C_PROMPT_CATALOG_CACHE


def _load_c_ontology() -> dict:
    global _C_ONTOLOGY_CACHE
    if _C_ONTOLOGY_CACHE is None:
        _C_ONTOLOGY_CACHE = _load_json(C_ONTOLOGY_PATH)
    return _C_ONTOLOGY_CACHE


def _target_supported(hview: dict, target_id: int) -> bool:
    return bool((hview.get("target_supported_under_hview") or {}).get(str(target_id)))


def _effective_universe(hview: dict, target_id: int) -> list[int]:
    return [int(v) for v in list((hview.get("effective_candidate_universe") or {}).get(str(target_id), []))]


def _build_candidate(
    *,
    cue_family: str,
    uniqueness_pass: bool,
    cue_count: int,
    anchor_count: int,
    naturalness_score: int,
    answer_leakage_risk: int,
    stability_score: float = 0.0,
    cue_type: str | None = None,
    cue_value: object = None,
    anchor_ids: list[int] | None = None,
    rejection_reason: str | None = None,
    relation_type_priority: int = 999,
    residual_cue_type: str | None = None,
    residual_cue_value: str | None = None,
    minimal_residual_cue: bool = False,
) -> dict:
    return {
        "cue_family": cue_family,
        "uniqueness_pass": bool(uniqueness_pass),
        "cue_count": int(cue_count),
        "anchor_count": int(anchor_count),
        "naturalness_score": int(naturalness_score),
        "answer_leakage_risk": int(answer_leakage_risk),
        "stability_score": float(stability_score),
        "cue_type": cue_type,
        "cue_value": cue_value,
        "anchor_ids": list(anchor_ids or []),
        "rejection_reason": rejection_reason,
        "relation_type_priority": int(relation_type_priority),
        "residual_cue_type": residual_cue_type,
        "residual_cue_value": residual_cue_value,
        "minimal_residual_cue": bool(minimal_residual_cue),
    }


def _candidate_sort_key(candidate: dict) -> tuple:
    return (
        0 if candidate["uniqueness_pass"] else 1,
        CUE_FAMILY_PRIORITY.get(candidate["cue_family"], 100),
        int(candidate.get("relation_type_priority", 999)),
        int(candidate["cue_count"]),
        int(candidate["anchor_count"]),
        -int(candidate["naturalness_score"]),
        -float(candidate.get("stability_score", 0.0)),
        int(candidate["answer_leakage_risk"]),
        0 if candidate.get("minimal_residual_cue") else 1,
        str(candidate["cue_family"]),
        str(candidate.get("cue_type")),
        str(candidate.get("cue_value")),
        str(candidate.get("residual_cue_type")),
        str(candidate.get("residual_cue_value")),
        tuple(int(v) for v in candidate.get("anchor_ids", [])),
    )


def _summarize_why_not_relation(projected_relations: list[dict]) -> str | None:
    if not projected_relations:
        return "no_projected_relation_candidates"
    usable = [row for row in projected_relations if row.get("projection_status") == "usable"]
    if usable:
        return None
    reasons: list[str] = []
    for row in projected_relations:
        reason = _normalized(row.get("rejected_projection_reason"))
        if reason is not None:
            reasons.append(reason)
    if not reasons:
        return "no_usable_relation_candidates"
    reasons.sort()
    return reasons[0]


def _relation_anchor_category(relation: dict) -> str | None:
    anchor_categories = list(relation.get("anchor_categories") or [])
    for category in anchor_categories:
        normalized = _normalized_lower(category)
        if normalized is not None:
            return normalized
    return None


def _is_generic_anchor_category(category: str | None) -> bool:
    normalized = _normalized_lower(category)
    return normalized in GENERIC_ANCHOR_CATEGORIES


def _relation_type_priority(relation_type: str | None) -> int:
    normalized = _normalized_lower(relation_type)
    return RELATION_TYPE_PRIORITY.get(normalized, 2)


def _category_candidate_pool(gt_row: dict, hview: dict) -> list[dict]:
    target_category = _normalized_lower(gt_row.get("canonical_category")) or _normalized_lower(gt_row.get("target_category"))
    if target_category is None:
        return []
    candidates = [
        obj
        for obj in _supported_visible_objects(hview)
        if _normalized_lower(obj.get("canonical_category")) == target_category
    ]
    candidates.sort(key=lambda obj: int(obj["object_id"]))
    return candidates


def _c_slot_semantic_categories(gt_row: dict) -> set[str] | None:
    family = _normalized_lower(gt_row.get("intent_family")) or _normalized_lower(gt_row.get("same_type_key"))
    prompt_slot = _normalized_lower(gt_row.get("prompt_slot"))
    if family is None:
        return None

    override = _C_SLOT_SEMANTIC_OVERRIDES.get((family, prompt_slot or ""))
    if override is not None:
        return set(override)

    prompt_catalog = _load_c_prompt_catalog()
    family_spec = dict((prompt_catalog.get("families") or {}).get(family) or {})
    category_to_slot = {
        _normalized_lower(category): _normalized_lower(slot)
        for category, slot in dict(family_spec.get("category_to_slot") or {}).items()
        if _normalized_lower(category) is not None and _normalized_lower(slot) is not None
    }
    if prompt_slot is not None:
        slot_categories = {category for category, slot in category_to_slot.items() if slot == prompt_slot}
        if slot_categories:
            return slot_categories

    proxy_groups = _load_c_proxy_groups()
    categories = {
        category
        for category in (_normalized_lower(item) for item in list(proxy_groups.get(family) or []))
        if category is not None
    }
    return categories or None


def _c_allowed_residual_cues(gt_row: dict) -> tuple[str, ...]:
    family = _normalized_lower(gt_row.get("intent_family")) or _normalized_lower(gt_row.get("same_type_key"))
    families = dict((_load_c_ontology().get("families") or {}))
    spec = dict(families.get(family) or {})
    cues = [
        cue
        for cue in (_normalized_lower(item) for item in list(spec.get("residual_cues_allowed") or []))
        if cue is not None
    ]
    return tuple(cues or DEFAULT_C_RESIDUAL_CUES)


def _c_semantic_candidate_pool(gt_row: dict, hview: dict) -> list[dict]:
    same_type_key = _normalized_lower(gt_row.get("same_type_key"))
    semantic_categories = _c_slot_semantic_categories(gt_row)
    candidates = []
    for obj in _supported_visible_objects(hview):
        object_same_type_key = _normalized_lower(obj.get("same_type_key"))
        if same_type_key is not None and object_same_type_key != same_type_key:
            continue
        category = _normalized_lower(obj.get("canonical_category"))
        if semantic_categories is not None and category not in semantic_categories:
            continue
        candidates.append(obj)
    candidates.sort(key=lambda obj: int(obj["object_id"]))
    return candidates


def _candidate_pool_for_prompt(gt_row: dict, hview: dict) -> list[dict]:
    task = _normalized_lower(gt_row.get("task"))
    if task == "c":
        return _c_semantic_candidate_pool(gt_row, hview)
    return _category_candidate_pool(gt_row, hview)


def _human_visible_natural_unique_candidate(*, target_id: int, candidate_pool: list[dict]) -> bool:
    return len(candidate_pool) == 1 and int(candidate_pool[0]["object_id"]) == target_id


def _near_requires_strict_margin(gt_row: dict) -> bool:
    task = _normalized_lower(gt_row.get("task"))
    if task != "c":
        return True
    family = _normalized_lower(gt_row.get("intent_family")) or _normalized_lower(gt_row.get("same_type_key"))
    return family in {"cooking_appliance"}


def _near_human_visual_rejection_reason(
    *,
    gt_row: dict,
    target_id: int,
    candidate_pool: list[dict],
    anchor_objs: list[dict],
) -> str | None:
    if not anchor_objs or len(candidate_pool) <= 1:
        return None
    strict_margin = _near_requires_strict_margin(gt_row)
    candidate_distances = _distances_to_anchor(candidate_pool, anchor_objs)
    target_anchor_distance = candidate_distances.get(target_id)
    ordered = sorted(candidate_distances.items(), key=lambda item: (item[1], item[0]))
    if not ordered or target_anchor_distance is None:
        return None
    if ordered[0][0] != target_id:
        return "near_not_human_visual_unique"
    if target_anchor_distance > NEAR_MAX_DISTANCE:
        return "near_not_human_visual_unique"
    if strict_margin and len(ordered) > 1 and ordered[1][1] - ordered[0][1] < NEAR_MIN_MARGIN:
        return "near_not_human_visual_unique"
    if strict_margin and sum(1 for _, distance in ordered if distance <= target_anchor_distance + NEAR_CLUSTER_PAD) > 1:
        return "near_not_human_visual_unique"
    return None


def _relation_layout_consistent(*, relation_type: str | None, target_obj: dict, anchor_obj: dict) -> bool:
    target_x, target_y = _screen_xy(target_obj)
    anchor_x, anchor_y = _screen_xy(anchor_obj)
    if None in (target_x, target_y, anchor_x, anchor_y):
        return False
    if relation_type in {"under", "on"} and abs(target_x - anchor_x) > 0.25:
        return False
    if relation_type == "under":
        return target_y < anchor_y - 0.03
    if relation_type == "on":
        return target_y > anchor_y + 0.03
    return True


def _select_relation_residual_cue(
    *,
    gt_row: dict,
    target_id: int,
    relation_type: str | None,
    candidate_pool: list[dict],
    anchor_objs: list[dict],
) -> tuple[str | None, str | None]:
    if _normalized_lower(gt_row.get("task")) != "c":
        return None, None
    if _normalized_lower(relation_type) != "near":
        return None, None
    if len(candidate_pool) <= 1 or not anchor_objs:
        return None, None

    allowed = set(_c_allowed_residual_cues(gt_row))
    candidate_distances = _distances_to_anchor(candidate_pool, anchor_objs)
    ordered_distances = sorted(candidate_distances.items(), key=lambda item: (item[1], item[0]))
    ordered_x = sorted(
        (
            float(x),
            int(obj["object_id"]),
        )
        for obj in candidate_pool
        if (x := _screen_x(obj)) is not None
    )
    if not ordered_distances:
        return None, None

    distance_risky = False
    target_anchor_distance = candidate_distances.get(target_id)
    if target_anchor_distance is not None and ordered_distances and ordered_distances[0][0] == target_id:
        second_margin = None
        if len(ordered_distances) > 1:
            second_margin = ordered_distances[1][1] - ordered_distances[0][1]
        local_cluster = sum(
            1 for _, distance in ordered_distances if distance <= target_anchor_distance + NEAR_CLUSTER_PAD
        )
        distance_risky = (second_margin is not None and second_margin < NEAR_MIN_MARGIN) or local_cluster > 1

    cue_priority = ("side", "ordinal", "distance") if distance_risky else ("distance", "side", "ordinal")

    def _distance_cue() -> tuple[str | None, str | None]:
        if "distance" not in allowed:
            return None, None
        ids = [candidate_id for candidate_id, _ in ordered_distances]
        if target_id not in ids:
            return None, None
        index = ids.index(target_id)
        if len(ids) == 2:
            margin = abs(ordered_distances[1][1] - ordered_distances[0][1])
            if margin < NEAR_MIN_MARGIN:
                return None, None
            return "distance", ("nearer" if index == 0 else "farther")
        if index == 0:
            margin = ordered_distances[1][1] - ordered_distances[0][1] if len(ordered_distances) > 1 else 999.0
            if margin >= NEAR_MIN_MARGIN:
                return "distance", "closest"
        if index == len(ids) - 1 and len(ids) > 1:
            margin = ordered_distances[-1][1] - ordered_distances[-2][1]
            if margin >= NEAR_MIN_MARGIN:
                return "distance", "farthest"
        return None, None

    def _side_cue() -> tuple[str | None, str | None]:
        if "side" not in allowed or len(ordered_x) < 2:
            return None, None
        ids = [candidate_id for _, candidate_id in ordered_x]
        xs = [x for x, _ in ordered_x]
        if ids[0] == target_id and xs[1] - xs[0] >= SIDE_MIN_GAP_X:
            return "side", "left"
        if ids[-1] == target_id and xs[-1] - xs[-2] >= SIDE_MIN_GAP_X:
            return "side", "right"
        return None, None

    def _ordinal_cue() -> tuple[str | None, str | None]:
        if "ordinal" not in allowed or len(ordered_x) < 2:
            return None, None
        ids = [candidate_id for _, candidate_id in ordered_x]
        xs = [x for x, _ in ordered_x]
        if ids[0] == target_id and xs[1] - xs[0] >= ORDINAL_MIN_GAP_X:
            return "ordinal", "first"
        if ids[1] == target_id and xs[1] - xs[0] >= ORDINAL_MIN_GAP_X:
            return "ordinal", "second"
        return None, None

    builders = {
        "distance": _distance_cue,
        "side": _side_cue,
        "ordinal": _ordinal_cue,
    }
    for cue_name in cue_priority:
        cue_type, cue_value = builders[cue_name]()
        if cue_type is not None and cue_value is not None:
            return cue_type, cue_value
    return None, None


def _relation_match_count_in_pool(
    *,
    relation: dict,
    projected_relations: list[dict],
    candidate_pool_ids: set[int],
) -> int:
    relation_type = _normalized_lower(relation.get("relation_type"))
    anchor_category = _relation_anchor_category(relation)
    anchor_ids = tuple(sorted(int(v) for v in list(relation.get("anchor_ids") or [])))
    match_count = 0
    for candidate in projected_relations:
        if candidate.get("projection_status") != "usable":
            continue
        candidate_target = candidate.get("target_id")
        if candidate_target is None or int(candidate_target) not in candidate_pool_ids:
            continue
        if _normalized_lower(candidate.get("relation_type")) != relation_type:
            continue
        candidate_anchor_category = _relation_anchor_category(candidate)
        if anchor_category is not None and candidate_anchor_category == anchor_category:
            match_count += 1
            continue
        candidate_anchor_ids = tuple(sorted(int(v) for v in list(candidate.get("anchor_ids") or [])))
        if anchor_category is None and candidate_anchor_ids == anchor_ids:
            match_count += 1
    return match_count


def build_resolved_cue_package(
    *,
    gt_row: dict,
    hview_confusable_universe: dict,
    view_relation_projection: dict,
) -> tuple[dict, dict]:
    question_id = str(gt_row["question_id"])
    target_id = int(gt_row["target_id"])
    target_category = _normalized(gt_row.get("canonical_category")) or _normalized(gt_row.get("target_category"))
    target_color = _normalized(gt_row.get("target_color_name")) or _normalized(gt_row.get("cue_value"))
    target_rule = _normalized(gt_row.get("target_rule"))
    projected_relations = list(view_relation_projection.get("projected_relations") or [])
    target_relations = [
        row for row in projected_relations if int(row.get("target_id", -1)) == target_id
    ]
    candidates: list[dict] = []

    target_supported = _target_supported(hview_confusable_universe, target_id)
    effective_universe = _effective_universe(hview_confusable_universe, target_id)
    candidate_pool = _candidate_pool_for_prompt(gt_row, hview_confusable_universe)
    candidate_pool_ids = {int(obj["object_id"]) for obj in candidate_pool} | {int(v) for v in effective_universe}
    objects_by_id = _objects_by_id(hview_confusable_universe)
    prompt_answer_leakage = bool(gt_row.get("prompt_answer_leakage", False))

    if target_supported and (
        effective_universe == [target_id]
        or _human_visible_natural_unique_candidate(target_id=target_id, candidate_pool=candidate_pool)
    ):
        candidates.append(
            _build_candidate(
                cue_family="natural_unique",
                uniqueness_pass=True,
                cue_count=0,
                anchor_count=0,
                naturalness_score=100,
                answer_leakage_risk=0,
                cue_type="none",
            )
        )

    usable_relations = [row for row in target_relations if row.get("projection_status") == "usable"]
    usable_relations.sort(
        key=lambda row: (
            _relation_type_priority(row.get("relation_type")),
            -float(row.get("relation_stability_margin", 0.0)),
            str(_normalized_lower(row.get("relation_type")) or ""),
            tuple(int(v) for v in row.get("anchor_ids", [])),
        )
    )
    relation_not_unique_under_hview = False
    relation_generic_anchor_present = False
    relation_near_not_human_visual_unique = False
    for relation in usable_relations:
        cue_match_count = _relation_match_count_in_pool(
            relation=relation,
            projected_relations=projected_relations,
            candidate_pool_ids=candidate_pool_ids,
        )
        relation_unique = cue_match_count == 1
        anchor_category = _relation_anchor_category(relation)
        anchor_ids = [int(v) for v in list(relation.get("anchor_ids") or [])]
        anchor_objs = [objects_by_id[anchor_id] for anchor_id in anchor_ids if anchor_id in objects_by_id]
        is_generic_anchor = _is_generic_anchor_category(anchor_category)
        if is_generic_anchor:
            relation_generic_anchor_present = True
        relation_type = _normalized_lower(relation.get("relation_type"))
        relation_type_priority = _relation_type_priority(relation_type)
        relation_naturalness = 70 if is_generic_anchor else (95 if relation_type_priority == 0 else 90)
        combo_relation_naturalness = 60 if is_generic_anchor else (85 if relation_type_priority == 0 else 80)
        relation_stability_margin = float(relation.get("relation_stability_margin", 0.0))
        near_rejection_reason = None
        layout_rejection_reason = None
        if _normalized_lower(gt_row.get("task")) == "c" and relation_type == "inside":
            relation_rejection_reason = "inside_not_allowed_for_c"
        else:
            relation_rejection_reason = None
        if relation_rejection_reason is None and relation_type == "near":
            near_rejection_reason = _near_human_visual_rejection_reason(
                gt_row=gt_row,
                target_id=target_id,
                candidate_pool=candidate_pool,
                anchor_objs=anchor_objs,
            )
            if near_rejection_reason is not None:
                relation_near_not_human_visual_unique = True
        elif relation_rejection_reason is None and relation_type in {"under", "on"} and anchor_objs:
            target_obj = objects_by_id.get(target_id)
            if target_obj is not None and not _relation_layout_consistent(
                relation_type=relation_type,
                target_obj=target_obj,
                anchor_obj=anchor_objs[0],
            ):
                layout_rejection_reason = f"{relation_type}_layout_inconsistent"
        if not relation_unique:
            relation_not_unique_under_hview = True
        if relation_rejection_reason is None and is_generic_anchor:
            relation_rejection_reason = "generic_anchor_category"
        elif relation_rejection_reason is None and near_rejection_reason is not None:
            relation_rejection_reason = near_rejection_reason
        elif relation_rejection_reason is None and layout_rejection_reason is not None:
            relation_rejection_reason = layout_rejection_reason
        elif relation_rejection_reason is None and not relation_unique:
            relation_rejection_reason = "relation_not_unique_under_hview"
        relation_candidate_unique = relation_rejection_reason is None
        residual_cue_type = None
        residual_cue_value = None
        minimal_residual_cue = False
        relation_cue_count = 1
        if _normalized_lower(gt_row.get("task")) == "c" and relation_type == "near" and len(candidate_pool) > 1:
            relation_cue_count = 2
        if relation_candidate_unique:
            residual_cue_type, residual_cue_value = _select_relation_residual_cue(
                gt_row=gt_row,
                target_id=target_id,
                relation_type=relation_type,
                candidate_pool=candidate_pool,
                anchor_objs=anchor_objs,
            )
            minimal_residual_cue = residual_cue_type is not None and residual_cue_value is not None
            if (
                _normalized_lower(gt_row.get("task")) == "c"
                and relation_type == "near"
                and len(candidate_pool) > 1
                and not minimal_residual_cue
            ):
                relation_rejection_reason = "near_requires_minimal_residual_cue"
                relation_candidate_unique = False
        candidates.append(
            _build_candidate(
                cue_family="relation",
                uniqueness_pass=relation_candidate_unique,
                cue_count=relation_cue_count,
                anchor_count=len(anchor_ids),
                naturalness_score=relation_naturalness,
                answer_leakage_risk=0,
                stability_score=relation_stability_margin,
                cue_type=relation_type,
                cue_value={
                    "relation_type": relation_type,
                    "anchor_category": anchor_category,
                    "cue_match_count": cue_match_count,
                },
                anchor_ids=anchor_ids,
                rejection_reason=relation_rejection_reason,
                relation_type_priority=relation_type_priority,
                residual_cue_type=residual_cue_type,
                residual_cue_value=residual_cue_value,
                minimal_residual_cue=minimal_residual_cue,
            )
        )
        if target_color is not None and bool(gt_row.get("has_unique_color")) and not prompt_answer_leakage:
            candidates.append(
                _build_candidate(
                    cue_family="relation + color",
                    uniqueness_pass=relation_candidate_unique,
                    cue_count=2,
                    anchor_count=len(anchor_ids),
                    naturalness_score=combo_relation_naturalness,
                    answer_leakage_risk=0,
                    stability_score=relation_stability_margin,
                    cue_type=relation_type,
                    cue_value={
                        "relation_type": relation_type,
                        "anchor_category": anchor_category,
                        "cue_match_count": cue_match_count,
                        "color": target_color,
                    },
                    anchor_ids=anchor_ids,
                    rejection_reason=relation_rejection_reason,
                    relation_type_priority=relation_type_priority,
                )
            )
        attribute_cue = gt_row.get("attribute_cue")
        if isinstance(attribute_cue, dict) and attribute_cue.get("value") is not None:
            candidates.append(
                _build_candidate(
                    cue_family="relation + attribute",
                    uniqueness_pass=relation_candidate_unique,
                    cue_count=2,
                    anchor_count=len(anchor_ids),
                    naturalness_score=combo_relation_naturalness,
                    answer_leakage_risk=0,
                    stability_score=relation_stability_margin,
                    cue_type=_normalized(attribute_cue.get("type")) or "attribute",
                    cue_value={
                        "relation_type": relation_type,
                        "anchor_category": anchor_category,
                        "cue_match_count": cue_match_count,
                        "attribute_type": _normalized(attribute_cue.get("type")) or "attribute",
                        "attribute_value": attribute_cue.get("value"),
                    },
                    anchor_ids=anchor_ids,
                    rejection_reason=relation_rejection_reason,
                    relation_type_priority=relation_type_priority,
                )
            )
        else:
            candidates.append(
                _build_candidate(
                    cue_family="relation + attribute",
                    uniqueness_pass=False,
                    cue_count=2,
                    anchor_count=len(anchor_ids),
                    naturalness_score=combo_relation_naturalness,
                    answer_leakage_risk=0,
                    stability_score=relation_stability_margin,
                    rejection_reason="attribute_source_missing_v1",
                    relation_type_priority=relation_type_priority,
                )
            )

    anchor_repair_relations = [
        row
        for row in target_relations
        if _normalized(row.get("rejected_projection_reason")) == "anchor_not_promptable_under_hview"
    ]
    if anchor_repair_relations:
        anchor_repair_cue = gt_row.get("anchor_repair_cue")
        if isinstance(anchor_repair_cue, dict) and anchor_repair_cue.get("value") is not None:
            relation = anchor_repair_relations[0]
            candidates.append(
                _build_candidate(
                    cue_family="relation + anchor_repair",
                    uniqueness_pass=True,
                    cue_count=2,
                    anchor_count=len(list(relation.get("anchor_ids") or [])),
                    naturalness_score=75,
                    answer_leakage_risk=0,
                    cue_type=_normalized(relation.get("relation_type")),
                    cue_value=anchor_repair_cue.get("value"),
                    anchor_ids=[int(v) for v in list(relation.get("anchor_ids") or [])],
                )
            )
        else:
            candidates.append(
                _build_candidate(
                    cue_family="relation + anchor_repair",
                    uniqueness_pass=False,
                    cue_count=2,
                    anchor_count=1,
                    naturalness_score=75,
                    answer_leakage_risk=0,
                    rejection_reason="anchor_repair_source_missing_v1",
                )
            )

    attribute_cue = gt_row.get("attribute_cue")
    if isinstance(attribute_cue, dict) and attribute_cue.get("value") is not None:
        candidates.append(
            _build_candidate(
                cue_family="attribute",
                uniqueness_pass=True,
                cue_count=1,
                anchor_count=0,
                naturalness_score=65,
                answer_leakage_risk=0,
                cue_type=_normalized(attribute_cue.get("type")) or "attribute",
                cue_value=attribute_cue.get("value"),
            )
        )

    if target_color is not None and bool(gt_row.get("has_unique_color")) and not prompt_answer_leakage:
        candidates.append(
            _build_candidate(
                cue_family="color",
                uniqueness_pass=True,
                cue_count=1,
                anchor_count=0,
                naturalness_score=55,
                answer_leakage_risk=0,
                cue_type="color",
                cue_value=target_color,
            )
        )

    if (
        bool(gt_row.get("has_unique_spatial"))
        and target_rule in {"nearest", "farthest"}
        and float(gt_row.get("reference_top2_gap_m", 0.0) or 0.0) >= STRICT_DISTANCE_MIN_GAP_M
        and not prompt_answer_leakage
    ):
        candidates.append(
            _build_candidate(
                cue_family="strict_distance",
                uniqueness_pass=True,
                cue_count=1,
                anchor_count=0,
                naturalness_score=40,
                answer_leakage_risk=0,
                cue_type="distance_rank",
                cue_value=target_rule,
            )
        )

    if not candidates:
        candidates.append(
            _build_candidate(
                cue_family="drop",
                uniqueness_pass=False,
                cue_count=0,
                anchor_count=0,
                naturalness_score=0,
                answer_leakage_risk=0,
                rejection_reason="no_conservative_unique_cue",
            )
        )

    selected = sorted(candidates, key=_candidate_sort_key)[0]
    if not selected["uniqueness_pass"]:
        selected = {
            **selected,
            "cue_family": "drop",
            "cue_type": None,
            "cue_value": None,
            "anchor_ids": [],
            "residual_cue_type": None,
            "residual_cue_value": None,
            "minimal_residual_cue": False,
            "rejection_reason": selected.get("rejection_reason") or "no_conservative_unique_cue",
        }

    selected_rejection = None if selected["cue_family"] != "drop" else (
        selected.get("rejection_reason") or "no_conservative_unique_cue"
    )
    resolved = {
        "schema_version": "resolved_cue_package_v1",
        "question_id": question_id,
        "cue_family": selected["cue_family"],
        "cue_type": selected.get("cue_type"),
        "cue_value": (
            selected.get("cue_value", {}).get("color")
            if isinstance(selected.get("cue_value"), dict) and selected["cue_family"] == "relation + color"
            else (
                selected.get("cue_value", {}).get("attribute_value")
                if isinstance(selected.get("cue_value"), dict) and selected["cue_family"] == "relation + attribute"
                else (
                    selected.get("cue_value", {}).get("relation_type")
                    if isinstance(selected.get("cue_value"), dict) and selected["cue_family"] in {"relation", "relation + anchor_repair"}
                    else selected.get("cue_value")
                )
            )
        ),
        "target_category": target_category,
        "target_cue_family": selected["cue_family"],
        "anchor_cue_family": "none" if not selected.get("anchor_ids") else selected["cue_family"],
        "anchor_ids": list(selected.get("anchor_ids") or []),
        "anchor_category": (
            selected.get("cue_value", {}).get("anchor_category")
            if isinstance(selected.get("cue_value"), dict)
            else None
        ),
        "cue_match_count": (
            int(selected.get("cue_value", {}).get("cue_match_count"))
            if isinstance(selected.get("cue_value"), dict) and selected.get("cue_value", {}).get("cue_match_count") is not None
            else (1 if selected["cue_family"] != "drop" else 0)
        ),
        "residual_cue_type": selected.get("residual_cue_type"),
        "residual_cue_value": selected.get("residual_cue_value"),
        "minimal_residual_cue": bool(selected.get("minimal_residual_cue")),
        "effective_uniqueness_witness": [target_id] if selected["cue_family"] != "drop" else [],
        "why_not_relation": (
            None
            if selected["cue_family"] in {"relation", "relation + color", "relation + attribute", "relation + anchor_repair"}
            else (
                "relation_not_unique_under_hview"
                if relation_not_unique_under_hview
                else (
                    "near_not_human_visual_unique"
                    if relation_near_not_human_visual_unique
                    else (
                        "generic_anchor_category"
                        if relation_generic_anchor_present
                        and any(
                            candidate.get("cue_family") == "relation"
                            and candidate.get("rejection_reason") == "generic_anchor_category"
                            for candidate in candidates
                        )
                        else (
                            _summarize_why_not_relation(target_relations)
                        )
                    )
                )
            )
        ),
        "drop_reason_code": selected_rejection,
    }

    chosen_vs_rejected = []
    for candidate in sorted(candidates, key=_candidate_sort_key):
        candidate_row = dict(candidate)
        candidate_row["selected"] = candidate_row == selected or (
            selected["cue_family"] == "drop" and candidate_row["cue_family"] == "drop"
        )
        chosen_vs_rejected.append(candidate_row)

    trace = {
        "schema_version": "cue_search_trace_v1",
        "question_id": question_id,
        "selected_candidate": {
            "cue_family": resolved["cue_family"],
            "cue_type": resolved.get("cue_type"),
            "cue_value": resolved.get("cue_value"),
            "anchor_ids": list(resolved.get("anchor_ids") or []),
            "anchor_category": resolved.get("anchor_category"),
            "cue_match_count": resolved.get("cue_match_count"),
        },
        "chosen_vs_rejected_cues": chosen_vs_rejected,
        "why_not_relation": resolved["why_not_relation"],
        "drop_reason_code": resolved["drop_reason_code"],
    }
    return resolved, trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Synthesize conservative cue package for a fixed GT row.")
    parser.add_argument("--gt-row", required=True)
    parser.add_argument("--hview-confusable-universe", required=True)
    parser.add_argument("--view-relation-projection", required=True)
    parser.add_argument("--resolved-cue-package-out", required=True)
    parser.add_argument("--cue-search-trace-out", required=True)
    args = parser.parse_args()

    resolved, trace = build_resolved_cue_package(
        gt_row=_load_json(Path(args.gt_row)),
        hview_confusable_universe=_load_json(Path(args.hview_confusable_universe)),
        view_relation_projection=_load_json(Path(args.view_relation_projection)),
    )
    resolved_path = Path(args.resolved_cue_package_out)
    trace_path = Path(args.cue_search_trace_out)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with open(resolved_path, "w") as f:
        json.dump(resolved, f, indent=2, ensure_ascii=True)
    with open(trace_path, "w") as f:
        json.dump(trace, f, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
