#!/usr/bin/env python3
"""
Build Next VQA manifests from gt_next_*.jsonl.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_navigation_gt_next import (
    TRUTH_REPAIR_PROTOCOL_PATH,
    get_truth_repair_protocol as get_shared_truth_repair_protocol,
)
from prompt_truth_contract import (
    build_canonical_prompt_truth_contract,
    extract_navigation_truth_contract_fields,
)


SYSTEM_A1 = "You are a navigation perception assistant."
SYSTEM_A2 = "You are a navigation planning assistant."
SYSTEM_B2 = (
    "You are an embodied navigation robot. Plan a collision-free path for your "
    "body using the numbered candidate points shown in the image."
)
SYSTEM_C = (
    "You are an embodied navigation robot. A user gives you a short request that "
    "implies the kind of object they want. Infer the target family from the request, "
    "resolve the final grounded target in the scene, and plan a collision-free path "
    "for your body."
)

RULE_TEXT = {
    "nearest": "nearest",
    "farthest": "farthest",
    "leftmost": "leftmost",
}
DEFAULT_TASKS = ("a1", "b1", "a2", "b2", "c")
ALLOWED_TASKS = frozenset(DEFAULT_TASKS)
ALLOWED_PROMPT_REPAIR_STATUSES = {"not_needed", "repaired", "unresolved"}
ALLOWED_CUE_TYPES = {"none", "color", "distance_rank", "side", "ordinal", "near", "on", "inside", "under"}
ALLOWED_PROMPT_TRUTH_TIERS = {"benchmark_grade", "val_grade", "train_grade", "reject"}
ALLOWED_SPLITS = {"train", "val", "benchmark"}
ALLOWED_HUMAN_VISIBLE_UNIQUENESS = {"unique", "multiple", "uncertain", "ambiguous"}
ALLOWED_HUMAN_VISIBLE_LEGALITY = {"legal", "illegal", "uncertain"}
REQUIRED_ADJUDICATION_FIELDS = (
    "benchmark_sample_id",
    "adjudication_status",
    "reviewer_a",
    "reviewer_b",
    "resolved_uniqueness",
    "notes",
)
REQUIRED_PRESERVED_FLAGS = (
    "task_family_preserved",
    "goal_anchor_preserved",
    "gt_path_preserved",
)
REQUIRED_C_PRESERVED_FLAGS = REQUIRED_PRESERVED_FLAGS + ("intent_family_preserved",)
EXPLICIT_TARGET_CATEGORY_HINTS = (
    "chair",
    "sofa",
    "couch",
    "table",
    "cabinet",
    "dresser",
    "desk",
    "shelf",
    "bed",
    "toilet",
    "sink",
    "lamp",
    "stool",
    "monitor",
    "tv",
    "refrigerator",
)
EXPLICIT_TARGET_COLOR_HINTS = (
    "red",
    "blue",
    "green",
    "yellow",
    "orange",
    "purple",
    "pink",
    "brown",
    "black",
    "white",
    "gray",
    "grey",
)
RELATION_CUE_TYPES = {"near", "on", "inside", "under"}
_C_MAIN_ONTOLOGY_CACHE: dict[Path, dict] = {}
_C_PROXY_GROUPS_CACHE: dict[Path, dict] = {}


def get_truth_repair_protocol() -> dict:
    return get_shared_truth_repair_protocol(TRUTH_REPAIR_PROTOCOL_PATH)


def load_c_main_candidate_ontology(path: Path | None = None) -> dict:
    ontology_path = (path or (SCRIPT_DIR.parent / "configs" / "c_main_vnext_candidate_ontology.json")).resolve()
    cached = _C_MAIN_ONTOLOGY_CACHE.get(ontology_path)
    if cached is not None:
        return cached
    with open(ontology_path, "r") as f:
        ontology = json.load(f)
    families = ontology.get("families")
    if not isinstance(families, dict) or not families:
        raise ValueError(f"c ontology missing families: {ontology_path}")
    _C_MAIN_ONTOLOGY_CACHE[ontology_path] = ontology
    return ontology


def load_c_proxy_groups(path: Path | None = None) -> dict:
    proxy_path = (path or (SCRIPT_DIR.parent / "configs" / "c_intent_family_proxy_groups_v1.json")).resolve()
    cached = _C_PROXY_GROUPS_CACHE.get(proxy_path)
    if cached is not None:
        return cached
    with open(proxy_path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"c proxy groups missing or invalid: {proxy_path}")
    _C_PROXY_GROUPS_CACHE[proxy_path] = payload
    return payload


def require_key(rec: dict, key: str, *, context: str) -> object:
    if key not in rec or rec[key] is None:
        raise ValueError(f"{context} missing required field: {key}")
    return rec[key]


def _normalized_optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _normalized_split_name(split: str) -> str:
    normalized = str(split).strip().lower()
    if normalized not in ALLOWED_SPLITS:
        raise ValueError(f"unsupported split={split}; expected one of {sorted(ALLOWED_SPLITS)}")
    return normalized


def _lookup_approved_unique_adjudication(
    rec: dict,
    *,
    context: str,
    adjudication_by_sample_id: dict[str, dict] | None,
) -> dict:
    benchmark_sample_id = _normalized_optional_text(rec.get("benchmark_sample_id"))
    if benchmark_sample_id is None:
        raise ValueError(f"{context} requires benchmark_sample_id for uncertain uniqueness adjudication")
    if not adjudication_by_sample_id:
        raise ValueError(f"{context} requires approved adjudication for uncertain uniqueness")
    adjudication = adjudication_by_sample_id.get(benchmark_sample_id)
    if adjudication is None:
        raise ValueError(
            f"{context} requires approved adjudication for uncertain uniqueness; missing benchmark_sample_id={benchmark_sample_id}"
        )
    for key in REQUIRED_ADJUDICATION_FIELDS:
        value = _normalized_optional_text(adjudication.get(key))
        if value is None:
            raise ValueError(f"{context} adjudication missing required field: {key}")
    status = _normalized_optional_text(adjudication.get("adjudication_status"))
    resolved = _normalized_optional_text(adjudication.get("resolved_uniqueness"))
    if status != "approved" or resolved != "unique":
        raise ValueError(
            f"{context} requires approved adjudication with resolved_uniqueness=unique; got status={status} resolved={resolved}"
        )
    return adjudication


def load_uncertain_adjudication_sidecar(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    rows = load_jsonl(path)
    adjudication_by_sample_id: dict[str, dict] = {}
    for row in rows:
        row_context = f"adjudication sidecar row={row}"
        for key in REQUIRED_ADJUDICATION_FIELDS:
            value = _normalized_optional_text(row.get(key))
            if value is None:
                raise ValueError(f"{row_context} missing required adjudication field: {key}")
        sample_id = _normalized_optional_text(row.get("benchmark_sample_id"))
        if sample_id is None:
            raise ValueError(f"{row_context} missing required adjudication field: benchmark_sample_id")
        if sample_id in adjudication_by_sample_id:
            raise ValueError(f"duplicate adjudication benchmark_sample_id={sample_id}")
        normalized_row = dict(row)
        normalized_row["benchmark_sample_id"] = sample_id
        adjudication_by_sample_id[sample_id] = normalized_row
    return adjudication_by_sample_id


def _derive_and_require_preserved_flags(
    rec: dict,
    *,
    context: str,
    split_name: str,
    include_intent_family: bool = False,
) -> None:
    if split_name not in {"benchmark", "val"}:
        return

    if "task_family_preserved" not in rec or rec.get("task_family_preserved") is None:
        rec["task_family_preserved"] = bool(rec.get("target_lock_preserved"))
    if "goal_anchor_preserved" not in rec or rec.get("goal_anchor_preserved") is None:
        rec["goal_anchor_preserved"] = (
            rec.get("goal_anchor_id") is not None
            or rec.get("goal_id") is not None
            or bool(rec.get("target_lock_preserved"))
        )
    if "gt_path_preserved" not in rec or rec.get("gt_path_preserved") is None:
        rec["gt_path_preserved"] = (
            _normalized_optional_text(rec.get("gt_path_id")) is not None
            or bool(rec.get("path_ids"))
            or bool(rec.get("target_lock_preserved"))
        )
    if include_intent_family and ("intent_family_preserved" not in rec or rec.get("intent_family_preserved") is None):
        rec["intent_family_preserved"] = bool(_normalized_optional_text(rec.get("intent_family")))

    required_flags = REQUIRED_C_PRESERVED_FLAGS if include_intent_family else REQUIRED_PRESERVED_FLAGS
    for key in required_flags:
        if not bool(rec.get(key)):
            raise ValueError(f"{context} requires {key}=true for split={split_name}")


def validate_truth_repair_metadata_for_split(
    rec: dict,
    *,
    split: str,
    context: str,
    adjudication_by_sample_id: dict[str, dict] | None = None,
) -> None:
    validate_truth_repair_metadata(rec, context=context)
    split_name = _normalized_split_name(split)
    human_visible_uniqueness = _normalized_optional_text(rec.get("human_visible_uniqueness"))
    if human_visible_uniqueness is not None and human_visible_uniqueness not in ALLOWED_HUMAN_VISIBLE_UNIQUENESS:
        raise ValueError(
            f"{context} has unsupported human_visible_uniqueness={human_visible_uniqueness}"
        )
    human_visible_legality = _normalized_optional_text(rec.get("human_visible_legality"))
    if human_visible_legality is not None and human_visible_legality not in ALLOWED_HUMAN_VISIBLE_LEGALITY:
        raise ValueError(
            f"{context} has unsupported human_visible_legality={human_visible_legality}"
        )
    prompt_answer_leakage = rec.get("prompt_answer_leakage")
    if prompt_answer_leakage is not None and not isinstance(prompt_answer_leakage, bool):
        raise ValueError(f"{context} prompt_answer_leakage must be bool when provided")
    if split_name in {"benchmark", "val"}:
        if human_visible_legality is None:
            human_visible_legality = "legal"
        if human_visible_legality != "legal":
            raise ValueError(f"{context} requires human_visible_legality=legal for split={split_name}")
        if prompt_answer_leakage is not False:
            raise ValueError(f"{context} requires prompt_answer_leakage=False for split={split_name}")
        if not bool(rec.get("target_lock_preserved")):
            raise ValueError(f"{context} requires target_lock_preserved=true for split={split_name}")
        _derive_and_require_preserved_flags(rec, context=context, split_name=split_name, include_intent_family=False)
        if human_visible_uniqueness == "uncertain":
            _lookup_approved_unique_adjudication(
                rec,
                context=context,
                adjudication_by_sample_id=adjudication_by_sample_id,
            )
        elif human_visible_uniqueness != "unique":
            raise ValueError(f"{context} requires human_visible_uniqueness=unique for split={split_name}")
    elif human_visible_legality == "illegal":
        raise ValueError(f"{context} does not allow human_visible_legality=illegal for split={split_name}")


def validate_c_metadata_for_split(
    rec: dict,
    *,
    split: str,
    context: str,
    adjudication_by_sample_id: dict[str, dict] | None = None,
) -> None:
    split_name = _normalized_split_name(split)
    if split_name in {"benchmark", "val"}:
        if "target_lock_preserved" not in rec or rec.get("target_lock_preserved") is None:
            rec["target_lock_preserved"] = True
        if "human_visible_uniqueness" not in rec or rec.get("human_visible_uniqueness") is None:
            rec["human_visible_uniqueness"] = "unique"
        if "prompt_answer_leakage" not in rec or rec.get("prompt_answer_leakage") is None:
            rec["prompt_answer_leakage"] = False
    if split_name == "benchmark":
        rec.setdefault("slot_key", f"{rec['scene_id']}::{int(rec['view_id'])}::{rec['routing_id']}")
        if "slot_identity_locked" not in rec or rec.get("slot_identity_locked") is None:
            rec["slot_identity_locked"] = True
        if "slot_generation_mode" not in rec or rec.get("slot_generation_mode") is None:
            rec["slot_generation_mode"] = "exact_slot_prompt_regeneration"
    human_visible_uniqueness = _normalized_optional_text(rec.get("human_visible_uniqueness"))
    if human_visible_uniqueness is not None and human_visible_uniqueness not in ALLOWED_HUMAN_VISIBLE_UNIQUENESS:
        raise ValueError(
            f"{context} has unsupported human_visible_uniqueness={human_visible_uniqueness}"
        )
    human_visible_legality = _normalized_optional_text(rec.get("human_visible_legality"))
    if human_visible_legality is not None and human_visible_legality not in ALLOWED_HUMAN_VISIBLE_LEGALITY:
        raise ValueError(
            f"{context} has unsupported human_visible_legality={human_visible_legality}"
        )
    prompt_answer_leakage = rec.get("prompt_answer_leakage")
    if prompt_answer_leakage is not None and not isinstance(prompt_answer_leakage, bool):
        raise ValueError(f"{context} prompt_answer_leakage must be bool when provided")
    if split_name in {"benchmark", "val"}:
        if human_visible_legality is None:
            human_visible_legality = "legal"
        if human_visible_legality != "legal":
            raise ValueError(f"{context} requires human_visible_legality=legal for split={split_name}")
        if prompt_answer_leakage is not False:
            raise ValueError(f"{context} requires prompt_answer_leakage=False for split={split_name}")
        if not bool(rec.get("target_lock_preserved")):
            raise ValueError(f"{context} requires target_lock_preserved=true for split={split_name}")
        _derive_and_require_preserved_flags(rec, context=context, split_name=split_name, include_intent_family=True)
        if human_visible_uniqueness == "uncertain":
            _lookup_approved_unique_adjudication(
                rec,
                context=context,
                adjudication_by_sample_id=adjudication_by_sample_id,
            )
        elif human_visible_uniqueness != "unique":
            raise ValueError(f"{context} requires human_visible_uniqueness=unique for split={split_name}")
    elif human_visible_legality == "illegal":
        raise ValueError(f"{context} does not allow human_visible_legality=illegal for split={split_name}")
    if split_name == "benchmark":
        intent_family = _normalized_optional_text(require_key(rec, "intent_family", context=context))
        ontology = load_c_main_candidate_ontology()
        families = ontology.get("families") or {}
        proxy_groups = load_c_proxy_groups()
        allowed_families = {str(key).strip().lower() for key in families} | {
            str(key).strip().lower() for key in proxy_groups
        }
        if intent_family not in allowed_families:
            raise ValueError(f"{context} has unsupported intent_family={intent_family}")
        _validate_c_exact_slot_protocol(rec, context=context)


def _validate_c_exact_slot_protocol(rec: dict, *, context: str) -> None:
    slot_key = str(require_key(rec, "slot_key", context=context)).strip()
    expected_slot_key = f"{rec['scene_id']}::{int(rec['view_id'])}::{rec['routing_id']}"
    if slot_key != expected_slot_key:
        raise ValueError(f"{context} slot_key must equal exact slot identity {expected_slot_key}")
    if not bool(require_key(rec, "slot_identity_locked", context=context)):
        raise ValueError(f"{context} requires slot_identity_locked=true")
    generation_mode = _normalized_optional_text(require_key(rec, "slot_generation_mode", context=context))
    if generation_mode != "exact_slot_prompt_regeneration":
        raise ValueError(f"{context} requires slot_generation_mode=exact_slot_prompt_regeneration")
    scene_token = f"/{rec['scene_id']}/"
    view_token = f"/view_{int(rec['view_id'])}/"
    sidecar_expectations = {
        # Inventory is view-scoped rather than route-scoped, and historical
        # releases used different inventory basenames per c-main variant.
        "candidate_inventory_ref": lambda name: name.endswith("inventory.json"),
        "h_view_ref": lambda name: name == "h_view.json",
    }
    for key, filename_validator in sidecar_expectations.items():
        if key not in rec or rec.get(key) is None:
            continue
        ref = str(require_key(rec, key, context=context)).strip()
        basename = Path(ref).name
        if scene_token not in ref or view_token not in ref or not filename_validator(basename):
            raise ValueError(f"{context} {key} must point back to the same scene/view sidecar")


def validate_truth_repair_metadata(rec: dict, *, context: str) -> None:
    prompt_repair_status = _normalized_optional_text(
        require_key(rec, "prompt_repair_status", context=context)
    )
    if prompt_repair_status not in ALLOWED_PROMPT_REPAIR_STATUSES:
        raise ValueError(f"{context} has unsupported prompt_repair_status={prompt_repair_status}")

    for key in (
        "target_lock_id",
        "target_lock_category",
        "same_type_key",
        "ambiguity_count_H_view",
        "cue_type",
        "target_lock_preserved",
    ):
        require_key(rec, key, context=context)

    try:
        int(rec["target_lock_id"])
    except Exception as exc:
        raise ValueError(f"{context} target_lock_id must be int-like") from exc

    try:
        ambiguity_count = int(rec["ambiguity_count_H_view"])
    except Exception as exc:
        raise ValueError(f"{context} ambiguity_count_H_view must be int-like") from exc
    if ambiguity_count < 0:
        raise ValueError(f"{context} ambiguity_count_H_view must be >= 0")

    cue_type = _normalized_optional_text(rec.get("cue_type"))
    if cue_type not in ALLOWED_CUE_TYPES:
        raise ValueError(f"{context} has unsupported cue_type={cue_type}")

    if not bool(rec.get("target_lock_preserved")):
        raise ValueError(f"{context} requires target_lock_preserved=true")

    prompt_truth_tier = _normalized_optional_text(
        require_key(rec, "prompt_truth_tier", context=context)
    )
    if prompt_truth_tier not in ALLOWED_PROMPT_TRUTH_TIERS:
        raise ValueError(f"{context} has unsupported prompt_truth_tier={prompt_truth_tier}")
    if prompt_truth_tier == "reject":
        raise ValueError(f"{context} has prompt_truth_tier=reject; fail-closed")

    if prompt_repair_status == "repaired":
        require_key(rec, "cue_value", context=context)
        if cue_type == "none":
            raise ValueError(f"{context} repaired rows cannot use cue_type=none")
        if cue_type == "distance_rank":
            raise ValueError(f"{context} repaired rows cannot use cue_type=distance_rank yet")
    elif prompt_repair_status == "unresolved":
        raise ValueError(f"{context} has unresolved truth-repair status; fail-closed")
    else:
        if cue_type != "none":
            raise ValueError(f"{context} non-repaired rows must use cue_type=none")


def validate_record_for_task(
    rec: dict,
    task: str,
    *,
    split: str = "train",
    adjudication_by_sample_id: dict[str, dict] | None = None,
) -> None:
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
        for key in (
            "routing_id",
            "start_id",
            "goal_id",
            "goal_ids",
            "path_ids",
            "target_rule",
            "optimal_length_m",
            "canonical_sparse_length_m",
        ):
            require_key(rec, key, context=f"{task} record")
        validate_truth_repair_metadata_for_split(
            rec,
            split=split,
            context=f"{task} record",
            adjudication_by_sample_id=adjudication_by_sample_id,
        )
    elif task == "c":
        for key in (
            "routing_id",
            "start_id",
            "goal_id",
            "path_ids",
            "optimal_length_m",
            "canonical_sparse_length_m",
        ):
            require_key(rec, key, context=f"{task} record")
        validate_c_metadata_for_split(
            rec,
            split=split,
            context=f"{task} record",
            adjudication_by_sample_id=adjudication_by_sample_id,
        )


def build_target_phrase(rec: dict) -> str:
    category = rec.get("canonical_category") or rec.get("target_category")
    if not category:
        raise ValueError("routing prompt missing target category")
    prompt_repair_status = _normalized_optional_text(rec.get("prompt_repair_status"))
    cue_type = _normalized_optional_text(rec.get("cue_type"))
    if prompt_repair_status == "repaired" and cue_type == "color":
        cue_value = str(require_key(rec, "cue_value", context="truth-repaired routing prompt")).strip()
        if not cue_value:
            raise ValueError("truth-repaired routing prompt missing required field: cue_value")
        return f"{cue_value} {category}"
    return str(category)


def _relation_modifier(cue_type: str, anchor_category: str | None) -> str:
    if anchor_category is None:
        raise ValueError(
            f"truth-repaired routing prompt missing anchor_category for relation cue_type={cue_type}"
        )
    if cue_type == "near":
        return f"near the {anchor_category}"
    if cue_type == "on":
        return f"on the {anchor_category}"
    if cue_type == "inside":
        return f"inside the {anchor_category}"
    if cue_type == "under":
        return f"under the {anchor_category}"
    raise ValueError(f"truth-repaired routing prompt has unsupported relation cue_type={cue_type}")


def build_repaired_selection_phrase(rec: dict) -> str:
    prompt_repair_status = _normalized_optional_text(rec.get("prompt_repair_status"))
    if prompt_repair_status != "repaired":
        raise ValueError("build_repaired_selection_phrase requires prompt_repair_status=repaired")
    category = str(require_key(rec, "canonical_category", context="truth-repaired routing prompt")).strip()
    if not category:
        raise ValueError("truth-repaired routing prompt missing target category")
    cue_bundle = rec.get("fixed_target_prompt_cue_bundle") or {}
    cue_family = _normalized_optional_text(cue_bundle.get("cue_family") or rec.get("cue_family"))
    cue_type = _normalized_optional_text(cue_bundle.get("cue_type") or rec.get("cue_type"))
    raw_cue_value = cue_bundle.get("cue_value", rec.get("cue_value"))
    cue_value = _normalized_optional_text(raw_cue_value) if not isinstance(raw_cue_value, dict) else None

    if cue_family == "relation + color" and isinstance(raw_cue_value, dict):
        relation_type = _normalized_optional_text(raw_cue_value.get("relation_type")) or cue_type
        anchor_category = _normalized_optional_text(raw_cue_value.get("anchor_category")) or _normalized_optional_text(
            cue_bundle.get("anchor_category")
        )
        color = _normalized_optional_text(raw_cue_value.get("color"))
        if color is None:
            raise ValueError("truth-repaired routing prompt missing relation+color color")
        return f"{color} {category} {_relation_modifier(str(relation_type), anchor_category)}"

    if cue_family == "relation + attribute" and isinstance(raw_cue_value, dict):
        relation_type = _normalized_optional_text(raw_cue_value.get("relation_type")) or cue_type
        anchor_category = _normalized_optional_text(raw_cue_value.get("anchor_category")) or _normalized_optional_text(
            cue_bundle.get("anchor_category")
        )
        attribute_value = _normalized_optional_text(raw_cue_value.get("attribute_value"))
        if attribute_value is None:
            raise ValueError("truth-repaired routing prompt missing relation+attribute attribute_value")
        return f"{category} {_relation_modifier(str(relation_type), anchor_category)} with {attribute_value}"

    if cue_family in {"relation", "relation + anchor_repair"} or cue_type in RELATION_CUE_TYPES:
        anchor_category = _normalized_optional_text(cue_bundle.get("anchor_category"))
        if isinstance(raw_cue_value, dict):
            cue_type = _normalized_optional_text(raw_cue_value.get("relation_type")) or cue_type
            anchor_category = _normalized_optional_text(raw_cue_value.get("anchor_category")) or anchor_category
        return f"{category} {_relation_modifier(str(cue_type), anchor_category)}"

    if cue_family == "attribute":
        if cue_value is None:
            raise ValueError("truth-repaired routing prompt missing attribute cue_value")
        return f"{category} with {cue_value}"

    if cue_type == "color":
        if cue_value is None:
            raise ValueError("truth-repaired routing prompt missing color cue_value")
        return f"{cue_value} {category}"

    if cue_type == "side":
        if cue_value is None:
            raise ValueError("truth-repaired routing prompt missing side cue_value")
        side_label = {
            "left": "leftmost",
            "right": "rightmost",
        }.get(cue_value, cue_value)
        return f"{side_label} {category}"

    if cue_type == "ordinal":
        if cue_value is None:
            raise ValueError("truth-repaired routing prompt missing ordinal cue_value")
        return f"{cue_value} {category} from left to right"

    if cue_type == "distance_rank":
        if cue_value is None:
            raise ValueError("truth-repaired routing prompt missing distance cue_value")
        return f"{cue_value} {category}"

    raise ValueError(f"truth-repaired routing prompt has unsupported cue_type={cue_type}")


def build_routing_selection_phrase(rec: dict) -> str:
    prompt_repair_status = _normalized_optional_text(rec.get("prompt_repair_status"))
    if prompt_repair_status == "repaired":
        return build_repaired_selection_phrase(rec)
    rule = RULE_TEXT.get(rec.get("target_rule", "nearest"), "nearest")
    return f"{rule} {build_target_phrase(rec)}"


def build_repair_cue_sentence(rec: dict) -> str | None:
    prompt_repair_status = _normalized_optional_text(rec.get("prompt_repair_status"))
    if prompt_repair_status != "repaired":
        return None
    cue_type = _normalized_optional_text(rec.get("cue_type"))
    cue_value = str(require_key(rec, "cue_value", context="truth-repaired routing prompt")).strip()
    if not cue_value:
        raise ValueError("truth-repaired routing prompt missing required field: cue_value")
    category = str(require_key(rec, "canonical_category", context="truth-repaired routing prompt"))
    if cue_type == "color":
        return None
    if cue_type in RELATION_CUE_TYPES:
        cue_bundle = rec.get("fixed_target_prompt_cue_bundle") or {}
        anchor_category = _normalized_optional_text(cue_bundle.get("anchor_category"))
        return f"Among the visible {category}s, choose the one {_relation_modifier(cue_type, anchor_category)}."
    if cue_type == "distance_rank":
        return f"Among the visible {category}s, choose the {cue_value} one."
    if cue_type == "side":
        return f"Among the visible {category}s, choose the {cue_value} one."
    if cue_type == "ordinal":
        return f"Among the visible {category}s, choose the {cue_value} one from left to right."
    raise ValueError(f"truth-repaired routing prompt has unsupported non-color cue_type={cue_type}")


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


def parse_tasks(raw: str) -> tuple[str, ...]:
    tasks = tuple(task.strip() for task in raw.split(",") if task.strip())
    if not tasks:
        raise ValueError("no tasks requested")
    invalid = tuple(task for task in tasks if task not in ALLOWED_TASKS)
    if invalid:
        raise ValueError(
            f"invalid task(s): {', '.join(invalid)}; allowed tasks: {', '.join(DEFAULT_TASKS)}"
        )
    return tasks


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
    validate_truth_repair_metadata(rec, context="routing prompt")
    sys = SYSTEM_B2 if embodied else SYSTEM_A2
    selection_phrase = build_routing_selection_phrase(rec)
    start_id = require_key(rec, "start_id", context="routing prompt")
    if embodied:
        robot_diameter_m = float(require_key(rec, "robot_diameter_m", context="embodied routing prompt"))
        user = (
            f"The image shows numbered candidate points. You are the robot, and your body diameter is {robot_diameter_m:.1f} m. "
            f"Start at display ID {start_id}. Navigate to the {selection_phrase}. "
            "Return a JSON array of display IDs representing a valid collision-free path for the robot."
        )
    else:
        user = (
            f"The image shows numbered candidate points. "
            f"Start at display ID {start_id}. Navigate to the {selection_phrase}. "
            "Output a JSON array of display IDs representing a valid path."
        )
    return {"system": sys, "user": user}


def build_c_prompt(rec: dict) -> dict:
    start_id = require_key(rec, "start_id", context="c prompt")
    generated_question = str(require_key(rec, "generated_question", context="c prompt")).strip()
    if not generated_question:
        raise ValueError("c prompt missing required field: generated_question")
    generated_question_lc = generated_question.lower()
    cue_bundle = rec.get("fixed_target_prompt_cue_bundle") or {}
    anchor_category = _normalized_optional_text(cue_bundle.get("anchor_category"))
    target_categories = {
        _normalized_optional_text(rec.get("canonical_category")),
        _normalized_optional_text(rec.get("target_category")),
        _normalized_optional_text(rec.get("target_lock_category")),
    }
    banned_categories = {
        category
        for category in EXPLICIT_TARGET_CATEGORY_HINTS
        if _normalized_optional_text(category) not in {None, anchor_category}
    }
    if anchor_category is not None and anchor_category in target_categories:
        banned_categories.add(anchor_category)
    if (
        re.search(r"\b(display\s*id|target\s*id|object\s*id|id)\s*[:#-]?\s*\d+\b", generated_question_lc)
        or re.search(r"\b(?:#)\s*\d+\b", generated_question_lc)
        or any(f"{color} {category}" in generated_question_lc for color in EXPLICIT_TARGET_COLOR_HINTS for category in banned_categories)
        or any(re.search(rf"\b{re.escape(category)}s?\b", generated_question_lc) for category in banned_categories)
    ):
        raise ValueError("c prompt generated_question must not explicitly name final target category or instance")
    require_key(rec, "intent_family", context="c prompt")
    route_source_task = str(require_key(rec, "route_source_task", context="c prompt")).strip().lower()
    if route_source_task != "b2":
        raise ValueError("c prompt requires route_source_task=b2")
    robot_diameter_m = float(require_key(rec, "robot_diameter_m", context="c prompt"))
    route_semantics = str(rec.get("route_semantics") or "embodied").strip().lower()
    if route_semantics != "embodied":
        raise ValueError("c prompt requires route_semantics=embodied")
    user = (
        f"The image shows numbered candidate points. You are the robot, and your body diameter is {robot_diameter_m:.1f} m. "
        f"Start at display ID {start_id}. A person says: \"{generated_question}\" "
        "Return a JSON array of display IDs representing a valid collision-free path to the resolved target."
    )
    return {"system": SYSTEM_C, "user": user}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Next VQA from gt_next jsonl files.")
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--split", default="train", choices=sorted(ALLOWED_SPLITS))
    parser.add_argument("--uncertain-adjudication-jsonl", default=None)
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_tasks = set(parse_tasks(args.tasks))
    split_name = _normalized_split_name(args.split)
    adjudication_by_sample_id = load_uncertain_adjudication_sidecar(
        Path(args.uncertain_adjudication_jsonl).resolve()
        if args.uncertain_adjudication_jsonl
        else None
    )

    tasks = ["a1", "b1", "a2", "b2"]
    for task in tasks:
        if task not in selected_tasks:
            continue
        gt_path = gt_dir / f"gt_next_{task}.jsonl"
        records = load_jsonl(gt_path)
        if not records:
            continue

        out_path = out_dir / f"vqa_next_{task}.jsonl"
        with open(out_path, "w") as f:
            for rec in records:
                validate_record_for_task(
                    rec,
                    task,
                    split=split_name,
                    adjudication_by_sample_id=adjudication_by_sample_id,
                )
                if task in ("a1", "b1"):
                    prompt = build_a1_prompt(rec, embodied=(task == "b1"))
                    answer = rec["walkable_ids"]
                else:
                    prompt = build_a2_prompt(rec, embodied=(task == "b2"))
                    answer = rec["path_ids"]
                contract = build_canonical_prompt_truth_contract(
                    rec,
                    prompt=prompt,
                    answer=answer,
                )

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
                    "question_source": contract["question_source"],
                    "prompt_text": contract["prompt_text"],
                    "protocol_version": contract["protocol_version"],
                    "immutable_identity": contract["immutable_identity"],
                    "fixed_target_contract_version": contract["fixed_target_contract_version"],
                    "fixed_gt_identity_hash": contract["fixed_gt_identity_hash"],
                    "gt_hash": contract["gt_hash"],
                    "prompt_hash": contract["prompt_hash"],
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
                        "reference_family_id": rec.get("reference_family_id"),
                        "target_color_name": rec.get("target_color_name"),
                        "target_id": rec.get("target_id"),
                        "target_distance_m": rec.get("target_distance_m"),
                        "optimal_length_m": rec.get("optimal_length_m"),
                        "canonical_sparse_length_m": rec.get("canonical_sparse_length_m"),
                        "instruction_identity": rec.get("instruction_identity"),
                        "instruction_selection": rec.get("instruction_selection"),
                        "routing_complexity": rec.get("routing_complexity"),
                        "navigation_reference_facts": rec.get("navigation_reference_facts"),
                        "navigation_geometry_facts": rec.get("navigation_geometry_facts"),
                        "navigation_embodiment_facts": rec.get("navigation_embodiment_facts"),
                        "navigation_axes": rec.get("navigation_axes"),
                        "navigation_validity": rec.get("navigation_validity"),
                        "affordance_axes": rec.get("affordance_axes"),
                        "affordance_validity": rec.get("affordance_validity"),
                        "facts_hash": rec.get("facts_hash"),
                        "robot_diameter_m": rec.get("robot_diameter_m"),
                        "target_lock_id": rec.get("target_lock_id"),
                        "target_lock_category": rec.get("target_lock_category"),
                        "same_type_key": rec.get("same_type_key"),
                        "ambiguity_count_H_view": rec.get("ambiguity_count_H_view"),
                        "cue_type": rec.get("cue_type"),
                        "cue_value": rec.get("cue_value"),
                        "prompt_repair_status": rec.get("prompt_repair_status"),
                        "prompt_truth_tier": rec.get("prompt_truth_tier"),
                        "target_lock_preserved": rec.get("target_lock_preserved"),
                        **extract_navigation_truth_contract_fields(rec),
                        **contract,
                    },
                }
                f.write(json.dumps(vqa, ensure_ascii=True) + "\n")

        print(f"Wrote {out_path}")

    c_path = gt_dir / "gt_next_c.jsonl"
    if "c" in selected_tasks and c_path.exists():
        out_c = out_dir / "vqa_next_c.jsonl"
        with open(out_c, "w") as f:
            for rec in load_jsonl(c_path):
                validate_record_for_task(
                    rec,
                    "c",
                    split=split_name,
                    adjudication_by_sample_id=adjudication_by_sample_id,
                )
                prompt = build_c_prompt(rec)
                contract = build_canonical_prompt_truth_contract(
                    rec,
                    prompt=prompt,
                    answer=rec["path_ids"],
                )
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
                    "question_source": contract["question_source"],
                    "prompt_text": contract["prompt_text"],
                    "protocol_version": contract["protocol_version"],
                    "immutable_identity": contract["immutable_identity"],
                    "fixed_target_contract_version": contract["fixed_target_contract_version"],
                    "fixed_gt_identity_hash": contract["fixed_gt_identity_hash"],
                    "gt_hash": contract["gt_hash"],
                    "prompt_hash": contract["prompt_hash"],
                    "question_text": rec.get("generated_question"),
                    "ground_truth": {
                        "answer": rec["path_ids"],
                        "start_id": rec.get("start_id"),
                        "goal_id": rec.get("goal_id"),
                        "goal_ids": rec.get("goal_ids"),
                        "path_ids": rec.get("path_ids"),
                        "optimal_length_m": rec.get("optimal_length_m"),
                        "canonical_sparse_length_m": rec.get("canonical_sparse_length_m"),
                        "target_id": rec.get("target_id"),
                        "target_category": rec.get("target_category"),
                        "canonical_category": rec.get("canonical_category"),
                        "semantic_group_id": rec.get("semantic_group_id"),
                        "reference_family_id": rec.get("reference_family_id"),
                        "target_color_name": rec.get("target_color_name"),
                        "instruction_identity": rec.get("instruction_identity"),
                        "instruction_selection": rec.get("instruction_selection"),
                        "routing_complexity": rec.get("routing_complexity"),
                        "navigation_reference_facts": rec.get("navigation_reference_facts"),
                        "navigation_geometry_facts": rec.get("navigation_geometry_facts"),
                        "navigation_embodiment_facts": rec.get("navigation_embodiment_facts"),
                        "navigation_axes": rec.get("navigation_axes"),
                        "navigation_validity": rec.get("navigation_validity"),
                        "facts_hash": rec.get("facts_hash"),
                        "generated_question": rec.get("generated_question"),
                        "label": rec.get("label"),
                        "intent_family": rec.get("intent_family"),
                        "intent_strength": rec.get("intent_strength"),
                        "residual_cue_type": rec.get("residual_cue_type"),
                        "resolved_goal_display_id": rec.get("resolved_goal_display_id"),
                        "route_source_task": rec.get("route_source_task"),
                        "route_semantics": rec.get("route_semantics"),
                        "candidate_inventory_ref": rec.get("candidate_inventory_ref"),
                        "robot_diameter_m": rec.get("robot_diameter_m"),
                        **extract_navigation_truth_contract_fields(rec),
                        **contract,
                    },
                }
                f.write(json.dumps(vqa, ensure_ascii=True) + "\n")
        print(f"Wrote {out_c}")


if __name__ == "__main__":
    main()
