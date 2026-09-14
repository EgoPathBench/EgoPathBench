from __future__ import annotations

import copy
import hashlib
import json
from functools import lru_cache
from pathlib import Path


QUESTION_SOURCE_REBUILD_FROM_GT = "canonical_rebuild_from_gt_v1"
DEFAULT_HUMAN_VISIBLE_PROTOCOL_VERSION = "hview_uview_v2"
FIXED_TARGET_CONTRACT_VERSION = "fixed_gt_carrier_v2"
ROOT_DIR = Path(__file__).resolve().parents[1]
PROMPT_RESOLVER_CONTRACT_PATH = (
    ROOT_DIR / "configs" / "prompt_resolver_contract_v1.json"
)
SCHEMA_DIR = ROOT_DIR / "schemas"
SCHEMA_BUNDLE_FILENAMES = (
    "cue_search_trace_v1.schema.json",
    "h_view_confusable_universe_v1.schema.json",
    "prompt_resolution_trace_v1.schema.json",
    "resolved_cue_package_v1.schema.json",
    "scene_relation_graph_v1.schema.json",
    "view_relation_projection_v1.schema.json",
)
# The explicit H-view protocol is required for fixed-target routing contracts.
# C-main uses its own intent/slot protocol and does not currently materialize this field.
FORMAL_PROTOCOL_TASKS = frozenset({"a2", "b2"})

NAVIGATION_TRUTH_CONTRACT_KEYS = (
    "source_group_id",
    "human_visible_protocol_version",
    "human_visible_legality",
    "human_visible_sidecar_ref",
    "benchmark_unit_type",
    "immutable_identity",
    "fixed_target_contract_version",
    "fixed_gt_identity_hash",
    "fixed_gt_source_refs",
    "human_visible_support",
    "human_visible_uniqueness",
    "has_unique_color",
    "has_unique_spatial",
    "has_unique_relation_anchor",
    "human_visible_same_type_object_ids",
    "supported_visible_same_type_ids",
    "uncertain_same_type_ids",
    "fixed_target_supported_under_hview",
    "natural_unique_under_hview",
    "unique_by_relation",
    "unique_by_attribute",
    "unique_by_color",
    "unique_by_distance",
    "prompt_unique_under_hview",
    "promptability_status",
    "promptability_resolution_trace",
    "cue_family",
    "fixed_target_prompt_cue_bundle",
    "fingerprint_leakage",
    "prompt_answer_leakage",
    "task_family",
    "goal_anchor_id",
    "gt_path_id",
    "target_lock_id",
    "target_lock_category",
    "same_type_key",
    "ambiguity_count_H_view",
    "cue_type",
    "cue_value",
    "prompt_repair_status",
    "prompt_truth_tier",
    "target_lock_preserved",
    "task_family_preserved",
    "goal_anchor_preserved",
    "gt_path_preserved",
    "intent_family_preserved",
    "acceptable_goal_ids",
    "goal_ring_min_distance_m",
    "goal_ring_threshold_m",
    "reference_path_display_ids",
    "direct_pairs_ref",
    "route_semantics",
    "implicit_resolution_mode",
    "cue_repair_protocol_version",
    "cue_repair_protocol_hash",
    "hview_confusable_universe_version",
    "hview_confusable_universe_hash",
    "schema_bundle_hash",
)


def _stable_json_dumps(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _compute_hash(payload: object) -> str:
    return hashlib.sha256(_stable_json_dumps(payload).encode("utf-8")).hexdigest()


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalized_optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _load_json_dict(path: Path, *, expected_schema_version: str | None = None) -> dict:
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a JSON object: {path}")
    if expected_schema_version is not None:
        actual = _normalized_optional_text(payload.get("schema_version"))
        if actual != expected_schema_version:
            raise ValueError(
                f"config {path} expected schema_version={expected_schema_version!r}, got {actual!r}"
            )
    return payload


def _validate_prompt_resolver_contract(contract: dict) -> dict:
    required_lists = (
        "shared_resolver_tasks",
        "allowed_record_fields",
        "allowed_cue_bundle_fields",
        "allowed_trace_fields",
        "forbidden_record_fields",
    )
    for key in required_lists:
        value = contract.get(key)
        if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
            raise ValueError(f"prompt resolver contract field {key!r} must be a non-empty string list")
    overlap = set(contract["allowed_record_fields"]) & set(contract["forbidden_record_fields"])
    if overlap:
        raise ValueError(f"prompt resolver contract has overlapping allowed/forbidden fields: {sorted(overlap)}")
    return contract


def load_prompt_resolver_contract(path: Path = PROMPT_RESOLVER_CONTRACT_PATH) -> dict:
    contract = _load_json_dict(path, expected_schema_version="prompt_resolver_contract_v1")
    return _validate_prompt_resolver_contract(contract)


@lru_cache(maxsize=1)
def get_prompt_resolver_contract() -> dict:
    return load_prompt_resolver_contract()


@lru_cache(maxsize=1)
def get_schema_bundle_hash() -> str:
    payload = {
        filename: _load_json_dict(SCHEMA_DIR / filename)
        for filename in SCHEMA_BUNDLE_FILENAMES
    }
    return _compute_hash(payload)


def _sanitize_nested_allowed_mapping(
    value: object,
    allowed_fields: list[str],
    *,
    allow_legacy_sequence: bool = False,
) -> dict | list | None:
    if value is None:
        return None
    if allow_legacy_sequence and isinstance(value, list):
        return copy.deepcopy(value)
    if not isinstance(value, dict):
        raise ValueError("resolver-visible nested payload must be a dict when present")
    allowed = set(allowed_fields)
    sanitized: dict = {}
    for key, nested_value in value.items():
        if key in allowed:
            sanitized[key] = copy.deepcopy(nested_value)
    return sanitized


def build_resolver_visible_record(rec: dict) -> dict:
    contract = get_prompt_resolver_contract()
    allowed_record_fields = contract["allowed_record_fields"]
    forbidden_record_fields = set(contract["forbidden_record_fields"])
    visible: dict = {}
    for key in allowed_record_fields:
        if key in forbidden_record_fields:
            raise ValueError(f"resolver-visible field {key!r} overlaps forbidden_record_fields")
        if key not in rec:
            continue
        value = rec.get(key)
        if key == "fixed_target_prompt_cue_bundle":
            value = _sanitize_nested_allowed_mapping(value, contract["allowed_cue_bundle_fields"])
        elif key == "promptability_resolution_trace":
            value = _sanitize_nested_allowed_mapping(
                value,
                contract["allowed_trace_fields"],
                allow_legacy_sequence=True,
            )
        else:
            value = copy.deepcopy(value)
        visible[key] = value
    return visible


def _canonical_immutable_identity(rec: dict) -> dict:
    routing_id = _normalized_optional_text(rec.get("routing_id"))
    route_semantics = _normalized_optional_text(rec.get("route_semantics"))
    gt_path_id = _normalized_optional_text(rec.get("gt_path_id"))
    if gt_path_id is None and routing_id is not None and route_semantics is not None:
        gt_path_id = f"{routing_id}::{route_semantics}"

    return {
        "scene_id": rec.get("scene_id"),
        "view_id": _int_or_none(rec.get("view_id")),
        "routing_id": routing_id,
        "target_id": _int_or_none(rec.get("target_lock_id"))
        if rec.get("target_lock_id") is not None
        else _int_or_none(rec.get("target_id")),
        "goal_anchor_id": _int_or_none(rec.get("goal_anchor_id"))
        if rec.get("goal_anchor_id") is not None
        else _int_or_none(rec.get("goal_id")),
        "gt_path_id": gt_path_id,
    }


def derive_immutable_identity(rec: dict) -> dict:
    return _canonical_immutable_identity(rec)


def _normalize_supplied_immutable_identity(raw_identity: object) -> dict:
    if not isinstance(raw_identity, dict):
        raise ValueError("formal canonical contract requires immutable_identity to be a dict")
    return {
        "scene_id": raw_identity.get("scene_id"),
        "view_id": _int_or_none(raw_identity.get("view_id")),
        "routing_id": _normalized_optional_text(raw_identity.get("routing_id")),
        "target_id": _int_or_none(raw_identity.get("target_id")),
        "goal_anchor_id": _int_or_none(raw_identity.get("goal_anchor_id")),
        "gt_path_id": _normalized_optional_text(raw_identity.get("gt_path_id")),
    }


def _immutable_identity_for_record(rec: dict, *, formal_release: bool) -> dict:
    raw_identity = rec.get("immutable_identity")
    if not formal_release and isinstance(raw_identity, dict):
        return _normalize_supplied_immutable_identity(raw_identity)
    return _canonical_immutable_identity(rec)


def derive_fixed_gt_identity_hash_from_identity(
    immutable_identity: dict,
    *,
    fixed_target_contract_version: str = FIXED_TARGET_CONTRACT_VERSION,
) -> str:
    payload = {
        "fixed_target_contract_version": fixed_target_contract_version,
        "immutable_identity": _normalize_supplied_immutable_identity(immutable_identity),
    }
    return _compute_hash(payload)


def derive_fixed_gt_identity_hash(
    rec: dict,
    *,
    formal_release: bool = False,
    fixed_target_contract_version: str = FIXED_TARGET_CONTRACT_VERSION,
) -> str:
    immutable_identity = _immutable_identity_for_record(rec, formal_release=formal_release)
    return derive_fixed_gt_identity_hash_from_identity(
        immutable_identity,
        fixed_target_contract_version=fixed_target_contract_version,
    )


def _protocol_version_for_record(rec: dict, *, formal_release: bool) -> str:
    protocol_version = _normalized_optional_text(rec.get("human_visible_protocol_version"))
    task = _normalized_optional_text(rec.get("task"))
    if protocol_version is None:
        if formal_release and task in FORMAL_PROTOCOL_TASKS:
            raise ValueError(
                "formal canonical contract requires explicit human_visible_protocol_version "
                f"for task={task}"
            )
        return DEFAULT_HUMAN_VISIBLE_PROTOCOL_VERSION
    return protocol_version


def _validate_formal_contract_inputs(rec: dict, contract: dict) -> None:
    if "immutable_identity" in rec and rec.get("immutable_identity") is not None:
        supplied_identity = _normalize_supplied_immutable_identity(rec.get("immutable_identity"))
        if supplied_identity != contract["immutable_identity"]:
            raise ValueError("formal canonical contract immutable_identity mismatch")

    supplied_protocol = _normalized_optional_text(rec.get("protocol_version"))
    if supplied_protocol is not None and supplied_protocol != contract["protocol_version"]:
        raise ValueError("formal canonical contract protocol_version mismatch")

    supplied_question_source = _normalized_optional_text(rec.get("question_source"))
    if supplied_question_source is not None and supplied_question_source != contract["question_source"]:
        raise ValueError("formal canonical contract question_source mismatch")

    supplied_fixed_contract_version = _normalized_optional_text(rec.get("fixed_target_contract_version"))
    if (
        supplied_fixed_contract_version is not None
        and supplied_fixed_contract_version != contract["fixed_target_contract_version"]
    ):
        raise ValueError("formal canonical contract fixed_target_contract_version mismatch")

    supplied_fixed_gt_identity_hash = _normalized_optional_text(rec.get("fixed_gt_identity_hash"))
    if (
        supplied_fixed_gt_identity_hash is not None
        and supplied_fixed_gt_identity_hash != contract["fixed_gt_identity_hash"]
    ):
        raise ValueError("formal canonical contract fixed_gt_identity_hash mismatch")

    supplied_prompt_text = rec.get("prompt_text")
    if supplied_prompt_text is not None and str(supplied_prompt_text) != contract["prompt_text"]:
        raise ValueError("formal canonical contract prompt_text mismatch")

    for key in ("gt_hash", "prompt_hash"):
        supplied_hash = _normalized_optional_text(rec.get(key))
        if supplied_hash is not None and supplied_hash != contract[key]:
            raise ValueError(f"formal canonical contract {key} mismatch")

    for key in ("resolver_contract_version", "resolver_contract_hash", "resolver_visible_hash"):
        supplied_value = _normalized_optional_text(rec.get(key))
        if supplied_value is not None and supplied_value != contract[key]:
            raise ValueError(f"formal canonical contract {key} mismatch")


def extract_navigation_truth_contract_fields(rec: dict) -> dict:
    fields: dict = {}
    for key in NAVIGATION_TRUTH_CONTRACT_KEYS:
        if key in rec:
            fields[key] = copy.deepcopy(rec.get(key))
    return fields


def build_canonical_prompt_truth_contract(
    rec: dict,
    *,
    prompt: dict,
    answer: object,
    formal_release: bool = False,
) -> dict:
    immutable_identity = _immutable_identity_for_record(rec, formal_release=formal_release)
    fixed_target_contract_version = (
        _normalized_optional_text(rec.get("fixed_target_contract_version"))
        or FIXED_TARGET_CONTRACT_VERSION
    )
    fixed_gt_identity_hash = derive_fixed_gt_identity_hash_from_identity(
        immutable_identity,
        fixed_target_contract_version=fixed_target_contract_version,
    )
    protocol_version = _protocol_version_for_record(rec, formal_release=formal_release)
    question_source = QUESTION_SOURCE_REBUILD_FROM_GT
    prompt_text = str(prompt.get("user") or "")
    resolver_contract = get_prompt_resolver_contract()
    resolver_visible_record = build_resolver_visible_record(rec)
    implicit_resolution_mode = (
        _normalized_optional_text(rec.get("implicit_resolution_mode"))
        or _normalized_optional_text(resolver_contract.get("default_implicit_resolution_mode"))
        or "explicit_or_not_applicable"
    )

    gt_payload = {
        "task": rec.get("task"),
        "question_id": rec.get("question_id"),
        "scene_id": rec.get("scene_id"),
        "view_id": _int_or_none(rec.get("view_id")),
        "routing_id": _normalized_optional_text(rec.get("routing_id")),
        "start_id": _int_or_none(rec.get("start_id")),
        "goal_id": _int_or_none(rec.get("goal_id")),
        "goal_ids": copy.deepcopy(rec.get("goal_ids")),
        "path_ids": copy.deepcopy(rec.get("path_ids")),
        "walkable_ids": copy.deepcopy(rec.get("walkable_ids")),
        "target_id": _int_or_none(rec.get("target_id")),
        "target_lock_id": _int_or_none(rec.get("target_lock_id")),
        "goal_anchor_id": _int_or_none(rec.get("goal_anchor_id")),
        "gt_path_id": _normalized_optional_text(rec.get("gt_path_id")),
        "route_semantics": _normalized_optional_text(rec.get("route_semantics")),
        "facts_hash": _normalized_optional_text(rec.get("facts_hash")),
        "answer": copy.deepcopy(answer),
        "immutable_identity": immutable_identity,
    }
    prompt_payload = {
        "protocol_version": protocol_version,
        "question_source": question_source,
        "system_prompt": prompt.get("system"),
        "user_prompt": prompt_text,
        "cue_family": rec.get("cue_family"),
        "fixed_target_prompt_cue_bundle": copy.deepcopy(rec.get("fixed_target_prompt_cue_bundle")),
        "generated_question": rec.get("generated_question"),
    }
    contract = {
        "protocol_version": protocol_version,
        "immutable_identity": immutable_identity,
        "fixed_target_contract_version": fixed_target_contract_version,
        "fixed_gt_identity_hash": fixed_gt_identity_hash,
        "gt_hash": _compute_hash(gt_payload),
        "prompt_hash": _compute_hash(prompt_payload),
        "question_source": question_source,
        "prompt_text": prompt_text,
        "resolver_contract_version": resolver_contract["schema_version"],
        "resolver_contract_hash": _compute_hash(resolver_contract),
        "resolver_visible_hash": _compute_hash(resolver_visible_record),
        "schema_bundle_hash": get_schema_bundle_hash(),
        "implicit_resolution_mode": implicit_resolution_mode,
    }
    if formal_release:
        _validate_formal_contract_inputs(rec, contract)
    return contract
