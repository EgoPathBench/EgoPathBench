#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from collections import Counter
from functools import lru_cache
from pathlib import Path


SUPPORT_STATUSES = ("supported", "uncertain", "contradicted")
SAME_TYPE_MODES = ("family_proxy", "canonical_category")
RISK_PATTERN_FLAGS = {
    "none",
    "zero_raw_visible_but_system_exists",
    "prior_manual_conflict",
}
ROOT_DIR = Path(__file__).resolve().parents[1]
HVIEW_CONFUSABLE_UNIVERSE_CONFIG_PATH = ROOT_DIR / "configs" / "hview_confusable_universe_v1.json"
SCHEMA_DIR = ROOT_DIR / "schemas"
SCHEMA_BUNDLE_FILENAMES = (
    "cue_search_trace_v1.schema.json",
    "h_view_confusable_universe_v1.schema.json",
    "prompt_resolution_trace_v1.schema.json",
    "resolved_cue_package_v1.schema.json",
    "scene_relation_graph_v1.schema.json",
    "view_relation_projection_v1.schema.json",
)


def _stable_json_dumps(payload: object) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _compute_hash(payload: object) -> str:
    return hashlib.sha256(_stable_json_dumps(payload).encode("utf-8")).hexdigest()


def _load_json_dict(path: Path, *, expected_schema_version: str | None = None) -> dict:
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a JSON object: {path}")
    if expected_schema_version is not None:
        actual = str(payload.get("schema_version") or "").strip()
        if actual != expected_schema_version:
            raise ValueError(
                f"config {path} expected schema_version={expected_schema_version!r}, got {actual!r}"
            )
    return payload


@lru_cache(maxsize=1)
def get_hview_confusable_universe_protocol() -> dict:
    return _load_json_dict(
        HVIEW_CONFUSABLE_UNIVERSE_CONFIG_PATH,
        expected_schema_version="hview_confusable_universe_v1",
    )


@lru_cache(maxsize=1)
def get_hview_confusable_universe_hash() -> str:
    return _compute_hash(get_hview_confusable_universe_protocol())


@lru_cache(maxsize=1)
def get_schema_bundle_hash() -> str:
    payload = {
        filename: _load_json_dict(SCHEMA_DIR / filename)
        for filename in SCHEMA_BUNDLE_FILENAMES
    }
    return _compute_hash(payload)


def _normalize(text: str | None) -> str | None:
    if text is None:
        return None
    value = str(text).strip()
    if not value:
        return None
    return value.lower()


def _parse_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _is_signed_screen_norm(value: float | None) -> bool:
    if value is None:
        return False
    return -1.0 <= value <= 1.0


def build_category_to_family_map(proxy_group_map: dict[str, list[str]]) -> dict[str, str]:
    category_to_family: dict[str, str] = {}
    for family, categories in proxy_group_map.items():
        norm_family = _normalize(family)
        if norm_family is None:
            continue
        for category in categories:
            norm_category = _normalize(category)
            if norm_category is None:
                continue
            previous = category_to_family.get(norm_category)
            if previous is not None and previous != norm_family:
                raise ValueError(
                    f"category {norm_category!r} mapped to multiple families: {previous!r} and {norm_family!r}"
                )
            category_to_family[norm_category] = norm_family
    return category_to_family


def family_for_category(category: str | None, category_to_family_map: dict[str, str]) -> str | None:
    norm = _normalize(category)
    if norm is None:
        return None
    return category_to_family_map.get(norm)


def _normalize_risk_flag(value: str | None, *, default: str = "none") -> str:
    norm = _normalize(value)
    if norm is None:
        return default
    return norm if norm in RISK_PATTERN_FLAGS else default


def load_proxy_group_map(path: Path) -> dict[str, list[str]]:
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"proxy group config must be a JSON object: {path}")
    groups: dict[str, list[str]] = {}
    for family, categories in raw.items():
        if not isinstance(family, str):
            raise ValueError(f"proxy group family must be a string: {family!r}")
        if not isinstance(categories, list) or not categories:
            raise ValueError(f"proxy group {family!r} must contain a non-empty category list")
        normalized_categories = []
        for category in categories:
            text = _normalize(category)
            if text is None:
                raise ValueError(f"proxy group {family!r} contains an empty category entry")
            normalized_categories.append(text)
        if len(set(normalized_categories)) != len(normalized_categories):
            raise ValueError(f"proxy group {family!r} contains duplicate categories")
        groups[family] = normalized_categories
    build_category_to_family_map(groups)
    return groups


def _same_type_key(
    canonical_category: str | None,
    category_to_family_map: dict[str, str],
    *,
    same_type_mode: str,
) -> str | None:
    norm = _normalize(canonical_category)
    if norm is None:
        return None
    if same_type_mode == "canonical_category":
        return norm
    if same_type_mode != "family_proxy":
        raise ValueError(f"unsupported same_type_mode={same_type_mode}")
    return category_to_family_map.get(norm, norm)


def _visibility_status(is_visible: bool, raycast_verified: bool, screen_x: float | None, screen_y: float | None) -> str:
    if not is_visible:
        return "not_reliable"
    # Render-side screen coordinates are normalized NDC in [-1, 1], not [0, 1].
    if raycast_verified and _is_signed_screen_norm(screen_x) and _is_signed_screen_norm(screen_y):
        return "visible"
    return "partial"


def _category_mapping_status(category_family: str | None, canonical_category: str | None, same_type_key: str | None) -> str:
    if canonical_category is None:
        return "unmapped"
    if category_family is not None:
        return "stable"
    if same_type_key == canonical_category:
        return "stable"
    return "unmapped"


def _support_status(visibility: str, category_mapping_status: str, risk_pattern_flag: str) -> str:
    if visibility == "visible" and category_mapping_status == "stable" and risk_pattern_flag == "none":
        return "supported"
    if risk_pattern_flag == "prior_manual_conflict":
        return "contradicted"
    if visibility == "partial" or category_mapping_status == "family_collapsed" or risk_pattern_flag == "zero_raw_visible_but_system_exists":
        return "uncertain"
    return "uncertain"


def _build_screen_anchor(screen_x: float | None, screen_y: float | None) -> dict:
    return {
        "type": "screen_anchor",
        "x": screen_x if screen_x is not None else None,
        "y": screen_y if screen_y is not None else None,
    }


def _group_object_ids_by_same_type_key(sorted_objects: list[dict], *, witness_only: bool) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = {}
    for obj in sorted_objects:
        same_type_key = obj.get("same_type_key")
        if not same_type_key:
            continue
        is_witness = (
            obj.get("support_status") == "supported" and obj.get("visibility_status") == "visible"
        )
        if witness_only and not is_witness:
            continue
        if not witness_only and is_witness:
            continue
        grouped.setdefault(str(same_type_key), []).append(int(obj["object_id"]))
    return grouped


def _human_visible_truth_ready(sorted_objects: list[dict], ambiguity_counts: dict[str, int]) -> bool:
    if not sorted_objects:
        return False

    supported_visible_keys = {
        str(obj["same_type_key"])
        for obj in sorted_objects
        if obj["support_status"] == "supported" and obj["visibility_status"] == "visible" and obj.get("same_type_key")
    }
    if not supported_visible_keys:
        return False
    if any(obj["support_status"] == "contradicted" for obj in sorted_objects):
        return False
    if any(count > 1 for count in ambiguity_counts.values()):
        return False

    for obj in sorted_objects:
        same_type_key = obj.get("same_type_key")
        if same_type_key not in supported_visible_keys:
            continue
        if obj["visibility_status"] == "partial":
            return False
        if obj["support_status"] == "uncertain":
            return False
    return True


def build_h_view(
    bundle: dict,
    *,
    routing_rows: list[dict] | None = None,
    proxy_group_map: dict[str, list[str]] | None = None,
    same_type_mode: str = "family_proxy",
) -> dict:
    if same_type_mode not in SAME_TYPE_MODES:
        raise ValueError(f"unsupported same_type_mode={same_type_mode}")
    hview_protocol = get_hview_confusable_universe_protocol()
    category_to_family_map = build_category_to_family_map(proxy_group_map or {})
    routing_rows = routing_rows or []
    routing_by_target: dict[int, dict] = {}
    for row in routing_rows:
        target_id = row.get("target_id")
        if target_id is None:
            continue
        try:
            key = int(target_id)
        except (TypeError, ValueError):
            continue
        routing_by_target[key] = row

    visible_objects = bundle.get("visible_objects") or []
    h_view_objects: dict[int, dict] = {}

    for visible in visible_objects:
        obj_id = visible.get("id")
        if obj_id is None:
            continue
        try:
            key = int(obj_id)
        except (TypeError, ValueError):
            continue
        canonical_category = _normalize(visible.get("canonical_category")) or _normalize(visible.get("category"))
        if canonical_category is None:
            continue
        screen_x = _parse_float(visible.get("screen_x"))
        screen_y = _parse_float(visible.get("screen_y"))
        raycast_verified = bool(visible.get("raycast_verified"))
        visibility = _visibility_status(True, raycast_verified, screen_x, screen_y)
        same_type_key = (
            _same_type_key(
                canonical_category,
                category_to_family_map,
                same_type_mode=same_type_mode,
            )
            or canonical_category
        )
        category_family = family_for_category(canonical_category, category_to_family_map)
        category_mapping_status = _category_mapping_status(category_family, canonical_category, same_type_key)
        risk_pattern_flag = _normalize_risk_flag(visible.get("risk_pattern_flag"), default="none")
        if key in routing_by_target and routing_by_target[key].get("target_id") is not None:
            evidence_sources = ["visible_objects", "routing_candidates"]
        else:
            evidence_sources = ["visible_objects"]
        support = _support_status(visibility, category_mapping_status, risk_pattern_flag)
        h_view_objects[key] = {
            "object_id": key,
            "canonical_category": canonical_category,
            "category_family": category_family,
            "visibility_status": visibility,
            "support_status": support,
            "category_mapping_status": category_mapping_status,
            "risk_pattern_flag": risk_pattern_flag,
            "same_type_key": same_type_key,
            "screen_box_or_anchor": _build_screen_anchor(screen_x, screen_y),
            "evidence_sources": evidence_sources,
        }

    for target_id, row in routing_by_target.items():
        if target_id in h_view_objects:
            continue
        canonical_category = _normalize(row.get("canonical_category")) or _normalize(row.get("target_category"))
        if canonical_category is None:
            continue
        visibility = _visibility_status(False, False, None, None)
        same_type_key = (
            _same_type_key(
                canonical_category,
                category_to_family_map,
                same_type_mode=same_type_mode,
            )
            or canonical_category
        )
        category_family = family_for_category(canonical_category, category_to_family_map)
        category_mapping_status = _category_mapping_status(category_family, canonical_category, same_type_key)
        risk_pattern_flag = _normalize_risk_flag(
            row.get("risk_pattern_flag"),
            default="zero_raw_visible_but_system_exists",
        )
        support = _support_status(visibility, category_mapping_status, risk_pattern_flag)
        h_view_objects[target_id] = {
            "object_id": target_id,
            "canonical_category": canonical_category,
            "category_family": category_family,
            "visibility_status": visibility,
            "support_status": support,
            "category_mapping_status": category_mapping_status,
            "risk_pattern_flag": risk_pattern_flag,
            "same_type_key": same_type_key,
            "screen_box_or_anchor": _build_screen_anchor(None, None),
            "evidence_sources": ["routing_candidates"],
        }

    sorted_objects = sorted(h_view_objects.values(), key=lambda obj: obj["object_id"])
    status_counts = Counter(obj["support_status"] for obj in sorted_objects)
    ambiguity_counts: dict[str, int] = {}
    for obj in sorted_objects:
        if obj["support_status"] != "supported" or obj["visibility_status"] != "visible":
            continue
        key = obj.get("same_type_key")
        if not key:
            continue
        ambiguity_counts[key] = ambiguity_counts.get(key, 0) + 1

    ready = _human_visible_truth_ready(sorted_objects, ambiguity_counts)
    support_summary = {status: int(status_counts.get(status, 0)) for status in SUPPORT_STATUSES}
    visibility_counts = Counter(obj["visibility_status"] for obj in sorted_objects)
    effective_uniqueness_witness_object_ids = sorted(
        int(obj["object_id"])
        for obj in sorted_objects
        if obj["support_status"] == "supported" and obj["visibility_status"] == "visible"
    )
    effective_uniqueness_witness_same_type_ids = _group_object_ids_by_same_type_key(
        sorted_objects,
        witness_only=True,
    )
    non_witness_same_type_ids = _group_object_ids_by_same_type_key(
        sorted_objects,
        witness_only=False,
    )
    if not sorted_objects:
        ambiguity_status = "empty"
    elif any(obj["support_status"] == "contradicted" for obj in sorted_objects):
        ambiguity_status = "contradicted"
    elif any(count > 1 for count in ambiguity_counts.values()):
        ambiguity_status = "ambiguous"
    elif ready:
        ambiguity_status = "unique"
    else:
        ambiguity_status = "uncertain"
    return {
        "schema_version": "h_view_v2",
        "protocol_version": "hview_uview_v2",
        "hview_confusable_universe_version": hview_protocol["schema_version"],
        "hview_confusable_universe_hash": get_hview_confusable_universe_hash(),
        "schema_bundle_hash": get_schema_bundle_hash(),
        "same_type_mode": same_type_mode,
        "objects": sorted_objects,
        "h_view_objects": sorted_objects,
        "support_status_counts": support_summary,
        "ambiguity_count_by_same_type_key": dict(ambiguity_counts),
        "ambiguity_status": ambiguity_status,
        "support_status_summary": support_summary,
        "uniqueness_witness_policy": {
            "supported_only": True,
            "allowed_support_statuses": ["supported"],
            "required_visibility_status": "visible",
        },
        "effective_uniqueness_witness_object_ids": effective_uniqueness_witness_object_ids,
        "effective_uniqueness_witness_same_type_ids": effective_uniqueness_witness_same_type_ids,
        "non_witness_same_type_ids": non_witness_same_type_ids,
        "visibility_status_counts": {
            "visible": int(visibility_counts.get("visible", 0)),
            "partial": int(visibility_counts.get("partial", 0)),
            "not_reliable": int(visibility_counts.get("not_reliable", 0)),
        },
        "human_visible_truth_ready": ready,
    }


def write_h_view_sidecar(path: Path, h_view: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(h_view, indent=2))
