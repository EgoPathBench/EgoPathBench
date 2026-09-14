#!/usr/bin/env python3
"""
Deterministically revalidate prompt-side uniqueness under H(view).

This resolver is intentionally conservative:
- it only consumes resolver-visible record fields plus H(view)
- it never reads direct GT identity fields to make the decision
- it fail-closes when surface form or H(view) support is insufficient
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

from prompt_truth_contract import build_resolver_visible_record, get_prompt_resolver_contract


ROOT_DIR = Path(__file__).resolve().parents[1]
CUE_REPAIR_PROTOCOL_PATH = ROOT_DIR / "configs" / "cue_repair_protocol_v1.json"
RELATION_CUE_TYPES = {"near", "on", "inside", "under"}
RELATION_SURFACE_PHRASES = {
    "near": ("near", "next to", "beside", "by", "close to"),
    "on": ("on", "on top of"),
    "inside": ("inside", "in"),
    "under": ("under", "beneath"),
}
STRICT_RELATION_SURFACE_PHRASES = {
    "near": ("near", "next to", "beside", "by", "close to"),
    "on": ("on top of", " on "),
    "inside": ("inside",),
    "under": ("under", "beneath"),
}
RESIDUAL_SURFACE_PHRASES = {
    "distance": {
        "nearer": ("nearer", "closer"),
        "farther": ("farther", "further"),
        "closest": ("closest",),
        "farthest": ("farthest", "furthest"),
    },
    "side": {
        "left": ("leftmost", "on the left"),
        "right": ("rightmost", "on the right"),
    },
    "ordinal": {
        "first": ("first",),
        "second": ("second",),
    },
}
IMPLICIT_ALLOWED_MODES = {
    "explicit_or_not_applicable",
    "implicit_only_pass",
    "implicit_plus_minimal_residual_cue_pass",
}


def _load_json(path: Path) -> dict:
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _normalized(text: object) -> str | None:
    if text is None:
        return None
    value = str(text).strip().lower()
    return value or None


def _cue_repair_protocol() -> dict:
    payload = _load_json(CUE_REPAIR_PROTOCOL_PATH)
    if payload.get("schema_version") != "cue_repair_protocol_v1":
        raise ValueError(f"unexpected cue repair protocol schema_version in {CUE_REPAIR_PROTOCOL_PATH}")
    return payload


def _supported_visible_objects(hview_confusable_universe: dict) -> list[dict]:
    objects = list(hview_confusable_universe.get("objects") or [])
    return [
        obj
        for obj in objects
        if _normalized(obj.get("support_status")) == "supported"
        and _normalized(obj.get("visibility_status")) == "visible"
    ]


def _supported_same_type_count(hview_confusable_universe: dict, same_type_key: str | None) -> int:
    if same_type_key is None:
        return 0
    return sum(
        1
        for obj in _supported_visible_objects(hview_confusable_universe)
        if _normalized(obj.get("same_type_key")) == same_type_key
    )


def _promptable_anchor_count(hview_confusable_universe: dict, anchor_category: str | None) -> int:
    if anchor_category is None:
        return 0
    anchor_confusable_ids = hview_confusable_universe.get("anchor_confusable_ids") or {}
    count = 0
    for obj in _supported_visible_objects(hview_confusable_universe):
        if _normalized(obj.get("canonical_category")) != anchor_category:
            continue
        confusers = list(anchor_confusable_ids.get(str(int(obj.get("object_id", -1))), []))
        if confusers:
            continue
        count += 1
    return count


def _supported_anchor_category_count(hview_confusable_universe: dict, anchor_category: str | None) -> int:
    if anchor_category is None:
        return 0
    return sum(
        1
        for obj in _supported_visible_objects(hview_confusable_universe)
        if _normalized(obj.get("canonical_category")) == anchor_category
    )


def _cue_bundle_from_record(rec: dict, resolved_cue_package: dict | None) -> dict:
    if isinstance(resolved_cue_package, dict) and resolved_cue_package:
        return {
            "cue_family": resolved_cue_package.get("cue_family"),
            "cue_type": resolved_cue_package.get("cue_type"),
            "cue_value": resolved_cue_package.get("cue_value"),
            "cue_match_count": resolved_cue_package.get("cue_match_count"),
            "anchor_category": resolved_cue_package.get("anchor_category"),
            "residual_cue_type": resolved_cue_package.get("residual_cue_type"),
            "residual_cue_value": resolved_cue_package.get("residual_cue_value"),
            "minimal_residual_cue": bool(resolved_cue_package.get("minimal_residual_cue")),
        }
    bundle = rec.get("fixed_target_prompt_cue_bundle")
    if isinstance(bundle, dict):
        return dict(bundle)
    return {
        "cue_family": rec.get("cue_family"),
        "cue_type": rec.get("cue_type"),
        "cue_value": rec.get("cue_value"),
    }


def _trace_from_record(rec: dict) -> dict:
    trace = rec.get("promptability_resolution_trace")
    return dict(trace) if isinstance(trace, dict) else {}


def _contains_phrase(prompt_text: str, phrase: str) -> bool:
    escaped = re.escape(phrase.strip().lower())
    return re.search(rf"\b{escaped}\b", prompt_text) is not None


def _contains_any_phrase(prompt_text: str, phrases: tuple[str, ...]) -> bool:
    return any(_contains_phrase(prompt_text, phrase) for phrase in phrases)


def _relation_surface_count(prompt_text: str, cue_type: str | None) -> int:
    count = 0
    for phrase in RELATION_SURFACE_PHRASES.get(cue_type or "", ()):
        count += len(re.findall(rf"\b{re.escape(phrase.strip().lower())}\b", prompt_text))
    return count


def _contains_unexpected_relation_surface(prompt_text: str, cue_type: str | None) -> bool:
    lowered = prompt_text.lower()
    for other_cue_type, phrases in STRICT_RELATION_SURFACE_PHRASES.items():
        if other_cue_type == cue_type:
            continue
        for phrase in phrases:
            if phrase.startswith(" ") and phrase.endswith(" "):
                if phrase in f" {lowered} ":
                    return True
            elif _contains_phrase(lowered, phrase):
                return True
    return False


def _surface_count(prompt_text: str, phrase: str) -> int:
    lowered = prompt_text.lower()
    return len(re.findall(rf"\b{re.escape(phrase.strip().lower())}\b", lowered))


def _residual_surface_present(prompt_text: str, cue_bundle: dict) -> bool:
    residual_cue_type = _normalized(cue_bundle.get("residual_cue_type"))
    residual_cue_value = _normalized(cue_bundle.get("residual_cue_value"))
    if residual_cue_type in {None, "none"} or residual_cue_value in {None, "none"}:
        return not bool(cue_bundle.get("minimal_residual_cue"))
    return _contains_any_phrase(
        prompt_text,
        tuple(RESIDUAL_SURFACE_PHRASES.get(residual_cue_type, {}).get(residual_cue_value, ())),
    )


def _explicit_target_leakage(question_text: str, visible_record: dict) -> bool:
    lowered = question_text.lower()
    explicit_categories = {
        _normalized(visible_record.get("canonical_category")),
        _normalized(visible_record.get("target_category")),
        _normalized(visible_record.get("target_lock_category")),
    }
    for category in explicit_categories:
        if category and re.search(rf"\b{re.escape(category)}s?\b", lowered):
            return True
    return bool(
        re.search(r"\b(display\s*id|target\s*id|object\s*id|id)\s*[:#-]?\s*\d+\b", lowered)
        or re.search(r"\b#\s*\d+\b", lowered)
    )


def _surface_contract_status(
    *,
    task: str,
    prompt_text: str,
    visible_record: dict,
    cue_bundle: dict,
) -> tuple[str, str | None]:
    cue_family = _normalized(cue_bundle.get("cue_family")) or _normalized(visible_record.get("cue_family"))
    cue_type = _normalized(cue_bundle.get("cue_type"))
    anchor_category = _normalized(cue_bundle.get("anchor_category"))
    prompt_lower = prompt_text.lower()

    if task == "c":
        generated_question = str(visible_record.get("generated_question") or "").strip()
        generated_lower = generated_question.lower()
        if not generated_question:
            return "implicit_missing", "generated_question_missing"
        if _explicit_target_leakage(generated_question, visible_record):
            return "implicit_leak", "implicit_prompt_explicit_target_leakage"
        if cue_family == "relation":
            if cue_type not in RELATION_CUE_TYPES or anchor_category is None:
                return "implicit_surface_mismatch", "implicit_relation_contract_incomplete"
            if not _contains_phrase(generated_lower, anchor_category):
                return "implicit_surface_mismatch", "implicit_relation_surface_mismatch"
            relation_terms = RELATION_SURFACE_PHRASES.get(cue_type, ())
            if not _contains_any_phrase(generated_lower, relation_terms):
                return "implicit_surface_mismatch", "implicit_relation_surface_mismatch"
            if _relation_surface_count(generated_lower, cue_type) > 1:
                return "implicit_surface_mismatch", "implicit_duplicate_relation_surface"
            if _contains_unexpected_relation_surface(generated_lower, cue_type):
                return "implicit_surface_mismatch", "implicit_extra_relation_surface"
            if not _residual_surface_present(generated_lower, cue_bundle):
                return "implicit_surface_mismatch", "implicit_residual_surface_mismatch"
        elif cue_family == "color":
            cue_value = _normalized(cue_bundle.get("cue_value"))
            if cue_value is None:
                return "implicit_surface_mismatch", "implicit_color_contract_incomplete"
            if cue_value not in generated_lower:
                return "implicit_surface_mismatch", "implicit_color_surface_mismatch"
            if _surface_count(generated_lower, cue_value) > 1:
                return "implicit_surface_mismatch", "implicit_duplicate_color_surface"
        return "implicit_ok", None

    if cue_family == "natural_unique":
        category = _normalized(visible_record.get("canonical_category")) or _normalized(visible_record.get("target_category"))
        if category is not None and _contains_phrase(prompt_lower, category):
            return "matched", None
        return "surface_mismatch", "natural_unique_surface_mismatch"

    if cue_family == "relation":
        if cue_type not in RELATION_CUE_TYPES or anchor_category is None:
            return "surface_mismatch", "relation_cue_contract_incomplete"
        relation_phrase = f"{cue_type} the {anchor_category}"
        if relation_phrase in prompt_lower:
            return "matched", None
        return "surface_mismatch", "relation_surface_mismatch"

    if cue_family == "color":
        cue_value = _normalized(cue_bundle.get("cue_value"))
        category = _normalized(visible_record.get("canonical_category")) or _normalized(visible_record.get("target_category"))
        if cue_value is None or category is None:
            return "surface_mismatch", "color_cue_contract_incomplete"
        if cue_value in prompt_lower and category in prompt_lower:
            return "matched", None
        return "surface_mismatch", "color_surface_mismatch"

    return "unsupported", "unsupported_cue_family_under_resolver"


def resolve_prompt_under_hview(
    *,
    rec: dict,
    prompt_text: str,
    hview_confusable_universe: dict,
    resolved_cue_package: dict | None = None,
) -> dict:
    contract = get_prompt_resolver_contract()
    visible_record = build_resolver_visible_record(rec)
    cue_bundle = _cue_bundle_from_record(visible_record, resolved_cue_package)
    trace = _trace_from_record(visible_record)
    task = _normalized(rec.get("task")) or "unknown"
    cue_family = _normalized(cue_bundle.get("cue_family")) or _normalized(visible_record.get("cue_family"))
    cue_type = _normalized(cue_bundle.get("cue_type"))
    anchor_category = _normalized(cue_bundle.get("anchor_category"))
    same_type_key = _normalized(visible_record.get("same_type_key"))
    supported_same_type_count = _supported_same_type_count(hview_confusable_universe, same_type_key)
    supported_anchor_category_count = _supported_anchor_category_count(
        hview_confusable_universe, anchor_category
    )
    promptable_anchor_count = _promptable_anchor_count(hview_confusable_universe, anchor_category)
    cue_match_count = cue_bundle.get("cue_match_count")
    cue_match_count = int(cue_match_count) if cue_match_count is not None else None
    human_visible_support = _normalized(visible_record.get("human_visible_support")) or _normalized(
        trace.get("human_visible_support")
    )
    human_visible_uniqueness = _normalized(visible_record.get("human_visible_uniqueness")) or _normalized(
        trace.get("human_visible_uniqueness")
    )
    prompt_repair_status = _normalized(visible_record.get("prompt_repair_status")) or _normalized(
        trace.get("prompt_repair_status")
    )
    implicit_mode = _normalized(visible_record.get("implicit_resolution_mode")) or _normalized(
        contract.get("default_implicit_resolution_mode")
    )
    if implicit_mode not in IMPLICIT_ALLOWED_MODES:
        raise ValueError(f"unsupported implicit_resolution_mode={implicit_mode}")

    surface_status, surface_drop_reason = _surface_contract_status(
        task=task,
        prompt_text=str(prompt_text or ""),
        visible_record=visible_record,
        cue_bundle=cue_bundle,
    )

    resolver_result = "drop"
    drop_reason_code = surface_drop_reason
    if surface_drop_reason is None:
        if cue_family == "natural_unique":
            if (
                supported_same_type_count == 1
                and human_visible_support == "supported"
                and human_visible_uniqueness == "unique"
            ):
                resolver_result = "pass"
            else:
                drop_reason_code = "natural_unique_not_unique_under_hview"
        elif cue_family == "relation":
            if cue_type not in RELATION_CUE_TYPES:
                drop_reason_code = "unsupported_relation_type"
            elif supported_anchor_category_count != 1:
                drop_reason_code = "anchor_category_not_unique_under_hview"
            elif promptable_anchor_count != 1:
                drop_reason_code = "anchor_category_not_unique_under_hview"
            elif cue_match_count is not None and cue_match_count != 1:
                drop_reason_code = "relation_not_unique_under_hview"
            elif human_visible_support != "supported" or human_visible_uniqueness != "unique":
                drop_reason_code = "relation_not_supported_under_hview"
            else:
                resolver_result = "pass"
        elif cue_family == "color":
            if prompt_repair_status != "repaired":
                drop_reason_code = "color_cue_not_repaired"
            elif human_visible_support != "supported" or human_visible_uniqueness != "unique":
                drop_reason_code = "color_not_unique_under_hview"
            else:
                resolver_result = "pass"
        else:
            drop_reason_code = "unsupported_cue_family_under_resolver"

    if task == "c" and resolver_result == "pass":
        if _normalized(visible_record.get("route_semantics")) != "embodied":
            resolver_result = "drop"
            drop_reason_code = "c_requires_embodied_route_semantics"

    return {
        "schema_version": "prompt_resolution_trace_v1",
        "question_id": str(rec.get("question_id") or ""),
        "task": task,
        "resolver_result": resolver_result,
        "drop_reason_code": drop_reason_code,
        "surface_contract_status": surface_status,
        "resolved_cue_family": cue_family,
        "resolved_cue_type": cue_type,
        "anchor_category": anchor_category,
        "cue_match_count": cue_match_count,
        "supported_same_type_count": supported_same_type_count,
        "supported_anchor_category_count": supported_anchor_category_count,
        "promptable_anchor_count": promptable_anchor_count,
        "implicit_resolution_mode": implicit_mode,
        "used_record_fields": sorted(visible_record.keys()),
        "resolver_contract_version": contract["schema_version"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve prompt uniqueness under H(view).")
    parser.add_argument("--record-json", required=True)
    parser.add_argument("--hview-confusable-universe", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--resolved-cue-package", default=None)
    parser.add_argument("--prompt-text", default=None)
    args = parser.parse_args()

    record = _load_json(Path(args.record_json))
    hview_confusable_universe = _load_json(Path(args.hview_confusable_universe))
    resolved_cue_package = (
        _load_json(Path(args.resolved_cue_package)) if args.resolved_cue_package is not None else None
    )
    prompt_text = str(args.prompt_text or record.get("generated_question") or "")

    trace = resolve_prompt_under_hview(
        rec=record,
        prompt_text=prompt_text,
        hview_confusable_universe=hview_confusable_universe,
        resolved_cue_package=resolved_cue_package,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(trace, f, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
