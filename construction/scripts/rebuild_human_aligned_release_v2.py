#!/usr/bin/env python3
"""
Rebuild a human-aligned source pool directly from full release roots.

This builder keeps fixed GT carriers frozen while regenerating prompt-side
human-visible uniqueness metadata using the new H(view) + relation projection
chain. It then emits clean GT/VQA pools that can be fed into the existing
benchmark-first split builder.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_c_main_vnext_pilot_from_release import (  # noqa: E402
    CUE_SUFFIXES,
    load_ontology_config,
    stable_index,
)
from build_hview_confusable_universe import (  # noqa: E402
    build_hview_confusable_universe,
    load_proxy_group_map,
)
from build_navigation_gt_next import (  # noqa: E402
    FIXED_TARGET_CONTRACT_VERSION,
    VIEW_BUNDLE_FILENAME,
    apply_resolved_prompt_package,
    build_prompt_aligned_navigation_reference_facts,
    build_truth_repair_metadata,
    derive_human_visible_legality,
    get_hview_confusable_universe_protocol,
    get_schema_bundle_hash,
    get_truth_repair_protocol,
    infer_source_group_id,
    load_view_bundle_for_view,
)
from build_next_release_datasets import _build_vqa_row_from_gt, build_release_datasets  # noqa: E402
from build_scene_relation_graph import build_scene_relation_graph  # noqa: E402
from build_view_relation_projection import build_view_relation_projection  # noqa: E402
from build_vqa_questions_next import build_c_prompt  # noqa: E402
from cue_synthesis import build_resolved_cue_package  # noqa: E402
from h_view_utils import build_h_view  # noqa: E402
from prompt_truth_contract import derive_fixed_gt_identity_hash_from_identity  # noqa: E402
from resolve_prompt_under_hview import resolve_prompt_under_hview  # noqa: E402
from tier_policy import classify_navigation_tier, classify_reference_axis  # noqa: E402


TASKS = ("a1", "b1", "a2", "b2", "c")
ROUTING_TASKS = ("a2", "b2", "c")
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROXY_GROUPS_PATH = ROOT_DIR / "configs" / "c_intent_family_proxy_groups_v1.json"
DEFAULT_C_ONTOLOGY_PATH = ROOT_DIR / "configs" / "c_main_vnext_candidate_ontology.json"
DEFAULT_C_PROMPT_CATALOG_PATH = ROOT_DIR / "configs" / "c_prompt_catalog_v1.json"
DEFAULT_SCENE_ROOT_CANDIDATES = (
    "scenes_v3_4src_relaxed_refilter_post_objaverse_20260325_124913",
    "scenes_v3_4src_relaxed_refilter_20260325_121234",
    "scenes_v3_4src_relaxed",
    "scenes_v3_4src",
    "scenes",
)


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _normalized_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalized_lower(value: object) -> str | None:
    text = _normalized_text(value)
    return text.lower() if text is not None else None


def _routing_semantics_for_task(task: str) -> str:
    return "embodied" if task in {"b2", "c"} else "pointmass"


def _cue_bundle_from_resolved(resolved_cue_package: dict) -> dict:
    return {
        "cue_family": resolved_cue_package.get("cue_family"),
        "cue_type": resolved_cue_package.get("cue_type"),
        "cue_value": resolved_cue_package.get("cue_value"),
        "anchor_category": resolved_cue_package.get("anchor_category"),
        "cue_match_count": resolved_cue_package.get("cue_match_count"),
        "residual_cue_type": resolved_cue_package.get("residual_cue_type"),
        "residual_cue_value": resolved_cue_package.get("residual_cue_value"),
        "minimal_residual_cue": bool(resolved_cue_package.get("minimal_residual_cue")),
    }


def _load_json(path: Path) -> dict | list:
    with open(path) as f:
        return json.load(f)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def _resolve_scene_dir(*, release_root: Path, scene_id: str, scenes_root: Path | None) -> Path:
    candidate_roots: list[Path] = []
    if scenes_root is not None:
        candidate_roots.append(Path(scenes_root).resolve())
    candidate_roots.append(release_root / "scenes")
    if release_root.parent.name == "releases":
        data_root = release_root.parent.parent
        for dirname in DEFAULT_SCENE_ROOT_CANDIDATES:
            candidate_roots.append(data_root / dirname)

    seen: set[Path] = set()
    searched: list[str] = []
    for root in candidate_roots:
        resolved_root = root.resolve()
        if resolved_root in seen:
            continue
        seen.add(resolved_root)
        scene_dir = resolved_root / scene_id
        searched.append(str(scene_dir))
        if (scene_dir / "layout.json").exists() and (scene_dir / "metadata.json").exists():
            return scene_dir
    raise FileNotFoundError(
        f"scene_id={scene_id} missing layout.json/metadata.json under any scene root candidate: {searched}"
    )


def _c_slot_key(row: dict) -> tuple[str, int, str]:
    return (str(row["scene_id"]), int(row["view_id"]), str(row["routing_id"]))


def _derive_c_question_id(b2_row: dict, source_row: dict | None) -> str:
    if source_row is not None and _normalized_text(source_row.get("question_id")) is not None:
        return str(source_row["question_id"])
    question_id = str(b2_row["question_id"])
    if question_id.endswith("_b2"):
        return f"{question_id[:-3]}_c"
    return f"{question_id}_c"


def _family_spec_for_category(ontology: dict, canonical_category: str | None, intent_family: str | None) -> dict:
    payload = ontology["payload"]
    family_specs = payload.get("families") or {}
    family = _normalized_lower(intent_family)
    category = _normalized_lower(canonical_category)
    if family is None:
        return {}
    spec = dict(family_specs.get(family) or {})
    safe_templates = spec.get("safe_category_question_templates") or {}
    if category is not None and category in safe_templates:
        spec = dict(spec)
        spec["question_templates"] = list(safe_templates[category])
    return spec


def load_c_prompt_catalog(path: Path) -> dict:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"c prompt catalog must be a JSON object: {path}")
    if _normalized_text(payload.get("schema_version")) != "c_prompt_catalog_v1":
        raise ValueError(f"c prompt catalog schema_version mismatch: {path}")
    raw_families = payload.get("families")
    if not isinstance(raw_families, dict) or not raw_families:
        raise ValueError(f"c prompt catalog missing families: {path}")

    families: dict[str, dict] = {}
    for raw_family, raw_spec in raw_families.items():
        family = _normalized_lower(raw_family)
        if family is None or not isinstance(raw_spec, dict):
            continue
        raw_slots = raw_spec.get("slots")
        if not isinstance(raw_slots, dict) or not raw_slots:
            raise ValueError(f"c prompt catalog family={family} missing slots")
        slots: dict[str, dict] = {}
        for raw_slot, raw_slot_spec in raw_slots.items():
            slot = _normalized_lower(raw_slot)
            if slot is None or not isinstance(raw_slot_spec, dict):
                continue
            templates = [
                str(text).strip()
                for text in list(raw_slot_spec.get("templates") or [])
                if str(text).strip()
            ]
            if not templates:
                raise ValueError(f"c prompt catalog family={family} slot={slot} missing templates")
            slots[slot] = {
                "templates": templates,
                "banned_phrases": [
                    str(text).strip().lower()
                    for text in list(raw_slot_spec.get("banned_phrases") or [])
                    if str(text).strip()
                ],
            }
        if not slots:
            raise ValueError(f"c prompt catalog family={family} has no usable slots")
        category_to_slot = {
            str(category).strip().lower(): str(slot).strip().lower()
            for category, slot in dict(raw_spec.get("category_to_slot") or {}).items()
            if str(category).strip() and str(slot).strip()
        }
        default_slot = _normalized_lower(raw_spec.get("default_slot"))
        families[family] = {
            "default_slot": default_slot,
            "category_to_slot": category_to_slot,
            "slots": slots,
        }
    return {
        "schema_version": "c_prompt_catalog_v1",
        "path": str(path.resolve()),
        "families": families,
    }


def _resolve_c_prompt_slot(
    *,
    prompt_catalog: dict,
    intent_family: str,
    canonical_category: str | None,
) -> tuple[str | None, dict | None]:
    family = _normalized_lower(intent_family)
    category = _normalized_lower(canonical_category)
    if family is None:
        return None, None
    family_spec = dict((prompt_catalog.get("families") or {}).get(family) or {})
    if not family_spec:
        return None, None
    category_to_slot = dict(family_spec.get("category_to_slot") or {})
    slot_name = category_to_slot.get(category or "") or family_spec.get("default_slot")
    slot_name = _normalized_lower(slot_name)
    if slot_name is None:
        return None, None
    slot_spec = dict((family_spec.get("slots") or {}).get(slot_name) or {})
    if not slot_spec:
        return None, None
    return slot_name, slot_spec


def _question_contains_phrase(text: str, phrase: str) -> bool:
    normalized_phrase = _normalized_lower(phrase)
    if normalized_phrase is None:
        return False
    return re.search(rf"\b{re.escape(normalized_phrase)}\b", text.lower()) is not None


def _preferred_c_question_for_slot(
    *,
    preferred_question: str | None,
    slot_spec: dict,
) -> str | None:
    candidate = _normalized_text(preferred_question)
    if candidate is None:
        return None
    lowered = candidate.lower()
    if re.search(r"\b(display\s*id|target\s*id|object\s*id|id)\s*[:#-]?\s*\d+\b", lowered):
        return None
    for phrase in list(slot_spec.get("banned_phrases") or []):
        if _question_contains_phrase(lowered, str(phrase)):
            return None
    return candidate


def _legacy_c_question_reuse_allowed(*, resolved_cue_package: dict) -> bool:
    cue_family = _normalized_lower(resolved_cue_package.get("cue_family"))
    return cue_family in {"natural_unique", "none"}


def _relation_fragment(cue_type: str, anchor_category: str | None) -> str | None:
    if anchor_category is None:
        return None
    if cue_type == "near":
        return f"near the {anchor_category}"
    if cue_type == "on":
        return f"on the {anchor_category}"
    if cue_type == "inside":
        return f"inside the {anchor_category}"
    if cue_type == "under":
        return f"under the {anchor_category}"
    return None


def _base_question_stem(text: str) -> str:
    stem = str(text).strip()
    while stem.endswith(("!", "?", ".")):
        stem = stem[:-1].rstrip()
    return stem


def _append_c_phrase(base_question: str, phrase: str, *, separator: str = " ") -> str:
    stem = _base_question_stem(base_question)
    if not stem:
        return phrase[:1].upper() + phrase[1:] + "."
    return f"{stem}{separator}{phrase}."


def _normalized_target_color(candidate: dict) -> str | None:
    for key in ("target_color_name", "reference_color_name", "visible_color_name", "material_color_name"):
        value = _normalized_lower(candidate.get(key))
        if value not in {None, "unknown", "none", "n/a", "na", "null"}:
            return value
    return None


def _normalize_truth_candidate(candidate: dict) -> dict:
    target_id = candidate.get("target_id")
    if target_id is None:
        raise ValueError("routing candidate missing target_id")
    canonical_category = (
        _normalized_lower(candidate.get("canonical_category"))
        or _normalized_lower(candidate.get("target_category"))
        or _normalized_lower(candidate.get("category"))
    )
    if canonical_category is None:
        raise ValueError(f"routing candidate target_id={target_id} missing canonical category")
    normalized = dict(candidate)
    normalized["target_id"] = int(target_id)
    normalized["canonical_category"] = canonical_category
    normalized["target_category"] = normalized.get("target_category") or canonical_category
    color = _normalized_target_color(candidate)
    if color is not None:
        normalized["target_color_name"] = color
    return normalized


def _append_c_residual_cue(base_question: str, resolved_cue_package: dict) -> str:
    cue_family = _normalized_lower(resolved_cue_package.get("cue_family")) or "drop"
    cue_type = _normalized_lower(resolved_cue_package.get("cue_type"))
    raw_cue_value = resolved_cue_package.get("cue_value")
    cue_value = _normalized_text(raw_cue_value)
    anchor_category = _normalized_lower(resolved_cue_package.get("anchor_category"))
    residual_cue_type = _normalized_lower(resolved_cue_package.get("residual_cue_type"))
    residual_cue_value = _normalized_text(resolved_cue_package.get("residual_cue_value"))
    if cue_family in {"natural_unique", "none"}:
        return base_question

    def _append_minimal_residual(text: str | None) -> str | None:
        if text is None or residual_cue_type in {None, "none"} or residual_cue_value is None:
            return text
        if residual_cue_type == "distance":
            label = {
                "nearer": "specifically the nearer option",
                "farther": "specifically the farther option",
                "closest": "specifically the closest option",
                "farthest": "specifically the farthest option",
            }.get(residual_cue_value)
            return _append_c_phrase(text, label, separator=", ") if label is not None else text
        if residual_cue_type == "side":
            label = {
                "left": "specifically the leftmost option",
                "right": "specifically the rightmost option",
            }.get(residual_cue_value)
            return _append_c_phrase(text, label, separator=", ") if label is not None else text
        if residual_cue_type == "ordinal":
            label = {
                "first": "specifically the first option from left to right",
                "second": "specifically the second option from left to right",
            }.get(residual_cue_value)
            return _append_c_phrase(text, label, separator=", ") if label is not None else text
        return text

    extra: str | None = None
    if cue_family == "relation":
        relation_fragment = _relation_fragment(cue_type or "", anchor_category)
        if relation_fragment is not None:
            extra = _append_minimal_residual(_append_c_phrase(base_question, relation_fragment))
    elif cue_family == "relation + color" and isinstance(raw_cue_value, dict):
        relation_type = _normalized_lower(raw_cue_value.get("relation_type"))
        relation_anchor = _normalized_lower(raw_cue_value.get("anchor_category")) or anchor_category
        color = _normalized_text(raw_cue_value.get("color"))
        relation_fragment = _relation_fragment(relation_type or "", relation_anchor)
        if relation_fragment is not None and color is not None:
            extra = _append_c_phrase(base_question, relation_fragment)
            extra = _append_c_phrase(extra, f"specifically the {color} one", separator=", ")
    elif cue_family == "relation + attribute" and isinstance(raw_cue_value, dict):
        relation_type = _normalized_lower(raw_cue_value.get("relation_type"))
        relation_anchor = _normalized_lower(raw_cue_value.get("anchor_category")) or anchor_category
        attribute_value = _normalized_text(raw_cue_value.get("attribute_value"))
        relation_fragment = _relation_fragment(relation_type or "", relation_anchor)
        if relation_fragment is not None and attribute_value is not None:
            extra = _append_c_phrase(base_question, relation_fragment)
            extra = _append_c_phrase(extra, f"specifically the one with {attribute_value}", separator=", ")
    elif cue_family == "attribute":
        if cue_value is not None:
            extra = _append_c_phrase(base_question, f"specifically the one with {cue_value}", separator=", ")
    elif cue_family == "color" and cue_value is not None:
        extra = _append_c_phrase(base_question, f"specifically the {cue_value} one", separator=", ")
    elif cue_family == "strict_distance" and cue_value is not None:
        if cue_value == "nearest":
            extra = _append_c_phrase(base_question, "specifically the nearer one", separator=", ")
        elif cue_value == "farthest":
            extra = _append_c_phrase(base_question, "specifically the farther one", separator=", ")

    if extra is None:
        return base_question

    if extra.lower() == base_question.lower():
        return base_question
    return extra


def _generate_c_question(
    *,
    question_id: str,
    canonical_category: str | None,
    intent_family: str,
    resolved_cue_package: dict,
    prompt_catalog: dict,
    preferred_question: str | None,
) -> tuple[str, str, str]:
    prompt_slot, slot_spec = _resolve_c_prompt_slot(
        prompt_catalog=prompt_catalog,
        intent_family=intent_family,
        canonical_category=canonical_category,
    )
    if prompt_slot is None or slot_spec is None:
        raise ValueError(
            f"no prompt slot for intent_family={intent_family} canonical_category={canonical_category}"
        )
    base_question = None
    if _legacy_c_question_reuse_allowed(resolved_cue_package=resolved_cue_package):
        base_question = _preferred_c_question_for_slot(
            preferred_question=preferred_question,
            slot_spec=slot_spec,
        )
    prompt_slot_source = "legacy_preferred_question" if base_question is not None else "prompt_catalog_template"
    if base_question is None:
        templates = list(slot_spec["templates"])
        base_question = templates[stable_index(question_id, len(templates))]
    return _append_c_residual_cue(base_question, resolved_cue_package), prompt_slot, prompt_slot_source


def _load_c_source_index(c_source_dirs: list[Path]) -> dict[tuple[str, int, str], dict]:
    indexed: dict[tuple[str, int, str], dict] = {}
    for source_dir in c_source_dirs:
        for split in ("benchmark", "val", "train"):
            gt_path = source_dir / split / "gt" / "gt_next_c.jsonl"
            for row in load_jsonl(gt_path):
                slot = _c_slot_key(row)
                if slot not in indexed:
                    indexed[slot] = dict(row)
    return indexed


def _find_target_hview_object(raw_h_view: dict, target_id: int) -> dict | None:
    for obj in list(raw_h_view.get("objects") or raw_h_view.get("h_view_objects") or []):
        if int(obj.get("object_id", -1)) == int(target_id):
            return obj
    for obj in list(raw_h_view.get("h_view_objects") or []):
        if int(obj.get("object_id", -1)) == int(target_id):
            return obj
    return None


def _same_type_members(raw_h_view: dict, same_type_key: str | None) -> tuple[list[int], list[int]]:
    if same_type_key is None:
        return [], []
    supported_ids: list[int] = []
    uncertain_ids: list[int] = []
    objects = list(raw_h_view.get("objects") or raw_h_view.get("h_view_objects") or [])
    for obj in objects:
        if _normalized_lower(obj.get("same_type_key")) != same_type_key:
            continue
        object_id = int(obj["object_id"])
        if obj.get("support_status") == "supported" and obj.get("visibility_status") == "visible":
            supported_ids.append(object_id)
        else:
            uncertain_ids.append(object_id)
    return sorted(set(supported_ids)), sorted(set(uncertain_ids))


def _resolve_target_candidate(bundle: dict, row: dict) -> dict:
    routing_id = _normalized_text(row.get("routing_id"))
    raw_target_id = row.get("target_id")
    target_id = int(raw_target_id) if raw_target_id is not None else None
    candidates = list(bundle.get("routing_candidates") or [])
    for candidate in candidates:
        if routing_id is not None and _normalized_text(candidate.get("routing_id")) == routing_id:
            candidate_target_id = candidate.get("target_id")
            if candidate_target_id is None:
                raise ValueError(f"routing_id={routing_id} candidate missing target_id")
            if target_id is not None and int(candidate_target_id) != target_id:
                raise ValueError(
                    f"routing_id={routing_id} target mismatch: row target_id={target_id} "
                    f"candidate target_id={candidate_target_id}"
                )
            return dict(candidate)
    if target_id is not None:
        for candidate in candidates:
            candidate_target_id = candidate.get("target_id")
            if candidate_target_id is not None and int(candidate_target_id) == target_id:
                return dict(candidate)
    if routing_id is not None:
        raise KeyError(f"routing_id={routing_id} missing from view bundle routing_candidates")
    if target_id is not None:
        raise KeyError(f"target_id={target_id} missing from view bundle routing_candidates")
    raise ValueError("routing row must provide routing_id or target_id")


def _hydrate_navigation_row_from_candidate(row: dict, *, task: str, target_candidate: dict) -> dict:
    rec = dict(row)
    routing_id = _normalized_text(rec.get("routing_id")) or _normalized_text(target_candidate.get("routing_id"))
    if routing_id is None:
        raise ValueError("routing row missing routing_id and bundle candidate missing routing_id")
    rec["routing_id"] = routing_id

    target_id = rec.get("target_id")
    if target_id is None:
        target_id = target_candidate.get("target_id")
    if target_id is None:
        raise ValueError(f"routing_id={routing_id} missing target_id in row and bundle candidate")
    rec["target_id"] = int(target_id)

    candidate_category = (
        _normalized_lower(target_candidate.get("canonical_category"))
        or _normalized_lower(target_candidate.get("target_category"))
        or _normalized_lower(target_candidate.get("category"))
    )
    if candidate_category is None:
        raise ValueError(f"routing_id={routing_id} bundle candidate missing canonical category")
    row_category = _normalized_lower(rec.get("canonical_category")) or _normalized_lower(rec.get("target_category"))
    if row_category is not None and row_category != candidate_category:
        raise ValueError(
            f"routing_id={routing_id} category mismatch: row={row_category} candidate={candidate_category}"
        )
    rec["canonical_category"] = rec.get("canonical_category") or candidate_category
    rec["target_category"] = rec.get("target_category") or candidate_category

    candidate_color = _normalized_target_color(target_candidate)
    if _normalized_lower(rec.get("target_color_name")) is None and candidate_color is not None:
        rec["target_color_name"] = candidate_color

    if rec.get("target_distance_m") is None:
        candidate_distance = target_candidate.get("distance_m", target_candidate.get("distance"))
        if candidate_distance is not None:
            rec["target_distance_m"] = candidate_distance

    goal_anchor_id = rec.get("goal_anchor_id")
    if goal_anchor_id is None:
        goal_anchor_id = rec.get("goal_id")
    if goal_anchor_id is None:
        goal_anchor_id = target_candidate.get("goal_anchor_id")
    if goal_anchor_id is None:
        goal_anchor_id = target_candidate.get("goal_id")
    if goal_anchor_id is not None:
        goal_anchor_id = int(goal_anchor_id)
        rec["goal_anchor_id"] = goal_anchor_id
        rec["goal_id"] = rec.get("goal_id") if rec.get("goal_id") is not None else goal_anchor_id

    if rec.get("gt_path_id") is None:
        rec["gt_path_id"] = f"{routing_id}::{_routing_semantics_for_task(task)}"
    return rec


def _prepare_navigation_carrier(row: dict, *, task: str, raw_h_view: dict) -> dict:
    route_semantics = _routing_semantics_for_task(task)
    target_id = int(row["target_id"])
    target_h_view = _find_target_hview_object(raw_h_view, target_id)
    same_type_key = _normalized_lower(
        (target_h_view or {}).get("same_type_key")
        or row.get("same_type_key")
        or row.get("canonical_category")
        or row.get("target_category")
    )

    rec = dict(row)
    rec["task"] = task
    rec["route_semantics"] = route_semantics
    rec["target_lock_id"] = target_id
    rec["target_lock_category"] = rec.get("canonical_category") or rec.get("target_category")
    rec["same_type_key"] = same_type_key
    rec["task_family"] = rec.get("task_family") or rec.get("tier_family") or "navigation"
    rec["goal_anchor_id"] = rec.get("goal_anchor_id") if rec.get("goal_anchor_id") is not None else rec.get("goal_id")
    rec["gt_path_id"] = rec.get("gt_path_id") or f"{rec['routing_id']}::{route_semantics}"
    rec["target_lock_preserved"] = True
    rec["task_family_preserved"] = True
    rec["goal_anchor_preserved"] = True
    rec["gt_path_preserved"] = True
    rec["fixed_target_contract_version"] = rec.get("fixed_target_contract_version") or FIXED_TARGET_CONTRACT_VERSION
    rec["immutable_identity"] = {
        "scene_id": rec.get("scene_id"),
        "view_id": int(rec.get("view_id")),
        "routing_id": rec.get("routing_id"),
        "target_id": target_id,
        "goal_anchor_id": int(rec["goal_anchor_id"]) if rec.get("goal_anchor_id") is not None else None,
        "gt_path_id": rec["gt_path_id"],
    }
    rec["fixed_gt_identity_hash"] = derive_fixed_gt_identity_hash_from_identity(
        rec["immutable_identity"],
        fixed_target_contract_version=rec["fixed_target_contract_version"],
    )
    return rec


def _repaired_status_fields(
    *,
    rec: dict,
    raw_h_view: dict,
    view_context: dict,
) -> dict:
    same_type_key = _normalized_lower(rec.get("same_type_key"))
    supported_ids, uncertain_ids = _same_type_members(raw_h_view, same_type_key)
    cue_family = _normalized_lower(rec.get("cue_family")) or "drop"
    prompt_unique = bool(rec.get("prompt_unique_under_hview"))
    human_visible_support = "supported" if prompt_unique else "uncertain"
    human_visible_uniqueness = "unique" if prompt_unique else "ambiguous"
    prompt_truth_tier = "benchmark_grade" if prompt_unique else "reject"
    prompt_answer_leakage = bool(rec.get("prompt_answer_leakage", False))
    fixed_target_supported = bool(rec.get("fixed_target_supported_under_hview", prompt_unique))
    natural_unique = cue_family == "natural_unique"
    unique_by_relation = cue_family == "relation"
    unique_by_color = cue_family == "color"
    unique_by_attribute = cue_family == "attribute"
    unique_by_distance = cue_family == "strict_distance"

    return {
        "human_visible_support": human_visible_support,
        "human_visible_uniqueness": human_visible_uniqueness,
        "human_visible_legality": derive_human_visible_legality(
            human_visible_support=human_visible_support,
            human_visible_uniqueness=human_visible_uniqueness,
            prompt_truth_tier=prompt_truth_tier,
            prompt_answer_leakage=prompt_answer_leakage,
        ),
        "human_visible_protocol_version": "hview_uview_v2",
        "human_visible_sidecar_ref": view_context["hview_relref"],
        "benchmark_unit_type": "route_bundle",
        "source_group_id": infer_source_group_id(str(rec.get("scene_id"))),
        "human_visible_same_type_object_ids": sorted(set(supported_ids + uncertain_ids)),
        "supported_visible_same_type_ids": supported_ids,
        "uncertain_same_type_ids": uncertain_ids,
        "fixed_target_supported_under_hview": fixed_target_supported,
        "natural_unique_under_hview": natural_unique,
        "unique_by_relation": unique_by_relation,
        "unique_by_attribute": unique_by_attribute,
        "unique_by_color": unique_by_color,
        "unique_by_distance": unique_by_distance,
        "cue_repair_protocol_version": rec.get("cue_repair_protocol_version") or "cue_repair_protocol_v1",
        "cue_repair_protocol_hash": rec.get("cue_repair_protocol_hash"),
        "hview_confusable_universe_version": rec.get("hview_confusable_universe_version")
        or view_context["hview"].get("hview_confusable_universe_version"),
        "hview_confusable_universe_hash": rec.get("hview_confusable_universe_hash")
        or view_context["hview"].get("hview_confusable_universe_hash"),
        "schema_bundle_hash": rec.get("schema_bundle_hash") or get_schema_bundle_hash(),
    }


def _build_prompt_row_for_resolution(
    *,
    rec: dict,
    resolved_cue_package: dict,
    generated_question: str | None = None,
    intent_family: str | None = None,
) -> dict:
    cue_family = _normalized_text(resolved_cue_package.get("cue_family")) or "drop"
    cue_type = resolved_cue_package.get("cue_type") or "none"
    prompt_row = dict(rec)
    prompt_row["cue_family"] = cue_family
    prompt_row["cue_type"] = cue_type
    prompt_row["cue_value"] = resolved_cue_package.get("cue_value")
    prompt_row["residual_cue_type"] = resolved_cue_package.get("residual_cue_type")
    prompt_row["residual_cue_value"] = resolved_cue_package.get("residual_cue_value")
    prompt_row["minimal_residual_cue"] = bool(resolved_cue_package.get("minimal_residual_cue"))
    prompt_row["fixed_target_prompt_cue_bundle"] = _cue_bundle_from_resolved(resolved_cue_package)
    prompt_row["human_visible_support"] = "supported" if cue_family != "drop" else "uncertain"
    prompt_row["human_visible_uniqueness"] = "unique" if cue_family != "drop" else "ambiguous"
    prompt_row["promptability_status"] = "natural_unique" if cue_family == "natural_unique" else (
        "cue_unique" if cue_family != "drop" else "not_promptable"
    )
    prompt_row["prompt_repair_status"] = "not_needed" if cue_family == "natural_unique" else "repaired"
    prompt_row["prompt_truth_tier"] = "benchmark_grade"
    prompt_row["promptability_resolution_trace"] = {
        **dict(prompt_row.get("promptability_resolution_trace") or {}),
        "cue_family": cue_family,
        "cue_type": cue_type,
        "cue_value": resolved_cue_package.get("cue_value"),
        "prompt_repair_status": "not_needed" if cue_family == "natural_unique" else "repaired",
        "prompt_truth_tier": "benchmark_grade",
        "human_visible_support": "supported" if cue_family != "drop" else "uncertain",
        "human_visible_uniqueness": "unique" if cue_family != "drop" else "ambiguous",
        "why_not_relation": resolved_cue_package.get("why_not_relation"),
    }
    if generated_question is not None:
        prompt_row["generated_question"] = generated_question
    if intent_family is not None:
        prompt_row["intent_family"] = intent_family
    return prompt_row


def _build_prompt_text_for_task(rec: dict) -> str:
    task = str(rec["task"])
    if task in {"a2", "b2"}:
        from build_vqa_questions_next import build_a2_prompt  # local import to avoid cycles during tests

        return build_a2_prompt(rec, embodied=(task == "b2"))["user"]
    if task == "c":
        return build_c_prompt(rec)["user"]
    raise ValueError(f"unsupported task for prompt build: {task}")


def _sync_navigation_difficulty_fields(rec: dict) -> dict:
    navigation_reference_facts = rec.get("navigation_reference_facts")
    if not isinstance(navigation_reference_facts, dict):
        raise ValueError("routing rebuild requires navigation_reference_facts dict")
    navigation_axes = rec.get("navigation_axes")
    if not isinstance(navigation_axes, dict):
        raise ValueError("routing rebuild requires navigation_axes dict")

    geometry_axis = _normalized_lower(navigation_axes.get("geometry_axis"))
    embodiment_axis = _normalized_lower(navigation_axes.get("embodiment_axis"))
    if geometry_axis is None:
        raise ValueError("routing rebuild requires navigation_axes.geometry_axis")
    if embodiment_axis is None:
        raise ValueError("routing rebuild requires navigation_axes.embodiment_axis")

    updated = dict(rec)
    prompt_aligned_reference_facts = build_prompt_aligned_navigation_reference_facts(
        original_reference_facts=navigation_reference_facts,
        truth_repair_metadata=updated,
    )
    reference_axis = classify_reference_axis(prompt_aligned_reference_facts)
    updated_axes = dict(navigation_axes)
    updated_axes["reference_axis"] = reference_axis

    updated["navigation_reference_facts"] = prompt_aligned_reference_facts
    updated["navigation_axes"] = updated_axes
    updated["tier"] = classify_navigation_tier(
        reference_axis=reference_axis,
        geometry_axis=geometry_axis,
        embodiment_axis=embodiment_axis,
    )
    return updated


def _build_c_candidate_from_b2(
    *,
    b2_row: dict,
    c_source_row: dict | None,
    resolved_cue_package: dict,
    ontology: dict,
    prompt_catalog: dict,
) -> dict | None:
    canonical_category = _normalized_lower(b2_row.get("canonical_category") or b2_row.get("target_category"))
    intent_family = ontology["category_to_family"].get(canonical_category or "")
    if intent_family is None:
        return None

    question_id = _derive_c_question_id(b2_row, c_source_row)
    preferred_question = _normalized_text((c_source_row or {}).get("generated_question"))
    try:
        generated_question, prompt_slot, prompt_slot_source = _generate_c_question(
            question_id=question_id,
            canonical_category=canonical_category,
            intent_family=intent_family,
            resolved_cue_package=resolved_cue_package,
            prompt_catalog=prompt_catalog,
            preferred_question=preferred_question,
        )
    except ValueError:
        return None

    return {
        **dict(b2_row),
        "question_id": question_id,
        "task": "c",
        "route_source_task": "b2",
        "route_semantics": "embodied",
        "intent_family": intent_family,
        "generated_question": generated_question,
        "prompt_slot": prompt_slot,
        "prompt_slot_source": prompt_slot_source,
        "prompt_catalog_version": prompt_catalog.get("schema_version"),
        "slot_key": f"{b2_row['scene_id']}::{int(b2_row['view_id'])}::{b2_row['routing_id']}",
        "slot_identity_locked": True,
        "slot_generation_mode": "exact_slot_prompt_regeneration",
    }


def _build_view_context(
    *,
    release_root: Path,
    output_dir: Path,
    scenes_root: Path | None,
    scene_id: str,
    view_id: int,
    proxy_group_map: dict[str, list[str]],
    scene_cache: dict[str, dict],
    view_cache: dict[tuple[str, int], dict],
) -> dict:
    key = (scene_id, int(view_id))
    cached = view_cache.get(key)
    if cached is not None:
        return cached

    scene_context = scene_cache.get(scene_id)
    if scene_context is None:
        scene_dir = _resolve_scene_dir(release_root=release_root, scene_id=scene_id, scenes_root=scenes_root)
        render_scene_dir = release_root / "renders" / scene_id
        layout = _load_json(scene_dir / "layout.json")
        metadata = _load_json(scene_dir / "metadata.json")
        scene_relation_graph, scene_relation_provenance = build_scene_relation_graph(
            scene_id=scene_id,
            layout=list(layout),
            metadata=dict(metadata),
        )
        scene_relation_path = output_dir / "sidecars" / "scene_relations" / scene_id / "scene_relation_graph.json"
        scene_relation_provenance_path = (
            output_dir / "sidecars" / "scene_relations" / scene_id / "scene_relation_provenance.json"
        )
        _write_json(scene_relation_path, scene_relation_graph)
        _write_json(scene_relation_provenance_path, scene_relation_provenance)
        scene_context = {
            "scene_dir": scene_dir,
            "render_scene_dir": render_scene_dir,
            "layout": list(layout),
            "metadata": dict(metadata),
            "scene_relation_graph": scene_relation_graph,
            "scene_relation_path": scene_relation_path,
        }
        scene_cache[scene_id] = scene_context

    bundle = load_view_bundle_for_view(scene_context["render_scene_dir"], int(view_id))
    visible_objects_path = scene_context["render_scene_dir"] / f"view_{int(view_id)}" / "visible_objects.json"
    audit_visible_objects = None
    if visible_objects_path.exists():
        payload = _load_json(visible_objects_path)
        if isinstance(payload, list):
            audit_visible_objects = payload
    raw_h_view = build_h_view(
        bundle,
        routing_rows=list(bundle.get("routing_candidates") or []),
        proxy_group_map=proxy_group_map,
        same_type_mode=str(get_hview_confusable_universe_protocol().get("default_same_type_mode") or "family_proxy"),
    )
    hview_universe, hview_audit = build_hview_confusable_universe(
        scene_id=scene_id,
        view_id=int(view_id),
        bundle=bundle,
        layout=scene_context["layout"],
        metadata=scene_context["metadata"],
        proxy_group_map=proxy_group_map,
        audit_visible_objects=audit_visible_objects,
    )
    hview_path = output_dir / "sidecars" / "h_view" / scene_id / f"view_{int(view_id)}.json"
    hview_audit_path = output_dir / "sidecars" / "h_view" / scene_id / f"view_{int(view_id)}.audit.json"
    _write_json(hview_path, hview_universe)
    _write_json(hview_audit_path, hview_audit)

    projection = build_view_relation_projection(
        scene_id=scene_id,
        view_id=int(view_id),
        scene_relation_graph=scene_context["scene_relation_graph"],
        hview_confusable_universe=hview_universe,
        bundle=bundle,
    )
    projection_path = output_dir / "sidecars" / "view_relations" / scene_id / f"view_{int(view_id)}.json"
    _write_json(projection_path, projection)

    context = {
        "bundle": bundle,
        "raw_h_view": raw_h_view,
        "hview": hview_universe,
        "projection": projection,
        "hview_path": hview_path,
        "hview_relref": str(Path("sidecars") / "h_view" / scene_id / f"view_{int(view_id)}.json"),
        "projection_path": projection_path,
    }
    view_cache[key] = context
    return context


def _rebuild_routing_row(
    *,
    row: dict,
    task: str,
    release_root: Path,
    output_dir: Path,
    scenes_root: Path | None,
    proxy_group_map: dict[str, list[str]],
    ontology: dict,
    prompt_catalog: dict,
    c_source_index: dict[tuple[str, int, str], dict],
    scene_cache: dict[str, dict],
    view_cache: dict[tuple[str, int], dict],
) -> dict | None:
    scene_id = str(row["scene_id"])
    view_id = int(row["view_id"])
    routing_id = str(row["routing_id"])
    view_context = _build_view_context(
        release_root=release_root,
        output_dir=output_dir,
        scenes_root=scenes_root,
        scene_id=scene_id,
        view_id=view_id,
        proxy_group_map=proxy_group_map,
        scene_cache=scene_cache,
        view_cache=view_cache,
    )

    base_task = "b2" if task == "c" else task
    target_candidate = _normalize_truth_candidate(_resolve_target_candidate(view_context["bundle"], row))
    hydrated_row = _hydrate_navigation_row_from_candidate(row, task=base_task, target_candidate=target_candidate)
    base_rec = _prepare_navigation_carrier(hydrated_row, task=task, raw_h_view=view_context["raw_h_view"])
    carrier_same_type_key = base_rec.get("same_type_key")
    truth_candidates = [
        _normalize_truth_candidate(candidate)
        for candidate in list(view_context["bundle"].get("routing_candidates") or [])
        if candidate.get("target_id") is not None
    ]
    metadata = build_truth_repair_metadata(
        target_candidate,
        truth_candidates,
        bundle=view_context["bundle"],
        source_row=base_rec,
    )
    base_rec.update(metadata)
    if carrier_same_type_key is not None:
        base_rec["same_type_key"] = carrier_same_type_key

    resolved_cue_package, cue_search_trace = build_resolved_cue_package(
        gt_row=base_rec,
        hview_confusable_universe=view_context["hview"],
        view_relation_projection=view_context["projection"],
    )

    c_source_row = c_source_index.get(_c_slot_key(base_rec)) if task == "c" else None
    if task == "c":
        c_rec = _build_c_candidate_from_b2(
            b2_row=base_rec,
            c_source_row=c_source_row,
            resolved_cue_package=resolved_cue_package,
            ontology=ontology,
            prompt_catalog=prompt_catalog,
        )
        if c_rec is None:
            return None
        prompt_row = _build_prompt_row_for_resolution(
            rec=c_rec,
            resolved_cue_package=resolved_cue_package,
            generated_question=str(c_rec["generated_question"]),
            intent_family=str(c_rec["intent_family"]),
        )
        prompt_row["implicit_resolution_mode"] = (
            "implicit_only_pass"
            if _normalized_lower(resolved_cue_package.get("cue_family")) == "natural_unique"
            else "implicit_plus_minimal_residual_cue_pass"
        )
    else:
        prompt_row = _build_prompt_row_for_resolution(rec=base_rec, resolved_cue_package=resolved_cue_package)

    try:
        prompt_text = _build_prompt_text_for_task(prompt_row)
    except Exception:
        return None

    prompt_resolution_trace = resolve_prompt_under_hview(
        rec=prompt_row,
        prompt_text=prompt_text,
        hview_confusable_universe=view_context["hview"],
        resolved_cue_package=resolved_cue_package,
    )
    repaired = apply_resolved_prompt_package(
        prompt_row,
        resolved_cue_package=resolved_cue_package,
        prompt_resolution_trace=prompt_resolution_trace,
    )
    if not bool(repaired.get("prompt_unique_under_hview")):
        return None

    repaired.update(_repaired_status_fields(rec=repaired, raw_h_view=view_context["raw_h_view"], view_context=view_context))
    repaired = _sync_navigation_difficulty_fields(repaired)
    repaired["task_family_preserved"] = True
    repaired["goal_anchor_preserved"] = True
    repaired["gt_path_preserved"] = True
    repaired["prompt_answer_leakage"] = False
    if task == "c":
        repaired["intent_family_preserved"] = True

    cue_dir = output_dir / "sidecars" / "cue_resolution" / task / scene_id / f"view_{view_id}"
    cue_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = cue_dir / f"{routing_id}.resolved_cue_package.json"
    trace_path = cue_dir / f"{routing_id}.cue_search_trace.json"
    resolution_path = cue_dir / f"{routing_id}.prompt_resolution_trace.json"
    _write_json(resolved_path, resolved_cue_package)
    _write_json(trace_path, cue_search_trace)
    _write_json(resolution_path, prompt_resolution_trace)
    repaired["resolved_cue_package_ref"] = str(resolved_path.relative_to(output_dir))
    repaired["cue_search_trace_ref"] = str(trace_path.relative_to(output_dir))
    repaired["prompt_resolution_trace_ref"] = str(resolution_path.relative_to(output_dir))
    return repaired


def build_human_aligned_source_pool(
    *,
    release_root: Path,
    output_dir: Path,
    scenes_root: Path | None = None,
    c_source_dirs: list[Path] | None = None,
    c_ontology_path: Path = DEFAULT_C_ONTOLOGY_PATH,
    c_prompt_catalog_path: Path = DEFAULT_C_PROMPT_CATALOG_PATH,
    build_release: bool = False,
    split_manifest: dict | None = None,
    target_questions: int = 1500,
    min_questions: int = 1000,
    max_questions: int = 2000,
    min_candidate_questions: int = 3000,
    max_candidate_questions: int = 4000,
    min_navigation_questions: int = 300,
) -> dict:
    release_root = Path(release_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    proxy_group_map = load_proxy_group_map(DEFAULT_PROXY_GROUPS_PATH)
    ontology = load_ontology_config(Path(c_ontology_path).resolve())
    prompt_catalog = load_c_prompt_catalog(Path(c_prompt_catalog_path).resolve())
    c_source_index = _load_c_source_index([Path(path).resolve() for path in (c_source_dirs or [])])

    source_gt_dir = output_dir / "source_gt"
    source_vqa_dir = output_dir / "source_vqa"

    summary = {
        "schema_version": "human_aligned_source_pool_v2",
        "release_root": str(release_root),
        "output_dir": str(output_dir),
        "scenes_root": str(Path(scenes_root).resolve()) if scenes_root is not None else None,
        "c_source_dirs": [str(Path(path).resolve()) for path in (c_source_dirs or [])],
        "c_prompt_catalog_path": str(Path(c_prompt_catalog_path).resolve()),
        "counts": {},
        "drops": {},
    }

    scene_cache: dict[str, dict] = {}
    view_cache: dict[tuple[str, int], dict] = {}

    passthrough_rows: dict[str, list[dict]] = {}
    for task in ("a1", "b1"):
        passthrough_rows[task] = load_jsonl(release_root / "gt" / f"gt_next_{task}.jsonl")

    rebuilt_rows: dict[str, list[dict]] = {"a2": [], "b2": [], "c": []}
    drop_counters: dict[str, Counter[str]] = {task: Counter() for task in ROUTING_TASKS}

    for task in ("a2", "b2"):
        for row in load_jsonl(release_root / "gt" / f"gt_next_{task}.jsonl"):
            rebuilt = _rebuild_routing_row(
                row=row,
                task=task,
                release_root=release_root,
                output_dir=output_dir,
                scenes_root=Path(scenes_root).resolve() if scenes_root is not None else None,
                proxy_group_map=proxy_group_map,
                ontology=ontology,
                prompt_catalog=prompt_catalog,
                c_source_index=c_source_index,
                scene_cache=scene_cache,
                view_cache=view_cache,
            )
            if rebuilt is None:
                drop_counters[task]["prompt_not_unique_under_hview"] += 1
                continue
            rebuilt_rows[task].append(rebuilt)

    for row in rebuilt_rows["b2"]:
        rebuilt_c = _rebuild_routing_row(
            row=row,
            task="c",
            release_root=release_root,
            output_dir=output_dir,
            scenes_root=Path(scenes_root).resolve() if scenes_root is not None else None,
            proxy_group_map=proxy_group_map,
            ontology=ontology,
            prompt_catalog=prompt_catalog,
            c_source_index=c_source_index,
            scene_cache=scene_cache,
            view_cache=view_cache,
        )
        if rebuilt_c is None:
            canonical_category = _normalized_lower(row.get("canonical_category") or row.get("target_category"))
            if ontology["category_to_family"].get(canonical_category or "") is None and _c_slot_key(row) not in c_source_index:
                drop_counters["c"]["missing_c_intent_source"] += 1
            else:
                drop_counters["c"]["prompt_not_unique_under_hview"] += 1
            continue
        rebuilt_rows["c"].append(rebuilt_c)

    rows_by_task = {
        "a1": passthrough_rows["a1"],
        "b1": passthrough_rows["b1"],
        "a2": rebuilt_rows["a2"],
        "b2": rebuilt_rows["b2"],
        "c": rebuilt_rows["c"],
    }

    vqa_by_task: dict[str, list[dict]] = {}
    for task, rows in rows_by_task.items():
        write_jsonl(source_gt_dir / f"gt_next_{task}.jsonl", rows)
        split_name = "train"
        vqa_by_task[task] = [
            _build_vqa_row_from_gt(rec, task, split_name=split_name, formal_release=True)
            for rec in rows
        ]
        write_jsonl(source_vqa_dir / f"vqa_next_{task}.jsonl", vqa_by_task[task])
        summary["counts"][task] = len(rows)
        if task in drop_counters:
            summary["drops"][task] = dict(sorted(drop_counters[task].items()))

    if build_release:
        split_manifest = split_manifest or _load_json(release_root / "splits" / "split_manifest.json")
        release_manifest = build_release_datasets(
            gt_dir=source_gt_dir,
            vqa_dir=source_vqa_dir,
            split_manifest=split_manifest,
            output_dir=output_dir / "release",
            target_questions=target_questions,
            min_questions=min_questions,
            max_questions=max_questions,
            min_candidate_questions=min_candidate_questions,
            max_candidate_questions=max_candidate_questions,
            min_navigation_questions=min_navigation_questions,
            validate_release=False,
            vqa_build_mode="rebuild_from_gt",
            formal_release=True,
        )
        summary["release_manifest_path"] = str(output_dir / "release" / "release_manifest.json")
        summary["release_manifest"] = release_manifest

    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild a human-aligned release source pool.")
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scenes-root", default=None)
    parser.add_argument("--c-source-dir", action="append", default=[])
    parser.add_argument("--c-ontology-path", default=str(DEFAULT_C_ONTOLOGY_PATH))
    parser.add_argument("--c-prompt-catalog-path", default=str(DEFAULT_C_PROMPT_CATALOG_PATH))
    parser.add_argument("--build-release", action="store_true")
    parser.add_argument("--target-benchmark-questions", type=int, default=1500)
    parser.add_argument("--min-benchmark-questions", type=int, default=1000)
    parser.add_argument("--max-benchmark-questions", type=int, default=2000)
    parser.add_argument("--min-candidate-questions", type=int, default=3000)
    parser.add_argument("--max-candidate-questions", type=int, default=4000)
    parser.add_argument("--min-navigation-questions", type=int, default=300)
    args = parser.parse_args()

    summary = build_human_aligned_source_pool(
        release_root=Path(args.release_root),
        output_dir=Path(args.output_dir),
        scenes_root=Path(args.scenes_root) if args.scenes_root else None,
        c_source_dirs=[Path(path) for path in args.c_source_dir],
        c_ontology_path=Path(args.c_ontology_path),
        c_prompt_catalog_path=Path(args.c_prompt_catalog_path),
        build_release=bool(args.build_release),
        target_questions=int(args.target_benchmark_questions),
        min_questions=int(args.min_benchmark_questions),
        max_questions=int(args.max_benchmark_questions),
        min_candidate_questions=int(args.min_candidate_questions),
        max_candidate_questions=int(args.max_candidate_questions),
        min_navigation_questions=int(args.min_navigation_questions),
    )
    print(json.dumps(summary["counts"], indent=2))


if __name__ == "__main__":
    main()
