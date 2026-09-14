#!/usr/bin/env python3
"""
Audit a2/b2/c rows for human-visual alignment risks.

This is a rule-based whole-package audit that traces each released row back to
the source view bundle, recomputes H(view)-style signals, and emits:

- row-level risk codes
- suspected leak stages
- high-risk shortlist
- aggregate summary
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import re
import sys
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_hview_confusable_universe import build_hview_confusable_universe
from build_navigation_gt_next import load_view_bundle_for_view
from h_view_utils import build_h_view, load_proxy_group_map
from resolve_prompt_under_hview import resolve_prompt_under_hview


SUPPORTED_TASKS = ("a2", "b2", "c")
DEFAULT_SPLITS = ("train", "val", "benchmark")
COLOR_TOKENS = {
    "black",
    "blue",
    "brown",
    "gray",
    "green",
    "grey",
    "orange",
    "pink",
    "purple",
    "red",
    "white",
    "yellow",
}
DISTANCE_CUE_PATTERNS = (
    r"\bnearest\b",
    r"\bclosest\b",
    r"\bcloser\b",
    r"\bclose by\b",
    r"\bclose to\b",
)
COLOR_BBOX_SHORT_SIDE_MIN_PX = 24
COLOR_VISIBLE_SURFACE_MIN = 0.15
HIGH_RISK_CODES = {
    "target_not_supported_under_hview",
    "same_type_ambiguous_under_hview",
    "cue_none_but_multi_under_hview",
    "color_claim_contradicted",
}
MEDIUM_RISK_CODES = {
    "same_type_uncertain_under_hview",
    "raw_zero_but_system_exists",
    "system_unique_but_raw_multi",
    "color_claim_uncertain",
    "cue_none_but_multi_under_raw_visible",
    "cue_none_but_multi_under_routing",
    "c_route_source_not_b2",
    "missing_human_visible_metadata",
    "c_prompt_mentions_distance_but_residual_none",
}

_PROXY_GROUP_MAP: dict[str, list[str]] | None = None


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


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _normalize_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _looks_like_color_claim(user_prompt: str, target_color_name: str | None) -> bool:
    prompt = _normalize_text(user_prompt) or ""
    if target_color_name is not None and target_color_name in prompt:
        return True
    prompt_tokens = set(re.findall(r"[a-z]+", prompt))
    return bool(prompt_tokens & COLOR_TOKENS)


def _prompt_mentions_distance_cue(user_prompt: str) -> bool:
    prompt = _normalize_text(user_prompt) or ""
    return any(re.search(pattern, prompt) for pattern in DISTANCE_CUE_PATTERNS)


def _load_c_proxy_group_map() -> dict[str, list[str]]:
    global _PROXY_GROUP_MAP
    if _PROXY_GROUP_MAP is None:
        _PROXY_GROUP_MAP = load_proxy_group_map(
            SCRIPT_DIR.parent / "configs" / "c_intent_family_proxy_groups_v1.json"
        )
    return _PROXY_GROUP_MAP


def _category_same_type_key(task: str, category: str | None) -> str | None:
    del task
    norm = _normalize_text(category)
    if norm is None:
        return None
    proxy_map = _load_c_proxy_group_map()
    for family, categories in proxy_map.items():
        if norm in categories:
            return family.strip().lower()
    return norm


def _row_same_type_key(task: str, row: dict, target_category: str | None) -> str | None:
    return _normalize_text(row.get("same_type_key")) or _category_same_type_key(task, target_category)


def _cue_bundle_from_row(row: dict) -> dict:
    bundle = row.get("fixed_target_prompt_cue_bundle")
    if isinstance(bundle, dict) and bundle:
        return dict(bundle)
    return {
        "cue_family": row.get("cue_family"),
        "cue_type": row.get("cue_type"),
        "cue_value": row.get("cue_value"),
        "anchor_category": row.get("anchor_category"),
        "cue_match_count": row.get("cue_match_count"),
        "residual_cue_type": row.get("residual_cue_type"),
    }


def _hydrated_promptability_trace(row: dict) -> dict:
    trace = row.get("promptability_resolution_trace")
    out = dict(trace) if isinstance(trace, dict) else {}
    cue_bundle = _cue_bundle_from_row(row)
    if out.get("resolved_cue_family") is None:
        out["resolved_cue_family"] = cue_bundle.get("cue_family") or row.get("cue_family")
    if out.get("resolved_cue_type") is None:
        out["resolved_cue_type"] = cue_bundle.get("cue_type") or row.get("cue_type")
    if out.get("anchor_category") is None:
        out["anchor_category"] = cue_bundle.get("anchor_category")
    if out.get("cue_match_count") is None:
        out["cue_match_count"] = cue_bundle.get("cue_match_count")
    if out.get("human_visible_support") is None:
        out["human_visible_support"] = row.get("human_visible_support")
    if out.get("human_visible_uniqueness") is None:
        out["human_visible_uniqueness"] = row.get("human_visible_uniqueness")
    if out.get("prompt_repair_status") is None:
        out["prompt_repair_status"] = row.get("prompt_repair_status")
    return out


def _collect_system_candidates(bundle: dict, *, task: str, target_same_type_key: str | None) -> list[dict]:
    out: list[dict] = []
    for candidate in bundle.get("routing_candidates", []):
        category = (
            _normalize_text(candidate.get("canonical_category"))
            or _normalize_text(candidate.get("target_category"))
            or _normalize_text(candidate.get("category"))
        )
        if _category_same_type_key(task, category) == target_same_type_key:
            out.append(candidate)
    return out


def _collect_raw_visible_same_type_ids(
    bundle: dict,
    *,
    task: str,
    target_same_type_key: str | None,
) -> list[int]:
    ids: list[int] = []
    for visible in bundle.get("visible_objects", []):
        category = _normalize_text(visible.get("canonical_category")) or _normalize_text(visible.get("category"))
        if _category_same_type_key(task, category) != target_same_type_key:
            continue
        obj_id = visible.get("id")
        if obj_id is None:
            continue
        try:
            ids.append(int(obj_id))
        except (TypeError, ValueError):
            continue
    return ids


def infer_base_release_root(release_package_root: Path, sample_row: dict) -> Path:
    if (release_package_root / "renders").exists():
        return release_package_root
    image_path = Path(str(sample_row.get("image_path") or ""))
    for idx, part in enumerate(image_path.parts):
        if part == "task_outputs":
            return Path(*image_path.parts[:idx])
    raise ValueError(
        f"cannot infer base release root from release_package_root={release_package_root} "
        f"and image_path={image_path}"
    )


def _build_h_view_for_row(bundle: dict, *, task: str) -> dict:
    del task
    proxy_group_map = _load_c_proxy_group_map()
    return build_h_view(
        bundle,
        routing_rows=bundle.get("routing_candidates", []),
        proxy_group_map=proxy_group_map,
        same_type_mode="family_proxy",
    )


def _build_hview_confusable_universe_for_row(bundle: dict, gt_row: dict) -> dict:
    scene_id = str(gt_row.get("scene_id") or bundle.get("scene_id") or "")
    view_id_raw = gt_row.get("view_id")
    try:
        view_id = int(view_id_raw) if view_id_raw is not None else int(bundle.get("view_id") or 0)
    except (TypeError, ValueError):
        view_id = 0
    universe, _ = build_hview_confusable_universe(
        scene_id=scene_id,
        view_id=view_id,
        bundle=bundle,
        layout=[],
        metadata={},
        proxy_group_map=_load_c_proxy_group_map(),
    )
    return universe


def _revalidate_prompt_resolution(
    *,
    task: str,
    vqa_row: dict,
    gt_row: dict,
    hview_confusable_universe: dict,
    target_same_type_key: str | None,
) -> dict:
    rec = dict(gt_row)
    if target_same_type_key is not None and rec.get("same_type_key") is None:
        rec["same_type_key"] = target_same_type_key
    if rec.get("promptability_resolution_trace") is None:
        rec["promptability_resolution_trace"] = _hydrated_promptability_trace(gt_row)
    else:
        rec["promptability_resolution_trace"] = _hydrated_promptability_trace(rec)
    if rec.get("fixed_target_prompt_cue_bundle") is None:
        rec["fixed_target_prompt_cue_bundle"] = _cue_bundle_from_row(gt_row)

    prompt_text = str(vqa_row.get("user_prompt") or "")
    if task == "c":
        if rec.get("generated_question") is None:
            rec["generated_question"] = prompt_text
        if rec.get("route_semantics") is None:
            rec["route_semantics"] = "embodied"
        if rec.get("implicit_resolution_mode") is None:
            rec["implicit_resolution_mode"] = "implicit_plus_minimal_residual_cue_pass"

    return resolve_prompt_under_hview(
        rec=rec,
        prompt_text=prompt_text,
        hview_confusable_universe=hview_confusable_universe,
    )


def _target_candidate_from_bundle(bundle: dict, row: dict) -> dict | None:
    routing_id = _normalize_text(row.get("routing_id"))
    target_id = row.get("target_id")
    if target_id is not None:
        try:
            target_id = int(target_id)
        except (TypeError, ValueError):
            target_id = None
    for candidate in bundle.get("routing_candidates", []):
        candidate_routing_id = _normalize_text(candidate.get("routing_id"))
        candidate_target_id = candidate.get("target_id")
        try:
            if candidate_target_id is not None:
                candidate_target_id = int(candidate_target_id)
        except (TypeError, ValueError):
            candidate_target_id = None
        if routing_id is not None and candidate_routing_id == routing_id:
            return candidate
        if target_id is not None and candidate_target_id == target_id:
            return candidate
    return None


def _color_claim_status(*, user_prompt: str, target_row: dict, target_candidate: dict | None) -> str:
    target_color = (
        _normalize_text(target_row.get("target_color_name"))
        or _normalize_text(target_row.get("reference_color_name"))
        or _normalize_text((target_row.get("instruction_identity") or {}).get("reference_color_name"))
        or _normalize_text((target_candidate or {}).get("reference_color_name"))
        or _normalize_text((target_candidate or {}).get("target_color_name"))
        or _normalize_text((target_candidate or {}).get("visible_color_name"))
    )
    if not _looks_like_color_claim(user_prompt, target_color):
        return "none"
    if target_candidate is None:
        return "uncertain"

    candidate_visible_color = _normalize_text(target_candidate.get("visible_color_name"))
    candidate_material_color = _normalize_text(target_candidate.get("material_color_name"))
    if target_color is not None and candidate_visible_color is not None and target_color != candidate_visible_color:
        return "contradicted"
    if (
        target_color is not None
        and candidate_visible_color is None
        and candidate_material_color is not None
        and target_color != candidate_material_color
    ):
        return "contradicted"

    routing_complexity = target_candidate.get("routing_complexity") or {}
    bbox_short_side = routing_complexity.get("target_bbox_short_side_px")
    visible_surface = routing_complexity.get("target_visible_surface_ratio")
    try:
        bbox_short_side = float(bbox_short_side) if bbox_short_side is not None else None
    except (TypeError, ValueError):
        bbox_short_side = None
    try:
        visible_surface = float(visible_surface) if visible_surface is not None else None
    except (TypeError, ValueError):
        visible_surface = None

    if candidate_visible_color is None and candidate_material_color is None:
        return "uncertain"
    if bbox_short_side is not None and bbox_short_side < COLOR_BBOX_SHORT_SIDE_MIN_PX:
        return "uncertain"
    if visible_surface is not None and visible_surface < COLOR_VISIBLE_SURFACE_MIN:
        return "uncertain"
    return "supported"


def _human_alignment_status(risk_codes: list[str]) -> str:
    if any(code in HIGH_RISK_CODES for code in risk_codes):
        return "contradicted"
    if any(code in MEDIUM_RISK_CODES for code in risk_codes):
        return "uncertain"
    return "aligned"


def _suspected_leak_stages(*, split: str, task: str, risk_codes: list[str]) -> list[str]:
    stages: list[str] = []
    if any(
        code in {
            "target_not_supported_under_hview",
            "same_type_uncertain_under_hview",
            "same_type_ambiguous_under_hview",
            "raw_zero_but_system_exists",
            "system_unique_but_raw_multi",
        }
        for code in risk_codes
    ):
        stages.append("candidate_build")
    if any(code in {"color_claim_contradicted", "color_claim_uncertain"} for code in risk_codes):
        stages.append("prompt_generation")
    if "missing_human_visible_metadata" in risk_codes:
        stages.append("release_packaging")
    if task == "c" and any(code.startswith("cue_none_but_") or code == "c_route_source_not_b2" for code in risk_codes):
        stages.append("c_rewrite")
    if task == "c" and "c_prompt_mentions_distance_but_residual_none" in risk_codes and "c_rewrite" not in stages:
        stages.append("c_rewrite")
    if split in {"benchmark", "val"} and risk_codes:
        stages.append("split_filter")
    return stages


def analyze_routing_row(
    *,
    split: str,
    task: str,
    vqa_row: dict,
    gt_row: dict,
    bundle: dict,
    hview_confusable_universe: dict,
) -> dict:
    question_id = str(gt_row.get("question_id") or vqa_row.get("question_id") or "")
    target_candidate = _target_candidate_from_bundle(bundle, gt_row)
    target_category = (
        _normalize_text(gt_row.get("canonical_category"))
        or _normalize_text(gt_row.get("target_category"))
        or _normalize_text((target_candidate or {}).get("canonical_category"))
        or _normalize_text((target_candidate or {}).get("target_category"))
    )
    target_same_type_key = _row_same_type_key(task, gt_row, target_category)
    h_view = _build_h_view_for_row(bundle, task=task)
    same_type_objects = [
        obj for obj in h_view.get("h_view_objects", []) if obj.get("same_type_key") == target_same_type_key
    ]
    target_id = gt_row.get("target_id")
    try:
        target_id = int(target_id) if target_id is not None else None
    except (TypeError, ValueError):
        target_id = None
    target_h_view = next((obj for obj in same_type_objects if int(obj.get("object_id", -1)) == target_id), None)
    supported_visible = [
        obj
        for obj in same_type_objects
        if obj.get("support_status") == "supported" and obj.get("visibility_status") == "visible"
    ]
    ambiguity_count_hview = len({int(obj["object_id"]) for obj in supported_visible})
    uncertain_same_type = [
        obj for obj in same_type_objects
        if obj.get("support_status") != "supported" or obj.get("visibility_status") != "visible"
    ]
    target_supported = bool(
        (hview_confusable_universe.get("target_supported_under_hview") or {}).get(str(target_id), False)
    ) or (
        target_h_view is not None
        and target_h_view.get("support_status") == "supported"
        and target_h_view.get("visibility_status") == "visible"
    )

    raw_visible_same_type_ids = _collect_raw_visible_same_type_ids(
        bundle, task=task, target_same_type_key=target_same_type_key
    )
    system_candidates = _collect_system_candidates(
        bundle, task=task, target_same_type_key=target_same_type_key
    )
    instruction_selection = (
        gt_row.get("instruction_selection")
        or (target_candidate or {}).get("instruction_selection")
        or {}
    )
    system_cohort_size = int(instruction_selection.get("cohort_size") or 1)
    color_claim_status = _color_claim_status(
        user_prompt=str(vqa_row.get("user_prompt") or ""),
        target_row=gt_row,
        target_candidate=target_candidate,
    )
    cue_bundle = _cue_bundle_from_row(gt_row)
    effective_cue_family = (
        _normalize_text(cue_bundle.get("cue_family"))
        or _normalize_text(gt_row.get("cue_family"))
    )
    prompt_resolution_trace = _revalidate_prompt_resolution(
        task=task,
        vqa_row=vqa_row,
        gt_row=gt_row,
        hview_confusable_universe=hview_confusable_universe,
        target_same_type_key=target_same_type_key,
    )
    prompt_resolution_pass = _normalize_text(prompt_resolution_trace.get("resolver_result")) == "pass"
    prompt_resolution_cue_family = (
        _normalize_text(prompt_resolution_trace.get("resolved_cue_family"))
        or effective_cue_family
    )
    prompt_uniquely_resolved = prompt_resolution_pass and prompt_resolution_cue_family in {
        "natural_unique",
        "relation",
        "color",
    }

    risk_codes: list[str] = []
    if not target_supported:
        risk_codes.append("target_not_supported_under_hview")
    elif uncertain_same_type and not prompt_uniquely_resolved:
        risk_codes.append("same_type_uncertain_under_hview")
    elif ambiguity_count_hview > 1 and not prompt_uniquely_resolved:
        risk_codes.append("same_type_ambiguous_under_hview")

    if len(raw_visible_same_type_ids) == 0 and len(system_candidates) > 0:
        risk_codes.append("raw_zero_but_system_exists")
    if len(raw_visible_same_type_ids) > max(system_cohort_size, 1) and not prompt_uniquely_resolved:
        risk_codes.append("system_unique_but_raw_multi")

    if color_claim_status == "contradicted":
        risk_codes.append("color_claim_contradicted")
    elif color_claim_status == "uncertain":
        risk_codes.append("color_claim_uncertain")

    residual_cue_type = _normalize_text(gt_row.get("residual_cue_type"))
    human_visible_fields = (
        gt_row.get("human_visible_support"),
        gt_row.get("human_visible_uniqueness"),
        gt_row.get("human_visible_legality"),
        gt_row.get("prompt_truth_tier"),
    )
    if all(value is None for value in human_visible_fields):
        risk_codes.append("missing_human_visible_metadata")
    if task == "c":
        if (
            effective_cue_family in (None, "none")
            and residual_cue_type in (None, "none")
            and _prompt_mentions_distance_cue(str(vqa_row.get("user_prompt") or ""))
        ):
            risk_codes.append("c_prompt_mentions_distance_but_residual_none")
        if effective_cue_family in (None, "none") and residual_cue_type in (None, "none"):
            if ambiguity_count_hview > 1 or uncertain_same_type:
                risk_codes.append("cue_none_but_multi_under_hview")
            if len(raw_visible_same_type_ids) > 1:
                risk_codes.append("cue_none_but_multi_under_raw_visible")
            if len(system_candidates) > 1:
                risk_codes.append("cue_none_but_multi_under_routing")
        route_source_task = _normalize_text(gt_row.get("route_source_task"))
        if route_source_task not in (None, "b2"):
            risk_codes.append("c_route_source_not_b2")

    seen = set()
    risk_codes = [code for code in risk_codes if not (code in seen or seen.add(code))]
    suspected_leak_stages = _suspected_leak_stages(split=split, task=task, risk_codes=risk_codes)
    alignment_status = _human_alignment_status(risk_codes)

    return {
        "question_id": question_id,
        "split": split,
        "task": task,
        "scene_id": gt_row.get("scene_id"),
        "view_id": gt_row.get("view_id"),
        "routing_id": gt_row.get("routing_id"),
        "target_id": target_id,
        "target_category": target_category,
        "intent_family": gt_row.get("intent_family"),
        "user_prompt": vqa_row.get("user_prompt"),
        "human_visual_alignment_status": alignment_status,
        "risk_codes": risk_codes,
        "suspected_leak_stages": suspected_leak_stages,
        "raw_visible_same_type_count": len(raw_visible_same_type_ids),
        "raw_visible_same_type_ids": raw_visible_same_type_ids,
        "system_candidate_count": len(system_candidates),
        "system_cohort_size": system_cohort_size,
        "ambiguity_count_hview": ambiguity_count_hview,
        "uncertain_same_type_count_hview": len(uncertain_same_type),
        "target_supported_under_hview": bool(target_supported),
        "residual_cue_type": residual_cue_type,
        "cue_family": effective_cue_family,
        "color_claim_status": color_claim_status,
        "prompt_resolution_result": prompt_resolution_trace.get("resolver_result"),
        "prompt_resolution_drop_reason_code": prompt_resolution_trace.get("drop_reason_code"),
        "prompt_resolution_surface_contract_status": prompt_resolution_trace.get("surface_contract_status"),
        "prompt_resolution_cue_family": prompt_resolution_trace.get("resolved_cue_family"),
    }


def _build_summary(rows: list[dict]) -> dict:
    status_counts = Counter(row["human_visual_alignment_status"] for row in rows)
    risk_code_counts = Counter()
    stage_counts = Counter()
    per_split_task = defaultdict(lambda: {"rows": 0, "contradicted": 0, "uncertain": 0, "aligned": 0})

    for row in rows:
        for code in row["risk_codes"]:
            risk_code_counts[code] += 1
        for stage in row["suspected_leak_stages"]:
            stage_counts[stage] += 1
        key = f"{row['split']}::{row['task']}"
        bucket = per_split_task[key]
        bucket["rows"] += 1
        bucket[row["human_visual_alignment_status"]] += 1

    return {
        "schema_version": "human_visual_alignment_audit_v1",
        "rows_total": len(rows),
        "status_counts": dict(status_counts),
        "risk_code_counts": dict(risk_code_counts),
        "suspected_leak_stage_counts": dict(stage_counts),
        "per_split_task": dict(per_split_task),
    }


def audit_release_package(
    *,
    release_package_root: Path,
    output_dir: Path,
    base_release_root: Path | None = None,
    splits: tuple[str, ...] = DEFAULT_SPLITS,
    tasks: tuple[str, ...] = SUPPORTED_TASKS,
) -> dict:
    package_root = release_package_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    bundle_cache: dict[tuple[str, int], dict] = {}
    hview_cache: dict[tuple[str, int], dict] = {}
    inferred_base_release_root: Path | None = None

    for split in splits:
        gt_dir = package_root / split / "gt"
        vqa_dir = package_root / split / "vqa"
        for task in tasks:
            gt_rows = load_jsonl(gt_dir / f"gt_next_{task}.jsonl")
            vqa_rows = load_jsonl(vqa_dir / f"vqa_next_{task}.jsonl")
            if not gt_rows or not vqa_rows:
                continue
            vqa_by_qid = {str(row.get("question_id")): row for row in vqa_rows}
            if inferred_base_release_root is None:
                inferred_base_release_root = infer_base_release_root(package_root, gt_rows[0])
            actual_base_release_root = (base_release_root or inferred_base_release_root).resolve()

            for gt_row in gt_rows:
                qid = str(gt_row.get("question_id"))
                vqa_row = vqa_by_qid.get(qid)
                if vqa_row is None:
                    continue
                scene_id = str(gt_row["scene_id"])
                view_id = int(gt_row["view_id"])
                cache_key = (scene_id, view_id)
                bundle = bundle_cache.get(cache_key)
                if bundle is None:
                    bundle = load_view_bundle_for_view(actual_base_release_root / "renders" / scene_id, view_id)
                    bundle_cache[cache_key] = bundle
                hview_confusable_universe = hview_cache.get(cache_key)
                if hview_confusable_universe is None:
                    hview_confusable_universe = _build_hview_confusable_universe_for_row(bundle, gt_row)
                    hview_cache[cache_key] = hview_confusable_universe
                rows.append(
                    analyze_routing_row(
                        split=split,
                        task=task,
                        vqa_row=vqa_row,
                        gt_row=gt_row,
                        bundle=bundle,
                        hview_confusable_universe=hview_confusable_universe,
                    )
                )

    rows.sort(key=lambda row: (row["split"], row["task"], row["question_id"]))
    summary = _build_summary(rows)
    high_risk = [row for row in rows if row["human_visual_alignment_status"] != "aligned"]
    high_risk.sort(
        key=lambda row: (
            row["human_visual_alignment_status"] != "contradicted",
            -len(row["risk_codes"]),
            row["split"],
            row["task"],
            row["question_id"],
        )
    )

    write_jsonl(output_dir / "all_audit_rows.jsonl", rows)
    write_jsonl(output_dir / "high_risk_shortlist.jsonl", high_risk)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    return {
        "rows": rows,
        "high_risk": high_risk,
        "summary": summary,
    }


def parse_csv_arg(raw: str, *, allowed: tuple[str, ...]) -> tuple[str, ...]:
    requested = tuple(part.strip() for part in raw.split(",") if part.strip())
    invalid = [part for part in requested if part not in allowed]
    if invalid:
        raise ValueError(f"unsupported values {invalid}; allowed={allowed}")
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit released routing rows for human-visual alignment risks.")
    parser.add_argument("--release-package-root", required=True, help="Release package root containing split/vqa and split/gt")
    parser.add_argument("--output-dir", required=True, help="Directory for audit outputs")
    parser.add_argument("--base-release-root", default=None, help="Optional root that contains renders/ and task_outputs/")
    parser.add_argument("--splits", default="train,val,benchmark", help="Comma-separated splits")
    parser.add_argument("--tasks", default="a2,b2,c", help="Comma-separated tasks")
    args = parser.parse_args()

    result = audit_release_package(
        release_package_root=Path(args.release_package_root),
        output_dir=Path(args.output_dir),
        base_release_root=Path(args.base_release_root) if args.base_release_root else None,
        splits=parse_csv_arg(args.splits, allowed=DEFAULT_SPLITS),
        tasks=parse_csv_arg(args.tasks, allowed=SUPPORTED_TASKS),
    )
    print(json.dumps(result["summary"], indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
