#!/usr/bin/env python3
"""
Build train / val / benchmark release subsets from full Next GT/VQA outputs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_policy import build_manifest, load_jsonl, write_jsonl
from build_vqa_questions_next import (
    build_a1_prompt,
    build_a2_prompt,
    build_c_prompt,
    load_uncertain_adjudication_sidecar,
    validate_record_for_task,
)
from build_truth_aligned_release_manifest import (
    build_truth_aligned_release_package as build_truth_aligned_release_package_layout,
)
from prompt_truth_contract import (
    build_canonical_prompt_truth_contract,
    extract_navigation_truth_contract_fields,
)


TASKS = ("a1", "b1", "a2", "b2", "c")
SPLITS = ("train", "val", "benchmark")
ALLOWED_VQA_BUILD_MODES = ("rebuild_from_gt", "source_passthrough_preview")
DEFAULT_TARGET_QUESTIONS = 1500
DEFAULT_MIN_QUESTIONS = 1350
DEFAULT_MAX_QUESTIONS = 1650
DEFAULT_MIN_CANDIDATE_QUESTIONS = 3000
DEFAULT_MAX_CANDIDATE_QUESTIONS = 4000
DEFAULT_TARGET_NAVIGATION_QUESTIONS = 300
DEFAULT_VAL_TARGET_QUESTIONS = 1200
DEFAULT_VAL_MIN_QUESTIONS = 1000
DEFAULT_VAL_MAX_QUESTIONS = 1400
DEFAULT_VAL_TASK_WEIGHTS = {
    "a1": 1.0,
    "b1": 1.0,
    "a2": 1.5,
    "b2": 1.5,
    "c": 1.0,
}


def _collect_rows_by_task(root: Path, prefix: str) -> dict[str, list[dict]]:
    rows_by_task: dict[str, list[dict]] = {}
    for task in TASKS:
        rows_by_task[task] = load_jsonl(root / f"{prefix}_next_{task}.jsonl")
    return rows_by_task


def _write_rows_by_task(root: Path, prefix: str, rows_by_task: dict[str, list[dict]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for task, rows in rows_by_task.items():
        if not rows:
            continue
        write_jsonl(root / f"{prefix}_next_{task}.jsonl", rows)


def _build_vqa_row_from_gt(
    rec: dict,
    task: str,
    *,
    split_name: str,
    adjudication_by_sample_id: dict[str, dict] | None = None,
    formal_release: bool = False,
) -> dict:
    validate_record_for_task(
        rec,
        task,
        split=split_name,
        adjudication_by_sample_id=adjudication_by_sample_id,
    )
    if task in ("a1", "b1"):
        prompt = build_a1_prompt(rec, embodied=(task == "b1"))
        answer = rec["walkable_ids"]
        contract = build_canonical_prompt_truth_contract(
            rec,
            prompt=prompt,
            answer=answer,
            formal_release=formal_release,
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
                "human_visible_uniqueness": rec.get("human_visible_uniqueness"),
                "prompt_unique_under_hview": rec.get("prompt_unique_under_hview"),
                "promptability_status": rec.get("promptability_status"),
                "promptability_resolution_trace": rec.get("promptability_resolution_trace"),
                "cue_family": rec.get("cue_family"),
                "fixed_target_prompt_cue_bundle": rec.get("fixed_target_prompt_cue_bundle"),
                "prompt_answer_leakage": rec.get("prompt_answer_leakage"),
                "task_family": rec.get("task_family"),
                "goal_anchor_id": rec.get("goal_anchor_id"),
                "gt_path_id": rec.get("gt_path_id"),
                "task_family_preserved": rec.get("task_family_preserved"),
                "goal_anchor_preserved": rec.get("goal_anchor_preserved"),
                "gt_path_preserved": rec.get("gt_path_preserved"),
                "acceptable_goal_ids": rec.get("acceptable_goal_ids"),
                "goal_ring_min_distance_m": rec.get("goal_ring_min_distance_m"),
                "goal_ring_threshold_m": rec.get("goal_ring_threshold_m"),
                "reference_path_display_ids": rec.get("reference_path_display_ids"),
                "direct_pairs_ref": rec.get("direct_pairs_ref"),
                "route_semantics": rec.get("route_semantics"),
                **contract,
            },
        }
        return vqa

    if task in ("a2", "b2"):
        prompt = build_a2_prompt(rec, embodied=(task == "b2"))
        contract = build_canonical_prompt_truth_contract(
            rec,
            prompt=prompt,
            answer=rec["path_ids"],
            formal_release=formal_release,
        )
        return {
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
                "answer": rec["path_ids"],
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
                **extract_navigation_truth_contract_fields(rec),
                **contract,
            },
        }

    if task == "c":
        prompt = build_c_prompt(rec)
        contract = build_canonical_prompt_truth_contract(
            rec,
            prompt=prompt,
            answer=rec["path_ids"],
            formal_release=formal_release,
        )
        return {
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

    raise ValueError(f"unsupported task for vqa rebuild: {task}")


def _rebuild_vqa_rows_by_task_from_gt(
    rows_by_task: dict[str, list[dict]],
    *,
    split_name: str,
    adjudication_by_sample_id: dict[str, dict] | None = None,
    formal_release: bool = False,
) -> dict[str, list[dict]]:
    rebuilt: dict[str, list[dict]] = {}
    for task, rows in rows_by_task.items():
        rebuilt[task] = [
            _build_vqa_row_from_gt(
                rec,
                task,
                split_name=split_name,
                adjudication_by_sample_id=adjudication_by_sample_id,
                formal_release=formal_release,
            )
            for rec in rows
        ]
    return rebuilt


def _index_by_question_id(rows_by_task: dict[str, list[dict]]) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for rows in rows_by_task.values():
        for row in rows:
            question_id = str(row.get("question_id", "")).strip()
            if not question_id:
                raise ValueError("row missing required field: question_id")
            if question_id in indexed:
                raise ValueError(f"duplicate question_id detected: {question_id}")
            indexed[question_id] = row
    return indexed


def _question_alignment_signature(row: dict) -> tuple[str, str, int, str | None]:
    return (
        str(row.get("task")),
        str(row.get("scene_id")),
        int(row.get("view_id")),
        str(row.get("routing_id")) if row.get("routing_id") is not None else None,
    )


def _validate_gt_vqa_alignment(
    gt_by_question_id: dict[str, dict],
    vqa_by_question_id: dict[str, dict],
) -> None:
    for question_id, gt_row in gt_by_question_id.items():
        vqa_row = vqa_by_question_id[question_id]
        if _question_alignment_signature(gt_row) != _question_alignment_signature(vqa_row):
            raise ValueError(
                "GT/VQA structural mismatch for question_id="
                f"{question_id}: gt={_question_alignment_signature(gt_row)} "
                f"vqa={_question_alignment_signature(vqa_row)}"
            )


def _split_name_for_row(row: dict, scene_to_split: dict[str, str]) -> str:
    scene_id = str(row.get("scene_id", "")).strip()
    if not scene_id:
        raise ValueError("row missing required field: scene_id")
    split_name = scene_to_split.get(scene_id)
    if split_name not in SPLITS:
        raise ValueError(f"scene_id={scene_id} missing valid split assignment")
    return str(split_name)


def _fallback_non_benchmark_split(split_name: str) -> str:
    return "train" if split_name == "benchmark" else split_name


def _advisory_scene_split_counts(split_manifest: dict) -> dict[str, int]:
    scene_to_split = split_manifest.get("scene_to_split")
    if not isinstance(scene_to_split, dict):
        raise ValueError("split_manifest missing required field: scene_to_split")
    counts = Counter()
    for scene_id, split_name in scene_to_split.items():
        _ = str(scene_id)
        normalized = str(split_name)
        if normalized not in SPLITS:
            raise ValueError(f"invalid advisory split assignment: scene_id={scene_id} split={split_name}")
        counts[normalized] += 1
    return {split_name: int(counts.get(split_name, 0)) for split_name in SPLITS}


def _derive_val_scene_target_count(
    split_manifest: dict,
    *,
    scene_rows: list[dict],
    excluded_scene_ids: set[str],
    excluded_source_group_ids: set[str],
) -> int:
    candidates = [
        scene
        for scene in scene_rows
        if str(scene["scene_id"]) not in excluded_scene_ids
        and str(scene["source_group_id"]) not in excluded_source_group_ids
    ]
    candidate_scene_count = len(candidates)
    if candidate_scene_count <= 0:
        return 0

    split_ratios = split_manifest.get("split_ratios")
    if isinstance(split_ratios, dict):
        try:
            train_ratio = float(split_ratios.get("train", 0.0) or 0.0)
            val_ratio = float(split_ratios.get("val", 0.0) or 0.0)
        except (TypeError, ValueError):
            train_ratio = 0.0
            val_ratio = 0.0
        non_benchmark_ratio = train_ratio + val_ratio
        if val_ratio > 0.0 and non_benchmark_ratio > 0.0:
            derived = int(round(candidate_scene_count * (val_ratio / non_benchmark_ratio)))
            return min(candidate_scene_count, max(1, derived))

    advisory_split_counts = _advisory_scene_split_counts(split_manifest)
    return min(candidate_scene_count, int(advisory_split_counts.get("val", 0)))


def _source_dataset(scene_id: str) -> str:
    if "__" not in scene_id:
        return "unknown"
    return scene_id.split("__", 1)[0]


def _source_group(scene_id: str) -> str:
    parts = [part for part in scene_id.split("__") if part]
    if len(parts) < 2:
        return scene_id
    if parts[0] == "matterport3d" and len(parts) >= 3:
        return "__".join(parts[:2])
    return scene_id


def _safe_mean(values: list[int | float]) -> float:
    if not values:
        return 0.0
    return round(float(mean(values)), 3)


def _weighted_task_targets(total_questions: int, weights: dict[str, float]) -> dict[str, int]:
    if total_questions <= 0:
        return {task: 0 for task in weights}
    weight_total = float(sum(weights.values()))
    if weight_total <= 0:
        raise ValueError("weights must sum to a positive value")

    raw_targets = {
        task: float(total_questions) * float(weight) / weight_total
        for task, weight in weights.items()
    }
    targets = {task: int(value) for task, value in raw_targets.items()}
    remainder = int(total_questions - sum(targets.values()))
    ranked_tasks = sorted(
        weights,
        key=lambda task: (raw_targets[task] - targets[task], float(weights[task]), task),
        reverse=True,
    )
    for idx in range(remainder):
        targets[ranked_tasks[idx % len(ranked_tasks)]] += 1
    return targets


def _derive_val_task_targets(target_questions: int = DEFAULT_VAL_TARGET_QUESTIONS) -> dict[str, int]:
    return _weighted_task_targets(target_questions, DEFAULT_VAL_TASK_WEIGHTS)


def _bundle_identity(row: dict) -> tuple[str, str]:
    task = str(row.get("task", "")).lower()
    scene_id = str(row["scene_id"])
    view_id = int(row["view_id"])
    if task in {"a1", "b1"}:
        return ("view_bundle", f"{scene_id}::v{view_id}")
    return ("route_bundle", f"{scene_id}::v{view_id}::{row['routing_id']}")


def _summarize_split(rows_by_task: dict[str, list[dict]]) -> dict:
    question_counts = {task: len(rows) for task, rows in rows_by_task.items() if rows}
    all_rows = [row for rows in rows_by_task.values() for row in rows]
    return {
        "question_counts_by_task": question_counts,
        "questions_total": int(sum(question_counts.values())),
        "scene_count": len(
            {
                str(row["scene_id"])
                for rows in rows_by_task.values()
                for row in rows
            }
        ),
        "reporting": {
            "question_variants": int(sum(question_counts.values())),
            "unique_bundles": len({_bundle_identity(row) for row in all_rows}),
            "unique_scenes": len({str(row["scene_id"]) for row in all_rows}),
            "unique_source_groups": len({_source_group(str(row["scene_id"])) for row in all_rows}),
        },
    }


def _build_hard_candidate_artifacts(
    *,
    all_selection_rows: list[dict],
    selection_source_dir: Path,
    selection_source_prefix: str,
    selection_dir: Path,
) -> tuple[dict, list[dict], Path, Path]:
    labels_path = selection_dir / "hard_candidate_labels.jsonl"
    manifest_path = selection_dir / "hard_candidate_manifest.json"
    candidate_manifest, label_rows = build_manifest(
        all_selection_rows,
        source_files=[
            str(selection_source_dir / f"{selection_source_prefix}_next_{task}.jsonl")
            for task in TASKS
            if (selection_source_dir / f"{selection_source_prefix}_next_{task}.jsonl").exists()
        ],
        labels_path=labels_path,
    )
    write_jsonl(labels_path, label_rows)
    with open(manifest_path, "w") as f:
        json.dump(candidate_manifest, f, indent=2)
    return candidate_manifest, label_rows, labels_path, manifest_path


def _summarize_candidate_scenes(label_rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in label_rows:
        if not bool(row.get("hard_candidate")):
            continue
        grouped[str(row["scene_id"])].append(row)

    scene_rows: list[dict] = []
    for scene_id, rows in sorted(grouped.items()):
        scores = [int(row.get("hardness_score", 0)) for row in rows]
        bundle_types = sorted({str(row["bundle_type"]) for row in rows})
        tasks = Counter(str(row["task"]) for row in rows)
        families = Counter(str(row["tier_family"]) for row in rows)
        navigation_scores = [int(row.get("hardness_score", 0)) for row in rows if str(row.get("tier_family")) == "navigation"]
        affordance_scores = [int(row.get("hardness_score", 0)) for row in rows if str(row.get("tier_family")) == "affordance"]
        scene_rows.append(
            {
                "scene_id": scene_id,
                "source_dataset": _source_dataset(scene_id),
                "source_group_id": _source_group(scene_id),
                "candidate_question_count": len(rows),
                "candidate_bundle_count": len({str(row["bundle_key"]) for row in rows}),
                "bundle_types": bundle_types,
                "bundle_diversity": len(bundle_types),
                "task_counts": dict(sorted(tasks.items())),
                "family_counts": dict(sorted(families.items())),
                "mean_hardness_score": _safe_mean(scores),
                "max_hardness_score": max(scores),
                "navigation_question_count": len(navigation_scores),
                "navigation_mean_score": _safe_mean(navigation_scores),
                "affordance_question_count": len(affordance_scores),
                "affordance_mean_score": _safe_mean(affordance_scores),
            }
        )
    return scene_rows


def _summarize_all_scenes(
    rows_by_task: dict[str, list[dict]],
    *,
    hard_candidate_question_ids: set[str],
) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for rows in rows_by_task.values():
        for row in rows:
            grouped[str(row["scene_id"])].append(row)

    scene_rows: list[dict] = []
    for scene_id, rows in sorted(grouped.items()):
        tasks = Counter(str(row["task"]) for row in rows)
        families = Counter(str(row.get("tier_family", "unknown")) for row in rows)
        hard_rows = [row for row in rows if str(row["question_id"]) in hard_candidate_question_ids]
        scene_rows.append(
            {
                "scene_id": scene_id,
                "source_dataset": _source_dataset(scene_id),
                "source_group_id": _source_group(scene_id),
                "questions_total": len(rows),
                "task_counts": dict(sorted(tasks.items())),
                "family_counts": dict(sorted(families.items())),
                "hard_candidate_question_count": len(hard_rows),
                "hard_candidate_bundle_count": len({_bundle_identity(row) for row in hard_rows}),
                "has_hard_candidates": bool(hard_rows),
            }
        )
    return scene_rows


def _can_use_balanced_val_selection(
    scene_rows: list[dict],
    *,
    excluded_scene_ids: set[str],
    excluded_source_group_ids: set[str],
    min_questions: int,
    target_task_counts: dict[str, int],
) -> bool:
    candidates = [
        scene
        for scene in scene_rows
        if str(scene["scene_id"]) not in excluded_scene_ids
        and str(scene["source_group_id"]) not in excluded_source_group_ids
    ]
    if len(candidates) < 100:
        return False
    if sum(int(scene["questions_total"]) for scene in candidates) < int(min_questions):
        return False
    for task, target in target_task_counts.items():
        available = sum(int(scene["task_counts"].get(task, 0)) for scene in candidates)
        if available < int(target):
            return False
    return True


def _select_val_scenes(
    scene_rows: list[dict],
    *,
    excluded_scene_ids: set[str],
    excluded_source_group_ids: set[str],
    target_scene_count: int,
    target_questions: int | None = None,
    min_questions: int | None = None,
    max_questions: int | None = None,
    target_task_counts: dict[str, int] | None = None,
) -> tuple[set[str], dict]:
    if target_scene_count <= 0:
        empty_summary = {
            "schema_version": "val_scene_selection_v2",
            "selection_order": ["benchmark", "val", "train"],
            "selection_policy": "post_rebuild_low_hardness_preference",
            "advisory_target_scene_count": 0,
            "candidate_scene_count": 0,
            "selected_scene_ids": [],
            "selected_source_group_ids": [],
        }
        return set(), empty_summary

    candidates = [
        scene
        for scene in scene_rows
        if str(scene["scene_id"]) not in excluded_scene_ids
        and str(scene["source_group_id"]) not in excluded_source_group_ids
    ]

    if (
        target_task_counts is not None
        and target_questions is not None
        and min_questions is not None
        and max_questions is not None
    ):
        selected: list[dict] = []
        source_counts: Counter[str] = Counter()
        selected_task_counts = {task: 0 for task in TASKS}
        selected_questions_total = 0
        remaining = list(candidates)

        def add_scene(scene: dict) -> None:
            nonlocal selected_questions_total
            selected.append(scene)
            selected_questions_total += int(scene["questions_total"])
            source_counts[str(scene["source_dataset"])] += 1
            for task in TASKS:
                selected_task_counts[task] += int(scene["task_counts"].get(task, 0))

        def projected_total(scene: dict) -> int:
            return selected_questions_total + int(scene["questions_total"])

        def common_penalty(
            scene: dict,
            *,
            question_weight: float,
            hard_question_weight: float,
            hard_bundle_weight: float,
            source_weight: float,
        ) -> float:
            overflow = max(0, projected_total(scene) - int(max_questions))
            return (
                int(scene["hard_candidate_question_count"]) * hard_question_weight
                + int(scene["hard_candidate_bundle_count"]) * hard_bundle_weight
                + int(scene["questions_total"]) * question_weight
                + source_counts[str(scene["source_dataset"])] * source_weight
                + float(overflow) * 50.0
            )

        while (
            remaining
            and len(selected) < target_scene_count
            and selected_task_counts["c"] < int(target_task_counts.get("c", 0))
            and selected_questions_total < int(max_questions)
        ):
            c_candidates = [scene for scene in remaining if int(scene["task_counts"].get("c", 0)) > 0]
            if not c_candidates:
                break
            best_scene = max(
                c_candidates,
                key=lambda scene: (
                    int(scene["task_counts"].get("c", 0)) * 40.0
                    + (int(scene["task_counts"].get("a2", 0)) + int(scene["task_counts"].get("b2", 0))) * 8.0
                    - common_penalty(
                        scene,
                        question_weight=2.0,
                        hard_question_weight=60.0,
                        hard_bundle_weight=40.0,
                        source_weight=2.0,
                    ),
                    -int(scene["questions_total"]),
                    str(scene["scene_id"]),
                ),
            )
            add_scene(best_scene)
            remaining = [scene for scene in remaining if scene["scene_id"] != best_scene["scene_id"]]

        while (
            remaining
            and len(selected) < target_scene_count
            and (
                selected_task_counts["a2"] < int(target_task_counts.get("a2", 0))
                or selected_task_counts["b2"] < int(target_task_counts.get("b2", 0))
            )
            and selected_questions_total < int(max_questions)
        ):
            nav_candidates = [
                scene
                for scene in remaining
                if int(scene["task_counts"].get("a2", 0)) + int(scene["task_counts"].get("b2", 0)) > 0
            ]
            if not nav_candidates:
                break
            deficit_a2 = max(0, int(target_task_counts.get("a2", 0)) - selected_task_counts["a2"])
            deficit_b2 = max(0, int(target_task_counts.get("b2", 0)) - selected_task_counts["b2"])
            deficit_c = max(0, int(target_task_counts.get("c", 0)) - selected_task_counts["c"])
            best_scene = max(
                nav_candidates,
                key=lambda scene: (
                    min(deficit_a2, int(scene["task_counts"].get("a2", 0))) * 20.0
                    + min(deficit_b2, int(scene["task_counts"].get("b2", 0))) * 20.0
                    + min(deficit_c, int(scene["task_counts"].get("c", 0))) * 10.0
                    - max(0, selected_task_counts["a1"] + int(scene["task_counts"].get("a1", 0)) - int(target_task_counts.get("a1", 0))) * 1.5
                    - max(0, selected_task_counts["b1"] + int(scene["task_counts"].get("b1", 0)) - int(target_task_counts.get("b1", 0))) * 1.5
                    - common_penalty(
                        scene,
                        question_weight=1.5,
                        hard_question_weight=50.0,
                        hard_bundle_weight=30.0,
                        source_weight=1.5,
                    ),
                    -int(scene["questions_total"]),
                    str(scene["scene_id"]),
                ),
            )
            add_scene(best_scene)
            remaining = [scene for scene in remaining if scene["scene_id"] != best_scene["scene_id"]]

        while (
            remaining
            and len(selected) < target_scene_count
            and (
                selected_task_counts["a1"] < int(target_task_counts.get("a1", 0))
                or selected_task_counts["b1"] < int(target_task_counts.get("b1", 0))
                or selected_questions_total < int(min_questions)
            )
            and selected_questions_total < int(max_questions)
        ):
            deficit_a1 = max(0, int(target_task_counts.get("a1", 0)) - selected_task_counts["a1"])
            deficit_b1 = max(0, int(target_task_counts.get("b1", 0)) - selected_task_counts["b1"])
            best_scene = max(
                remaining,
                key=lambda scene: (
                    min(deficit_a1, int(scene["task_counts"].get("a1", 0))) * 12.0
                    + min(deficit_b1, int(scene["task_counts"].get("b1", 0))) * 12.0
                    - max(0, selected_task_counts["a2"] + int(scene["task_counts"].get("a2", 0)) - int(target_task_counts.get("a2", 0))) * 0.5
                    - max(0, selected_task_counts["b2"] + int(scene["task_counts"].get("b2", 0)) - int(target_task_counts.get("b2", 0))) * 0.5
                    - max(0, selected_task_counts["c"] + int(scene["task_counts"].get("c", 0)) - int(target_task_counts.get("c", 0))) * 0.5
                    - common_penalty(
                        scene,
                        question_weight=1.2,
                        hard_question_weight=40.0,
                        hard_bundle_weight=25.0,
                        source_weight=1.0,
                    ),
                    -int(scene["questions_total"]),
                    str(scene["scene_id"]),
                ),
            )
            add_scene(best_scene)
            remaining = [scene for scene in remaining if scene["scene_id"] != best_scene["scene_id"]]

        selected_ids = {str(scene["scene_id"]) for scene in selected}
        summary = {
            "schema_version": "val_scene_selection_v3",
            "selection_order": ["benchmark", "val", "train"],
            "selection_policy": "balanced_navigation_dev_split",
            "advisory_target_scene_count": int(target_scene_count),
            "target_questions": int(target_questions),
            "target_window": {
                "min_questions": int(min_questions),
                "max_questions": int(max_questions),
            },
            "target_task_counts": {task: int(target_task_counts.get(task, 0)) for task in TASKS},
            "candidate_scene_count": len(candidates),
            "selected_scene_count": len(selected),
            "selected_scene_ids": sorted(selected_ids),
            "selected_source_group_ids": sorted({str(scene["source_group_id"]) for scene in selected}),
            "selected_questions_total": int(selected_questions_total),
            "selected_task_counts": {task: int(selected_task_counts.get(task, 0)) for task in TASKS},
            "selected_scene_rows": selected,
        }
        return selected_ids, summary

    selected: list[dict] = []
    source_counts: Counter[str] = Counter()

    while candidates and len(selected) < target_scene_count:
        best_scene = min(
            candidates,
            key=lambda scene: (
                int(scene["hard_candidate_question_count"]),
                int(scene["hard_candidate_bundle_count"]),
                source_counts[str(scene["source_dataset"])],
                int(scene["questions_total"]),
                str(scene["scene_id"]),
            ),
        )
        selected.append(best_scene)
        source_counts[str(best_scene["source_dataset"])] += 1
        candidates = [scene for scene in candidates if scene["scene_id"] != best_scene["scene_id"]]

    selected_ids = {str(scene["scene_id"]) for scene in selected}
    summary = {
        "schema_version": "val_scene_selection_v2",
        "selection_order": ["benchmark", "val", "train"],
        "selection_policy": "post_rebuild_low_hardness_preference",
        "advisory_target_scene_count": int(target_scene_count),
        "candidate_scene_count": len(
            [
                scene
                for scene in scene_rows
                if str(scene["scene_id"]) not in excluded_scene_ids
                and str(scene["source_group_id"]) not in excluded_source_group_ids
            ]
        ),
        "selected_scene_ids": sorted(selected_ids),
        "selected_source_group_ids": sorted({str(scene["source_group_id"]) for scene in selected}),
        "selected_scene_rows": selected,
    }
    return selected_ids, summary


def _current_total_objective(*, total: int, target_questions: int, min_questions: int, max_questions: int) -> float:
    score = -abs(target_questions - total) * 50.0
    if min_questions <= total <= max_questions:
        score += 10000.0
    if total > max_questions:
        score -= (total - max_questions) * 200.0
    return score


def _evaluate_scene_addition(
    *,
    current_total: int,
    current_navigation_total: int,
    current_affordance_total: int,
    scene: dict,
    selected_scenes: list[dict],
    target_questions: int,
    min_questions: int,
    max_questions: int,
    target_navigation_questions: int,
) -> float:
    new_total = current_total + int(scene["candidate_question_count"])
    new_navigation_total = current_navigation_total + int(scene["family_counts"].get("navigation", 0))
    new_affordance_total = current_affordance_total + int(scene["family_counts"].get("affordance", 0))
    score = _current_total_objective(
        total=new_total,
        target_questions=target_questions,
        min_questions=min_questions,
        max_questions=max_questions,
    )

    target_affordance_questions = max(target_questions - target_navigation_questions, 0)
    score += (abs(target_navigation_questions - current_navigation_total) - abs(target_navigation_questions - new_navigation_total)) * 90.0
    score += (abs(target_affordance_questions - current_affordance_total) - abs(target_affordance_questions - new_affordance_total)) * 20.0

    score += float(scene["mean_hardness_score"]) * 10.0
    score += float(scene["max_hardness_score"]) * 2.0
    score += float(scene["navigation_mean_score"]) * 6.0
    score += float(scene["bundle_diversity"]) * 30.0

    existing_source_counts = Counter(str(item["source_dataset"]) for item in selected_scenes)
    existing_source_counts[str(scene["source_dataset"])] += 1
    source_peak = max(existing_source_counts.values()) if existing_source_counts else 1
    score -= max(0, source_peak - 1) * 8.0

    return score


def _select_benchmark_scenes(
    scene_rows: list[dict],
    *,
    target_questions: int,
    min_questions: int,
    max_questions: int,
    target_navigation_questions: int,
) -> tuple[set[str], dict]:
    selected: list[dict] = []
    remaining = list(scene_rows)
    total_questions = 0
    total_navigation_questions = 0
    total_affordance_questions = 0

    while remaining:
        best_scene = max(
            remaining,
            key=lambda scene: (
                _evaluate_scene_addition(
                    current_total=total_questions,
                    current_navigation_total=total_navigation_questions,
                    current_affordance_total=total_affordance_questions,
                    scene=scene,
                    selected_scenes=selected,
                    target_questions=target_questions,
                    min_questions=min_questions,
                    max_questions=max_questions,
                    target_navigation_questions=target_navigation_questions,
                ),
                float(scene["navigation_mean_score"]),
                float(scene["mean_hardness_score"]),
                float(scene["max_hardness_score"]),
                -int(scene["candidate_question_count"]),
                str(scene["scene_id"]),
            ),
        )

        best_objective = _evaluate_scene_addition(
            current_total=total_questions,
            current_navigation_total=total_navigation_questions,
            current_affordance_total=total_affordance_questions,
            scene=best_scene,
            selected_scenes=selected,
            target_questions=target_questions,
            min_questions=min_questions,
            max_questions=max_questions,
            target_navigation_questions=target_navigation_questions,
        )
        current_objective = _current_total_objective(
            total=total_questions,
            target_questions=target_questions,
            min_questions=min_questions,
            max_questions=max_questions,
        )

        if total_questions >= min_questions and total_navigation_questions >= target_navigation_questions and best_objective <= current_objective:
            break

        selected.append(best_scene)
        remaining = [scene for scene in remaining if scene["scene_id"] != best_scene["scene_id"]]
        total_questions += int(best_scene["candidate_question_count"])
        total_navigation_questions += int(best_scene["family_counts"].get("navigation", 0))
        total_affordance_questions += int(best_scene["family_counts"].get("affordance", 0))

    selected_ids = {str(scene["scene_id"]) for scene in selected}
    selection_summary = {
        "schema_version": "benchmark_scene_selection_v3",
        "selection_order": ["benchmark", "val", "train"],
        "selection_policy": "target_proximity_after_full_hard_candidate_mining",
        "target_questions": target_questions,
        "target_navigation_questions": target_navigation_questions,
        "target_window": {
            "min_questions": min_questions,
            "max_questions": max_questions,
        },
        "candidate_scene_count": len(scene_rows),
        "selected_scene_count": len(selected),
        "selected_scene_ids": sorted(selected_ids),
        "selected_source_group_ids": sorted({str(scene["source_group_id"]) for scene in selected}),
        "selected_questions_total": sum(int(scene["candidate_question_count"]) for scene in selected),
        "selected_navigation_questions_total": sum(int(scene["family_counts"].get("navigation", 0)) for scene in selected),
        "selected_affordance_questions_total": sum(int(scene["family_counts"].get("affordance", 0)) for scene in selected),
        "selected_source_counts": dict(sorted(Counter(str(scene["source_dataset"]) for scene in selected).items())),
        "scene_rows": scene_rows,
        "selected_scene_rows": selected,
    }
    return selected_ids, selection_summary


def _validate_release(
    *,
    candidate_manifest: dict,
    split_vqa: dict[str, dict[str, list[dict]]],
    target_questions: int,
    min_questions: int,
    max_questions: int,
    min_candidate_questions: int,
    max_candidate_questions: int,
    min_navigation_questions: int,
) -> None:
    candidate_total = int(candidate_manifest["policy"]["question_counts"]["hard_candidate"])
    benchmark_navigation_total = sum(len(split_vqa["benchmark"][task]) for task in ("a2", "b2", "c"))
    benchmark_total = sum(len(rows) for rows in split_vqa["benchmark"].values())

    if not (min_candidate_questions <= candidate_total <= max_candidate_questions):
        raise ValueError(
            f"hard candidate pool out of range: total={candidate_total} expected=[{min_candidate_questions}, {max_candidate_questions}]"
        )
    if not (min_questions <= benchmark_total <= max_questions):
        raise ValueError(
            f"benchmark question total out of range: total={benchmark_total} expected=[{min_questions}, {max_questions}]"
        )
    if benchmark_navigation_total < min_navigation_questions:
        raise ValueError(
            f"benchmark navigation total below minimum: total={benchmark_navigation_total} required={min_navigation_questions}"
        )
    if benchmark_total < target_questions and benchmark_navigation_total < min_navigation_questions:
        raise ValueError("benchmark under target and lacks navigation coverage")


def build_release_datasets(
    *,
    gt_dir: Path,
    vqa_dir: Path,
    split_manifest: dict,
    output_dir: Path,
    target_questions: int = DEFAULT_TARGET_QUESTIONS,
    min_questions: int = DEFAULT_MIN_QUESTIONS,
    max_questions: int = DEFAULT_MAX_QUESTIONS,
    min_candidate_questions: int = DEFAULT_MIN_CANDIDATE_QUESTIONS,
    max_candidate_questions: int = DEFAULT_MAX_CANDIDATE_QUESTIONS,
    min_navigation_questions: int = DEFAULT_TARGET_NAVIGATION_QUESTIONS,
    adjudication_by_sample_id: dict[str, dict] | None = None,
    validate_release: bool = True,
    vqa_build_mode: str = "rebuild_from_gt",
    formal_release: bool = True,
) -> dict:
    if min_questions > max_questions:
        raise ValueError("min_questions must be <= max_questions")
    if vqa_build_mode not in ALLOWED_VQA_BUILD_MODES:
        raise ValueError(
            f"unsupported vqa_build_mode={vqa_build_mode}; expected one of {ALLOWED_VQA_BUILD_MODES}"
        )
    if formal_release and vqa_build_mode == "source_passthrough_preview":
        raise ValueError(
            "formal release does not allow vqa_build_mode=source_passthrough_preview; "
            "use rebuild_from_gt or explicitly set formal_release=False for legacy preview runs"
        )

    gt_rows_by_task = _collect_rows_by_task(gt_dir, "gt")
    vqa_rows_by_task = _collect_rows_by_task(vqa_dir, "vqa")
    gt_by_question_id = _index_by_question_id(gt_rows_by_task)
    vqa_by_question_id = _index_by_question_id(vqa_rows_by_task)

    if set(gt_by_question_id) != set(vqa_by_question_id):
        missing_gt = sorted(set(vqa_by_question_id) - set(gt_by_question_id))
        missing_vqa = sorted(set(gt_by_question_id) - set(vqa_by_question_id))
        raise ValueError(
            f"GT/VQA question mismatch: missing_gt={missing_gt[:5]} missing_vqa={missing_vqa[:5]}"
        )
    _validate_gt_vqa_alignment(gt_by_question_id, vqa_by_question_id)

    selection_rows_by_task = gt_rows_by_task
    all_selection_rows = [row for task in TASKS for row in selection_rows_by_task[task]]
    selection_dir = output_dir / "benchmark_selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    candidate_manifest, label_rows, labels_path, manifest_path = _build_hard_candidate_artifacts(
        all_selection_rows=all_selection_rows,
        selection_source_dir=gt_dir,
        selection_source_prefix="gt",
        selection_dir=selection_dir,
    )

    candidate_question_ids = {
        str(label["question_id"])
        for label in label_rows
        if bool(label.get("hard_candidate"))
    }
    all_scene_rows = _summarize_all_scenes(
        selection_rows_by_task,
        hard_candidate_question_ids=candidate_question_ids,
    )
    scene_rows = _summarize_candidate_scenes(label_rows)
    available_navigation_questions = sum(int(scene["family_counts"].get("navigation", 0)) for scene in scene_rows)
    target_navigation_questions = min(max(min_navigation_questions, target_questions // 5), available_navigation_questions)
    selected_benchmark_scenes, selection_manifest = _select_benchmark_scenes(
        scene_rows,
        target_questions=target_questions,
        min_questions=min_questions,
        max_questions=max_questions,
        target_navigation_questions=target_navigation_questions,
    )
    selection_manifest_path = selection_dir / "benchmark_scene_selection.json"
    with open(selection_manifest_path, "w") as f:
        json.dump(selection_manifest, f, indent=2)

    split_gt: dict[str, dict[str, list[dict]]] = {split_name: {task: [] for task in TASKS} for split_name in SPLITS}
    split_vqa_source: dict[str, dict[str, list[dict]]] = {split_name: {task: [] for task in TASKS} for split_name in SPLITS}
    selected_question_ids: set[str] = set()
    hard_nonbenchmark_question_ids: set[str] = set()
    hard_retained_by_split: dict[str, set[str]] = {"train": set(), "val": set()}
    selected_benchmark_source_groups = {
        _source_group(scene_id) for scene_id in selected_benchmark_scenes
    }
    val_target_task_counts = _derive_val_task_targets(DEFAULT_VAL_TARGET_QUESTIONS)
    use_balanced_val_selection = _can_use_balanced_val_selection(
        all_scene_rows,
        excluded_scene_ids=set(selected_benchmark_scenes),
        excluded_source_group_ids=set(selected_benchmark_source_groups),
        min_questions=DEFAULT_VAL_MIN_QUESTIONS,
        target_task_counts=val_target_task_counts,
    )
    selected_val_scenes, val_selection_manifest = _select_val_scenes(
        all_scene_rows,
        excluded_scene_ids=set(selected_benchmark_scenes),
        excluded_source_group_ids=set(selected_benchmark_source_groups),
        target_scene_count=_derive_val_scene_target_count(
            split_manifest,
            scene_rows=all_scene_rows,
            excluded_scene_ids=set(selected_benchmark_scenes),
            excluded_source_group_ids=set(selected_benchmark_source_groups),
        ),
        target_questions=DEFAULT_VAL_TARGET_QUESTIONS if use_balanced_val_selection else None,
        min_questions=DEFAULT_VAL_MIN_QUESTIONS if use_balanced_val_selection else None,
        max_questions=DEFAULT_VAL_MAX_QUESTIONS if use_balanced_val_selection else None,
        target_task_counts=val_target_task_counts if use_balanced_val_selection else None,
    )

    for task in TASKS:
        for gt_row in gt_rows_by_task[task]:
            question_id = str(gt_row["question_id"])
            scene_id = str(gt_row["scene_id"])
            source_group_id = _source_group(scene_id)
            if scene_id in selected_benchmark_scenes:
                if question_id not in candidate_question_ids:
                    continue
                split_name = "benchmark"
                selected_question_ids.add(question_id)
            else:
                if source_group_id in selected_benchmark_source_groups:
                    continue
                split_name = "val" if scene_id in selected_val_scenes else "train"
                if question_id in candidate_question_ids:
                    hard_nonbenchmark_question_ids.add(question_id)
                    if split_name in hard_retained_by_split:
                        hard_retained_by_split[split_name].add(question_id)

            split_vqa_source[split_name][task].append(vqa_by_question_id[question_id])
            split_gt[split_name][task].append(gt_row)

    if vqa_build_mode == "rebuild_from_gt":
        split_vqa = {
            split_name: _rebuild_vqa_rows_by_task_from_gt(
                split_gt[split_name],
                split_name=split_name,
                adjudication_by_sample_id=adjudication_by_sample_id,
                formal_release=formal_release,
            )
            for split_name in SPLITS
        }
    else:
        split_vqa = split_vqa_source

    validation_error: str | None = None
    try:
        _validate_release(
            candidate_manifest=candidate_manifest,
            split_vqa=split_vqa,
            target_questions=target_questions,
            min_questions=min_questions,
            max_questions=max_questions,
            min_candidate_questions=min_candidate_questions,
            max_candidate_questions=max_candidate_questions,
            min_navigation_questions=min_navigation_questions,
        )
    except ValueError as exc:
        validation_error = str(exc)
        if validate_release:
            raise

    for split_name in SPLITS:
        _write_rows_by_task(output_dir / split_name / "gt", "gt", split_gt[split_name])
        _write_rows_by_task(output_dir / split_name / "vqa", "vqa", split_vqa[split_name])

    val_selection_dir = output_dir / "val_selection"
    val_selection_dir.mkdir(parents=True, exist_ok=True)
    val_selection_manifest_path = val_selection_dir / "val_scene_selection.json"
    with open(val_selection_manifest_path, "w") as f:
        json.dump(val_selection_manifest, f, indent=2)

    manifest = {
        "schema_version": "next_release_dataset_manifest_v4",
        "vqa_build_mode": vqa_build_mode,
        "formal_release": bool(formal_release),
        "preview_only": vqa_build_mode != "rebuild_from_gt",
        "policy_name": candidate_manifest["policy"]["name"],
        "policy_description": candidate_manifest["policy"]["description"],
        "source_split_manifest": split_manifest,
        "splits": {
            split_name: _summarize_split(split_vqa[split_name])
            for split_name in SPLITS
        },
        "benchmark_selection": {
            "target_questions": target_questions,
            "min_questions": min_questions,
            "max_questions": max_questions,
            "min_candidate_questions": min_candidate_questions,
            "max_candidate_questions": max_candidate_questions,
            "min_navigation_questions": min_navigation_questions,
            "target_navigation_questions": target_navigation_questions,
            "hard_candidate_questions_total": candidate_manifest["policy"]["question_counts"]["hard_candidate"],
            "hard_candidate_question_counts_by_task": candidate_manifest["policy"]["question_counts_by_task"],
            "hard_candidate_question_counts_by_source": candidate_manifest["policy"]["question_counts_by_source"],
            "selected_questions_total": len(selected_question_ids),
            "hard_non_benchmark_questions_total": len(hard_nonbenchmark_question_ids),
            "hard_train_questions_total": len(hard_retained_by_split["train"]),
            "hard_val_questions_total": len(hard_retained_by_split["val"]),
            "selected_scene_count": len(selected_benchmark_scenes),
            "selected_scene_ids": sorted(selected_benchmark_scenes),
            "selected_source_group_ids": sorted(selected_benchmark_source_groups),
            "reporting": {
                "question_variants": int(candidate_manifest["policy"]["reporting"]["question_variants"]["hard_candidate"]),
                "unique_bundles": int(candidate_manifest["policy"]["reporting"]["unique_bundles"]["hard_candidate"]),
                "unique_scenes": int(candidate_manifest["policy"]["reporting"]["unique_scenes"]["hard_candidate"]),
                "unique_source_groups": int(candidate_manifest["policy"]["reporting"]["unique_source_groups"]["hard_candidate"]),
            },
            "hard_candidate_labels_jsonl": str(labels_path),
            "hard_candidate_manifest_json": str(manifest_path),
            "benchmark_scene_selection_json": str(selection_manifest_path),
        },
        "val_selection": {
            "selected_scene_count": len(val_selection_manifest["selected_scene_ids"]),
            "selected_scene_ids": val_selection_manifest["selected_scene_ids"],
            "selected_source_group_ids": val_selection_manifest["selected_source_group_ids"],
            "val_scene_selection_json": str(val_selection_manifest_path),
        },
    }
    if vqa_build_mode != "rebuild_from_gt":
        manifest["preview_note"] = (
            "Split counts and leakage accounting are authoritative, but VQA rows were copied "
            "from the source package instead of being regenerated from truth-aligned GT."
        )
    if validation_error is not None:
        manifest["validation_error"] = validation_error
    with open(output_dir / "release_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def build_truth_aligned_release_package(
    *,
    legacy_release_dir: Path,
    repaired_release_dir: Path,
    output_root: Path,
    release_version: str,
    supersedes: str | None = None,
) -> dict:
    return build_truth_aligned_release_package_layout(
        legacy_release_dir=legacy_release_dir,
        repaired_release_dir=repaired_release_dir,
        output_root=output_root,
        release_version=release_version,
        supersedes=supersedes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/val/benchmark release subsets from full Next outputs.")
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--vqa-dir", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-benchmark-questions", type=int, default=DEFAULT_TARGET_QUESTIONS)
    parser.add_argument("--min-benchmark-questions", type=int, default=DEFAULT_MIN_QUESTIONS)
    parser.add_argument("--max-benchmark-questions", type=int, default=DEFAULT_MAX_QUESTIONS)
    parser.add_argument("--min-candidate-questions", type=int, default=DEFAULT_MIN_CANDIDATE_QUESTIONS)
    parser.add_argument("--max-candidate-questions", type=int, default=DEFAULT_MAX_CANDIDATE_QUESTIONS)
    parser.add_argument("--min-navigation-questions", type=int, default=DEFAULT_TARGET_NAVIGATION_QUESTIONS)
    parser.add_argument("--uncertain-adjudication-jsonl", default=None)
    parser.add_argument(
        "--vqa-build-mode",
        default="rebuild_from_gt",
        choices=sorted(ALLOWED_VQA_BUILD_MODES),
        help="Use source_passthrough_preview only together with --non-formal-release for legacy preview inputs.",
    )
    parser.add_argument(
        "--non-formal-release",
        action="store_true",
        help="Allow legacy preview passthrough; formal release defaults to rebuild_from_gt only.",
    )
    args = parser.parse_args()

    with open(Path(args.split_manifest).resolve(), "r") as f:
        split_manifest = json.load(f)
    adjudication_by_sample_id = load_uncertain_adjudication_sidecar(
        Path(args.uncertain_adjudication_jsonl).resolve()
        if args.uncertain_adjudication_jsonl
        else None
    )

    manifest = build_release_datasets(
        gt_dir=Path(args.gt_dir).resolve(),
        vqa_dir=Path(args.vqa_dir).resolve(),
        split_manifest=split_manifest,
        output_dir=Path(args.output_dir).resolve(),
        target_questions=int(args.target_benchmark_questions),
        min_questions=int(args.min_benchmark_questions),
        max_questions=int(args.max_benchmark_questions),
        min_candidate_questions=int(args.min_candidate_questions),
        max_candidate_questions=int(args.max_candidate_questions),
        min_navigation_questions=int(args.min_navigation_questions),
        adjudication_by_sample_id=adjudication_by_sample_id,
        vqa_build_mode=str(args.vqa_build_mode),
        formal_release=not bool(args.non_formal_release),
    )
    print(json.dumps({split: payload["question_counts_by_task"] for split, payload in manifest["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
