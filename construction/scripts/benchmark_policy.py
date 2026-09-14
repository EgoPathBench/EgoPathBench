#!/usr/bin/env python3
"""
Shared global hard-candidate policy for NavBench3D-Next benchmark construction.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean


FINAL_LABELS = ("hard_candidate", "non_candidate")
POLICY_NAME = "global_hard_candidate_v2"
POLICY_DESCRIPTION = (
    "Global hard-candidate miner for the strict Next benchmark. "
    "A1/B1 require complete eligible hard affordance bundles with high clutter and at least one of "
    "boundary or embodiment-gap difficulty high, then use score-based calibration to keep only the hardest view bundles. "
    "A2/B2 define the required complete eligible hard navigation bundle while C is an optional implicit companion question; "
    "navigation admission still requires non-category-only references, reference_axis above low, geometry_axis above low, "
    "and a geometry-dominant hardness score so high-geometry paths outrank medium-geometry backfill routes."
)
REQUIRED_TASKS_BY_BUNDLE = {
    "view_bundle": ("a1", "b1"),
    "route_bundle": ("a2", "b2"),
}
OPTIONAL_TASKS_BY_BUNDLE = {
    "view_bundle": (),
    "route_bundle": ("c",),
}
AXIS_SCORE = {"low": 0, "medium": 1, "high": 2}
REFERENCE_MODE_SCORE = {
    "category_only": 0,
    "distance_only": 1,
    "color_only": 2,
    "color_and_distance": 3,
    "relation_only": 3,
}
CANDIDATE_POOL_TARGET_QUESTIONS = 3500
CANDIDATE_POOL_TARGET_BY_BUNDLE_TYPE = {
    "view_bundle": {
        "question_target": 2450,
        "bundle_cap": 1225,
    },
    "route_bundle": {
        "question_target": 1050,
        "bundle_cap": 350,
    },
}
POLICY_THRESHOLDS = {
    "affordance": {
        "eligible_required": True,
        "tier_required": "hard",
        "clutter_axis_required": "high",
        "boundary_or_gap_required": "high",
    },
    "navigation": {
        "eligible_required": True,
        "tier_required": "hard",
        "reference_resolution_forbidden": ["category_only"],
        "reference_axis_minimum": "medium",
        "geometry_axis_minimum": "medium",
        "geometry_axis_preferred": "high",
        "geometry_priority": "dominant_weight",
    },
}
POLICY_CALIBRATION = {
    "candidate_pool_target_questions": CANDIDATE_POOL_TARGET_QUESTIONS,
    "bundle_caps": CANDIDATE_POOL_TARGET_BY_BUNDLE_TYPE,
}


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _require_text(value: object, *, context: str) -> str:
    if value is None:
        raise ValueError(f"{context} missing required text value")
    text = str(value).strip()
    if not text:
        raise ValueError(f"{context} missing required text value")
    return text


def _nested_payload(row: dict) -> dict:
    payload = row.get("ground_truth")
    if isinstance(payload, dict):
        return payload
    return {}


def extract_field(row: dict, key: str) -> object:
    if key in row:
        return row.get(key)
    return _nested_payload(row).get(key)


def _require_text_field(row: dict, key: str, *, context: str) -> str:
    return _require_text(extract_field(row, key), context=f"{context}.{key}")


def _require_dict_field(row: dict, key: str, *, context: str) -> dict:
    value = extract_field(row, key)
    if not isinstance(value, dict):
        raise ValueError(f"{context} missing required dict field: {key}")
    return value


def _require_bool(value: object, *, context: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    raise ValueError(f"{context} must be a bool")


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


def _human_visible_legality(row: dict) -> str:
    legality = extract_field(row, "human_visible_legality")
    if legality is None:
        return "legal"
    normalized = _require_text(legality, context="question.human_visible_legality").lower()
    if normalized not in {"legal", "illegal", "uncertain"}:
        raise ValueError(f"unsupported human_visible_legality: {normalized}")
    return normalized


def _score_axis(label: str, *, context: str) -> int:
    normalized = _require_text(label, context=context).lower()
    if normalized not in AXIS_SCORE:
        raise ValueError(f"{context} unsupported axis label: {normalized}")
    return AXIS_SCORE[normalized]


def _score_reference_mode(label: str, *, context: str) -> int:
    normalized = _require_text(label, context=context).lower()
    if normalized not in REFERENCE_MODE_SCORE:
        raise ValueError(f"{context} unsupported reference mode: {normalized}")
    return REFERENCE_MODE_SCORE[normalized]


def question_family(row: dict) -> str:
    family = _require_text_field(row, "tier_family", context="question").lower()
    if family not in {"affordance", "navigation"}:
        raise ValueError(f"unsupported tier_family: {family}")
    return family


def question_task(row: dict) -> str:
    return _require_text_field(row, "task", context="question").lower()


def question_bundle_type(row: dict) -> str:
    family = question_family(row)
    return "view_bundle" if family == "affordance" else "route_bundle"


def question_bundle_key(row: dict) -> str:
    scene_id = _require_text_field(row, "scene_id", context="question")
    view_id = extract_field(row, "view_id")
    if view_id is None:
        raise ValueError("question.view_id missing required text value")
    if question_bundle_type(row) == "view_bundle":
        return f"{scene_id}::v{int(view_id)}"
    routing_id = _require_text_field(row, "routing_id", context="question")
    return f"{scene_id}::v{int(view_id)}::{routing_id}"


def _classify_affordance_question(row: dict) -> dict:
    legality = _human_visible_legality(row)
    tier = _require_text_field(row, "tier", context="affordance").lower()
    axes = _require_dict_field(row, "affordance_axes", context="affordance")
    validity = _require_dict_field(row, "affordance_validity", context="affordance")

    clutter_axis = _require_text(axes.get("clutter_axis"), context="affordance_axes.clutter_axis").lower()
    boundary_axis = _require_text(axes.get("boundary_axis"), context="affordance_axes.boundary_axis").lower()
    embodiment_gap_axis = _require_text(
        axes.get("embodiment_gap_axis"),
        context="affordance_axes.embodiment_gap_axis",
    ).lower()
    eligible = _require_bool(validity.get("eligible"), context="affordance_validity.eligible")

    reasons: list[str] = []
    if legality != "legal":
        reasons.append(f"human_visible_{legality}")
    if not eligible:
        reasons.append("affordance_ineligible")
    if tier != "hard":
        reasons.append("affordance_tier_not_hard")
    if clutter_axis != "high":
        reasons.append("clutter_axis_not_high")
    if boundary_axis != "high" and embodiment_gap_axis != "high":
        reasons.append("boundary_or_gap_not_high")

    candidate_eligible = not reasons
    hardness_score = 0
    if candidate_eligible:
        boundary_score = _score_axis(boundary_axis, context="affordance_axes.boundary_axis")
        gap_score = _score_axis(embodiment_gap_axis, context="affordance_axes.embodiment_gap_axis")
        hardness_score = 25 + (boundary_score * 5) + (gap_score * 7)
        if boundary_axis == "high" and embodiment_gap_axis == "high":
            hardness_score += 6

    return {
        "bundle_type": "view_bundle",
        "final_label": "hard_candidate" if candidate_eligible else "non_candidate",
        "hard_candidate": candidate_eligible,
        "candidate_eligible": candidate_eligible,
        "bundle_accept": candidate_eligible,
        "hardness_score": hardness_score,
        "rejection_reasons": reasons,
        "benchmark_evidence": {
            "affordance_tier": tier,
            "affordance_axes": axes,
            "affordance_validity": validity,
        },
    }


def _classify_navigation_question(row: dict) -> dict:
    legality = _human_visible_legality(row)
    tier = _require_text_field(row, "tier", context="navigation").lower()
    axes = _require_dict_field(row, "navigation_axes", context="navigation")
    reference_facts = _require_dict_field(row, "navigation_reference_facts", context="navigation")
    validity = _require_dict_field(row, "navigation_validity", context="navigation")

    reference_axis = _require_text(axes.get("reference_axis"), context="navigation_axes.reference_axis").lower()
    geometry_axis = _require_text(axes.get("geometry_axis"), context="navigation_axes.geometry_axis").lower()
    embodiment_axis = _require_text(axes.get("embodiment_axis", "medium"), context="navigation_axes.embodiment_axis").lower()
    resolution_mode = _require_text(
        reference_facts.get("reference_resolution_mode"),
        context="navigation_reference_facts.reference_resolution_mode",
    ).lower()
    eligible = _require_bool(validity.get("eligible"), context="navigation_validity.eligible")

    reasons: list[str] = []
    if legality != "legal":
        reasons.append(f"human_visible_{legality}")
    if not eligible:
        reasons.append("navigation_ineligible")
    if tier != "hard":
        reasons.append("navigation_tier_not_hard")
    if resolution_mode == "category_only":
        reasons.append("reference_resolution_category_only")
    if reference_axis == "low":
        reasons.append("reference_axis_low")
    if geometry_axis == "low":
        reasons.append("geometry_axis_low")

    candidate_eligible = not reasons
    hardness_score = 0
    if candidate_eligible:
        reference_score = _score_axis(reference_axis, context="navigation_axes.reference_axis")
        geometry_score = _score_axis(geometry_axis, context="navigation_axes.geometry_axis")
        embodiment_score = _score_axis(embodiment_axis, context="navigation_axes.embodiment_axis")
        reference_mode_score = _score_reference_mode(
            resolution_mode,
            context="navigation_reference_facts.reference_resolution_mode",
        )
        # Geometry dominates admission ranking. Medium-geometry routes can backfill only when they outrank peers by score.
        hardness_score = (
            geometry_score * 60
            + reference_score * 8
            + reference_mode_score * 5
            + embodiment_score * 2
        )
        if geometry_axis == "high":
            hardness_score += 20

    return {
        "bundle_type": "route_bundle",
        "final_label": "hard_candidate" if candidate_eligible else "non_candidate",
        "hard_candidate": candidate_eligible,
        "candidate_eligible": candidate_eligible,
        "bundle_accept": candidate_eligible,
        "hardness_score": hardness_score,
        "rejection_reasons": reasons,
        "benchmark_evidence": {
            "navigation_tier": tier,
            "navigation_axes": axes,
            "navigation_reference_facts": reference_facts,
            "navigation_validity": validity,
        },
    }


def classify_question(row: dict) -> dict:
    bundle_type = question_bundle_type(row)
    bundle_key = question_bundle_key(row)
    question_id = _require_text_field(row, "question_id", context="question")
    task = question_task(row)
    scene_id = _require_text_field(row, "scene_id", context="question")
    if bundle_type == "view_bundle":
        classification = _classify_affordance_question(row)
    else:
        classification = _classify_navigation_question(row)
    return {
        "question_id": question_id,
        "task": task,
        "tier_family": question_family(row),
        "scene_id": scene_id,
        "source_dataset": _source_dataset(scene_id),
        "source_group_id": _source_group(scene_id),
        "view_id": int(extract_field(row, "view_id")),
        "routing_id": extract_field(row, "routing_id"),
        "bundle_key": bundle_key,
        **classification,
    }


def _base_decision_signature(decision: dict) -> str:
    payload = {
        "bundle_type": decision["bundle_type"],
        "candidate_eligible": decision["candidate_eligible"],
        "bundle_accept": decision["bundle_accept"],
        "hardness_score": decision["hardness_score"],
        "rejection_reasons": list(decision["rejection_reasons"]),
        "benchmark_evidence": decision["benchmark_evidence"],
    }
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def _sorted_counts(counter: Counter) -> dict[str, int]:
    return {label: int(counter.get(label, 0)) for label in FINAL_LABELS}


def _entity_count(records: list[dict], key_fn) -> dict[str, int]:
    grouped = {label: set() for label in FINAL_LABELS}
    for record in records:
        grouped[str(record["final_label"])].add(key_fn(record))
    return {label: len(grouped[label]) for label in FINAL_LABELS}


def _score_summary(scores: list[int]) -> dict[str, float | int | None]:
    if not scores:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
        }
    return {
        "count": len(scores),
        "min": min(scores),
        "max": max(scores),
        "mean": round(mean(scores), 3),
    }


def _select_candidate_bundle_keys(bundle_records: list[dict]) -> set[tuple[str, str]]:
    selected: set[tuple[str, str]] = set()
    for bundle_type, calibration in CANDIDATE_POOL_TARGET_BY_BUNDLE_TYPE.items():
        eligible_records = [
            record
            for record in bundle_records
            if record["bundle_type"] == bundle_type and record["candidate_eligible"]
        ]
        eligible_records.sort(
            key=lambda record: (
                -int(record["hardness_score"]),
                str(record["scene_id"]),
                str(record["bundle_key"]),
            )
        )
        for record in eligible_records[: int(calibration["bundle_cap"])]:
            selected.add((str(record["bundle_type"]), str(record["bundle_key"])))
    return selected


def _validate_bundle_tasks(bundle_type: str, actual_tasks: set[str], bundle_key: str) -> None:
    required_tasks = set(REQUIRED_TASKS_BY_BUNDLE[bundle_type])
    optional_tasks = set(OPTIONAL_TASKS_BY_BUNDLE[bundle_type])
    allowed_tasks = required_tasks | optional_tasks
    missing_required = sorted(required_tasks - actual_tasks)
    unexpected_tasks = sorted(actual_tasks - allowed_tasks)
    if missing_required or unexpected_tasks:
        raise ValueError(
            f"incomplete {bundle_type} {bundle_key}: required_tasks={sorted(required_tasks)} "
            f"optional_tasks={sorted(optional_tasks)} got={sorted(actual_tasks)}"
        )


def _reference_bundle_decision(bundle_type: str, decisions: list[dict], bundle_key: str) -> dict:
    required_tasks = set(REQUIRED_TASKS_BY_BUNDLE[bundle_type])
    required_decisions = [decision for decision in decisions if str(decision["task"]) in required_tasks]
    if not required_decisions:
        raise ValueError(f"missing required decisions for {bundle_type} {bundle_key}")
    signature = _base_decision_signature(required_decisions[0])
    for decision in required_decisions[1:]:
        if _base_decision_signature(decision) != signature:
            raise ValueError(f"inconsistent required-task bundle decision for {bundle_type} {bundle_key}")
    return required_decisions[0]


def build_manifest(rows: list[dict], *, source_files: list[str], labels_path: Path) -> tuple[dict, list[dict]]:
    grouped_rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped_rows[(question_bundle_type(row), question_bundle_key(row))].append(row)

    bundle_records: list[dict] = []
    for (bundle_type, bundle_key), bundle_rows in sorted(grouped_rows.items()):
        actual_tasks = {question_task(row) for row in bundle_rows}
        _validate_bundle_tasks(bundle_type, actual_tasks, bundle_key)

        decisions = [classify_question(row) for row in sorted(bundle_rows, key=question_task)]
        reference_decision = _reference_bundle_decision(bundle_type, decisions, bundle_key)

        bundle_records.append(
            {
                "bundle_type": bundle_type,
                "bundle_key": bundle_key,
                "scene_id": str(reference_decision["scene_id"]),
                "source_dataset": str(reference_decision["source_dataset"]),
                "candidate_eligible": bool(reference_decision["candidate_eligible"]),
                "hardness_score": int(reference_decision["hardness_score"]),
                "decisions": decisions,
            }
        )

    selected_bundle_keys = _select_candidate_bundle_keys(bundle_records)

    label_rows: list[dict] = []
    question_counter = Counter()
    task_counters: dict[str, Counter] = defaultdict(Counter)
    bundle_counters: dict[str, Counter] = {bundle_type: Counter() for bundle_type in REQUIRED_TASKS_BY_BUNDLE}
    source_counters: dict[str, Counter] = defaultdict(Counter)
    accepted_scores: list[int] = []
    accepted_scores_by_bundle: dict[str, list[int]] = defaultdict(list)
    eligible_counts_by_bundle_type: Counter = Counter()

    for record in bundle_records:
        bundle_type = str(record["bundle_type"])
        bundle_key = str(record["bundle_key"])
        bundle_candidate_eligible = bool(record["candidate_eligible"])
        selected = (bundle_type, bundle_key) in selected_bundle_keys
        bundle_final_label = "hard_candidate" if bundle_candidate_eligible and selected else "non_candidate"
        if bundle_candidate_eligible:
            eligible_counts_by_bundle_type[bundle_type] += 1
        if bundle_final_label == "hard_candidate":
            accepted_scores.append(int(record["hardness_score"]))
            accepted_scores_by_bundle[bundle_type].append(int(record["hardness_score"]))
        bundle_counters[bundle_type][bundle_final_label] += 1

        for original_decision in record["decisions"]:
            decision = dict(original_decision)
            final_label = (
                "hard_candidate"
                if selected and bool(original_decision["candidate_eligible"])
                else "non_candidate"
            )
            decision["base_hard_candidate"] = bool(original_decision["candidate_eligible"])
            decision["hard_candidate"] = final_label == "hard_candidate"
            decision["bundle_accept"] = final_label == "hard_candidate"
            decision["final_label"] = final_label
            if bool(original_decision["candidate_eligible"]) and not selected:
                decision["rejection_reasons"] = list(original_decision["rejection_reasons"]) + [
                    f"candidate_pool_trimmed_{bundle_type}"
                ]
            question_counter[final_label] += 1
            task_counters[decision["task"]][final_label] += 1
            source_counters[decision["source_dataset"]][final_label] += 1
            label_rows.append(decision)

    reporting = {
        "question_variants": _sorted_counts(question_counter),
        "unique_bundles": _entity_count(label_rows, lambda row: (str(row["bundle_type"]), str(row["bundle_key"]))),
        "unique_scenes": _entity_count(label_rows, lambda row: str(row["scene_id"])),
        "unique_source_groups": _entity_count(label_rows, lambda row: str(row["source_group_id"])),
    }

    manifest = {
        "schema_version": "hard_candidate_manifest_v2",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "source_files": source_files,
        "policy": {
            "name": POLICY_NAME,
            "description": POLICY_DESCRIPTION,
            "thresholds": POLICY_THRESHOLDS,
            "calibration": {
                **POLICY_CALIBRATION,
                "eligible_bundle_counts_by_type": dict(sorted(eligible_counts_by_bundle_type.items())),
            },
            "question_counts": _sorted_counts(question_counter),
            "reporting": reporting,
            "question_counts_by_task": {
                task: _sorted_counts(counter)
                for task, counter in sorted(task_counters.items())
            },
            "question_counts_by_source": {
                source: _sorted_counts(counter)
                for source, counter in sorted(source_counters.items())
            },
            "bundle_counts": {
                bundle_type: _sorted_counts(counter)
                for bundle_type, counter in sorted(bundle_counters.items())
            },
            "score_summary": {
                "all_hard_candidates": _score_summary(accepted_scores),
                "by_bundle_type": {
                    bundle_type: _score_summary(scores)
                    for bundle_type, scores in sorted(accepted_scores_by_bundle.items())
                },
            },
        },
        "corpus": {
            "vqa_records_total": len(rows),
            "view_bundle_total": sum(1 for bundle_type, _ in grouped_rows if bundle_type == "view_bundle"),
            "route_bundle_total": sum(1 for bundle_type, _ in grouped_rows if bundle_type == "route_bundle"),
            "scene_total": len({_require_text_field(row, "scene_id", context="question") for row in rows}),
        },
        "outputs": {
            "labels_jsonl": str(labels_path),
        },
    }
    return manifest, label_rows
