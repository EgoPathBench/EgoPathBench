#!/usr/bin/env python3
"""
Aggregate machine-readable and human-readable stats for Next dataset artifacts.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from task_outputs_utils import load_scene_classification_views


TIER_FAMILIES = ("navigation", "affordance")
TIER_LEVELS = ("easy", "medium", "hard")
AXIS_LEVELS = ("low", "medium", "high")
NAVIGATION_AXIS_NAMES = ("reference_axis", "geometry_axis", "embodiment_axis")
AFFORDANCE_AXIS_NAMES = ("clutter_axis", "boundary_axis", "embodiment_gap_axis")
MIN_READY_NONTRIVIAL_REFERENCE_FRACTION = 0.25
MIN_READY_NONTRIVIAL_GEOMETRY_FRACTION = 0.35
MAX_READY_HARD_EMBODIMENT_ONLY_FRACTION = 0.34


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


def require_dict(payload: object, *, context: str) -> dict:
    if not isinstance(payload, dict):
        raise ValueError(f"{context} must be a dict")
    return payload


def require_tier_family(row: dict, *, context: str) -> str:
    value = row.get("tier_family")
    if value is None:
        raise ValueError(f"{context} missing required field: tier_family")
    family = str(value).strip().lower()
    if family not in TIER_FAMILIES:
        raise ValueError(f"{context} tier_family must be one of {TIER_FAMILIES}, got {value!r}")
    return family


def require_axes(row: dict, key: str, *, axis_names: tuple[str, ...], context: str) -> dict[str, str]:
    payload = require_dict(row.get(key), context=f"{context}.{key}")
    axes: dict[str, str] = {}
    for axis_name in axis_names:
        value = payload.get(axis_name)
        if value is None:
            raise ValueError(f"{context}.{key} missing required field: {axis_name}")
        axis = str(value).strip().lower()
        if axis not in AXIS_LEVELS:
            raise ValueError(f"{context}.{key}.{axis_name} must be one of {AXIS_LEVELS}, got {value!r}")
        axes[axis_name] = axis
    return axes


def require_tier_label(row: dict, *, context: str) -> str:
    value = row.get("tier")
    if value is None:
        raise ValueError(f"{context} missing required field: tier")
    tier = str(value).strip().lower()
    if tier not in TIER_LEVELS:
        raise ValueError(f"{context} tier must be one of {TIER_LEVELS}, got {value!r}")
    return tier


def normalize_counter(counter: Counter, *, preferred_order: tuple[str, ...] | None = None) -> dict[str, int]:
    if preferred_order is None:
        keys = sorted(counter.keys())
    else:
        ordered_keys = [key for key in preferred_order if counter[key] > 0]
        remaining_keys = sorted(key for key in counter.keys() if key not in preferred_order and counter[key] > 0)
        keys = ordered_keys + remaining_keys
    return {key: counter[key] for key in keys if counter[key] > 0}


def summarize_axis_counters(axis_counters: dict[str, Counter]) -> dict[str, dict[str, int]]:
    return {
        axis_name: normalize_counter(counter, preferred_order=AXIS_LEVELS)
        for axis_name, counter in sorted(axis_counters.items())
        if counter
    }


def accumulate_failed_checks(counter: Counter, validity_payload: object, *, context: str) -> None:
    payload = require_dict(validity_payload, context=context)
    failed_checks = payload.get("failed_checks")
    if failed_checks is None:
        raise ValueError(f"{context} missing required field: failed_checks")
    if not isinstance(failed_checks, list):
        raise ValueError(f"{context}.failed_checks must be a list")
    for failed_check in failed_checks:
        counter[str(failed_check)] += 1


def round3(value: float) -> float:
    return round(float(value), 3)


def summarize_numeric(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "median": None}
    vals = [float(v) for v in values]
    return {
        "count": len(vals),
        "min": round3(min(vals)),
        "max": round3(max(vals)),
        "mean": round3(sum(vals) / float(len(vals))),
        "median": round3(statistics.median(vals)),
    }


def require_route_key(row: dict, *, context: str) -> tuple[str, int, str]:
    scene_id = row.get("scene_id")
    view_id = row.get("view_id")
    routing_id = row.get("routing_id")
    if scene_id is None:
        raise ValueError(f"{context} missing required field: scene_id")
    if view_id is None:
        raise ValueError(f"{context} missing required field: view_id")
    if routing_id is None:
        raise ValueError(f"{context} missing required field: routing_id")
    return (str(scene_id), int(view_id), str(routing_id))


def _source_group(scene_id: str) -> str:
    parts = [part for part in scene_id.split("__") if part]
    if len(parts) < 2:
        return scene_id
    if parts[0] == "matterport3d" and len(parts) >= 3:
        return "__".join(parts[:2])
    return scene_id


def _bundle_identity(row: dict, *, context: str) -> tuple[str, str]:
    task = str(row.get("task", "")).strip().lower()
    scene_id = str(row.get("scene_id", "")).strip()
    if not scene_id:
        question_id = str(row.get("question_id", "")).strip()
        if not question_id:
            raise ValueError(f"{context} missing required field: scene_id")
        return ("unknown_bundle", question_id)
    view_id = row.get("view_id")
    if view_id is None:
        raise ValueError(f"{context} missing required field: view_id")
    if task in {"a1", "b1"}:
        return ("view_bundle", f"{scene_id}::v{int(view_id)}")
    routing_id = row.get("routing_id")
    if routing_id is None:
        raise ValueError(f"{context} missing required field: routing_id")
    return ("route_bundle", f"{scene_id}::v{int(view_id)}::{routing_id}")


def require_reference_resolution_mode(row: dict, *, context: str) -> str:
    payload = require_dict(row.get("navigation_reference_facts"), context=f"{context}.navigation_reference_facts")
    value = payload.get("reference_resolution_mode")
    if value is None:
        raise ValueError(f"{context}.navigation_reference_facts missing required field: reference_resolution_mode")
    return str(value)


def dedupe_navigation_rows(rows: list[dict]) -> tuple[list[dict], dict[tuple[str, int, str], set[str]]]:
    unique_rows: dict[tuple[str, int, str], dict] = {}
    route_task_sets: dict[tuple[str, int, str], set[str]] = defaultdict(set)
    for idx, row in enumerate(rows):
        key = require_route_key(row, context=f"navigation_row[{idx}]")
        route_task_sets[key].add(str(row.get("task", "")))
        if key not in unique_rows:
            unique_rows[key] = row
            continue
        ref_row = unique_rows[key]
        if require_tier_label(ref_row, context=f"unique_navigation_row[{idx}]") != require_tier_label(
            row,
            context=f"navigation_row[{idx}]",
        ):
            raise ValueError(f"conflicting tier for route key={key}")
        if require_axes(
            ref_row,
            "navigation_axes",
            axis_names=NAVIGATION_AXIS_NAMES,
            context=f"unique_navigation_row[{idx}]",
        ) != require_axes(
            row,
            "navigation_axes",
            axis_names=NAVIGATION_AXIS_NAMES,
            context=f"navigation_row[{idx}]",
        ):
            raise ValueError(f"conflicting navigation_axes for route key={key}")
        ref_hash = ref_row.get("facts_hash")
        cur_hash = row.get("facts_hash")
        if ref_hash is not None and cur_hash is not None and str(ref_hash) != str(cur_hash):
            raise ValueError(f"conflicting facts_hash for route key={key}")
    ordered_keys = sorted(unique_rows.keys())
    return [unique_rows[key] for key in ordered_keys], route_task_sets


def navigation_axis_cell_key(axes: dict[str, str]) -> str:
    return f"{axes['reference_axis']}/{axes['geometry_axis']}/{axes['embodiment_axis']}"


def all_navigation_axis_cell_keys() -> list[str]:
    return [
        f"{reference_axis}/{geometry_axis}/{embodiment_axis}"
        for reference_axis in AXIS_LEVELS
        for geometry_axis in AXIS_LEVELS
        for embodiment_axis in AXIS_LEVELS
    ]


def normalize_fixed_counter(counter: Counter, ordered_keys: list[str]) -> dict[str, int]:
    return {key: int(counter.get(key, 0)) for key in ordered_keys}


def compute_axis_collapse_warnings(axis_counters: dict[str, Counter], *, total: int, threshold: float = 0.90) -> list[str]:
    warnings: list[str] = []
    if total <= 0:
        return warnings
    for axis_name, counter in sorted(axis_counters.items()):
        for axis_level in AXIS_LEVELS:
            count = int(counter.get(axis_level, 0))
            if (float(count) / float(total)) > float(threshold):
                warnings.append(f"{axis_name} collapsed to {axis_level} ({round3(count / float(total))})")
    return warnings


def fraction(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round3(float(count) / float(total))


def collect_gt_records(gt_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(gt_dir.glob("gt_next_*.jsonl")):
        rows.extend(load_jsonl(path))
    return rows


def collect_vqa_records(vqa_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(vqa_dir.glob("vqa_next_*.jsonl")):
        rows.extend(load_jsonl(path))
    return rows


def collect_render_bundle_rows(renders_dir: Path, task_outputs_dir: Path | None = None) -> list[dict]:
    rows: list[dict] = []
    classification_cache: dict[str, tuple[dict[int, dict], bool, Path]] = {}
    camera_ready_cache: dict[str, bool] = {}
    for path in sorted(renders_dir.glob("*/*/next_view_bundle_v3.json")):
        with open(path, "r") as f:
            bundle = json.load(f)
        if bundle.get("schema_version") != "next_view_bundle_v3":
            raise ValueError(f"unexpected bundle schema in {path}")
        scene_id = path.parent.parent.name
        view_name = path.parent.name
        view_id = int(view_name.split("view_", 1)[1])
        if scene_id not in camera_ready_cache:
            cameras_path = path.parent.parent / "cameras.json"
            if not cameras_path.exists():
                camera_ready_cache[scene_id] = False
            else:
                with open(cameras_path, "r") as f:
                    camera_ready_cache[scene_id] = bool(json.load(f))
        if not camera_ready_cache[scene_id]:
            continue
        if scene_id not in classification_cache:
            classification_cache[scene_id] = load_scene_classification_views(
                scene_id=scene_id,
                scene_dir=path.parent.parent,
                task_outputs_root=task_outputs_dir,
            )
        classification_views_by_id, using_task_outputs, task_outputs_path = classification_cache[scene_id]
        classification_view = classification_views_by_id.get(view_id)
        if using_task_outputs and classification_view is None:
            raise ValueError(
                f"scene={scene_id} view={view_id} missing protocolized classification view in {task_outputs_path}"
            )
        classification_source = classification_view if classification_view is not None else bundle
        classification_count = classification_source.get("candidate_count")
        if classification_view is not None:
            if classification_count is None:
                raise ValueError(
                    f"scene={scene_id} view={view_id} classification_view missing required field: candidate_count"
                )
            classification_count = int(classification_count)
        else:
            classification_count = len(bundle.get("classification_candidates", []))
        rows.append(
            {
                "scene_id": scene_id,
                "view_id": view_name,
                "classification_count": classification_count,
                "routing_count": len(bundle.get("routing_candidates", [])),
                "affordance_validity": classification_source.get("affordance_validity"),
                "affordance_axes": classification_source.get("affordance_axes"),
                "affordance_tier": classification_source.get("affordance_tier"),
                "routing_candidates": list(bundle.get("routing_candidates", [])),
                "rejection_reasons": list(bundle.get("rejection_reasons", [])),
            }
        )
    return rows


def build_stats(*, renders_dir: Path, gt_dir: Path, vqa_dir: Path, task_outputs_dir: Path | None = None) -> dict:
    gt_rows = collect_gt_records(gt_dir)
    vqa_rows = collect_vqa_records(vqa_dir)
    render_rows = collect_render_bundle_rows(renders_dir, task_outputs_dir=task_outputs_dir)

    gt_task_counts = Counter(str(row.get("task")) for row in gt_rows if row.get("task"))
    vqa_task_counts = Counter(str(row.get("task")) for row in vqa_rows if row.get("task"))
    gt_family_counts = Counter(require_tier_family(row, context=f"gt[{idx}]") for idx, row in enumerate(gt_rows))
    vqa_family_counts = Counter(require_tier_family(row, context=f"vqa[{idx}]") for idx, row in enumerate(vqa_rows))

    navigation_rows: list[dict] = []
    affordance_rows: list[dict] = []
    for idx, row in enumerate(gt_rows):
        family = require_tier_family(row, context=f"gt[{idx}]")
        if family == "navigation":
            navigation_rows.append(row)
        else:
            affordance_rows.append(row)
    unique_navigation_rows, route_task_sets = dedupe_navigation_rows(navigation_rows)
    vqa_navigation_rows = [
        row for idx, row in enumerate(vqa_rows)
        if require_tier_family(row, context=f"vqa[{idx}]") == "navigation"
    ]
    unique_vqa_navigation_route_keys = {
        require_route_key(row, context=f"vqa_navigation[{idx}]")
        for idx, row in enumerate(vqa_navigation_rows)
        if row.get("routing_id") is not None
    }

    category_counter = Counter()
    color_counter = Counter()
    reference_family_counter = Counter()
    reference_resolution_counter = Counter()
    navigation_tier_counter = Counter()
    affordance_tier_counter = Counter()
    navigation_axis_counters = {axis_name: Counter() for axis_name in NAVIGATION_AXIS_NAMES}
    affordance_axis_counters = {axis_name: Counter() for axis_name in AFFORDANCE_AXIS_NAMES}
    navigation_axis_cell_counter = Counter()
    path_lengths: list[float] = []
    ambiguity_competition = 0
    same_group_diff_color_total = 0
    same_group_same_color_total = 0
    eligibility_failures = {
        "navigation": Counter(),
        "affordance": Counter(),
    }
    rejection_reason_counter = Counter()
    unique_category_counter = Counter()
    unique_color_counter = Counter()
    unique_reference_family_counter = Counter()
    unique_reference_resolution_counter = Counter()
    unique_navigation_tier_counter = Counter()
    unique_navigation_axis_counters = {axis_name: Counter() for axis_name in NAVIGATION_AXIS_NAMES}
    unique_navigation_axis_cell_counter = Counter()

    for row_idx, row in enumerate(navigation_rows):
        category = str(row.get("canonical_category") or row.get("target_category") or "unknown")
        category_counter[category] += 1

        color_name = row.get("target_color_name")
        if color_name:
            color_counter[str(color_name)] += 1
        reference_family = row.get("reference_family_id") or row.get("semantic_group_id")
        if reference_family:
            reference_family_counter[str(reference_family)] += 1
        reference_resolution_counter[
            require_reference_resolution_mode(row, context=f"gt_navigation[{row_idx}]")
        ] += 1

        complexity = row.get("routing_complexity") or {}
        if "path_length_raw_m" in complexity:
            path_lengths.append(float(complexity["path_length_raw_m"]))

        tier = require_tier_label(row, context=f"gt_navigation[{row_idx}]")
        navigation_tier_counter[tier] += 1
        axes = require_axes(
            row,
            "navigation_axes",
            axis_names=NAVIGATION_AXIS_NAMES,
            context=f"gt_navigation[{row_idx}]",
        )
        for axis_name, axis_value in axes.items():
            navigation_axis_counters[axis_name][axis_value] += 1
        navigation_axis_cell_counter[navigation_axis_cell_key(axes)] += 1

        reference_facts = row.get("navigation_reference_facts") or {}
        same_group_diff = max(
            int(complexity.get("num_same_group_diff_color", 0)),
            int(reference_facts.get("same_semantic_group_diff_color_count", 0) or 0),
        )
        same_group_same = max(
            int(complexity.get("num_same_group_same_color", 0)),
            int(reference_facts.get("same_semantic_group_same_color_count", 0) or 0),
        )
        if (same_group_diff + same_group_same) > 0:
            ambiguity_competition += 1
        same_group_diff_color_total += same_group_diff
        same_group_same_color_total += same_group_same

    for row_idx, row in enumerate(unique_navigation_rows):
        category = str(row.get("canonical_category") or row.get("target_category") or "unknown")
        unique_category_counter[category] += 1
        color_name = row.get("target_color_name")
        if color_name:
            unique_color_counter[str(color_name)] += 1
        reference_family = row.get("reference_family_id") or row.get("semantic_group_id")
        if reference_family:
            unique_reference_family_counter[str(reference_family)] += 1
        unique_reference_resolution_counter[
            require_reference_resolution_mode(row, context=f"unique_navigation[{row_idx}]")
        ] += 1
        tier = require_tier_label(row, context=f"unique_navigation[{row_idx}]")
        unique_navigation_tier_counter[tier] += 1
        axes = require_axes(
            row,
            "navigation_axes",
            axis_names=NAVIGATION_AXIS_NAMES,
            context=f"unique_navigation[{row_idx}]",
        )
        for axis_name, axis_value in axes.items():
            unique_navigation_axis_counters[axis_name][axis_value] += 1
        unique_navigation_axis_cell_counter[navigation_axis_cell_key(axes)] += 1

    for row_idx, row in enumerate(affordance_rows):
        tier = require_tier_label(row, context=f"gt_affordance[{row_idx}]")
        affordance_tier_counter[tier] += 1
        axes = require_axes(
            row,
            "affordance_axes",
            axis_names=AFFORDANCE_AXIS_NAMES,
            context=f"gt_affordance[{row_idx}]",
        )
        for axis_name, axis_value in axes.items():
            affordance_axis_counters[axis_name][axis_value] += 1

    for render_idx, row in enumerate(render_rows):
        accumulate_failed_checks(
            eligibility_failures["affordance"],
            row.get("affordance_validity"),
            context=f"render[{render_idx}].affordance_validity",
        )
        for routing_idx, routing in enumerate(row.get("routing_candidates", [])):
            accumulate_failed_checks(
                eligibility_failures["navigation"],
                routing.get("navigation_validity"),
                context=f"render[{render_idx}].routing[{routing_idx}].navigation_validity",
            )
        for rejection in row.get("rejection_reasons", []):
            if not isinstance(rejection, dict):
                continue
            reason = rejection.get("reason")
            if reason:
                rejection_reason_counter[str(reason)] += 1

    classification_counts = [row["classification_count"] for row in render_rows]
    routing_counts = [row["routing_count"] for row in render_rows]
    views_with_same_family_competition = 0
    for row in render_rows:
        family_counter = Counter()
        for candidate in row.get("routing_candidates", []):
            family_id = candidate.get("reference_family_id") or candidate.get("semantic_group_id")
            if family_id:
                family_counter[str(family_id)] += 1
        if any(count >= 2 for count in family_counter.values()):
            views_with_same_family_competition += 1

    hard_by_embodiment_only = 0
    nontrivial_reference_routes = 0
    nontrivial_geometry_routes = 0
    for row_idx, row in enumerate(unique_navigation_rows):
        axes = require_axes(
            row,
            "navigation_axes",
            axis_names=NAVIGATION_AXIS_NAMES,
            context=f"validation_navigation[{row_idx}]",
        )
        if axes["reference_axis"] != "low":
            nontrivial_reference_routes += 1
        if axes["geometry_axis"] != "low":
            nontrivial_geometry_routes += 1
        if (
            require_tier_label(row, context=f"validation_navigation[{row_idx}]") == "hard"
            and axes["reference_axis"] == "low"
            and axes["geometry_axis"] == "low"
            and axes["embodiment_axis"] == "high"
        ):
            hard_by_embodiment_only += 1

    navigation_axis_collapse_warnings = compute_axis_collapse_warnings(
        unique_navigation_axis_counters,
        total=len(unique_navigation_rows),
    )
    unique_navigation_nonzero_axis_cells = sum(
        1 for value in unique_navigation_axis_cell_counter.values() if value > 0
    )
    monotonicity_present = False
    fraction_hard_embodiment_only = fraction(hard_by_embodiment_only, len(unique_navigation_rows))
    fraction_nontrivial_reference = fraction(nontrivial_reference_routes, len(unique_navigation_rows))
    fraction_nontrivial_geometry = fraction(nontrivial_geometry_routes, len(unique_navigation_rows))
    readiness_gate_failures: list[str] = []
    if navigation_axis_collapse_warnings:
        readiness_gate_failures.append("axis_collapse_present")
    if unique_navigation_nonzero_axis_cells < 6:
        readiness_gate_failures.append("mixed_axis_nonzero_cells_lt_6")
    if fraction_nontrivial_reference < MIN_READY_NONTRIVIAL_REFERENCE_FRACTION:
        readiness_gate_failures.append("reference_support_too_thin")
    if fraction_nontrivial_geometry < MIN_READY_NONTRIVIAL_GEOMETRY_FRACTION:
        readiness_gate_failures.append("geometry_support_too_thin")
    if fraction_hard_embodiment_only > MAX_READY_HARD_EMBODIMENT_ONLY_FRACTION:
        readiness_gate_failures.append("hard_routes_dominated_by_embodiment_only")

    navigation_validation_label = "dev-only"
    if not readiness_gate_failures:
        navigation_validation_label = "almost-paper-ready"
    if not readiness_gate_failures and monotonicity_present:
        navigation_validation_label = "paper-ready-evidence-present"

    a2_b2_c_tasks = {"a2", "b2", "c"}
    a2_b2_c_vqa_rows = [row for row in vqa_navigation_rows if str(row.get("task")) in a2_b2_c_tasks]
    a2_b2_c_unique_routes = {
        key for key, tasks in route_task_sets.items()
        if tasks & a2_b2_c_tasks
    }
    a2_b2_c_shared_routes = sum(
        1
        for tasks in route_task_sets.values()
        if len(tasks & a2_b2_c_tasks) >= 2
    )

    vqa_bundle_keys = {_bundle_identity(row, context=f"vqa_bundle[{idx}]") for idx, row in enumerate(vqa_rows)}
    gt_bundle_keys = {_bundle_identity(row, context=f"gt_bundle[{idx}]") for idx, row in enumerate(gt_rows)}
    gt_scene_ids = {str(row["scene_id"]) for row in gt_rows if row.get("scene_id") is not None}
    vqa_scene_ids = {str(row["scene_id"]) for row in vqa_rows if row.get("scene_id") is not None}
    reporting = {
        "question_variants": {
            "gt_total": len(gt_rows),
            "vqa_total": len(vqa_rows),
        },
        "unique_bundles": {
            "gt_total": len(gt_bundle_keys),
            "vqa_total": len(vqa_bundle_keys),
        },
        "unique_scenes": {
            "gt_total": len(gt_scene_ids),
            "vqa_total": len(vqa_scene_ids),
        },
        "unique_source_groups": {
            "gt_total": len({_source_group(scene_id) for scene_id in gt_scene_ids}),
            "vqa_total": len({_source_group(scene_id) for scene_id in vqa_scene_ids}),
        },
    }

    return {
        "gt_task_counts": dict(sorted(gt_task_counts.items())),
        "vqa_task_counts": dict(sorted(vqa_task_counts.items())),
        "gt_family_counts": normalize_counter(gt_family_counts, preferred_order=TIER_FAMILIES),
        "vqa_family_counts": normalize_counter(vqa_family_counts, preferred_order=TIER_FAMILIES),
        "routing_category_distribution": dict(sorted(category_counter.items())),
        "routing_category_distribution_unique_routes": dict(sorted(unique_category_counter.items())),
        "routing_color_distribution": dict(sorted(color_counter.items())),
        "routing_color_distribution_unique_routes": dict(sorted(unique_color_counter.items())),
        "routing_reference_family_distribution": dict(sorted(reference_family_counter.items())),
        "routing_reference_family_distribution_unique_routes": dict(sorted(unique_reference_family_counter.items())),
        "navigation_reference_resolution_distribution": dict(sorted(reference_resolution_counter.items())),
        "navigation_reference_resolution_distribution_unique_routes": dict(sorted(unique_reference_resolution_counter.items())),
        "navigation_tier_distribution": normalize_counter(navigation_tier_counter, preferred_order=TIER_LEVELS),
        "navigation_tier_distribution_unique_routes": normalize_counter(
            unique_navigation_tier_counter,
            preferred_order=TIER_LEVELS,
        ),
        "affordance_tier_distribution": normalize_counter(affordance_tier_counter, preferred_order=TIER_LEVELS),
        "navigation_axis_distribution": summarize_axis_counters(navigation_axis_counters),
        "navigation_axis_distribution_unique_routes": summarize_axis_counters(unique_navigation_axis_counters),
        "navigation_axis_cell_distribution": normalize_fixed_counter(
            navigation_axis_cell_counter,
            all_navigation_axis_cell_keys(),
        ),
        "navigation_axis_cell_distribution_unique_routes": normalize_fixed_counter(
            unique_navigation_axis_cell_counter,
            all_navigation_axis_cell_keys(),
        ),
        "affordance_axis_distribution": summarize_axis_counters(affordance_axis_counters),
        "unique_route_counts": {
            "navigation_gt_unique_routes": len(unique_navigation_rows),
            "navigation_vqa_unique_routes": len(unique_vqa_navigation_route_keys),
        },
        "eligibility_failures": {
            family: normalize_counter(counter)
            for family, counter in eligibility_failures.items()
        },
        "rejection_reason_distribution": dict(sorted(rejection_reason_counter.items())),
        "routing_path_length_raw_m": summarize_numeric(path_lengths),
        "render_candidate_counts": {
            "classification_per_view": summarize_numeric(classification_counts),
            "routing_per_view": summarize_numeric(routing_counts),
        },
        "routing_ambiguity": {
            "questions_with_competition": ambiguity_competition,
            "same_group_diff_color_total": same_group_diff_color_total,
            "same_group_same_color_total": same_group_same_color_total,
            "views_with_same_family_competition": views_with_same_family_competition,
        },
        "a2_b2_c_route_reuse": {
            "vqa_item_count": len(a2_b2_c_vqa_rows),
            "unique_route_count": len(a2_b2_c_unique_routes),
            "duplicate_item_count": max(0, len(a2_b2_c_vqa_rows) - len(a2_b2_c_unique_routes)),
            "shared_multi_task_route_count": int(a2_b2_c_shared_routes),
            "shared_multi_task_route_fraction": fraction(a2_b2_c_shared_routes, len(a2_b2_c_unique_routes)),
            "vqa_items_per_unique_route": round3(
                float(len(a2_b2_c_vqa_rows)) / float(max(len(a2_b2_c_unique_routes), 1))
            ),
        },
        "navigation_tier_validation": {
            "label": navigation_validation_label,
            "axis_collapse_warnings": navigation_axis_collapse_warnings,
            "readiness_gate_failures": readiness_gate_failures,
            "mixed_axis_nonzero_cells_unique_routes": unique_navigation_nonzero_axis_cells,
            "fraction_hard_embodiment_only": fraction_hard_embodiment_only,
            "fraction_nontrivial_reference_unique_routes": fraction_nontrivial_reference,
            "fraction_nontrivial_geometry_unique_routes": fraction_nontrivial_geometry,
            "monotonicity_evidence_present": monotonicity_present,
        },
        "corpus": {
            "gt_records_total": len(gt_rows),
            "vqa_records_total": len(vqa_rows),
            "render_views_total": len(render_rows),
            "render_scenes_total": len({row["scene_id"] for row in render_rows}),
        },
        "reporting": reporting,
    }


def build_markdown(stats: dict) -> str:
    lines = [
        "# Next Dataset Stats",
        "",
        "## GT Task Counts",
        "",
    ]
    for task, count in stats.get("gt_task_counts", {}).items():
        lines.append(f"- `{task}`: {count}")

    lines.extend(
        [
            "",
            "## VQA Task Counts",
            "",
        ]
    )
    for task, count in stats.get("vqa_task_counts", {}).items():
        lines.append(f"- `{task}`: {count}")

    lines.extend(
        [
            "",
            "## Reporting",
            "",
        ]
    )
    for key, payload in stats.get("reporting", {}).items():
        lines.append(f"- `{key}`: {payload}")

    lines.extend(
        [
            "",
            "## GT Family Counts",
            "",
        ]
    )
    for family, count in stats.get("gt_family_counts", {}).items():
        lines.append(f"- `{family}`: {count}")

    lines.extend(
        [
            "",
            "## VQA Family Counts",
            "",
        ]
    )
    for family, count in stats.get("vqa_family_counts", {}).items():
        lines.append(f"- `{family}`: {count}")

    lines.extend(
        [
            "",
            "## Navigation Tier Distribution",
            "",
        ]
    )
    for tier, count in stats.get("navigation_tier_distribution", {}).items():
        lines.append(f"- `{tier}`: {count}")

    lines.extend(
        [
            "",
            "## Navigation Tier Distribution Unique Routes",
            "",
        ]
    )
    for tier, count in stats.get("navigation_tier_distribution_unique_routes", {}).items():
        lines.append(f"- `{tier}`: {count}")

    lines.extend(
        [
            "",
            "## Affordance Tier Distribution",
            "",
        ]
    )
    for tier, count in stats.get("affordance_tier_distribution", {}).items():
        lines.append(f"- `{tier}`: {count}")

    lines.extend(
        [
            "",
            "## Routing Category Distribution",
            "",
        ]
    )
    for category, count in stats.get("routing_category_distribution", {}).items():
        lines.append(f"- `{category}`: {count}")

    lines.extend(
        [
            "",
            "## Routing Color Distribution",
            "",
        ]
    )
    for color, count in stats.get("routing_color_distribution", {}).items():
        lines.append(f"- `{color}`: {count}")

    lines.extend(
        [
            "",
            "## Routing Reference Family Distribution",
            "",
        ]
    )
    for family, count in stats.get("routing_reference_family_distribution", {}).items():
        lines.append(f"- `{family}`: {count}")

    lines.extend(
        [
            "",
            "## Navigation Reference Resolution Distribution",
            "",
            f"- {json.dumps(stats.get('navigation_reference_resolution_distribution', {}), ensure_ascii=False)}",
            "",
            "## Navigation Reference Resolution Distribution Unique Routes",
            "",
            f"- {json.dumps(stats.get('navigation_reference_resolution_distribution_unique_routes', {}), ensure_ascii=False)}",
            "",
            "## Path Length Raw",
            "",
            f"- {json.dumps(stats.get('routing_path_length_raw_m', {}), ensure_ascii=False)}",
            "",
            "## Navigation Axis Distribution",
            "",
            f"- {json.dumps(stats.get('navigation_axis_distribution', {}), ensure_ascii=False)}",
            "",
            "## Navigation Axis Distribution Unique Routes",
            "",
            f"- {json.dumps(stats.get('navigation_axis_distribution_unique_routes', {}), ensure_ascii=False)}",
            "",
            "## Navigation Axis Cell Distribution",
            "",
            f"- {json.dumps(stats.get('navigation_axis_cell_distribution', {}), ensure_ascii=False)}",
            "",
            "## Navigation Axis Cell Distribution Unique Routes",
            "",
            f"- {json.dumps(stats.get('navigation_axis_cell_distribution_unique_routes', {}), ensure_ascii=False)}",
            "",
            "## Affordance Axis Distribution",
            "",
            f"- {json.dumps(stats.get('affordance_axis_distribution', {}), ensure_ascii=False)}",
            "",
            "## Unique Route Counts",
            "",
            f"- {json.dumps(stats.get('unique_route_counts', {}), ensure_ascii=False)}",
            "",
            "## Eligibility Failures",
            "",
            f"- {json.dumps(stats.get('eligibility_failures', {}), ensure_ascii=False)}",
            "",
            "## Rejection Reasons",
            "",
            f"- {json.dumps(stats.get('rejection_reason_distribution', {}), ensure_ascii=False)}",
            "",
            "## Render Candidate Counts",
            "",
            f"- classification_per_view: {json.dumps(stats['render_candidate_counts']['classification_per_view'], ensure_ascii=False)}",
            f"- routing_per_view: {json.dumps(stats['render_candidate_counts']['routing_per_view'], ensure_ascii=False)}",
            "",
            "## Routing Ambiguity",
            "",
            f"- {json.dumps(stats.get('routing_ambiguity', {}), ensure_ascii=False)}",
            "",
            "## A2B2C Route Reuse",
            "",
            f"- {json.dumps(stats.get('a2_b2_c_route_reuse', {}), ensure_ascii=False)}",
            "",
            "## Navigation Tier Validation",
            "",
            f"- {json.dumps(stats.get('navigation_tier_validation', {}), ensure_ascii=False)}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Report Next dataset stats from renders / gt / vqa artifacts.")
    parser.add_argument("--renders-dir", required=True)
    parser.add_argument("--gt-dir", required=True)
    parser.add_argument("--vqa-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-outputs-dir", default=None)
    args = parser.parse_args()

    renders_dir = Path(args.renders_dir).resolve()
    gt_dir = Path(args.gt_dir).resolve()
    vqa_dir = Path(args.vqa_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    task_outputs_dir = Path(args.task_outputs_dir).resolve() if args.task_outputs_dir else None

    stats = build_stats(
        renders_dir=renders_dir,
        gt_dir=gt_dir,
        vqa_dir=vqa_dir,
        task_outputs_dir=task_outputs_dir,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    with open(output_dir / "dataset_stats.md", "w") as f:
        f.write(build_markdown(stats))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
