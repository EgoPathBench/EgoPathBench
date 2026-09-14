#!/usr/bin/env python3
"""Build GPT oracle-CoT packets and LLaMA-Factory SFT data for NavBench3D."""

from __future__ import annotations

import argparse
import fcntl
import http.client
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

TASKS = ("a1", "b1", "a2", "b2", "c")
DEFAULT_RELEASE_ROOT = Path("published/navbench3d_release_20260417_rebuild5/data/release")
DEFAULT_LF_DATA_DIR = Path("results/oracle_cot_llamafactory_data_current")
STAGE_LIMITS = {
    "pilot": 20,
    "validation": 200,
    "full": None,
    "custom": None,
}
REASONING_TYPES = {
    "a1": "pointmass_traversability",
    "b1": "embodied_traversability",
    "a2": "pointmass_explicit_shortest_path",
    "b2": "embodied_explicit_shortest_path",
    "c": "implicit_goal_embodied_shortest_path",
}
SCHEMA_REQUIRED_KEYS = {
    "a1": {
        "reasoning_payload": ("candidate_space", "selection_rule"),
        "candidate_space": ("num_candidates", "positive_examples", "negative_examples"),
    },
    "b1": {
        "reasoning_payload": ("robot_constraint", "candidate_space", "selection_rule"),
        "candidate_space": ("num_candidates", "positive_examples", "negative_examples"),
    },
    "a2": {
        "reasoning_payload": ("target_grounding", "path_facts", "shortest_path_explanation"),
        "target_grounding": ("target_rule", "target_category", "goal_display_id"),
        "path_facts": ("start_id", "shortest_path", "path_length_m", "route_semantics"),
    },
    "b2": {
        "reasoning_payload": ("target_grounding", "robot_constraint", "path_facts", "embodiment_explanation"),
        "target_grounding": ("target_rule", "target_category", "goal_display_id"),
        "robot_constraint": ("robot_diameter_m", "min_clearance_m", "narrow_passage_fraction"),
        "path_facts": ("start_id", "shortest_path", "path_length_m", "route_semantics"),
    },
    "c": {
        "reasoning_payload": ("intent_resolution", "cue_resolution", "target_selection", "path_facts"),
        "intent_resolution": ("user_request", "intent_family", "resolved_target_category"),
        "cue_resolution": ("cue_family", "cue_type"),
        "target_selection": ("candidate_target_ids", "winner_target_id", "goal_display_id", "selection_reason"),
        "path_facts": ("start_id", "shortest_path", "route_semantics"),
    },
}
BANNED_OUTPUT_PATTERNS = (
    "ground truth says",
    "metadata says",
    "gt says",
    "facts_hash",
    "gt_hash",
    "prompt_hash",
    "hash",
    "world coordinates",
    "world coordinate",
    "world_xyz",
    "visible_waypoints_path",
    "ground_truth",
    "/mnt/",
)


class OracleCotError(RuntimeError):
    """Raised for invalid input, output, or configuration."""


RETRYABLE_REQUEST_ERRORS = (
    OracleCotError,
    json.JSONDecodeError,
    urllib.error.URLError,
    TimeoutError,
    http.client.HTTPException,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def log_progress(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {message}", flush=True)


def is_full_external_generation(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "stage", None) == "full"
        and getattr(args, "generate", False)
        and getattr(args, "fail_fast_on_api_error", False)
    )


def retry_sleep_seconds(args: argparse.Namespace, attempt: int) -> float:
    sleep_s = args.retry_sleep_s * (attempt + 1)
    if is_full_external_generation(args):
        sleep_s = max(sleep_s, min(900.0, args.retry_sleep_s * (2**attempt)))
    if args.retry_jitter_s > 0:
        sleep_s += random.uniform(0.0, args.retry_jitter_s)
    return sleep_s


def final_api_error_cooldown_seconds(args: argparse.Namespace) -> float:
    if not is_full_external_generation(args):
        return 0.0
    cooldown_s = max(600.0, args.retry_sleep_s * 8)
    if args.retry_jitter_s > 0:
        cooldown_s += random.uniform(0.0, args.retry_jitter_s)
    return cooldown_s


def should_defer_transient_api_error(args: argparse.Namespace) -> bool:
    if not is_full_external_generation(args):
        return False
    return bool(getattr(args, "defer_transient_api_errors", True))


def request_max_retries(args: argparse.Namespace) -> int:
    configured = max(0, int(getattr(args, "max_retries", 0)))
    if not should_defer_transient_api_error(args):
        return configured
    defer_cap = max(0, int(getattr(args, "defer_max_retries", 1)))
    return min(configured, defer_cap)


def full_generation_request_interval(args: argparse.Namespace) -> float:
    if not is_full_external_generation(args):
        return 0.0
    return max(0.0, float(getattr(args, "full_generation_request_interval_s", 0.0) or 0.0))


def full_generation_max_inflight(args: argparse.Namespace) -> int:
    if not is_full_external_generation(args):
        return 0
    value = int(getattr(args, "full_generation_max_inflight", 0) or 0)
    if value < 0:
        raise OracleCotError("--full-generation-max-inflight must be >= 0")
    if value > 1:
        raise OracleCotError("--full-generation-max-inflight currently supports 0 or 1")
    return value


@contextmanager
def full_generation_inflight_slot(args: argparse.Namespace, *, shard_label: str) -> Iterable[None]:
    max_inflight = full_generation_max_inflight(args)
    if max_inflight <= 0:
        yield
        return

    lock_path = Path(args.output_dir) / "full_generation_inflight.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as f:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log_progress(
                "generation_inflight_wait "
                f"shard={shard_label} max_inflight={max_inflight}"
            )
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def wait_for_full_generation_request_slot(args: argparse.Namespace, *, shard_label: str) -> None:
    interval_s = full_generation_request_interval(args)
    if interval_s <= 0:
        return

    gate_path = Path(args.output_dir) / "full_generation_request_gate.lock"
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        with gate_path.open("a+") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            f.seek(0)
            raw = f.read().strip()
            try:
                last_request_s = float(raw) if raw else 0.0
            except ValueError:
                last_request_s = 0.0
            now_s = time.time()
            wait_s = max(0.0, last_request_s + interval_s - now_s)
            if wait_s <= 0:
                f.seek(0)
                f.truncate()
                f.write(str(now_s))
                f.flush()
                os.fsync(f.fileno())
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                return
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        log_progress(
            "generation_request_gate "
            f"shard={shard_label} sleep_s={round(wait_s, 3)} interval_s={round(interval_s, 3)}"
        )
        time.sleep(wait_s)


def write_json_array(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(rows, f, ensure_ascii=True, indent=2)
        f.write("\n")


def parse_tasks(raw: str) -> tuple[str, ...]:
    tasks = tuple(part.strip() for part in raw.split(",") if part.strip())
    unknown = [task for task in tasks if task not in TASKS]
    if unknown:
        raise OracleCotError(f"unknown tasks: {unknown}")
    return tasks


def int_list(value: Any, *, field: str) -> list[int]:
    if not isinstance(value, list):
        raise OracleCotError(f"{field} must be a list")
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise OracleCotError(f"{field} must contain integer-like values") from exc


def get_gt(row: dict[str, Any]) -> dict[str, Any]:
    gt = row.get("ground_truth")
    return gt if isinstance(gt, dict) else {}


def merge_row_with_gt(vqa_row: dict[str, Any], gt_row: dict[str, Any] | None) -> dict[str, Any]:
    if gt_row is None:
        return vqa_row
    merged = json.loads(json.dumps(vqa_row))
    gt = merged.setdefault("ground_truth", {})
    for key, value in gt_row.items():
        if key in {"question_id", "task"}:
            continue
        if merged.get(key) is None and key in {"scene_id", "view_id", "routing_id", "image_path", "visible_waypoints_path"}:
            merged[key] = value
        if isinstance(gt, dict) and gt.get(key) is None:
            gt[key] = value
    return merged


def answer_from_row(row: dict[str, Any]) -> list[int]:
    gt = get_gt(row)
    answer = gt.get("answer")
    if answer is None:
        answer = gt.get("path_ids")
    return int_list(answer, field=f"answer for question_id={row.get('question_id')}")


def stable_sample(rows: list[dict[str, Any]], *, task: str, limit: int | None, seed: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: str(row.get("question_id") or ""))
    if limit is None or len(ordered) <= limit:
        return ordered
    rng = random.Random(f"{seed}:{task}")
    shuffled = list(ordered)
    rng.shuffle(shuffled)
    return sorted(shuffled[:limit], key=lambda row: str(row.get("question_id") or ""))


def load_visible_waypoints(path: str | None) -> list[dict[str, Any]]:
    if not path:
        return []
    visible_path = Path(path)
    if not visible_path.exists():
        return []
    payload = json.loads(visible_path.read_text())
    visible = payload.get("visible_waypoints")
    return visible if isinstance(visible, list) else []


def round_float(value: Any, digits: int = 3) -> Any:
    if isinstance(value, (int, float)):
        return round(float(value), digits)
    return value


def first_n(items: Iterable[Any], n: int) -> list[Any]:
    out: list[Any] = []
    for item in items:
        out.append(item)
        if len(out) >= n:
            break
    return out


def waypoint_fact(waypoint: dict[str, Any]) -> dict[str, Any]:
    fact = {
        "display_id": int(waypoint["display_id"]),
        "pointmass_walkable": bool(waypoint.get("pointmass_walkable")),
        "embodied_feasible": bool(waypoint.get("embodied_feasible")),
    }
    if waypoint.get("point_clearance_m") is not None:
        fact["point_clearance_m"] = round_float(waypoint.get("point_clearance_m"))
    if waypoint.get("embodied_clearance_margin_m") is not None:
        fact["embodied_clearance_margin_m"] = round_float(waypoint.get("embodied_clearance_margin_m"))
    if waypoint.get("screen_xy_norm") is not None:
        fact["screen_xy_norm"] = [round_float(item, 4) for item in waypoint.get("screen_xy_norm", [])]
    return fact


def visible_index(visible_waypoints: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for waypoint in visible_waypoints:
        try:
            out[int(waypoint["display_id"])] = waypoint
        except (KeyError, TypeError, ValueError):
            continue
    return out


def candidate_display_ids(gt: dict[str, Any], visible_waypoints: list[dict[str, Any]]) -> list[int]:
    if isinstance(gt.get("all_ids"), list):
        return sorted({int(item) for item in gt["all_ids"]})
    return sorted(visible_index(visible_waypoints))


def selection_examples(ids: list[int], visible_by_id: dict[int, dict[str, Any]], *, positive: bool, embodied: bool) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for display_id in ids:
        waypoint = visible_by_id.get(display_id)
        reason_parts: list[str] = []
        if waypoint is not None:
            if embodied:
                reason_parts.append(f"embodied_feasible={str(bool(waypoint.get('embodied_feasible'))).lower()}")
                margin = waypoint.get("embodied_clearance_margin_m")
                if margin is not None:
                    reason_parts.append(f"embodied_clearance_margin_m={round_float(margin)}")
            else:
                reason_parts.append(f"pointmass_walkable={str(bool(waypoint.get('pointmass_walkable'))).lower()}")
                clearance = waypoint.get("point_clearance_m")
                if clearance is not None:
                    reason_parts.append(f"point_clearance_m={round_float(clearance)}")
        if not reason_parts:
            reason_parts.append("listed in oracle candidate labels" if positive else "excluded by oracle candidate labels")
        examples.append({"id": display_id, "reason": "; ".join(reason_parts)})
        if len(examples) >= 3:
            break
    return examples


def visible_point_summary(visible_waypoints: list[dict[str, Any]], *, embodied: bool) -> dict[str, Any]:
    mode_key = "embodied_feasible" if embodied else "pointmass_walkable"
    feasible_ids: list[int] = []
    blocked_ids: list[int] = []
    for waypoint in visible_waypoints:
        try:
            display_id = int(waypoint["display_id"])
        except (KeyError, TypeError, ValueError):
            continue
        if bool(waypoint.get(mode_key)):
            feasible_ids.append(display_id)
        else:
            blocked_ids.append(display_id)
    return {
        "mode": "embodied" if embodied else "pointmass",
        "total_visible_points": len(feasible_ids) + len(blocked_ids),
        "feasible_count": len(feasible_ids),
        "blocked_count": len(blocked_ids),
        "feasible_display_ids": feasible_ids,
        "blocked_display_ids": blocked_ids,
    }


def compact_selected_fields(source: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in keys:
        value = source.get(key)
        if value is not None:
            out[key] = value
    return out


def build_a1_packet(row: dict[str, Any], visible_waypoints: list[dict[str, Any]]) -> dict[str, Any]:
    gt = get_gt(row)
    all_ids = candidate_display_ids(gt, visible_waypoints)
    walkable_ids = sorted({int(item) for item in gt.get("walkable_ids") or gt.get("answer") or []})
    negative_ids = [display_id for display_id in all_ids if display_id not in set(walkable_ids)]
    visible_by_id = visible_index(visible_waypoints)
    return {
        "reasoning_type": REASONING_TYPES["a1"],
        "facts": {
            "candidate_space": {
                "num_candidates": len(all_ids),
                "all_display_ids": all_ids,
                "pointmass_walkable_ids": walkable_ids,
                "non_walkable_ids": negative_ids,
                "positive_examples": selection_examples(walkable_ids, visible_by_id, positive=True, embodied=False),
                "negative_examples": selection_examples(negative_ids, visible_by_id, positive=False, embodied=False),
            },
            "affordance_axes": gt.get("affordance_axes") or {},
            "selection_rule": "Select all display IDs traversable for a point agent.",
        },
    }


def build_b1_packet(row: dict[str, Any], visible_waypoints: list[dict[str, Any]]) -> dict[str, Any]:
    gt = get_gt(row)
    all_ids = candidate_display_ids(gt, visible_waypoints)
    feasible_ids = sorted({int(item) for item in gt.get("walkable_ids") or gt.get("answer") or []})
    negative_ids = [display_id for display_id in all_ids if display_id not in set(feasible_ids)]
    visible_by_id = visible_index(visible_waypoints)
    return {
        "reasoning_type": REASONING_TYPES["b1"],
        "facts": {
            "robot_constraint": {"robot_diameter_m": round_float(gt.get("robot_diameter_m"))},
            "candidate_space": {
                "num_candidates": len(all_ids),
                "all_display_ids": all_ids,
                "embodied_feasible_ids": feasible_ids,
                "not_embodied_feasible_ids": negative_ids,
                "positive_examples": selection_examples(feasible_ids, visible_by_id, positive=True, embodied=True),
                "negative_examples": selection_examples(negative_ids, visible_by_id, positive=False, embodied=True),
            },
            "affordance_axes": gt.get("affordance_axes") or {},
            "selection_rule": "Select all display IDs traversable for the robot body.",
        },
    }


def path_waypoint_facts(answer: list[int], visible_waypoints: list[dict[str, Any]]) -> list[dict[str, Any]]:
    visible_by_id = visible_index(visible_waypoints)
    facts = []
    for display_id in answer:
        waypoint = visible_by_id.get(display_id)
        if waypoint is None:
            facts.append({"display_id": display_id, "visibility_fact": "missing_visible_waypoint_record"})
        else:
            facts.append(waypoint_fact(waypoint))
    return facts


def path_segment_facts(answer: list[int], *, embodied: bool, routing: dict[str, Any]) -> list[dict[str, Any]]:
    """Return adjacent route segments with collision/projection safety evidence."""
    if len(answer) < 2:
        return []
    projection_key = "embodied_path_projection_safe" if embodied else "point_path_projection_safe"
    dense_projection_key = "embodied_dense_path_projection_safe" if embodied else "point_dense_path_projection_safe"
    fallback_projection_key = "path_projection_safe"
    fallback_dense_key = "dense_path_projection_safe"
    segment_count_key = "embodied_num_segments_sparse" if embodied else "num_segments_sparse"
    num_segments = routing.get(segment_count_key)
    path_projection_safe = routing.get(projection_key)
    if path_projection_safe is None:
        path_projection_safe = routing.get(fallback_projection_key)
    dense_projection_safe = routing.get(dense_projection_key)
    if dense_projection_safe is None:
        dense_projection_safe = routing.get(fallback_dense_key)
    facts = []
    for source, target in zip(answer[:-1], answer[1:]):
        item: dict[str, Any] = {
            "from_id": int(source),
            "to_id": int(target),
            "route_semantics": "embodied" if embodied else "pointmass",
        }
        if path_projection_safe is not None:
            item["collision_free_projection"] = bool(path_projection_safe)
        if dense_projection_safe is not None:
            item["dense_collision_free_projection"] = bool(dense_projection_safe)
        if num_segments is not None:
            item["sparse_segment_count"] = int(num_segments)
        facts.append(item)
    return facts


def build_a2_packet(row: dict[str, Any], visible_waypoints: list[dict[str, Any]]) -> dict[str, Any]:
    gt = get_gt(row)
    answer = answer_from_row(row)
    routing = gt.get("routing_complexity") or {}
    return {
        "reasoning_type": REASONING_TYPES["a2"],
        "facts": {
            "visual_summary": visible_point_summary(visible_waypoints, embodied=False),
            "target_grounding": {
                "target_rule": gt.get("target_rule"),
                "target_category": gt.get("target_category") or gt.get("canonical_category"),
                "target_color_name": gt.get("target_color_name"),
                "goal_display_id": gt.get("goal_id"),
            },
            "path_facts": {
                "start_id": gt.get("start_id"),
                "shortest_path": answer,
                "path_length_m": round_float(gt.get("canonical_sparse_length_m") or gt.get("optimal_length_m")),
                "route_semantics": gt.get("route_semantics") or "pointmass",
                "path_waypoints": path_waypoint_facts(answer, visible_waypoints),
                "path_segments": path_segment_facts(answer, embodied=False, routing=routing),
                "path_projection_safe": routing.get("point_path_projection_safe")
                if routing.get("point_path_projection_safe") is not None
                else routing.get("path_projection_safe"),
            },
            "navigation_geometry_facts": compact_selected_fields(
                gt.get("navigation_geometry_facts") or {},
                ("path_length_point_m", "detour_ratio_point", "turn_count_point", "decision_count_point", "geometry_profile"),
            ),
            "instruction_selection": compact_selected_fields(
                gt.get("instruction_selection") or {},
                ("cohort_target_ids", "cohort_size", "winner_target_ids", "winner_count", "distance_rank", "selection_status"),
            ),
        },
    }


def build_b2_packet(row: dict[str, Any], visible_waypoints: list[dict[str, Any]]) -> dict[str, Any]:
    gt = get_gt(row)
    answer = answer_from_row(row)
    embodiment = gt.get("navigation_embodiment_facts") or {}
    routing = gt.get("routing_complexity") or {}
    return {
        "reasoning_type": REASONING_TYPES["b2"],
        "facts": {
            "visual_summary": visible_point_summary(visible_waypoints, embodied=True),
            "target_grounding": {
                "target_rule": gt.get("target_rule"),
                "target_category": gt.get("target_category") or gt.get("canonical_category"),
                "target_color_name": gt.get("target_color_name"),
                "goal_display_id": gt.get("goal_id"),
            },
            "robot_constraint": {
                "robot_diameter_m": round_float(gt.get("robot_diameter_m")),
                "min_clearance_m": round_float(
                    embodiment.get("embodied_clearance_margin_min_m") or routing.get("embodied_min_clearance_m")
                ),
                "narrow_passage_fraction": round_float(
                    embodiment.get("narrow_passage_fraction") or routing.get("embodied_narrow_passage_fraction")
                ),
            },
            "path_facts": {
                "start_id": gt.get("start_id"),
                "shortest_path": answer,
                "path_length_m": round_float(gt.get("canonical_sparse_length_m") or gt.get("optimal_length_m")),
                "route_semantics": gt.get("route_semantics") or "embodied",
                "path_waypoints": path_waypoint_facts(answer, visible_waypoints),
                "path_segments": path_segment_facts(answer, embodied=True, routing=routing),
                "path_projection_safe": routing.get("embodied_path_projection_safe")
                if routing.get("embodied_path_projection_safe") is not None
                else routing.get("path_projection_safe"),
            },
            "navigation_embodiment_facts": compact_selected_fields(
                embodiment,
                (
                    "path_length_embodied_m",
                    "detour_ratio_embodied",
                    "turn_count_embodied",
                    "decision_count_embodied",
                    "path_overlap_point_vs_embodied",
                    "embodied_extra_length_m",
                    "embodied_clearance_margin_min_m",
                    "narrow_passage_fraction",
                ),
            ),
            "routing_complexity": compact_selected_fields(
                routing,
                (
                    "embodied_num_segments_sparse",
                    "embodied_min_clearance_m",
                    "embodied_mean_clearance_m",
                    "embodied_narrow_passage_fraction",
                    "embodied_detour_ratio_sparse",
                ),
            ),
            "instruction_selection": compact_selected_fields(
                gt.get("instruction_selection") or {},
                ("cohort_target_ids", "cohort_size", "winner_target_ids", "winner_count", "distance_rank", "selection_status"),
            ),
        },
    }


def build_c_packet(row: dict[str, Any], visible_waypoints: list[dict[str, Any]]) -> dict[str, Any]:
    gt = get_gt(row)
    answer = answer_from_row(row)
    cue_bundle = gt.get("fixed_target_prompt_cue_bundle") or {}
    instruction = gt.get("instruction_selection") or {}
    candidate_target_ids = instruction.get("cohort_target_ids") or gt.get("supported_visible_same_type_ids") or []
    embodiment = gt.get("navigation_embodiment_facts") or {}
    routing = gt.get("routing_complexity") or {}
    return {
        "reasoning_type": REASONING_TYPES["c"],
        "facts": {
            "visual_summary": visible_point_summary(visible_waypoints, embodied=True),
            "intent_resolution": {
                "user_request": gt.get("generated_question") or row.get("question_text"),
                "intent_family": gt.get("intent_family"),
                "same_type_key": gt.get("same_type_key"),
                "resolved_target_category": gt.get("target_category") or gt.get("canonical_category"),
            },
            "cue_resolution": {
                "cue_family": cue_bundle.get("cue_family") or gt.get("cue_family"),
                "cue_type": cue_bundle.get("cue_type") or gt.get("cue_type"),
                "anchor_category": cue_bundle.get("anchor_category"),
                "cue_value": cue_bundle.get("cue_value") or gt.get("cue_value"),
                "residual_cue": cue_bundle.get("residual_cue_value") or gt.get("residual_cue_value"),
                "minimal_residual_cue": cue_bundle.get("minimal_residual_cue"),
            },
            "target_selection": {
                "candidate_target_ids": candidate_target_ids,
                "candidate_target_count": len(candidate_target_ids),
                "winner_target_id": gt.get("target_id"),
                "goal_display_id": gt.get("goal_id"),
                "selection_status": instruction.get("selection_status"),
            },
            "same_type_visibility": {
                "same_type_supported_visible_count": gt.get("same_type_supported_visible_count"),
                "supported_visible_same_type_ids": gt.get("supported_visible_same_type_ids") or [],
                "human_visible_same_type_object_ids": gt.get("human_visible_same_type_object_ids") or [],
            },
            "robot_constraint": {"robot_diameter_m": round_float(gt.get("robot_diameter_m"))},
            "path_facts": {
                "start_id": gt.get("start_id"),
                "shortest_path": answer,
                "path_length_m": round_float(gt.get("canonical_sparse_length_m") or gt.get("optimal_length_m")),
                "route_semantics": gt.get("route_semantics") or "embodied",
                "path_waypoints": path_waypoint_facts(answer, visible_waypoints),
                "path_segments": path_segment_facts(answer, embodied=True, routing=routing),
                "path_projection_safe": routing.get("embodied_path_projection_safe")
                if routing.get("embodied_path_projection_safe") is not None
                else routing.get("path_projection_safe"),
            },
            "navigation_embodiment_facts": compact_selected_fields(
                embodiment,
                (
                    "path_length_embodied_m",
                    "detour_ratio_embodied",
                    "turn_count_embodied",
                    "decision_count_embodied",
                    "path_overlap_point_vs_embodied",
                    "embodied_extra_length_m",
                    "embodied_clearance_margin_min_m",
                    "narrow_passage_fraction",
                ),
            ),
            "routing_complexity": compact_selected_fields(
                routing,
                (
                    "embodied_num_segments_sparse",
                    "embodied_min_clearance_m",
                    "embodied_mean_clearance_m",
                    "embodied_narrow_passage_fraction",
                    "embodied_detour_ratio_sparse",
                    "detour_ratio_sparse",
                    "sparse_turn_count",
                    "start_goal_l2_m",
                    "path_projection_safe",
                    "embodied_path_projection_safe",
                ),
            ),
        },
    }


def build_fact_packet(row: dict[str, Any]) -> dict[str, Any]:
    task = str(row.get("task") or "")
    if task not in TASKS:
        raise OracleCotError(f"unsupported task={task!r} question_id={row.get('question_id')}")
    gt = get_gt(row)
    answer = answer_from_row(row)
    visible_waypoints = load_visible_waypoints(row.get("visible_waypoints_path") or gt.get("visible_waypoints_path"))
    all_ids = candidate_display_ids(gt, visible_waypoints)
    base = {
        "question_id": str(row.get("question_id")),
        "task": task,
        "task_schema": REASONING_TYPES[task],
        "original_prompt": row.get("user_prompt") or row.get("prompt_text") or "",
        "system_prompt": row.get("system_prompt") or "",
        "image_path": row.get("image_path") or gt.get("image_path"),
        "candidate_display_ids": all_ids,
        "required_final_json": answer,
        "audit_refs": {
            "visible_waypoints_path": row.get("visible_waypoints_path") or gt.get("visible_waypoints_path"),
            "image_path": row.get("image_path") or gt.get("image_path"),
            "direct_pairs_ref": gt.get("direct_pairs_ref") or row.get("direct_pairs_ref"),
            "route_semantics": gt.get("route_semantics"),
        },
        "source_refs": {
            "split": "train",
            "scene_id": row.get("scene_id"),
            "view_id": row.get("view_id"),
            "routing_id": row.get("routing_id"),
        },
    }
    if task == "a1":
        task_payload = build_a1_packet(row, visible_waypoints)
    elif task == "b1":
        task_payload = build_b1_packet(row, visible_waypoints)
    elif task == "a2":
        task_payload = build_a2_packet(row, visible_waypoints)
    elif task == "b2":
        task_payload = build_b2_packet(row, visible_waypoints)
    else:
        task_payload = build_c_packet(row, visible_waypoints)
    base.update(task_payload)
    return base


def collect_rows(release_root: Path, split: str, tasks: tuple[str, ...], limit_per_task: int | None, seed: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        path = release_root / split / "vqa" / f"vqa_next_{task}.jsonl"
        if not path.exists():
            raise OracleCotError(f"missing VQA file: {path}")
        gt_path = release_root / split / "gt" / f"gt_next_{task}.jsonl"
        gt_by_qid = {
            str(row.get("question_id")): row
            for row in load_jsonl(gt_path)
            if row.get("question_id") is not None
        }
        task_rows = [
            merge_row_with_gt(row, gt_by_qid.get(str(row.get("question_id"))))
            for row in load_jsonl(path)
        ]
        sampled = stable_sample(task_rows, task=task, limit=limit_per_task, seed=seed)
        rows.extend(sampled)
    return rows


def shard_path(root: Path, kind: str, task: str, row_index: int, shard_size: int) -> Path:
    shard_id = row_index // shard_size
    return root / kind / task / f"shard_{shard_id:05d}.jsonl"


def write_fact_packets(output_dir: Path, packets: list[dict[str, Any]], shard_size: int) -> None:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    task_offsets: Counter[str] = Counter()
    for packet in packets:
        task = str(packet["task"])
        task_index = task_offsets[task]
        task_offsets[task] += 1
        grouped[(task, task_index // shard_size)].append(packet)
    for (task, shard_id), rows in grouped.items():
        write_jsonl(output_dir / "fact_packets" / task / f"shard_{shard_id:05d}.jsonl", rows)


def select_task_shard(packets: list[dict[str, Any]], *, shard_index: int, shard_count: int) -> list[dict[str, Any]]:
    if shard_count < 1:
        raise OracleCotError("--task-shard-count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise OracleCotError("--task-shard-index must satisfy 0 <= index < count")
    if shard_count == 1:
        return packets

    grouped_offsets: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    for packet in packets:
        task = str(packet["task"])
        task_index = grouped_offsets[task]
        grouped_offsets[task] += 1
        if task_index % shard_count == shard_index:
            selected_packet = dict(packet)
            selected_packet["_task_index_for_shard"] = task_index
            selected.append(selected_packet)
    return selected


def system_generation_prompt() -> str:
    return (
        "You write compact oracle-shortest-path rationales for NavBench3D training data. "
        "Use only the supplied fact packet. Do not solve a new path, do not change the required final JSON, "
        "and do not mention internal metadata, hashes, file paths, or world coordinates. "
        "Return exactly one JSON object."
    )


def packet_for_generation(packet: dict[str, Any]) -> dict[str, Any]:
    """Return the fact packet subset that is safe and useful for GPT rationale writing."""
    generation_packet = {
        key: value
        for key, value in packet.items()
        if key
        not in {
            "image_path",
            "system_prompt",
            "source_refs",
            "audit_refs",
            "_task_index_for_shard",
            "task_schema",
            "reasoning_type",
        }
    }
    generation_packet["facts"] = sanitize_facts_for_generation(
        str(packet.get("task") or ""), packet.get("facts") if isinstance(packet.get("facts"), dict) else {}
    )
    return generation_packet


def sanitize_waypoint_fact_for_generation(waypoint: Any) -> dict[str, Any]:
    if not isinstance(waypoint, dict):
        return {}
    allowed_keys = (
        "display_id",
        "pointmass_walkable",
        "embodied_feasible",
        "point_clearance_m",
        "embodied_clearance_margin_m",
        "visibility_fact",
    )
    return {key: waypoint[key] for key in allowed_keys if key in waypoint and waypoint[key] is not None}


def sanitize_path_facts_for_generation(path_facts: Any) -> dict[str, Any]:
    if not isinstance(path_facts, dict):
        return {}
    allowed_keys = ("start_id", "shortest_path", "path_length_m", "route_semantics")
    sanitized = {key: path_facts[key] for key in allowed_keys if key in path_facts and path_facts[key] is not None}
    waypoints = path_facts.get("path_waypoints")
    if isinstance(waypoints, list):
        sanitized["path_waypoints"] = [
            item for item in (sanitize_waypoint_fact_for_generation(waypoint) for waypoint in waypoints) if item
        ]
    segments = path_facts.get("path_segments")
    if isinstance(segments, list):
        sanitized["path_segments"] = [
            {
                key: segment[key]
                for key in (
                    "from_id",
                    "to_id",
                    "route_semantics",
                    "collision_free_projection",
                    "dense_collision_free_projection",
                    "sparse_segment_count",
                )
                if isinstance(segment, dict) and key in segment and segment[key] is not None
            }
            for segment in segments
            if isinstance(segment, dict)
        ]
    if path_facts.get("path_projection_safe") is not None:
        sanitized["path_projection_safe"] = bool(path_facts.get("path_projection_safe"))
    return sanitized


def target_grounding_for_generation(target_grounding: Any) -> dict[str, Any]:
    if not isinstance(target_grounding, dict):
        return {}
    allowed_keys = ("target_rule", "target_category", "goal_display_id")
    return {key: target_grounding[key] for key in allowed_keys if key in target_grounding and target_grounding[key] is not None}


def robot_constraint_for_generation(robot_constraint: Any) -> dict[str, Any]:
    if not isinstance(robot_constraint, dict):
        return {}
    allowed_keys = ("robot_diameter_m", "min_clearance_m", "narrow_passage_fraction")
    return {key: robot_constraint[key] for key in allowed_keys if key in robot_constraint and robot_constraint[key] is not None}


def sanitize_facts_for_generation(task: str, facts: dict[str, Any]) -> dict[str, Any]:
    if task == "a1":
        candidate_space = facts.get("candidate_space") if isinstance(facts.get("candidate_space"), dict) else {}
        sanitized_candidate_space = {
            key: candidate_space[key]
            for key in (
                "num_candidates",
                "all_display_ids",
                "pointmass_walkable_ids",
                "non_walkable_ids",
                "positive_examples",
                "negative_examples",
            )
            if key in candidate_space and candidate_space[key] is not None
        }
        return {
            key: value
            for key, value in {
                "candidate_space": sanitized_candidate_space,
                "selection_rule": facts.get("selection_rule"),
            }.items()
            if value is not None
        }
    if task == "b1":
        candidate_space = facts.get("candidate_space") if isinstance(facts.get("candidate_space"), dict) else {}
        sanitized_candidate_space = {
            key: candidate_space[key]
            for key in (
                "num_candidates",
                "all_display_ids",
                "embodied_feasible_ids",
                "not_embodied_feasible_ids",
                "positive_examples",
                "negative_examples",
            )
            if key in candidate_space and candidate_space[key] is not None
        }
        return {
            key: value
            for key, value in {
                "robot_constraint": facts.get("robot_constraint"),
                "candidate_space": sanitized_candidate_space,
                "selection_rule": facts.get("selection_rule"),
            }.items()
            if value is not None
        }
    if task == "a2":
        sanitized = {
            key: facts[key]
            for key in ("visual_summary", "target_grounding", "shortest_path_explanation", "navigation_geometry_facts")
            if key in facts and facts[key] is not None
        }
        sanitized["target_grounding"] = target_grounding_for_generation(facts.get("target_grounding"))
        sanitized["path_facts"] = sanitize_path_facts_for_generation(facts.get("path_facts"))
        sanitized["visual_summary"] = facts.get("visual_summary") or {}
        sanitized["navigation_geometry_facts"] = facts.get("navigation_geometry_facts") or {}
        return sanitized
    if task == "b2":
        sanitized = {
            key: facts[key]
            for key in (
                "visual_summary",
                "target_grounding",
                "robot_constraint",
                "embodiment_explanation",
                "navigation_embodiment_facts",
                "routing_complexity",
                "instruction_selection",
            )
            if key in facts and facts[key] is not None
        }
        sanitized["target_grounding"] = target_grounding_for_generation(facts.get("target_grounding"))
        sanitized["robot_constraint"] = robot_constraint_for_generation(facts.get("robot_constraint"))
        sanitized["path_facts"] = sanitize_path_facts_for_generation(facts.get("path_facts"))
        sanitized["visual_summary"] = facts.get("visual_summary") or {}
        sanitized["navigation_embodiment_facts"] = facts.get("navigation_embodiment_facts") or {}
        sanitized["routing_complexity"] = facts.get("routing_complexity") or {}
        return sanitized
    if task == "c":
        sanitized = {
            key: facts[key]
            for key in (
                "visual_summary",
                "intent_resolution",
                "cue_resolution",
                "target_selection",
                "robot_constraint",
                "navigation_embodiment_facts",
                "routing_complexity",
            )
            if key in facts and facts[key] is not None
        }
        sanitized["robot_constraint"] = robot_constraint_for_generation(facts.get("robot_constraint"))
        sanitized["same_type_visibility"] = facts.get("same_type_visibility") or {}
        sanitized["path_facts"] = sanitize_path_facts_for_generation(facts.get("path_facts"))
        sanitized["visual_summary"] = facts.get("visual_summary") or {}
        sanitized["navigation_embodiment_facts"] = facts.get("navigation_embodiment_facts") or {}
        sanitized["routing_complexity"] = facts.get("routing_complexity") or {}
        return sanitized
    return {}


def compact_examples(examples: Any) -> list[dict[str, Any]]:
    if not isinstance(examples, list):
        return []
    out: list[dict[str, Any]] = []
    for example in examples[:3]:
        if isinstance(example, dict):
            item: dict[str, Any] = {}
            if example.get("id") is not None:
                item["id"] = example.get("id")
            if example.get("reason") is not None:
                item["reason"] = example.get("reason")
            if item:
                out.append(item)
    return out


def path_facts_template(facts: dict[str, Any], *, include_length: bool = True) -> dict[str, Any]:
    path_facts = facts.get("path_facts") if isinstance(facts.get("path_facts"), dict) else {}
    template = {
        "start_id": path_facts.get("start_id"),
        "shortest_path": path_facts.get("shortest_path"),
        "route_semantics": path_facts.get("route_semantics"),
    }
    if include_length:
        template["path_length_m"] = path_facts.get("path_length_m")
    return template


def reasoning_payload_template(packet: dict[str, Any]) -> dict[str, Any]:
    task = str(packet["task"])
    facts = packet.get("facts") if isinstance(packet.get("facts"), dict) else {}
    if task in {"a1", "b1"}:
        candidate_space = facts.get("candidate_space") if isinstance(facts.get("candidate_space"), dict) else {}
        payload = {
            "candidate_space": {
                "num_candidates": candidate_space.get("num_candidates"),
                "positive_examples": compact_examples(candidate_space.get("positive_examples")),
                "negative_examples": compact_examples(candidate_space.get("negative_examples")),
            },
            "selection_rule": facts.get("selection_rule"),
        }
        if task == "b1":
            payload = {
                "robot_constraint": facts.get("robot_constraint") or {},
                **payload,
            }
        return payload
    if task == "a2":
        return {
            "visual_summary": facts.get("visual_summary") or {},
            "target_grounding": facts.get("target_grounding") or {},
            "path_facts": path_facts_template(facts),
            "shortest_path_explanation": facts.get("shortest_path_explanation"),
        }
    if task == "b2":
        return {
            "visual_summary": facts.get("visual_summary") or {},
            "target_grounding": facts.get("target_grounding") or {},
            "robot_constraint": facts.get("robot_constraint") or {},
            "path_facts": path_facts_template(facts),
            "embodiment_explanation": facts.get("embodiment_explanation"),
        }
    return {
        "visual_summary": facts.get("visual_summary") or {},
        "intent_resolution": facts.get("intent_resolution") or {},
        "cue_resolution": facts.get("cue_resolution") or {},
        "target_selection": facts.get("target_selection") or {},
        "same_type_visibility": facts.get("same_type_visibility") or {},
        "navigation_embodiment_facts": facts.get("navigation_embodiment_facts") or {},
        "routing_complexity": facts.get("routing_complexity") or {},
        "path_facts": path_facts_template(facts),
    }


def required_output_schema(packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": packet["question_id"],
        "task": packet["task"],
        "reasoning_type": packet["reasoning_type"],
        "reasoning_payload": reasoning_payload_template(packet),
        "final_json": packet["required_final_json"],
    }


def generation_constraints(*, include_top_level_constraint: bool = True) -> list[str]:
    constraints = []
    if include_top_level_constraint:
        constraints.append("Return the exact top-level keys shown in required_output_envelope.")
    constraints.extend([
        "Return the same task-specific reasoning_payload object shape shown in required_output_envelope.",
        "Do not replace reasoning_payload with a free-form rationale, note, explanation, or summary field.",
        "You may only shorten natural-language reason/explanation strings; preserve numeric facts and ID lists.",
        "final_json must exactly equal required_final_json from the fact packet.",
        "Do not invent or alter the shortest_path.",
        "Do not output the phrases 'ground truth says' or 'metadata says'.",
        "Do not output hashes, file paths, internal references, world coordinates, or world_xyz.",
        "Every display ID mentioned in start_id, goal_display_id, shortest_path, or final_json must be in candidate_display_ids.",
        "Keep the rationale concise and task-specific.",
    ])
    return constraints


def user_generation_prompt(packet: dict[str, Any]) -> str:
    schema = required_output_schema(packet)
    payload = {
        "required_output_envelope": schema,
        "hard_constraints": generation_constraints(),
        "fact_packet": packet_for_generation(packet),
    }
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)


def batch_generation_prompt(packets: list[dict[str, Any]]) -> str:
    items = [
        {
            "required_output_envelope": required_output_schema(packet),
            "fact_packet": packet_for_generation(packet),
        }
        for packet in packets
    ]
    payload = {
        "required_output": {
            "records": [required_output_schema(packet) for packet in packets],
        },
        "hard_constraints": [
            "Return exactly one JSON object with a top-level records array.",
            "The records array must contain exactly one object per input item.",
            "Each record must match the corresponding required_output_envelope question_id and task.",
            *generation_constraints(include_top_level_constraint=False),
        ],
        "items": items,
    }
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)


def resolve_api_base(api_base_arg: str | None, api_base_env: str) -> str:
    if api_base_arg:
        return api_base_arg.rstrip("/")
    for env_name in (api_base_env, "OPENAI_BASE_URL", "OPENAI_API_BASE"):
        value = os.getenv(env_name)
        if value:
            return value.rstrip("/")
    raise OracleCotError(f"missing API base; pass --api-base or set {api_base_env}/OPENAI_BASE_URL")


def resolve_api_key(api_key_env: str) -> str:
    value = os.getenv(api_key_env)
    if not value:
        raise OracleCotError(f"missing API key env var: {api_key_env}")
    return value


def require_external_upload_ack(args: argparse.Namespace) -> None:
    if not args.generate:
        return
    if args.allow_external_upload:
        return
    api_base = resolve_api_base(args.api_base, args.api_base_env)
    raise OracleCotError(
        "refusing to send oracle-CoT fact packets to an external API without explicit acknowledgement. "
        f"api_base={api_base!r}; rerun with --allow-external-upload only after user approval for sending "
        "sanitized question text, oracle final_json/path facts, candidate IDs, clearance, and intent/cue facts."
    )


def require_safe_sft_export_target(args: argparse.Namespace, data_dir: Path) -> None:
    if not args.generate:
        return
    if args.stage == "full":
        return
    if args.allow_partial_current_sft_export:
        return
    if data_dir.resolve() != DEFAULT_LF_DATA_DIR.resolve():
        return
    raise OracleCotError(
        "refusing to export partial pilot/validation SFT data into the current full training data directory. "
        f"stage={args.stage!r}; data_dir={str(data_dir)!r}. Use a stage-specific --llamafactory-data-dir "
        "or pass --allow-partial-current-sft-export if this overwrite is intentional."
    )


def chat_completion_request(
    *,
    api_base: str,
    api_key: str,
    model: str,
    packet: dict[str, Any],
    max_completion_tokens: int,
    timeout_s: float,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    return chat_completion_request_for_prompt(
        api_base=api_base,
        api_key=api_key,
        model=model,
        prompt=user_generation_prompt(packet),
        max_completion_tokens=max_completion_tokens,
        timeout_s=timeout_s,
        temperature=temperature,
        top_p=top_p,
    )


def chat_completion_request_for_prompt(
    *,
    api_base: str,
    api_key: str,
    model: str,
    prompt: str,
    max_completion_tokens: int,
    timeout_s: float,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_generation_prompt()},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_completion_tokens": max_completion_tokens,
        "response_format": {"type": "json_object"},
        "store": False,
    }
    data = json.dumps(body, ensure_ascii=True).encode("utf-8")
    request = urllib.request.Request(
        f"{api_base}/chat/completions",
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw)


def extract_response_text(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise OracleCotError("empty chat completion choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts).strip()
    raise OracleCotError("cannot extract assistant text from chat completion")


def strip_code_fence(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_code_fence(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end < start:
            raise
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise OracleCotError("assistant output is not a JSON object")
    return parsed


def parse_batch_records(text: str, expected_qids: list[str]) -> dict[str, dict[str, Any]]:
    parsed = parse_json_object(text)
    records = parsed.get("records")
    if not isinstance(records, list):
        raise OracleCotError("batch assistant output missing records array")
    if len(records) != len(expected_qids):
        raise OracleCotError(f"batch records length mismatch: expected={len(expected_qids)} got={len(records)}")
    by_qid: dict[str, dict[str, Any]] = {}
    for item in records:
        if not isinstance(item, dict):
            raise OracleCotError("batch record is not a JSON object")
        qid = item.get("question_id")
        if qid is None:
            raise OracleCotError("batch record missing question_id")
        by_qid[str(qid)] = item
    missing = [qid for qid in expected_qids if qid not in by_qid]
    if missing:
        raise OracleCotError(f"batch output missing question_ids: {missing[:3]}")
    return by_qid


def collect_display_id_values(value: Any, *, path: tuple[str, ...] = ()) -> list[tuple[tuple[str, ...], int]]:
    found: list[tuple[tuple[str, ...], int]] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = (*path, str(key))
            lowered = str(key).lower()
            if lowered in {"start_id", "goal_display_id", "display_id"} or lowered.endswith("_display_id"):
                try:
                    found.append((child_path, int(child)))
                except (TypeError, ValueError):
                    pass
            elif lowered == "id" and any(part in {"positive_examples", "negative_examples"} for part in path):
                try:
                    found.append((child_path, int(child)))
                except (TypeError, ValueError):
                    pass
            elif lowered in {"shortest_path", "final_json"} and isinstance(child, list):
                for item in child:
                    try:
                        found.append((child_path, int(item)))
                    except (TypeError, ValueError):
                        pass
            else:
                found.extend(collect_display_id_values(child, path=child_path))
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            found.extend(collect_display_id_values(child, path=(*path, str(idx))))
    return found


def require_object(payload: dict[str, Any], key: str, issues: list[str], *, parent: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        issues.append(f"{parent}.{key}_missing_or_not_object")
        return {}
    return value


def missing_keys(payload: dict[str, Any], required: tuple[str, ...], *, prefix: str) -> list[str]:
    missing = []
    for key in required:
        if key not in payload or payload.get(key) is None:
            missing.append(f"{prefix}.{key}")
    return missing


def validate_task_schema(parsed: dict[str, Any], packet: dict[str, Any], final_json: list[int]) -> list[str]:
    task = str(packet["task"])
    issues: list[str] = []
    payload = parsed.get("reasoning_payload")
    if not isinstance(payload, dict):
        return ["reasoning_payload_not_object"]

    schema = SCHEMA_REQUIRED_KEYS[task]
    for missing in missing_keys(payload, schema["reasoning_payload"], prefix="reasoning_payload"):
        issues.append(f"schema_missing:{missing}")

    if task in {"a1", "b1"}:
        candidate_space = require_object(payload, "candidate_space", issues, parent="reasoning_payload")
        for missing in missing_keys(candidate_space, schema["candidate_space"], prefix="reasoning_payload.candidate_space"):
            issues.append(f"schema_missing:{missing}")
        expected_count = ((packet.get("facts") or {}).get("candidate_space") or {}).get("num_candidates")
        if expected_count is not None and candidate_space.get("num_candidates") is not None:
            try:
                if int(candidate_space["num_candidates"]) != int(expected_count):
                    issues.append("candidate_space_num_candidates_mismatch")
            except (TypeError, ValueError):
                issues.append("candidate_space_num_candidates_not_int")
        for field in ("positive_examples", "negative_examples"):
            examples = candidate_space.get(field)
            if not isinstance(examples, list):
                issues.append(f"schema_invalid:reasoning_payload.candidate_space.{field}_not_list")
        return issues

    path_facts = require_object(payload, "path_facts", issues, parent="reasoning_payload")
    target_grounding: dict[str, Any] = {}
    if task in {"a2", "b2"}:
        target_grounding = require_object(payload, "target_grounding", issues, parent="reasoning_payload")
        for missing in missing_keys(target_grounding, schema.get("target_grounding", ()), prefix="reasoning_payload.target_grounding"):
            issues.append(f"schema_missing:{missing}")
    for missing in missing_keys(path_facts, schema["path_facts"], prefix="reasoning_payload.path_facts"):
        issues.append(f"schema_missing:{missing}")

    if task in {"b2"}:
        robot_constraint = require_object(payload, "robot_constraint", issues, parent="reasoning_payload")
        for missing in missing_keys(robot_constraint, schema.get("robot_constraint", ()), prefix="reasoning_payload.robot_constraint"):
            issues.append(f"schema_missing:{missing}")
    if task == "c":
        intent_resolution = require_object(payload, "intent_resolution", issues, parent="reasoning_payload")
        cue_resolution = require_object(payload, "cue_resolution", issues, parent="reasoning_payload")
        target_selection = require_object(payload, "target_selection", issues, parent="reasoning_payload")
        for missing in missing_keys(intent_resolution, schema["intent_resolution"], prefix="reasoning_payload.intent_resolution"):
            issues.append(f"schema_missing:{missing}")
        for missing in missing_keys(cue_resolution, schema["cue_resolution"], prefix="reasoning_payload.cue_resolution"):
            issues.append(f"schema_missing:{missing}")
        for missing in missing_keys(target_selection, schema["target_selection"], prefix="reasoning_payload.target_selection"):
            issues.append(f"schema_missing:{missing}")

    try:
        payload_path = int_list(path_facts.get("shortest_path"), field="reasoning_payload.path_facts.shortest_path")
        if payload_path != final_json:
            issues.append("reasoning_payload_shortest_path_mismatch")
    except OracleCotError:
        issues.append("reasoning_payload_shortest_path_not_int_list")

    packet_path_facts = ((packet.get("facts") or {}).get("path_facts") or {})
    expected_start = packet_path_facts.get("start_id")
    if expected_start is not None and path_facts.get("start_id") is not None:
        try:
            if int(path_facts["start_id"]) != int(expected_start):
                issues.append("reasoning_payload_start_id_mismatch")
        except (TypeError, ValueError):
            issues.append("reasoning_payload_start_id_not_int")

    if task == "c":
        output_goal_holder = payload.get("target_selection") if isinstance(payload.get("target_selection"), dict) else {}
        packet_goal_holder = ((packet.get("facts") or {}).get("target_selection") or {})
    else:
        output_goal_holder = target_grounding
        packet_goal_holder = ((packet.get("facts") or {}).get("target_grounding") or {})
    expected_goal = packet_goal_holder.get("goal_display_id")
    if expected_goal is not None and output_goal_holder.get("goal_display_id") is not None:
        try:
            if int(output_goal_holder["goal_display_id"]) != int(expected_goal):
                issues.append("reasoning_payload_goal_display_id_mismatch")
        except (TypeError, ValueError):
            issues.append("reasoning_payload_goal_display_id_not_int")

    return issues


def strings_in_payload(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings_in_payload(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings_in_payload(child)


def audit_path_edge_legality(packet: dict[str, Any], final_json: list[int]) -> dict[str, Any]:
    task = str(packet.get("task") or "")
    if task not in {"a2", "b2", "c"}:
        return {"status": "not_applicable", "ok": None}
    if not final_json:
        return {"status": "not_checked_empty_path", "ok": None}

    refs = packet.get("audit_refs") if isinstance(packet.get("audit_refs"), dict) else {}
    visible_ref = refs.get("visible_waypoints_path")
    if not visible_ref:
        return {"status": "not_checked_missing_visible_waypoints_path", "ok": None}

    try:
        from evaluate_next import (  # type: ignore
            load_direct_pairs,
            load_visible_payload,
            resolve_optional_path,
        )

        visible_path = Path(str(visible_ref))
        if not visible_path.exists():
            return {"status": "not_checked_missing_visible_waypoints_file", "ok": None}
        visible_payload = load_visible_payload(visible_path)
        visible_waypoints = list(visible_payload.get("visible_waypoints", []))
        all_ids = {
            int(waypoint["display_id"])
            for waypoint in visible_waypoints
            if waypoint.get("display_id") is not None
        }
        if any(display_id not in all_ids for display_id in final_json):
            return {"status": "checked_failed_invalid_display_id", "ok": False}

        expected_start = ((packet.get("facts") or {}).get("path_facts") or {}).get("start_id")
        if expected_start is None or len(final_json) < 2 or int(final_json[0]) != int(expected_start):
            return {"status": "checked_failed_start_not_anchored", "ok": False}

        image_ref = refs.get("image_path") or packet.get("image_path")
        image_path = Path(str(image_ref)) if image_ref else None
        anchors = [visible_path.parent]
        if image_path is not None:
            anchors.append(image_path.parent)
        anchors.append(Path.cwd())

        direct_pairs_ref = refs.get("direct_pairs_ref")
        direct_pairs_path = resolve_optional_path(str(direct_pairs_ref), anchors) if direct_pairs_ref else None
        if direct_pairs_ref is not None and direct_pairs_path is None:
            return {"status": "not_checked_unresolved_direct_pairs_ref", "ok": None}

        if direct_pairs_path is not None:
            adj = load_direct_pairs(direct_pairs_path, visible_waypoints)
            waypoint_sequence = [int(display_id) for display_id in final_json]
            routing_mode = "direct_pairs"
        else:
            # Train answers are canonical sparse visible-waypoint paths. Adjacent
            # display IDs can be connected by hidden graph waypoints, so graph
            # fallback direct-edge checks create false negatives without a
            # direct_pairs_ref expansion.
            return {
                "status": "checked_ok_oracle_sparse_path",
                "ok": True,
                "routing_mode": "oracle_sparse",
                "num_edges": max(len(final_json) - 1, 0),
            }

        path_length_m = 0.0
        for source, target in zip(waypoint_sequence[:-1], waypoint_sequence[1:]):
            if source not in adj or target not in adj[source]:
                return {
                    "status": "checked_failed_missing_edge",
                    "ok": False,
                    "routing_mode": routing_mode,
                    "edge": [source, target],
                }
            path_length_m += float(adj[source][target])

        return {
            "status": "checked_ok",
            "ok": True,
            "routing_mode": routing_mode,
            "num_edges": max(len(final_json) - 1, 0),
            "path_length_m": round(path_length_m, 3),
        }
    except Exception as exc:  # pragma: no cover - defensive for external data drift
        return {"status": "not_checked_error", "ok": None, "error": str(exc)[:240]}


def validate_generated(parsed: dict[str, Any], packet: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    qid = packet["question_id"]
    task = packet["task"]
    allowed = {int(item) for item in packet.get("candidate_display_ids") or []}
    required_answer = [int(item) for item in packet["required_final_json"]]

    if parsed.get("question_id") != qid:
        issues.append("question_id_mismatch")
    if parsed.get("task") != task:
        issues.append("task_mismatch")
    if parsed.get("reasoning_type") != REASONING_TYPES[task]:
        issues.append("reasoning_type_mismatch")
    if not isinstance(parsed.get("reasoning_payload"), dict):
        issues.append("reasoning_payload_not_object")

    try:
        final_json = int_list(parsed.get("final_json"), field="final_json")
    except OracleCotError:
        final_json = []
        issues.append("final_json_not_int_list")
    if final_json != required_answer:
        issues.append("final_json_mismatch")

    issues.extend(validate_task_schema(parsed, packet, final_json))

    if allowed:
        for display_path, display_id in collect_display_id_values(parsed):
            if display_id not in allowed:
                issues.append(f"display_id_outside_space:{'.'.join(display_path)}={display_id}")

    lowered_strings = "\n".join(strings_in_payload(parsed)).lower()
    for pattern in BANNED_OUTPUT_PATTERNS:
        if pattern in lowered_strings:
            issues.append(f"banned_leakage:{pattern}")

    if task in {"a2", "b2", "c"} and final_json:
        facts = packet.get("facts") or {}
        path_facts = facts.get("path_facts") or {}
        start_id = path_facts.get("start_id")
        goal_id = path_facts.get("shortest_path", [None])[-1] if path_facts.get("shortest_path") else None
        if start_id is not None and final_json[0] != int(start_id):
            issues.append("path_start_mismatch")
        if goal_id is not None and final_json[-1] != int(goal_id):
            issues.append("path_goal_mismatch")
        expected_route_semantics = path_facts.get("route_semantics")
        payload_path_facts = (
            parsed.get("reasoning_payload", {}).get("path_facts", {})
            if isinstance(parsed.get("reasoning_payload"), dict)
            else {}
        )
        if (
            expected_route_semantics is not None
            and isinstance(payload_path_facts, dict)
            and payload_path_facts.get("route_semantics") is not None
            and str(payload_path_facts.get("route_semantics")) != str(expected_route_semantics)
        ):
            issues.append("reasoning_payload_route_semantics_mismatch")
        expected_length = path_facts.get("path_length_m")
        actual_length = payload_path_facts.get("path_length_m") if isinstance(payload_path_facts, dict) else None
        if expected_length is not None and actual_length is not None:
            try:
                if abs(float(actual_length) - float(expected_length)) > 0.01:
                    issues.append("reasoning_payload_path_length_mismatch")
            except (TypeError, ValueError):
                issues.append("reasoning_payload_path_length_not_float")

    edge_audit = audit_path_edge_legality(packet, final_json)
    if edge_audit.get("ok") is False:
        issues.append(f"path_edge_legality:{edge_audit.get('status')}")

    return {
        "ok": not issues,
        "issues": issues,
        "path_edge_legality": edge_audit,
    }


def existing_status(output_dir: Path) -> tuple[set[str], set[str]]:
    accepted: set[str] = set()
    rejected: set[str] = set()
    for path in (output_dir / "parsed").glob("*/*.jsonl"):
        for row in iter_jsonl(path):
            if row.get("question_id"):
                accepted.add(str(row["question_id"]))
    for path in (output_dir / "rejected").glob("*/*.jsonl"):
        for row in iter_jsonl(path):
            if row.get("question_id"):
                rejected.add(str(row["question_id"]))
    return accepted, rejected


def progress_message(
    *,
    shard_label: str,
    processed: int,
    accepted_written: int,
    rejected_written: int,
    deferred_api_errors: int,
    skipped_accepted: int,
    skipped_rejected: int,
) -> str:
    return (
        "generation_progress "
        f"shard={shard_label} processed={processed} accepted_written={accepted_written} "
        f"rejected_written={rejected_written} deferred_api_errors={deferred_api_errors} "
        f"skipped_accepted={skipped_accepted} "
        f"skipped_rejected={skipped_rejected}"
    )


def raise_if_deferred_api_errors(args: argparse.Namespace, *, deferred_api_errors: int, shard_label: str) -> None:
    if deferred_api_errors <= 0 or not should_defer_transient_api_error(args):
        return
    raise OracleCotError(
        "generation deferred transient API errors; "
        f"shard={shard_label} deferred_api_errors={deferred_api_errors}"
    )


def write_generated_record(
    *,
    output_dir: Path,
    args: argparse.Namespace,
    item: dict[str, Any],
    api_base: str,
    raw_record: dict[str, Any] | None,
    parsed: dict[str, Any] | None,
    error_message: str | None,
) -> bool | None:
    packet = item["packet"]
    qid = str(item["qid"])
    task = str(item["task"])
    task_index = int(item["task_index"])
    raw_path = shard_path(output_dir, "raw_responses", task, task_index, args.shard_size)
    parsed_path = shard_path(output_dir, "parsed", task, task_index, args.shard_size)
    rejected_path = shard_path(output_dir, "rejected", task, task_index, args.shard_size)
    if raw_record is not None:
        append_jsonl(raw_path, raw_record)

    if parsed is None:
        if args.fail_fast_on_api_error:
            if should_defer_transient_api_error(args):
                log_progress(
                    "generation_transient_api_defer "
                    f"qid={qid} task={task} error={error_message}"
                )
                return None
            cooldown_s = final_api_error_cooldown_seconds(args)
            if cooldown_s > 0:
                log_progress(
                    "generation_fail_fast_cooldown "
                    f"qid={qid} task={task} sleep_s={round(cooldown_s, 3)} error={error_message}"
                )
                time.sleep(cooldown_s)
            raise OracleCotError(f"generation failed for question_id={qid}: {error_message}")
        append_jsonl(
            rejected_path,
            {
                "question_id": qid,
                "task": task,
                "ok": False,
                "issues": ["api_or_parse_error"],
                "error": error_message,
            },
        )
        return False

    qc = validate_generated(parsed, packet)
    record = {
        "question_id": qid,
        "task": task,
        "parsed": parsed,
        "qc": qc,
    }
    if qc["ok"]:
        append_jsonl(parsed_path, record)
        return True

    append_jsonl(rejected_path, record)
    return False


def request_single_generated(
    *,
    args: argparse.Namespace,
    api_base: str,
    api_key: str,
    item: dict[str, Any],
    shard_label: str,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str | None]:
    packet = item["packet"]
    qid = str(item["qid"])
    task = str(item["task"])
    error_message: str | None = None
    max_retries = request_max_retries(args)
    for attempt in range(max_retries + 1):
        try:
            with full_generation_inflight_slot(args, shard_label=shard_label):
                wait_for_full_generation_request_slot(args, shard_label=shard_label)
                response = chat_completion_request(
                    api_base=api_base,
                    api_key=api_key,
                    model=args.model,
                    packet=packet,
                    max_completion_tokens=args.max_completion_tokens,
                    timeout_s=args.timeout_s,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            text = extract_response_text(response)
            raw_record = {
                "question_id": qid,
                "task": task,
                "model": args.model,
                "api_base": api_base,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "response_text": text,
                "response_usage": response.get("usage"),
                "response_id": response.get("id"),
                "request_mode": "single",
            }
            return raw_record, parse_json_object(text), None
        except RETRYABLE_REQUEST_ERRORS as exc:
            error_message = str(exc)
            if attempt >= max_retries:
                break
            sleep_s = retry_sleep_seconds(args, attempt)
            log_progress(
                "generation_retry "
                f"shard={shard_label} qid={qid} task={task} attempt={attempt + 1} "
                f"sleep_s={round(sleep_s, 3)} error={error_message}"
            )
            time.sleep(sleep_s)
    return None, None, error_message


def request_batch_generated(
    *,
    args: argparse.Namespace,
    api_base: str,
    api_key: str,
    items: list[dict[str, Any]],
    shard_label: str,
) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]] | None, str | None]:
    packets = [item["packet"] for item in items]
    qids = [str(item["qid"]) for item in items]
    task_counts = dict(sorted(Counter(str(item["task"]) for item in items).items()))
    error_message: str | None = None
    max_tokens = args.batch_max_completion_tokens or (args.max_completion_tokens * len(items))
    timeout_s = max(args.timeout_s, args.timeout_s * len(items))
    max_retries = request_max_retries(args)
    for attempt in range(max_retries + 1):
        try:
            with full_generation_inflight_slot(args, shard_label=shard_label):
                wait_for_full_generation_request_slot(args, shard_label=shard_label)
                response = chat_completion_request_for_prompt(
                    api_base=api_base,
                    api_key=api_key,
                    model=args.model,
                    prompt=batch_generation_prompt(packets),
                    max_completion_tokens=max_tokens,
                    timeout_s=timeout_s,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
            text = extract_response_text(response)
            raw_record = {
                "model": args.model,
                "api_base": api_base,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "response_text": text,
                "response_usage": response.get("usage"),
                "response_id": response.get("id"),
                "request_mode": "batch",
                "batch_size": len(items),
                "batch_question_ids": qids,
                "batch_task_counts": task_counts,
            }
            return raw_record, parse_batch_records(text, qids), None
        except RETRYABLE_REQUEST_ERRORS as exc:
            error_message = str(exc)
            if attempt >= max_retries:
                break
            sleep_s = retry_sleep_seconds(args, attempt)
            log_progress(
                "generation_batch_retry "
                f"shard={shard_label} batch_size={len(items)} attempt={attempt + 1} "
                f"sleep_s={round(sleep_s, 3)} error={error_message}"
            )
            time.sleep(sleep_s)
    return None, None, error_message


def chunk_items(items: list[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    if size <= 1:
        for item in items:
            yield [item]
        return
    for index in range(0, len(items), size):
        yield items[index : index + size]


def generate_for_packets(args: argparse.Namespace, output_dir: Path, packets: list[dict[str, Any]]) -> None:
    api_base = resolve_api_base(args.api_base, args.api_base_env)
    api_key = resolve_api_key(args.api_key_env)
    accepted_qids, rejected_qids = existing_status(output_dir) if args.resume else (set(), set())
    processed = 0
    skipped_accepted = 0
    skipped_rejected = 0
    accepted_written = 0
    rejected_written = 0
    deferred_api_errors = 0
    shard_label = f"{args.task_shard_index}/{args.task_shard_count}"

    log_progress(
        "generation_start "
        f"shard={shard_label} selected_packets={len(packets)} "
        f"existing_accepted={len(accepted_qids)} existing_rejected={len(rejected_qids)}"
    )

    task_offsets: Counter[str] = Counter()
    work_items: list[dict[str, Any]] = []
    for packet in packets:
        qid = packet["question_id"]
        task = packet["task"]
        if packet.get("_task_index_for_shard") is not None:
            task_index = int(packet["_task_index_for_shard"])
        else:
            task_index = task_offsets[task]
            task_offsets[task] += 1
        if qid in accepted_qids:
            skipped_accepted += 1
            continue
        if qid in rejected_qids and not args.retry_rejected:
            skipped_rejected += 1
            continue
        work_items.append({"packet": packet, "qid": qid, "task": task, "task_index": task_index})

    for batch in chunk_items(work_items, args.batch_size):
        batch_raw: dict[str, Any] | None = None
        parsed_by_qid: dict[str, dict[str, Any]] | None = None
        batch_error: str | None = None
        if len(batch) > 1:
            batch_raw, parsed_by_qid, batch_error = request_batch_generated(
                args=args,
                api_base=api_base,
                api_key=api_key,
                items=batch,
                shard_label=shard_label,
            )
            if parsed_by_qid is None:
                log_progress(
                    "generation_batch_fallback "
                    f"shard={shard_label} batch_size={len(batch)} error={batch_error}"
                )

        for item in batch:
            qid = str(item["qid"])
            if parsed_by_qid is not None and batch_raw is not None:
                raw_record = {**batch_raw, "question_id": qid, "task": str(item["task"])}
                parsed = parsed_by_qid.get(qid)
                error_message = None
            else:
                raw_record, parsed, error_message = request_single_generated(
                    args=args,
                    api_base=api_base,
                    api_key=api_key,
                    item=item,
                    shard_label=shard_label,
                )

            ok = write_generated_record(
                output_dir=output_dir,
                args=args,
                item=item,
                api_base=api_base,
                raw_record=raw_record,
                parsed=parsed,
                error_message=error_message,
            )
            if ok is True:
                accepted_written += 1
            elif ok is False:
                rejected_written += 1
            else:
                deferred_api_errors += 1
            processed += 1

            if args.progress_every > 0 and processed % args.progress_every == 0:
                log_progress(
                    progress_message(
                        shard_label=shard_label,
                        processed=processed,
                        accepted_written=accepted_written,
                        rejected_written=rejected_written,
                        deferred_api_errors=deferred_api_errors,
                        skipped_accepted=skipped_accepted,
                        skipped_rejected=skipped_rejected,
                    )
                )
        if args.request_sleep_s > 0:
            time.sleep(args.request_sleep_s)

    log_progress(
        "generation_done "
        f"shard={shard_label} processed={processed} accepted_written={accepted_written} "
        f"rejected_written={rejected_written} deferred_api_errors={deferred_api_errors} "
        f"skipped_accepted={skipped_accepted} "
        f"skipped_rejected={skipped_rejected}"
    )
    raise_if_deferred_api_errors(
        args,
        deferred_api_errors=deferred_api_errors,
        shard_label=shard_label,
    )


def load_generated_by_qid(output_dir: Path) -> dict[str, dict[str, Any]]:
    by_qid: dict[str, dict[str, Any]] = {}
    for path in sorted((output_dir / "parsed").glob("*/*.jsonl")):
        for row in iter_jsonl(path):
            if row.get("question_id") and row.get("parsed"):
                by_qid[str(row["question_id"])] = row
    return by_qid


def build_user_message(packet: dict[str, Any]) -> str:
    prompt = str(packet.get("original_prompt") or "").strip()
    return f"<image>{prompt}"


def build_direct_sft_row(packet: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": build_user_message(packet)},
            {"role": "assistant", "content": json.dumps(packet["required_final_json"], ensure_ascii=True)},
        ],
        "images": [packet["image_path"]],
        "metadata": {
            "question_id": packet["question_id"],
            "task": packet["task"],
            "supervision": "direct_sft",
        },
    }


def group_rows_by_task(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {task: [] for task in TASKS}
    for row in rows:
        task = str((row.get("metadata") or {}).get("task") or row.get("task") or "")
        if task in grouped:
            grouped[task].append(row)
    return grouped


def stratified_limit(rows: list[dict[str, Any]], *, total_limit: int) -> list[dict[str, Any]]:
    if total_limit <= 0 or len(rows) <= total_limit:
        return rows
    grouped = group_rows_by_task(rows)
    nonempty_tasks = [task for task in TASKS if grouped[task]]
    if not nonempty_tasks:
        return rows[:total_limit]
    base = total_limit // len(nonempty_tasks)
    remainder = total_limit % len(nonempty_tasks)
    selected: list[dict[str, Any]] = []
    leftovers: list[dict[str, Any]] = []
    for idx, task in enumerate(nonempty_tasks):
        task_rows = grouped[task]
        quota = base + (1 if idx < remainder else 0)
        selected.extend(task_rows[:quota])
        leftovers.extend(task_rows[quota:])
    if len(selected) < total_limit:
        selected.extend(leftovers[: total_limit - len(selected)])
    return selected[:total_limit]


def stratified_fraction(rows: list[dict[str, Any]], *, fraction: float) -> list[dict[str, Any]]:
    grouped = group_rows_by_task(rows)
    selected: list[dict[str, Any]] = []
    for task in TASKS:
        task_rows = grouped[task]
        if not task_rows:
            continue
        quota = max(1, int(len(task_rows) * fraction))
        selected.extend(task_rows[:quota])
    return selected


def compact_value(value: Any, default: str = "unknown") -> str:
    if value is None or value == "":
        return default
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def format_distance_m(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return compact_value(value)
    amount = float(value)
    if 0 < abs(amount) < 1:
        cm = round(amount * 100, 1)
        return f"about {cm:g} cm"
    return f"about {round(amount, 2):g} m"


def format_fraction_as_percent(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return compact_value(value)
    return f"{round(float(value) * 100):g}%"


def format_turn_count(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return compact_value(value)
    count = int(round(float(value)))
    if count == 0:
        return "no turns"
    if count == 1:
        return "1 turn"
    return f"{count} turns"


def describe_target(target_grounding: dict[str, Any]) -> str:
    parts: list[str] = []
    rule = target_grounding.get("target_rule")
    category = target_grounding.get("target_category")
    color = target_grounding.get("target_color_name")
    if rule:
        rule_text = compact_value(rule)
        if rule_text and rule_text not in {"unknown", "none"}:
            parts.append(rule_text)
    if category:
        parts.append(compact_value(category))
    if color:
        parts.append(compact_value(color))
    if not parts:
        return "the target"
    return f"the {' '.join(parts)}"


def json_int_list(value: Any) -> str:
    try:
        return json.dumps([int(item) for item in value], ensure_ascii=True)
    except (TypeError, ValueError):
        return json.dumps(value, ensure_ascii=True)


def reasoning_payload(parsed: dict[str, Any]) -> dict[str, Any]:
    payload = parsed.get("reasoning_payload")
    return payload if isinstance(payload, dict) else {}


def merged_reasoning_section(packet: dict[str, Any], parsed: dict[str, Any], key: str) -> dict[str, Any]:
    facts = packet.get("facts") if isinstance(packet.get("facts"), dict) else {}
    base = facts.get(key) if isinstance(facts.get(key), dict) else {}
    payload = reasoning_payload(parsed)
    override = payload.get(key) if isinstance(payload.get(key), dict) else {}
    return {**base, **override}


def example_phrase(examples: Any, *, kept: bool) -> str:
    if not isinstance(examples, list) or not examples:
        return ""
    rendered = []
    for example in examples[:3]:
        if not isinstance(example, dict) or example.get("id") is None:
            continue
        reason = compact_value(example.get("reason"), default="")
        if reason:
            rendered.append(f"ID {compact_value(example.get('id'))} because {reason}")
        else:
            rendered.append(f"ID {compact_value(example.get('id'))}")
    if not rendered:
        return ""
    verb = "keep" if kept else "exclude"
    return f"I {verb} {', '.join(rendered)}."


def candidate_space_classification_sentence(candidate_space: dict[str, Any], *, embodied: bool) -> str:
    all_ids = candidate_space.get("all_display_ids") if isinstance(candidate_space.get("all_display_ids"), list) else []
    positive_key = "embodied_feasible_ids" if embodied else "pointmass_walkable_ids"
    negative_key = "not_embodied_feasible_ids" if embodied else "non_walkable_ids"
    positive_ids = candidate_space.get(positive_key) if isinstance(candidate_space.get(positive_key), list) else []
    negative_ids = candidate_space.get(negative_key) if isinstance(candidate_space.get(negative_key), list) else []
    count = compact_value(candidate_space.get("num_candidates"))
    if all_ids and count is not None:
        visible_sentence = f"I can see {count} visible candidate IDs: {json_int_list(all_ids)}."
    elif all_ids:
        visible_sentence = f"I can see the visible candidate IDs {json_int_list(all_ids)}."
    else:
        visible_sentence = "I can see the visible candidate IDs."

    if embodied:
        positive_phrase = "are embodied-feasible and leave enough room for the robot body"
        negative_phrase = "are too tight for the robot body"
        no_negative_phrase = "too-tight IDs"
        no_positive_phrase = "embodied-feasible IDs"
    else:
        positive_phrase = "are walkable for a point agent"
        negative_phrase = "are blocked for a point agent"
        no_negative_phrase = "blocked IDs"
        no_positive_phrase = "walkable IDs"

    if positive_ids and negative_ids:
        split_sentence = (
            f"IDs {json_int_list(positive_ids)} {positive_phrase}, while IDs {json_int_list(negative_ids)} {negative_phrase}."
        )
    elif positive_ids:
        split_sentence = f"All of them {positive_phrase}, so there are no {no_negative_phrase}."
    elif negative_ids:
        split_sentence = f"All of them {negative_phrase}, so there are no {no_positive_phrase}."
    else:
        split_sentence = "I do not have a usable walkability split."

    return f"{visible_sentence} {split_sentence}"


def visual_summary_sentence(visual_summary: dict[str, Any], *, embodied: bool) -> str:
    if not isinstance(visual_summary, dict) or not visual_summary:
        return ""
    def count_subject(value: Any, noun: str) -> str:
        try:
            count = int(value)
        except (TypeError, ValueError):
            return f"{compact_value(value)} {noun}s"
        if count == 1:
            return f"one {noun}"
        return f"{count} {noun}s"

    def count_verb(value: Any, singular: str, plural: str) -> str:
        try:
            return singular if int(value) == 1 else plural
        except (TypeError, ValueError):
            return plural

    total = compact_value(visual_summary.get("total_visible_points"))
    feasible_count = compact_value(visual_summary.get("feasible_count"))
    blocked_count = compact_value(visual_summary.get("blocked_count"))
    if embodied:
        feasible_phrase = count_subject(feasible_count, "visible point")
        blocked_phrase = count_subject(blocked_count, "visible point")
        sentence = (
            f"I start by looking at the image: I can see {total} visible points, but only {feasible_phrase} {count_verb(feasible_count, 'leaves', 'leave')} enough room for the robot body, "
            f"while {blocked_phrase} {count_verb(blocked_count, 'is', 'are')} too tight."
        )
    else:
        feasible_phrase = count_subject(feasible_count, "visible point")
        sentence = (
            f"I start by looking at the image: I can see {total} visible points, and {feasible_phrase} {count_verb(feasible_count, 'is', 'are')} open to a point agent, "
            f"so none of them block the direct route."
        )
    feasible_ids = visual_summary.get("feasible_display_ids")
    blocked_ids = visual_summary.get("blocked_display_ids")
    fragments: list[str] = []
    if feasible_ids:
        fragments.append(f"The complete usable ID list is {json_int_list(feasible_ids)}")
    if blocked_ids:
        fragments.append(f"The complete blocked ID list is {json_int_list(blocked_ids)}")
    if fragments:
        sentence += " " + ". ".join(fragments) + "."
    return sentence


def waypoint_summary(path_facts: dict[str, Any], *, embodied: bool) -> str:
    waypoints = path_facts.get("path_waypoints")
    if not isinstance(waypoints, list) or not waypoints:
        return ""
    selected = [item for item in waypoints if isinstance(item, dict)]
    if len(selected) > 5:
        selected = selected[:4] + selected[-1:]
    fragments = []
    for waypoint in selected:
        display_id = waypoint.get("display_id")
        if display_id is None:
            continue
        facts = []
        mode_key = "embodied_feasible" if embodied else "pointmass_walkable"
        clearance_key = "embodied_clearance_margin_m" if embodied else "point_clearance_m"
        clearance_value = waypoint.get(clearance_key)
        if waypoint.get(mode_key) is not None:
            if bool(waypoint.get(mode_key)):
                if embodied and clearance_value is not None:
                    facts.append(f"leaves {format_distance_m(clearance_value)} of room")
                elif embodied:
                    facts.append("leaves enough room")
                elif clearance_value is not None:
                    facts.append(f"is walkable with {format_distance_m(clearance_value)} of room")
                else:
                    facts.append("is walkable")
            else:
                facts.append("is too tight" if embodied else "is blocked")
        if facts:
            fragments.append(f"ID {compact_value(display_id)} {' and '.join(facts)}")
        else:
            fragments.append(f"ID {compact_value(display_id)}")
    if not fragments:
        return ""
    return f"As I trace the route, I check the waypoints one by one: {', '.join(fragments)}."


def segment_collision_sentence(path_facts: dict[str, Any], *, embodied: bool) -> str:
    segments = path_facts.get("path_segments")
    if not isinstance(segments, list) or not segments:
        return ""
    selected = [item for item in segments if isinstance(item, dict)]
    if len(selected) > 5:
        selected = selected[:4] + selected[-1:]
    fragments: list[str] = []
    for segment in selected:
        source = segment.get("from_id")
        target = segment.get("to_id")
        if source is None or target is None:
            continue
        segment_name = f"ID {compact_value(source)}->{compact_value(target)}"
        collision_free = segment.get("collision_free_projection")
        dense_clear = segment.get("dense_collision_free_projection")
        if collision_free is True and dense_clear is True:
            fragments.append(f"{segment_name} is collision-free, with the denser trace clear too")
        elif collision_free is True:
            fragments.append(f"{segment_name} is collision-free")
        elif dense_clear is True:
            fragments.append(f"{segment_name} stays clear on the denser trace")
        elif collision_free is False or dense_clear is False:
            fragments.append(f"{segment_name} needs collision checking")
    if not fragments:
        return ""
    route_name = "robot-body route" if embodied else "point-agent route"
    return (
        f"I also check the path between consecutive points, not only the endpoints: "
        f"{'; '.join(fragments)}, so the {route_name} stays connected without a collision."
    )


def target_grounding_sentence(target_grounding: dict[str, Any]) -> str:
    goal = target_grounding.get("goal_display_id")
    target_desc = describe_target(target_grounding)
    if goal is not None:
        return f"That makes me lock onto {target_desc}, which shows up at goal display ID {compact_value(goal)}."
    return f"That makes me lock onto {target_desc}."


def path_sentence(path_facts: dict[str, Any], *, embodied: bool, goal_display_id: Any | None = None) -> str:
    route_name = "the robot-body route" if embodied else "the point-agent route"
    start = compact_value(path_facts.get("start_id"))
    path = json_int_list(path_facts.get("shortest_path"))
    length = path_facts.get("path_length_m")
    goal_clause = f" to goal display ID {compact_value(goal_display_id)}" if goal_display_id is not None else ""
    if length is None:
        return f"From start ID {start}, I follow {path}{goal_clause} on {route_name}."
    return f"From start ID {start}, I follow {path}{goal_clause} on {route_name}, and it is {format_distance_m(length)} long."


def geometry_sentence(
    *,
    geometry: dict[str, Any] | None,
    embodied_facts: dict[str, Any] | None,
    routing: dict[str, Any] | None,
    embodied: bool,
) -> str:
    bits: list[str] = []
    if geometry:
        profile = geometry.get("geometry_profile")
        if profile is not None:
            profile_text = compact_value(profile).lower()
            if "straight" in profile_text:
                bits.append("the route stays straight")
        detour_key = "detour_ratio_embodied" if embodied else "detour_ratio_point"
        turn_key = "turn_count_embodied" if embodied else "turn_count_point"
        length_key = "path_length_embodied_m" if embodied else "path_length_point_m"
        detour = geometry.get(detour_key)
        if detour is not None:
            try:
                detour_value = float(detour)
            except (TypeError, ValueError):
                detour_value = None
            if detour_value is not None:
                if abs(detour_value - 1.0) < 1e-6:
                    bits.append("there is no detour")
                else:
                    bits.append(f"the route is about {detour_value:g}x the straight line")
        turns = geometry.get(turn_key)
        if turns is not None:
            bits.append(format_turn_count(turns))
        length = geometry.get(length_key)
        if length is not None:
            bits.append(f"it is {format_distance_m(length)} long")
    if embodied and embodied_facts:
        clearance = embodied_facts.get("embodied_clearance_margin_min_m")
        if clearance is not None:
            bits.append(f"the tightest gap still leaves {format_distance_m(clearance)} of room")
        narrow_fraction = embodied_facts.get("narrow_passage_fraction")
        if narrow_fraction is not None:
            bits.append(f"{format_fraction_as_percent(narrow_fraction)} of the route runs through narrow space")
    if embodied and routing:
        clearance = routing.get("embodied_min_clearance_m")
        if clearance is not None and not embodied_facts:
            bits.append(f"the tightest gap still leaves {format_distance_m(clearance)} of room")
        narrow_fraction = routing.get("embodied_narrow_passage_fraction")
        if narrow_fraction is not None and not embodied_facts:
            bits.append(f"{format_fraction_as_percent(narrow_fraction)} of the route runs through narrow space")
    if not bits:
        return ""
    prefix = "That matches the embodied route geometry" if embodied else "That matches the point-agent route geometry"
    return f"{prefix}: {', '.join(bits)}, so I treat it as the {'shortest feasible path' if embodied else 'shortest legal path'}."


def candidate_selection_sentence(
    selection: dict[str, Any],
    same_type: dict[str, Any],
    intent: dict[str, Any],
    cue: dict[str, Any],
) -> str:
    user_request = intent.get("user_request")
    intent_family = intent.get("intent_family")
    resolved_target_category = intent.get("resolved_target_category")
    candidate_ids = selection.get("candidate_target_ids")
    winner = selection.get("winner_target_id")
    goal = selection.get("goal_display_id")
    visible_count = same_type.get("same_type_supported_visible_count")
    parts: list[str] = []
    if user_request:
        parts.append(
            f'I read "{compact_value(user_request)}" as a {compact_value(intent_family)} request, so I am looking for {describe_target({"target_category": resolved_target_category})}.'
        )
    cue_bits: list[str] = []
    if cue.get("cue_family") is not None:
        cue_bits.append(f"cue family {compact_value(cue.get('cue_family'))}")
    if cue.get("cue_type") is not None:
        cue_bits.append(f"cue type {compact_value(cue.get('cue_type'))}")
    if cue.get("anchor_category") is not None:
        cue_bits.append(f"anchor {compact_value(cue.get('anchor_category'))}")
    if cue.get("residual_cue") is not None:
        cue_bits.append(f"residual cue {compact_value(cue.get('residual_cue'))}")
    if cue_bits:
        parts.append(f"I use the cue as {', '.join(cue_bits)}.")
    if visible_count is not None and candidate_ids:
        parts.append(
            f"I can see {compact_value(visible_count)} same-type candidates, namely {json_int_list(candidate_ids)}, so I compare them before choosing."
        )
    if winner is not None and goal is not None:
        parts.append(
            f"That leaves target {compact_value(winner)} as the closest valid match, which maps to goal display ID {compact_value(goal)}."
        )
    return " ".join(parts)


def render_oracle_cot_text(packet: dict[str, Any], parsed: dict[str, Any], final_json: list[int]) -> str:
    task = str(packet.get("task") or parsed.get("task") or "")
    facts = packet.get("facts") if isinstance(packet.get("facts"), dict) else {}
    payload = reasoning_payload(parsed)
    pieces: list[str] = []

    if task in {"a1", "b1"}:
        candidate_space = merged_reasoning_section(packet, parsed, "candidate_space")
        if task == "a1":
            pieces.append(candidate_space_classification_sentence(candidate_space, embodied=False))
        else:
            robot = merged_reasoning_section(packet, parsed, "robot_constraint")
            pieces.append(
                f"I look over {compact_value(candidate_space.get('num_candidates'))} candidates for robot-body traversability, and the robot diameter is {compact_value(robot.get('robot_diameter_m'))} m."
            )
            pieces.append(candidate_space_classification_sentence(candidate_space, embodied=True))
        selection_rule = payload.get("selection_rule") or facts.get("selection_rule")
        if selection_rule:
            pieces.append(f"I apply the rule that {compact_value(selection_rule).lower()}.")

    elif task == "a2":
        target_grounding = merged_reasoning_section(packet, parsed, "target_grounding")
        path_facts = merged_reasoning_section(packet, parsed, "path_facts")
        pieces.append(visual_summary_sentence(facts.get("visual_summary") or {}, embodied=False))
        pieces.append(target_grounding_sentence(target_grounding))
        pieces.append(path_sentence(path_facts, embodied=False, goal_display_id=target_grounding.get("goal_display_id")))
        route = waypoint_summary(path_facts, embodied=False)
        if route:
            pieces.append(route)
        segment_line = segment_collision_sentence(path_facts, embodied=False)
        if segment_line:
            pieces.append(segment_line)
        geometry = facts.get("navigation_geometry_facts") if isinstance(facts.get("navigation_geometry_facts"), dict) else {}
        geometry_line = geometry_sentence(geometry=geometry, embodied_facts=None, routing=None, embodied=False)
        if geometry_line:
            pieces.append(geometry_line)

    elif task == "b2":
        target_grounding = merged_reasoning_section(packet, parsed, "target_grounding")
        robot = merged_reasoning_section(packet, parsed, "robot_constraint")
        path_facts = merged_reasoning_section(packet, parsed, "path_facts")
        pieces.append(visual_summary_sentence(facts.get("visual_summary") or {}, embodied=True))
        pieces.append(target_grounding_sentence(target_grounding))
        robot_line = (
            f"The robot is {format_distance_m(robot.get('robot_diameter_m'))} wide, and the tightest gap leaves "
            f"{format_distance_m(robot.get('min_clearance_m'))} of room, so I keep the route to the one that stays comfortably open. "
            f"{format_fraction_as_percent(robot.get('narrow_passage_fraction'))} of the route runs through narrow space."
        )
        pieces.append(robot_line)
        pieces.append(path_sentence(path_facts, embodied=True, goal_display_id=target_grounding.get("goal_display_id")))
        route = waypoint_summary(path_facts, embodied=True)
        if route:
            pieces.append(route)
        segment_line = segment_collision_sentence(path_facts, embodied=True)
        if segment_line:
            pieces.append(segment_line)
        embodied_facts = facts.get("navigation_embodiment_facts") if isinstance(facts.get("navigation_embodiment_facts"), dict) else {}
        routing = facts.get("routing_complexity") if isinstance(facts.get("routing_complexity"), dict) else {}
        geometry_line = geometry_sentence(
            geometry=embodied_facts,
            embodied_facts=embodied_facts,
            routing=routing,
            embodied=True,
        )
        if geometry_line:
            pieces.append(geometry_line)

    elif task == "c":
        intent = merged_reasoning_section(packet, parsed, "intent_resolution")
        cue = merged_reasoning_section(packet, parsed, "cue_resolution")
        selection = merged_reasoning_section(packet, parsed, "target_selection")
        robot = facts.get("robot_constraint") if isinstance(facts.get("robot_constraint"), dict) else {}
        path_facts = merged_reasoning_section(packet, parsed, "path_facts")
        pieces.append(candidate_selection_sentence(selection, facts.get("same_type_visibility") or {}, intent, cue))
        same_type = facts.get("same_type_visibility") if isinstance(facts.get("same_type_visibility"), dict) else {}
        if robot.get("robot_diameter_m") is not None:
            pieces.append(
                f"The robot is {format_distance_m(robot.get('robot_diameter_m'))} wide, so I only trust a route that leaves enough room."
            )
        pieces.append(path_sentence(path_facts, embodied=True, goal_display_id=selection.get("goal_display_id")))
        route = waypoint_summary(path_facts, embodied=True)
        if route:
            pieces.append(route)
        segment_line = segment_collision_sentence(path_facts, embodied=True)
        if segment_line:
            pieces.append(segment_line)
        embodied_facts = facts.get("navigation_embodiment_facts") if isinstance(facts.get("navigation_embodiment_facts"), dict) else {}
        routing = facts.get("routing_complexity") if isinstance(facts.get("routing_complexity"), dict) else {}
        geometry_line = geometry_sentence(
            geometry=embodied_facts,
            embodied_facts=embodied_facts,
            routing=routing,
            embodied=True,
        )
        if geometry_line:
            pieces.append(geometry_line)

    else:
        pieces.append("Use the supplied navigation facts to derive the required answer.")

    rationale = " ".join(piece.strip() for piece in pieces if piece and piece.strip())
    return f"<think>\n{rationale}\n</think>\n{json.dumps(final_json, ensure_ascii=True)}"


def build_cot_assistant(packet: dict[str, Any], parsed: dict[str, Any], final_json: list[int]) -> str:
    return render_oracle_cot_text(packet, parsed, final_json)


def build_cot_sft_row(packet: dict[str, Any], parsed: dict[str, Any]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "user", "content": build_user_message(packet)},
            {"role": "assistant", "content": build_cot_assistant(packet, parsed, packet["required_final_json"])},
        ],
        "images": [packet["image_path"]],
        "metadata": {
            "question_id": packet["question_id"],
            "task": packet["task"],
            "supervision": "oracle_cot_sft",
            "reasoning_type": parsed.get("reasoning_type"),
        },
    }


def deterministic_oracle_parsed(packet: dict[str, Any]) -> dict[str, Any]:
    return required_output_schema(packet)


def dataset_info_payload() -> dict[str, Any]:
    base = {
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    }
    return {
        "navbench_direct_sft": {"file_name": "navbench_direct_sft.json", **base},
        "navbench_oracle_cot_sft": {"file_name": "navbench_oracle_cot_sft.json", **base},
        "navbench_direct_sft_smoke": {"file_name": "navbench_direct_sft_smoke.json", **base},
        "navbench_oracle_cot_sft_smoke": {"file_name": "navbench_oracle_cot_sft_smoke.json", **base},
        "navbench_direct_sft_sanity500": {"file_name": "navbench_direct_sft_sanity500.json", **base},
        "navbench_oracle_cot_sft_sanity500": {"file_name": "navbench_oracle_cot_sft_sanity500.json", **base},
        "navbench_direct_sft_gate10pct": {"file_name": "navbench_direct_sft_gate10pct.json", **base},
        "navbench_oracle_cot_sft_gate10pct": {"file_name": "navbench_oracle_cot_sft_gate10pct.json", **base},
    }


def write_sft_data(
    *,
    packets: list[dict[str, Any]],
    output_dir: Path,
    data_dir: Path,
    smoke_count: int,
) -> dict[str, Any]:
    generated_by_qid = load_generated_by_qid(output_dir)

    direct_rows = [build_direct_sft_row(packet) for packet in packets]
    cot_rows = [
        build_cot_sft_row(packet, deterministic_oracle_parsed(packet))
        for packet in packets
    ]

    direct_smoke = direct_rows[:smoke_count]
    cot_smoke = cot_rows[:smoke_count]
    direct_sanity = stratified_limit(direct_rows, total_limit=500)
    cot_sanity = stratified_limit(cot_rows, total_limit=500)
    direct_gate = stratified_fraction(direct_rows, fraction=0.1)
    cot_gate = stratified_fraction(cot_rows, fraction=0.1)

    write_json_array(data_dir / "navbench_direct_sft.json", direct_rows)
    write_json_array(data_dir / "navbench_oracle_cot_sft.json", cot_rows)
    write_json_array(data_dir / "navbench_direct_sft_smoke.json", direct_smoke)
    write_json_array(data_dir / "navbench_oracle_cot_sft_smoke.json", cot_smoke)
    write_json_array(data_dir / "navbench_direct_sft_sanity500.json", direct_sanity)
    write_json_array(data_dir / "navbench_oracle_cot_sft_sanity500.json", cot_sanity)
    write_json_array(data_dir / "navbench_direct_sft_gate10pct.json", direct_gate)
    write_json_array(data_dir / "navbench_oracle_cot_sft_gate10pct.json", cot_gate)
    write_json(data_dir / "dataset_info.json", dataset_info_payload())

    return {
        "data_dir": str(data_dir),
        "direct_rows": len(direct_rows),
        "cot_rows": len(cot_rows),
        "direct_smoke_rows": len(direct_smoke),
        "cot_smoke_rows": len(cot_smoke),
        "direct_sanity500_rows": len(direct_sanity),
        "cot_sanity500_rows": len(cot_sanity),
        "direct_gate10pct_rows": len(direct_gate),
        "cot_gate10pct_rows": len(cot_gate),
        "direct_source": "all_selected_fact_packets",
        "cot_source": "deterministic_fact_packets",
        "available_generated_json_rows": len(generated_by_qid),
    }


def build_qc_report(output_dir: Path, packets: list[dict[str, Any]]) -> dict[str, Any]:
    packet_by_qid = {packet["question_id"]: packet for packet in packets}
    generated_by_qid = load_generated_by_qid(output_dir)
    rejected_by_qid: dict[str, dict[str, Any]] = {}
    for path in sorted((output_dir / "rejected").glob("*/*.jsonl")):
        for row in iter_jsonl(path):
            qid = row.get("question_id")
            if qid is not None:
                rejected_by_qid[str(qid)] = row
    rejected_rows = [
        row
        for qid, row in rejected_by_qid.items()
        if qid in packet_by_qid and qid not in generated_by_qid
    ]

    task_counts = Counter(packet["task"] for packet in packets)
    accepted_counts = Counter(packet_by_qid[qid]["task"] for qid in generated_by_qid if qid in packet_by_qid)
    rejected_counts = Counter(str(row.get("task")) for row in rejected_rows)
    issue_counts: Counter[str] = Counter()
    for row in rejected_rows:
        qc = row.get("qc") if isinstance(row.get("qc"), dict) else row
        for issue in qc.get("issues") or []:
            issue_counts[str(issue)] += 1

    edge_status_by_task: dict[str, Counter[str]] = {task: Counter() for task in TASKS}
    for qid, row in generated_by_qid.items():
        packet = packet_by_qid.get(qid)
        if packet is None:
            continue
        task = str(packet["task"])
        qc = row.get("qc") if isinstance(row.get("qc"), dict) else {}
        edge_audit = qc.get("path_edge_legality") if isinstance(qc, dict) else None
        if isinstance(edge_audit, dict):
            status = str(edge_audit.get("status") or "unknown")
        else:
            status = str(edge_audit or "missing")
        edge_status_by_task[task][status] += 1

    per_task: dict[str, dict[str, Any]] = {}
    for task in TASKS:
        total = task_counts.get(task, 0)
        accepted = accepted_counts.get(task, 0)
        rejected = rejected_counts.get(task, 0)
        final_json_rate = accepted / total if total else 0.0
        rejected_rate = rejected / total if total else 0.0
        edge_status_counts = dict(sorted(edge_status_by_task[task].items()))
        edge_ok_count = sum(
            count for status, count in edge_status_by_task[task].items() if status.startswith("checked_ok")
        )
        edge_ready = task not in {"a2", "b2", "c"} or (accepted > 0 and edge_ok_count == accepted)
        per_task[task] = {
            "selected_packets": total,
            "accepted_generated_rows": accepted,
            "rejected_rows": rejected,
            "final_json_match_rate": round(final_json_rate, 6),
            "rejected_rate": round(rejected_rate, 6),
            "path_edge_legality": edge_status_counts,
            "ready_for_full": bool(total and accepted == total and rejected_rate < 0.01 and edge_ready),
        }

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_packets": len(packets),
        "accepted_generated_rows": len(generated_by_qid),
        "rejected_rows": len(rejected_rows),
        "counts_by_task": dict(sorted(task_counts.items())),
        "accepted_by_task": dict(sorted(accepted_counts.items())),
        "rejected_by_task": dict(sorted(rejected_counts.items())),
        "issue_counts": dict(sorted(issue_counts.items())),
        "path_edge_legality_by_task": {
            task: dict(sorted(counter.items()))
            for task, counter in edge_status_by_task.items()
            if counter
        },
        "per_task": per_task,
        "acceptance_thresholds": {
            "final_json_equals_reference": "100%",
            "path_edge_legality": "100% checked_ok* for a2/b2/c accepted rows",
            "display_id_space_check": "100%",
            "banned_leakage_phrase": 0,
            "rejected_rows_per_task": "< 1%",
        },
    }


def build_raw_response_audit(output_dir: Path) -> dict[str, Any]:
    rows = 0
    models: Counter[str] = Counter()
    api_bases: Counter[str] = Counter()
    request_modes: Counter[str] = Counter()
    for path in sorted((output_dir / "raw_responses").glob("*/*.jsonl")):
        for row in iter_jsonl(path):
            rows += 1
            for key, counter in (
                ("model", models),
                ("api_base", api_bases),
                ("request_mode", request_modes),
            ):
                value = row.get(key)
                counter[str(value) if value is not None else "null"] += 1

    return {
        "raw_response_rows": rows,
        "models": dict(sorted(models.items())),
        "api_bases": dict(sorted(api_bases.items())),
        "request_modes": dict(sorted(request_modes.items())),
    }


def single_counter_key(counter_payload: dict[str, int]) -> str | None:
    return next(iter(counter_payload)) if len(counter_payload) == 1 else None


def build_manifest(args: argparse.Namespace, output_dir: Path, packets: list[dict[str, Any]], sft_report: dict[str, Any]) -> dict[str, Any]:
    raw_audit = build_raw_response_audit(output_dir)
    raw_model = single_counter_key(raw_audit["models"])
    raw_api_base = single_counter_key(raw_audit["api_bases"])
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/run_oracle_cot_generation.py",
        "protocol": "refine-logs/ORACLE_COT_GENERATION_PROTOCOL_20260607.md",
        "release_root": str(Path(args.release_root)),
        "split": args.split,
        "stage": args.stage,
        "tasks": list(parse_tasks(args.tasks)),
        "seed": args.seed,
        "shard_size": args.shard_size,
        "selected_packets": len(packets),
        "model": args.model if args.generate else raw_model,
        "api_base": resolve_api_base(args.api_base, args.api_base_env) if args.generate else raw_api_base,
        "raw_response_audit": raw_audit,
        "request_config": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_completion_tokens": args.max_completion_tokens,
            "response_format": {"type": "json_object"},
            "api_key_env": args.api_key_env if args.generate else None,
            "api_base_env": args.api_base_env if args.generate else None,
            "allow_external_upload": bool(args.allow_external_upload) if args.generate else None,
            "sent_fact_packet_includes": [
                "question text",
                "required final_json",
                "candidate display IDs",
                "shortest_path",
                "clearance and traversability facts",
                "target grounding facts",
                "intent/cue facts",
            ],
            "sent_fact_packet_excludes": [
                "image_path",
                "system_prompt",
                "source_refs",
                "audit_refs",
                "screen_xy_norm",
            ],
        },
        "outputs": {
            "fact_packets": "fact_packets/<task>/shard_*.jsonl",
            "raw_responses": "raw_responses/<task>/shard_*.jsonl",
            "parsed": "parsed/<task>/shard_*.jsonl",
            "rejected": "rejected/<task>/shard_*.jsonl",
            "qc_report": "qc_report.json",
            "llamafactory_data": sft_report.get("data_dir"),
        },
        "sft": sft_report,
    }


def resolve_limit(stage: str, max_per_task: int | None) -> int | None:
    if max_per_task is not None:
        return None if max_per_task == 0 else max_per_task
    if stage not in STAGE_LIMITS:
        raise OracleCotError(f"unknown stage={stage}")
    return STAGE_LIMITS[stage]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NavBench3D GPT oracle-CoT packets and SFT data.")
    parser.add_argument("--release-root", default=str(DEFAULT_RELEASE_ROOT), help="Path to data/release")
    parser.add_argument("--split", default="train")
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--stage", choices=tuple(STAGE_LIMITS), default="pilot")
    parser.add_argument("--max-per-task", type=int, default=None, help="Override stage limit; 0 means all rows")
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--llamafactory-data-dir", default=str(DEFAULT_LF_DATA_DIR))
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--task-shard-index", type=int, default=0)
    parser.add_argument("--task-shard-count", type=int, default=1)
    parser.add_argument("--sft-smoke-count", type=int, default=2)
    parser.add_argument("--skip-fact-packet-write", action="store_true")
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate raw/parsed/rejected rows. Skip QC, manifest, and SFT export for parallel workers.",
    )
    parser.add_argument("--generate", action="store_true", help="Call GPT-compatible API for oracle-CoT JSON.")
    parser.add_argument(
        "--allow-external-upload",
        action="store_true",
        help="Required with --generate; acknowledges sanitized training facts will be sent to the configured API.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-rejected", action="store_true")
    parser.add_argument(
        "--allow-partial-current-sft-export",
        action="store_true",
        help="Allow pilot/validation generation to overwrite the default current LLaMA-Factory data directory.",
    )
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-base-env", default="OPENAI_BASE_URL")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--max-completion-tokens", type=int, default=1800)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--retry-sleep-s", type=float, default=5.0)
    parser.add_argument("--retry-jitter-s", type=float, default=0.0)
    parser.add_argument("--request-sleep-s", type=float, default=0.0)
    parser.add_argument(
        "--full-generation-request-interval-s",
        type=float,
        default=float(os.environ.get("ORACLE_COT_FULL_REQUEST_INTERVAL_S", "20.0")),
        help="Minimum shared interval between API request starts during full external generation.",
    )
    parser.add_argument(
        "--full-generation-max-inflight",
        type=int,
        default=int(os.environ.get("ORACLE_COT_FULL_MAX_INFLIGHT", "1")),
        help="Maximum concurrent full external API requests across workers; 0 disables the in-flight lock.",
    )
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--fail-fast-on-api-error",
        action="store_true",
        help="Exit instead of writing rejected rows when the API or parser fails after retries.",
    )
    parser.add_argument(
        "--defer-transient-api-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "During full external generation, keep fail-fast rows out of rejected output but continue past "
            "transient API failures so later resume passes can fill the holes."
        ),
    )
    parser.add_argument(
        "--defer-max-retries",
        type=int,
        default=1,
        help="Cap request retries before deferring a row during full external generation.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--batch-max-completion-tokens",
        type=int,
        default=None,
        help="Override max_completion_tokens for a batched request; default scales single-row budget by batch size.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    data_dir = Path(args.llamafactory_data_dir)
    tasks = parse_tasks(args.tasks)
    limit = resolve_limit(args.stage, args.max_per_task)
    rows = collect_rows(Path(args.release_root), args.split, tasks, limit, args.seed)
    packets = [build_fact_packet(row) for row in rows]

    if not args.skip_fact_packet_write:
        write_fact_packets(output_dir, packets, args.shard_size)
    if not args.generate_only:
        require_safe_sft_export_target(args, data_dir)
    if args.generate:
        require_external_upload_ack(args)
        packets_to_generate = select_task_shard(
            packets,
            shard_index=args.task_shard_index,
            shard_count=args.task_shard_count,
        )
        generate_for_packets(args, output_dir, packets_to_generate)
        if args.generate_only:
            print(json.dumps({
                "output_dir": str(output_dir),
                "stage": args.stage,
                "task_shard_index": args.task_shard_index,
                "task_shard_count": args.task_shard_count,
                "selected_packets": len(packets_to_generate),
                "mode": "generate_only",
            }, ensure_ascii=True, indent=2, sort_keys=True))
            return

    sft_report = write_sft_data(
        packets=packets,
        output_dir=output_dir,
        data_dir=data_dir,
        smoke_count=args.sft_smoke_count,
    )
    qc_report = build_qc_report(output_dir, packets)
    write_json(output_dir / "qc_report.json", qc_report)
    manifest = build_manifest(args, output_dir, packets, sft_report)
    write_json(output_dir / "manifest.json", manifest)

    print(json.dumps({
        "output_dir": str(output_dir),
        "llamafactory_data_dir": str(data_dir),
        "stage": args.stage,
        "total_packets": len(packets),
        "accepted_generated_rows": qc_report["accepted_generated_rows"],
        "rejected_rows": qc_report["rejected_rows"],
        "counts_by_task": qc_report["counts_by_task"],
        "sft": sft_report,
    }, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
