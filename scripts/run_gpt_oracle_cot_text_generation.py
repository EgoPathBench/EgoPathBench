#!/usr/bin/env python3
"""Generate GPT-written natural-language oracle CoT text for NavBench3D."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_oracle_cot_generation as oracle  # noqa: E402


DEFAULT_OUTPUT_DIR = Path("results/oracle_cot_gpt_text_v1_20260611")
CODEX_USER_AGENT = "codex_exec/0.133.0 (Ubuntu 22.4.0; x86_64) dumb (codex_exec; 0.133.0)"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def existing_accepted_qids(output_dir: Path) -> set[str]:
    qids: set[str] = set()
    for kind in ("accepted", "accepted_reqc"):
        for path in sorted((output_dir / kind).glob("*/*.jsonl")):
            for row in iter_jsonl(path):
                if row.get("question_id"):
                    qids.add(str(row["question_id"]))
    return qids


def system_prompt() -> str:
    return (
        "You write concise natural-language training rationales for NavBench3D. "
        "Treat the evidence packet as visual evidence already extracted from the scene, not as metadata to cite. "
        "Write a first-person reasoning paragraph that sounds like active visual navigation: I inspect what is visible, compare candidates, rule out unsafe or irrelevant choices, trace a route, and then verify the shortest valid path. "
        "Do not write a report, checklist, field summary, or explanation that simply announces the answer first. "
        "Use only the supplied evidence. Do not solve a new path. Do not change the final JSON. "
        "Do not mention ground truth, metadata, hashes, local paths, or world coordinates. "
        "For a2, b2, and c, turn the evidence into a causal chain rather than copying facts mechanically. "
        "Return plain text only, not JSON or markdown."
    )


def planning_rationale_contract(task: str) -> list[str]:
    if task == "a2":
        return [
            "Write one flowing first-person paragraph, not a numbered list or labeled outline.",
            "Choose your own sentence order, but the paragraph must connect scene inspection, target discrimination, point-agent feasibility, route tracing, and shortest-route verification.",
            "Do not begin by simply declaring the final path; build toward it from visible evidence.",
            "Use the target category, target rule, any same-category candidate IDs, and the goal display ID to explain why this target is selected.",
            "Use the usable and blocked point IDs to explain which numbered points can support a point-agent route.",
            "Also explain that each adjacent route segment is collision-free, not only that the endpoint IDs are usable.",
            "Trace the route from the start ID to the goal and mention the path list and path length.",
            "Explain shortestness with route shape, detour, turn count, and path length rather than only the length number.",
            "Use causal words such as because, so, therefore, which means, or that means.",
            "Keep robot-body, embodied-routing, intent, and cue language out of the rationale.",
        ]
    if task == "b2":
        return [
            "Write one flowing first-person paragraph, not a numbered list or labeled outline.",
            "Choose your own sentence order, but the paragraph must connect scene inspection, target discrimination, robot-body feasibility, route tracing, and shortest embodied-route verification.",
            "Do not begin by simply declaring the final path; build toward it from visible evidence.",
            "Use the target category, target rule, any same-category candidate IDs, and the goal display ID to explain why this target is selected.",
            "Treat robot diameter, clearance, usable point IDs, too-tight point IDs, and collision-free adjacent route segments as passability evidence for the robot body.",
            "Trace the route from the start ID to the goal and mention the path list and path length.",
            "Explain shortestness with route shape, clearance, narrow passages, detour, turn count, and path length rather than only the length number.",
            "Use causal words such as because, so, therefore, which means, or that means.",
            "Do not describe the route as pointmass-only.",
        ]
    return [
        "Write one flowing first-person paragraph, not a numbered list or labeled outline.",
        "Choose your own sentence order, but the paragraph must connect request interpretation, cue resolution, target discrimination, robot-body feasibility, route tracing, and shortest embodied-route verification.",
        "Do not begin by simply declaring the final path; build toward it from visible evidence.",
        "Explain how the request, intent, cue, same-type candidate IDs, winner target ID, and goal display ID resolve the implicit target.",
        "Treat robot diameter, clearance, usable point IDs, too-tight point IDs, and collision-free adjacent route segments as passability evidence for the robot body.",
        "Trace the route from the start ID to the goal and mention the path list and path length.",
        "Explain shortestness with route shape, clearance, narrow passages, detour, turn count, and path length rather than only the length number.",
        "Use causal words such as because, so, therefore, which means, or that means.",
    ]


def task_requirements(task: str) -> list[str]:
    if task == "a1":
        return [
            "Explain that this is point-agent traversability.",
            "Use the full visible candidate set, not just examples.",
            "Explicitly split the candidates into walkable and blocked IDs and show the complete split.",
            "The visible candidate set may be summarized as a continuous range when it has no gaps.",
            "Copy the exact bracketed walkable and blocked ID lists; do not summarize those subsets as ranges.",
            "State that the final list selects all pointmass-walkable display IDs.",
        ]
    if task == "b1":
        return [
            "Explain that this is robot-body traversability.",
            "Mention robot diameter and clearance/embodied feasibility.",
            "Use the full visible candidate set, not just examples.",
            "Explicitly split the candidates into embodied-feasible and too-tight IDs and show the complete split.",
            "The visible candidate set may be summarized as a continuous range when it has no gaps.",
            "Copy the exact bracketed feasible and too-tight ID lists; do not summarize those subsets as ranges.",
        ]
    if task in {"a2", "b2", "c"}:
        return planning_rationale_contract(task)
    return []


def planning_flow_instruction(task: str) -> str:
    if task == "a2":
        return (
            "Use the evidence to form a natural point-agent navigation rationale; let the sentence order emerge from the scene rather than following a rigid checklist."
        )
    if task == "b2":
        return (
            "Use the evidence to form a natural robot-body navigation rationale; let the sentence order emerge from the scene rather than following a rigid checklist."
        )
    return (
        "Use the evidence to form a natural implicit-goal robot-body navigation rationale; let the sentence order emerge from the scene rather than following a rigid checklist."
    )


def append_evidence_line(lines: list[str], label: str, value: Any) -> None:
    if value in (None, "", [], {}, "unknown"):
        return
    lines.append(f"{label}: {value}.")


def optional_int_list_text(value: Any) -> str | None:
    if not isinstance(value, list) or not value:
        return None
    return oracle.json_int_list(value)


def route_detour_text(value: Any) -> str | None:
    try:
        detour_value = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    if detour_value is None:
        return None
    if abs(detour_value - 1.0) < 1e-6:
        return "no detour"
    return f"detour ratio {detour_value:g}"


def waypoint_evidence_text(waypoints: Any, *, embodied: bool) -> str | None:
    if not isinstance(waypoints, list) or not waypoints:
        return None
    parts: list[str] = []
    for waypoint in waypoints:
        if not isinstance(waypoint, dict) or waypoint.get("display_id") is None:
            continue
        item_parts = [f"ID {waypoint.get('display_id')}"]
        if waypoint.get("pointmass_walkable") is not None:
            item_parts.append("point-agent usable" if waypoint.get("pointmass_walkable") else "point-agent blocked")
        if embodied and waypoint.get("embodied_feasible") is not None:
            item_parts.append("robot-body usable" if waypoint.get("embodied_feasible") else "too tight for the robot body")
        if embodied and waypoint.get("embodied_clearance_margin_m") is not None:
            item_parts.append(f"clearance margin {oracle.format_distance_m(waypoint.get('embodied_clearance_margin_m'))}")
        elif waypoint.get("point_clearance_m") is not None:
            item_parts.append(f"open-space cue {oracle.format_distance_m(waypoint.get('point_clearance_m'))}")
        parts.append(" / ".join(item_parts))
    if not parts:
        return None
    return "; ".join(parts)


def segment_evidence_text(segments: Any) -> str | None:
    if not isinstance(segments, list) or not segments:
        return None
    parts: list[str] = []
    for segment in segments:
        if not isinstance(segment, dict):
            continue
        source = segment.get("from_id")
        target = segment.get("to_id")
        if source is None or target is None:
            continue
        status_bits = [f"{source}->{target}"]
        if segment.get("collision_free_projection") is not None:
            status_bits.append(
                "collision-free projection"
                if segment.get("collision_free_projection")
                else "projection may collide"
            )
        if segment.get("dense_collision_free_projection") is not None:
            status_bits.append(
                "dense path also clear"
                if segment.get("dense_collision_free_projection")
                else "dense path not clear"
            )
        parts.append(" / ".join(status_bits))
    if not parts:
        return None
    return "; ".join(parts)


def planning_context_lines(packet: dict[str, Any]) -> list[str]:
    task = str(packet["task"])
    facts = packet.get("facts") if isinstance(packet.get("facts"), dict) else {}
    path_facts = facts.get("path_facts") if isinstance(facts.get("path_facts"), dict) else {}
    lines: list[str] = []

    visual = facts.get("visual_summary") if isinstance(facts.get("visual_summary"), dict) else {}
    if visual:
        total = visual.get("total_visible_points")
        feasible_ids = visual.get("feasible_display_ids") or []
        blocked_ids = visual.get("blocked_display_ids") or []
        if task == "a2":
            lines.append(
                f"Visible numbered points: {total}; point-agent usable IDs {oracle.json_int_list(feasible_ids)}; point-agent blocked IDs {oracle.json_int_list(blocked_ids)}."
            )
        else:
            lines.append(
                f"Visible numbered points: {total}; robot-body usable IDs {oracle.json_int_list(feasible_ids)}; too-tight or blocked IDs {oracle.json_int_list(blocked_ids)}."
            )

    if task in {"a2", "b2"}:
        target = facts.get("target_grounding") if isinstance(facts.get("target_grounding"), dict) else {}
        if target:
            append_evidence_line(lines, "Target category", target.get("target_category"))
            append_evidence_line(lines, "Target selection rule", target.get("target_rule"))
            append_evidence_line(lines, "Target maps to goal display ID", target.get("goal_display_id"))
        selection = facts.get("instruction_selection") if isinstance(facts.get("instruction_selection"), dict) else {}
        if selection:
            append_evidence_line(lines, "Same-category candidate target IDs", optional_int_list_text(selection.get("cohort_target_ids")))
            append_evidence_line(lines, "Rule-winning target IDs", optional_int_list_text(selection.get("winner_target_ids")))
            append_evidence_line(lines, "Target selection status", selection.get("selection_status"))
            append_evidence_line(lines, "Target distance rank", selection.get("distance_rank"))

    if task in {"b2", "c"}:
        robot = facts.get("robot_constraint") if isinstance(facts.get("robot_constraint"), dict) else {}
        if robot:
            robot_diameter = oracle.format_distance_m(robot.get("robot_diameter_m"))
            append_evidence_line(lines, "Robot diameter", robot_diameter)
            append_evidence_line(lines, "Tightest usable clearance on the route", oracle.format_distance_m(robot.get("min_clearance_m")))
            append_evidence_line(lines, "Narrow-passage share of the route", oracle.format_fraction_as_percent(robot.get("narrow_passage_fraction")))

    route_semantics = path_facts.get("route_semantics")
    path = path_facts.get("shortest_path")
    if path_facts:
        append_evidence_line(lines, "Route semantics", route_semantics)
        append_evidence_line(lines, "Start ID", path_facts.get("start_id"))
        append_evidence_line(lines, "Route to explain", oracle.json_int_list(path))
        append_evidence_line(lines, "Path length", oracle.format_distance_m(path_facts.get("path_length_m")))
        waypoint_text = waypoint_evidence_text(path_facts.get("path_waypoints"), embodied=task in {"b2", "c"})
        append_evidence_line(lines, "Waypoint checks along this route", waypoint_text)
        segment_text = segment_evidence_text(path_facts.get("path_segments"))
        append_evidence_line(lines, "Adjacent segment checks along this route", segment_text)
        if path_facts.get("path_projection_safe") is not None:
            append_evidence_line(
                lines,
                "Whole-route collision projection",
                "clear" if path_facts.get("path_projection_safe") else "not clear",
            )

    if task == "a2":
        geometry = facts.get("navigation_geometry_facts") if isinstance(facts.get("navigation_geometry_facts"), dict) else {}
        if geometry:
            detour = geometry.get("detour_ratio_point")
            shortest_cues = [
                route_detour_text(detour),
                oracle.format_turn_count(geometry.get("turn_count_point")),
                geometry.get("geometry_profile"),
            ]
            lines.append(f"Shortest-route cues: {', '.join(str(item) for item in shortest_cues if item)}.")
    if task == "b2":
        embodiment = facts.get("navigation_embodiment_facts") if isinstance(facts.get("navigation_embodiment_facts"), dict) else {}
        routing = facts.get("routing_complexity") if isinstance(facts.get("routing_complexity"), dict) else {}
        if embodiment or routing:
            shortest_cues = [
                route_detour_text(embodiment.get("detour_ratio_embodied") or routing.get("embodied_detour_ratio_sparse")),
                oracle.format_turn_count(embodiment.get("turn_count_embodied")),
                f"extra embodied length {oracle.format_distance_m(embodiment.get('embodied_extra_length_m'))}" if embodiment.get("embodied_extra_length_m") is not None else None,
                f"narrow-passage share {oracle.format_fraction_as_percent(embodiment.get('narrow_passage_fraction') or routing.get('embodied_narrow_passage_fraction'))}",
            ]
            lines.append(f"Shortest embodied-route cues: {', '.join(str(item) for item in shortest_cues if item)}.")
    if task == "c":
        intent = facts.get("intent_resolution") if isinstance(facts.get("intent_resolution"), dict) else {}
        cue = facts.get("cue_resolution") if isinstance(facts.get("cue_resolution"), dict) else {}
        selection = facts.get("target_selection") if isinstance(facts.get("target_selection"), dict) else {}
        same_type = facts.get("same_type_visibility") if isinstance(facts.get("same_type_visibility"), dict) else {}
        embodiment = facts.get("navigation_embodiment_facts") if isinstance(facts.get("navigation_embodiment_facts"), dict) else {}
        routing = facts.get("routing_complexity") if isinstance(facts.get("routing_complexity"), dict) else {}
        if intent:
            append_evidence_line(lines, "User request", json.dumps(intent.get("user_request"), ensure_ascii=True))
            append_evidence_line(lines, "Intent family", intent.get("intent_family"))
            append_evidence_line(lines, "Resolved target category", intent.get("resolved_target_category"))
        if cue:
            cue_bits = []
            if cue.get("cue_family") is not None:
                cue_bits.append(f"cue family {cue.get('cue_family')}")
            if cue.get("cue_type") is not None:
                cue_bits.append(f"cue type {cue.get('cue_type')}")
            if cue.get("anchor_category") is not None:
                cue_bits.append(f"anchor {cue.get('anchor_category')}")
            if cue.get("residual_cue") is not None:
                cue_bits.append(f"residual cue {cue.get('residual_cue')}")
            if cue_bits:
                append_evidence_line(lines, "Cue evidence", ", ".join(cue_bits))
        if selection:
            candidate_ids = selection.get("candidate_target_ids") or []
            append_evidence_line(lines, "Same-type candidate target count", selection.get("candidate_target_count"))
            append_evidence_line(lines, "Same-type candidate target IDs", oracle.json_int_list(candidate_ids))
            append_evidence_line(lines, "Winner target ID", selection.get("winner_target_id"))
            append_evidence_line(lines, "Winner maps to goal display ID", selection.get("goal_display_id"))
            append_evidence_line(lines, "Target selection status", selection.get("selection_status"))
        if same_type:
            append_evidence_line(lines, "Visible same-type object IDs", oracle.json_int_list(same_type.get("supported_visible_same_type_ids") or []))
        if embodiment or routing:
            shortest_cues = [
                route_detour_text(routing.get("embodied_detour_ratio_sparse") or routing.get("detour_ratio_sparse")),
                oracle.format_turn_count(routing.get("sparse_turn_count") or embodiment.get("turn_count_embodied")),
                f"narrow-passage share {oracle.format_fraction_as_percent(routing.get('embodied_narrow_passage_fraction') or embodiment.get('narrow_passage_fraction'))}",
                f"minimum route clearance {oracle.format_distance_m(routing.get('embodied_min_clearance_m') or embodiment.get('embodied_clearance_margin_min_m'))}",
            ]
            lines.append(f"Shortest embodied-route cues: {', '.join(str(item) for item in shortest_cues if item)}.")
    return lines


def compact_planning_fact_packet(packet: dict[str, Any]) -> dict[str, Any]:
    safe_packet = oracle.packet_for_generation(packet)
    task = str(safe_packet.get("task") or "")
    if task not in {"a2", "b2", "c"}:
        return safe_packet
    facts = safe_packet.get("facts") if isinstance(safe_packet.get("facts"), dict) else {}
    compact_facts: dict[str, Any] = {}
    for key in (
        "visual_summary",
        "target_grounding",
        "instruction_selection",
        "robot_constraint",
        "intent_resolution",
        "cue_resolution",
        "target_selection",
        "same_type_visibility",
        "navigation_geometry_facts",
        "navigation_embodiment_facts",
        "routing_complexity",
    ):
        if key in facts:
            compact_facts[key] = facts[key]
    path_facts = facts.get("path_facts") if isinstance(facts.get("path_facts"), dict) else {}
    compact_path = {
        key: path_facts[key]
        for key in ("start_id", "shortest_path", "path_length_m", "route_semantics")
        if key in path_facts
    }
    if "path_projection_safe" in path_facts:
        compact_path["path_projection_safe"] = path_facts["path_projection_safe"]
    waypoints = path_facts.get("path_waypoints")
    if isinstance(waypoints, list):
        compact_path["path_waypoints"] = [
            {
                key: value
                for key, value in waypoint.items()
                if key
                in {
                    "display_id",
                    "pointmass_walkable",
                    "embodied_feasible",
                    "point_clearance_m",
                    "embodied_clearance_margin_m",
                }
            }
            for waypoint in waypoints[:4]
            if isinstance(waypoint, dict)
        ]
    segments = path_facts.get("path_segments")
    if isinstance(segments, list):
        compact_path["path_segments"] = [
            {
                key: value
                for key, value in segment.items()
                if key
                in {
                    "from_id",
                    "to_id",
                    "route_semantics",
                    "collision_free_projection",
                    "dense_collision_free_projection",
                    "sparse_segment_count",
                }
            }
            for segment in segments[:4]
            if isinstance(segment, dict)
        ]
    compact_facts["path_facts"] = compact_path
    safe_packet["facts"] = compact_facts
    return safe_packet


def gpt_fact_packet(packet: dict[str, Any], *, compact_planning_prompt: bool = False) -> dict[str, Any]:
    if compact_planning_prompt:
        safe_packet = compact_planning_fact_packet(packet)
    else:
        safe_packet = oracle.packet_for_generation(packet)
    if safe_packet.get("task") != "a2":
        return safe_packet
    facts = safe_packet.get("facts")
    if not isinstance(facts, dict):
        return safe_packet
    path_facts = facts.get("path_facts")
    if not isinstance(path_facts, dict):
        return safe_packet
    waypoints = path_facts.get("path_waypoints")
    if not isinstance(waypoints, list):
        return safe_packet
    path_facts["path_waypoints"] = [
        {
            key: value
            for key, value in waypoint.items()
            if key not in {"embodied_feasible", "embodied_clearance_margin_m"}
        }
        for waypoint in waypoints
        if isinstance(waypoint, dict)
    ]
    return safe_packet


def user_prompt(packet: dict[str, Any], *, compact_planning_prompt: bool = False) -> str:
    return structured_user_prompt(packet, compact_planning_prompt=compact_planning_prompt)


def traversability_user_prompt(packet: dict[str, Any], *, compact_planning_prompt: bool = False) -> str:
    prompt_packet = gpt_fact_packet(packet, compact_planning_prompt=compact_planning_prompt)
    task = str(prompt_packet["task"])
    if task not in {"a1", "b1"}:
        return structured_user_prompt(packet, compact_planning_prompt=compact_planning_prompt)
    final_json = json.dumps(prompt_packet["required_final_json"], ensure_ascii=True)
    facts = prompt_packet.get("facts") if isinstance(prompt_packet.get("facts"), dict) else {}
    candidate_space = facts.get("candidate_space") if isinstance(facts.get("candidate_space"), dict) else {}
    all_ids = candidate_space.get("all_display_ids") or []
    if task == "a1":
        positive_label = "Pointmass-walkable IDs"
        negative_label = "Pointmass-blocked IDs"
        positive_ids = candidate_space.get("pointmass_walkable_ids") or []
        negative_ids = candidate_space.get("non_walkable_ids") or []
        task_focus = (
            "This is a point-agent traversability task: inspect all visible candidate IDs, "
            "separate the pointmass-walkable IDs from the blocked IDs, then keep exactly the walkable list."
        )
    else:
        positive_label = "Robot-body feasible IDs"
        negative_label = "Robot-body too-tight IDs"
        positive_ids = candidate_space.get("embodied_feasible_ids") or []
        negative_ids = candidate_space.get("not_embodied_feasible_ids") or []
        robot = facts.get("robot_constraint") if isinstance(facts.get("robot_constraint"), dict) else {}
        task_focus = (
            "This is a robot-body traversability task: inspect all visible candidate IDs, "
            f"apply the robot diameter {oracle.format_distance_m(robot.get('robot_diameter_m'))}, "
            "separate embodied-feasible IDs from too-tight IDs, then keep exactly the feasible list."
        )
    lines = [
        "Write one flowing first-person reasoning paragraph in <think> tags.",
        "Do not use headings, bullets, numbered steps, markdown, JSON envelopes, or the label 'Final JSON'.",
        "The paragraph may describe the visible candidate set as a continuous range when it has no gaps.",
        "The walkable/blocked or feasible/too-tight subset lists must stay exact bracketed lists.",
        "Output format:",
        "<think>",
        "reasoning paragraph",
        "</think>",
        final_json,
        "The final JSON list must appear on its own line after </think>, not inside the think block.",
        f"Task: {task}",
        f"Final JSON: {final_json}",
        f"Thinking route: {task_focus}",
        f"Visible candidate IDs: {oracle.json_int_list(all_ids)}.",
        f"{positive_label}: {oracle.json_int_list(positive_ids)}.",
        f"{negative_label}: {oracle.json_int_list(negative_ids)}.",
        f"Reasoning contract: {'; '.join(task_requirements(task))}",
    ]
    return "\n".join(lines)


def planning_user_prompt(
    packet: dict[str, Any],
    *,
    compact_planning_prompt: bool = False,
    minimal_planning_prompt: bool = False,
    plain_planning_prompt: bool = False,
) -> str:
    if str(packet.get("task")) in {"a1", "b1"}:
        return traversability_user_prompt(packet, compact_planning_prompt=compact_planning_prompt)
    if minimal_planning_prompt:
        return minimal_planning_user_prompt(packet, compact_planning_prompt=compact_planning_prompt)
    if plain_planning_prompt:
        return plain_planning_user_prompt(packet, compact_planning_prompt=compact_planning_prompt)
    return plain_planning_user_prompt(packet, compact_planning_prompt=compact_planning_prompt)


def minimal_planning_user_prompt(packet: dict[str, Any], *, compact_planning_prompt: bool = False) -> str:
    prompt_packet = gpt_fact_packet(packet, compact_planning_prompt=compact_planning_prompt)
    task = str(prompt_packet["task"])
    if task not in {"a2", "b2", "c"}:
        return structured_user_prompt(packet, compact_planning_prompt=compact_planning_prompt)
    final_json = json.dumps(prompt_packet["required_final_json"], ensure_ascii=True)
    task_focus = planning_flow_instruction(task)
    lines = [
        "Write one flowing first-person reasoning paragraph inside <think> tags.",
        "Do not use headings, bullets, numbered steps, or labels such as 'task analysis' or 'target judgment'.",
        "Sound like a person thinking through the image, not a summary of fields.",
        "Use the evidence as visual observations already extracted from the image; do not mention metadata, labels, oracle, ground truth, or supplied facts.",
        "Do not copy the evidence mechanically. Turn it into a causal chain with comparisons, exclusions, route tracing, and shortest-route verification.",
        f"Writing goal: {task_focus}",
        "Output format:",
        "<think>",
        "reasoning paragraph",
        "</think>",
        final_json,
        "The final JSON list must appear on its own line after </think>, not inside the think block.",
        "Do not write a JSON envelope, markdown, metadata, hashes, file paths, world coordinates, or the label 'Final JSON'.",
        f"Task: {task}",
        f"Required final answer line: {final_json}",
        "Evidence:",
    ]
    lines.extend(planning_context_lines(prompt_packet))
    lines.append(f"Reasoning contract: {'; '.join(planning_rationale_contract(task))}")
    return "\n".join(lines)


def plain_planning_user_prompt(packet: dict[str, Any], *, compact_planning_prompt: bool = False) -> str:
    prompt_packet = gpt_fact_packet(packet, compact_planning_prompt=compact_planning_prompt)
    task = str(prompt_packet["task"])
    if task not in {"a2", "b2", "c"}:
        return minimal_planning_user_prompt(packet, compact_planning_prompt=compact_planning_prompt)
    final_json = json.dumps(prompt_packet["required_final_json"], ensure_ascii=True)
    task_focus = planning_flow_instruction(task)
    lines = [
        "Write one flowing first-person reasoning paragraph in <think> tags.",
        "Do not use headings, bullets, numbered steps, or labels such as 'task analysis' or 'target judgment'.",
        "Make the reasoning sound like a person inspecting the scene, comparing visible options, rejecting bad choices, tracing the route, and checking shortestness before concluding.",
        "Use the evidence below as visual observations already extracted from the image; do not mention metadata, labels, oracle, ground truth, or supplied facts.",
        "Do not copy the evidence mechanically or announce the final path first. Convert the evidence into a causal navigation rationale.",
        f"Writing goal: {task_focus}",
        "Output format:",
        "<think>",
        "reasoning paragraph",
        "</think>",
        final_json,
        "The final JSON list must appear on its own line after </think>, not inside the think block.",
        f"Task: {task}",
        f"Required final answer line: {final_json}",
        "Evidence:",
    ]
    lines.extend(planning_context_lines(prompt_packet))
    lines.append(f"Reasoning contract: {'; '.join(planning_rationale_contract(task))}")
    return "\n".join(lines)


def structured_user_prompt(packet: dict[str, Any], *, compact_planning_prompt: bool = False) -> str:
    final_json = packet["required_final_json"]
    payload = {
        "task": packet["task"],
        "question_id": packet["question_id"],
        "writing_contract": {
            "format": "plain_text",
            "length": "3 to 7 short sentences inside the think block plus the final answer line",
            "required_shape": (
                "<think>\\n"
                "natural-language rationale\\n"
                "</think>\\n"
                f"{json.dumps(final_json, ensure_ascii=True)}"
            ),
            "must_end_with": json.dumps(final_json, ensure_ascii=True),
            "final_answer_line": "The last line must be exactly the JSON list, outside the think block, with no label or prefix.",
            "forbidden": [
                "JSON envelope",
                "markdown code block",
                "Final JSON label",
                "ground truth says",
                "metadata says",
                "hashes",
                "local file paths",
                "world coordinates",
            ],
        },
        "task_requirements": task_requirements(str(packet["task"])),
        "fact_packet": gpt_fact_packet(packet, compact_planning_prompt=compact_planning_prompt),
    }
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)


def build_request_headers(api_key: str, *, codex_request_headers: bool) -> dict[str, str]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if not codex_request_headers:
        return headers

    session_id = str(uuid.uuid4())
    turn_id = str(uuid.uuid4())
    metadata = {
        "session_id": session_id,
        "thread_id": session_id,
        "thread_source": "user",
        "turn_id": turn_id,
        "sandbox": "seccomp",
        "turn_started_at_unix_ms": int(time.time() * 1000),
    }
    headers.update(
        {
            "Accept": "text/event-stream",
            "originator": "codex_exec",
            "user-agent": CODEX_USER_AGENT,
            "x-codex-beta-features": "terminal_resize_reflow",
            "x-codex-turn-metadata": json.dumps(metadata, separators=(",", ":")),
            "x-codex-window-id": f"{session_id}:0",
            "x-client-request-id": session_id,
            "session-id": session_id,
            "thread-id": session_id,
        }
    )
    return headers


def request_text_completion(
    *,
    api_base: str,
    api_key: str,
    model: str,
    packet: dict[str, Any],
    max_completion_tokens: int,
    timeout_s: float,
    temperature: float,
    top_p: float,
    api_endpoint: str,
    compact_planning_prompt: bool,
    minimal_planning_prompt: bool,
    plain_planning_prompt: bool,
    codex_request_headers: bool,
) -> dict[str, Any]:
    if api_endpoint == "responses":
        return responses_text_request(
            api_base=api_base,
            api_key=api_key,
            model=model,
            packet=packet,
            max_completion_tokens=max_completion_tokens,
            timeout_s=timeout_s,
            temperature=temperature,
            top_p=top_p,
            compact_planning_prompt=compact_planning_prompt,
            minimal_planning_prompt=minimal_planning_prompt,
            plain_planning_prompt=plain_planning_prompt,
            codex_request_headers=codex_request_headers,
        )
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt()},
            {
                "role": "user",
                "content": planning_user_prompt(
                    packet,
                    compact_planning_prompt=compact_planning_prompt,
                    minimal_planning_prompt=minimal_planning_prompt,
                    plain_planning_prompt=plain_planning_prompt,
                ),
            },
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_completion_tokens": max_completion_tokens,
        "store": False,
    }
    data = json.dumps(body, ensure_ascii=True).encode("utf-8")
    request = urllib.request.Request(
        f"{api_base}/chat/completions",
        data=data,
        headers=build_request_headers(api_key, codex_request_headers=codex_request_headers),
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw)


def responses_text_request(
    *,
    api_base: str,
    api_key: str,
    model: str,
    packet: dict[str, Any],
    max_completion_tokens: int,
    timeout_s: float,
    temperature: float,
    top_p: float,
    compact_planning_prompt: bool,
    minimal_planning_prompt: bool,
    plain_planning_prompt: bool,
    codex_request_headers: bool,
) -> dict[str, Any]:
    body = {
        "model": model,
        "instructions": system_prompt(),
        "input": planning_user_prompt(
            packet,
            compact_planning_prompt=compact_planning_prompt,
            minimal_planning_prompt=minimal_planning_prompt,
            plain_planning_prompt=plain_planning_prompt,
        ),
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_completion_tokens,
        "store": False,
    }
    data = json.dumps(body, ensure_ascii=True).encode("utf-8")
    request = urllib.request.Request(
        f"{api_base}/responses",
        data=data,
        headers=build_request_headers(api_key, codex_request_headers=codex_request_headers),
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        raw = response.read().decode("utf-8")
    return json.loads(raw)


def response_text(response: dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()
    output = response.get("output")
    if isinstance(output, list):
        parts = []
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for child in content:
                    if isinstance(child, dict):
                        if isinstance(child.get("text"), str):
                            parts.append(child["text"])
                        elif isinstance(child.get("content"), str):
                            parts.append(child["content"])
        text = "".join(parts).strip()
        if text:
            return text
    choices = response.get("choices") or []
    if not choices:
        raise oracle.OracleCotError("empty choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [item.get("text", "") for item in content if isinstance(item, dict)]
        return "".join(parts).strip()
    raise oracle.OracleCotError("missing text content")


def parse_final_json(text: str) -> list[int]:
    cleaned = text.strip()
    match = re.search(r"(\[[^\n]*\])\s*$", cleaned)
    candidate = match.group(1) if match else None
    if candidate is None:
        start = cleaned.rfind("[")
        end = cleaned.rfind("]")
        if start < 0 or end < start:
            raise oracle.OracleCotError("missing final JSON list line")
        candidate = cleaned[start : end + 1]
    parsed = json.loads(candidate)
    if not isinstance(parsed, list):
        raise oracle.OracleCotError("Final JSON is not a list")
    return [int(item) for item in parsed]


def ordered_phrase_groups(text: str, groups: list[tuple[str, ...]]) -> bool:
    cursor = 0
    for group in groups:
        best_index: int | None = None
        best_phrase = ""
        for phrase in group:
            index = text.find(phrase, cursor)
            if index < 0:
                continue
            if best_index is None or index < best_index:
                best_index = index
                best_phrase = phrase
        if best_index is None:
            return False
        cursor = best_index + max(len(best_phrase), 1)
    return True


def ids_are_continuous_from_one(ids: list[Any]) -> bool:
    try:
        normalized = [int(item) for item in ids]
    except (TypeError, ValueError):
        return False
    return bool(normalized) and normalized == list(range(1, max(normalized) + 1))


def mentions_complete_visible_candidate_set(text: str, ids: list[Any]) -> bool:
    if json.dumps(ids, ensure_ascii=True) in text:
        return True
    if not ids_are_continuous_from_one(ids):
        return False
    end = int(ids[-1])
    range_patterns = [
        rf"\b1\s+through\s+{end}\b",
        rf"\b1\s+to\s+{end}\b",
        rf"\b1\s*-\s*{end}\b",
        rf"\bfrom\s+1\s+through\s+{end}\b",
        rf"\bfrom\s+1\s+to\s+{end}\b",
    ]
    return any(re.search(pattern, text) for pattern in range_patterns)


def contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def mentions_route_segment(text: str) -> bool:
    if contains_any(
        text,
        (
            "segment",
            "segments",
            "edge",
            "edges",
            "connection",
            "connections",
            "link",
            "links",
            "adjacent",
            "between each",
            "point-to-point",
            "point to point",
            "move from",
            "straight from",
        ),
    ):
        return True
    return bool(re.search(r"\b\d+\s*(?:->|to|-)\s*\d+\b", text))


def mentions_collision_free_segment(text: str) -> bool:
    if not mentions_route_segment(text):
        return False
    safety_patterns = (
        r"collision[- ]free",
        r"without collision",
        r"does not collide",
        r"do not collide",
        r"no collision",
        r"unobstructed",
        r"\bclear\s+(?:segment|segments|edge|edges|connection|connections|link|links|path|route|line|trace)\b",
        r"\b(?:segment|segments|edge|edges|connection|connections|link|links|path|route|line|trace)\s+"
        r"(?:is|are|stays|stay|remains|remain|looks|look|also)?\s*(?:clear|safe|open)\b",
        r"\b(?:segment|segments|edge|edges|connection|connections|link|links)\b[^.;]{0,80}\b"
        r"(?:is|are|stays|stay|remains|remain|looks|look)?\s*(?:clear|safe|open)\b",
        r"\bdense(?:r)?\s+(?:path|trace|check)\s+(?:is\s+|also\s+)?clear\b",
        r"\bdense(?:r)?\s+(?:checked\s+)?trace\b[^.;]{0,80}\bclear\b",
        r"\bprojected\s+(?:path|body path|route)\s+(?:is\s+|also\s+)?clear\b",
    )
    return any(re.search(pattern, text) for pattern in safety_patterns)


def mentions_same_type_candidate_resolution(text: str) -> bool:
    return contains_any(
        text,
        (
            "same-type",
            "same type",
            "candidate target",
            "candidate targets",
            "candidate target count",
            "candidate ids",
            "candidate object",
            "candidate objects",
            "candidate",
            "only one",
            "unique",
            "target to choose",
            "target to choose from",
            "object to choose",
            "object to choose from",
            "competing",
            "multiple",
            "option",
            "options",
            "sink-like target",
            "chair-like target",
            "table-like target",
        ),
    )


def require_planning_elements(issues: list[str], lowered: str, task: str) -> None:
    if not contains_any(
        lowered,
        (
            "target",
            "goal display",
            "goal id",
            "nearest",
            "winner",
            "same-type",
            "same type",
            "candidate",
            "request",
            "intent",
        ),
    ):
        issues.append("missing_target_resolution_element")
    if not contains_any(
        lowered,
        (
            "usable",
            "walkable",
            "blocked",
            "too tight",
            "clearance",
            "open",
            "fit",
            "passable",
            "feasible",
        ),
    ):
        issues.append("missing_waypoint_feasibility_element")
    if not mentions_collision_free_segment(lowered):
        issues.append("missing_segment_collision_free_element")
    if task in {"b2", "c"} and not contains_any(
        lowered,
        (
            "robot",
            "robot body",
            "embodied",
            "clearance",
            "fit",
            "passable",
            "too tight",
        ),
    ):
        issues.append("missing_embodied_feasibility_element")


def validate_cot_text(text: str, packet: dict[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    stripped = text.strip()
    if stripped.startswith("{"):
        issues.append("json_envelope_output")
    if "```" in stripped:
        issues.append("markdown_code_fence")
    if not stripped.startswith("<think>"):
        issues.append("missing_open_think_tag")
    if "</think>" not in stripped:
        issues.append("missing_close_think_tag")
    if stripped.rfind("</think>") > stripped.rfind("["):
        issues.append("final_answer_inside_think")
    if "Final JSON:" in stripped:
        issues.append("final_json_label_present")
    lowered = stripped.lower()
    for pattern in oracle.BANNED_OUTPUT_PATTERNS:
        if pattern in lowered:
            issues.append(f"banned_leakage:{pattern}")
    if "reasoning_payload" in stripped or "question_id" in stripped:
        issues.append("internal_schema_field_leak")

    try:
        final_json = parse_final_json(stripped)
    except Exception as exc:
        final_json = []
        issues.append(f"final_json_parse_error:{str(exc)[:80]}")
    expected = [int(item) for item in packet["required_final_json"]]
    if final_json != expected:
        issues.append("final_json_mismatch")

    task = str(packet["task"])

    def require_any(issue: str, *phrases: str) -> None:
        if not any(phrase in lowered for phrase in phrases):
            issues.append(issue)

    if task in {"a1", "b1"}:
        candidate_space = ((packet.get("facts") or {}).get("candidate_space") or {}) if isinstance(packet.get("facts"), dict) else {}
        all_ids = candidate_space.get("all_display_ids") or []
        if not mentions_complete_visible_candidate_set(lowered, all_ids):
            issues.append("missing_visible_candidate_list")
        if task == "a1":
            positive_ids = candidate_space.get("pointmass_walkable_ids") or []
            negative_ids = candidate_space.get("non_walkable_ids") or []
            require_any("missing_pointmass_language", "pointmass", "point agent")
            require_any("missing_walkability_language", "walkable", "blocked")
            if positive_ids and negative_ids:
                if json.dumps(positive_ids, ensure_ascii=True) not in stripped:
                    issues.append("missing_complete_walkable_split")
                if json.dumps(negative_ids, ensure_ascii=True) not in stripped:
                    issues.append("missing_complete_blocked_split")
            elif positive_ids and not any(phrase in lowered for phrase in ("all of them are walkable", "all of them are point-agent", "so there are no blocked ids", "blocked ids are []")):
                issues.append("missing_complete_walkable_split")
            elif negative_ids and not any(phrase in lowered for phrase in ("all of them are blocked", "so there are no walkable ids", "walkable ids are []")):
                issues.append("missing_complete_walkable_split")
        else:
            positive_ids = candidate_space.get("embodied_feasible_ids") or []
            negative_ids = candidate_space.get("not_embodied_feasible_ids") or []
            require_any("missing_robot_language", "robot", "robot body")
            require_any("missing_embodied_language", "embodied", "embodied feasible", "embodied-feasible", "clearance")
            require_any("missing_clearance_language", "clearance", "too tight", "leave enough room")
            if positive_ids and negative_ids:
                if json.dumps(positive_ids, ensure_ascii=True) not in stripped:
                    issues.append("missing_complete_embodied_feasible_split")
                if json.dumps(negative_ids, ensure_ascii=True) not in stripped:
                    issues.append("missing_complete_too_tight_split")
            elif positive_ids and not any(phrase in lowered for phrase in ("all of them leave enough room", "so there are no too-tight ids", "too-tight ids are []", "too tight ids are []")):
                issues.append("missing_complete_embodied_feasible_split")
            elif negative_ids and not any(phrase in lowered for phrase in ("all of them are too tight", "so there are no embodied-feasible ids", "embodied-feasible ids are []", "embodied feasible ids are []")):
                issues.append("missing_complete_embodied_feasible_split")

    if any(
        phrase in lowered
        for phrase in (
            "task analysis:",
            "target judgment:",
            "candidate feasibility judgment:",
            "path planning:",
            "shortest-path judgment:",
            "shortest embodied path judgment:",
        )
    ):
        issues.append("label_style_not_allowed")

    if task in {"a2", "b2", "c"}:
        require_any("missing_first_person_voice", " i ", "i first", "i inspect", "i scan", "i look", "i read", "i trace", "i check", "i conclude")
        require_any("missing_causal_language", "because", "since", "therefore", "which means", "that means", "this means", "so i", "so the", "so,")
        require_any("missing_visual_analysis_language", "inspect", "scan", "look at", "visible")
        require_any(
            "missing_shortest_path_language",
            "shortest",
            "shortest path",
            "shortest embodied path",
            "shortest route",
            "shortest legal",
            "shortest practical",
            "direct route",
            "direct connection",
            "direct",
            "short segment",
            "single short",
            "no wandering",
        )
        require_any("missing_route_geometry_language", "detour", "turn", "geometry", "straight", "profile")
        if (
            "path length" not in lowered
            and "total length" not in lowered
            and not re.search(r"\b\d+(?:\.\d+)?\s*(?:m|cm|meter|meters|centimeter|centimeters)\b", lowered)
        ):
            issues.append("missing_path_length_language")
        require_any("missing_start_id_language", "start id", "starting from", "starting at", "start ", "starts at", "starts from", "from start", "start at")
        if json.dumps(expected, ensure_ascii=True) not in stripped:
            issues.append("missing_path_list_text")
        require_planning_elements(issues, lowered, task)

    if task == "a2":
        if "robot" in lowered or "embodied" in lowered:
            issues.append("a2_mentions_robot_or_embodied")
        require_any("missing_pointmass_language", "pointmass", "point-mass", "point-agent", "point agent")
        require_any("missing_walkability_language", "walkable", "blocked")
    if task in {"b1", "b2"}:
        require_any("missing_robot_language", "robot", "robot body")
    if task == "b2":
        require_any("missing_embodied_language", "embodied", "embodied feasible")
        require_any("missing_clearance_language", "clearance", "margin", "room", "passage", "gap", "narrow")
        require_any("missing_goal_display_id_language", "goal display", "goal id", "goal ", "display ")
        require_any("missing_blocked_point_language", "blocked", "cannot pass")
    if task == "c":
        require_any("missing_robot_language", "robot", "robot body", "clearance", "embodied")
        require_any("missing_intent_language", "intent", "request", "interpret", "looking for", "place to put", "want to", "need to")
        require_any("missing_cue_language", "cue", "relation", "near", "closest", "only one", "same-type", "same type", "candidate")
        if not mentions_same_type_candidate_resolution(lowered):
            issues.append("missing_same_type_candidate_language")
        require_any("missing_winner_target_id_language", "winner target", "winner target id", "winning target", "unique winning target", "unique winner", "winner is", "only one", "unique")
        require_any("missing_goal_display_id_language", "goal display", "goal id", "goal ", "display ", "at id")
        require_any("missing_embodied_language", "embodied", "embodied feasible", "clearance")
        require_any("missing_clearance_language", "clearance", "margin", "room", "passage", "gap", "narrow")

    return {"ok": not issues, "issues": issues, "final_json": final_json}


def evaluate_generated_text(
    text: str,
    packet: dict[str, Any],
    *,
    skip_validation: bool,
) -> dict[str, Any]:
    if not skip_validation:
        qc = validate_cot_text(text, packet)
        qc["validation_skipped"] = False
        return qc

    issues: list[str] = []
    try:
        final_json = parse_final_json(text.strip())
    except Exception as exc:
        final_json = []
        issues.append(f"final_json_parse_error:{str(exc)[:80]}")
    return {
        "ok": not issues,
        "issues": issues,
        "final_json": final_json,
        "validation_skipped": True,
    }


def selected_packets(args: argparse.Namespace) -> list[dict[str, Any]]:
    tasks = oracle.parse_tasks(args.tasks)
    limit = None if args.max_per_task == 0 else args.max_per_task
    rows = oracle.collect_rows(Path(args.release_root), args.split, tasks, limit, args.seed)
    return [oracle.build_fact_packet(row) for row in rows]


def call_with_retries(args: argparse.Namespace, api_base: str, api_key: str, packet: dict[str, Any]) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(args.max_retries + 1):
        try:
            return request_text_completion(
                api_base=api_base,
                api_key=api_key,
                model=args.model,
                packet=packet,
                max_completion_tokens=args.max_completion_tokens,
                timeout_s=args.timeout_s,
                temperature=args.temperature,
                top_p=args.top_p,
                api_endpoint=args.api_endpoint,
                compact_planning_prompt=args.compact_planning_prompt,
                minimal_planning_prompt=args.minimal_planning_prompt,
                plain_planning_prompt=args.plain_planning_prompt,
                codex_request_headers=args.codex_request_headers,
            )
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, oracle.OracleCotError) as exc:
            last_error = exc
            if attempt >= args.max_retries:
                break
            time.sleep(args.retry_sleep_s + random.random() * args.retry_jitter_s)
    raise oracle.OracleCotError(f"request failed after retries: {last_error}")


def output_path(output_dir: Path, kind: str, task: str, index: int, shard_size: int) -> Path:
    return output_dir / kind / task / f"shard_{index // shard_size:05d}.jsonl"


def report_path(output_dir: Path, *, shard_index: int, shard_count: int) -> Path:
    if shard_count <= 1:
        return output_dir / "qc_report.json"
    return output_dir / "qc_reports" / f"shard_{shard_index:02d}_of_{shard_count:02d}.json"


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.allow_external_upload:
        raise oracle.OracleCotError(
            "refusing to send fact packets without --allow-external-upload and explicit user approval"
        )
    if args.shard_count < 1:
        raise oracle.OracleCotError("--shard-count must be >= 1")
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise oracle.OracleCotError("--shard-index must satisfy 0 <= index < count")
    api_base = oracle.resolve_api_base(args.api_base, args.api_base_env)
    api_key = oracle.resolve_api_key(args.api_key_env)
    output_dir = Path(args.output_dir)
    packets = selected_packets(args)
    accepted_qids = existing_accepted_qids(output_dir) if args.resume else set()
    counts: Counter[str] = Counter()
    issues: Counter[str] = Counter()
    selected_for_shard = 0

    for index, packet in enumerate(packets):
        if index % args.shard_count != args.shard_index:
            continue
        selected_for_shard += 1
        task = str(packet["task"])
        qid = str(packet["question_id"])
        if qid in accepted_qids:
            counts["skipped_accepted"] += 1
            continue
        try:
            response = call_with_retries(args, api_base, api_key, packet)
            text = response_text(response)
            qc = evaluate_generated_text(text, packet, skip_validation=args.skip_validation)
            raw = {
                "question_id": qid,
                "task": task,
                "model": args.model,
                "api_base": api_base,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "usage": response.get("usage"),
                "text": text,
            }
            append_jsonl(output_path(output_dir, "raw_responses", task, index, args.shard_size), raw)
            row = {
                "question_id": qid,
                "task": task,
                "text": text,
                "final_json": qc.get("final_json"),
                "qc": qc,
            }
            if qc["ok"]:
                append_jsonl(output_path(output_dir, "accepted", task, index, args.shard_size), row)
                counts["accepted"] += 1
            else:
                append_jsonl(output_path(output_dir, "rejected", task, index, args.shard_size), row)
                counts["rejected"] += 1
                for issue in qc["issues"]:
                    issues[str(issue)] += 1
        except Exception as exc:
            append_jsonl(
                output_path(output_dir, "rejected", task, index, args.shard_size),
                {
                    "question_id": qid,
                    "task": task,
                    "error": str(exc)[:500],
                    "qc": {"ok": False, "issues": ["api_or_parse_error"]},
                },
            )
            counts["api_or_parse_error"] += 1

        if args.progress_every > 0 and (index + 1) % args.progress_every == 0:
            print(json.dumps({"processed": index + 1, **counts}, ensure_ascii=True, sort_keys=True), flush=True)
        if args.request_sleep_s > 0:
            time.sleep(args.request_sleep_s)

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "model": args.model,
        "api_base": api_base,
        "selected_packets_total": len(packets),
        "selected_packets_shard": selected_for_shard,
        "counts": dict(sorted(counts.items())),
        "issue_counts": dict(sorted(issues.items())),
        "request_config": {
            "api_endpoint": args.api_endpoint,
            "compact_planning_prompt": args.compact_planning_prompt,
            "minimal_planning_prompt": args.minimal_planning_prompt,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_completion_tokens": args.max_completion_tokens,
            "response_format": "plain_text",
            "codex_request_headers": bool(args.codex_request_headers),
            "skip_validation": bool(args.skip_validation),
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
        },
    }
    write_json(report_path(output_dir, shard_index=args.shard_index, shard_count=args.shard_count), report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GPT natural-language oracle CoT text.")
    parser.add_argument("--release-root", default=str(oracle.DEFAULT_RELEASE_ROOT))
    parser.add_argument("--split", default="train")
    parser.add_argument("--tasks", default=",".join(oracle.TASKS))
    parser.add_argument("--max-per-task", type=int, default=3, help="0 means all rows")
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--shard-size", type=int, default=500)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-base-env", default="OPENAI_BASE_URL")
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-endpoint", choices=("chat", "responses"), default="chat")
    parser.add_argument("--codex-request-headers", action="store_true")
    parser.add_argument("--compact-planning-prompt", action="store_true")
    parser.add_argument("--minimal-planning-prompt", action="store_true")
    parser.add_argument("--plain-planning-prompt", action="store_true")
    parser.add_argument("--allow-external-upload", action="store_true")
    parser.add_argument("--max-completion-tokens", type=int, default=1000)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--retry-sleep-s", type=float, default=5.0)
    parser.add_argument("--retry-jitter-s", type=float, default=0.0)
    parser.add_argument("--request-sleep-s", type=float, default=0.0)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--progress-every", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(run(args), ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
