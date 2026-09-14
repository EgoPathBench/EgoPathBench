#!/usr/bin/env python3
"""Visualize per-question results for Next VQA tasks (A1/A2/B1/B2).

Generates one PNG per question with overlays for GT/Pred, plus an index.html.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Optional


TASK_COLORS = {
    "a1": (255, 0, 0),
    "b1": (0, 255, 0),
    "a2": (0, 0, 255),
    "b2": (255, 255, 0),
    "c": (255, 215, 0),
}

AB_STATE_COLORS = {
    "both_walkable": (0, 255, 0),
    "a1_walkable_b1_blocked": (255, 255, 0),
    "both_blocked": (255, 0, 0),
    "invariant_violation": (255, 0, 255),
}


def get_task_colors() -> dict[str, tuple[int, int, int]]:
    return dict(TASK_COLORS)


def load_smoke_artifacts(view_dir: Path) -> dict[str, list[dict]]:
    artifacts: dict[str, list[dict]] = {}
    for task_name in ("a1", "b1", "a2", "b2", "c"):
        path = view_dir / f"visible_waypoints_{task_name}.json"
        if not path.exists():
            continue
        with open(path) as f:
            artifacts[task_name] = list(json.load(f).get("visible_waypoints", []))
    return artifacts


def _infer_scene_and_view(view_dir: Path) -> tuple[str, str]:
    scene_id = view_dir.parent.name or "unknown_scene"
    suffix = view_dir.name
    if suffix.startswith("view_"):
        suffix = suffix.split("view_", 1)[1]
    return scene_id, suffix


def _read_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _load_visible_waypoints(
    view_dir: Path, task_name: str
) -> tuple[dict[int, dict], list[str]]:
    path = view_dir / f"visible_waypoints_{task_name}.json"
    if not path.exists():
        return {}, []
    payload = _read_json(path)
    result: dict[int, dict] = {}
    warnings: list[str] = []
    for waypoint in payload.get("visible_waypoints", []):
        display_id = waypoint.get("display_id")
        if display_id is None:
            warnings.append(
                f"warning: fallback {task_name} candidate missing display_id"
            )
            continue
        result[int(display_id)] = dict(waypoint)
    return result, warnings


def _read_materialized_candidates(task_outputs_path: Path | None) -> list[dict]:
    if task_outputs_path is None or not task_outputs_path.exists():
        return []
    payload = _read_json(task_outputs_path)
    for key in ("ab_candidates", "shared_ab_candidates", "materialized_ab_candidates"):
        if isinstance(payload.get(key), list):
            return [dict(candidate) for candidate in payload[key]]
    return []


def _extract_bool(record: dict[str, Any], *keys: str) -> bool | None:
    for key in keys:
        if key in record:
            value = record[key]
            if value is None:
                return None
            return bool(value)
    return None


def _extract_waypoint_id(record: dict[str, Any]) -> int | None:
    value = record.get("waypoint_id")
    if value is None:
        return None
    return int(value)


def _normalize_fallback_walkability(record: dict[str, Any] | None) -> bool | None:
    if record is None:
        return None
    return _extract_bool(
        record, "is_walkable", "pointmass_walkable", "embodied_feasible"
    )


def _resolve_materialized_semantics(
    raw: dict[str, Any]
) -> tuple[bool | None, bool | None]:
    a1_walkable = _extract_bool(raw, "pointmass_walkable", "a1_is_walkable")
    b1_walkable = _extract_bool(raw, "embodied_feasible", "b1_is_walkable")
    return a1_walkable, b1_walkable


def _resolve_fallback_image_xy(
    display_id: int,
    a1_record: dict[str, Any] | None,
    b1_record: dict[str, Any] | None,
    warnings: list[str],
) -> list[int] | None:
    a1_xy = a1_record.get("image_xy") if a1_record else None
    b1_xy = b1_record.get("image_xy") if b1_record else None
    if isinstance(a1_xy, list) and len(a1_xy) == 2:
        if isinstance(b1_xy, list) and len(b1_xy) == 2 and a1_xy != b1_xy:
            warnings.append(
                f"warning: fallback sources disagree on image_xy for display_id={display_id}"
            )
        return list(a1_xy)
    if isinstance(b1_xy, list) and len(b1_xy) == 2:
        return list(b1_xy)
    return None


def _warn_fallback_disagreement(
    display_id: int,
    a1_record: dict[str, Any] | None,
    b1_record: dict[str, Any] | None,
    warnings: list[str],
) -> None:
    a1_has_display = a1_record is not None and a1_record.get("display_id") is not None
    b1_has_display = b1_record is not None and b1_record.get("display_id") is not None
    if a1_has_display != b1_has_display or (a1_record is None) != (b1_record is None):
        warnings.append(
            f"warning: fallback sources disagree on candidate membership for display_id={display_id}"
        )


def _candidate_from_fallback(
    display_id: int, a1: dict | None, b1: dict | None, warnings: list[str]
) -> dict[str, Any]:
    candidate: dict[str, Any] = {"display_id": display_id}
    source = a1 or b1 or {}
    image_xy = _resolve_fallback_image_xy(display_id, a1, b1, warnings)
    if image_xy is not None:
        candidate["image_xy"] = image_xy
    waypoint_id = _extract_waypoint_id(source)
    if waypoint_id is not None:
        candidate["waypoint_id"] = waypoint_id
    a1_walkable = _normalize_fallback_walkability(a1)
    b1_walkable = _normalize_fallback_walkability(b1)
    if a1_walkable is not None:
        candidate["a1_is_walkable"] = a1_walkable
    if b1_walkable is not None:
        candidate["b1_is_walkable"] = b1_walkable
    return candidate


def _resolve_materialized_candidate(
    raw: dict[str, Any],
    display_id: int,
    a1_record: dict[str, Any] | None,
    b1_record: dict[str, Any] | None,
    warnings: list[str],
) -> dict[str, Any]:
    candidate: dict[str, Any] = {"display_id": display_id}

    waypoint_id = _extract_waypoint_id(raw)
    if waypoint_id is None:
        fallback_waypoint_id = _extract_waypoint_id(a1_record or b1_record or {})
        if fallback_waypoint_id is not None:
            waypoint_id = fallback_waypoint_id
    if waypoint_id is not None:
        candidate["waypoint_id"] = waypoint_id

    if isinstance(raw.get("image_xy"), list) and len(raw["image_xy"]) == 2:
        candidate["image_xy"] = list(raw["image_xy"])
    else:
        fallback_xy = _resolve_fallback_image_xy(
            display_id, a1_record, b1_record, warnings
        )
        if fallback_xy is not None:
            warnings.append(
                f"warning: fallback image_xy used for display_id={display_id}"
            )
            candidate["image_xy"] = fallback_xy
        else:
            warnings.append(f"warning: missing image_xy for display_id={display_id}")

    a1_walkable, b1_walkable = _resolve_materialized_semantics(raw)
    if a1_walkable is None:
        a1_walkable = _normalize_fallback_walkability(a1_record)
        if a1_walkable is not None:
            warnings.append(
                f"warning: fallback a1 semantics used for display_id={display_id}"
            )
    if b1_walkable is None:
        b1_walkable = _normalize_fallback_walkability(b1_record)
        if b1_walkable is not None:
            warnings.append(
                f"warning: fallback b1 semantics used for display_id={display_id}"
            )

    if a1_walkable is not None:
        candidate["a1_is_walkable"] = a1_walkable
    if b1_walkable is not None:
        candidate["b1_is_walkable"] = b1_walkable
    return candidate


def _materialized_has_any_ab_fields(raw: dict[str, Any]) -> bool:
    return any(
        key in raw
        for key in (
            "image_xy",
            "pointmass_walkable",
            "embodied_feasible",
            "a1_is_walkable",
            "b1_is_walkable",
        )
    )


def _resolve_materialized_fallback_records(
    display_id: int, a1_fallback: dict[int, dict], b1_fallback: dict[int, dict]
) -> tuple[dict | None, dict | None]:
    return a1_fallback.get(display_id), b1_fallback.get(display_id)


def _maybe_warn_true_fallback_disagreement(
    materialized: list[dict],
    raw: dict[str, Any],
    display_id: int,
    a1_record: dict[str, Any] | None,
    b1_record: dict[str, Any] | None,
    warnings: list[str],
) -> None:
    if materialized and _materialized_has_any_ab_fields(raw):
        return
    _warn_fallback_disagreement(display_id, a1_record, b1_record, warnings)


def _build_materialized_candidate(
    raw: dict[str, Any],
    display_id: int,
    a1_fallback: dict[int, dict],
    b1_fallback: dict[int, dict],
    warnings: list[str],
    materialized: list[dict],
) -> dict[str, Any]:
    a1_record, b1_record = _resolve_materialized_fallback_records(
        display_id, a1_fallback, b1_fallback
    )
    candidate = _resolve_materialized_candidate(
        raw, display_id, a1_record, b1_record, warnings
    )
    _maybe_warn_true_fallback_disagreement(
        materialized, raw, display_id, a1_record, b1_record, warnings
    )
    return candidate


def _append_partial_unknown_warning(
    candidate: dict[str, Any], warnings: list[str], source: str
) -> None:
    display_id = candidate.get("display_id")
    warnings.append(
        f"warning: partial/unknown {source} AB semantics for display_id={display_id}"
    )


def _record_partial_warning_if_needed(
    candidate: dict[str, Any], warnings: list[str], source: str
) -> None:
    if "a1_is_walkable" not in candidate or "b1_is_walkable" not in candidate:
        _append_partial_unknown_warning(candidate, warnings, source)


def _should_skip_candidate_without_semantics(candidate: dict[str, Any]) -> bool:
    return "a1_is_walkable" not in candidate and "b1_is_walkable" not in candidate


def _should_skip_candidate_without_pixel(candidate: dict[str, Any]) -> bool:
    return not (isinstance(candidate.get("image_xy"), list) and len(candidate["image_xy"]) == 2)


def _fallback_display_ids(a1_fallback: dict[int, dict], b1_fallback: dict[int, dict]) -> set[int]:
    return set(a1_fallback.keys()) | set(b1_fallback.keys())


def _should_use_fallback_entry(
    display_id: int, seen_ids: set[int], materialized: list[dict]
) -> bool:
    return display_id not in seen_ids or not materialized


def _dedupe_warnings(warnings: list[str]) -> list[str]:
    seen: set[str] = set()
    out = []
    for w in warnings:
        if w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out


def _load_fallback_sources(view_dir: Path) -> tuple[dict[int, dict], dict[int, dict], list[str]]:
    warnings: list[str] = []
    a1_fallback, warn_a1 = _load_visible_waypoints(view_dir, "a1")
    b1_fallback, warn_b1 = _load_visible_waypoints(view_dir, "b1")
    warnings.extend(warn_a1)
    warnings.extend(warn_b1)
    return a1_fallback, b1_fallback, warnings


def load_ab_candidates(
    view_dir: Path, task_outputs_path: Path | None
) -> tuple[list[dict], list[str]]:
    a1_fallback, b1_fallback, warnings = _load_fallback_sources(view_dir)
    materialized = _read_materialized_candidates(task_outputs_path)

    candidates: list[dict[str, Any]] = []
    seen_ids: set[int] = set()

    for raw in materialized:
        display_id = raw.get("display_id")
        if display_id is None:
            warnings.append(
                "warning: skipping materialized AB candidate without display_id"
            )
            continue
        display_id = int(display_id)
        seen_ids.add(display_id)

        candidate = _build_materialized_candidate(
            raw, display_id, a1_fallback, b1_fallback, warnings, materialized
        )
        if _should_skip_candidate_without_semantics(candidate):
            warnings.append(
                f"warning: skipping unresolved candidate display_id={display_id}"
            )
            continue
        if _should_skip_candidate_without_pixel(candidate):
            warnings.append(
                f"warning: skipping candidate without pixel display_id={display_id}"
            )
            continue
        _record_partial_warning_if_needed(candidate, warnings, "materialized")
        candidates.append(candidate)

    if not materialized:
        warnings.append(
            "warning: materialized AB candidates unavailable; using fallback sources"
        )

    for display_id in _fallback_display_ids(a1_fallback, b1_fallback):
        if not _should_use_fallback_entry(display_id, seen_ids, materialized):
            continue
        a1_record = a1_fallback.get(display_id)
        b1_record = b1_fallback.get(display_id)
        candidate = _candidate_from_fallback(display_id, a1_record, b1_record, warnings)
        _warn_fallback_disagreement(display_id, a1_record, b1_record, warnings)
        if _should_skip_candidate_without_semantics(candidate):
            warnings.append(
                f"warning: skipping fallback candidate display_id={display_id}"
            )
            continue
        if _should_skip_candidate_without_pixel(candidate):
            warnings.append(
                f"warning: fallback candidate missing image_xy for display_id={display_id}"
            )
            continue
        _record_partial_warning_if_needed(candidate, warnings, "fallback")
        candidates.append(candidate)

    if not candidates:
        warnings.append("warning: no AB candidates resolved")

    candidates.sort(key=lambda item: item.get("display_id", 0))
    return candidates, _dedupe_warnings(warnings)


def _describe_candidate(candidate: dict[str, Any], state: str) -> str:
    waypoint_fragment = ""
    if candidate.get("waypoint_id") is not None:
        waypoint_fragment = f" waypoint_id={candidate['waypoint_id']}"
    return f"candidate display_id={candidate.get('display_id')}{waypoint_fragment}: {state}"


def draw_metadata_block(
    image: Image.Image, title: str, lines: list[str]
) -> Image.Image:
    font = ImageFont.load_default()
    normalized_lines = [title] + list(lines)
    padding = 8
    line_spacing = 4

    measurement_canvas = Image.new("RGB", (1, 1), (255, 255, 255))
    temp_draw = ImageDraw.Draw(measurement_canvas)
    line_heights = []
    text_width = 0
    for line in normalized_lines:
        left, top, right, bottom = temp_draw.textbbox((0, 0), line, font=font)
        text_width = max(text_width, right - left)
        line_heights.append(max(12, bottom - top))

    block_height = max(
        28,
        padding * 2
        + sum(line_heights)
        + line_spacing * max(0, len(normalized_lines) - 1),
    )
    block_width = max(image.width, text_width + padding * 2)
    canvas = Image.new(
        "RGB", (block_width, image.height + block_height), (255, 255, 255)
    )
    canvas.paste(image.convert("RGB"), (0, 0))

    draw = ImageDraw.Draw(canvas)
    top = image.height
    draw.rectangle(
        (0, top, block_width, image.height + block_height),
        fill=(245, 245, 245),
        outline=(180, 180, 180),
    )
    y = top + padding
    for index, line in enumerate(normalized_lines):
        fill = (0, 0, 0)
        if index == 0:
            fill = (20, 20, 20)
        elif (
            "warning" in line.lower()
            or "fallback" in line.lower()
            or "disagree" in line.lower()
            or "missing" in line.lower()
        ):
            fill = (180, 40, 40)
        draw.text((padding, y), line, fill=fill, font=font)
        _, top, _, bottom = temp_draw.textbbox((0, 0), line, font=font)
        y += max(12, bottom - top) + line_spacing

    return canvas


def _warning_footer_lines(warnings: list[str]) -> list[str]:
    if not warnings:
        return []
    return ["warnings:", *warnings]


def _maybe_fill_warning_hint(lines: list[str]) -> list[str]:
    if not lines:
        return []
    return ["legend: warning lines are listed below", *lines]


def _build_ab_metadata_lines(
    view_dir: Path, candidates: list[dict[str, Any]]
) -> list[str]:
    scene_id, view_id = _infer_scene_and_view(view_dir)
    return [
        f"scene_id={scene_id}",
        f"view_id={view_id}",
        f"resolved_candidates={len(candidates)}",
        "legend: labels are display_id values beside each point",
        "legend: green=both_walkable yellow=a1_walkable_b1_blocked red=both_blocked magenta=invariant_violation",
    ]


def classify_ab_state(
    candidate: dict,
) -> tuple[str | None, tuple[int, int, int] | None, list[str]]:
    warnings: list[str] = []
    a1 = candidate.get("a1_is_walkable")
    b1 = candidate.get("b1_is_walkable")
    display_id = candidate.get("display_id")

    if a1 is None or b1 is None:
        warnings.append(
            f"warning: partial/unknown AB semantics for display_id={display_id}"
        )
        return None, None, warnings
    if a1 is False and b1 is True:
        warnings.append(f"warning: invariant violation for display_id={display_id}")
        return "invariant_violation", AB_STATE_COLORS["invariant_violation"], warnings
    if a1 is True and b1 is False:
        return (
            "a1_walkable_b1_blocked",
            AB_STATE_COLORS["a1_walkable_b1_blocked"],
            warnings,
        )
    if a1 is False and b1 is False:
        return "both_blocked", AB_STATE_COLORS["both_blocked"], warnings
    return "both_walkable", AB_STATE_COLORS["both_walkable"], warnings


def _draw_ab_marker(
    draw: ImageDraw.ImageDraw, x: int, y: int, color: tuple[int, int, int], label: str
) -> None:
    r = 3
    draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=(0, 0, 0), width=1)
    draw.text((x + 5, y - 8), label, fill=(0, 0, 0), font=ImageFont.load_default())


def render_ab_semantics_debug(
    view_dir: Path, output_path: Path, task_outputs_path: Path | None = None
) -> Path:
    base_path = view_dir / "rgb.png"
    if not base_path.exists():
        raise FileNotFoundError(f"Missing required base image: {base_path}")

    with Image.open(base_path) as base_image:
        canvas = base_image.convert("RGB").copy()

    draw = ImageDraw.Draw(canvas)
    candidates, warnings = load_ab_candidates(view_dir, task_outputs_path)
    metadata_lines = _build_ab_metadata_lines(view_dir, candidates)

    for candidate in candidates:
        image_xy = candidate.get("image_xy")
        if not isinstance(image_xy, list) or len(image_xy) != 2:
            continue
        x, y = int(image_xy[0]), int(image_xy[1])
        state, color, state_warnings = classify_ab_state(candidate)
        metadata_lines.append(
            _describe_candidate(candidate, state or "partial_unknown")
        )
        warnings.extend(state_warnings)
        if color is not None:
            _draw_ab_marker(draw, x, y, color, str(candidate.get("display_id")))

    metadata_lines.extend(_maybe_fill_warning_hint(_warning_footer_lines(warnings)))
    canvas = draw_metadata_block(canvas, "AB semantics debug", metadata_lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def routing_debug_filename(target_id: int, target_rule: str, single_unit: bool) -> str:
    if single_unit:
        return "routing_semantics_debug.png"
    return f"routing_semantics_debug_t{target_id}_{target_rule}.png"


def _resolve_waypoint_xy(
    waypoint_map: dict[int, dict[str, Any]], display_id: int | None
) -> tuple[int, int] | None:
    if display_id is None:
        return None
    waypoint = waypoint_map.get(int(display_id))
    if waypoint is None:
        return None
    image_xy = waypoint.get("image_xy")
    if not isinstance(image_xy, list) or len(image_xy) != 2:
        return None
    return int(image_xy[0]), int(image_xy[1])


def _draw_routing_marker(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int] | None,
    color: tuple[int, int, int],
    label: str,
) -> None:
    if xy is None:
        return
    x, y = xy
    r = 4
    draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=(0, 0, 0), width=1)
    draw.text((x + 5, y - 8), label, fill=(0, 0, 0), font=ImageFont.load_default())


def _draw_nested_overlap_marker(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int] | None,
    outer_color: tuple[int, int, int],
    inner_color: tuple[int, int, int],
) -> None:
    if xy is None:
        return
    x, y = xy
    draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=outer_color, outline=(0, 0, 0), width=1)
    draw.point((x, y), fill=inner_color)
    draw.point((x + 1, y), fill=outer_color)
    draw.point((x - 1, y), fill=outer_color)
    draw.point((x, y + 1), fill=outer_color)
    draw.point((x, y - 1), fill=outer_color)
    draw.point((x, y), fill=inner_color)


def _draw_overlap_if_needed(
    draw: ImageDraw.ImageDraw,
    a2_xy: tuple[int, int] | None,
    b2_xy: tuple[int, int] | None,
) -> bool:
    if a2_xy is None or b2_xy is None or a2_xy != b2_xy:
        return False
    _draw_nested_overlap_marker(draw, a2_xy, TASK_COLORS["b2"], TASK_COLORS["a2"])
    return True


def _record_waypoint_xy(
    waypoint_map: dict[int, dict[str, Any]], record: dict[str, Any] | None, key: str
) -> tuple[int, int] | None:
    if not isinstance(record, dict):
        return None
    return _resolve_waypoint_xy(waypoint_map, record.get(key))


def _draw_record_markers(
    draw: ImageDraw.ImageDraw,
    waypoint_map: dict[int, dict[str, Any]],
    record: dict[str, Any] | None,
    color: tuple[int, int, int],
    start_label: str,
    goal_label: str,
    skip_start: bool = False,
    skip_goal: bool = False,
) -> None:
    if not isinstance(record, dict):
        return
    if not skip_start:
        _draw_routing_marker(
            draw,
            _resolve_waypoint_xy(waypoint_map, record.get("start_id")),
            color,
            start_label,
        )
    if not skip_goal:
        _draw_routing_marker(
            draw,
            _resolve_waypoint_xy(waypoint_map, record.get("goal_id")),
            color,
            goal_label,
        )


def _draw_routing_path(
    draw: ImageDraw.ImageDraw,
    waypoint_map: dict[int, dict[str, Any]],
    path_ids: list[int],
    color: tuple[int, int, int],
) -> list[int]:
    points = []
    missing = 0
    for display_id in path_ids:
        xy = _resolve_waypoint_xy(waypoint_map, display_id)
        if xy is None:
            missing += 1
            continue
        points.append(xy)
    if len(points) >= 2:
        draw.line(points, fill=color, width=5)
    return [missing, len(path_ids)]


def _build_routing_metadata_lines(view_dir: Path, routing_unit: dict[str, Any]) -> list[str]:
    scene_id, view_id = _infer_scene_and_view(view_dir)
    return [
        f"scene_id={scene_id}",
        f"view_id={view_id}",
        f"target_id={routing_unit.get('target_id')}",
        f"target_rule={routing_unit.get('target_rule')}",
        f"target_category={routing_unit.get('target_category')}",
        "legend: red=target blue=a2 yellow=b2 gray=context",
    ]


def _draw_gray_context_candidates(
    draw: ImageDraw.ImageDraw,
    waypoint_map: dict[int, dict[str, Any]],
    used_ids: set[int],
) -> None:
    for display_id, waypoint in waypoint_map.items():
        if display_id in used_ids:
            continue
        image_xy = waypoint.get("image_xy")
        if not isinstance(image_xy, list) or len(image_xy) != 2:
            continue
        x, y = int(image_xy[0]), int(image_xy[1])
        draw.point((x, y), fill=(128, 128, 128))


def _load_shared_routing_candidates(view_dir: Path) -> tuple[dict[int, dict[str, Any]], list[str]]:
    a2_map, a2_warnings = _load_visible_waypoints(view_dir, "a2")
    if a2_map:
        return a2_map, a2_warnings
    b2_map, b2_warnings = _load_visible_waypoints(view_dir, "b2")
    return b2_map, b2_warnings


def _resolve_target_highlight_base(view_dir: Path, routing_unit: dict[str, Any]) -> tuple[Path, list[str]]:
    warnings: list[str] = []
    target_id = routing_unit.get("target_id")
    base_path = view_dir / "rgb.png"
    if target_id is not None:
        highlight_path = view_dir / f"path_target_{int(target_id)}.png"
        if highlight_path.exists():
            return highlight_path, warnings
    warnings.append("warning: target highlight unavailable; using bbox/rgb")
    return base_path, warnings


def render_routing_semantics_debug(
    view_dir: Path, output_path: Path, routing_unit: dict[str, Any]
) -> Path:
    base_path, highlight_warnings = _resolve_target_highlight_base(view_dir, routing_unit)
    if not base_path.exists():
        raise FileNotFoundError(f"Missing required base image: {base_path}")

    with Image.open(base_path) as base_image:
        canvas = base_image.convert("RGB").copy()

    draw = ImageDraw.Draw(canvas)
    warnings: list[str] = []
    warnings.extend(highlight_warnings)
    metadata_lines = _build_routing_metadata_lines(view_dir, routing_unit)
    shared_waypoints, shared_warnings = _load_shared_routing_candidates(view_dir)
    warnings.extend(shared_warnings)

    bbox = routing_unit.get("target_bbox")
    if isinstance(bbox, list) and len(bbox) == 4:
        left, top, right, bottom = [int(value) for value in bbox]
        draw.rectangle(
            (left, top, right, bottom), fill=(255, 210, 210), outline=(255, 0, 0), width=2
        )
    else:
        warnings.append("warning: target localization unavailable")

    used_ids: set[int] = set()
    a2_record = routing_unit.get("a2_record")
    if isinstance(a2_record, dict):
        used_ids.update(int(value) for value in a2_record.get("path_ids", []))
        if a2_record.get("start_id") is not None:
            used_ids.add(int(a2_record["start_id"]))
        if a2_record.get("goal_id") is not None:
            used_ids.add(int(a2_record["goal_id"]))
        missing, total = _draw_routing_path(draw, shared_waypoints, list(a2_record.get("path_ids", [])), TASK_COLORS["a2"])
        if missing:
            warnings.append(f"warning: A2 path missing {missing}/{total} nodes")
    else:
        warnings.append("warning: missing A2 record")

    b2_record = routing_unit.get("b2_record")
    if isinstance(b2_record, dict):
        used_ids.update(int(value) for value in b2_record.get("path_ids", []))
        if b2_record.get("start_id") is not None:
            used_ids.add(int(b2_record["start_id"]))
        if b2_record.get("goal_id") is not None:
            used_ids.add(int(b2_record["goal_id"]))
        missing, total = _draw_routing_path(draw, shared_waypoints, list(b2_record.get("path_ids", [])), TASK_COLORS["b2"])
        if missing:
            warnings.append(f"warning: B2 path missing {missing}/{total} nodes")
    else:
        warnings.append("warning: missing B2 record")

    a2_start_xy = _record_waypoint_xy(shared_waypoints, a2_record, "start_id")
    a2_goal_xy = _record_waypoint_xy(shared_waypoints, a2_record, "goal_id")
    b2_start_xy = _record_waypoint_xy(shared_waypoints, b2_record, "start_id")
    b2_goal_xy = _record_waypoint_xy(shared_waypoints, b2_record, "goal_id")

    start_overlapped = _draw_overlap_if_needed(draw, a2_start_xy, b2_start_xy)
    goal_overlapped = _draw_overlap_if_needed(draw, a2_goal_xy, b2_goal_xy)

    _draw_record_markers(
        draw,
        shared_waypoints,
        a2_record,
        TASK_COLORS["a2"],
        "A2 start",
        "A2 goal",
        skip_start=start_overlapped,
        skip_goal=goal_overlapped,
    )
    _draw_record_markers(
        draw,
        shared_waypoints,
        b2_record,
        TASK_COLORS["b2"],
        "B2 start",
        "B2 goal",
        skip_start=start_overlapped,
        skip_goal=goal_overlapped,
    )

    _draw_gray_context_candidates(draw, shared_waypoints, used_ids)

    metadata_lines.extend(_maybe_fill_warning_hint(_warning_footer_lines(warnings)))
    canvas = draw_metadata_block(canvas, "Routing semantics debug", metadata_lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def _draw_task_panel(
    base: Image.Image, candidates: list[dict], color: tuple[int, int, int]
) -> Image.Image:
    panel = base.copy()
    draw = ImageDraw.Draw(panel)
    for candidate in candidates:
        image_xy = candidate.get("image_xy")
        if not isinstance(image_xy, list) or len(image_xy) != 2:
            continue
        x, y = int(image_xy[0]), int(image_xy[1])
        r = 4
        draw.ellipse(
            (x - r, y - r, x + r, y + r), fill=color, outline=(0, 0, 0), width=1
        )
    return panel


def render_combined_debug_visualization(view_dir: Path, output_path: Path) -> Path:
    artifacts = load_smoke_artifacts(view_dir)
    panels = []
    for task_name in ("a1", "b1", "a2", "b2", "c"):
        overlay_path = view_dir / f"rgb_overlay_{task_name}.png"
        base_path = overlay_path if overlay_path.exists() else view_dir / "rgb.png"
        if not base_path.exists():
            continue
        with Image.open(base_path) as img:
            panels.append(
                _draw_task_panel(
                    img, artifacts.get(task_name, []), TASK_COLORS[task_name]
                )
            )

    if not panels:
        raise ValueError(f"No visualization panels found in {view_dir}")

    width = max(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    columns = 3 if len(panels) > 4 else 2
    rows = (len(panels) + columns - 1) // columns
    canvas = Image.new("RGB", (width * columns, height * rows), (255, 255, 255))
    for index, panel in enumerate(panels):
        x = (index % columns) * width
        y = (index // columns) * height
        canvas.paste(panel, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path



# Pillow is faster for large batches; fallback to matplotlib if unavailable.
try:
    from PIL import Image, ImageDraw, ImageFont
    PIL_OK = True
except Exception:  # pragma: no cover
    PIL_OK = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
except Exception:  # pragma: no cover
    plt = None
    mpimg = None


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def load_font(size: int, bold: bool = False):
    if not PIL_OK:
        return None
    candidates = []
    if bold:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        ])
    else:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        ])
    for font_path in candidates:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def parse_ids(raw) -> list[int] | None:
    if raw is None:
        return None
    if isinstance(raw, list):
        try:
            return [int(x) for x in raw]
        except Exception:
            return None
    if isinstance(raw, str):
        s = raw.strip()
        s = s.replace("```json", "").replace("```", "").strip()
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [int(x) for x in parsed]
        except Exception:
            return None
    return None


def load_visible(path: Path) -> tuple[list[dict], dict[int, tuple[int, int]]]:
    data = json.loads(path.read_text())
    v = data.get("visible_waypoints", [])
    mapping: dict[int, tuple[int, int]] = {}
    for wp in v:
        try:
            did = int(wp["display_id"])
            xy = wp.get("image_xy")
            if xy is None:
                continue
            mapping[did] = (int(xy[0]), int(xy[1]))
        except Exception:
            continue
    return v, mapping


def build_display_to_waypoint(visible_waypoints: list[dict]) -> dict[int, int]:
    return {int(wp["display_id"]): int(wp["waypoint_id"]) for wp in visible_waypoints}


def load_graph(scene_dir: Path, embodied: bool) -> dict:
    if embodied:
        p = scene_dir / "waypoint_graph_embodied.json"
        if not p.exists():
            p = scene_dir / "waypoint_graph.json"
    else:
        p = scene_dir / "waypoint_graph_pointmass.json"
        if not p.exists():
            p = scene_dir / "waypoint_graph.json"
    return json.loads(p.read_text())


def build_adj(edges: list[dict]) -> dict[int, dict[int, float]]:
    adj: dict[int, dict[int, float]] = {}
    for e in edges:
        u = int(e.get("source"))
        v = int(e.get("target"))
        w = float(e.get("distance", 1.0))
        adj.setdefault(u, {})[v] = w
        adj.setdefault(v, {})[u] = w
    return adj


import heapq


def dijkstra(adj: dict[int, dict[int, float]], start: int, goal: int) -> Optional[float]:
    if start not in adj or goal not in adj:
        return None
    heap = [(0.0, start)]
    dist = {start: 0.0}
    while heap:
        d, u = heapq.heappop(heap)
        if u == goal:
            return d
        if d > dist.get(u, float("inf")):
            continue
        for v, w in adj[u].items():
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(heap, (nd, v))
    return None


def shortest_hops(adj: dict[int, dict[int, float]], start: int, goal: int) -> Optional[int]:
    if start not in adj or goal not in adj:
        return None
    if start == goal:
        return 0
    visited = {start}
    queue = [(start, 0)]
    while queue:
        node, dist = queue.pop(0)
        for nb in adj.get(node, {}):
            if nb in visited:
                continue
            if nb == goal:
                return dist + 1
            visited.add(nb)
            queue.append((nb, dist + 1))
    return None


# --- Drawing helpers ---

def draw_circle(draw, xy, r, color, outline=(0, 0, 0, 255)):
    x, y = xy
    draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=outline, width=1)


def draw_square(draw, xy, r, color, outline=(0, 0, 0, 255)):
    x, y = xy
    draw.rectangle((x - r, y - r, x + r, y + r), fill=color, outline=outline, width=1)


def draw_line(draw, pts, color, width=3):
    if len(pts) < 2:
        return
    draw.line(pts, fill=color, width=width)


def line_height(font) -> int:
    bbox = font.getbbox("Ag")
    return (bbox[3] - bbox[1]) + 4


def wrap_text(text: str,
              width: int = 72,
              font=None,
              max_width_px: float | None = None) -> list[str]:
    words = text.split()
    if not words:
        return [""]
    lines = []
    cur = words[0]
    for word in words[1:]:
        cand = cur + " " + word
        if font is not None and max_width_px is not None:
            fits = font.getlength(cand) <= max_width_px
        else:
            fits = len(cand) <= width
        if fits:
            cur = cand
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    return lines


def text_block(draw, xy, lines, font, fg=(255, 255, 255, 255)):
    x, y = xy
    line_h = line_height(font)
    max_w = max((font.getlength(s) for s in lines), default=0)
    pad = 6
    box = (x, y, x + max_w + pad * 2, y + line_h * len(lines) + pad * 2)
    draw.rectangle(box, fill=(0, 0, 0, 170))
    ty = y + pad
    for s in lines:
        draw.text((x + pad, ty), s, font=font, fill=fg)
        ty += line_h
    return box


def draw_badge(draw, xy, text: str, font, fill, outline=(0, 0, 0, 255), fg=(255, 255, 255, 255)):
    x, y = xy
    pad_x = 10
    pad_y = 5
    bbox = font.getbbox(text)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    box = (x, y, x + tw + pad_x * 2, y + th + pad_y * 2)
    draw.rounded_rectangle(box, radius=8, fill=fill, outline=outline, width=2)
    draw.text((x + pad_x, y + pad_y - bbox[1]), text, font=font, fill=fg)
    return box


def make_footer(base_img, footer_h: int = 280):
    canvas = Image.new("RGB", (base_img.width, base_img.height + footer_h), (18, 18, 18))
    canvas.paste(base_img, (0, 0))
    return canvas


def draw_legend(draw, x: int, y: int, task: str, font, max_width_px: float | None = None):
    lines = ["Legend:"]
    marker_rows = []

    if task in ("a1", "b1"):
        items = [
            ((0, 255, 0, 220), "green", "TP: pred=walkable and GT=walkable"),
            ((255, 0, 0, 220), "red", "FP: pred=walkable but GT=blocked"),
            ((255, 200, 0, 220), "orange", "FN: GT=walkable but model missed"),
        ]
        for color, name, desc in items:
            marker_rows.append((len(lines), color, name))
            lines.extend(wrap_text(
                f"{name}: {desc}",
                font=font,
                max_width_px=max_width_px - 30 if max_width_px else None,
            ))
        text_block(draw, (x + 24, y), lines, font)
        for row, color, _ in marker_rows:
            cy = y + 14 + row * line_height(font)
            draw_circle(draw, (x + 10, cy), 6, color)
        return

    items = [
        ((0, 120, 255, 220), "blue circle", "start waypoint"),
        ((0, 255, 0, 200), "green circle", "GT goal waypoint(s)"),
        ((255, 215, 0, 240), "yellow target", "projected target object center"),
        ((0, 200, 255, 180), "cyan line", "GT path"),
        ((255, 0, 255, 220), "magenta line", "predicted path"),
        ((255, 0, 255, 220), "magenta square", "predicted final node"),
    ]
    for color, name, desc in items:
        marker_rows.append((len(lines), color, name))
        lines.extend(wrap_text(
            f"{name}: {desc}",
            font=font,
            max_width_px=max_width_px - 30 if max_width_px else None,
        ))
    lines.extend(wrap_text(
        "red underlay on base image: target object / render path cue",
        font=font,
        max_width_px=max_width_px - 30 if max_width_px else None,
    ))
    text_block(draw, (x + 24, y), lines, font)
    for row, color, name in marker_rows:
        cy = y + 14 + row * line_height(font)
        if "square" in name:
            draw_square(draw, (x + 10, cy), 6, color)
        elif "line" in name:
            draw_line(draw, [(x + 4, cy), (x + 16, cy)], color, width=3)
        elif "target" in name:
            draw_target_marker(draw, (x + 10, cy), color)
        else:
            draw_circle(draw, (x + 10, cy), 6, color)


def screen_norm_to_image_xy(screen_x: float, screen_y: float, res_x: int, res_y: int) -> tuple[int, int]:
    px = int(round((screen_x + 1.0) * 0.5 * res_x))
    py = int(round((1.0 - (screen_y + 1.0) * 0.5) * res_y))
    return px, py


def draw_target_marker(draw, xy, color=(255, 215, 0, 240), size: int = 12):
    x, y = xy
    draw.line((x - size, y, x + size, y), fill=color, width=3)
    draw.line((x, y - size, x, y + size), fill=color, width=3)
    draw.ellipse((x - 5, y - 5, x + 5, y + 5), outline=(0, 0, 0, 255), width=2)
    draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=color, outline=(0, 0, 0, 255), width=1)


def draw_target_label(draw, xy, text: str, font):
    x, y = xy
    tx = min(max(12, x + 18), max(12, x - 180))
    ty = max(12, y - 34)
    text_block(draw, (tx, ty), [text], font, fg=(255, 240, 140, 255))
    draw.line((x + 5, y - 5, tx, ty + 12), fill=(255, 215, 0, 240), width=2)


def load_cameras(scene_dir: Path) -> list[dict]:
    cameras_path = scene_dir / "cameras.json"
    if not cameras_path.exists():
        return []
    try:
        return json.loads(cameras_path.read_text())
    except Exception:
        return []


def find_target_projection(cameras: list[dict], view_id: int, target_id: int, res_x: int, res_y: int) -> Optional[tuple[int, int]]:
    if view_id < 0 or view_id >= len(cameras):
        return None
    cam = cameras[view_id]
    for obj in cam.get("visible_objects", []):
        try:
            if int(obj.get("id", -1)) != target_id:
                continue
            sx = float(obj.get("screen_x"))
            sy = float(obj.get("screen_y"))
            return screen_norm_to_image_xy(sx, sy, res_x, res_y)
        except Exception:
            continue
    return None


def estimate_footer_height(prompt_lines: list[str], eval_lines: list[str], task: str, font) -> int:
    if font is None:
        return 280
    legend_items = 5 if task in ("a1", "b1") else 10
    legend_lines = legend_items + 1
    lh = line_height(font)
    left_h = (len(prompt_lines) + len(eval_lines) + 1) * lh
    right_h = legend_lines * lh
    body_h = max(left_h, right_h)
    return max(280, body_h + 32)


def classification_eval_summary(tp: int, fp: int, fn: int) -> tuple[str, tuple[int, int, int, int], list[str]]:
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 1.0
    f1 = f1_score(tp, fp, fn)
    exact = (fp == 0 and fn == 0)
    if exact:
        status = "PASS"
        color = (36, 160, 80, 235)
    elif tp > 0:
        status = "PARTIAL"
        color = (220, 150, 20, 235)
    else:
        status = "FAIL"
        color = (200, 50, 50, 235)
    lines = [
        f"eval={status}",
        f"precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}",
        f"tp={tp} fp={fp} fn={fn}",
    ]
    return status, color, lines


def routing_eval_summary(pred_ids: list[int], metrics: dict) -> tuple[str, tuple[int, int, int, int], list[str]]:
    if metrics.get("success"):
        status = "SUCCESS"
        color = (36, 160, 80, 235)
    elif not pred_ids:
        status = "NO_PRED"
        color = (120, 120, 120, 235)
    elif not metrics.get("valid_ids"):
        status = "INVALID_IDS"
        color = (200, 50, 50, 235)
    elif not metrics.get("valid_edges"):
        status = "INVALID_PATH"
        color = (200, 90, 40, 235)
    else:
        status = "WRONG_GOAL"
        color = (220, 150, 20, 235)
    lines = [
        f"eval={status}",
        f"success={metrics.get('success')} valid_ids={metrics.get('valid_ids')}",
        f"valid_edges={metrics.get('valid_edges')} pred_len={metrics.get('length', 0.0):.2f}",
        f"goal_candidates={len(metrics.get('goal_ids', []))} pred_nodes={len(pred_ids)}",
    ]
    return status, color, lines


# --- Metrics helpers ---

def f1_score(tp: int, fp: int, fn: int) -> float:
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 1.0


def eval_routing(q, pred_ids, all_ids, disp_to_wp, adj, goal_hop=0, goal_dist_m=0.0, weak_connectivity=False):
    if pred_ids is None:
        pred_ids = []
    pred_ids = [int(x) for x in pred_ids]

    valid_ids = True
    if not pred_ids:
        valid_ids = False
    else:
        if any(pid not in all_ids for pid in pred_ids):
            valid_ids = False

    # Edge validity + length
    valid_edges = False
    length = 0.0
    if valid_ids and len(pred_ids) >= 2:
        try:
            wp_seq = [disp_to_wp[int(pid)] for pid in pred_ids]
            ok = True
            if weak_connectivity:
                for a, b in zip(wp_seq[:-1], wp_seq[1:]):
                    dist = dijkstra(adj, a, b)
                    if dist is None:
                        ok = False
                        break
                    length += dist
            else:
                for a, b in zip(wp_seq[:-1], wp_seq[1:]):
                    if a not in adj or b not in adj[a]:
                        ok = False
                        break
                    length += adj[a][b]
            valid_edges = ok
        except Exception:
            valid_edges = False

    gt = q.get("ground_truth", {})
    goal_ids = gt.get("goal_ids") or ([] if gt.get("goal_id") is None else [gt.get("goal_id")])
    goal_ids = [int(x) for x in goal_ids if x is not None]

    success = False
    if valid_ids and goal_ids:
        pred_goal_id = int(pred_ids[-1])
        success = pred_goal_id in goal_ids
        if not success and (goal_hop > 0 or goal_dist_m > 0.0):
            try:
                pred_wp = disp_to_wp[pred_goal_id]
            except Exception:
                pred_wp = None
            if pred_wp is not None:
                for gid in goal_ids:
                    try:
                        goal_wp = disp_to_wp[int(gid)]
                    except Exception:
                        continue
                    if goal_hop > 0:
                        hops = shortest_hops(adj, pred_wp, goal_wp)
                        if hops is not None and hops <= goal_hop:
                            success = True
                            break
                    if not success and goal_dist_m > 0.0:
                        dist = dijkstra(adj, pred_wp, goal_wp)
                        if dist is not None and dist <= goal_dist_m:
                            success = True
                            break

    return {
        "valid_ids": valid_ids,
        "valid_edges": valid_edges,
        "success": success,
        "length": length,
        "goal_ids": goal_ids,
    }


def load_image(path: Path):
    if PIL_OK:
        return Image.open(path).convert("RGB")
    if mpimg is None:
        raise RuntimeError("Neither PIL nor matplotlib is available")
    img = mpimg.imread(str(path))
    if img.dtype != "uint8":
        img = (img * 255).astype("uint8")
    return Image.fromarray(img)


def save_image(img, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def resolve_routing_base_image(img_path: Path, q: dict) -> Path:
    target_id = q.get("ground_truth", {}).get("target_id")
    try:
        target_id_int = int(target_id) if target_id is not None else None
    except (TypeError, ValueError):
        target_id_int = None
    if target_id_int is not None:
        target_path = img_path.parent / f"path_target_{target_id_int}.png"
        if target_path.exists():
            return target_path
    return img_path


def resolve_question_image_path(q: dict) -> Path:
    img_path = Path(q.get("image_path", ""))
    if q.get("task") in ("a2", "b2"):
        return resolve_routing_base_image(img_path, q)
    return img_path


def load_base_image_for_question(q: dict, task: str) -> Image.Image:
    return load_image(resolve_question_image_path(q))


def main():
    parser = argparse.ArgumentParser(description="Visualize per-question results for Next VQA tasks")
    parser.add_argument("--vqa", required=True, help="vqa_next_*.jsonl")
    parser.add_argument("--predictions", required=True, help="predictions jsonl")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--gt", default=None, help="optional gt_next_*.jsonl for routing path overlay")
    parser.add_argument("--restrict-to-predictions", action="store_true",
                        help="Only visualize questions whose question_id appears in predictions. "
                             "Useful for smoke / partial benchmark runs.")
    parser.add_argument("--limit", type=int, default=0, help="limit number of questions (0=all)")
    parser.add_argument("--skip-existing", action="store_true", help="skip existing images")
    parser.add_argument("--goal-hop", type=int, default=0)
    parser.add_argument("--goal-dist-m", type=float, default=0.0)
    parser.add_argument("--weak-connectivity", action="store_true")
    parser.add_argument("--no-index", action="store_true", help="do not write index.html")
    args = parser.parse_args()

    vqa = load_jsonl(Path(args.vqa))
    preds_list = load_jsonl(Path(args.predictions))
    preds = {p.get("question_id"): p for p in preds_list}
    pred_qids = {qid for qid in preds.keys() if qid}

    if args.restrict_to_predictions:
        vqa = [q for q in vqa if q.get("question_id") in pred_qids]

    gt_map = {}
    if args.gt:
        gt_list = load_jsonl(Path(args.gt))
        if args.restrict_to_predictions:
            gt_list = [g for g in gt_list if g.get("question_id") in pred_qids]
        gt_map = {g.get("question_id"): g for g in gt_list}

    out_dir = Path(args.output)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    font = load_font(18) if PIL_OK else None
    font_small = load_font(16) if PIL_OK else None
    font_title = load_font(19, bold=True) if PIL_OK else None

    # Caches
    graph_cache: dict[tuple[str, bool], dict] = {}
    adj_cache: dict[tuple[str, bool], dict] = {}
    camera_cache: dict[str, list[dict]] = {}

    results = []

    for i, q in enumerate(vqa):
        if args.limit and i >= args.limit:
            break
        qid = q.get("question_id")
        if not qid:
            continue
        out_path = img_dir / f"{qid}.png"
        if args.skip_existing and out_path.exists():
            continue

        img_path = resolve_question_image_path(q)
        if not img_path.exists():
            continue

        vis_path = Path(q.get("visible_waypoints_path", ""))
        if not vis_path.exists():
            continue
        visible, id_to_xy = load_visible(vis_path)
        all_ids = set(id_to_xy.keys())
        disp_to_wp = build_display_to_waypoint(visible)

        task = q.get("task")
        gt = q.get("ground_truth", {})

        try:
            base_img = load_base_image_for_question(q, task=task)
        except Exception:
            continue
        prompt_width_px = max(320, base_img.width * 0.45)
        eval_lines = ["Evaluation Result:", "pending"]
        prompt_lines = [
            "System Prompt:",
            *wrap_text(q.get("system_prompt", ""), font=font, max_width_px=prompt_width_px),
            "",
            "User Prompt:",
            *wrap_text(q.get("user_prompt", ""), font=font, max_width_px=prompt_width_px),
        ]

        pred = preds.get(qid, {})
        pred_ids = parse_ids(pred.get("output")) or []
        pred_ids = [int(x) for x in pred_ids if isinstance(x, int) or str(x).isdigit()]
        info_lines = [f"{qid}", f"task={task}"]
        badge_text = "Evaluation: UNSUPPORTED"
        badge_color = (120, 120, 120, 235)
        render_data: dict = {"task": task}

        if task in ("a1", "b1"):
            gt_pos = set(int(x) for x in (gt.get("answer") or gt.get("walkable_ids") or []))
            pred_set = set(pred_ids) & all_ids
            tp = pred_set & gt_pos
            fp = pred_set - gt_pos
            fn = gt_pos - pred_set
            f1 = f1_score(len(tp), len(fp), len(fn))
            eval_status, eval_color, eval_detail_lines = classification_eval_summary(len(tp), len(fp), len(fn))
            eval_lines = ["Evaluation Result:", *eval_detail_lines]
            info_lines = [
                f"{qid}",
                f"task={task}",
                f"tp={len(tp)} fp={len(fp)} fn={len(fn)} f1={f1:.3f}",
                f"pred={len(pred_set)} gt={len(gt_pos)}",
            ]
            badge_text = f"Evaluation: {eval_status}"
            badge_color = eval_color
            render_data.update({
                "tp": tp,
                "fp": fp,
                "fn": fn,
            })

            results.append({
                "question_id": qid,
                "task": task,
                "tp": len(tp),
                "fp": len(fp),
                "fn": len(fn),
                "f1": round(f1, 4),
                "evaluation_status": eval_status,
            })

        elif task in ("a2", "b2"):
            scene_dir = img_path.parent.parent
            embodied = (task == "b2")
            key = (str(scene_dir), embodied)
            if key not in graph_cache:
                graph_cache[key] = load_graph(scene_dir, embodied)
                adj_cache[key] = build_adj(graph_cache[key].get("edges", []))
            adj = adj_cache[key]

            gt_rec = gt_map.get(qid)
            start_id = gt.get("start_id")
            goal_ids = gt.get("goal_ids") or ([] if gt.get("goal_id") is None else [gt.get("goal_id")])
            goal_ids = [int(x) for x in goal_ids if x is not None]
            metrics = eval_routing(
                q,
                pred_ids,
                all_ids,
                disp_to_wp,
                adj,
                goal_hop=args.goal_hop,
                goal_dist_m=args.goal_dist_m,
                weak_connectivity=args.weak_connectivity,
            )
            eval_status, eval_color, eval_detail_lines = routing_eval_summary(pred_ids, metrics)
            eval_lines = ["Evaluation Result:", *eval_detail_lines]
            info_lines = [
                f"{qid}",
                f"task={task}",
                f"valid_ids={metrics['valid_ids']} valid_edges={metrics['valid_edges']}",
                f"success={metrics['success']} pred_len={metrics['length']:.2f}",
                f"pred_nodes={len(pred_ids)} goals={len(metrics['goal_ids'])}",
                f"target={gt.get('target_category', 'object')}#{gt.get('target_id')}",
            ]
            badge_text = f"Evaluation: {eval_status}"
            badge_color = eval_color
            if str(scene_dir) not in camera_cache:
                camera_cache[str(scene_dir)] = load_cameras(scene_dir)
            target_id = gt.get("target_id")
            view_id = int(q.get("view_id", -1))
            target_xy = None
            if target_id is not None:
                target_xy = find_target_projection(
                    camera_cache[str(scene_dir)],
                    view_id=view_id,
                    target_id=int(target_id),
                    res_x=base_img.width,
                    res_y=base_img.height,
                )
            render_data.update({
                "gt_rec": gt_rec,
                "pred_ids": pred_ids,
                "start_id": start_id,
                "goal_ids": goal_ids,
                "target_id": target_id,
                "target_xy": target_xy,
            })

            results.append({
                "question_id": qid,
                "task": task,
                **metrics,
                "pred_nodes": len(pred_ids),
                "evaluation_status": eval_status,
            })

        else:
            eval_lines = ["Evaluation Result:", "eval=UNSUPPORTED"]
            info_lines = [f"{qid}", f"task={task}", "unsupported"]
            results.append({"question_id": qid, "task": task, "note": "unsupported", "evaluation_status": "UNSUPPORTED"})

        footer_h = estimate_footer_height(prompt_lines, eval_lines, task, font)
        img = make_footer(base_img, footer_h=footer_h)
        draw = ImageDraw.Draw(img, "RGBA")

        if task in ("a1", "b1"):
            for did in render_data.get("tp", set()):
                xy = id_to_xy.get(did)
                if xy:
                    draw_circle(draw, xy, 6, (0, 255, 0, 180))
            for did in render_data.get("fp", set()):
                xy = id_to_xy.get(did)
                if xy:
                    draw_circle(draw, xy, 6, (255, 0, 0, 180))
            for did in render_data.get("fn", set()):
                xy = id_to_xy.get(did)
                if xy:
                    draw_circle(draw, xy, 6, (255, 200, 0, 180))
        elif task in ("a2", "b2"):
            gt_rec = render_data.get("gt_rec")
            if gt_rec and gt_rec.get("path_ids"):
                pts = [id_to_xy.get(int(pid)) for pid in gt_rec.get("path_ids", [])]
                pts = [p for p in pts if p is not None]
                draw_line(draw, pts, (0, 200, 255, 180), width=3)

            pred_pts = [id_to_xy.get(int(pid)) for pid in render_data.get("pred_ids", [])]
            pred_pts = [p for p in pred_pts if p is not None]
            draw_line(draw, pred_pts, (255, 0, 255, 220), width=3)
            if pred_pts:
                draw_square(draw, pred_pts[-1], 6, (255, 0, 255, 220))

            start_id = render_data.get("start_id")
            if start_id is not None and int(start_id) in id_to_xy:
                draw_circle(draw, id_to_xy[int(start_id)], 7, (0, 120, 255, 220))

            for gid in render_data.get("goal_ids", []):
                xy = id_to_xy.get(gid)
                if xy:
                    draw_circle(draw, xy, 7, (0, 255, 0, 200))

            if render_data.get("target_xy") is not None and font_title:
                target_xy = render_data["target_xy"]
                draw_target_marker(draw, target_xy)
                draw_target_label(
                    draw,
                    target_xy,
                    f"TARGET {gt.get('target_category', 'object')}#{int(render_data.get('target_id'))}",
                    font_title,
                )

        info_box = None
        if font_small:
            info_box = text_block(draw, (10, 10), info_lines, font_small)
        if font_title:
            badge_y = (info_box[3] + 8) if info_box else 10
            draw_badge(draw, (10, badge_y), badge_text, font_title, badge_color)

        if font:
            footer_y = base_img.height + 10
            prompt_box = text_block(draw, (10, footer_y), prompt_lines, font)
            text_block(draw, (10, prompt_box[3] + 10), eval_lines, font)
            draw_legend(
                draw,
                int(base_img.width * 0.56),
                footer_y,
                task,
                font,
                max_width_px=base_img.width * 0.4,
            )

        save_image(img, out_path)

    # Write results jsonl
    with open(out_dir / "results.jsonl", "w") as f:
        for rec in results:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")

    # Index HTML
    if not args.no_index:
        html_path = out_dir / "index.html"
        with open(html_path, "w") as f:
            f.write("<html><head><meta charset='utf-8'><title>Next Results</title></head><body>\n")
            f.write("<h2>Next VQA Results</h2>\n")
            f.write("<p>Images: {}</p>\n".format(len(results)))
            f.write("<style>img{max-width:420px;height:auto;border:1px solid #333;margin:6px;} .card{display:inline-block;vertical-align:top;margin:6px;} .qid{font-family:monospace;font-size:12px;}</style>\n")
            for rec in results:
                qid = rec.get("question_id")
                if not qid:
                    continue
                img_rel = f"images/{qid}.png"
                f.write("<div class='card'>")
                f.write(f"<div class='qid'>{qid}</div>")
                f.write(f"<img src='{img_rel}' />")
                f.write("</div>\n")
            f.write("</body></html>\n")


if __name__ == "__main__":
    main()
