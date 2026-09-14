#!/usr/bin/env python3
"""Build a benchmark-only routing eval-fix package for Next VQA."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_navigation_gt import build_occupancy_grid
from render_scenes import select_goal_routing_candidates


ALL_TASKS = ("a1", "b1", "a2", "b2", "c")
ROUTING_TASKS = ("a2", "b2", "c")


def load_json(path: Path):
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
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


def count_sidecar_files(task_sidecar_root: Path) -> int:
    if not task_sidecar_root.exists():
        return 0
    return sum(1 for _ in task_sidecar_root.rglob("*.direct_pairs.json"))


def resolve_path(value: str | None, anchors: list[Path]) -> Path:
    if not value:
        raise FileNotFoundError("missing path value")
    path = Path(value)
    if path.is_absolute():
        return path
    for anchor in anchors:
        candidate = anchor / path
        if candidate.exists():
            return candidate
    if path.exists():
        return path
    raise FileNotFoundError(f"could not resolve path: {value}")


def infer_release_root_from_asset_path(path: Path) -> Path | None:
    parts = list(path.parts)
    for marker in ("task_outputs", "renders"):
        if marker in parts:
            idx = parts.index(marker)
            if idx > 0:
                return Path(*parts[:idx])
    return None


def resolve_render_root(*, release_package_root: Path, asset_paths: list[Path]) -> Path:
    candidates: list[Path] = []
    direct = release_package_root.parent / "renders"
    if direct.exists():
        candidates.append(direct)
    for asset_path in asset_paths:
        inferred_root = infer_release_root_from_asset_path(asset_path)
        if inferred_root is None:
            continue
        candidate = inferred_root / "renders"
        if candidate.exists():
            candidates.append(candidate)
    if not candidates:
        raise FileNotFoundError(
            f"could not resolve renders root for release_package_root={release_package_root}"
        )
    return candidates[0]


def load_config(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def pointmass_config(config: dict) -> dict:
    cfg = json.loads(json.dumps(config))
    cfg["navigation"]["agent_radius"] = 0.0
    cfg["navigation"]["passable_margin"] = 0.0
    return cfg


def is_embodied_routing_row(task: str, gt: dict) -> bool:
    if task == "b2":
        return True
    route_semantics = str(gt.get("route_semantics") or "").strip().lower()
    if route_semantics == "embodied":
        return True
    route_source_task = str(gt.get("route_source_task") or "").strip().lower()
    if route_source_task == "b2":
        return True
    robot_diameter_m = gt.get("robot_diameter_m")
    if robot_diameter_m is not None:
        try:
            return float(robot_diameter_m) > 0.0
        except (TypeError, ValueError):
            return False
    return task == "c"


def feasible_key_for_row(task: str, gt: dict) -> str:
    return "embodied_feasible" if is_embodied_routing_row(task, gt) else "pointmass_walkable"


def route_semantics_for_row(task: str, gt: dict) -> str:
    return "embodied" if is_embodied_routing_row(task, gt) else "pointmass"


def goal_ring_tolerance_m(config: dict) -> float:
    targets_cfg = config.get("targets", {})
    value = targets_cfg.get("goal_ring_tolerance_m")
    if value is not None:
        return float(value)
    dense_spacing = float(targets_cfg.get("dense_candidate_spacing_m", 0.0))
    grid_resolution = float(config.get("navigation", {}).get("grid_resolution", 0.05))
    return max(grid_resolution, dense_spacing, 0.05)


def extract_layout_objects(payload) -> list[dict]:
    if isinstance(payload, list):
        return [dict(obj) for obj in payload]
    if isinstance(payload, dict):
        for key in ("objects", "layout", "instances"):
            value = payload.get(key)
            if isinstance(value, list):
                return [dict(obj) for obj in value]
    raise ValueError("unsupported layout payload")


def infer_scene_and_view(row: dict) -> tuple[str, int]:
    scene_id = row.get("scene_id")
    view_id = row.get("view_id")
    if scene_id is not None and view_id is not None:
        return str(scene_id), int(view_id)
    visible_path = Path(str(row.get("visible_waypoints_path", "")))
    if visible_path.parent.name.startswith("view_") and visible_path.parent.parent.name:
        return visible_path.parent.parent.name, int(visible_path.parent.name.split("_", 1)[1])
    raise ValueError(f"could not infer scene/view for question_id={row.get('question_id')}")


def is_direct_segment_traversable(nav_grid, start_xy: tuple[float, float], end_xy: tuple[float, float]) -> bool:
    x0, y0 = float(start_xy[0]), float(start_xy[1])
    x1, y1 = float(end_xy[0]), float(end_xy[1])
    gx0, gy0 = nav_grid.world_to_grid(x0, y0)
    gx1, gy1 = nav_grid.world_to_grid(x1, y1)
    dx = abs(gx1 - gx0)
    dy = abs(gy1 - gy0)
    sx = 1 if gx0 < gx1 else -1
    sy = 1 if gy0 < gy1 else -1
    err = dx - dy
    gx, gy = gx0, gy0
    while True:
        if not nav_grid.is_free(gx, gy):
            return False
        if gx == gx1 and gy == gy1:
            break
        err2 = 2 * err
        if err2 > -dy:
            err -= dy
            gx += sx
        if err2 < dx:
            err += dx
            gy += sy
    return True


def build_direct_pairs_sidecar(
    visible_waypoints: list[dict],
    nav_grid,
    feasible_key: str,
    task: str,
    scene_id: str,
    view_id: int,
) -> dict:
    feasible = []
    for waypoint in visible_waypoints:
        display_id = waypoint.get("display_id")
        world_xyz = waypoint.get("world_xyz") or []
        if display_id is None or len(world_xyz) != 3:
            continue
        if not bool(waypoint.get(feasible_key, False)):
            continue
        feasible.append(
            {
                "display_id": int(display_id),
                "world_xyz": [float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2])],
            }
        )
    feasible.sort(key=lambda item: item["display_id"])
    neighbors: dict[str, list[dict]] = {str(item["display_id"]): [] for item in feasible}
    for idx, src in enumerate(feasible):
        x0, y0 = src["world_xyz"][:2]
        for dst in feasible[idx + 1 :]:
            x1, y1 = dst["world_xyz"][:2]
            if not is_direct_segment_traversable(nav_grid, (x0, y0), (x1, y1)):
                continue
            distance = math.hypot(x1 - x0, y1 - y0)
            neighbors[str(src["display_id"])].append(
                {"display_id": int(dst["display_id"]), "distance_m": round(distance, 4)}
            )
            neighbors[str(dst["display_id"])].append(
                {"display_id": int(src["display_id"]), "distance_m": round(distance, 4)}
            )
    for entries in neighbors.values():
        entries.sort(key=lambda item: item["display_id"])
    return {
        "schema_version": "next_direct_pairs_v1",
        "task": task,
        "scene_id": scene_id,
        "view_id": int(view_id),
        "display_ids": [item["display_id"] for item in feasible],
        "direct_neighbors": neighbors,
    }


def build_goal_ring_info(
    visible_waypoints: list[dict],
    bbox: list[float],
    ring_tolerance: float,
    feasible_key: str,
) -> dict:
    goal_candidates = select_goal_routing_candidates(
        visible_waypoints,
        bbox=bbox,
        ring_tolerance_m=ring_tolerance,
        feasible_key=feasible_key,
    )
    acceptable_goal_ids = sorted(
        int(candidate["display_id"])
        for candidate in goal_candidates
        if candidate.get("display_id") is not None
    )
    if not acceptable_goal_ids:
        raise ValueError("empty acceptable_goal_ids")
    return {
        "acceptable_goal_ids": acceptable_goal_ids,
        "goal_ring_min_distance_m": float(goal_candidates[0]["goal_ring_min_distance_m"]),
        "goal_ring_threshold_m": float(goal_candidates[0]["goal_ring_threshold_m"]),
    }


def load_route_candidate(render_root: Path, scene_id: str, view_id: int, routing_id: str) -> dict:
    bundle_path = render_root / scene_id / f"view_{view_id}" / "next_view_bundle_v3.json"
    payload = load_json(bundle_path)
    for candidate in payload.get("routing_candidates", []):
        if str(candidate.get("routing_id")) == str(routing_id):
            return dict(candidate)
    raise KeyError(f"routing_id={routing_id} missing from {bundle_path}")


def reference_path_display_ids(gt_row: dict, route_candidate: dict, view_bundle: dict, embodied: bool) -> list[int]:
    existing_path_ids = gt_row.get("path_ids") or []
    if existing_path_ids:
        return [int(x) for x in existing_path_ids if x is not None]
    field = "gt_path_candidate_ids_embodied" if embodied else "gt_path_candidate_ids_pointmass"
    raw = route_candidate.get(field) or []
    candidate_to_display = {
        str(candidate.get("candidate_id")): int(candidate.get("display_id"))
        for candidate in view_bundle.get("classification_candidates", [])
        if candidate.get("candidate_id") is not None and candidate.get("display_id") is not None
    }
    resolved: list[int] = []
    for item in raw:
        if item is None:
            continue
        try:
            resolved_id = int(item)
        except (TypeError, ValueError):
            lookup = candidate_to_display.get(str(item))
            if lookup is None:
                raise ValueError(f"cannot resolve reference-path item {item!r} to display_id")
            resolved_id = int(lookup)
        if not resolved or resolved[-1] != resolved_id:
            resolved.append(resolved_id)
    return resolved


def validate_reference_path(
    row_id: str,
    reference_ids: list[int],
    direct_neighbors: dict[str, list[dict]],
    acceptable_goal_ids: list[int],
) -> dict:
    if not reference_ids:
        raise ValueError(f"question_id={row_id} missing reference_path_display_ids")
    if reference_ids[-1] not in acceptable_goal_ids:
        raise ValueError(
            f"question_id={row_id} reference endpoint {reference_ids[-1]} not in acceptable_goal_ids"
        )
    if len(reference_ids) == 1:
        return {"reference_path_legal": True, "reference_endpoint_ok": True}
    neighbor_sets = {
        int(src): {int(item["display_id"]) for item in entries}
        for src, entries in direct_neighbors.items()
    }
    for src, dst in zip(reference_ids[:-1], reference_ids[1:]):
        if int(dst) not in neighbor_sets.get(int(src), set()):
            raise ValueError(f"question_id={row_id} illegal reference segment {src}->{dst}")
    return {"reference_path_legal": True, "reference_endpoint_ok": True}


def reference_path_length_m(reference_ids: list[int], direct_neighbors: dict[str, list[dict]]) -> float:
    if len(reference_ids) < 2:
        return 0.0
    adjacency = {
        int(src): {int(item["display_id"]): float(item["distance_m"]) for item in entries}
        for src, entries in direct_neighbors.items()
    }
    total = 0.0
    for src, dst in zip(reference_ids[:-1], reference_ids[1:]):
        weight = adjacency.get(int(src), {}).get(int(dst))
        if weight is None:
            raise ValueError(f"illegal reference segment {src}->{dst} when summing path length")
        total += weight
    return total


def shortest_reference_path(
    start_id: int,
    acceptable_goal_ids: list[int],
    direct_neighbors: dict[str, list[dict]],
) -> list[int]:
    adjacency = {
        int(src): {int(item["display_id"]): float(item["distance_m"]) for item in entries}
        for src, entries in direct_neighbors.items()
    }
    if start_id not in adjacency:
        return []
    goals = {int(goal_id) for goal_id in acceptable_goal_ids}
    heap = [(0.0, int(start_id))]
    prev: dict[int, int | None] = {int(start_id): None}
    dist = {int(start_id): 0.0}
    found_goal: int | None = None
    while heap:
        current_dist, node = heapq.heappop(heap)
        if current_dist > dist.get(node, float("inf")):
            continue
        if node in goals:
            found_goal = node
            break
        for neighbor, weight in adjacency.get(node, {}).items():
            next_dist = current_dist + float(weight)
            if next_dist >= dist.get(neighbor, float("inf")):
                continue
            dist[neighbor] = next_dist
            prev[neighbor] = node
            heapq.heappush(heap, (next_dist, neighbor))
    if found_goal is None:
        return []
    path = [found_goal]
    while prev[path[-1]] is not None:
        path.append(int(prev[path[-1]]))
    path.reverse()
    return path


def patch_ground_truth_dict(ground_truth: dict, patch_fields: dict) -> dict:
    patched = dict(ground_truth or {})
    patched.update(patch_fields)
    return patched


def validate_c_main_row(gt_row: dict, vqa_row: dict) -> None:
    generated_question = str(gt_row.get("generated_question") or "").strip()
    intent_family = str(gt_row.get("intent_family") or "").strip()
    route_source_task = str(gt_row.get("route_source_task") or "").strip().lower()
    robot_diameter_m = gt_row.get("robot_diameter_m")
    if not generated_question:
        raise ValueError(
            f"question_id={gt_row.get('question_id')} placeholder C row is not allowed: missing generated_question; "
            "use a C-main / C-main-vnext export instead"
        )
    if not intent_family:
        raise ValueError(
            f"question_id={gt_row.get('question_id')} placeholder C row is not allowed: missing intent_family"
        )
    if route_source_task != "b2":
        raise ValueError(
            f"question_id={gt_row.get('question_id')} placeholder C row is not allowed: route_source_task must be b2"
        )
    try:
        if robot_diameter_m is None or float(robot_diameter_m) <= 0.0:
            raise ValueError
    except (TypeError, ValueError):
        raise ValueError(
            f"question_id={gt_row.get('question_id')} placeholder C row is not allowed: invalid robot_diameter_m"
        ) from None
    question_text = str(vqa_row.get("question_text") or "").strip()
    if question_text != generated_question:
        raise ValueError(
            f"question_id={gt_row.get('question_id')} C-main export is malformed: question_text must match generated_question"
        )


def build_benchmark_evalfix_package(
    *,
    release_package_root: Path,
    scenes_root: Path,
    output_root: Path,
    config_path: Path,
    split: str = "benchmark",
    tasks: tuple[str, ...] = ROUTING_TASKS,
) -> dict:
    config = load_config(config_path)
    ring_tolerance = goal_ring_tolerance_m(config)
    full_release_root = release_package_root.parent
    render_root: Path | None = None

    out_split_dir = output_root / split
    out_gt_dir = out_split_dir / "gt"
    out_vqa_dir = out_split_dir / "vqa"
    out_sidecar_root = out_split_dir / "sidecars"
    benchmark_tasks = [
        task
        for task in ROUTING_TASKS
        if task in tasks
        or (
            (out_gt_dir / f"gt_next_{task}.jsonl").exists()
            and (out_vqa_dir / f"vqa_next_{task}.jsonl").exists()
        )
    ]

    layout_cache: dict[str, list[dict]] = {}
    target_cache: dict[str, dict[int, dict]] = {}
    nav_grid_cache: dict[tuple[str, str], object] = {}
    direct_pairs_cache: dict[tuple[str, str, int, str], dict] = {}
    summary = {
        "schema_version": "next_benchmark_evalfix_package_v1",
        "source_package_root": str(release_package_root),
        "source_release_root": str(full_release_root),
        "source_scenes_root": str(scenes_root),
        "config_path": str(config_path),
        "package_name": output_root.name,
        "split": split,
        "benchmark_tasks": benchmark_tasks,
        "route_semantics_by_task": {
            "a2": "pointmass",
            "b2": "embodied",
            "c": "embodied_or_row_defined",
        },
        "row_counts_by_task": {},
        "sidecar_counts_by_task": {},
        "audits": {
            "nonempty_acceptable_goal_rows": 0,
            "reference_path_legal_rows": 0,
            "reference_endpoint_ok_rows": 0,
            "reference_path_missing_rows": 0,
        },
        "notes": "Benchmark-only evaluation-protocol correction. Train/val are intentionally unchanged.",
    }

    for task in ALL_TASKS:
        src_gt_path = release_package_root / split / "gt" / f"gt_next_{task}.jsonl"
        src_vqa_path = release_package_root / split / "vqa" / f"vqa_next_{task}.jsonl"
        out_gt_path = out_gt_dir / src_gt_path.name
        out_vqa_path = out_vqa_dir / src_vqa_path.name
        if not src_gt_path.exists() or not src_vqa_path.exists():
            continue
        gt_rows = load_jsonl(src_gt_path)
        vqa_rows = load_jsonl(src_vqa_path)
        if task not in tasks:
            if out_gt_path.exists() and out_vqa_path.exists():
                kept_gt_rows = load_jsonl(out_gt_path)
                kept_vqa_rows = load_jsonl(out_vqa_path)
                summary["row_counts_by_task"][task] = len(kept_gt_rows)
                summary["sidecar_counts_by_task"][task] = count_sidecar_files(out_sidecar_root / task)
            else:
                write_jsonl(out_gt_path, gt_rows)
                write_jsonl(out_vqa_path, vqa_rows)
                summary["row_counts_by_task"][task] = len(gt_rows)
                summary["sidecar_counts_by_task"][task] = count_sidecar_files(out_sidecar_root / task)
            continue

        gt_by_qid = {str(row["question_id"]): dict(row) for row in gt_rows}
        patched_gt_rows = []
        patched_vqa_rows = []
        sidecar_count = 0

        for vqa_row in vqa_rows:
            qid = str(vqa_row.get("question_id"))
            gt_row = gt_by_qid.get(qid)
            if gt_row is None:
                raise KeyError(f"missing gt row for question_id={qid}")
            if task == "c":
                validate_c_main_row(gt_row, vqa_row)

            scene_id, view_id = infer_scene_and_view(gt_row)
            if scene_id not in layout_cache:
                layout_path = scenes_root / scene_id / "layout.json"
                layout_cache[scene_id] = extract_layout_objects(load_json(layout_path))
                target_cache[scene_id] = {
                    int(obj["id"]): dict(obj)
                    for obj in layout_cache[scene_id]
                    if obj.get("id") is not None
                }
            target_id = int(gt_row["target_id"])
            target_record = target_cache[scene_id].get(target_id)
            if target_record is None or not isinstance(target_record.get("bbox"), list):
                raise KeyError(f"scene={scene_id} target_id={target_id} missing bbox")

            visible_path = resolve_path(
                gt_row.get("visible_waypoints_path") or vqa_row.get("visible_waypoints_path"),
                [release_package_root, full_release_root],
            )
            image_path = resolve_path(
                gt_row.get("image_path") or vqa_row.get("image_path"),
                [release_package_root, full_release_root],
            )
            if render_root is None:
                render_root = resolve_render_root(
                    release_package_root=release_package_root,
                    asset_paths=[visible_path, image_path],
                )
            visible_payload = load_json(visible_path)
            visible_waypoints = list(visible_payload.get("visible_waypoints", []))
            embodied = is_embodied_routing_row(task, vqa_row.get("ground_truth", {}))
            route_semantics = "embodied" if embodied else "pointmass"
            feasible_key = feasible_key_for_row(task, vqa_row.get("ground_truth", {}))
            sidecar_path = out_sidecar_root / task / scene_id / f"view_{view_id}.direct_pairs.json"
            sidecar_cache_key = (task, scene_id, int(view_id), feasible_key)
            sidecar_payload = direct_pairs_cache.get(sidecar_cache_key)
            if sidecar_payload is None:
                if sidecar_path.exists():
                    sidecar_payload = load_json(sidecar_path)
                else:
                    nav_key = (scene_id, route_semantics)
                    if nav_key not in nav_grid_cache:
                        scene_layout = layout_cache[scene_id]
                        grid_config = config if embodied else pointmass_config(config)
                        nav_grid_cache[nav_key] = build_occupancy_grid(
                            scene_layout,
                            grid_config,
                            scene_path=scenes_root / scene_id,
                        )
                    nav_grid = nav_grid_cache[nav_key]
                    sidecar_payload = build_direct_pairs_sidecar(
                        visible_waypoints=visible_waypoints,
                        nav_grid=nav_grid,
                        feasible_key=feasible_key,
                        task=task,
                        scene_id=scene_id,
                        view_id=view_id,
                    )
                    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
                    sidecar_path.write_text(json.dumps(sidecar_payload, indent=2, ensure_ascii=True) + "\n")
                    sidecar_count += 1
                direct_pairs_cache[sidecar_cache_key] = sidecar_payload

            goal_info = build_goal_ring_info(
                visible_waypoints=visible_waypoints,
                bbox=target_record["bbox"],
                ring_tolerance=ring_tolerance,
                feasible_key=feasible_key,
            )
            route_candidate = load_route_candidate(
                render_root=render_root,
                scene_id=scene_id,
                view_id=view_id,
                routing_id=str(gt_row["routing_id"]),
            )
            reference_ids = shortest_reference_path(
                start_id=int(gt_row["start_id"]),
                acceptable_goal_ids=goal_info["acceptable_goal_ids"],
                direct_neighbors=sidecar_payload["direct_neighbors"],
            )
            summary["audits"]["nonempty_acceptable_goal_rows"] += 1
            if reference_ids:
                audit = validate_reference_path(
                    row_id=qid,
                    reference_ids=reference_ids,
                    direct_neighbors=sidecar_payload["direct_neighbors"],
                    acceptable_goal_ids=goal_info["acceptable_goal_ids"],
                )
                summary["audits"]["reference_path_legal_rows"] += int(audit["reference_path_legal"])
                summary["audits"]["reference_endpoint_ok_rows"] += int(audit["reference_endpoint_ok"])
            else:
                summary["audits"]["reference_path_missing_rows"] += 1

            optimal_length_m = gt_row.get("optimal_length_m")
            if reference_ids:
                optimal_length_m = round(
                    reference_path_length_m(reference_ids, sidecar_payload["direct_neighbors"]),
                    4,
                )

            patch_fields = {
                "acceptable_goal_ids": goal_info["acceptable_goal_ids"],
                "goal_ring_min_distance_m": round(goal_info["goal_ring_min_distance_m"], 4),
                "goal_ring_threshold_m": round(goal_info["goal_ring_threshold_m"], 4),
                "reference_path_display_ids": reference_ids,
                "direct_pairs_ref": str(Path("sidecars") / task / scene_id / f"view_{view_id}.direct_pairs.json"),
                "route_semantics": route_semantics,
                "optimal_length_m": optimal_length_m,
            }
            patched_gt = dict(gt_row)
            patched_gt.update(patch_fields)
            patched_vqa = dict(vqa_row)
            patched_vqa["ground_truth"] = patch_ground_truth_dict(
                vqa_row.get("ground_truth", {}),
                patch_fields,
            )
            patched_gt_rows.append(patched_gt)
            patched_vqa_rows.append(patched_vqa)

        write_jsonl(out_gt_path, patched_gt_rows)
        write_jsonl(out_vqa_path, patched_vqa_rows)
        summary["row_counts_by_task"][task] = len(patched_gt_rows)
        summary["sidecar_counts_by_task"][task] = count_sidecar_files(out_sidecar_root / task)

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "release_manifest.json"
    summary_path = output_root / "summary.json"
    manifest_text = json.dumps(summary, indent=2, ensure_ascii=True) + "\n"
    manifest_path.write_text(manifest_text)
    summary_path.write_text(manifest_text)
    return summary


def parse_tasks(raw: str) -> tuple[str, ...]:
    tasks = tuple(task.strip() for task in raw.split(",") if task.strip())
    if not tasks:
        raise ValueError("no tasks requested")
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Build benchmark-only routing eval-fix package.")
    parser.add_argument("--release-package-root", required=True)
    parser.add_argument("--scenes-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="benchmark")
    parser.add_argument("--tasks", default="a2,b2,c")
    args = parser.parse_args()

    summary = build_benchmark_evalfix_package(
        release_package_root=Path(args.release_package_root),
        scenes_root=Path(args.scenes_root),
        output_root=Path(args.output_root),
        config_path=Path(args.config),
        split=args.split,
        tasks=parse_tasks(args.tasks),
    )
    print(json.dumps(summary, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
