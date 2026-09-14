#!/usr/bin/env python3
"""
Evaluate Next VQA tasks (A1/A2/B1/B2) with ID-based outputs.
"""

from __future__ import annotations

import argparse
import heapq
import json
import math
from pathlib import Path
from typing import Optional


ROUTING_GT_FIELDS = (
    "start_id",
    "goal_id",
    "goal_ids",
    "path_ids",
    "optimal_length_m",
    "canonical_sparse_length_m",
    "target_id",
    "resolved_goal_display_id",
    "label",
    "intent_family",
    "intent_strength",
    "residual_cue_type",
    "route_source_task",
    "route_semantics",
    "candidate_inventory_ref",
    "acceptable_goal_ids",
    "goal_ring_min_distance_m",
    "goal_ring_threshold_m",
    "reference_path_display_ids",
    "direct_pairs_ref",
    "robot_diameter_m",
    "target_category",
    "canonical_category",
)


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


def merge_vqa_with_gt(vqa: list[dict], gt_rows: list[dict]) -> list[dict]:
    if not gt_rows:
        return vqa
    gt_by_qid = {str(row.get("question_id")): row for row in gt_rows if row.get("question_id") is not None}
    merged = []
    for row in vqa:
        qid = str(row.get("question_id"))
        gt_row = gt_by_qid.get(qid)
        if gt_row is None:
            merged.append(row)
            continue
        new_row = json.loads(json.dumps(row))
        gt = new_row.setdefault("ground_truth", {})
        for key in ROUTING_GT_FIELDS:
            if gt.get(key) is None and gt_row.get(key) is not None:
                gt[key] = gt_row.get(key)
        merged.append(new_row)
    return merged


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
        # Strip code fences
        s = s.replace("```json", "").replace("```", "").strip()
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [int(x) for x in parsed]
        except Exception:
            return None
    return None


def prediction_output(pred: dict) -> object:
    if pred.get("success") is False:
        return None
    return pred.get("output")


def build_display_to_waypoint(visible_waypoints: list[dict]) -> dict[int, int]:
    return {int(wp["display_id"]): int(wp["waypoint_id"]) for wp in visible_waypoints}


def load_visible_ids(path: Path) -> tuple[set[int], dict[int, int]]:
    data = json.loads(path.read_text())
    v = data.get("visible_waypoints", [])
    ids = {int(w["display_id"]) for w in v}
    return ids, build_display_to_waypoint(v)


def load_visible_payload(path: Path) -> dict:
    return json.loads(path.read_text())


def resolve_optional_path(ref: str | None, anchors: list[Path]) -> Path | None:
    if not ref:
        return None
    path = Path(ref)
    if path.is_absolute():
        return path if path.exists() else None
    for anchor in anchors:
        candidate = anchor / path
        if candidate.exists():
            return candidate
    if path.exists():
        return path
    return None


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


def build_display_world_xyz(visible_waypoints: list[dict]) -> dict[int, tuple[float, float, float]]:
    mapping: dict[int, tuple[float, float, float]] = {}
    for waypoint in visible_waypoints:
        display_id = waypoint.get("display_id")
        world_xyz = waypoint.get("world_xyz") or []
        if display_id is None or len(world_xyz) != 3:
            continue
        mapping[int(display_id)] = (float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2]))
    return mapping


def euclidean_distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def load_direct_pairs(path: Path, visible_waypoints: list[dict]) -> dict[int, dict[int, float]]:
    payload = json.loads(path.read_text())
    neighbors = payload.get("direct_neighbors", {})
    world_xyz_by_display = build_display_world_xyz(visible_waypoints)
    adj: dict[int, dict[int, float]] = {}
    for src_raw, entries in neighbors.items():
        try:
            src = int(src_raw)
        except (TypeError, ValueError):
            continue
        src_xyz = world_xyz_by_display.get(src)
        src_adj = adj.setdefault(src, {})
        for entry in entries or []:
            dst_raw = entry.get("display_id")
            if dst_raw is None:
                continue
            try:
                dst = int(dst_raw)
            except (TypeError, ValueError):
                continue
            distance = entry.get("distance_m")
            if distance is None and src_xyz is not None:
                dst_xyz = world_xyz_by_display.get(dst)
                if dst_xyz is not None:
                    distance = euclidean_distance(src_xyz, dst_xyz)
            if distance is None:
                continue
            src_adj[dst] = float(distance)
    return adj


def _visible_to_graph_distance(visible_wp: dict, graph_wp: dict) -> tuple[float, int]:
    world_a = visible_wp.get("world_xyz") or []
    world_b = graph_wp.get("world_xyz") or []
    if len(world_a) == 3 and len(world_b) == 3:
        dx = float(world_a[0]) - float(world_b[0])
        dy = float(world_a[1]) - float(world_b[1])
        dz = float(world_a[2]) - float(world_b[2])
        world_dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    else:
        world_dist = float("inf")

    grid_a = visible_wp.get("grid_xy") or []
    grid_b = graph_wp.get("grid_xy") or []
    if len(grid_a) == 2 and len(grid_b) == 2:
        grid_dist = abs(int(grid_a[0]) - int(grid_b[0])) + abs(int(grid_a[1]) - int(grid_b[1]))
    else:
        grid_dist = 10**9
    return world_dist, grid_dist


def build_display_to_graph_waypoint(
    visible_waypoints: list[dict],
    graph_waypoints: list[dict],
    graph_waypoint_ids: set[int],
) -> dict[int, int]:
    if not graph_waypoint_ids:
        return build_display_to_waypoint(visible_waypoints)

    usable_graph_waypoints = [wp for wp in graph_waypoints if int(wp.get("waypoint_id", -1)) in graph_waypoint_ids]
    mapping: dict[int, int] = {}
    for visible_wp in visible_waypoints:
        display_id = visible_wp.get("display_id")
        waypoint_id = visible_wp.get("waypoint_id")
        if display_id is None or waypoint_id is None:
            continue
        display_id = int(display_id)
        waypoint_id = int(waypoint_id)
        if waypoint_id in graph_waypoint_ids:
            mapping[display_id] = waypoint_id
            continue
        if not usable_graph_waypoints:
            continue
        best = min(
            (
                *_visible_to_graph_distance(visible_wp, graph_wp),
                int(graph_wp["waypoint_id"]),
            )
            for graph_wp in usable_graph_waypoints
        )
        mapping[display_id] = int(best[2])
    return mapping


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


def require_optimal_length_m(gt: dict, *, qid: str) -> float:
    value = gt.get("optimal_length_m")
    if value is None:
        raise ValueError(f"question_id={qid} missing optimal_length_m")
    optimal = float(value)
    if optimal <= 0.0:
        raise ValueError(f"question_id={qid} has non-positive optimal_length_m")
    return optimal


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


def require_goal_display_ids(
    gt: dict,
    *,
    qid: str,
    require_acceptable_goal_ids: bool = False,
) -> list[int]:
    if require_acceptable_goal_ids:
        goal_ids = gt.get("acceptable_goal_ids")
        if goal_ids is None:
            raise ValueError(
                f"question_id={qid} missing acceptable_goal_ids for direct-pairs goal-ring routing semantics"
            )
        return [int(x) for x in goal_ids if x is not None]

    goal_ids = gt.get("acceptable_goal_ids")
    if goal_ids is None:
        goal_ids = gt.get("goal_ids")
    if goal_ids is None:
        gid = gt.get("goal_id")
        goal_ids = [gid] if gid is not None else []
    return [int(x) for x in goal_ids if x is not None]


def eval_classification(vqa: list[dict], preds: dict[str, dict]) -> dict:
    tp = fp = fn = tn = invalid = 0
    for q in vqa:
        qid = q.get("question_id")
        gt = q.get("ground_truth", {})
        all_ids = gt.get("all_ids")
        if all_ids is None:
            # fallback to visible_waypoints
            vp_path = Path(q.get("visible_waypoints_path"))
            if vp_path.exists():
                all_ids, _ = load_visible_ids(vp_path)
            else:
                all_ids = []
        all_ids = set(int(x) for x in all_ids)
        gt_pos = set(int(x) for x in (gt.get("walkable_ids") or gt.get("answer") or []))

        pred = preds.get(qid, {})
        pred_ids = parse_ids(prediction_output(pred))
        if pred_ids is None:
            pred_ids = []
        pred_set = set(int(x) for x in pred_ids)

        # Invalid IDs
        invalid += len([x for x in pred_set if x not in all_ids])
        pred_set = pred_set & all_ids

        tp += len(pred_set & gt_pos)
        fp += len(pred_set - gt_pos)
        fn += len(gt_pos - pred_set)
        tn += len(all_ids - (pred_set | gt_pos))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    bal_acc = (recall + tnr) / 2
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "invalid_ids": invalid,
    }


def eval_routing(vqa: list[dict], preds: dict[str, dict], embodied: bool,
                 goal_hop: int = 0, goal_dist_m: float = 0.0,
                 weak_connectivity: bool = False) -> dict:
    total = 0
    valid_ids = 0
    valid_paths = 0
    success = 0
    spl_sum = 0.0

    for q in vqa:
        qid = q.get("question_id")
        total += 1
        pred = preds.get(qid, {})
        pred_ids = parse_ids(prediction_output(pred))
        if pred_ids is None or len(pred_ids) == 0:
            continue

        vp_path = Path(q.get("visible_waypoints_path"))
        if not vp_path.exists():
            continue
        visible_payload = load_visible_payload(vp_path)
        visible_waypoints = list(visible_payload.get("visible_waypoints", []))
        all_ids = {int(w["display_id"]) for w in visible_waypoints if w.get("display_id") is not None}

        # Filter invalid ids
        if any(pid not in all_ids for pid in pred_ids):
            continue
        valid_ids += 1

        gt = q.get("ground_truth", {})
        start_id = gt.get("start_id")
        image_path_value = q.get("image_path")
        image_path = Path(image_path_value) if image_path_value else None
        row_source_dir = Path(q.get("__row_source_dir", "")) if q.get("__row_source_dir") else None
        anchors = [vp_path.parent]
        if image_path is not None:
            anchors.append(image_path.parent)
        if row_source_dir is not None:
            anchors.extend([row_source_dir, row_source_dir.parent])
        anchors.append(Path.cwd())
        direct_pairs_ref = gt.get("direct_pairs_ref") or q.get("direct_pairs_ref")
        direct_pairs_path = resolve_optional_path(direct_pairs_ref, anchors)
        evalfix_expected = (
            direct_pairs_ref is not None
            or gt.get("acceptable_goal_ids") is not None
            or gt.get("goal_ring_threshold_m") is not None
        )
        if evalfix_expected and direct_pairs_ref is None:
            raise ValueError(f"question_id={qid} missing required direct_pairs_ref for eval-fix routing row")
        if direct_pairs_ref is not None and direct_pairs_path is None:
            raise ValueError(f"question_id={qid} cannot resolve direct_pairs_ref={direct_pairs_ref!r}")

        adj: dict[int, dict[int, float]]
        disp_to_wp: dict[int, int] = {}
        using_direct_pairs = direct_pairs_path is not None
        if using_direct_pairs:
            if weak_connectivity:
                raise ValueError(f"question_id={qid} cannot use --weak-connectivity with direct-pairs routing semantics")
            if goal_hop > 0 or goal_dist_m > 0.0:
                raise ValueError(f"question_id={qid} cannot use goal-hop/dist tolerance with direct-pairs routing semantics")
            adj = load_direct_pairs(direct_pairs_path, visible_waypoints)
        else:
            if image_path is None:
                raise ValueError(f"question_id={qid} cannot use legacy routing fallback without image_path")
            # Legacy fallback: sparse graph adjacency on graph waypoint ids.
            scene_dir = image_path.parent.parent
            graph = load_graph(scene_dir, embodied=embodied)
            adj = build_adj(graph.get("edges", []))
            disp_to_wp = build_display_to_graph_waypoint(
                visible_waypoints,
                graph.get("waypoints", []),
                set(adj.keys()),
            )

        # Convert to waypoint IDs
        if using_direct_pairs:
            wp_seq = [int(pid) for pid in pred_ids]
        else:
            try:
                wp_seq = [disp_to_wp[int(pid)] for pid in pred_ids]
            except KeyError:
                continue

        # Edge validity + length
        length = 0.0
        ok = False
        start_anchored = (
            start_id is not None
            and len(pred_ids) >= 2
            and int(pred_ids[0]) == int(start_id)
        )
        if start_anchored:
            if weak_connectivity:
                ok = True
                for a, b in zip(wp_seq[:-1], wp_seq[1:]):
                    dist = dijkstra(adj, a, b)
                    if dist is None:
                        ok = False
                        break
                    length += dist
            else:
                ok = True
                for a, b in zip(wp_seq[:-1], wp_seq[1:]):
                    if a not in adj or b not in adj[a]:
                        ok = False
                        break
                    length += adj[a][b]
        if not ok:
            continue
        valid_paths += 1

        goal_ids = require_goal_display_ids(
            gt,
            qid=str(qid),
            require_acceptable_goal_ids=using_direct_pairs,
        )
        if not goal_ids:
            continue
        pred_goal_id = int(pred_ids[-1])
        ok_success = pred_goal_id in goal_ids
        if not using_direct_pairs and not ok_success and (goal_hop > 0 or goal_dist_m > 0.0):
            try:
                pred_wp = disp_to_wp[pred_goal_id]
            except KeyError:
                pred_wp = None
            if pred_wp is not None:
                for gid in goal_ids:
                    try:
                        goal_wp = disp_to_wp[gid]
                    except KeyError:
                        continue
                    if goal_hop > 0:
                        hops = shortest_hops(adj, pred_wp, goal_wp)
                        if hops is not None and hops <= goal_hop:
                            ok_success = True
                            break
                    if not ok_success and goal_dist_m > 0.0:
                        dist = dijkstra(adj, pred_wp, goal_wp)
                        if dist is not None and dist <= goal_dist_m:
                            ok_success = True
                            break

        if ok_success:
            success += 1
            # SPL
            if start_id is None:
                continue
            optimal = require_optimal_length_m(gt, qid=str(qid))
            if using_direct_pairs and int(start_id) not in all_ids:
                continue
            if not using_direct_pairs:
                try:
                    start_wp = disp_to_wp[int(start_id)]
                except KeyError:
                    continue
                if start_wp not in adj:
                    continue
            spl_sum += optimal / max(optimal, length)

    return {
        "total": total,
        "success": success,
        "valid_id_rate": round(valid_ids / max(total, 1), 4),
        "valid_path_rate": round(valid_paths / max(total, 1), 4),
        "valid_edge_rate": round(valid_paths / max(total, 1), 4),
        "success_rate": round(success / max(total, 1), 4),
        # SPL over successful episodes only (standard definition).
        "spl": round(spl_sum / max(success, 1), 4),
        # SPL over all episodes (treat failures as 0), useful for intuition.
        "spl_all": round(spl_sum / max(total, 1), 4),
    }


def is_embodied_routing_question(question: dict) -> bool:
    task = str(question.get("task") or "").strip().lower()
    if task == "b2":
        return True
    gt = question.get("ground_truth", {}) or {}
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
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Next VQA tasks.")
    parser.add_argument("--vqa", required=True, help="vqa_next_*.jsonl")
    parser.add_argument("--gt", default=None, help="optional gt_next_*.jsonl for routing metadata merge")
    parser.add_argument("--predictions", required=True, help="predictions jsonl")
    parser.add_argument("--output", required=True, help="output metrics json")
    parser.add_argument("--restrict-to-predictions", action="store_true",
                        help="Only evaluate questions whose question_id appears in predictions. "
                             "Useful for smoke runs generated with --max-questions.")
    parser.add_argument("--goal-hop", type=int, default=0, help="Allow success if within N hops from goal")
    parser.add_argument("--goal-dist-m", type=float, default=0.0, help="Allow success if within graph distance (m)")
    parser.add_argument("--weak-connectivity", action="store_true",
                        help="Relax edge validity: each step only needs to be graph-connected (not direct edge)")
    args = parser.parse_args()

    vqa = load_jsonl(Path(args.vqa))
    vqa_source_dir = Path(args.vqa).resolve().parent
    for row in vqa:
        row["__row_source_dir"] = str(vqa_source_dir)
    if args.gt:
        vqa = merge_vqa_with_gt(vqa, load_jsonl(Path(args.gt)))
    preds_list = load_jsonl(Path(args.predictions))
    preds = {p.get("question_id"): p for p in preds_list}

    if args.restrict_to_predictions:
        pred_qids = {qid for qid in preds.keys() if qid}
        vqa = [q for q in vqa if q.get("question_id") in pred_qids]

    if not vqa:
        raise SystemExit("No VQA records found")

    task = vqa[0].get("task")
    if task in ("a1", "b1"):
        metrics = eval_classification(vqa, preds)
    elif task in ("a2", "b2", "c"):
        metrics = eval_routing(vqa, preds, embodied=is_embodied_routing_question(vqa[0]),
                               goal_hop=args.goal_hop,
                               goal_dist_m=args.goal_dist_m,
                               weak_connectivity=args.weak_connectivity)
    else:
        metrics = {"note": "task not supported"}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"task": task, "metrics": metrics}, f, indent=2)

    print(json.dumps({"task": task, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
