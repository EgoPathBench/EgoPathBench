"""
NavBench3D - Step 6: Evaluation Metrics.

Computes navigation evaluation metrics:
- Success Rate (SR)
- Success weighted by Path Length (SPL)
- Collision Rate (via traversability grid)
- Path Deviation (average distance from predicted path to GT path)
- Path Smoothness

The evaluation validates VLM-predicted paths (3D coordinate arrays) against
pre-computed traversability grids with Minkowski inflation (agent_radius=0.30m).

Usage:
    python scripts/evaluate.py --predictions results/gpt4o_predictions.json \
        --gt data/vqa_questions/all_questions.json \
        --gt-dir data/gt_paths
"""

import json
import math
import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Import traversability-based validation from GT module
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_navigation_gt import validate_vlm_path


def parse_prediction(raw_output: str) -> Optional[list[list[float]]]:
    """Parse VLM output into a list of 3D coordinates."""
    import re

    # Try to extract JSON array from the output
    # Handle cases where VLM wraps in markdown code blocks
    raw_output = raw_output.strip()
    raw_output = re.sub(r"```json\s*", "", raw_output)
    raw_output = re.sub(r"```\s*$", "", raw_output)
    raw_output = raw_output.strip()

    try:
        parsed = json.loads(raw_output)
        if isinstance(parsed, list):
            # Validate structure: list of [x, y, z] coordinates
            path = []
            for point in parsed:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    path.append([float(point[0]), float(point[1]), float(point[2]) if len(point) > 2 else 0.0])
            if path:
                return path
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Try to find array-like patterns in text
    pattern = r'\[\s*(-?\d+\.?\d*)\s*,\s*(-?\d+\.?\d*)\s*(?:,\s*(-?\d+\.?\d*))?\s*\]'
    matches = re.findall(pattern, raw_output)
    if matches:
        path = []
        for m in matches:
            x, y, z = float(m[0]), float(m[1]), float(m[2]) if m[2] else 0.0
            path.append([x, y, z])
        if path:
            return path

    return None


def path_length(path: list[list[float]]) -> float:
    """Compute total path length."""
    total = 0.0
    for i in range(1, len(path)):
        total += math.sqrt(sum((a - b) ** 2 for a, b in zip(path[i], path[i-1])))
    return total


def point_distance(p1: list[float], p2: list[float]) -> float:
    """Euclidean distance between two 3D points."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def point_distance_2d(p1: list[float], p2: list[float]) -> float:
    """2D Euclidean distance (x, y only) — used for floor-plane navigation."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def success_rate(pred_path: list[list[float]], target_pos: list[float],
                 goal_pos: list[float] = None,
                 threshold: float = 1.0) -> bool:
    """Check if the prediction ends near the target/goal.

    Uses 2D (x, y) distance since navigation is on the floor plane.
    If goal_pos is provided, uses it instead of target_pos. goal_pos is
    the nearest reachable point to the target (handles large objects like beds
    whose center may be >1m from any free cell).
    """
    if not pred_path:
        return False
    final_pos = pred_path[-1]
    check_pos = goal_pos if goal_pos is not None else target_pos
    dist = point_distance_2d(final_pos, check_pos)
    return dist <= threshold


def spl(pred_path: list[list[float]], gt_path: list[list[float]],
        target_pos: list[float], threshold: float = 1.0,
        goal_pos: list[float] = None) -> float:
    """Success weighted by Path Length."""
    is_success = success_rate(pred_path, target_pos, goal_pos, threshold)
    if not is_success:
        return 0.0

    gt_len = path_length(gt_path)
    pred_len = path_length(pred_path)

    if pred_len == 0:
        return 0.0

    return gt_len / max(gt_len, pred_len)


def collision_check_traversability(
    pred_path: list[list[float]],
    gt_dir: str,
) -> dict:
    """
    Check path traversability using the pre-computed traversability grid.

    The traversability grid is already Minkowski-inflated by agent_radius
    (0.30m), so checking the center-line is sufficient.

    Returns dict with:
      - traversable: bool
      - collision_rate: float (fraction of steps in collision)
      - collision_point: [x, y] or None
      - min_clearance: float (meters)
      - path_length: float (meters)
    """
    if not pred_path or not gt_dir:
        return {
            "traversable": False,
            "collision_rate": 1.0,
            "collision_point": None,
            "min_clearance": 0.0,
            "path_length": 0.0,
        }

    try:
        result = validate_vlm_path(gt_dir, pred_path)
        return {
            "traversable": result["traversable"],
            "collision_rate": round(1.0 - result["traversable_fraction"], 4),
            "collision_point": result["collision_point"],
            "min_clearance": result["min_clearance"],
            "path_length": result["path_length"],
        }
    except Exception:
        # Fallback: no traversability data available
        return {
            "traversable": False,
            "collision_rate": -1,  # Unknown
            "collision_point": None,
            "min_clearance": 0.0,
            "path_length": path_length(pred_path),
        }


def collision_rate_simple(pred_path: list[list[float]], layout: list[dict],
                          agent_radius: float = 0.30) -> float:
    """
    Fallback: simplified 2D AABB collision detection (used when no traversability grid).
    Uses object bounding boxes from layout.json.
    """
    if not pred_path:
        return 1.0

    obstacles = []
    for obj in layout:
        bbox = obj.get("bbox", [])
        if len(bbox) >= 6:
            # Object center and half-extents
            cx, cy = bbox[0], bbox[1]
            hx, hy = bbox[3] / 2, bbox[4] / 2
            height = bbox[5] if len(bbox) > 5 else 1.0
            base_z = bbox[2]

            if base_z < 1.5 and height > 0.1:
                obstacles.append((cx, cy, hx + agent_radius, hy + agent_radius))

    collisions = 0
    for point in pred_path:
        px, py = point[0], point[1]
        for cx, cy, hx, hy in obstacles:
            if abs(px - cx) < hx and abs(py - cy) < hy:
                collisions += 1
                break

    return collisions / len(pred_path)


def path_deviation(pred_path: list[list[float]],
                   gt_path: list[list[float]]) -> float:
    """
    Average deviation of predicted path from GT optimal path.
    For each predicted waypoint, compute minimum 2D distance to any GT waypoint.
    Returns average minimum distance in meters (lower is better).
    Uses 2D distance since all navigation is on the floor plane.
    """
    if not pred_path or not gt_path:
        return float("inf")

    deviations = []
    for pred_point in pred_path:
        min_dist = float("inf")
        for gt_point in gt_path:
            dist = point_distance_2d(pred_point, gt_point)
            min_dist = min(min_dist, dist)
        deviations.append(min_dist)

    return sum(deviations) / len(deviations)


def path_smoothness(pred_path: list[list[float]]) -> float:
    """
    Measure path smoothness via average angular change between consecutive segments.
    Lower = smoother. Returns average angle in radians.
    """
    if len(pred_path) < 3:
        return 0.0

    angles = []
    for i in range(1, len(pred_path) - 1):
        # Vectors
        v1 = [pred_path[i][j] - pred_path[i-1][j] for j in range(min(len(pred_path[i]), 3))]
        v2 = [pred_path[i+1][j] - pred_path[i][j] for j in range(min(len(pred_path[i]), 3))]

        # Normalize
        len1 = math.sqrt(sum(x**2 for x in v1))
        len2 = math.sqrt(sum(x**2 for x in v2))

        if len1 < 1e-8 or len2 < 1e-8:
            continue

        v1 = [x / len1 for x in v1]
        v2 = [x / len2 for x in v2]

        # Dot product → angle
        dot = sum(a * b for a, b in zip(v1, v2))
        dot = max(-1.0, min(1.0, dot))
        angle = math.acos(dot)
        angles.append(angle)

    return sum(angles) / len(angles) if angles else 0.0


def evaluate_single(prediction: dict, question: dict,
                    gt_dir: str = None, layout: Optional[list[dict]] = None) -> dict:
    """Evaluate a single prediction against ground truth.

    Uses traversability grid (if gt_dir provided) for accurate collision checking,
    falling back to simplified AABB checking with layout.json.
    """
    gt = question["ground_truth"]

    # Parse prediction
    pred_path = parse_prediction(prediction.get("output", ""))

    result = {
        "question_id": question["question_id"],
        "level": question["level"],
        "level_name": question["level_name"],
        "scene_id": question["scene_id"],
        "source": question.get("source", "internscene"),
        "parsed_successfully": pred_path is not None,
    }

    if pred_path is None:
        result.update({
            "success": False,
            "spl": 0.0,
            "collision_rate": 1.0,
            "traversable": False,
            "path_deviation": float("inf"),
            "path_smoothness": float("inf"),
            "pred_path_length": 0.0,
            "gt_path_length": gt["path_length"],
        })
        return result

    threshold = 1.0  # meters

    goal_pos = gt.get("goal_position", gt["target_position"])
    result["success"] = success_rate(pred_path, gt["target_position"], goal_pos, threshold)
    result["spl"] = spl(pred_path, gt["optimal_path"], gt["target_position"], threshold, goal_pos)
    result["path_deviation"] = path_deviation(pred_path, gt["optimal_path"])
    result["path_smoothness"] = path_smoothness(pred_path)
    result["pred_path_length"] = path_length(pred_path)
    result["gt_path_length"] = gt["path_length"]
    result["pred_num_points"] = len(pred_path)

    # Collision checking: prefer traversability grid, fall back to layout AABB
    if gt_dir:
        scene_gt_dir = str(Path(gt_dir) / question["scene_id"])
        trav_result = collision_check_traversability(pred_path, scene_gt_dir)
        result["collision_rate"] = trav_result["collision_rate"]
        result["traversable"] = trav_result["traversable"]
        result["min_clearance"] = trav_result["min_clearance"]
    elif layout:
        result["collision_rate"] = collision_rate_simple(pred_path, layout)
        result["traversable"] = result["collision_rate"] == 0.0
        result["min_clearance"] = -1  # Unknown
    else:
        result["collision_rate"] = -1  # Unknown
        result["traversable"] = False
        result["min_clearance"] = -1

    return result


def aggregate_results(results: list[dict]) -> dict:
    """Aggregate evaluation results across all questions."""
    if not results:
        return {}

    # Overall metrics
    total = len(results)
    parsed = sum(1 for r in results if r["parsed_successfully"])

    overall = {
        "total_questions": total,
        "parsed_successfully": parsed,
        "parse_rate": parsed / total if total > 0 else 0,
        "success_rate": sum(1 for r in results if r.get("success")) / total,
        "avg_spl": np.mean([r["spl"] for r in results]),
        "avg_collision_rate": np.mean([r["collision_rate"] for r in results if r["collision_rate"] >= 0]),
        "avg_path_deviation": np.mean([r["path_deviation"] for r in results if r["path_deviation"] != float("inf")]) if any(r["path_deviation"] != float("inf") for r in results) else float("inf"),
        "avg_path_smoothness": np.mean([r["path_smoothness"] for r in results if r["path_smoothness"] != float("inf")]) if any(r["path_smoothness"] != float("inf") for r in results) else float("inf"),
    }

    # Per-level metrics
    per_level = {}
    for level in sorted(set(r["level"] for r in results)):
        level_results = [r for r in results if r["level"] == level]
        n = len(level_results)
        level_name = level_results[0]["level_name"] if level_results else f"level_{level}"

        per_level[f"level_{level}_{level_name}"] = {
            "count": n,
            "parse_rate": sum(1 for r in level_results if r["parsed_successfully"]) / n,
            "success_rate": sum(1 for r in level_results if r.get("success")) / n,
            "avg_spl": float(np.mean([r["spl"] for r in level_results])),
            "avg_collision_rate": float(np.mean([r["collision_rate"] for r in level_results if r["collision_rate"] >= 0])) if any(r["collision_rate"] >= 0 for r in level_results) else -1,
            "avg_path_deviation": float(np.mean([r["path_deviation"] for r in level_results if r["path_deviation"] != float("inf")])) if any(r["path_deviation"] != float("inf") for r in level_results) else float("inf"),
        }

    # Per-source metrics
    per_source = {}
    for source in sorted(set(r.get("source", "internscene") for r in results)):
        source_results = [r for r in results if r.get("source", "internscene") == source]
        n = len(source_results)
        per_source[source] = {
            "count": n,
            "parse_rate": sum(1 for r in source_results if r["parsed_successfully"]) / n,
            "success_rate": sum(1 for r in source_results if r.get("success")) / n,
            "avg_spl": float(np.mean([r["spl"] for r in source_results])),
            "avg_collision_rate": float(np.mean([r["collision_rate"] for r in source_results if r["collision_rate"] >= 0])) if any(r["collision_rate"] >= 0 for r in source_results) else -1,
            "avg_path_deviation": float(np.mean([r["path_deviation"] for r in source_results if r["path_deviation"] != float("inf")])) if any(r["path_deviation"] != float("inf") for r in source_results) else float("inf"),
        }

    return {"overall": overall, "per_level": per_level, "per_source": per_source}


def print_results_table(aggregated: dict):
    """Print formatted results table."""
    print("\n" + "=" * 80)
    print("NavBench3D Evaluation Results")
    print("=" * 80)

    overall = aggregated.get("overall", {})
    print(f"\nOverall ({overall.get('total_questions', 0)} questions):")
    print(f"  Parse Rate:       {overall.get('parse_rate', 0):.1%}")
    print(f"  Success Rate:     {overall.get('success_rate', 0):.1%}")
    print(f"  SPL:              {overall.get('avg_spl', 0):.3f}")
    print(f"  Collision Rate:   {overall.get('avg_collision_rate', 0):.1%}")
    print(f"  Path Deviation:   {overall.get('avg_path_deviation', 0):.3f}m")
    print(f"  Path Smoothness:  {overall.get('avg_path_smoothness', 0):.3f}rad")

    print(f"\nPer-Level Breakdown:")
    print(f"{'Level':<25} {'Count':>6} {'Parse%':>8} {'SR':>8} {'SPL':>8} {'Coll%':>8} {'Dev(m)':>8}")
    print("-" * 75)

    for level_key, metrics in sorted(aggregated.get("per_level", {}).items()):
        print(f"{level_key:<25} {metrics['count']:>6} {metrics['parse_rate']:>7.1%} "
              f"{metrics['success_rate']:>7.1%} {metrics['avg_spl']:>7.3f} "
              f"{metrics['avg_collision_rate']:>7.1%} "
              f"{metrics['avg_path_deviation']:>7.3f}" if metrics['avg_path_deviation'] != float('inf') else
              f"{level_key:<25} {metrics['count']:>6} {metrics['parse_rate']:>7.1%} "
              f"{metrics['success_rate']:>7.1%} {metrics['avg_spl']:>7.3f} "
              f"{metrics['avg_collision_rate']:>7.1%} {'inf':>7}")

    print(f"\nPer-Source Breakdown:")
    print(f"{'Source':<16} {'Count':>6} {'Parse%':>8} {'SR':>8} {'SPL':>8} {'Coll%':>8} {'Dev(m)':>8}")
    print("-" * 70)
    for source, metrics in sorted(aggregated.get("per_source", {}).items()):
        if metrics["avg_path_deviation"] != float("inf"):
            print(
                f"{source:<16} {metrics['count']:>6} {metrics['parse_rate']:>7.1%} "
                f"{metrics['success_rate']:>7.1%} {metrics['avg_spl']:>7.3f} "
                f"{metrics['avg_collision_rate']:>7.1%} {metrics['avg_path_deviation']:>7.3f}"
            )
        else:
            print(
                f"{source:<16} {metrics['count']:>6} {metrics['parse_rate']:>7.1%} "
                f"{metrics['success_rate']:>7.1%} {metrics['avg_spl']:>7.3f} "
                f"{metrics['avg_collision_rate']:>7.1%} {'inf':>7}"
            )

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Evaluate NavBench3D Predictions")
    parser.add_argument("--predictions", required=True, help="JSON file with VLM predictions")
    parser.add_argument("--gt", required=True, help="Ground truth questions JSON")
    parser.add_argument("--gt-dir", default=None, help="GT paths dir for traversability-based collision checking")
    parser.add_argument("--scenes-dir", default=None, help="Scenes dir for fallback AABB collision checking")
    parser.add_argument("--output", default=None, help="Output results JSON")
    parser.add_argument("--threshold", type=float, default=1.0, help="Success distance threshold (m)")
    args = parser.parse_args()

    # Load data
    with open(args.predictions) as f:
        predictions = json.load(f)

    with open(args.gt) as f:
        questions = json.load(f)

    # Build prediction lookup
    pred_lookup = {}
    for pred in predictions:
        pred_lookup[pred["question_id"]] = pred

    # Load layouts for fallback collision checking
    layouts = {}
    if args.scenes_dir:
        scenes_path = Path(args.scenes_dir)
        for scene_dir in scenes_path.iterdir():
            layout_file = scene_dir / "layout.json"
            if layout_file.exists():
                with open(layout_file) as f:
                    layouts[scene_dir.name] = json.load(f)

    # Evaluate
    results = []
    for q in questions:
        qid = q["question_id"]
        pred = pred_lookup.get(qid, {"question_id": qid, "output": ""})
        layout = layouts.get(q["scene_id"])

        result = evaluate_single(pred, q, gt_dir=args.gt_dir, layout=layout)
        results.append(result)

    # Aggregate
    aggregated = aggregate_results(results)

    # Print
    print_results_table(aggregated)

    # Save
    output_path = args.output or "evaluation_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "aggregated": aggregated,
            "per_question": results,
        }, f, indent=2, default=str)
    print(f"\nDetailed results saved to {output_path}")


if __name__ == "__main__":
    main()
