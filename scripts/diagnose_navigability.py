#!/usr/bin/env python3
"""
Offline navigability diagnostic — no Blender needed.
Reads scene JSON, builds occupancy grid, does flood-fill from
multiple candidate positions, reports reachable area and
whether any distant object is reachable.

Usage:
    python scripts/diagnose_navigability.py \
        --scenes-dir /datadisk/NavBench3D/scenes \
        --scene-ids scene1 scene2 ...
"""
import argparse
import json
import math
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np

# Must match render_scenes.py
REMOVABLE_CATEGORIES = {
    "shoe", "shoes", "sneakers", "boots", "sandals", "slippers",
    "bag", "backpack", "suitcase", "purse", "briefcase",
    "hat", "cap", "helmet",
    "toy", "teddy bear", "stuffed animal", "doll",
    "bottle", "can", "cup", "mug", "glass",
    "book", "books", "magazine", "paper", "newspaper",
    "clothes", "clothing", "jacket", "coat", "shirt", "pants",
    "towel", "blanket", "rug", "mat", "carpet",
    "pillow", "cushion",
    "box", "carton", "package",
    "basket", "bin", "trash", "garbage", "waste",
    "case", "container", "crate", "bucket",
    "hamper", "laundry",
    "board",
    "cable", "cord", "wire",
    "plant", "flower", "pot",
    "scale", "object", "item", "stuff", "misc",
    "umbrella", "broom", "mop",
    "toilet paper", "tissue",
    "stopcock", "dispenser",
    "teapot", "kettle",
}

KEEP_CATEGORIES = {
    "table", "desk", "counter", "countertop",
    "chair", "stool", "bench", "seat", "armchair",
    "couch", "sofa", "loveseat",
    "bed", "crib", "bunk",
    "cabinet", "dresser", "wardrobe", "closet", "shelf", "shelves",
    "stand", "nightstand", "bookshelf", "bookcase",
    "refrigerator", "fridge", "oven", "stove", "microwave",
    "sink", "bathtub", "shower", "toilet",
    "door", "window",
    "tv", "television", "monitor", "screen",
    "lamp", "light", "chandelier",
    "radiator", "heater", "air conditioner",
    "washer", "dryer", "dishwasher",
    "piano", "fireplace",
    "computer",
}


def _is_removable(obj):
    bbox = obj.get("bbox", [])
    if len(bbox) < 6:
        return False
    cat = obj.get("category", "").lower().strip()
    x, y, z, sx, sy, sz = bbox[:6]
    obj_bottom = z - sz / 2
    obj_top = z + sz / 2
    vol = sx * sy * sz
    footprint = sx * sy
    for kc in KEEP_CATEGORIES:
        if kc in cat or cat in kc:
            return False
    if obj_bottom > 0.20:
        return False
    if obj_top > 0.6 or footprint > 0.5 or vol > 0.2:
        return False
    for rc in REMOVABLE_CATEGORIES:
        if rc in cat or cat in rc:
            return True
    if vol < 0.05 and footprint < 0.15:
        return True
    return False


def build_occupancy(layout, room_bounds, resolution=0.1,
                    robot_radius=0.30, robot_height=1.4):
    min_x, min_y, max_x, max_y = room_bounds
    w = max(1, int((max_x - min_x) / resolution))
    h = max(1, int((max_y - min_y) / resolution))
    grid = np.zeros((h, w), dtype=bool)

    wall_cells = max(1, int(robot_radius / resolution))
    grid[:wall_cells, :] = True
    grid[-wall_cells:, :] = True
    grid[:, :wall_cells] = True
    grid[:, -wall_cells:] = True

    for obj in layout:
        bbox = obj.get("bbox", [])
        if len(bbox) < 6:
            continue
        x, y, z = bbox[0], bbox[1], bbox[2]
        sx, sy, sz = bbox[3], bbox[4], bbox[5]
        obj_bottom = z - sz / 2
        obj_top = z + sz / 2
        if obj_bottom < robot_height and obj_top > 0.10:
            half_sx = sx / 2 + robot_radius
            half_sy = sy / 2 + robot_radius
            gx1 = max(0, int((x - half_sx - min_x) / resolution))
            gy1 = max(0, int((y - half_sy - min_y) / resolution))
            gx2 = min(w, int((x + half_sx - min_x) / resolution))
            gy2 = min(h, int((y + half_sy - min_y) / resolution))
            grid[gy1:gy2, gx1:gx2] = True

    return grid


def flood_fill(grid, start_r, start_c):
    h, w = grid.shape
    sr, sc = int(start_r), int(start_c)
    if not (0 <= sr < h and 0 <= sc < w and not grid[sr, sc]):
        return set()
    visited = {(sr, sc)}
    q = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and not grid[nr, nc]:
                visited.add((nr, nc))
                q.append((nr, nc))
    return visited


def get_room_bounds_from_layout(layout):
    """Estimate room bounds from object bounding boxes (no Blender)."""
    xs, ys = [], []
    for obj in layout:
        bbox = obj.get("bbox", [])
        if len(bbox) >= 6:
            x, y, sx, sy = bbox[0], bbox[1], bbox[3], bbox[4]
            xs.extend([x - sx/2, x + sx/2])
            ys.extend([y - sy/2, y + sy/2])
    if not xs:
        return (-3, -3, 3, 3)
    margin = 0.3
    return (min(xs) - margin, min(ys) - margin,
            max(xs) + margin, max(ys) + margin)


def diagnose_scene(scene_dir, scene_id):
    layout_file = os.path.join(scene_dir, scene_id, "layout.json")
    if not os.path.exists(layout_file):
        return {"scene_id": scene_id, "error": "no layout.json"}

    with open(layout_file) as f:
        layout = json.load(f)

    n_objs = len([o for o in layout if o.get("model_uid")])
    room_bounds = get_room_bounds_from_layout(layout)
    min_x, min_y, max_x, max_y = room_bounds
    room_w = max_x - min_x
    room_h = max_y - min_y
    room_area = room_w * room_h
    room_diag = math.hypot(room_w, room_h)

    resolution = 0.1
    grid = build_occupancy(layout, room_bounds, resolution)
    h, w = grid.shape
    total_cells = h * w
    free_cells = int(np.sum(~grid))
    free_area = free_cells * resolution * resolution
    free_pct = free_cells / total_cells * 100

    # Find best flood-fill area from free cells (sample multiple starts)
    free_indices = np.argwhere(~grid)
    if len(free_indices) == 0:
        return {
            "scene_id": scene_id,
            "n_objs": n_objs,
            "room_area": room_area,
            "room_diag": round(room_diag, 2),
            "free_area": 0,
            "free_pct": 0,
            "max_reachable_area": 0,
            "nav_pass": False,
            "n_far": 0,
            "n_reachable": 0,
            "n_floor_clutter": 0,
            "floor_clutter": [],
            "reason": "no free cells",
        }

    # Sample up to 20 random start points
    n_samples = min(20, len(free_indices))
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(len(free_indices), n_samples, replace=False)

    best_reachable = set()
    for idx in sample_idx:
        r, c = free_indices[idx]
        reach = flood_fill(grid, r, c)
        if len(reach) > len(best_reachable):
            best_reachable = reach

    max_reach_area = len(best_reachable) * resolution * resolution
    MIN_REACHABLE_AREA = 2.0

    # Check reachable targets
    MIN_NAV_DIST = max(1.0, min(2.0, room_diag * 0.15))
    GOAL_SEARCH_R = 15

    # Pick a sample camera position from the reachable set
    if best_reachable:
        # Pick a cell near the edge of the reachable set
        sample_cells = list(best_reachable)
        mid_idx = len(sample_cells) // 2
        cam_r, cam_c = sample_cells[mid_idx]
        cx = min_x + (cam_c + 0.5) * resolution
        cy = min_y + (cam_r + 0.5) * resolution
    else:
        cx, cy = (min_x + max_x) / 2, (min_y + max_y) / 2

    # Check how many objects are "far" and "reachable"
    n_far = 0
    n_reachable = 0
    blocking_objs = []  # objects on floor that block navigation
    for obj in layout:
        bbox = obj.get("bbox", [])
        if len(bbox) < 6:
            continue
        ox, oy, oz = bbox[0], bbox[1], bbox[2]
        sx, sy, sz = bbox[3], bbox[4], bbox[5]
        cat = obj.get("category", "?")

        obj_dist = math.hypot(ox - cx, oy - cy)
        if obj_dist >= MIN_NAV_DIST:
            n_far += 1
            obj_r = int((oy - min_y) / resolution)
            obj_c = int((ox - min_x) / resolution)
            found = False
            for dr in range(-GOAL_SEARCH_R, GOAL_SEARCH_R + 1):
                for dc in range(-GOAL_SEARCH_R, GOAL_SEARCH_R + 1):
                    if (obj_r + dr, obj_c + dc) in best_reachable:
                        found = True
                        break
                if found:
                    break
            if found:
                n_reachable += 1

        # Identify small floor objects that might be removable
        obj_bottom = oz - sz / 2
        obj_top = oz + sz / 2
        obj_vol = sx * sy * sz
        obj_footprint = sx * sy
        if (obj_bottom < 0.15 and obj_top < 0.8
                and obj_vol < 0.3 and obj_footprint < 0.5
                and obj.get("model_uid")):
            blocking_objs.append({
                "category": cat,
                "id": obj.get("id", 0),
                "pos": (ox, oy, oz),
                "size": (sx, sy, sz),
                "vol": obj_vol,
                "footprint": obj_footprint,
            })

    nav_pass = max_reach_area >= MIN_REACHABLE_AREA and n_reachable > 0

    return {
        "scene_id": scene_id,
        "n_objs": n_objs,
        "room_area": round(room_area, 1),
        "room_diag": round(room_diag, 2),
        "free_area": round(free_area, 1),
        "free_pct": round(free_pct, 1),
        "max_reachable_area": round(max_reach_area, 1),
        "MIN_NAV_DIST": round(MIN_NAV_DIST, 2),
        "n_far": n_far,
        "n_reachable": n_reachable,
        "nav_pass": nav_pass,
        "n_floor_clutter": len(blocking_objs),
        "floor_clutter": blocking_objs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes-dir", required=True)
    parser.add_argument("--scene-ids", nargs="+", default=None,
                        help="Scene IDs to check (all if omitted)")
    parser.add_argument("--simulate-cleanup", action="store_true",
                        help="Simulate removing floor clutter for failing scenes")
    args = parser.parse_args()

    if args.scene_ids:
        scene_ids = args.scene_ids
    else:
        scene_ids = sorted(os.listdir(args.scenes_dir))

    results = []
    pass_count = 0
    fail_count = 0
    clutter_fixable = 0
    cleanup_fixed = 0

    for sid in scene_ids:
        try:
            r = diagnose_scene(args.scenes_dir, sid)
        except Exception as e:
            print(f"  ERR  {sid}: {e}")
            fail_count += 1
            results.append({"scene_id": sid, "error": str(e)})
            continue
        results.append(r)
        if r.get("error"):
            print(f"  ERR  {sid}: {r['error']}")
            fail_count += 1
            continue

        status = "PASS" if r["nav_pass"] else "FAIL"
        clutter_info = ""
        if not r["nav_pass"] and r.get("n_floor_clutter", 0) > 0:
            clutter_fixable += 1
            cats = [o["category"] for o in r.get("floor_clutter", [])]
            clutter_info = f" | clutter: {', '.join(cats[:5])}"

        print(f"  {status}  {sid}: "
              f"objs={r.get('n_objs', '?')}, room={r.get('room_area', '?')}m², "
              f"free={r.get('free_area', '?')}m² ({r.get('free_pct', '?')}%), "
              f"reachable={r.get('max_reachable_area', '?')}m², "
              f"far={r.get('n_far', '?')}, reach_far={r.get('n_reachable', '?')}"
              f"{clutter_info}")

        # Simulate cleanup for failing scenes
        if not r.get("nav_pass") and args.simulate_cleanup:
            try:
                layout_file = os.path.join(args.scenes_dir, sid, "layout.json")
                layout = json.load(open(layout_file))
                room_bounds = get_room_bounds_from_layout(layout)
                removable = [(i, o) for i, o in enumerate(layout) if _is_removable(o)]
                if removable:
                    cleaned = [o for i, o in enumerate(layout)
                              if not _is_removable(o)]
                    new_area = 0
                    grid = build_occupancy(cleaned, room_bounds)
                    free_idx = np.argwhere(~grid)
                    if len(free_idx) > 0:
                        rng = np.random.RandomState(42)
                        n_s = min(10, len(free_idx))
                        si = rng.choice(len(free_idx), n_s, replace=False)
                        best = 0
                        for ix in si:
                            reach = flood_fill(grid, int(free_idx[ix, 0]), int(free_idx[ix, 1]))
                            if len(reach) > best:
                                best = len(reach)
                        new_area = best * 0.1 * 0.1
                    removed_cats = [o.get("category", "?") for _, o in removable]
                    fixed = new_area >= 2.0
                    if fixed:
                        cleanup_fixed += 1
                    print(f"        -> cleanup: removed {len(removable)} "
                          f"({', '.join(removed_cats[:5])}), "
                          f"area={new_area:.1f}m² {'FIXED!' if fixed else 'still fail'}")
            except Exception as e:
                print(f"        -> cleanup error: {e}")
        if r.get("nav_pass"):
            pass_count += 1
        else:
            fail_count += 1

    total = pass_count + fail_count
    print(f"\n{'='*60}")
    print(f"SUMMARY: {pass_count}/{total} pass ({pass_count/total*100:.1f}%), "
          f"{fail_count} fail")
    print(f"  Of {fail_count} failures, {clutter_fixable} have floor clutter "
          f"that could be cleaned")
    if args.simulate_cleanup:
        print(f"  Cleanup simulation: {cleanup_fixed}/{fail_count} failures "
              f"fixed by removing clutter")
        print(f"  After cleanup: {pass_count + cleanup_fixed}/{total} pass "
              f"({(pass_count + cleanup_fixed)/total*100:.1f}%)")

    # List all floor clutter categories across failing scenes
    clutter_cats = {}
    for r in results:
        if not r.get("nav_pass") and r.get("floor_clutter"):
            for o in r["floor_clutter"]:
                cat = o["category"]
                clutter_cats[cat] = clutter_cats.get(cat, 0) + 1
    if clutter_cats:
        print(f"\nFloor clutter categories in failing scenes:")
        for cat, cnt in sorted(clutter_cats.items(), key=lambda x: -x[1]):
            print(f"  {cat}: {cnt}")


if __name__ == "__main__":
    main()
