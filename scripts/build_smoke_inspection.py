#!/usr/bin/env python3
"""Build smoke-test inspection overlays for navigability fixes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


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


def load_visible(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return data.get("visible_waypoints", [])


def choose_records(records: list[dict], limit: int) -> list[dict]:
    chosen = []
    seen_scenes: set[str] = set()
    for rec in records:
        scene_id = str(rec.get("scene_id", ""))
        if not scene_id or scene_id in seen_scenes:
            continue
        chosen.append(rec)
        seen_scenes.add(scene_id)
        if len(chosen) >= limit:
            break
    return chosen


def load_font(size: int):
    for font_path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()


def draw_marker(draw: ImageDraw.ImageDraw, xy: tuple[int, int], radius: int,
                fill: tuple[int, int, int, int], outline=(0, 0, 0, 255)) -> None:
    x, y = xy
    draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                 fill=fill, outline=outline, width=2)


def draw_square(draw: ImageDraw.ImageDraw, xy: tuple[int, int], radius: int,
                fill: tuple[int, int, int, int], outline=(0, 0, 0, 255)) -> None:
    x, y = xy
    draw.rectangle((x - radius, y - radius, x + radius, y + radius),
                   fill=fill, outline=outline, width=2)


def draw_cross(draw: ImageDraw.ImageDraw, xy: tuple[int, int], radius: int,
               color: tuple[int, int, int, int]) -> None:
    x, y = xy
    draw.line((x - radius, y, x + radius, y), fill=color, width=3)
    draw.line((x, y - radius, x, y + radius), fill=color, width=3)


def label_point(draw: ImageDraw.ImageDraw, font, xy: tuple[int, int], text: str) -> None:
    tx = xy[0] + 10
    ty = xy[1] - 10
    bbox = font.getbbox(text)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    draw.rectangle((tx - 3, ty - 2, tx + w + 3, ty + h + 2), fill=(0, 0, 0, 165))
    draw.text((tx, ty), text, font=font, fill=(255, 255, 255, 255))


def text_block(draw: ImageDraw.ImageDraw, font, lines: list[str]) -> None:
    line_h = (font.getbbox("Ag")[3] - font.getbbox("Ag")[1]) + 4
    widths = [font.getbbox(line)[2] - font.getbbox(line)[0] for line in lines]
    box_w = max(widths, default=0) + 16
    box_h = max(1, len(lines)) * line_h + 12
    draw.rectangle((10, 10, 10 + box_w, 10 + box_h), fill=(0, 0, 0, 180))
    y = 16
    for line in lines:
        draw.text((18, y), line, font=font, fill=(255, 255, 255, 255))
        y += line_h


def build_mapping(visible: list[dict]) -> dict[int, dict]:
    out = {}
    for wp in visible:
        try:
            out[int(wp["display_id"])] = wp
        except Exception:
            continue
    return out


def draw_path(draw: ImageDraw.ImageDraw, pts: list[tuple[int, int]]) -> None:
    if len(pts) < 2:
        return
    draw.line(pts, fill=(0, 220, 255, 220), width=4)


def point_radius_px(depth_m: float,
                    img_h: int,
                    fov_deg: float,
                    min_px: int = 6,
                    max_px: int = 10) -> int:
    if depth_m <= 1e-6:
        return max_px
    half_h = depth_m * math.tan(math.radians(fov_deg) / 2.0)
    if half_h <= 1e-6:
        return max_px
    px_per_meter = img_h / (2.0 * half_h)
    radius_px = int(round(px_per_meter * 0.045))
    return max(min_px, min(radius_px, max_px))


def render_record(rec: dict, out_path: Path, fov_deg: float) -> dict | None:
    task = str(rec.get("task"))
    view_dir = Path(rec.get("visible_waypoints_path", "")).parent
    target_id = rec.get("target_id")
    base_path = view_dir / f"path_target_{target_id}.png"
    if not base_path.exists():
        base_path = view_dir / "rgb.png"
    if not base_path.exists():
        return None

    cls_key = "pointmass_walkable" if task.startswith("a") else "embodied_feasible"
    route_visible = load_visible(Path(rec.get("visible_waypoints_path", "")))
    if not route_visible:
        return None

    route_map = build_mapping(route_visible)
    img = Image.open(base_path).convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")
    font = load_font(18)
    small_font = load_font(16)
    img_h = img.size[1]

    walkable_count = 0
    blocked_count = 0
    for wp in route_visible:
        xy_raw = wp.get("image_xy")
        if xy_raw is None:
            continue
        xy = (int(xy_raw[0]), int(xy_raw[1]))
        radius = point_radius_px(float(wp.get("depth", 1.0)), img_h, fov_deg)
        is_free = bool(wp.get(cls_key, False))
        if is_free:
            walkable_count += 1
            draw_marker(draw, xy, radius, (46, 204, 113, 210))
        else:
            blocked_count += 1
            draw_marker(draw, xy, radius, (231, 76, 60, 220))
            draw_cross(draw, xy, max(4, radius - 2), (255, 245, 245, 220))
        label_point(draw, small_font, xy, str(wp.get("display_id", "?")))

    path_pts = []
    for display_id in rec.get("path_ids", []):
        wp = route_map.get(int(display_id))
        if wp and wp.get("image_xy") is not None:
            xy = wp["image_xy"]
            path_pts.append((int(xy[0]), int(xy[1])))
    draw_path(draw, path_pts)

    start_id = rec.get("start_id")
    if start_id is not None and int(start_id) in route_map:
        xy = route_map[int(start_id)]["image_xy"]
        start_xy = (int(xy[0]), int(xy[1]))
        draw_square(draw, start_xy, 12, (52, 152, 219, 220))
        label_point(draw, font, start_xy, f"S{int(start_id)}")

    for goal_id in rec.get("goal_ids", []):
        if int(goal_id) not in route_map:
            continue
        xy = route_map[int(goal_id)]["image_xy"]
        goal_xy = (int(xy[0]), int(xy[1]))
        draw_square(draw, goal_xy, 12, (241, 196, 15, 230))
        label_point(draw, font, goal_xy, f"G{int(goal_id)}")

    lines = [
        f"{rec.get('scene_id')}  view={rec.get('view_id')}  task={task}",
        f"target={rec.get('target_category')}#{target_id}  dist={rec.get('target_distance_m')}m",
        f"walkable={walkable_count}  blocked={blocked_count}  path_nodes={len(rec.get('path_ids', []))}",
        "green=walkable  red=blocked  blue=start  yellow=goal  cyan=GT path",
    ]
    text_block(draw, font, lines)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return {
        "question_id": rec.get("question_id"),
        "task": task,
        "scene_id": rec.get("scene_id"),
        "view_id": rec.get("view_id"),
        "target_id": target_id,
        "target_category": rec.get("target_category"),
        "output_image": str(out_path),
    }


def write_index(rows: list[dict], out_dir: Path) -> None:
    index_path = out_dir / "index.html"
    with open(index_path, "w") as f:
        f.write("<html><head><meta charset='utf-8'><title>Smoke Inspection</title>")
        f.write("<style>body{font-family:Arial,sans-serif;background:#111;color:#eee;}")
        f.write(".card{display:inline-block;vertical-align:top;width:460px;margin:12px;}")
        f.write("img{max-width:440px;border:1px solid #444;} .meta{font-size:12px;line-height:1.4;}</style>")
        f.write("</head><body><h2>NavBench3D Smoke Inspection</h2>")
        f.write(f"<p>Images: {len(rows)}</p>")
        for row in rows:
            rel = Path(row["output_image"]).relative_to(out_dir)
            f.write("<div class='card'>")
            f.write(f"<img src='{rel.as_posix()}' alt='{row['question_id']}'><div class='meta'>")
            f.write(f"<div>{row['question_id']}</div>")
            f.write(f"<div>scene={row['scene_id']} view={row['view_id']} task={row['task']}</div>")
            f.write(f"<div>target={row['target_category']}#{row['target_id']}</div>")
            f.write("</div></div>")
        f.write("</body></html>")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build smoke inspection overlays.")
    parser.add_argument("--gt-dir", required=True, help="Directory with gt_next_*.jsonl")
    parser.add_argument("--output-dir", required=True, help="Inspection output directory")
    parser.add_argument("--limit", type=int, default=20, help="Max scenes per task")
    parser.add_argument("--fov-deg", type=float, default=120.0, help="Render vertical FOV")
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for task in ("a2", "b2"):
        records = load_jsonl(gt_dir / f"gt_next_{task}.jsonl")
        for rec in choose_records(records, args.limit):
            scene_id = str(rec.get("scene_id"))
            qid = str(rec.get("question_id"))
            out_path = out_dir / task / scene_id / f"{qid}.png"
            row = render_record(rec, out_path, args.fov_deg)
            if row is not None:
                all_rows.append(row)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(all_rows, f, indent=2)
    write_index(all_rows, out_dir)
    print(f"Wrote {len(all_rows)} inspection images to {out_dir}")


if __name__ == "__main__":
    main()
