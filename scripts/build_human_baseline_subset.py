#!/usr/bin/env python3
"""Build a fixed human-baseline subset and browser task package."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path


TASK_NAMES = {
    "a1": "点级可通行性",
    "b1": "机器人可通行性",
    "a2": "显式目标点路径",
    "b2": "显式目标机器人路径",
    "c": "意图目标机器人路径",
}

OBJECT_NAMES_ZH = {
    "backpack": "背包",
    "basket": "篮子",
    "book": "书",
    "bottle": "瓶子",
    "bowl": "碗",
    "cabinet": "柜子",
    "carpet": "地毯",
    "case": "箱子",
    "chair": "椅子",
    "clothes": "衣服",
    "couch": "沙发",
    "cup": "杯子",
    "cushion": "垫子",
    "desk": "书桌",
    "door": "门",
    "fireplace": "壁炉",
    "fruit": "水果",
    "guitar": "吉他",
    "knife": "刀",
    "lamp": "灯",
    "microwave": "微波炉",
    "monitor": "显示器",
    "oven": "烤箱",
    "pack": "包",
    "picture": "画",
    "pillow": "枕头",
    "plant": "植物",
    "rack": "架子",
    "radiator": "暖气片",
    "refrigerator": "冰箱",
    "shelf": "架子",
    "shoe": "鞋",
    "sink": "水槽",
    "stool": "凳子",
    "stove": "炉灶",
    "table": "桌子",
    "towel": "毛巾",
    "toy": "玩具",
    "vase": "花瓶",
    "washing machine": "洗衣机",
    "water cooler": "饮水机",
    "window": "窗户",
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def visible_display_ids(row: dict) -> list[int]:
    path = Path(row["visible_waypoints_path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    waypoints = payload["visible_waypoints"] if isinstance(payload, dict) else payload
    ids = []
    for waypoint in waypoints:
        display_id = waypoint.get("display_id")
        if display_id is not None:
            ids.append(int(display_id))
    return sorted(set(ids))


def object_zh(name: str) -> str:
    return OBJECT_NAMES_ZH.get(name.strip().lower(), name.strip())


def relation_target_zh(target: str) -> str:
    target = target.strip().removeprefix("the ")
    for relation, template in [
        ("under the ", "{anchor}下方"),
        ("on the ", "{anchor}上面"),
        ("near the ", "{anchor}附近"),
        ("inside the ", "{anchor}里面"),
    ]:
        if target.startswith(relation):
            anchor = target.removeprefix(relation)
            return template.format(anchor=object_zh(anchor))
    for relation, template in [
        (" near ", "靠近{anchor}的{obj}"),
        (" under ", "在{anchor}下方的{obj}"),
        (" on ", "在{anchor}上面的{obj}"),
        (" inside ", "在{anchor}里面的{obj}"),
    ]:
        marker = f"{relation}the "
        if marker in target:
            obj, anchor = target.split(marker, 1)
            return template.format(obj=object_zh(obj), anchor=object_zh(anchor))
    return object_zh(target)


def option_zh(option: str) -> str:
    return {"closest": "最近", "nearer": "较近"}.get(option, option)


def translate_request(request: str) -> str:
    request = request.strip()
    patterns = [
        (
            "I want to set something down near the ",
            "我想在靠近{anchor}的地方放下东西，具体选择{option}的那个选项。",
        ),
        (
            "I want to put something away near the ",
            "我想把东西收起来，位置靠近{anchor}，具体选择{option}的那个选项。",
        ),
        (
            "I want to take a seat near the ",
            "我想在靠近{anchor}的地方坐下，具体选择{option}的那个选项。",
        ),
        (
            "I want to sit down near the ",
            "我想在靠近{anchor}的地方坐下，具体选择{option}的那个选项。",
        ),
        (
            "I need a place to sit near the ",
            "我需要一个靠近{anchor}、可以坐下的地方，具体选择{option}的那个选项。",
        ),
        (
            "I need somewhere to put something away near the ",
            "我需要一个靠近{anchor}、可以把东西收起来的地方，具体选择{option}的那个选项。",
        ),
        (
            "I need somewhere to place something near the ",
            "我需要一个靠近{anchor}、可以放东西的地方，具体选择{option}的那个选项。",
        ),
    ]
    for prefix, template in patterns:
        if request.startswith(prefix):
            rest = request[len(prefix) :]
            suffix = ", specifically the "
            if suffix in rest:
                anchor, option = rest.split(suffix, 1)
                option = option.removesuffix(" option.")
                return template.format(anchor=object_zh(anchor), option=option_zh(option))

    if request.startswith("I want to head out near the "):
        rest = request.removeprefix("I want to head out near the ")
        anchor, option = rest.split(", specifically the ", 1)
        option = option.removesuffix(" option.")
        return f"我想找靠近{object_zh(anchor)}的出口，具体选择{option_zh(option)}的那个选项。"

    storage_prefixes = [
        ("I need a place where I can store something ", "我需要一个可以收纳东西的地方，位置在{target}。"),
        ("I want to put something away ", "我想把东西收起来，位置在{target}。"),
        ("I need somewhere to put something away ", "我需要一个可以把东西收起来的地方，位置在{target}。"),
    ]
    for prefix, template in storage_prefixes:
        if request.startswith(prefix):
            target = request.removeprefix(prefix).removesuffix(".")
            return template.format(target=relation_target_zh(target))

    return request


def prompt_zh(row: dict) -> str:
    prompt = row.get("prompt_text") or row.get("user_prompt")
    task = row["task"]
    if task == "a1":
        return "图中显示了带编号的候选点。请输出所有可步行到达的显示 ID，答案必须是 JSON 数组。"
    if task == "b1":
        return "图中显示了带编号的候选点。机器人直径为 0.6 米。请输出机器人可以安全通行的所有显示 ID，答案必须是 JSON 数组。"

    if task in {"a2", "b2"}:
        start = "Navigate to "
        end = ". Output a JSON array" if task == "a2" else ". Return a JSON array"
        target = prompt.split(start, 1)[1].split(end, 1)[0]
        target_text = relation_target_zh(target)
        if task == "a2":
            return (
                "图中显示了带编号的候选点。请从显示 ID 1 出发，"
                f"导航到{target_text}。请输出一个由显示 ID 组成的 JSON 数组，表示一条有效路径。"
            )
        return (
            "图中显示了带编号的候选点。你是一个直径 0.6 米的机器人。请从显示 ID 1 出发，"
            f"导航到{target_text}。请输出一个由显示 ID 组成的 JSON 数组，表示一条对机器人有效且无碰撞的路径。"
        )

    request = prompt.split('A person says: "', 1)[1].split('" Return a JSON array', 1)[0]
    return (
        "图中显示了带编号的候选点。你是一个直径 0.6 米的机器人。请从显示 ID 1 出发。"
        f'有人说：“{translate_request(request)}”'
        "请输出一个由显示 ID 组成的 JSON 数组，表示一条到达你推断出的目标、且对机器人有效无碰撞的路径。"
    )


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_row(row: dict, image_name: str, ordinal: int) -> dict:
    return {
        "subset_index": ordinal,
        "question_id": row["question_id"],
        "task": row["task"],
        "task_name": TASK_NAMES[row["task"]],
        "scene_id": row.get("scene_id"),
        "view_id": row.get("view_id"),
        "routing_id": row.get("routing_id"),
        "image": f"images/{image_name}",
        "prompt": prompt_zh(row),
        "prompt_language": "zh-CN",
        "expected_answer_format": "由显示 ID 组成的 JSON 数组",
    }


def key_row(row: dict, image_name: str, image_sha256: str) -> dict:
    ground_truth = row.get("ground_truth") or {}
    return {
        "question_id": row["question_id"],
        "task": row["task"],
        "task_name": TASK_NAMES[row["task"]],
        "scene_id": row.get("scene_id"),
        "view_id": row.get("view_id"),
        "routing_id": row.get("routing_id"),
        "source_image_path": row.get("image_path"),
        "visible_waypoints_path": row.get("visible_waypoints_path"),
        "packaged_image": f"images/{image_name}",
        "image_sha256": image_sha256,
        "candidate_ids": visible_display_ids(row),
        "prompt_en": row.get("prompt_text") or row.get("user_prompt"),
        "prompt_zh": prompt_zh(row),
        "answer": ground_truth.get("answer"),
        "ground_truth": ground_truth,
        "gt_hash": row.get("gt_hash"),
        "prompt_hash": row.get("prompt_hash"),
        "fixed_gt_identity_hash": row.get("fixed_gt_identity_hash"),
    }


def build_subset(args: argparse.Namespace) -> None:
    vqa_root = args.release_root / "data/benchmark_evalfix/benchmark/vqa"
    if not vqa_root.exists():
        vqa_root = args.release_root / "data/release/benchmark/vqa"
    if not vqa_root.exists():
        raise FileNotFoundError(f"Could not find benchmark VQA root under {args.release_root}")

    rng = random.Random(args.seed)
    output_dir = args.output_dir
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    public_questions: list[dict] = []
    private_key: list[dict] = []
    sample_summary: dict[str, dict] = {}

    for task in TASK_NAMES:
        path = vqa_root / f"vqa_next_{task}.jsonl"
        rows = sorted(read_jsonl(path), key=lambda item: item["question_id"])
        available = []
        missing_images = []
        for row in rows:
            image_path = Path(row["image_path"])
            if image_path.exists():
                available.append(row)
            else:
                missing_images.append(row["question_id"])
        if len(available) < args.per_task:
            raise RuntimeError(
                f"Task {task} has only {len(available)} rows with available images, "
                f"but {args.per_task} are required."
            )

        selected = rng.sample(available, args.per_task)
        selected.sort(key=lambda item: item["question_id"])
        sample_summary[task] = {
            "task_name": TASK_NAMES[task],
            "requested": args.per_task,
            "available_rows": len(rows),
            "available_images": len(available),
            "missing_images": missing_images[:20],
        }

        for row in selected:
            source_image = Path(row["image_path"])
            image_name = f"{row['question_id']}.png"
            destination = images_dir / image_name
            shutil.copy2(source_image, destination)
            image_sha256 = digest_file(destination)
            public_questions.append(clean_row(row, image_name, len(public_questions) + 1))
            private_key.append(key_row(row, image_name, image_sha256))

    rng.shuffle(public_questions)
    for index, row in enumerate(public_questions, start=1):
        row["subset_index"] = index

    key_by_id = {row["question_id"]: row for row in private_key}
    private_key = [key_by_id[row["question_id"]] for row in public_questions]

    manifest = {
        "name": "EgoPathBench human baseline subset",
        "seed": args.seed,
        "per_task": args.per_task,
        "total_questions": len(public_questions),
        "source_vqa_root": str(vqa_root),
        "task_counts": {
            task: sum(1 for row in public_questions if row["task"] == task) for task in TASK_NAMES
        },
        "sample_summary": sample_summary,
        "public_questions": "questions.json",
        "private_answer_key": "answer_key_private.json",
    }

    write_json(output_dir / "questions.json", {"manifest": manifest, "questions": public_questions})
    write_json(output_dir / "answer_key_private.json", {"manifest": manifest, "answers": private_key})
    write_json(output_dir / "manifest.json", manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--release-root",
        type=Path,
        default=Path("published/navbench3d_release_20260417_rebuild5"),
        help="NavBench3D release root containing data/benchmark_evalfix.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("human_baseline/subset_n20_seed20260630"),
        help="Directory for the browser package.",
    )
    parser.add_argument("--per-task", type=int, default=20, help="Number of questions per task.")
    parser.add_argument("--seed", type=int, default=20260630, help="Sampling seed.")
    return parser.parse_args()


if __name__ == "__main__":
    build_subset(parse_args())
