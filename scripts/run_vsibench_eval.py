#!/usr/bin/env python3
"""Prepare and evaluate VSI-Bench with a local LLaMA-Factory video VLM."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable

import numpy as np

DEFAULT_MODEL = "/mnt/data/omnimodel/weights/base/Qwen3.5-4B"
DEFAULT_LLAMAFACTORY_ROOT = "/mnt/data/wuchanglin/LLaMA-Factory"
DEFAULT_OUTPUT_ROOT = Path("results/vsibench_eval_20260724")
DATASET_NAME = "nyu-visionx/VSI-Bench"
VIDEO_ARCHIVES = ("arkitscenes.zip", "scannet.zip", "scannetpp.zip")
DEFAULT_VIDEO_FPS = 2.0
DEFAULT_VIDEO_MAXLEN = 32
DEFAULT_VIDEO_MAX_PIXELS = 262144
MCA_QUESTION_TYPES = {
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
}
NA_QUESTION_TYPES = {
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        return
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def build_prompt(row: dict[str, Any]) -> str:
    question = str(row["question"])
    question_type = str(row["question_type"])
    if question_type in NA_QUESTION_TYPES:
        return f"{question}\nPlease answer the question using a single word or phrase."
    if question_type in MCA_QUESTION_TYPES:
        options = "\n".join(str(option) for option in row["options"])
        return (
            f"{question}\nOptions:\n{options}\n"
            "Answer with the option's letter from the given choices directly."
        )
    raise ValueError(f"Unknown VSI-Bench question type: {question_type}")


def fuzzy_first_token(output: object) -> str:
    text = str(output or "").strip()
    if not text:
        return ""
    return text.split()[0].rstrip(".").strip()


def to_float(value: object) -> float | None:
    token = fuzzy_first_token(value)
    try:
        parsed = float(token)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def mean_relative_accuracy(prediction: float | None, target: float | None) -> float:
    if prediction is None or target is None or target == 0:
        return 0.0
    relative_error = abs(prediction - target) / abs(target)
    thresholds = [0.5 + 0.05 * idx for idx in range(10)]
    return sum(relative_error <= 1 - threshold for threshold in thresholds) / len(thresholds)


def score_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[float]] = defaultdict(list)
    success_count = 0
    for row in rows:
        success_count += int(bool(row.get("success")))
        question_type = str(row["question_type"])
        if question_type in MCA_QUESTION_TYPES:
            prediction = fuzzy_first_token(row.get("output"))
            score = float(prediction.lower() == str(row["ground_truth"]).lower())
            values[f"{question_type}_accuracy"].append(score)
        elif question_type in NA_QUESTION_TYPES:
            score = mean_relative_accuracy(to_float(row.get("output")), to_float(row["ground_truth"]))
            values[f"{question_type}_MRA:.5:.95:.05"].append(score)
        else:
            raise ValueError(f"Unknown VSI-Bench question type: {question_type}")

    metric_values = {key: sum(items) / len(items) for key, items in values.items()}
    direction_keys = [
        "object_rel_direction_easy_accuracy",
        "object_rel_direction_medium_accuracy",
        "object_rel_direction_hard_accuracy",
    ]
    if all(key in metric_values for key in direction_keys):
        metric_values["object_rel_direction_accuracy"] = sum(metric_values.pop(key) for key in direction_keys) / 3

    overall = sum(metric_values.values()) / len(metric_values) if metric_values else 0.0
    summary: dict[str, Any] = {
        "count": len(rows),
        "successful_requests": success_count,
        "request_success_rate": round(success_count / len(rows), 6) if rows else 0.0,
        "overall": round(overall * 100, 6),
    }
    summary.update({key: round(value * 100, 6) for key, value in sorted(metric_values.items())})
    return summary


def video_sample_indices(total_frames: int, duration_s: float, fps: float, maxlen: int) -> list[int]:
    sample_frames = max(1, math.floor(duration_s * fps))
    sample_frames = min(total_frames, maxlen, sample_frames)
    return np.linspace(0, total_frames - 1, sample_frames).astype(np.int32).tolist()


def build_cached_video(source_path: Path, cache_path: Path, *, fps: float, maxlen: int) -> None:
    """Cache the uniformly sampled frames that LLaMA-Factory would otherwise decode per question."""
    import av

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return

    source = av.open(str(source_path), "r")
    stream = next(item for item in source.streams if item.type == "video")
    if stream.frames == 0 or stream.duration is None:
        raise ValueError(f"VSI-Bench video lacks finite frame metadata: {source_path}")
    indices = set(
        video_sample_indices(
            int(stream.frames),
            float(stream.duration * stream.time_base),
            fps,
            maxlen,
        )
    )
    frames = []
    source.seek(0)
    for frame_index, frame in enumerate(source.decode(stream)):
        if frame_index in indices:
            frames.append(frame)
    source.close()
    if not frames:
        raise ValueError(f"No frames sampled from VSI-Bench video: {source_path}")

    temporary_path = cache_path.with_suffix(".tmp.mp4")
    target = av.open(str(temporary_path), "w")
    target_stream = target.add_stream("libx264", rate=Fraction(str(fps)))
    target_stream.width = frames[0].width
    target_stream.height = frames[0].height
    target_stream.pix_fmt = "yuv420p"
    for frame in frames:
        for packet in target_stream.encode(frame):
            target.mux(packet)
    for packet in target_stream.encode():
        target.mux(packet)
    target.close()
    temporary_path.replace(cache_path)


def prepare_dataset(output_root: Path, *, cache_videos: bool, video_fps: float, video_maxlen: int) -> Path:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    data_root = output_root / "data"
    video_root = data_root / "videos"
    video_root.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(DATASET_NAME, "full", split="test")
    for archive_name in VIDEO_ARCHIVES:
        dataset_dir = video_root / archive_name.removesuffix(".zip")
        if dataset_dir.exists() and any(dataset_dir.glob("*.mp4")):
            continue
        archive_path = data_root / "archives" / archive_name
        if not archive_path.exists():
            archive_path = Path(hf_hub_download(DATASET_NAME, archive_name, repo_type="dataset"))
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(video_root)

    records = []
    for row in dataset:
        record = dict(row)
        source_path = video_root / row["dataset"] / f"{row['scene_name']}.mp4"
        if cache_videos:
            cached_path = data_root / "sampled_videos" / row["dataset"] / f"{row['scene_name']}.mp4"
            build_cached_video(source_path, cached_path, fps=video_fps, maxlen=video_maxlen)
            record["video_path"] = str(cached_path)
        else:
            record["video_path"] = str(source_path)
        records.append(record)

    manifest_path = data_root / "manifest.jsonl"
    with manifest_path.open("w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
    missing = [row["video_path"] for row in records if not Path(row["video_path"]).exists()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} VSI-Bench videos; first missing path: {missing[0]}")
    return manifest_path


def load_cached_video_frames(video_path: Path) -> tuple[list[Any], float]:
    """Read an already uniformly sampled MP4 into Qwen3.5's native video-input format."""
    import av

    container = av.open(str(video_path), "r")
    stream = next(item for item in container.streams if item.type == "video")
    frames = [frame.to_image() for frame in container.decode(stream)]
    duration_s = float(stream.duration * stream.time_base) if stream.duration is not None else 0.0
    container.close()
    if not frames:
        raise ValueError(f"No frames decoded from VSI-Bench video: {video_path}")
    return frames, len(frames) / duration_s if duration_s > 0 else DEFAULT_VIDEO_FPS


def load_native_qwen(model_name_or_path: str, adapter_name_or_path: str | None):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_name_or_path:
        model = PeftModel.from_pretrained(model, adapter_name_or_path).merge_and_unload()
    model.eval()
    return model, processor, tokenizer


def native_video_response(
    model: Any,
    processor: Any,
    tokenizer: Any,
    *,
    prompt: str,
    video_path: Path,
    video_max_pixels: int,
    max_new_tokens: int,
) -> tuple[str, int, int, str]:
    import torch

    frames, _ = load_cached_video_frames(video_path)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": "cached-video",
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = processor(text=[rendered_prompt], videos=[frames], return_tensors="pt")
    inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated_ids = generated[:, inputs["input_ids"].shape[1] :]
    text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0].strip()
    return text, int(generated_ids.shape[1]), int(inputs["input_ids"].shape[1]), "stop"


def run_model(args: argparse.Namespace) -> None:
    output_root = Path(args.output_root).resolve()
    manifest_path = (
        Path(args.manifest).resolve()
        if args.manifest
        else prepare_dataset(
            output_root,
            cache_videos=args.cache_videos,
            video_fps=args.video_fps,
            video_maxlen=args.video_maxlen,
        )
    )
    rows = list(iter_jsonl(manifest_path))
    if args.shard_count > 1:
        rows = [row for row in rows if int(row["id"]) % args.shard_count == args.shard_index]
    if args.max_questions is not None:
        rows = rows[: args.max_questions]

    run_dir = output_root / "runs" / args.model_tag
    suffix = f"_shard{args.shard_index:02d}of{args.shard_count:02d}" if args.shard_count > 1 else ""
    prediction_path = run_dir / f"predictions{suffix}.jsonl"
    completed = {int(row["id"]) for row in iter_jsonl(prediction_path)} if args.resume else set()
    model, processor, tokenizer = load_native_qwen(args.model_name_or_path, args.adapter_name_or_path)

    for index, row in enumerate(rows, start=1):
        if int(row["id"]) in completed:
            continue
        record = {
            **row,
            "model_tag": args.model_tag,
            "model_name_or_path": args.model_name_or_path,
            "adapter_name_or_path": args.adapter_name_or_path,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        started = time.time()
        try:
            output, response_length, prompt_length, finish_reason = native_video_response(
                model,
                processor,
                tokenizer,
                prompt=build_prompt(row),
                video_path=Path(row["video_path"]),
                video_max_pixels=args.video_max_pixels,
                max_new_tokens=args.max_new_tokens,
            )
            record.update(
                {
                    "success": True,
                    "output": output,
                    "response_length": response_length,
                    "prompt_length": prompt_length,
                    "finish_reason": finish_reason,
                }
            )
        except Exception as exc:
            record.update({"success": False, "output": None, "error": str(exc)})
        record["latency_s"] = round(time.time() - started, 3)
        append_jsonl(prediction_path, record)
        if index == 1 or index % args.log_every == 0:
            print(f"[{args.model_tag}] {index}/{len(rows)} -> {prediction_path}", flush=True)

    predictions = list(iter_jsonl(prediction_path))
    by_id = {int(row["id"]): row for row in predictions}
    selected = [by_id[int(row["id"])] for row in rows if int(row["id"]) in by_id]
    full_summary = score_rows(selected)
    debiased_summary = score_rows([row for row in selected if not row["pruned"]])
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_tag": args.model_tag,
        "model_name_or_path": args.model_name_or_path,
        "adapter_name_or_path": args.adapter_name_or_path,
        "protocol": {
            "dataset": DATASET_NAME,
            "video_fps": args.video_fps,
            "video_maxlen": args.video_maxlen,
            "video_max_pixels": args.video_max_pixels,
            "max_new_tokens": args.max_new_tokens,
            "temperature": 0.0,
            "enable_thinking": False,
            "official_metric_compatible": True,
            "sampled_video_cache": args.cache_videos,
            "shard_index": args.shard_index,
            "shard_count": args.shard_count,
        },
        "full": full_summary,
        "debiased": debiased_summary,
        "prediction_path": str(prediction_path),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / f"summary{suffix}.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    prepare.add_argument("--no-cache-videos", action="store_false", dest="cache_videos")
    prepare.add_argument("--video-fps", type=float, default=DEFAULT_VIDEO_FPS)
    prepare.add_argument("--video-maxlen", type=int, default=DEFAULT_VIDEO_MAXLEN)
    prepare.set_defaults(cache_videos=True)

    run = subparsers.add_parser("run")
    run.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    run.add_argument("--manifest", default=None)
    run.add_argument("--model-tag", required=True)
    run.add_argument("--model-name-or-path", default=DEFAULT_MODEL)
    run.add_argument("--adapter-name-or-path", default=None)
    run.add_argument("--video-max-pixels", type=int, default=DEFAULT_VIDEO_MAX_PIXELS)
    run.add_argument("--video-fps", type=float, default=DEFAULT_VIDEO_FPS)
    run.add_argument("--video-maxlen", type=int, default=DEFAULT_VIDEO_MAXLEN)
    run.add_argument("--max-new-tokens", type=int, default=16)
    run.add_argument("--max-questions", type=int, default=None)
    run.add_argument("--log-every", type=int, default=25)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--shard-count", type=int, default=1)
    run.add_argument("--no-cache-videos", action="store_false", dest="cache_videos")
    run.set_defaults(cache_videos=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        print(
            prepare_dataset(
                Path(args.output_root).resolve(),
                cache_videos=args.cache_videos,
                video_fps=args.video_fps,
                video_maxlen=args.video_maxlen,
            )
        )
    else:
        run_model(args)


if __name__ == "__main__":
    main()
