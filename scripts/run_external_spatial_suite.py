#!/usr/bin/env python3
"""Download external spatial benchmarks and run local VLM evaluation."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import zipfile
import hashlib
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import urlopen


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_llamafactory_navbench_predict import chat_model_args, prepare_llamafactory_imports


LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DEFAULT_RESULTS_ROOT = Path("results/external_spatial_suite_20260619")
DEFAULT_MODEL = "/mnt/data/omnimodel/weights/base/Qwen3.5-4B"
DEFAULT_LLAMAFACTORY_ROOT = "/mnt/data/wuchanglin/LLaMA-Factory"
DEFAULT_PAPER_BENCHMARKS = ["embspatial", "spatialeval_vqa", "spatialeval_vtqa", "3dsrbench"]
SYSTEM_PROMPT = "You are a careful visual-spatial reasoning assistant. Answer with only the final answer and no explanation."
ANSWER_SUFFIX = "Answer with the correct option letter only."
DEFAULT_MAX_NEW_TOKENS = 4096
PREDICTION_PARSE_POLICY = "strict_after_think_or_direct_short_answer_v1"
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
ANSWER_PREFIX_RE = re.compile(
    r"^(?:the\s+)?(?:final\s+)?(?:correct\s+)?(?:answer|option)\s*(?:is|:)?\s*([A-F])\b",
    flags=re.I,
)
OPTION_MARKER_RE = re.compile(r"\b[A-F]\s*[\.\):]", flags=re.I)


@dataclass(frozen=True)
class BenchmarkSpec:
    key: str
    dataset_name: str
    dataset_config: str | None
    split: str
    benchmark_name: str


SPECS: dict[str, BenchmarkSpec] = {
    "embspatial": BenchmarkSpec(
        key="embspatial",
        dataset_name="FlagEval/EmbSpatial-Bench",
        dataset_config=None,
        split="test",
        benchmark_name="EmbSpatial-Bench",
    ),
    "spatialeval_vqa": BenchmarkSpec(
        key="spatialeval_vqa",
        dataset_name="MilaWang/SpatialEval",
        dataset_config="vqa",
        split="test",
        benchmark_name="SpatialEval-vqa",
    ),
    "spatialeval_vtqa": BenchmarkSpec(
        key="spatialeval_vtqa",
        dataset_name="MilaWang/SpatialEval",
        dataset_config="vtqa",
        split="test",
        benchmark_name="SpatialEval-vtqa",
    ),
    "viewspatial": BenchmarkSpec(
        key="viewspatial",
        dataset_name="lidingm/ViewSpatial-Bench",
        dataset_config="ViewSpatial-Bench",
        split="test",
        benchmark_name="ViewSpatial-Bench",
    ),
    "3dsrbench": BenchmarkSpec(
        key="3dsrbench",
        dataset_name="ccvl/3DSRBench",
        dataset_config="benchmark",
        split="test",
        benchmark_name="3DSRBench",
    ),
}


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def safe_slug(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return slug[:160] or "item"


def parse_option_letter(raw: object, *, options: list[str] | None = None) -> str:
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    upper = text.upper()
    if len(upper) == 1 and upper in {"A", "B", "C", "D", "E", "F"}:
        return upper
    if upper and upper[0] in {"A", "B", "C", "D", "E", "F"}:
        if upper[0] == upper[:1] and (len(upper) == 1 or upper[1] in {".", ")", ":"}):
            return upper[0]
    match = re.search(r"\b([A-F])\b", upper)
    if match:
        return match.group(1)
    if options:
        normalized = normalize_text(text)
        for idx, option in enumerate(options):
            opt_norm = normalize_text(option)
            if not opt_norm:
                continue
            if normalized == opt_norm or normalized in opt_norm or opt_norm in normalized:
                return LETTERS[idx]
    if text.isdigit():
        idx = int(text)
        if 0 <= idx < len(LETTERS):
            return LETTERS[idx]
    return ""


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    stripped = re.sub(r"^```[a-zA-Z0-9_+-]*\s*", "", stripped)
    stripped = re.sub(r"\s*```$", "", stripped)
    return stripped.strip()


def valid_option_letter(letter: str, options: list[str] | None = None) -> bool:
    if letter not in LETTERS:
        return False
    if not options:
        return letter in {"A", "B", "C", "D", "E", "F"}
    idx = LETTERS.index(letter)
    return idx < len(options)


def parse_independent_option_answer(raw: object, *, options: list[str] | None = None) -> str:
    if raw is None:
        return ""
    text = strip_code_fence(str(raw).strip())
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    candidate = " ".join(lines)
    if len(candidate) > 160:
        return ""
    if len(OPTION_MARKER_RE.findall(candidate)) > 1:
        return ""

    upper = candidate.upper()
    if len(upper) == 1 and valid_option_letter(upper, options):
        return upper
    if upper and upper[0] in LETTERS and len(upper) > 1 and upper[1] in {".", ")", ":"}:
        letter = upper[0]
        return letter if valid_option_letter(letter, options) else ""

    match = ANSWER_PREFIX_RE.match(candidate)
    if match:
        letter = match.group(1).upper()
        return letter if valid_option_letter(letter, options) else ""

    if options:
        normalized = normalize_text(candidate)
        for idx, option in enumerate(options):
            if normalized and normalized == normalize_text(option):
                return LETTERS[idx]

    if candidate.isdigit():
        idx = int(candidate)
        if 0 <= idx < len(LETTERS) and (not options or idx < len(options)):
            return LETTERS[idx]
    return ""


def parse_model_option_letter(raw: object, *, options: list[str] | None = None) -> str:
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    if THINK_CLOSE in text:
        return parse_independent_option_answer(text.rsplit(THINK_CLOSE, 1)[1], options=options)
    if THINK_OPEN in text:
        return ""
    if "\n" in text or len(text) > 80:
        return ""
    return parse_independent_option_answer(text, options=options)


def refresh_prediction_parse(record: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    choices = list(row.get("choices") or [])
    pred_letter = parse_model_option_letter(record.get("output"), options=choices)
    pred_text = ""
    if pred_letter and choices:
        idx = LETTERS.index(pred_letter)
        if idx < len(choices):
            pred_text = choices[idx]
    elif pred_letter:
        pred_text = pred_letter
    elif record.get("output"):
        pred_text = str(record.get("output"))
    gold_letter = str(row.get("gold_letter") or record.get("gold_letter") or "").strip().upper()
    refreshed = dict(record)
    refreshed.update(
        {
            "pred_letter": pred_letter,
            "pred_text": pred_text,
            "gold_letter": gold_letter,
            "gold_text": row.get("gold_text", record.get("gold_text")),
            "correct": bool(pred_letter) and pred_letter == gold_letter,
            "parse_policy": PREDICTION_PARSE_POLICY,
        }
    )
    return refreshed


def parse_viewspatial_choices(raw: object) -> list[str]:
    if isinstance(raw, list):
        return [str(item).strip() for item in raw if str(item).strip()]
    if not isinstance(raw, str):
        return []
    options: list[str] = []
    for line in raw.splitlines():
        text = line.strip()
        if not text:
            continue
        match = re.match(r"^[A-F][\.\):\s]+(.*)$", text)
        options.append(match.group(1).strip() if match else text)
    return options


def extract_spatialeval_choices(question: str) -> list[str]:
    match = re.search(r"Available options:\s*(.*)\s*$", question, flags=re.S | re.I)
    if not match:
        return []
    chunk = match.group(1).strip()
    if not chunk:
        return []
    options: list[str] = []
    for line in chunk.splitlines():
        text = line.strip()
        if not text:
            continue
        parsed = re.match(r"^[A-F][\.\):\s]+(.*)$", text)
        options.append(parsed.group(1).strip() if parsed else text)
    return options


def format_options(options: list[str]) -> str:
    return "\n".join(f"{LETTERS[idx]}. {option}" for idx, option in enumerate(options))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


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


def save_pil_image(image, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    image.save(dst)


def download_url(url: str, dst: Path, *, retries: int = 3) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            with urlopen(url, timeout=60) as resp, dst.open("wb") as f:
                shutil.copyfileobj(resp, f)
            return
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt + 1 < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"failed to download {url}: {last_exc}") from last_exc


def extract_hf_zip_member(repo_id: str, archive_name: str, member_path: str, asset_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    archive_path = Path(hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=archive_name))
    target_path = asset_dir / member_path
    if target_path.exists():
        return target_path
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as zf:
        if member_path not in zf.namelist():
            raise FileNotFoundError(f"{member_path} not found in {archive_name}")
        with zf.open(member_path) as src, target_path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    return target_path


def build_embspatial_row(row: dict[str, Any], asset_dir: Path) -> dict[str, Any]:
    qid = str(row.get("question_id") or row.get("id") or "").strip()
    if not qid:
        raise ValueError("EmbSpatial row missing question_id")
    image_path = asset_dir / f"{safe_slug(qid)}.png"
    if not image_path.exists():
        save_pil_image(row["image"], image_path)
    options = [str(x).strip() for x in row.get("answer_options") or [] if str(x).strip()]
    answer_idx = int(row.get("answer"))
    if not options or answer_idx < 0 or answer_idx >= len(options):
        raise ValueError(f"EmbSpatial row has invalid answer/options: {qid}")
    gold_letter = LETTERS[answer_idx]
    gold_text = options[answer_idx]
    user_prompt = (
        f"{str(row.get('question') or '').strip()}\n\n"
        f"Options:\n{format_options(options)}\n\n"
        f"{ANSWER_SUFFIX}"
    )
    return {
        "question_id": qid,
        "benchmark": "EmbSpatial-Bench",
        "source_id": row.get("question_type") or row.get("relation") or "",
        "task_type": row.get("relation") or row.get("question_type") or "",
        "image_paths": [str(image_path)],
        "image_path": str(image_path),
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "choices": options,
        "gold_letter": gold_letter,
        "gold_text": gold_text,
        "raw_answer": answer_idx,
    }


def build_spatialeval_row(row: dict[str, Any], asset_dir: Path, *, benchmark_name: str) -> dict[str, Any]:
    qid = str(row.get("id") or "").strip()
    if not qid:
        raise ValueError(f"{benchmark_name} row missing id")
    image_path = asset_dir / f"{safe_slug(qid)}.png"
    if not image_path.exists():
        save_pil_image(row["image"], image_path)
    question = str(row.get("text") or "").strip()
    options = extract_spatialeval_choices(question)
    gold_letter = str(row.get("oracle_option") or "").strip().upper()
    gold_text = str(row.get("oracle_answer") or "").strip()
    user_prompt = question
    if ANSWER_SUFFIX.lower() not in question.lower():
        user_prompt = f"{question}\n\n{ANSWER_SUFFIX}"
    return {
        "question_id": qid,
        "benchmark": benchmark_name,
        "source_id": qid,
        "task_type": qid.split(".", 1)[0] if "." in qid else "",
        "image_paths": [str(image_path)],
        "image_path": str(image_path),
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "choices": options,
        "gold_letter": gold_letter,
        "gold_text": gold_text,
    }


def build_viewspatial_row(row: dict[str, Any], asset_dir: Path, *, repo_id: str, row_index: int) -> dict[str, Any]:
    source_paths = row.get("image_path") or []
    if isinstance(source_paths, str):
        source_paths = [source_paths]
    if not isinstance(source_paths, list) or not source_paths:
        raise ValueError("ViewSpatial row missing image_path")
    local_paths: list[str] = []
    for source_path in source_paths:
        source_path = str(source_path)
        relative_path = source_path
        prefix = "ViewSpatial-Bench/"
        if relative_path.startswith(prefix):
            relative_path = relative_path[len(prefix) :]
        if relative_path.startswith("scannetv2_val/"):
            archive_name = "scannetv2_val.zip"
        elif relative_path.startswith("val2017/"):
            archive_name = "val2017.zip"
        else:
            raise ValueError(f"unsupported ViewSpatial image path: {source_path}")
        local_path = asset_dir / relative_path
        if not local_path.exists():
            local_path = extract_hf_zip_member(repo_id, archive_name, relative_path, asset_dir)
        local_paths.append(str(local_path))
    options = parse_viewspatial_choices(row.get("choices"))
    if not options:
        raise ValueError("ViewSpatial row has no parseable choices")
    gold_letter = parse_option_letter(row.get("answer"), options=options)
    if not gold_letter:
        raise ValueError(f"ViewSpatial row has invalid answer: {row.get('answer')!r}")
    gold_text = options[LETTERS.index(gold_letter)]
    qid = f"viewspatial__{row_index:05d}"
    user_prompt = (
        f"{str(row.get('question') or '').strip()}\n\n"
        f"Options:\n{format_options(options)}\n\n"
        f"{ANSWER_SUFFIX}"
    )
    return {
        "question_id": qid,
        "benchmark": "ViewSpatial-Bench",
        "source_id": row.get("question_type") or "",
        "task_type": row.get("question_type") or "",
        "image_paths": local_paths,
        "image_path": local_paths[0],
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "choices": options,
        "gold_letter": gold_letter,
        "gold_text": gold_text,
        "raw_answer": row.get("answer"),
    }


def build_3dsrbench_row(row: dict[str, Any], asset_dir: Path) -> dict[str, Any]:
    index = str(row.get("index") or "").strip()
    if not index:
        raise ValueError("3DSRBench row missing index")
    image_url = str(row.get("image_url") or "").strip()
    if not image_url:
        raise ValueError("3DSRBench row missing image_url")
    url_hash = hashlib.md5(image_url.encode("utf-8")).hexdigest()[:12]
    image_path = asset_dir / "by_url" / f"{url_hash}{Path(image_url).suffix or '.jpg'}"
    if not image_path.exists():
        download_url(image_url, image_path)
    raw_options = [str(row.get(letter) or "").strip() for letter in ("A", "B", "C", "D")]
    options = [option for option in raw_options if option and option.lower() != "none"]
    if len(options) < 2:
        raise ValueError(f"3DSRBench row has invalid options for index={index}")
    gold_letter = parse_option_letter(row.get("answer"))
    if not gold_letter:
        raise ValueError(f"3DSRBench row has invalid answer: {row.get('answer')!r}")
    gold_idx = LETTERS.index(gold_letter)
    if gold_idx >= len(raw_options) or not raw_options[gold_idx] or raw_options[gold_idx].lower() == "none":
        raise ValueError(f"3DSRBench row answer points to empty option: index={index}")
    gold_text = raw_options[gold_idx]
    qid = f"3dsrbench__{index}"
    user_prompt = (
        f"{str(row.get('question') or '').strip()}\n\n"
        f"Options:\n{format_options(options)}\n\n"
        f"{ANSWER_SUFFIX}"
    )
    return {
        "question_id": qid,
        "benchmark": "3DSRBench",
        "source_id": str(row.get("image_source") or ""),
        "task_type": str(row.get("category") or ""),
        "image_paths": [str(image_path)],
        "image_path": str(image_path),
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "choices": options,
        "gold_letter": gold_letter,
        "gold_text": gold_text,
        "raw_answer": row.get("answer"),
    }


def build_manifest_for_spec(spec: BenchmarkSpec, out_root: Path) -> Path:
    from datasets import load_dataset

    asset_dir = out_root / "assets" / spec.key
    manifest_dir = out_root / "manifests" / spec.key
    manifest_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)

    if spec.dataset_config is None:
        dataset = load_dataset(spec.dataset_name, split=spec.split)
    else:
        dataset = load_dataset(spec.dataset_name, spec.dataset_config, split=spec.split)

    rows: list[dict[str, Any]] = []
    if spec.key == "embspatial":
        for row in dataset:
            rows.append(build_embspatial_row(row, asset_dir))
    elif spec.key in {"spatialeval_vqa", "spatialeval_vtqa"}:
        for row in dataset:
            rows.append(build_spatialeval_row(row, asset_dir, benchmark_name=spec.benchmark_name))
    elif spec.key == "viewspatial":
        for row_index, row in enumerate(dataset):
            rows.append(build_viewspatial_row(row, asset_dir, repo_id=spec.dataset_name, row_index=row_index))
    elif spec.key == "3dsrbench":
        for row in dataset:
            rows.append(build_3dsrbench_row(row, asset_dir))
    else:
        raise ValueError(f"unsupported benchmark spec: {spec.key}")

    manifest_path = manifest_dir / "manifest.jsonl"
    write_jsonl(manifest_path, rows)
    summary = {
        "benchmark": spec.benchmark_name,
        "dataset_name": spec.dataset_name,
        "dataset_config": spec.dataset_config,
        "split": spec.split,
        "count": len(rows),
        "manifest_path": str(manifest_path),
        "asset_dir": str(asset_dir),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (manifest_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n")
    print(f"[build] {spec.key}: {len(rows)} rows -> {manifest_path}")
    return manifest_path


def load_manifest(path: Path) -> list[dict[str, Any]]:
    return load_jsonl(path)


def build_prediction(client, *, model: str, row: dict[str, Any], api_base: str, max_new_tokens: int) -> dict[str, Any]:
    image_paths = row.get("image_paths") or []
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    prompt = str(row.get("user_prompt") or "")
    system = str(row.get("system_prompt") or SYSTEM_PROMPT)
    start = time.time()
    responses = client.chat(
        messages=[{"role": "user", "content": prompt}],
        system=system,
        images=[str(path) for path in image_paths] if image_paths else None,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        do_sample=False,
    )
    response_text = responses[0].response_text if responses else ""
    if not isinstance(response_text, str):
        response_text = str(response_text)
    raw_output = response_text.strip()
    benchmark = str(row.get("benchmark") or "")
    record = {
        "question_id": row.get("question_id"),
        "benchmark": benchmark,
        "model": model,
        "output": raw_output,
        "latency_s": round(time.time() - start, 3),
        "max_new_tokens": max_new_tokens,
    }
    return refresh_prediction_parse(record, row)


def run_model(
    *,
    model_tag: str,
    model_name_or_path: str,
    adapter_name_or_path: str | None,
    manifest_root: Path,
    output_root: Path,
    repo_root: Path,
    llamafactory_root: str,
    benchmarks: list[str],
    max_questions: int | None,
    max_new_tokens: int,
    resume: bool,
) -> dict[str, Any]:
    prepare_llamafactory_imports(repo_root, Path(llamafactory_root))
    from llamafactory.chat import ChatModel

    args = argparse.Namespace(
        model_name_or_path=model_name_or_path,
        template="qwen2_vl",
        adapter_name_or_path=adapter_name_or_path,
        image_max_pixels=262144,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        top_p=1.0,
        enable_thinking=False,
    )
    model_args = chat_model_args(args)
    chat_model = ChatModel(model_args)

    run_dir = output_root / "runs" / model_tag
    pred_root = run_dir / "predictions"
    pred_root.mkdir(parents=True, exist_ok=True)
    benchmark_metrics: dict[str, Any] = {}
    all_counts = {"total": 0, "correct": 0, "parsed": 0}

    for benchmark in benchmarks:
        manifest_path = manifest_root / benchmark / "manifest.jsonl"
        if not manifest_path.exists():
            raise FileNotFoundError(f"missing manifest: {manifest_path}")
        rows = load_manifest(manifest_path)
        if max_questions is not None:
            rows = rows[: max_questions]
        rows_by_qid = {str(row.get("question_id")): row for row in rows}
        bench_pred_path = pred_root / benchmark / "predictions.jsonl"
        existing: dict[str, dict[str, Any]] = {}
        if resume and bench_pred_path.exists():
            for rec in load_jsonl(bench_pred_path):
                qid = rec.get("question_id")
                if qid:
                    row = rows_by_qid.get(str(qid), {})
                    existing[str(qid)] = refresh_prediction_parse(rec, row) if row else rec
        results = [existing[str(row.get("question_id"))] for row in rows if str(row.get("question_id")) in existing]
        bench_pred_path.parent.mkdir(parents=True, exist_ok=True)
        to_run = [row for row in rows if str(row.get("question_id")) not in existing]
        if to_run:
            print(f"[run] model={model_tag} benchmark={benchmark} rows={len(to_run)}")
        for idx, row in enumerate(to_run, start=1):
            qid = str(row.get("question_id") or "")
            started = time.time()
            try:
                record = build_prediction(
                    chat_model,
                    model=model_name_or_path,
                    row=row,
                    api_base="",
                    max_new_tokens=max_new_tokens,
                )
                record.update(
                    {
                        "question_id": qid,
                        "benchmark": row.get("benchmark"),
                        "row_index": idx,
                        "success": True,
                        "latency_s": round(time.time() - started, 3),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                record = {
                    "question_id": qid,
                    "benchmark": row.get("benchmark"),
                    "model": model_tag,
                    "output": "",
                    "pred_letter": "",
                    "pred_text": "",
                    "gold_letter": row.get("gold_letter"),
                    "gold_text": row.get("gold_text"),
                    "correct": False,
                    "parse_policy": PREDICTION_PARSE_POLICY,
                    "success": False,
                    "error": str(exc),
                    "latency_s": round(time.time() - started, 3),
                    "row_index": idx,
                }
            results.append(record)
            with bench_pred_path.open("w") as f:
                for rec in results:
                    f.write(json.dumps(rec, ensure_ascii=True) + "\n")
            print(f"[run] {model_tag} {benchmark} {idx}/{len(to_run)} saved -> {bench_pred_path}")

        total = len(rows)
        correct = sum(1 for rec in results if rec.get("correct"))
        parsed = sum(1 for rec in results if rec.get("pred_letter"))
        metric = {
            "benchmark": benchmark,
            "total": total,
            "correct": correct,
            "accuracy": round(correct / total, 4) if total else 0.0,
            "parse_rate": round(parsed / total, 4) if total else 0.0,
            "prediction_path": str(bench_pred_path),
        }
        benchmark_metrics[benchmark] = metric
        all_counts["total"] += total
        all_counts["correct"] += correct
        all_counts["parsed"] += parsed
        (run_dir / f"{benchmark}.json").write_text(json.dumps(metric, ensure_ascii=True, indent=2) + "\n")

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_tag": model_tag,
        "model_name_or_path": model_name_or_path,
        "adapter_name_or_path": adapter_name_or_path,
        "prediction_config": {
            "max_new_tokens": max_new_tokens,
            "parse_policy": PREDICTION_PARSE_POLICY,
        },
        "benchmarks": benchmark_metrics,
        "aggregate": {
            "total": all_counts["total"],
            "correct": all_counts["correct"],
            "parsed": all_counts["parsed"],
            "accuracy": round(all_counts["correct"] / all_counts["total"], 4) if all_counts["total"] else 0.0,
            "parse_rate": round(all_counts["parsed"] / all_counts["total"], 4) if all_counts["total"] else 0.0,
        },
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps(summary["aggregate"], ensure_ascii=True, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and evaluate external spatial benchmarks.")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Download datasets and build local manifests.")
    build.add_argument("--output-root", default=str(DEFAULT_RESULTS_ROOT))
    build.add_argument("--benchmarks", nargs="+", default=DEFAULT_PAPER_BENCHMARKS)

    run = sub.add_parser("run", help="Run a local VLM on the prepared manifests.")
    run.add_argument("--output-root", default=str(DEFAULT_RESULTS_ROOT))
    run.add_argument("--manifest-root", default=None)
    run.add_argument("--benchmarks", nargs="+", default=DEFAULT_PAPER_BENCHMARKS)
    run.add_argument("--model-tag", required=True)
    run.add_argument("--model-name-or-path", default=DEFAULT_MODEL)
    run.add_argument("--adapter-name-or-path", default=None)
    run.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    run.add_argument("--llamafactory-root", default=DEFAULT_LLAMAFACTORY_ROOT)
    run.add_argument("--max-questions", type=int, default=None)
    run.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    run.add_argument("--no-resume", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.output_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    if args.command == "build":
        for benchmark in args.benchmarks:
            spec = SPECS.get(benchmark)
            if spec is None:
                raise SystemExit(f"unknown benchmark: {benchmark}")
            build_manifest_for_spec(spec, out_root)
        return

    if args.command == "run":
        manifest_root = Path(args.manifest_root).resolve() if args.manifest_root else out_root / "manifests"
        run_model(
            model_tag=args.model_tag,
            model_name_or_path=args.model_name_or_path,
            adapter_name_or_path=args.adapter_name_or_path,
            manifest_root=manifest_root,
            output_root=out_root,
            repo_root=Path(args.repo_root).resolve(),
            llamafactory_root=args.llamafactory_root,
            benchmarks=args.benchmarks,
            max_questions=args.max_questions,
            max_new_tokens=args.max_new_tokens,
            resume=not args.no_resume,
        )
        return

    raise SystemExit(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
