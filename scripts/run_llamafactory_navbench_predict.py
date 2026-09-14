#!/usr/bin/env python3
"""Run local LLaMA-Factory VLM inference on NavBench3D VQA rows."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MODEL = "/mnt/data/omnimodel/weights/base/Qwen3.5-4B"
DEFAULT_LLAMAFACTORY_ROOT = "/mnt/data/wuchanglin/LLaMA-Factory"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def existing_qids(path: Path) -> set[str]:
    return {str(row["question_id"]) for row in iter_jsonl(path) if row.get("question_id") is not None}


def row_prompt(row: dict[str, Any]) -> str:
    return str(row.get("user_prompt") or row.get("prompt_text") or row.get("question_text") or "").strip()


def row_system(row: dict[str, Any]) -> str | None:
    system = str(row.get("system_prompt") or "").strip()
    return system or None


def prepare_llamafactory_imports(repo_root: Path, llamafactory_root: Path) -> None:
    compat_dir = repo_root / "tools" / "llamafactory_compat"
    src_dir = llamafactory_root / "src"
    for path in (str(compat_dir), str(src_dir)):
        if path not in sys.path:
            sys.path.insert(0, path)
    os.environ.setdefault("DISABLE_VERSION_CHECK", "1")

    # Import explicitly because sitecustomize is only auto-loaded from PYTHONPATH
    # during interpreter startup.
    import sitecustomize  # noqa: F401


def chat_model_args(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model_name_or_path": args.model_name_or_path,
        "template": args.template,
        "infer_backend": "huggingface",
        "trust_remote_code": True,
        "image_max_pixels": args.image_max_pixels,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "enable_thinking": args.enable_thinking,
    }
    if args.adapter_name_or_path:
        payload["finetuning_type"] = "lora"
        payload["adapter_name_or_path"] = args.adapter_name_or_path
    return payload


def predict_rows(args: argparse.Namespace) -> None:
    repo_root = Path(args.repo_root).resolve()
    prepare_llamafactory_imports(repo_root, Path(args.llamafactory_root))

    from llamafactory.chat import ChatModel

    rows = load_jsonl(Path(args.vqa))
    if args.max_questions is not None:
        rows = rows[: args.max_questions]

    output_path = Path(args.output)
    done = existing_qids(output_path) if args.resume else set()
    model_args = chat_model_args(args)
    chat_model = ChatModel(model_args)

    for index, row in enumerate(rows):
        qid = str(row.get("question_id") or "")
        if not qid:
            continue
        if qid in done:
            continue

        image_path = str(row.get("image_path") or "").strip()
        prompt = row_prompt(row)
        record = {
            "question_id": qid,
            "task": row.get("task"),
            "model_name_or_path": args.model_name_or_path,
            "adapter_name_or_path": args.adapter_name_or_path,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "request_config": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "max_new_tokens": args.max_new_tokens,
                "template": args.template,
                "image_max_pixels": args.image_max_pixels,
                "enable_thinking": args.enable_thinking,
            },
        }

        started = time.time()
        try:
            responses = chat_model.chat(
                messages=[{"role": "user", "content": prompt}],
                system=row_system(row),
                images=[image_path] if image_path else None,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                do_sample=args.temperature > 0,
            )
            response = responses[0]
            record.update(
                {
                    "success": True,
                    "output": response.response_text.strip(),
                    "response_length": response.response_length,
                    "prompt_length": response.prompt_length,
                    "finish_reason": response.finish_reason,
                    "latency_s": round(time.time() - started, 3),
                    "row_index": index,
                }
            )
        except Exception as exc:
            record.update(
                {
                    "success": False,
                    "output": None,
                    "error": str(exc),
                    "latency_s": round(time.time() - started, 3),
                    "row_index": index,
                }
            )

        append_jsonl(output_path, record)
        if args.request_sleep_s > 0:
            time.sleep(args.request_sleep_s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local LLaMA-Factory predictions for NavBench3D VQA rows.")
    parser.add_argument("--vqa", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--llamafactory-root", default=DEFAULT_LLAMAFACTORY_ROOT)
    parser.add_argument("--model-name-or-path", default=DEFAULT_MODEL)
    parser.add_argument("--adapter-name-or-path", default=None)
    parser.add_argument("--template", default="qwen2_vl")
    parser.add_argument("--image-max-pixels", type=int, default=262144)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--request-sleep-s", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    predict_rows(args)


if __name__ == "__main__":
    main()
