#!/usr/bin/env python3
"""
Run Next VQA benchmark using OpenAI-compatible API.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import time
from pathlib import Path

from typing import Optional


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


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_user_content(question: dict) -> list[dict]:
    img_path = question.get("image_path")
    user_prompt = question.get("user_prompt", "")
    content = []
    if img_path and Path(img_path).exists():
        b64 = encode_image(img_path)
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{b64}",
                    "detail": "high",
                },
            }
        )
    content.append({"type": "text", "text": user_prompt})
    return content


def extract_chat_stream_text(stream, *, qid: str, model: str, api_base: str) -> str:
    delta_chunks: list[str] = []
    for chunk in stream:
        choices = getattr(chunk, "choices", None)
        if not choices:
            continue
        choice = choices[0]
        delta = getattr(choice, "delta", None)
        if delta is None:
            continue
        content = getattr(delta, "content", None)
        if isinstance(content, str) and content:
            delta_chunks.append(content)
    content = "".join(delta_chunks).strip()
    if not content:
        raise RuntimeError(
            f"question_id={qid} model={model} api_base={api_base} empty assistant streamed text"
        )
    return content


def normalize_benchmark_output(text: str, *, qid: str) -> str:
    stripped = text.strip()
    try:
        json.loads(stripped)
        return stripped
    except json.JSONDecodeError:
        pass

    for match in re.finditer(r"\[[^\[\]]*\]", stripped, flags=re.DOTALL):
        candidate = match.group(0).strip()
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, list):
            return candidate

    return stripped


def validate_benchmark_output(text: str, *, qid: str) -> None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raw_prefix = repr(text[:160])
        raise RuntimeError(
            f"question_id={qid} benchmark output is not valid JSON array: {exc}; "
            f"raw_prefix={raw_prefix}"
        ) from exc
    if not isinstance(payload, list):
        raise RuntimeError(f"question_id={qid} benchmark output must be a JSON array")
    for item in payload:
        if isinstance(item, bool):
            raise RuntimeError(
                f"question_id={qid} benchmark output must contain integer-like IDs, got bool"
            )
        if isinstance(item, int):
            continue
        if isinstance(item, str) and item.strip().lstrip("-").isdigit():
            continue
        raise RuntimeError(
            f"question_id={qid} benchmark output must contain integer-like IDs, got {item!r}"
        )


def request_benchmark_output(client, *, model: str, question: dict, api_base: str) -> str:
    qid = str(question.get("question_id") or "")
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": question.get("system_prompt", "")},
            {"role": "user", "content": build_user_content(question)},
        ],
        stream=True,
        store=False,
        max_completion_tokens=512,
    )
    output = extract_chat_stream_text(stream, qid=qid, model=model, api_base=api_base)
    output = normalize_benchmark_output(output, qid=qid)
    validate_benchmark_output(output, qid=qid)
    return output


def is_retriable_benchmark_error(error_text: str) -> bool:
    lowered = error_text.lower()
    return (
        "timed out" in lowered
        or "timeout" in lowered
        or "empty assistant streamed text" in lowered
        or "rate limit" in lowered
        or "too many requests" in lowered
        or "bad gateway" in lowered
        or "service unavailable" in lowered
        or "gateway timeout" in lowered
        or "upstream connect error" in lowered
        or "connection reset" in lowered
        or "connection aborted" in lowered
        or "connection error" in lowered
        or "temporarily unavailable" in lowered
        or "server disconnected" in lowered
        or "remote protocol error" in lowered
        or "502" in lowered
        or "503" in lowered
        or "504" in lowered
    )


def benchmark_retry_sleep_seconds(attempt: int) -> float:
    return min(30.0, float(2 ** max(0, attempt - 1)))


def run_preflight_check(
    client,
    *,
    model: str,
    questions: list[dict],
    api_base: str,
    max_retries: int,
) -> None:
    if not questions:
        raise RuntimeError("cannot run benchmark preflight without questions")
    sample = questions[0]
    qid = str(sample.get("question_id") or "")
    print(f"[preflight] model={model} api_base={api_base} question_id={qid}")
    attempt = 0
    last_exc = None
    while attempt <= max_retries:
        try:
            request_benchmark_output(client, model=model, question=sample, api_base=api_base)
            print(f"[preflight] ok question_id={qid}")
            return
        except Exception as exc:
            last_exc = exc
            error_text = str(exc)
            if attempt >= max_retries or not is_retriable_benchmark_error(error_text):
                raise RuntimeError(
                    f"Preflight failed for model={model} api_base={api_base} question_id={qid}: {exc}"
                ) from exc
            attempt += 1
            print(f"[preflight] retry {attempt}/{max_retries} question_id={qid} reason={error_text}")
            time.sleep(benchmark_retry_sleep_seconds(attempt))
    raise RuntimeError(
        f"Preflight failed for model={model} api_base={api_base} question_id={qid}: {last_exc}"
    )


def run_openai(questions: list[dict], model: str, output_path: Path,
               max_questions: Optional[int] = None, resume: bool = True,
               timeout_s: float = 60.0, max_retries: int = 1) -> None:
    from openai import OpenAI

    api_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    client = OpenAI(
        timeout=timeout_s,
        base_url=api_base,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if resume and output_path.exists():
        for rec in load_jsonl(output_path):
            qid = rec.get("question_id")
            if qid:
                existing[qid] = rec

    if max_questions is not None and len(existing) > max_questions:
        raise RuntimeError(
            f"existing predictions already exceed max_questions: "
            f"{len(existing)} > {max_questions} for {output_path}"
        )

    to_run = [q for q in questions if q.get("question_id") not in existing]
    if max_questions is not None:
        remaining_budget = max_questions - len(existing)
        if remaining_budget <= 0:
            to_run = []
        else:
            to_run = to_run[:remaining_budget]

    results = list(existing.values())
    if to_run:
        run_preflight_check(
            client,
            model=model,
            questions=to_run,
            api_base=api_base,
            max_retries=max_retries,
        )

    def persist_results() -> None:
        with open(output_path, "w") as f:
            for rec in results:
                f.write(json.dumps(rec, ensure_ascii=True) + "\n")

    for i, q in enumerate(to_run, start=1):
        qid = q.get("question_id")
        print(f"[run] {i}/{len(to_run)} question_id={qid}")
        start = time.time()
        attempt = 0
        output = ""
        success = False
        error = None
        while attempt <= max_retries:
            try:
                output = request_benchmark_output(
                    client,
                    model=model,
                    question=q,
                    api_base=api_base,
                )
                success = True
                error = None
                break
            except Exception as e:
                error = str(e)
                if "benchmark output" in error and not is_retriable_benchmark_error(error):
                    raise RuntimeError(
                        f"question_id={qid} model={model} api_base={api_base} fatal benchmark output error: {error}"
                    ) from e
                attempt += 1
                if attempt > max_retries:
                    break
                print(f"[retry] {attempt}/{max_retries} question_id={qid} reason={error}")
                time.sleep(benchmark_retry_sleep_seconds(attempt))

        results.append({
            "question_id": qid,
            "output": output,
            "success": success,
            "error": error,
            "latency": round(time.time() - start, 3),
        })

        persist_results()
        print(f"[{i}/{len(to_run)}] saved {output_path}")

    # final save
    persist_results()
    print(f"Done. Results: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Next VQA benchmark (OpenAI).")
    parser.add_argument("--vqa", required=True, help="Path to vqa_next_*.jsonl")
    parser.add_argument("--model", required=True, help="OpenAI model name (e.g., gpt-4o)")
    parser.add_argument("--output", required=True, help="Output predictions jsonl")
    parser.add_argument("--max-questions", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=1)
    args = parser.parse_args()

    vqa = load_jsonl(Path(args.vqa))
    if not vqa:
        raise SystemExit("No VQA records found")

    run_openai(vqa, args.model, Path(args.output),
               max_questions=args.max_questions,
               resume=not args.no_resume,
               timeout_s=args.timeout_s,
               max_retries=args.max_retries)


if __name__ == "__main__":
    main()
