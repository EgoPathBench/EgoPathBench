#!/usr/bin/env python3
"""
Run Next VQA benchmark using OpenAI-compatible API.
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import os
import re
import time
from pathlib import Path

from typing import Any, Optional


class BenchmarkOutputError(RuntimeError):
    def __init__(self, message: str, *, output: str = "") -> None:
        super().__init__(message)
        self.output = output


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


class ModelAdapter:
    def __init__(
        self,
        *,
        name: str,
        stream: bool,
        output_mode: str,
        response_format: dict[str, Any] | None = None,
        extra_body: dict[str, Any] | None = None,
        max_completion_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        reasoning_policy: str = "controlled",
    ) -> None:
        self.name = name
        self.stream = stream
        self.output_mode = output_mode
        self.response_format = response_format
        self.extra_body = extra_body
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.reasoning_policy = reasoning_policy

    def request_config(self) -> dict[str, Any]:
        return {
            "adapter": self.name,
            "stream": self.stream,
            "output_mode": self.output_mode,
            "response_format": self.response_format,
            "extra_body": self.extra_body,
            "max_completion_tokens": self.max_completion_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "reasoning_policy": self.reasoning_policy,
            "skip_preflight": os.getenv("BENCHMARK_SKIP_PREFLIGHT", "").strip().lower()
            in {"1", "true", "yes"},
        }


def select_model_adapter(model: str) -> ModelAdapter:
    lowered = model.lower()
    reasoning_policy = os.getenv("BENCHMARK_REASONING_POLICY", "controlled").strip().lower()
    native_reasoning = reasoning_policy == "native"

    def finalize(adapter: ModelAdapter) -> ModelAdapter:
        adapter.reasoning_policy = reasoning_policy
        override = os.getenv("BENCHMARK_MAX_COMPLETION_TOKENS")
        if override:
            try:
                value = int(override)
            except ValueError as exc:
                raise RuntimeError(
                    f"BENCHMARK_MAX_COMPLETION_TOKENS must be an integer, got {override!r}"
                ) from exc
            if value <= 0:
                raise RuntimeError("BENCHMARK_MAX_COMPLETION_TOKENS must be positive")
            adapter.max_completion_tokens = value
        return adapter

    if (
        lowered.startswith("moonshotai/kimi-")
        or lowered.startswith("kimi/")
        or lowered.startswith("kimi-")
    ):
        return finalize(ModelAdapter(
            name="kimi_object",
            stream=False,
            output_mode="object",
            response_format={"type": "json_object"},
            extra_body=None if native_reasoning else {"thinking": {"type": "disabled"}},
        ))
    if lowered.startswith("qwen/") or lowered.startswith("qwen"):
        return finalize(ModelAdapter(
            name="qwen_object",
            stream=False,
            output_mode="object",
            response_format={"type": "json_object"},
            extra_body=None if native_reasoning else {"enable_thinking": False},
        ))
    if lowered.startswith("gemini-"):
        return finalize(ModelAdapter(
            name="gemini_object",
            stream=False,
            output_mode="object",
            response_format={"type": "json_object"},
            extra_body=None if native_reasoning else {"thinking_config": {"thinking_budget": 0}},
            max_completion_tokens=8192,
        ))
    if lowered.startswith("stepfun-ai/step-3.7-flash") or lowered.startswith("stepfun-ai/step-3.5-flash"):
        return finalize(ModelAdapter(
            name="stepfun_reasoning_object",
            stream=False,
            output_mode="object",
            response_format={"type": "json_object"},
            max_completion_tokens=4096,
        ))
    if (
        lowered.startswith("meta/llama-4-")
        or lowered.startswith("mistralai/mistral-large-3")
        or lowered.startswith("mistralai/mistral-medium-3.5")
    ):
        return finalize(ModelAdapter(
            name="vlm_object",
            stream=False,
            output_mode="object",
            response_format={"type": "json_object"},
        ))
    return finalize(ModelAdapter(
        name="default_object",
        stream=False,
        output_mode="object",
        response_format={"type": "json_object"},
    ))


def build_user_content(question: dict, *, adapter: ModelAdapter) -> list[dict]:
    img_path = question.get("image_path")
    user_prompt = question.get("user_prompt", "")
    if adapter.output_mode == "object":
        user_prompt = (
            f"{user_prompt}\n"
            'Return only a JSON object of the form {"display_ids":[...]} with no extra text.'
        )
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


def extract_sse_chat_completion_text(text: str, *, qid: str, model: str, api_base: str) -> str:
    delta_chunks: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        data = line.removeprefix("data:").strip()
        if not data or data == "[DONE]":
            continue
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            continue
        choices = payload.get("choices")
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if isinstance(content, str) and content:
            delta_chunks.append(content)
    content = "".join(delta_chunks).strip()
    if not content:
        raise RuntimeError(
            f"question_id={qid} model={model} api_base={api_base} empty assistant streamed text"
        )
    return content


def extract_chat_completion_text(response, *, qid: str, model: str, api_base: str) -> str:
    if isinstance(response, str) and "data:" in response:
        return extract_sse_chat_completion_text(
            response, qid=qid, model=model, api_base=api_base
        )
    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError(
            f"question_id={qid} model={model} api_base={api_base} empty assistant completion"
        )
    message = getattr(choices[0], "message", None)
    if message is None:
        raise RuntimeError(
            f"question_id={qid} model={model} api_base={api_base} missing assistant message"
        )
    content = getattr(message, "content", None)
    parts: list[str] = []
    if isinstance(content, str):
        parts.append(content)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            else:
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
    text = "".join(parts).strip()
    if not text:
        raise RuntimeError(
            f"question_id={qid} model={model} api_base={api_base} empty assistant completion"
        )
    return text


def chat_completion_metadata(response) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    choices = getattr(response, "choices", None)
    if choices:
        metadata["finish_reason"] = getattr(choices[0], "finish_reason", None)
    usage = getattr(response, "usage", None)
    if usage is not None:
        if hasattr(usage, "model_dump"):
            metadata["usage"] = usage.model_dump()
        elif isinstance(usage, dict):
            metadata["usage"] = usage
        else:
            metadata["usage"] = {
                key: getattr(usage, key)
                for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                if hasattr(usage, key)
            }
    return metadata


def _coerce_display_ids_payload(payload: Any, *, qid: str) -> list[int]:
    if isinstance(payload, dict):
        payload = payload.get("display_ids")
    if not isinstance(payload, list):
        raise RuntimeError(f"question_id={qid} benchmark output must be a JSON array")
    out: list[int] = []
    for item in payload:
        if isinstance(item, bool):
            raise RuntimeError(
                f"question_id={qid} benchmark output must contain integer-like IDs, got bool"
            )
        if isinstance(item, int):
            out.append(item)
            continue
        if isinstance(item, str) and item.strip().lstrip("-").isdigit():
            out.append(int(item.strip()))
            continue
        if isinstance(item, dict):
            item_id = item.get("id", item.get("display_id", item.get("displayId")))
            if isinstance(item_id, bool):
                raise RuntimeError(
                    f"question_id={qid} benchmark output must contain integer-like IDs, got bool"
                )
            if isinstance(item_id, int):
                out.append(item_id)
                continue
            if isinstance(item_id, str) and item_id.strip().lstrip("-").isdigit():
                out.append(int(item_id.strip()))
                continue
        raise RuntimeError(
            f"question_id={qid} benchmark output must contain integer-like IDs, got {item!r}"
        )
    return out


def _try_parse_display_ids(candidate: str, *, qid: str) -> list[int] | None:
    candidate = candidate.strip()
    if not candidate:
        return None
    parsers = [json.loads, ast.literal_eval]
    for parser in parsers:
        try:
            payload = parser(candidate)
        except (json.JSONDecodeError, SyntaxError, ValueError):
            continue
        try:
            return _coerce_display_ids_payload(payload, qid=qid)
        except RuntimeError:
            continue
    return None


def normalize_benchmark_output(text: str, *, qid: str) -> str:
    stripped = text.strip()
    parsed = _try_parse_display_ids(stripped, qid=qid)
    if parsed is not None:
        return json.dumps(parsed)

    if stripped.startswith("```") and stripped.endswith("```"):
        inner = re.sub(r"^```[a-zA-Z0-9_+-]*\n?", "", stripped)
        inner = re.sub(r"\n?```$", "", inner)
        parsed = _try_parse_display_ids(inner, qid=qid)
        if parsed is not None:
            return json.dumps(parsed)

    for pattern in (r"\[[^\[\]]*\]", r"\{[^\{\}]*\}"):
        for match in re.finditer(pattern, stripped, flags=re.DOTALL):
            parsed = _try_parse_display_ids(match.group(0), qid=qid)
            if parsed is not None:
                return json.dumps(parsed)

    return stripped


def validate_benchmark_output(text: str, *, qid: str, question: dict | None = None) -> None:
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raw_prefix = repr(text[:160])
        raise BenchmarkOutputError(
            f"question_id={qid} benchmark output is not valid JSON array: {exc}; "
            f"raw_prefix={raw_prefix}",
            output=text,
        ) from exc
    try:
        ids = _coerce_display_ids_payload(payload, qid=qid)
    except RuntimeError as exc:
        raise BenchmarkOutputError(str(exc), output=text) from exc
    allowed = None
    if question:
        allowed = ((question.get("ground_truth") or {}).get("all_ids"))
    if allowed:
        allowed_set = {int(x) for x in allowed}
        out_of_range = [item for item in ids if item not in allowed_set]
        if out_of_range:
            raise BenchmarkOutputError(
                f"question_id={qid} benchmark output contains IDs outside candidate ID space: "
                f"{out_of_range}",
                output=text,
            )


def parse_openai_default_headers_env() -> dict[str, str] | None:
    raw = os.getenv("OPENAI_DEFAULT_HEADERS_JSON", "").strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"OPENAI_DEFAULT_HEADERS_JSON must be a JSON object: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError("OPENAI_DEFAULT_HEADERS_JSON must be a JSON object")

    headers: dict[str, str] = {}
    for key, value in payload.items():
        if not isinstance(key, str) or not key.strip():
            raise RuntimeError("OPENAI_DEFAULT_HEADERS_JSON header names must be non-empty strings")
        if not isinstance(value, str):
            raise RuntimeError(
                f"OPENAI_DEFAULT_HEADERS_JSON header {key!r} value must be a string"
            )
        headers[key] = value
    return headers or None


def make_openai_client(*, timeout_s: float, api_base: str, api_key: str | None):
    from openai import OpenAI

    return OpenAI(
        timeout=timeout_s,
        base_url=api_base,
        api_key=api_key,
        default_headers=parse_openai_default_headers_env(),
    )


def request_benchmark_result(client, *, model: str, question: dict, api_base: str) -> dict[str, Any]:
    qid = str(question.get("question_id") or "")
    adapter = select_model_adapter(model)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": question.get("system_prompt", "")},
            {"role": "user", "content": build_user_content(question, adapter=adapter)},
        ],
        stream=adapter.stream,
        store=False,
        max_completion_tokens=adapter.max_completion_tokens,
        temperature=adapter.temperature,
        top_p=adapter.top_p,
        response_format=adapter.response_format,
        extra_body=adapter.extra_body,
    )
    if adapter.stream:
        output = extract_chat_stream_text(response, qid=qid, model=model, api_base=api_base)
        metadata: dict[str, Any] = {}
    else:
        output = extract_chat_completion_text(response, qid=qid, model=model, api_base=api_base)
        metadata = chat_completion_metadata(response)
    output = normalize_benchmark_output(output, qid=qid)
    validate_benchmark_output(output, qid=qid, question=question)
    return {
        "output": output,
        "response_metadata": metadata,
    }


def request_benchmark_output(client, *, model: str, question: dict, api_base: str) -> str:
    return str(request_benchmark_result(
        client,
        model=model,
        question=question,
        api_base=api_base,
    )["output"])


def is_retriable_benchmark_error(error_text: str) -> bool:
    lowered = error_text.lower()
    return (
        "asset upload returned 403" in lowered
        or "timed out" in lowered
        or "timeout" in lowered
        or "empty assistant streamed text" in lowered
        or "empty assistant completion" in lowered
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


def is_benchmark_output_error(error_text: str) -> bool:
    return "benchmark output" in error_text.lower()


def benchmark_retry_sleep_seconds(attempt: int) -> float:
    return min(30.0, float(2 ** max(0, attempt - 1)))


def run_preflight_check(
    client,
    *,
    model: str,
    questions: list[dict],
    api_base: str,
    max_retries: int,
    record_output_errors: bool = False,
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
            if (
                record_output_errors
                and is_benchmark_output_error(error_text)
                and not is_retriable_benchmark_error(error_text)
            ):
                print(
                    f"[preflight] benchmark output error will be recorded "
                    f"question_id={qid} reason={error_text}"
                )
                return
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
               timeout_s: float = 60.0, max_retries: int = 1,
               record_output_errors: bool = False) -> None:
    api_base = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    request_config = select_model_adapter(model).request_config()
    client = make_openai_client(
        timeout_s=timeout_s,
        api_base=api_base,
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
    skip_preflight = os.getenv("BENCHMARK_SKIP_PREFLIGHT", "").strip().lower() in {"1", "true", "yes"}
    if to_run and not skip_preflight:
        run_preflight_check(
            client,
            model=model,
            questions=to_run,
            api_base=api_base,
            max_retries=max_retries,
            record_output_errors=record_output_errors,
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
        response_metadata = None
        success = False
        error = None
        while attempt <= max_retries:
            try:
                result = request_benchmark_result(
                    client,
                    model=model,
                    question=q,
                    api_base=api_base,
                )
                output = str(result["output"])
                response_metadata = result.get("response_metadata")
                success = True
                error = None
                break
            except Exception as e:
                error = str(e)
                if is_benchmark_output_error(error) and not is_retriable_benchmark_error(error):
                    output = getattr(e, "output", "") or ""
                    if record_output_errors:
                        break
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
            "request_config": request_config,
            "response_metadata": response_metadata,
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
    parser.add_argument("--record-output-errors", action="store_true",
                        help="Record non-retriable benchmark output errors as failed rows and continue.")
    args = parser.parse_args()

    vqa = load_jsonl(Path(args.vqa))
    if not vqa:
        raise SystemExit("No VQA records found")

    run_openai(vqa, args.model, Path(args.output),
               max_questions=args.max_questions,
               resume=not args.no_resume,
               timeout_s=args.timeout_s,
               max_retries=args.max_retries,
               record_output_errors=args.record_output_errors)


if __name__ == "__main__":
    main()
