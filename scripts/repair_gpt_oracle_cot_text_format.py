#!/usr/bin/env python3
"""Repair GPT-written Oracle-CoT text formatting without changing rationales."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


FINAL_LIST_AT_END_RE = re.compile(r"(\[[^\n]*\])\s*$")
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


class RepairError(ValueError):
    """Raised when a row cannot be safely format-repaired."""


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def normalize_final_json(value: Any) -> list[int]:
    if not isinstance(value, list):
        raise RepairError("missing_final_json")
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise RepairError("invalid_final_json") from exc


def parse_json_list(text: str) -> list[int]:
    parsed = json.loads(text)
    if not isinstance(parsed, list):
        raise RepairError("trailing_json_is_not_list")
    try:
        return [int(item) for item in parsed]
    except (TypeError, ValueError) as exc:
        raise RepairError("trailing_json_list_not_ints") from exc


def trailing_final_list(text: str) -> tuple[str, list[int]] | None:
    match = FINAL_LIST_AT_END_RE.search(text.strip())
    if match is None:
        return None
    return match.group(1), parse_json_list(match.group(1))


def remove_trailing_expected_list(text: str, expected: list[int]) -> tuple[str, bool]:
    stripped = text.strip()
    match = FINAL_LIST_AT_END_RE.search(stripped)
    if match is None:
        return stripped, False
    try:
        parsed = parse_json_list(match.group(1))
    except (json.JSONDecodeError, RepairError):
        return stripped, False
    if parsed != expected:
        return stripped, False
    return stripped[: match.start()].rstrip(), True


def final_nonempty_line(text: str) -> str:
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return lines[-1] if lines else ""


def classify_text(text: str) -> str:
    stripped = text.strip()
    starts_open = stripped.startswith(THINK_OPEN)
    has_open = THINK_OPEN in stripped
    has_close = THINK_CLOSE in stripped
    final_line = final_nonempty_line(stripped)
    final_line_is_list = bool(re.fullmatch(r"\[[^\n]*\]", final_line))
    final_line_after_close = not has_close or stripped.rfind(THINK_CLOSE) < stripped.rfind(final_line)
    has_independent_final_list = final_line_is_list and final_line_after_close

    if starts_open and has_close and has_independent_final_list:
        return "ok"
    if not has_open and has_independent_final_list:
        return "plain_plus_tail"
    if starts_open and not has_close:
        return "open_no_close"
    if starts_open and has_close and not has_independent_final_list:
        return "json_inside_or_missing_tail"
    if not has_open and not has_independent_final_list:
        return "plain_no_tail"
    return "other"


def clean_rationale(text: str, expected: list[int]) -> str:
    rationale, _ = remove_trailing_expected_list(text, expected)
    rationale = rationale.strip()
    if not rationale:
        raise RepairError("empty_rationale_after_repair")
    return rationale


def build_clean_text(rationale: str, expected: list[int]) -> str:
    return f"{THINK_OPEN}\n{rationale.strip()}\n{THINK_CLOSE}\n{json.dumps(expected, ensure_ascii=True)}"


def repair_text(text: str, expected: list[int]) -> tuple[str, str, bool]:
    stripped = text.strip()
    category = classify_text(stripped)
    expected_text = json.dumps(expected, ensure_ascii=True)

    if stripped.startswith(THINK_OPEN):
        after_open = stripped[len(THINK_OPEN) :].lstrip("\n")
        if THINK_CLOSE in after_open:
            inner, _after_close = after_open.split(THINK_CLOSE, 1)
            rationale = clean_rationale(inner, expected)
        else:
            rationale = clean_rationale(after_open, expected)
    else:
        trailing = trailing_final_list(stripped)
        if trailing is not None and trailing[1] != expected:
            raise RepairError("trailing_final_json_mismatch")
        rationale = clean_rationale(stripped, expected)

    repaired = build_clean_text(rationale, expected)
    if final_nonempty_line(repaired) != expected_text:
        raise RepairError("repaired_final_line_not_canonical")
    validate_clean_text(repaired, expected)
    return repaired, category, repaired != stripped


def validate_clean_text(text: str, expected: list[int]) -> None:
    stripped = text.strip()
    expected_text = json.dumps(expected, ensure_ascii=True)
    if not stripped.startswith(f"{THINK_OPEN}\n"):
        raise RepairError("clean_missing_open_think")
    if f"\n{THINK_CLOSE}\n" not in stripped:
        raise RepairError("clean_missing_close_think")
    before_close, after_close = stripped.rsplit(THINK_CLOSE, 1)
    if not before_close[len(THINK_OPEN) :].strip():
        raise RepairError("clean_empty_think_body")
    if after_close.strip() != expected_text:
        raise RepairError("clean_final_json_not_independent")
    try:
        parsed = parse_json_list(after_close.strip())
    except (json.JSONDecodeError, RepairError) as exc:
        raise RepairError("clean_final_json_parse_error") from exc
    if parsed != expected:
        raise RepairError("clean_final_json_mismatch")


def repair_row(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = normalize_final_json(row.get("final_json"))
    raw_text = str(row.get("text") or "")
    repaired_text, category, changed = repair_text(raw_text, expected)
    repaired = dict(row)
    repaired["raw_text"] = raw_text
    repaired["text"] = repaired_text
    repaired["format_repair"] = {
        "changed": changed,
        "input_category": category,
        "schema_version": "gpt_oracle_cot_text_format_repair_v1",
    }
    return repaired, repaired["format_repair"]


def repair_directory(*, input_dir: Path, output_dir: Path, report_path: Path, failed_path: Path, overwrite: bool) -> dict[str, Any]:
    if output_dir.exists():
        if not overwrite:
            raise RepairError(f"output_dir_exists:{output_dir}")
        shutil.rmtree(output_dir)
    if failed_path.exists():
        if not overwrite:
            raise RepairError(f"failed_path_exists:{failed_path}")
        failed_path.unlink()
    if report_path.exists() and overwrite:
        report_path.unlink()

    counts: Counter[str] = Counter()
    counts_by_task: dict[str, Counter[str]] = defaultdict(Counter)
    changed_by_task: Counter[str] = Counter()
    failed: Counter[str] = Counter()
    rows = 0

    for path in sorted(input_dir.glob("*/*.jsonl")):
        rel_path = path.relative_to(input_dir)
        out_path = output_dir / rel_path
        task = rel_path.parts[0] if rel_path.parts else str(path.parent.name)
        for row in iter_jsonl(path):
            rows += 1
            try:
                repaired, info = repair_row(row)
            except (json.JSONDecodeError, RepairError, ValueError) as exc:
                reason = str(exc)
                failed[reason] += 1
                append_jsonl(
                    failed_path,
                    {
                        "error": reason,
                        "question_id": row.get("question_id"),
                        "source_path": str(path),
                        "task": row.get("task") or task,
                    },
                )
                continue
            category = str(info["input_category"])
            counts[category] += 1
            counts_by_task[str(repaired.get("task") or task)][category] += 1
            if info["changed"]:
                changed_by_task[str(repaired.get("task") or task)] += 1
            append_jsonl(out_path, repaired)

    repaired_rows = sum(counts.values())
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "report_path": str(report_path),
        "failed_path": str(failed_path),
        "input_rows": rows,
        "output_rows": repaired_rows,
        "failed_rows": sum(failed.values()),
        "changed_rows": sum(changed_by_task.values()),
        "input_format_categories": dict(sorted(counts.items())),
        "input_format_categories_by_task": {
            task: dict(sorted(counter.items())) for task, counter in sorted(counts_by_task.items())
        },
        "changed_rows_by_task": dict(sorted(changed_by_task.items())),
        "failed_reasons": dict(sorted(failed.items())),
        "repair_contract": {
            "does_not_change_rationale_words": True,
            "does_not_change_final_json": True,
            "adds_raw_text_field": True,
            "clean_shape": "<think>\\nrationale\\n</think>\\n[final_json]",
        },
    }
    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Repair GPT Oracle-CoT text format wrappers.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results/oracle_cot_gpt_text_full_20260611/accepted"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/oracle_cot_gpt_text_full_20260611/accepted_format_repaired"),
    )
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument("--failed-path", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    report_path = args.report_path or args.output_dir.parent / "format_repair_report.json"
    failed_path = args.failed_path or args.output_dir.parent / "format_repair_failed.jsonl"
    report = repair_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        report_path=report_path,
        failed_path=failed_path,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
