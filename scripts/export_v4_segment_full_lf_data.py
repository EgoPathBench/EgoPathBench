#!/usr/bin/env python3
"""Export full v4 segment-aware GPT-CoT rows to LLaMA-Factory data."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import repair_gpt_oracle_cot_text_format as format_repair  # noqa: E402
import run_oracle_cot_generation as oracle  # noqa: E402


DEFAULT_COT_DIR = Path("results/oracle_cot_gpt_text_v4_segment_full_20260616/accepted_format_repaired")
DEFAULT_OUTPUT_DIR = Path("results/lf_data_v4_segment_full_20260616")


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


def write_json_array(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(rows, f, ensure_ascii=True, indent=2)
        f.write("\n")


def load_cot_rows(input_dir: Path) -> dict[str, dict[str, Any]]:
    by_qid: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for path in sorted(input_dir.glob("*/*.jsonl")):
        for row in iter_jsonl(path):
            qid = str(row.get("question_id") or "")
            if not qid:
                continue
            if qid in by_qid:
                duplicates.append(qid)
            row = dict(row)
            row["_source_path"] = str(path)
            by_qid[qid] = row
    if duplicates:
        raise oracle.OracleCotError(f"duplicate CoT question_ids in {input_dir}: {duplicates[:5]}")
    return by_qid


def load_cot_rows_from_dirs(input_dirs: list[Path]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    duplicates: list[str] = []
    for input_dir in input_dirs:
        rows = load_cot_rows(input_dir)
        for qid, row in rows.items():
            if qid in merged:
                duplicates.append(qid)
            merged[qid] = row
    if duplicates:
        raise oracle.OracleCotError(f"duplicate CoT question_ids across dirs: {duplicates[:5]}")
    return merged


def load_packets(release_root: Path, split: str, tasks: tuple[str, ...], seed: int) -> list[dict[str, Any]]:
    rows = oracle.collect_rows(release_root, split, tasks, limit_per_task=None, seed=seed)
    return [oracle.build_fact_packet(row) for row in rows]


def build_v4_gpt_cot_sft_row(packet: dict[str, Any], cot_row: dict[str, Any], *, cot_source: str) -> dict[str, Any]:
    expected = [int(item) for item in packet["required_final_json"]]
    row_final = format_repair.normalize_final_json(cot_row.get("final_json"))
    if row_final != expected:
        raise oracle.OracleCotError(f"final_json mismatch for question_id={packet['question_id']}")
    cot_text = str(cot_row.get("text") or "")
    format_repair.validate_clean_text(cot_text, expected)
    return {
        "messages": [
            {"role": "user", "content": oracle.build_user_message(packet)},
            {"role": "assistant", "content": cot_text},
        ],
        "images": [packet["image_path"]],
        "metadata": {
            "question_id": packet["question_id"],
            "task": packet["task"],
            "supervision": "v4_segment_gpt_oracle_cot_sft",
            "cot_source": cot_source,
            "cot_source_path": cot_row.get("_source_path"),
        },
    }


def dataset_info_payload() -> dict[str, Any]:
    base = {
        "formatting": "sharegpt",
        "columns": {"messages": "messages", "images": "images"},
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
        },
    }
    return {
        "navbench_v4segfull_direct_sft": {
            "file_name": "navbench_v4segfull_direct_sft.json",
            **base,
        },
        "navbench_v4segfull_new_gpt_cot_sft": {
            "file_name": "navbench_v4segfull_new_gpt_cot_sft.json",
            **base,
        },
        "navbench_v4segfull_new_gpt_cot_sft_smoke": {
            "file_name": "navbench_v4segfull_new_gpt_cot_sft_smoke.json",
            **base,
        },
        "navbench_v4segfull_new_gpt_cot_sft_sanity500": {
            "file_name": "navbench_v4segfull_new_gpt_cot_sft_sanity500.json",
            **base,
        },
        "navbench_v4segfull_new_gpt_cot_sft_gate10pct": {
            "file_name": "navbench_v4segfull_new_gpt_cot_sft_gate10pct.json",
            **base,
        },
    }


def export_lf_data(
    *,
    release_root: Path,
    split: str,
    tasks: tuple[str, ...],
    cot_dirs: list[Path],
    output_dir: Path,
    seed: int,
    smoke_count: int,
    require_full: bool,
    expected_rows: int,
) -> dict[str, Any]:
    packets = load_packets(release_root, split, tasks, seed)
    cot_by_qid = load_cot_rows_from_dirs(cot_dirs)

    direct_rows: list[dict[str, Any]] = []
    cot_rows: list[dict[str, Any]] = []
    missing_cot: list[str] = []
    missing_images: list[str] = []
    source_label = "+".join(str(path) for path in cot_dirs)

    for packet in packets:
        qid = str(packet["question_id"])
        if not packet.get("image_path") or not Path(str(packet["image_path"])).exists():
            missing_images.append(qid)
        direct_rows.append(oracle.build_direct_sft_row(packet))
        cot_row = cot_by_qid.get(qid)
        if cot_row is None:
            missing_cot.append(qid)
            continue
        cot_rows.append(build_v4_gpt_cot_sft_row(packet, cot_row, cot_source=source_label))

    if require_full and missing_cot:
        raise oracle.OracleCotError(f"missing {len(missing_cot)} CoT rows; examples={missing_cot[:10]}")
    if require_full and expected_rows > 0 and len(cot_rows) != expected_rows:
        raise oracle.OracleCotError(f"cot row count {len(cot_rows)} != expected {expected_rows}")

    smoke_rows = cot_rows[:smoke_count]
    sanity_rows = oracle.stratified_limit(cot_rows, total_limit=500)
    gate_rows = oracle.stratified_fraction(cot_rows, fraction=0.1)

    write_json_array(output_dir / "navbench_v4segfull_direct_sft.json", direct_rows)
    write_json_array(output_dir / "navbench_v4segfull_new_gpt_cot_sft.json", cot_rows)
    write_json_array(output_dir / "navbench_v4segfull_new_gpt_cot_sft_smoke.json", smoke_rows)
    write_json_array(output_dir / "navbench_v4segfull_new_gpt_cot_sft_sanity500.json", sanity_rows)
    write_json_array(output_dir / "navbench_v4segfull_new_gpt_cot_sft_gate10pct.json", gate_rows)
    write_json(output_dir / "dataset_info.json", dataset_info_payload())

    task_counts = Counter(str((row.get("metadata") or {}).get("task")) for row in cot_rows)
    direct_task_counts = Counter(str((row.get("metadata") or {}).get("task")) for row in direct_rows)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "release_root": str(release_root),
        "split": split,
        "tasks": list(tasks),
        "seed": seed,
        "cot_dirs": [str(path) for path in cot_dirs],
        "direct_rows": len(direct_rows),
        "cot_rows": len(cot_rows),
        "expected_rows": expected_rows,
        "require_full": require_full,
        "smoke_rows": len(smoke_rows),
        "sanity500_rows": len(sanity_rows),
        "gate10pct_rows": len(gate_rows),
        "rows_by_task": dict(sorted(task_counts.items())),
        "direct_rows_by_task": dict(sorted(direct_task_counts.items())),
        "available_cot_rows": len(cot_by_qid),
        "missing_cot_rows": len(missing_cot),
        "missing_cot_examples": missing_cot[:20],
        "missing_images": len(missing_images),
        "missing_image_examples": missing_images[:20],
        "contract": {
            "cot_text_source": "gpt_accepted_format_repaired",
            "uses_deterministic_oracle_parsed": False,
            "validates_final_json_equals_gt": True,
            "validates_clean_think_format": True,
        },
    }
    write_json(output_dir / "export_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export full v4 GPT Oracle-CoT LF data.")
    parser.add_argument("--release-root", type=Path, default=oracle.DEFAULT_RELEASE_ROOT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--tasks", default=",".join(oracle.TASKS))
    parser.add_argument("--cot-dir", type=Path, action="append", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--smoke-count", type=int, default=2)
    parser.add_argument("--require-full", action="store_true")
    parser.add_argument("--expected-rows", type=int, default=31852)
    args = parser.parse_args()

    report = export_lf_data(
        release_root=args.release_root,
        split=args.split,
        tasks=oracle.parse_tasks(args.tasks),
        cot_dirs=args.cot_dir or [DEFAULT_COT_DIR],
        output_dir=args.output_dir,
        seed=args.seed,
        smoke_count=args.smoke_count,
        require_full=args.require_full,
        expected_rows=args.expected_rows,
    )
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
