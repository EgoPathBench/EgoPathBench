#!/usr/bin/env python3
"""Re-run current GPT Oracle-CoT text QC over previously rejected rows."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_gpt_oracle_cot_text_generation as gpt_cot  # noqa: E402
import run_oracle_cot_generation as oracle  # noqa: E402


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, sort_keys=True)
        f.write("\n")


def load_packets_by_qid(release_root: Path, split: str, tasks: tuple[str, ...], seed: int) -> dict[str, dict[str, Any]]:
    rows = oracle.collect_rows(release_root, split, tasks, limit_per_task=None, seed=seed)
    return {str(packet["question_id"]): packet for packet in (oracle.build_fact_packet(row) for row in rows)}


def load_existing_qids(paths: Iterable[Path]) -> set[str]:
    qids: set[str] = set()
    for root in paths:
        if not root.exists():
            continue
        for path in sorted(root.glob("*/*.jsonl")):
            for row in iter_jsonl(path):
                qid = row.get("question_id")
                if qid:
                    qids.add(str(qid))
    return qids


def reqc_directory(
    *,
    input_dir: Path,
    accepted_dir: Path,
    rejected_dir: Path,
    report_path: Path,
    release_root: Path,
    split: str,
    tasks: tuple[str, ...],
    seed: int,
    overwrite: bool,
) -> dict[str, Any]:
    if overwrite:
        if accepted_dir.exists():
            shutil.rmtree(accepted_dir)
        if rejected_dir.exists():
            shutil.rmtree(rejected_dir)
        if report_path.exists():
            report_path.unlink()
    elif accepted_dir.exists() or rejected_dir.exists():
        raise oracle.OracleCotError("output reqc dirs already exist; pass --overwrite to rebuild")

    packets_by_qid = load_packets_by_qid(release_root, split, tasks, seed)
    already_accepted = load_existing_qids([input_dir.parent / "accepted"])
    counts: Counter[str] = Counter()
    issues: Counter[str] = Counter()
    counts_by_task: dict[str, Counter[str]] = defaultdict(Counter)
    missing_packet_examples: list[str] = []

    for path in sorted(input_dir.glob("*/*.jsonl")):
        task_from_path = path.parent.name
        out_rel = path.relative_to(input_dir)
        for row in iter_jsonl(path):
            qid = str(row.get("question_id") or "")
            task = str(row.get("task") or task_from_path)
            if not qid:
                counts["missing_qid"] += 1
                counts_by_task[task]["missing_qid"] += 1
                continue
            if qid in already_accepted:
                counts["skipped_already_accepted"] += 1
                counts_by_task[task]["skipped_already_accepted"] += 1
                continue
            packet = packets_by_qid.get(qid)
            if packet is None:
                counts["missing_packet"] += 1
                counts_by_task[task]["missing_packet"] += 1
                if len(missing_packet_examples) < 20:
                    missing_packet_examples.append(qid)
                continue
            text = row.get("text")
            if not isinstance(text, str) or not text.strip():
                counts["no_text"] += 1
                counts_by_task[task]["no_text"] += 1
                continue

            qc = gpt_cot.validate_cot_text(text, packet)
            updated = dict(row)
            updated["final_json"] = qc.get("final_json")
            updated["qc"] = qc
            updated["reqc"] = {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "schema_version": "gpt_oracle_cot_text_reqc_v1",
                "source_path": str(path),
            }
            if qc["ok"]:
                append_jsonl(accepted_dir / out_rel, updated)
                counts["accepted_reqc"] += 1
                counts_by_task[task]["accepted_reqc"] += 1
            else:
                append_jsonl(rejected_dir / out_rel, updated)
                counts["rejected_reqc"] += 1
                counts_by_task[task]["rejected_reqc"] += 1
                for issue in qc["issues"]:
                    issues[str(issue)] += 1

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "accepted_dir": str(accepted_dir),
        "rejected_dir": str(rejected_dir),
        "release_root": str(release_root),
        "split": split,
        "tasks": list(tasks),
        "seed": seed,
        "counts": dict(sorted(counts.items())),
        "counts_by_task": {task: dict(sorted(counter.items())) for task, counter in sorted(counts_by_task.items())},
        "issue_counts": dict(sorted(issues.items())),
        "missing_packet_examples": missing_packet_examples,
        "contract": {
            "does_not_call_api": True,
            "does_not_rewrite_text": True,
            "uses_current_validate_cot_text": True,
        },
    }
    write_json(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-QC rejected GPT Oracle-CoT text rows.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/oracle_cot_gpt_text_v4_segment_500_20260615"))
    parser.add_argument("--release-root", type=Path, default=oracle.DEFAULT_RELEASE_ROOT)
    parser.add_argument("--split", default="train")
    parser.add_argument("--tasks", default=",".join(oracle.TASKS))
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    report = reqc_directory(
        input_dir=output_dir / "rejected",
        accepted_dir=output_dir / "accepted_reqc",
        rejected_dir=output_dir / "rejected_reqc",
        report_path=output_dir / "reqc_report.json",
        release_root=args.release_root,
        split=args.split,
        tasks=oracle.parse_tasks(args.tasks),
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
