#!/usr/bin/env python3
"""Generate oracle predictions from vqa_next jsonl files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Make oracle predictions for Next VQA.")
    parser.add_argument("--vqa", required=True, help="Path to vqa_next_*.jsonl")
    parser.add_argument("--output", required=True, help="Output predictions jsonl")
    args = parser.parse_args()

    records = load_jsonl(Path(args.vqa))
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        for rec in records:
            gt = rec.get("ground_truth", {})
            answer = gt.get("answer", [])
            pred = {
                "question_id": rec.get("question_id"),
                "output": json.dumps(answer),
                "success": True,
            }
            f.write(json.dumps(pred, ensure_ascii=True) + "\n")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
