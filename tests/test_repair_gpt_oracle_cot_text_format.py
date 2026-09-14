from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "repair_gpt_oracle_cot_text_format.py"
SPEC = importlib.util.spec_from_file_location("repair_gpt_oracle_cot_text_format", SCRIPT_PATH)
repair_mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(repair_mod)


def test_repair_wraps_plain_text_with_independent_tail() -> None:
    text = "I inspect the candidate points and keep the feasible ones.\n[1, 3]"

    repaired, category, changed = repair_mod.repair_text(text, [1, 3])

    assert category == "plain_plus_tail"
    assert changed is True
    assert repaired == "<think>\nI inspect the candidate points and keep the feasible ones.\n</think>\n[1, 3]"


def test_repair_splits_plain_text_with_inline_tail() -> None:
    text = "I inspect the candidate points and keep the feasible ones. [1, 3]"

    repaired, category, changed = repair_mod.repair_text(text, [1, 3])

    assert category == "plain_no_tail"
    assert changed is True
    assert repaired == "<think>\nI inspect the candidate points and keep the feasible ones.\n</think>\n[1, 3]"


def test_repair_moves_final_json_out_of_think_block() -> None:
    text = "<think>\nI trace the shortest embodied path to the target. [1, 73]\n</think>"

    repaired, category, changed = repair_mod.repair_text(text, [1, 73])

    assert category == "json_inside_or_missing_tail"
    assert changed is True
    assert repaired == "<think>\nI trace the shortest embodied path to the target.\n</think>\n[1, 73]"


def test_repair_closes_open_think_and_appends_final_json() -> None:
    text = "<think>\nI trace the route from start ID 1 to the goal without extra bends."

    repaired, category, changed = repair_mod.repair_text(text, [1, 30])

    assert category == "open_no_close"
    assert changed is True
    assert repaired == "<think>\nI trace the route from start ID 1 to the goal without extra bends.\n</think>\n[1, 30]"


def test_repair_normalizes_existing_think_tag_spacing() -> None:
    text = "<think>I inspect the route and keep the answer.</think>\n[1, 3]"

    repaired, category, changed = repair_mod.repair_text(text, [1, 3])

    assert category == "ok"
    assert changed is True
    assert repaired == "<think>\nI inspect the route and keep the answer.\n</think>\n[1, 3]"


def test_repair_rejects_mismatched_trailing_answer() -> None:
    text = "I inspect the route and conclude with the path.\n[1, 4]"

    with pytest.raises(repair_mod.RepairError, match="trailing_final_json_mismatch"):
        repair_mod.repair_text(text, [1, 3])


def test_repair_directory_preserves_raw_text_and_writes_report(tmp_path: Path) -> None:
    input_dir = tmp_path / "accepted"
    shard = input_dir / "a1" / "shard_00000.jsonl"
    shard.parent.mkdir(parents=True)
    row = {
        "final_json": [1, 3],
        "question_id": "q1",
        "task": "a1",
        "text": "I inspect all candidates and keep the walkable ones.\n[1, 3]",
    }
    shard.write_text(json.dumps(row) + "\n")

    report = repair_mod.repair_directory(
        input_dir=input_dir,
        output_dir=tmp_path / "clean",
        report_path=tmp_path / "report.json",
        failed_path=tmp_path / "failed.jsonl",
        overwrite=False,
    )

    out_rows = [
        json.loads(line)
        for line in (tmp_path / "clean" / "a1" / "shard_00000.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert report["input_rows"] == 1
    assert report["changed_rows"] == 1
    assert report["failed_rows"] == 0
    assert out_rows[0]["raw_text"] == row["text"]
    assert out_rows[0]["text"].startswith("<think>\n")
    assert out_rows[0]["text"].endswith("\n[1, 3]")
    assert out_rows[0]["format_repair"]["input_category"] == "plain_plus_tail"
