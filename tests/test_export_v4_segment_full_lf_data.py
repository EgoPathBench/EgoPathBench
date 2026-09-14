from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "export_v4_segment_full_lf_data.py"
SPEC = importlib.util.spec_from_file_location("export_v4_segment_full_lf_data", SCRIPT_PATH)
export_mod = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(export_mod)


def packet(tmp_path: Path, qid: str, task: str = "a2") -> dict:
    image = tmp_path / f"{qid}.png"
    image.write_bytes(b"png")
    return {
        "question_id": qid,
        "task": task,
        "original_prompt": "Return a JSON array.",
        "image_path": str(image),
        "required_final_json": [1, 3],
    }


def cot_row(qid: str, task: str = "a2", final_json: list[int] | None = None) -> dict:
    return {
        "question_id": qid,
        "task": task,
        "final_json": final_json or [1, 3],
        "text": "<think>\nI inspect the scene, compare options, and trace [1, 3].\n</think>\n[1, 3]",
        "_source_path": f"/tmp/{qid}.jsonl",
    }


def test_build_v4_gpt_cot_sft_row_uses_gpt_text(tmp_path: Path) -> None:
    row = export_mod.build_v4_gpt_cot_sft_row(
        packet(tmp_path, "q1"),
        cot_row("q1"),
        cot_source="accepted_format_repaired",
    )

    assert row["messages"][0]["content"] == "<image>Return a JSON array."
    assert row["messages"][1]["content"].startswith("<think>\nI inspect")
    assert row["metadata"]["supervision"] == "v4_segment_gpt_oracle_cot_sft"
    assert row["metadata"]["cot_source"] == "accepted_format_repaired"


def test_build_v4_gpt_cot_sft_row_rejects_mismatched_final_json(tmp_path: Path) -> None:
    with pytest.raises(export_mod.oracle.OracleCotError, match="final_json mismatch"):
        export_mod.build_v4_gpt_cot_sft_row(
            packet(tmp_path, "q1"),
            cot_row("q1", final_json=[1, 4]),
            cot_source="accepted_format_repaired",
        )


def test_export_lf_data_writes_full_dataset_names(tmp_path: Path, monkeypatch) -> None:
    packets = [packet(tmp_path, "q1", "a2"), packet(tmp_path, "q2", "b2")]
    cot_rows = {"q1": cot_row("q1", "a2"), "q2": cot_row("q2", "b2")}

    monkeypatch.setattr(export_mod, "load_packets", lambda *args, **kwargs: packets)
    monkeypatch.setattr(export_mod, "load_cot_rows_from_dirs", lambda _dirs: cot_rows)

    output_dir = tmp_path / "lf"
    report = export_mod.export_lf_data(
        release_root=tmp_path,
        split="train",
        tasks=("a2", "b2"),
        cot_dirs=[tmp_path / "accepted_format_repaired"],
        output_dir=output_dir,
        seed=1,
        smoke_count=1,
        require_full=True,
        expected_rows=2,
    )

    assert report["cot_rows"] == 2
    assert report["contract"]["uses_deterministic_oracle_parsed"] is False
    dataset_info = json.loads((output_dir / "dataset_info.json").read_text())
    assert "navbench_v4segfull_new_gpt_cot_sft" in dataset_info
    rows = json.loads((output_dir / "navbench_v4segfull_new_gpt_cot_sft.json").read_text())
    assert [row["metadata"]["question_id"] for row in rows] == ["q1", "q2"]
    assert rows[0]["messages"][1]["content"].startswith("<think>\n")
