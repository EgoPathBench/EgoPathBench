from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "reqc_gpt_oracle_cot_text.py"
SPEC = importlib.util.spec_from_file_location("reqc_gpt_oracle_cot_text", SCRIPT_PATH)
reqc = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(reqc)


def test_reqc_directory_accepts_current_valid_rejected_text(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "out"
    rejected = output_dir / "rejected" / "a2" / "shard_00000.jsonl"
    rejected.parent.mkdir(parents=True)
    text = "<think>\nI inspect the visible route and choose the shortest pointmass path.\n</think>\n[1, 3]"
    rejected.write_text(json.dumps({"question_id": "q1", "task": "a2", "text": text}) + "\n")
    packet = {"question_id": "q1", "task": "a2", "required_final_json": [1, 3]}

    monkeypatch.setattr(reqc, "load_packets_by_qid", lambda *args, **kwargs: {"q1": packet})
    monkeypatch.setattr(
        reqc.gpt_cot,
        "validate_cot_text",
        lambda got_text, got_packet: {"ok": True, "issues": [], "final_json": [1, 3]},
    )

    report = reqc.reqc_directory(
        input_dir=output_dir / "rejected",
        accepted_dir=output_dir / "accepted_reqc",
        rejected_dir=output_dir / "rejected_reqc",
        report_path=output_dir / "reqc_report.json",
        release_root=tmp_path,
        split="train",
        tasks=("a2",),
        seed=1,
        overwrite=True,
    )

    rows = [
        json.loads(line)
        for line in (output_dir / "accepted_reqc" / "a2" / "shard_00000.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert report["counts"] == {"accepted_reqc": 1}
    assert rows[0]["text"] == text
    assert rows[0]["final_json"] == [1, 3]
