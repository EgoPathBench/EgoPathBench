from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "paper" / "scripts" / "build_paper_assets.py"
spec = spec_from_file_location("build_paper_assets_for_test", MODULE_PATH)
mod = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(f'{mod.json.dumps(row)}\n' for row in rows))


def test_merge_prediction_files_keeps_all_disjoint_shards(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    third = tmp_path / "third.jsonl"
    _write_jsonl(first, [{"question_id": "q1", "correct": True}])
    _write_jsonl(second, [{"question_id": "q2", "correct": False}])
    _write_jsonl(third, [{"question_id": "q3", "correct": True}])

    merged = mod.merge_prediction_files([first, second, third])

    assert set(merged) == {"q1", "q2", "q3"}


def test_merge_prediction_files_rejects_duplicate_question_ids(tmp_path: Path) -> None:
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    _write_jsonl(first, [{"question_id": "q1", "correct": True}])
    _write_jsonl(second, [{"question_id": "q1", "correct": False}])

    with pytest.raises(ValueError, match="Duplicate question_id q1"):
        mod.merge_prediction_files([first, second])
