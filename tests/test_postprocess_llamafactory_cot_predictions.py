from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "postprocess_llamafactory_cot_predictions.py"
SPEC = importlib.util.spec_from_file_location("postprocess_llamafactory_cot_predictions", SCRIPT_PATH)
postprocess = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(postprocess)


def test_extract_final_json_accepts_plain_json_list() -> None:
    assert postprocess.extract_final_json("[1, 3]") == [1, 3]


def test_extract_final_json_accepts_fenced_json_after_think() -> None:
    text = "<think>\nreasoning\n</think>\n```json\n[\n  1,\n  3\n]\n```"
    assert postprocess.extract_final_json(text) == [1, 3]


def test_extract_final_json_accepts_multiline_json_after_think() -> None:
    text = "<think>\nreasoning\n</think>\n[\n 1,\n 3\n]"
    assert postprocess.extract_final_json(text) == [1, 3]


def test_extract_final_json_rejects_missing_tail() -> None:
    with pytest.raises(postprocess.PostprocessError, match="missing_trailing_final_json"):
        postprocess.extract_final_json("<think>\nreasoning\n</think>\nno json here")
