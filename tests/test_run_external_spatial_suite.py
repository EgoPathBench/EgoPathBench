from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_external_spatial_suite.py"
SPEC = importlib.util.spec_from_file_location("run_external_spatial_suite", SCRIPT_PATH)
external = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = external
SPEC.loader.exec_module(external)


class _Response:
    def __init__(self, text: str) -> None:
        self.response_text = text


class _Client:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return [_Response(self.text)]


def test_parse_model_option_letter_uses_only_after_think_answer() -> None:
    text = "<think>\nA. wrong option appears in reasoning.\n</think>\nD"

    assert external.parse_model_option_letter(text, options=["one", "two", "three", "four"]) == "D"


def test_parse_model_option_letter_rejects_unclosed_think_tail_enum() -> None:
    text = (
        "<think>\n"
        "The reasoning is still enumerating options.\n"
        "A. fireplace: farther away.\n"
        "B. pool table: farther away.\n"
        "C. railing: farther away.\n"
        "D. pillow"
    )

    assert external.parse_model_option_letter(text, options=["fireplace", "pool table", "railing", "pillow"]) == ""


def test_parse_model_option_letter_rejects_multiple_tail_options() -> None:
    text = "<think>\nreasoning\n</think>\nA. northeast\nB. northwest"

    assert external.parse_model_option_letter(text, options=["northeast", "northwest"]) == ""


def test_parse_model_option_letter_accepts_direct_short_answer_without_think() -> None:
    assert external.parse_model_option_letter("B. northwest", options=["northeast", "northwest"]) == "B"


def test_refresh_prediction_parse_removes_old_reasoning_pollution() -> None:
    row = {
        "choices": ["fireplace", "pool table", "railing", "pillow"],
        "gold_letter": "D",
        "gold_text": "pillow",
    }
    record = {
        "output": "<think>\nI see a room and start comparing options.\nA. fireplace is farther away.",
        "pred_letter": "A",
        "correct": False,
    }

    refreshed = external.refresh_prediction_parse(record, row)

    assert refreshed["pred_letter"] == ""
    assert refreshed["correct"] is False
    assert refreshed["parse_policy"] == external.PREDICTION_PARSE_POLICY


def test_build_prediction_uses_4096_and_strict_after_think_parse() -> None:
    client = _Client("<think>\nA appears during reasoning.\n</think>\nC. railing")
    row = {
        "question_id": "q1",
        "benchmark": "embspatial",
        "user_prompt": "Which option is correct?",
        "system_prompt": external.SYSTEM_PROMPT,
        "choices": ["fireplace", "pool table", "railing", "pillow"],
        "gold_letter": "C",
        "gold_text": "railing",
    }

    record = external.build_prediction(
        client,
        model="/models/base",
        row=row,
        api_base="",
        max_new_tokens=external.DEFAULT_MAX_NEW_TOKENS,
    )

    assert client.calls[0]["max_new_tokens"] == 4096
    assert record["pred_letter"] == "C"
    assert record["correct"] is True
