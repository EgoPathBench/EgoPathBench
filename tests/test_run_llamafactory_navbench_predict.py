from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_llamafactory_navbench_predict.py"
SPEC = importlib.util.spec_from_file_location("run_llamafactory_navbench_predict", SCRIPT_PATH)
predict = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(predict)


def test_row_prompt_and_system_prefer_benchmark_fields() -> None:
    row = {
        "user_prompt": "Use this prompt.",
        "prompt_text": "Do not use this one.",
        "question_text": "Or this one.",
        "system_prompt": "Navigation system.",
    }
    assert predict.row_prompt(row) == "Use this prompt."
    assert predict.row_system(row) == "Navigation system."


def test_chat_model_args_records_adapter_and_generation_config() -> None:
    args = argparse.Namespace(
        model_name_or_path="/models/base",
        adapter_name_or_path="/adapters/direct",
        template="qwen2_vl",
        image_max_pixels=262144,
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=128,
        enable_thinking=False,
    )
    payload = predict.chat_model_args(args)
    assert payload["model_name_or_path"] == "/models/base"
    assert payload["adapter_name_or_path"] == "/adapters/direct"
    assert payload["infer_backend"] == "huggingface"
    assert payload["do_sample"] is False
    assert payload["max_new_tokens"] == 128
    assert payload["finetuning_type"] == "lora"


def test_chat_model_args_base_omits_lora_finetuning_type() -> None:
    args = argparse.Namespace(
        model_name_or_path="/models/base",
        adapter_name_or_path=None,
        template="qwen2_vl",
        image_max_pixels=262144,
        temperature=0.0,
        top_p=1.0,
        max_new_tokens=128,
        enable_thinking=False,
    )
    payload = predict.chat_model_args(args)
    assert "adapter_name_or_path" not in payload
    assert "finetuning_type" not in payload


def test_existing_qids_reads_prediction_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "preds.jsonl"
    path.write_text(
        json.dumps({"question_id": "q1", "output": "[1]"}) + "\n"
        + json.dumps({"question_id": "q2", "output": "[2]"}) + "\n"
    )
    assert predict.existing_qids(path) == {"q1", "q2"}
