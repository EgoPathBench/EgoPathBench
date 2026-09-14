from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
import types

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_benchmark_next.py"
spec = spec_from_file_location("run_benchmark_next", MODULE_PATH)
mod = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


class _FakeDelta:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, *, delta_content=None, message_content=None):
        self.delta = _FakeDelta(delta_content) if delta_content is not None else None
        self.message = types.SimpleNamespace(content=message_content)


class _FakeChunk:
    def __init__(self, delta_content=None):
        self.choices = [_FakeChoice(delta_content=delta_content)]


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __iter__(self):
        return iter(self._chunks)


class _FakeCompletionResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(message_content=content)]


class _FakeChatCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class _FakeClient:
    def __init__(self, responses):
        self.chat = types.SimpleNamespace(completions=_FakeChatCompletions(responses))


def test_extract_chat_stream_text_concatenates_chunks():
    stream = _FakeStream([_FakeChunk("[1"), _FakeChunk(",2]")])
    assert mod.extract_chat_stream_text(
        stream,
        qid="q0",
        model="gpt-5.4",
        api_base="http://example.test/v1",
    ) == "[1,2]"


def test_extract_chat_stream_text_rejects_empty_stream():
    stream = _FakeStream([_FakeChunk(None)])
    with pytest.raises(RuntimeError, match="empty assistant streamed text"):
        mod.extract_chat_stream_text(
            stream,
            qid="q0",
            model="gpt-5.4",
            api_base="http://example.test/v1",
        )


def test_extract_chat_completion_text_reads_message_content():
    response = _FakeCompletionResponse('{"display_ids":[1,2]}')
    assert mod.extract_chat_completion_text(
        response,
        qid="q0",
        model="moonshotai/kimi-k2.5",
        api_base="http://example.test/v1",
    ) == '{"display_ids":[1,2]}'


def test_extract_chat_completion_text_reads_sse_string_response():
    response = "\n\n".join(
        [
            'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}',
            'data: {"choices":[{"delta":{"reasoning_content":"Thinking"}}]}',
            'data: {"choices":[{"delta":{"content":"[1"}}]}',
            'data: {"choices":[{"delta":{"content":",2"}}]}',
            'data: {"choices":[{"delta":{"content":"]"}}]}',
            "data: [DONE]",
        ]
    )

    assert mod.extract_chat_completion_text(
        response,
        qid="q0",
        model="grok-4.20-fast",
        api_base="http://example.test/v1",
    ) == "[1,2]"


def test_parse_openai_default_headers_env(monkeypatch):
    monkeypatch.setenv(
        "OPENAI_DEFAULT_HEADERS_JSON",
        '{"User-Agent":"Mozilla/5.0","Accept":"application/json"}',
    )

    assert mod.parse_openai_default_headers_env() == {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json",
    }


def test_parse_openai_default_headers_env_rejects_non_string_values(monkeypatch):
    monkeypatch.setenv("OPENAI_DEFAULT_HEADERS_JSON", '{"User-Agent":123}')

    with pytest.raises(RuntimeError, match="value must be a string"):
        mod.parse_openai_default_headers_env()


def test_select_model_adapter_chooses_provider_specific_modes():
    qwen = mod.select_model_adapter("qwen/qwen3.5-397b-a17b")
    assert qwen.name == "qwen_object"
    assert qwen.stream is False
    assert qwen.response_format == {"type": "json_object"}
    assert qwen.extra_body == {"enable_thinking": False}
    assert qwen.temperature == 0.0
    assert qwen.top_p == 1.0

    dashscope_qwen = mod.select_model_adapter("qwen3.6-plus")
    assert dashscope_qwen.name == "qwen_object"
    assert dashscope_qwen.stream is False
    assert dashscope_qwen.response_format == {"type": "json_object"}
    assert dashscope_qwen.extra_body == {"enable_thinking": False}

    llama4 = mod.select_model_adapter("meta/llama-4-maverick-17b-128e-instruct")
    assert llama4.name == "vlm_object"
    assert llama4.stream is False
    assert llama4.response_format == {"type": "json_object"}

    mistral = mod.select_model_adapter("mistralai/mistral-large-3-675b-instruct-2512")
    assert mistral.name == "vlm_object"
    assert mistral.stream is False
    assert mistral.response_format == {"type": "json_object"}

    gemini = mod.select_model_adapter("gemini-3.1-pro-preview")
    assert gemini.name == "gemini_object"
    assert gemini.stream is False
    assert gemini.response_format == {"type": "json_object"}
    assert gemini.extra_body == {"thinking_config": {"thinking_budget": 0}}
    assert gemini.max_completion_tokens == 8192

    kimi = mod.select_model_adapter("moonshotai/kimi-k2.5")
    assert kimi.name == "kimi_object"
    assert kimi.stream is False
    assert kimi.response_format == {"type": "json_object"}
    assert kimi.extra_body == {"thinking": {"type": "disabled"}}

    dashscope_kimi = mod.select_model_adapter("kimi-k2.6")
    assert dashscope_kimi.name == "kimi_object"
    assert dashscope_kimi.stream is False
    assert dashscope_kimi.response_format == {"type": "json_object"}
    assert dashscope_kimi.extra_body == {"thinking": {"type": "disabled"}}

    namespaced_dashscope_kimi = mod.select_model_adapter("kimi/kimi-k2.6")
    assert namespaced_dashscope_kimi.name == "kimi_object"
    assert namespaced_dashscope_kimi.stream is False
    assert namespaced_dashscope_kimi.response_format == {"type": "json_object"}
    assert namespaced_dashscope_kimi.extra_body == {"thinking": {"type": "disabled"}}

    default = mod.select_model_adapter("gpt-5.4")
    assert default.name == "default_object"
    assert default.stream is False
    assert default.response_format == {"type": "json_object"}


def test_normalize_benchmark_output_extracts_json_array_suffix():
    text = "I'll quickly inspect the workspace first. [1,4,11,21]"
    assert mod.normalize_benchmark_output(text, qid="q5") == "[1, 4, 11, 21]"


def test_normalize_benchmark_output_accepts_json_object_display_ids():
    text = '{"display_ids":[1,2,3]}'
    assert mod.normalize_benchmark_output(text, qid="q6") == "[1, 2, 3]"


def test_normalize_benchmark_output_accepts_object_items_with_ids():
    text = '[{"id": "1"}, {"display_id": 2}, {"displayId": "3"}]'
    assert mod.normalize_benchmark_output(text, qid="q6b") == "[1, 2, 3]"


def test_normalize_benchmark_output_accepts_python_dict_display_ids():
    text = "{'display_ids': ['1', '2']}"
    assert mod.normalize_benchmark_output(text, qid="q7") == "[1, 2]"


def test_validate_benchmark_output_requires_json_array_and_integer_like_ids():
    with pytest.raises(RuntimeError, match=r"JSON array.*raw_prefix='not-json'"):
        mod.validate_benchmark_output("not-json", qid="q1")

    with pytest.raises(RuntimeError, match="integer-like"):
        mod.validate_benchmark_output('[1, "x"]', qid="q2")

    mod.validate_benchmark_output("[1, 2, 3]", qid="q3")
    mod.validate_benchmark_output('["1", "2"]', qid="q4")


def test_validate_benchmark_output_rejects_ids_outside_candidate_space():
    question = {"ground_truth": {"all_ids": [1, 2, 3]}}
    with pytest.raises(RuntimeError, match="outside candidate ID space"):
        mod.validate_benchmark_output("[495, 989]", qid="q8", question=question)


def test_request_benchmark_output_uses_kimi_object_mode():
    response = _FakeCompletionResponse('{"display_ids":[1,2]}')
    client = _FakeClient([response])
    question = {
        "question_id": "q0",
        "system_prompt": "system",
        "user_prompt": "Output walkable IDs",
    }

    output = mod.request_benchmark_output(
        client,
        model="moonshotai/kimi-k2.5",
        question=question,
        api_base="http://example.test/v1",
    )

    assert output == "[1, 2]"
    call = client.chat.completions.calls[0]
    assert call["stream"] is False
    assert call["temperature"] == 0.0
    assert call["top_p"] == 1.0
    assert call["response_format"] == {"type": "json_object"}
    assert call["extra_body"] == {"thinking": {"type": "disabled"}}
    assert "display_ids" in call["messages"][1]["content"][-1]["text"]


def test_request_benchmark_output_uses_gemini_object_mode():
    response = _FakeCompletionResponse('{"display_ids":[1,2]}')
    client = _FakeClient([response])
    question = {
        "question_id": "q0",
        "system_prompt": "system",
        "user_prompt": "Output walkable IDs",
    }

    output = mod.request_benchmark_output(
        client,
        model="gemini-3.1-pro-preview",
        question=question,
        api_base="http://example.test/v1",
    )

    assert output == "[1, 2]"
    call = client.chat.completions.calls[0]
    assert call["stream"] is False
    assert call["response_format"] == {"type": "json_object"}
    assert call["extra_body"] == {"thinking_config": {"thinking_budget": 0}}
    assert call["max_completion_tokens"] == 8192
    assert call["temperature"] == 0.0
    assert call["top_p"] == 1.0


def test_run_openai_fails_in_preflight_before_writing_bad_results(tmp_path, monkeypatch):
    client = _FakeClient([_FakeStream([_FakeChunk(None)])])
    monkeypatch.setattr(mod, "make_openai_client", lambda **_: client)

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"png")
    output_path = tmp_path / "preds.jsonl"

    questions = [
        {
            "question_id": "q0",
            "image_path": str(image_path),
            "system_prompt": "You are a navigation perception assistant.",
            "user_prompt": "Output a JSON array of display IDs that are walkable.",
        }
    ]

    with pytest.raises(RuntimeError, match="Preflight failed"):
        mod.run_openai(
            questions,
            "gpt-5.4",
            output_path,
            max_questions=1,
            resume=False,
            timeout_s=5.0,
            max_retries=0,
        )

    assert not output_path.exists()


def test_run_openai_resume_respects_total_max_questions(tmp_path, monkeypatch):
    client = _FakeClient(
        [
            _FakeCompletionResponse('{"display_ids":[3]}'),
            _FakeCompletionResponse('{"display_ids":[3]}'),
        ]
    )
    monkeypatch.setattr(mod, "make_openai_client", lambda **_: client)

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"png")
    output_path = tmp_path / "preds.jsonl"
    output_path.write_text(
        '{"question_id":"q0","output":"[0]","success":true,"error":null,"latency":0.1}\n'
        '{"question_id":"q1","output":"[1]","success":true,"error":null,"latency":0.1}\n'
    )

    questions = [
        {
            "question_id": "q0",
            "image_path": str(image_path),
            "system_prompt": "You are a navigation perception assistant.",
            "user_prompt": "Output a JSON array of display IDs that are walkable.",
        },
        {
            "question_id": "q1",
            "image_path": str(image_path),
            "system_prompt": "You are a navigation perception assistant.",
            "user_prompt": "Output a JSON array of display IDs that are walkable.",
        },
        {
            "question_id": "q2",
            "image_path": str(image_path),
            "system_prompt": "You are a navigation perception assistant.",
            "user_prompt": "Output a JSON array of display IDs that are walkable.",
        },
        {
            "question_id": "q3",
            "image_path": str(image_path),
            "system_prompt": "You are a navigation perception assistant.",
            "user_prompt": "Output a JSON array of display IDs that are walkable.",
        },
    ]

    mod.run_openai(
        questions,
        "gpt-5.4-mini",
        output_path,
        max_questions=3,
        resume=True,
        timeout_s=5.0,
        max_retries=0,
    )

    rows = [line for line in output_path.read_text().splitlines() if line.strip()]
    assert len(rows) == 3


def test_run_openai_can_record_non_retriable_output_errors(tmp_path, monkeypatch):
    response = _FakeCompletionResponse('{"display_ids":[0]}')
    client = _FakeClient([response, response])
    monkeypatch.setattr(mod, "make_openai_client", lambda **_: client)

    image_path = tmp_path / "sample.png"
    image_path.write_bytes(b"png")
    output_path = tmp_path / "preds.jsonl"
    questions = [
        {
            "question_id": "q0",
            "image_path": str(image_path),
            "system_prompt": "You are a navigation perception assistant.",
            "user_prompt": "Output a JSON array of display IDs that are walkable.",
            "ground_truth": {"all_ids": [1, 2, 3]},
        }
    ]

    mod.run_openai(
        questions,
        "meta/llama-4-maverick-17b-128e-instruct",
        output_path,
        max_questions=1,
        resume=False,
        timeout_s=5.0,
        max_retries=0,
        record_output_errors=True,
    )

    row = mod.load_jsonl(output_path)[0]
    assert row["question_id"] == "q0"
    assert row["output"] == "[0]"
    assert row["success"] is False
    assert "outside candidate ID space" in row["error"]
    assert row["request_config"]["temperature"] == 0.0
    assert row["request_config"]["top_p"] == 1.0
