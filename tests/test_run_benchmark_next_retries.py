from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_benchmark_next.py"
spec = spec_from_file_location("run_benchmark_next_retry", MODULE_PATH)
mod = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_is_retriable_benchmark_error_covers_provider_instability():
    assert mod.is_retriable_benchmark_error("502 Bad Gateway")
    assert mod.is_retriable_benchmark_error("503 Service Unavailable")
    assert mod.is_retriable_benchmark_error("rate limit exceeded")
    assert mod.is_retriable_benchmark_error("connection reset by peer")
    assert mod.is_retriable_benchmark_error("upstream connect error or disconnect/reset before headers")
    assert mod.is_retriable_benchmark_error("Error code: 403 - {'error': {'message': 'Asset upload returned 403'}}")


def test_run_preflight_check_retries_provider_error(monkeypatch):
    attempts = {"count": 0}

    def fake_request(*args, **kwargs):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("502 Bad Gateway")
        return "[1,2]"

    monkeypatch.setattr(mod, "request_benchmark_output", fake_request)
    monkeypatch.setattr(mod.time, "sleep", lambda *_args, **_kwargs: None)

    mod.run_preflight_check(
        client=object(),
        model="gpt-5.4",
        questions=[{"question_id": "q0"}],
        api_base="http://example.test/v1",
        max_retries=2,
    )

    assert attempts["count"] == 2


def test_run_preflight_check_still_rejects_non_retriable_output_error(monkeypatch):
    def fake_request(*args, **kwargs):
        raise RuntimeError("question_id=q0 benchmark output is not valid JSON array")

    monkeypatch.setattr(mod, "request_benchmark_output", fake_request)
    monkeypatch.setattr(mod.time, "sleep", lambda *_args, **_kwargs: None)

    with pytest.raises(RuntimeError, match="Preflight failed"):
        mod.run_preflight_check(
            client=object(),
            model="gpt-5.4",
            questions=[{"question_id": "q0"}],
            api_base="http://example.test/v1",
            max_retries=2,
        )
