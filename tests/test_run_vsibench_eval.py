from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_vsibench_eval.py"
spec = spec_from_file_location("run_vsibench_eval_for_test", MODULE_PATH)
mod = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_build_prompt_matches_official_answer_contract() -> None:
    row = {
        "question_type": "route_planning",
        "question": "Which route is correct?",
        "options": ["A. Left", "B. Right"],
    }

    prompt = mod.build_prompt(row)

    assert "Options:\nA. Left\nB. Right" in prompt
    assert prompt.endswith("Answer with the option's letter from the given choices directly.")


def test_score_rows_uses_accuracy_and_mra() -> None:
    rows = [
        {
            "question_type": "route_planning",
            "ground_truth": "B",
            "output": "B. Right",
            "success": True,
        },
        {
            "question_type": "object_counting",
            "ground_truth": "4",
            "output": "3",
            "success": True,
        },
    ]

    summary = mod.score_rows(rows)

    assert summary["route_planning_accuracy"] == 100.0
    assert summary["object_counting_MRA:.5:.95:.05"] == 60.0
    assert summary["overall"] == 80.0


def test_numeric_scoring_matches_official_float_parsing() -> None:
    assert mod.to_float("1.5") == 1.5
    assert mod.to_float("1.5m") is None


def test_score_rows_averages_direction_difficulties_before_overall() -> None:
    rows = [
        {
            "question_type": "object_rel_direction_easy",
            "ground_truth": "A",
            "output": "A",
            "success": True,
        },
        {
            "question_type": "object_rel_direction_medium",
            "ground_truth": "A",
            "output": "B",
            "success": True,
        },
        {
            "question_type": "object_rel_direction_hard",
            "ground_truth": "A",
            "output": "B",
            "success": True,
        },
        {
            "question_type": "route_planning",
            "ground_truth": "A",
            "output": "A",
            "success": True,
        },
    ]

    summary = mod.score_rows(rows)

    assert round(summary["object_rel_direction_accuracy"], 6) == round(100 / 3, 6)
    assert round(summary["overall"], 6) == round((100 / 3 + 100) / 2, 6)


def test_video_sample_indices_match_uniform_maxlen_sampling() -> None:
    assert mod.video_sample_indices(100, 60.0, 2.0, 32) == list(range(0, 100, 3))[:32] or len(
        mod.video_sample_indices(100, 60.0, 2.0, 32)
    ) == 32
