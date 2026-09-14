from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "paper" / "scripts" / "build_paper_assets.py"
spec = spec_from_file_location("build_paper_assets", MODULE_PATH)
mod = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_baseline_adjusted_egopath_score_maps_no_skill_to_zero():
    metrics = {
        "a1_ba": 0.5,
        "b1_ba": 0.5,
        "a2_sr": 0.0,
        "b2_sr": 0.0,
        "c_sr": 0.0,
    }
    assert mod.egopath_score(metrics) == 0.0


def test_baseline_adjusted_egopath_score_maps_perfect_to_one():
    metrics = {
        "a1_ba": 1.0,
        "b1_ba": 1.0,
        "a2_sr": 1.0,
        "b2_sr": 1.0,
        "c_sr": 1.0,
    }
    assert mod.egopath_score(metrics) == 1.0


def test_baseline_adjusted_egopath_score_uses_adjusted_ba_and_raw_sr():
    metrics = {
        "a1_ba": 0.75,
        "b1_ba": 0.80,
        "a2_sr": 0.30,
        "b2_sr": 0.10,
        "c_sr": 0.05,
    }
    expected = (0.50 + 0.60 + 0.30 + 0.10 + 0.05) / 5
    assert abs(mod.egopath_score(metrics) - expected) < 1e-12
