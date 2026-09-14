from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import pytest


spec = spec_from_file_location("prepare_public_dataset", Path(__file__).resolve().parents[1] / "scripts/prepare_public_dataset.py")
mod = module_from_spec(spec)
spec.loader.exec_module(mod)


def source_fixture(tmp_path):
    source = tmp_path / "source"
    view = tmp_path / "task_outputs/scene/view_0"
    view.mkdir(parents=True)
    (view / "image.png").write_bytes(b"test-image")
    (view / "visible_waypoints.json").write_text('{"visible_waypoints": []}')
    (view.parent / "waypoint_graph.json").write_text('{"edges": []}')
    row = {"question_id": "q1", "image_path": str(view / "image.png"),
           "visible_waypoints_path": str(view / "visible_waypoints.json"),
           "ground_truth": {"answer": [1, 2]}}
    for kind in ("vqa", "gt"):
        path = source / f"release/benchmark/{kind}/{kind}_next_a2.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(row) + "\n")
    manifest = {"formal_release": True, "splits": {"benchmark": {"question_counts_by_task": {"a2": 1}}}}
    (source / "release/release_manifest.json").write_text(json.dumps(manifest))
    return source, view


def test_export_survives_relocation_and_preserves_answers(tmp_path):
    source, view = source_fixture(tmp_path)
    output = tmp_path / "output"
    report = mod.prepare(source, output)
    relocated = tmp_path / "relocated"
    output.rename(relocated)
    row = json.loads((relocated / "release/benchmark/vqa/vqa_next_a2.jsonl").read_text())
    assert (relocated / row["image_path"]).read_bytes() == b"test-image"
    assert (relocated / row["visible_waypoints_path"]).is_file()
    assert (relocated / "assets/scene/waypoint_graph.json").is_file()
    assert row["ground_truth"]["answer"] == [1, 2]
    assert report["asset_files"] == 3


def test_missing_image_fails_before_reporting_success(tmp_path):
    source, view = source_fixture(tmp_path)
    (view / "image.png").unlink()
    output = tmp_path / "output"
    with pytest.raises(FileNotFoundError):
        mod.prepare(source, output)
    assert not (output / "preparation_report.json").exists()
