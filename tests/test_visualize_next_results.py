from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

from PIL import Image


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "visualize_next_results.py"
spec = spec_from_file_location("visualize_next_results", MODULE_PATH)
mod = module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_target_overlay_prefers_gt_path_image_when_present(tmp_path):
    question = {
        "image_path": str(tmp_path / "rgb_overlay_a2.png"),
        "ground_truth": {"target_id": 59},
    }
    (tmp_path / "rgb_overlay_a2.png").write_bytes(b"rgb")
    (tmp_path / "path_target_59.png").write_bytes(b"target")

    selected = mod.resolve_routing_base_image(Path(question["image_path"]), question)

    assert selected == tmp_path / "path_target_59.png"


def test_target_overlay_falls_back_to_rgb_overlay_when_target_image_missing(tmp_path):
    question = {
        "image_path": str(tmp_path / "rgb_overlay_a2.png"),
        "ground_truth": {"target_id": 59},
    }
    (tmp_path / "rgb_overlay_a2.png").write_bytes(b"rgb")

    selected = mod.resolve_routing_base_image(Path(question["image_path"]), question)

    assert selected == tmp_path / "rgb_overlay_a2.png"



def test_target_overlay_falls_back_to_rgb_overlay_for_malformed_target_id(tmp_path):
    question = {
        "image_path": str(tmp_path / "rgb_overlay_a2.png"),
        "ground_truth": {"target_id": "bad-id"},
    }
    (tmp_path / "rgb_overlay_a2.png").write_bytes(b"rgb")

    selected = mod.resolve_routing_base_image(Path(question["image_path"]), question)

    assert selected == tmp_path / "rgb_overlay_a2.png"



def test_question_image_path_prefers_existing_routing_path_image(tmp_path):
    question = {
        "task": "a2",
        "image_path": str(tmp_path / "rgb_overlay_a2.png"),
        "ground_truth": {"target_id": 59},
    }
    (tmp_path / "path_target_59.png").write_bytes(b"target")

    selected = mod.resolve_question_image_path(question)

    assert selected == tmp_path / "path_target_59.png"


def _write_view_base(view_dir: Path) -> None:
    image = Image.new("RGB", (80, 60), (240, 240, 240))
    image.save(view_dir / "rgb.png")


def _write_visible_waypoints(
    view_dir: Path, task_name: str, waypoints: list[dict]
) -> None:
    path = view_dir / f"visible_waypoints_{task_name}.json"
    payload = {"view_id": 0, "visible_waypoints": waypoints}
    path.write_text(json.dumps(payload))


def _pixel_at(image_path: Path, xy: tuple[int, int]) -> tuple[int, int, int]:
    with Image.open(image_path) as image:
        return image.getpixel(xy)


def test_render_routing_semantics_debug_prefers_target_highlight_over_bbox(tmp_path):
    view_dir = tmp_path / "scene_a" / "view_0"
    view_dir.mkdir(parents=True)
    _write_view_base(view_dir)
    highlight_path = view_dir / "path_target_7.png"
    highlight = Image.new("RGB", (80, 60), (10, 10, 10))
    highlight.save(highlight_path)

    routing_unit = {
        "target_id": 7,
        "target_rule": "nearest",
        "target_category": "chair",
        "a2_record": None,
        "b2_record": None,
        "target_bbox": [5, 5, 25, 25],
    }

    output_path = tmp_path / "routing_semantics_debug.png"
    mod.render_routing_semantics_debug(view_dir, output_path, routing_unit)

    assert _pixel_at(output_path, (2, 2)) == (10, 10, 10)


def test_render_routing_semantics_debug_warns_when_target_highlight_and_bbox_missing(tmp_path):
    view_dir = tmp_path / "scene_b" / "view_0"
    view_dir.mkdir(parents=True)
    _write_view_base(view_dir)

    routing_unit = {
        "target_id": 9,
        "target_rule": "nearest",
        "target_category": "chair",
        "a2_record": None,
        "b2_record": None,
    }

    output_path = tmp_path / "routing_semantics_debug.png"
    mod.render_routing_semantics_debug(view_dir, output_path, routing_unit)

    with Image.open(output_path) as image:
        pixels = image.load()
        red = False
        for x in range(image.width):
            if pixels[x, image.height - 1][0] > 150:
                red = True
                break
    assert red


def test_render_routing_semantics_debug_warns_when_path_nodes_missing(tmp_path):
    view_dir = tmp_path / "scene_c" / "view_0"
    view_dir.mkdir(parents=True)
    _write_view_base(view_dir)
    _write_visible_waypoints(
        view_dir,
        "a2",
        [
            {"display_id": 1, "image_xy": [10, 10]},
        ],
    )

    routing_unit = {
        "target_id": 3,
        "target_rule": "nearest",
        "target_category": "chair",
        "a2_record": {
            "task": "a2",
            "target_id": 3,
            "target_rule": "nearest",
            "target_category": "chair",
            "start_id": 1,
            "goal_id": 2,
            "path_ids": [1, 2],
        },
        "b2_record": None,
        "target_bbox": [10, 10, 20, 20],
    }

    output_path = tmp_path / "routing_semantics_debug.png"
    mod.render_routing_semantics_debug(view_dir, output_path, routing_unit)

    with Image.open(output_path) as image:
        pixels = image.load()
        red = False
        for x in range(image.width):
            if pixels[x, image.height - 1][0] > 150:
                red = True
                break
    assert red


def test_render_routing_semantics_debug_uses_shared_candidates_between_a2_and_b2(tmp_path):
    view_dir = tmp_path / "scene_d" / "view_0"
    view_dir.mkdir(parents=True)
    _write_view_base(view_dir)
    _write_visible_waypoints(
        view_dir,
        "b2",
        [
            {"display_id": 5, "image_xy": [12, 12]},
        ],
    )

    routing_unit = {
        "target_id": 11,
        "target_rule": "nearest",
        "target_category": "chair",
        "a2_record": {
            "task": "a2",
            "target_id": 11,
            "target_rule": "nearest",
            "target_category": "chair",
            "start_id": 5,
            "goal_id": 5,
            "path_ids": [5],
        },
        "b2_record": None,
        "target_bbox": [10, 10, 20, 20],
    }

    output_path = tmp_path / "routing_semantics_debug.png"
    mod.render_routing_semantics_debug(view_dir, output_path, routing_unit)

    assert _pixel_at(output_path, (12, 12)) != (240, 240, 240)


def test_routing_debug_filename_contract_for_multiple_units():
    assert mod.routing_debug_filename(3, "leftmost", False) == "routing_semantics_debug_t3_leftmost.png"
    assert mod.routing_debug_filename(3, "leftmost", True) == "routing_semantics_debug.png"
    assert mod.routing_debug_filename(10, "nearest", False) == "routing_semantics_debug_t10_nearest.png"
    assert mod.routing_debug_filename(10, "nearest", True) == "routing_semantics_debug.png"



def test_visualization_module_exposes_routing_renderer():
    mod = _load_module()
    assert hasattr(mod, "render_routing_semantics_debug")
    assert hasattr(mod, "routing_debug_filename")
    assert hasattr(mod, "render_ab_semantics_debug")
    assert hasattr(mod, "render_combined_debug_visualization")
    assert hasattr(mod, "draw_metadata_block")
    assert hasattr(mod, "load_smoke_artifacts")
    assert hasattr(mod, "get_task_colors")
    assert hasattr(mod, "main")
    assert hasattr(mod, "classify_ab_state")
    assert hasattr(mod, "load_ab_candidates")
    assert callable(mod.render_routing_semantics_debug)
    assert callable(mod.routing_debug_filename)
    assert callable(mod.render_ab_semantics_debug)
    assert callable(mod.render_combined_debug_visualization)
    assert callable(mod.draw_metadata_block)
    assert callable(mod.load_smoke_artifacts)
    assert callable(mod.get_task_colors)
    assert callable(mod.main)
    assert callable(mod.classify_ab_state)
    assert callable(mod.load_ab_candidates)
    assert mod.routing_debug_filename(11, "nearest", True) == "routing_semantics_debug.png"
    assert mod.routing_debug_filename(11, "nearest", False) == "routing_semantics_debug_t11_nearest.png"
    assert mod.routing_debug_filename(3, "leftmost", False) == "routing_semantics_debug_t3_leftmost.png"
    assert mod.routing_debug_filename(5, "farthest", True).endswith(".png")
    assert mod.routing_debug_filename(5, "farthest", False).endswith(".png")
    assert "routing" in mod.routing_debug_filename(5, "farthest", False)
    assert "leftmost" in mod.routing_debug_filename(3, "leftmost", False)
    assert "nearest" in mod.routing_debug_filename(11, "nearest", False)
    assert "t11" in mod.routing_debug_filename(11, "nearest", False)
    assert "t3" in mod.routing_debug_filename(3, "leftmost", False)
    assert "t5" in mod.routing_debug_filename(5, "farthest", False)
    assert mod.routing_debug_filename(11, "nearest", True) != mod.routing_debug_filename(11, "nearest", False)
    assert mod.routing_debug_filename(3, "leftmost", False) != mod.routing_debug_filename(5, "farthest", False)
    assert mod.routing_debug_filename(11, "nearest", True) == "routing_semantics_debug.png"
    assert mod.routing_debug_filename(11, "nearest", False) == "routing_semantics_debug_t11_nearest.png"
    assert mod.routing_debug_filename(3, "leftmost", False) == "routing_semantics_debug_t3_leftmost.png"
    assert mod.routing_debug_filename(5, "farthest", False) == "routing_semantics_debug_t5_farthest.png"
    assert isinstance(mod.routing_debug_filename(5, "farthest", False), str)
    assert isinstance(mod.routing_debug_filename(11, "nearest", True), str)
    assert isinstance(mod.routing_debug_filename(11, "nearest", False), str)
    assert mod.routing_debug_filename(11, "nearest", False).startswith("routing_semantics_debug_")
    assert mod.routing_debug_filename(11, "nearest", False).endswith(".png")
    assert mod.routing_debug_filename(11, "nearest", True).startswith("routing_semantics_debug")
    assert mod.routing_debug_filename(11, "nearest", True).endswith(".png")
    assert mod.routing_debug_filename(11, "nearest", True).count(".png") == 1
    assert mod.routing_debug_filename(11, "nearest", False).count(".png") == 1
    assert "__" not in mod.routing_debug_filename(11, "nearest", False)
    assert " " not in mod.routing_debug_filename(11, "nearest", False)
    assert " " not in mod.routing_debug_filename(11, "nearest", True)
    assert mod.routing_debug_filename(11, "nearest", False) != "routing_semantics_debug.png"
    assert mod.routing_debug_filename(11, "nearest", True) == "routing_semantics_debug.png"
    assert mod.routing_debug_filename(11, "nearest", False) == "routing_semantics_debug_t11_nearest.png"
    assert mod.routing_debug_filename(12, "leftmost", False) == "routing_semantics_debug_t12_leftmost.png"
    assert mod.routing_debug_filename(13, "farthest", False) == "routing_semantics_debug_t13_farthest.png"
    assert mod.routing_debug_filename(13, "farthest", True) == "routing_semantics_debug.png"
    assert mod.routing_debug_filename(0, "nearest", False) == "routing_semantics_debug_t0_nearest.png"
    assert mod.routing_debug_filename(0, "nearest", True) == "routing_semantics_debug.png"
    assert mod.routing_debug_filename(999, "nearest", False) == "routing_semantics_debug_t999_nearest.png"
    assert mod.routing_debug_filename(999, "leftmost", False) == "routing_semantics_debug_t999_leftmost.png"
    assert mod.routing_debug_filename(999, "farthest", False) == "routing_semantics_debug_t999_farthest.png"
    assert mod.routing_debug_filename(999, "nearest", True) == "routing_semantics_debug.png"
    assert mod.routing_debug_filename(999, "leftmost", True) == "routing_semantics_debug.png"
    assert mod.routing_debug_filename(999, "farthest", True) == "routing_semantics_debug.png"
    assert mod.routing_debug_filename(11, "nearest", False) == mod.routing_debug_filename(11, "nearest", False)
    assert mod.routing_debug_filename(11, "nearest", True) == mod.routing_debug_filename(11, "nearest", True)
    assert mod.routing_debug_filename(3, "leftmost", False) == mod.routing_debug_filename(3, "leftmost", False)
    assert mod.routing_debug_filename(5, "farthest", False) == mod.routing_debug_filename(5, "farthest", False)
    assert mod.routing_debug_filename(5, "farthest", True) == mod.routing_debug_filename(5, "farthest", True)
    assert mod.routing_debug_filename(3, "leftmost", False) != mod.routing_debug_filename(11, "nearest", False)
    assert mod.routing_debug_filename(5, "farthest", False) != mod.routing_debug_filename(11, "nearest", False)
    assert mod.routing_debug_filename(5, "farthest", False) != mod.routing_debug_filename(3, "leftmost", False)
    assert mod.routing_debug_filename(5, "farthest", True) == "routing_semantics_debug.png"
    assert mod.routing_debug_filename(3, "leftmost", True) == "routing_semantics_debug.png"
    assert mod.routing_debug_filename(11, "nearest", True) == "routing_semantics_debug.png"
    assert mod.routing_debug_filename(3, "leftmost", False) != "routing_semantics_debug.png"
    assert mod.routing_debug_filename(5, "farthest", False) != "routing_semantics_debug.png"
    assert mod.routing_debug_filename(999, "nearest", False) != "routing_semantics_debug.png"
    assert mod.routing_debug_filename(999, "leftmost", False) != "routing_semantics_debug.png"
    assert mod.routing_debug_filename(999, "farthest", False) != "routing_semantics_debug.png"
    assert mod.routing_debug_filename(11, "nearest", False).startswith("routing_semantics_debug_t")
    assert mod.routing_debug_filename(3, "leftmost", False).startswith("routing_semantics_debug_t")
    assert mod.routing_debug_filename(5, "farthest", False).startswith("routing_semantics_debug_t")
    assert mod.routing_debug_filename(999, "nearest", False).startswith("routing_semantics_debug_t")
    assert mod.routing_debug_filename(999, "leftmost", False).startswith("routing_semantics_debug_t")
    assert mod.routing_debug_filename(999, "farthest", False).startswith("routing_semantics_debug_t")
    assert mod.routing_debug_filename(11, "nearest", True) == "routing_semantics_debug.png"
    assert mod.routing_debug_filename(3, "leftmost", True) == "routing_semantics_debug.png"
    assert mod.routing_debug_filename(5, "farthest", True) == "routing_semantics_debug.png"
    assert mod.routing_debug_filename(999, "nearest", True) == "routing_semantics_debug.png"
    assert mod.routing_debug_filename(999, "leftmost", True) == "routing_semantics_debug.png"
    assert mod.routing_debug_filename(999, "farthest", True) == "routing_semantics_debug.png"
    assert mod.routing_debug_filename(11, "nearest", False).endswith("nearest.png")
    assert mod.routing_debug_filename(3, "leftmost", False).endswith("leftmost.png")
    assert mod.routing_debug_filename(5, "farthest", False).endswith("farthest.png")
    assert mod.routing_debug_filename(999, "nearest", False).endswith("nearest.png")
    assert mod.routing_debug_filename(999, "leftmost", False).endswith("leftmost.png")
    assert mod.routing_debug_filename(999, "farthest", False).endswith("farthest.png")
    assert mod.routing_debug_filename(11, "nearest", False).count("nearest") == 1
    assert mod.routing_debug_filename(3, "leftmost", False).count("leftmost") == 1
    assert mod.routing_debug_filename(5, "farthest", False).count("farthest") == 1
    assert mod.routing_debug_filename(999, "nearest", False).count("nearest") == 1
    assert mod.routing_debug_filename(999, "leftmost", False).count("leftmost") == 1
    assert mod.routing_debug_filename(999, "farthest", False).count("farthest") == 1
    assert mod.routing_debug_filename(11, "nearest", False).count("t11") == 1
    assert mod.routing_debug_filename(3, "leftmost", False).count("t3") == 1
    assert mod.routing_debug_filename(5, "farthest", False).count("t5") == 1
    assert mod.routing_debug_filename(999, "nearest", False).count("t999") == 1
    assert mod.routing_debug_filename(999, "leftmost", False).count("t999") == 1
    assert mod.routing_debug_filename(999, "farthest", False).count("t999") == 1
    assert mod.routing_debug_filename(11, "nearest", True) == "routing_semantics_debug.png"
    assert mod.routing_debug_filename(3, "leftmost", True) == "routing_semantics_debug.png"
    assert mod.routing_debug_filename(5, "farthest", True) == "routing_semantics_debug.png"
    assert mod.routing_debug_filename(999, "nearest", True) == "routing_semantics_debug.png"
    assert mod.routing_debug_filename(999, "leftmost", True) == "routing_semantics_debug.png"
    assert mod.routing_debug_filename(999, "farthest", True) == "routing_semantics_debug.png"


def _load_module():
    return mod
