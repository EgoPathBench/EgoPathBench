from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


GEN_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_oracle_cot_generation.py"
GEN_SPEC = importlib.util.spec_from_file_location("run_oracle_cot_generation", GEN_PATH)
oracle_cot = importlib.util.module_from_spec(GEN_SPEC)
assert GEN_SPEC.loader is not None
GEN_SPEC.loader.exec_module(oracle_cot)

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_gpt_oracle_cot_text_generation.py"
SPEC = importlib.util.spec_from_file_location("run_gpt_oracle_cot_text_generation", SCRIPT_PATH)
gpt_cot = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(gpt_cot)


def write_visible(tmp_path: Path) -> str:
    path = tmp_path / "visible_waypoints.json"
    payload = {
        "visible_waypoints": [
            {
                "display_id": 1,
                "pointmass_walkable": True,
                "embodied_feasible": True,
                "point_clearance_m": 0.45,
                "embodied_clearance_margin_m": 0.2,
            },
            {
                "display_id": 2,
                "pointmass_walkable": True,
                "embodied_feasible": False,
                "point_clearance_m": 0.05,
                "embodied_clearance_margin_m": 0.0,
            },
            {
                "display_id": 3,
                "pointmass_walkable": True,
                "embodied_feasible": True,
                "point_clearance_m": 0.4,
                "embodied_clearance_margin_m": 0.15,
            },
        ]
    }
    path.write_text(json.dumps(payload))
    return str(path.resolve())


def base_row(tmp_path: Path, task: str) -> dict:
    gt = {
        "answer": [1, 3] if task in {"a2", "b2", "c", "b1"} else [1, 2, 3],
        "all_ids": [1, 2, 3],
        "walkable_ids": [1, 2, 3],
        "start_id": 1,
        "goal_id": 3,
        "target_id": 7 if task == "c" else 99,
        "target_category": "chair",
        "target_rule": "nearest",
        "robot_diameter_m": 0.6,
        "route_semantics": "embodied" if task in {"b2", "c"} else "pointmass",
        "canonical_sparse_length_m": 1.25,
        "navigation_geometry_facts": {
            "path_length_point_m": 1.25,
            "detour_ratio_point": 1.0,
            "turn_count_point": 0,
            "decision_count_point": 0,
            "geometry_profile": "straight_short",
        },
        "navigation_embodiment_facts": {
            "path_length_embodied_m": 1.25 if task in {"b2", "c"} else None,
            "detour_ratio_embodied": 1.0 if task in {"b2", "c"} else None,
            "turn_count_embodied": 0 if task in {"b2", "c"} else None,
            "decision_count_embodied": 0 if task in {"b2", "c"} else None,
            "path_overlap_point_vs_embodied": 1.0 if task in {"b2", "c"} else None,
            "embodied_extra_length_m": 0.0 if task in {"b2", "c"} else None,
            "embodied_clearance_margin_min_m": 0.05,
            "narrow_passage_fraction": 0.3,
        },
        "routing_complexity": {
            "num_segments_sparse": 1 if task == "a2" else None,
            "embodied_num_segments_sparse": 1 if task in {"b2", "c"} else None,
            "embodied_min_clearance_m": 0.05 if task in {"b2", "c"} else None,
            "embodied_mean_clearance_m": 0.12 if task in {"b2", "c"} else None,
            "embodied_narrow_passage_fraction": 0.3 if task in {"b2", "c"} else None,
            "embodied_detour_ratio_sparse": 1.0 if task in {"b2", "c"} else None,
            "detour_ratio_sparse": 1.0 if task in {"b2", "c"} else None,
            "sparse_turn_count": 0 if task in {"b2", "c"} else None,
            "start_goal_l2_m": 1.25 if task in {"b2", "c"} else None,
            "path_projection_safe": True if task in {"a2", "b2", "c"} else None,
            "point_path_projection_safe": True if task == "a2" else None,
            "embodied_path_projection_safe": True if task in {"b2", "c"} else None,
        },
        "instruction_selection": {
            "cohort_target_ids": [4, 6, 7, 18] if task == "c" else [99, 100],
            "selection_status": "unique_winner",
        },
        "fixed_target_prompt_cue_bundle": {
            "cue_family": "relation",
            "cue_type": "near",
            "anchor_category": "table",
            "residual_cue_value": "closest",
        },
        "intent_family": "sit_rest",
        "same_type_key": "sit_rest",
        "same_type_supported_visible_count": 4 if task == "c" else None,
        "supported_visible_same_type_ids": [4, 6, 7, 18] if task == "c" else None,
        "human_visible_same_type_object_ids": [4, 6, 7, 18] if task == "c" else None,
        "generated_question": "I want to sit near the table.",
    }
    if task == "b1":
        gt["walkable_ids"] = [1, 3]
    return {
        "question_id": f"q_{task}",
        "task": task,
        "user_prompt": "Return the path.",
        "system_prompt": "System",
        "image_path": "/tmp/image.png",
        "visible_waypoints_path": write_visible(tmp_path),
        "ground_truth": gt,
    }


class _DummyHTTPResponse:
    def __init__(self, payload: str) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload.encode("utf-8")

    def __enter__(self) -> "_DummyHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


@pytest.mark.parametrize("task", oracle_cot.TASKS)
def test_validate_gpt_cot_text_accepts_plain_text(tmp_path: Path, task: str) -> None:
    packet = oracle_cot.build_fact_packet(base_row(tmp_path, task))
    answer = json.dumps(packet["required_final_json"], ensure_ascii=True)
    if task == "a1":
        text = (
            "<think>\n"
            "I first look over the visible candidate IDs [1, 2, 3]. "
            "For a point agent, all of them are walkable, so there are no blocked IDs. "
            "The walkable IDs are [1, 2, 3], and I apply the rule that select all display IDs traversable for a point agent.\n"
            "</think>\n"
            "[1, 2, 3]"
        )
    elif task == "b1":
        text = (
            "<think>\n"
            "I first look over the visible candidate IDs [1, 2, 3]. "
            "For the robot body, IDs [1, 3] are embodied-feasible and leave enough room, while ID [2] is too tight. "
            "I keep the embodied-feasible set [1, 3] and apply the rule that select all display IDs traversable for the robot body.\n"
            "</think>\n"
            "[1, 3]"
        )
    elif task == "a2":
            text = (
                "<think>\n"
                "I first inspect the visible points in the image and count the walkable ones. "
                "The visual summary says all 3 visible points are pointmass-walkable and none are blocked. "
                "I ground the target as chair at goal display ID 3, so from start ID 1 I trace [1, 3] to the goal. "
                "The adjacent segment from 1 to 3 is collision-free, and the geometry facts show path length 1.25, a straight-short route with zero turns, and detour ratio 1.0, so the direct path is the shortest legal one rather than a detour.\n"
                "</think>\n"
                "[1, 3]"
            )
    elif task == "c":
        text = (
            "<think>\n"
            "I first read the request and cue, then inspect the visible points in the scene: place_object with the under relation. "
                "The same-type visibility summary shows 4 candidate targets, and the candidate target IDs are [4, 6, 7, 18], so I can compare the options before choosing the winner target ID 7 and goal display ID 3. "
                "The visual summary shows the embodied points I can use, and the robot constraint plus clearance facts mean I should avoid blocked or risky points. "
                "From start ID 1, I follow the path [1, 3], and the adjacent segment is collision-free while the route facts show path length 1.25, detour ratio 1.0, and zero turns, so the embodied route is shortest without leaning only on the length number.\n"
                "</think>\n"
                "[1, 3]"
            )
    elif task == "b2":
        text = (
            "<think>\n"
            "I first inspect the visible points and the robot constraints. "
                "The visual summary shows 3 visible points, with one blocked point and two embodied-feasible points, so I avoid the blocked point and follow the route that stays feasible for the robot body. "
                "The grounded target is the table at goal display ID 3, and the robot needs diameter 0.6 with clearance 0.05, so the path must respect the embodied constraint. "
                "I trace the route from start ID 1 to [1, 3], and the adjacent segment is collision-free while the embodied facts show path length 1.25, detour ratio 1.0, zero turns, and no extra embodied length, which supports that this is the shortest embodied path.\n"
                "</think>\n"
                "[1, 3]"
            )
    else:
        text = f"<think>\nThe pointmass traversability facts select the final IDs.\n</think>\n{answer}"

    qc = gpt_cot.validate_cot_text(text, packet)

    assert qc["ok"]
    assert qc["final_json"] == packet["required_final_json"]


def test_validate_gpt_cot_text_rejects_json_envelope(tmp_path: Path) -> None:
    packet = oracle_cot.build_fact_packet(base_row(tmp_path, "b2"))
    text = json.dumps({"reasoning_payload": {}, "final_json": [1, 3]})

    qc = gpt_cot.validate_cot_text(text, packet)

    assert not qc["ok"]
    assert "json_envelope_output" in qc["issues"]
    assert "internal_schema_field_leak" in qc["issues"]


def test_response_text_extracts_responses_api_output() -> None:
    response = {
        "output": [
            {
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "<think>\nReasoning text.\n</think>\n[1, 3]"}
                ],
            }
        ]
    }

    assert gpt_cot.response_text(response) == "<think>\nReasoning text.\n</think>\n[1, 3]"


def test_user_prompt_requests_plain_text_not_json(tmp_path: Path) -> None:
    packet = oracle_cot.build_fact_packet(base_row(tmp_path, "c"))
    prompt = gpt_cot.user_prompt(packet)

    assert '"format": "plain_text"' in prompt
    assert "JSON envelope" in prompt
    assert "<think>" in prompt
    assert "</think>" in prompt
    assert "image_path" not in prompt
    assert "visible_waypoints_path" not in prompt


def test_a2_gpt_prompt_and_qc_exclude_embodied_language(tmp_path: Path) -> None:
    packet = oracle_cot.build_fact_packet(base_row(tmp_path, "a2"))
    safe_packet = gpt_cot.gpt_fact_packet(packet)
    waypoints = safe_packet["facts"]["path_facts"]["path_waypoints"]

    assert all("embodied_feasible" not in waypoint for waypoint in waypoints)
    assert all("embodied_clearance_margin_m" not in waypoint for waypoint in waypoints)

    qc = gpt_cot.validate_cot_text(
        "<think>\nThe robot embodied route is [1, 3].\n</think>\n[1, 3]",
        packet,
    )
    assert not qc["ok"]
    assert "a2_mentions_robot_or_embodied" in qc["issues"]

    label_qc = gpt_cot.validate_cot_text(
        "<think>\nTask analysis: pointmass routing. Target judgment: ...\n</think>\n[1, 3]",
        packet,
    )
    assert not label_qc["ok"]
    assert "label_style_not_allowed" in label_qc["issues"]


@pytest.mark.parametrize("task", ["a1", "b1"])
def test_gpt_fact_packet_keeps_full_candidate_split(tmp_path: Path, task: str) -> None:
    packet = oracle_cot.build_fact_packet(base_row(tmp_path, task))
    safe_packet = gpt_cot.gpt_fact_packet(packet)
    candidate_space = safe_packet["facts"]["candidate_space"]

    assert candidate_space["all_display_ids"] == [1, 2, 3]
    if task == "a1":
        assert candidate_space["pointmass_walkable_ids"] == [1, 2, 3]
        assert candidate_space["non_walkable_ids"] == []
    else:
        assert candidate_space["embodied_feasible_ids"] == [1, 3]
        assert candidate_space["not_embodied_feasible_ids"] == [2]


@pytest.mark.parametrize("task", ["a1", "b1"])
def test_validate_gpt_cot_text_rejects_examples_only_traversability(tmp_path: Path, task: str) -> None:
    packet = oracle_cot.build_fact_packet(base_row(tmp_path, task))
    if task == "a1":
        text = "<think>\nI keep ID 1 because it is walkable and ID 2 because it is blocked.\n</think>\n[1, 2, 3]"
    else:
        text = "<think>\nI keep ID 1 because it leaves enough room and ID 2 because it is too tight.\n</think>\n[1, 3]"

    qc = gpt_cot.validate_cot_text(text, packet)

    assert not qc["ok"]
    assert any(issue.startswith("missing_complete_") for issue in qc["issues"])


def test_compact_planning_prompt_keeps_path_facts(tmp_path: Path) -> None:
    packet = oracle_cot.build_fact_packet(base_row(tmp_path, "b2"))
    full_prompt = gpt_cot.user_prompt(packet)
    compact_prompt = gpt_cot.user_prompt(packet, compact_planning_prompt=True)

    assert len(compact_prompt) <= len(full_prompt)
    assert '"shortest_path"' in compact_prompt
    assert '"target_grounding"' in compact_prompt
    assert '"robot_constraint"' in compact_prompt
    assert '"embodied_clearance_margin_m"' in compact_prompt


def test_fact_packet_carries_visual_and_same_type_summaries(tmp_path: Path) -> None:
    a2_packet = oracle_cot.build_fact_packet(base_row(tmp_path, "a2"))
    b2_packet = oracle_cot.build_fact_packet(base_row(tmp_path, "b2"))
    c_packet = oracle_cot.build_fact_packet(base_row(tmp_path, "c"))

    assert a2_packet["facts"]["visual_summary"]["total_visible_points"] == 3
    assert a2_packet["facts"]["visual_summary"]["blocked_count"] == 0
    assert b2_packet["facts"]["visual_summary"]["blocked_count"] == 1
    assert c_packet["facts"]["target_selection"]["candidate_target_count"] == 4
    assert c_packet["facts"]["same_type_visibility"]["same_type_supported_visible_count"] == 4
    assert a2_packet["facts"]["path_facts"]["path_projection_safe"] is True
    assert b2_packet["facts"]["path_facts"]["path_segments"][0]["collision_free_projection"] is True


def test_planning_prompt_uses_complete_visual_id_lists(tmp_path: Path) -> None:
    visible_path = tmp_path / "visible_many.json"
    visible_path.write_text(
        json.dumps(
            {
                "visible_waypoints": [
                    {
                        "display_id": index,
                        "pointmass_walkable": index <= 12,
                        "embodied_feasible": index % 2 == 0,
                    }
                    for index in range(1, 15)
                ]
            }
        )
    )
    row = base_row(tmp_path, "a2")
    row["visible_waypoints_path"] = str(visible_path)
    row["ground_truth"]["all_ids"] = list(range(1, 15))
    row["ground_truth"]["answer"] = [1, 12]
    row["ground_truth"]["goal_id"] = 12
    packet = oracle_cot.build_fact_packet(row)

    prompt = gpt_cot.plain_planning_user_prompt(packet)

    assert "Visible numbered points: 14; point-agent usable IDs [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]; point-agent blocked IDs [13, 14]." in prompt
    assert "Route to explain: [1, 12]." in prompt
    assert "Shortest-route cues:" in prompt
    assert "The usable IDs are [1, 2, 3, 4, 5, 6, 7, 8]" not in prompt


@pytest.mark.parametrize("task", ["a1", "b1"])
def test_traversability_prompt_uses_exact_complete_lists(tmp_path: Path, task: str) -> None:
    packet = oracle_cot.build_fact_packet(base_row(tmp_path, task))

    prompt = gpt_cot.planning_user_prompt(packet)

    assert "Visible candidate IDs: [1, 2, 3]." in prompt
    if task == "a1":
        assert "Pointmass-walkable IDs: [1, 2, 3]." in prompt
        assert "Pointmass-blocked IDs: []" in prompt
    else:
        assert "Robot-body feasible IDs: [1, 3]." in prompt
        assert "Robot-body too-tight IDs: [2]." in prompt
    assert "visible candidate set as a continuous range" in prompt
    assert "subset lists must stay exact bracketed lists" in prompt
    assert "1 through 3" not in prompt
    assert '"fact_packet"' not in prompt


@pytest.mark.parametrize(
    ("task", "expected_phrases"),
    [
        ("a2", ["first-person reasoning paragraph", "visual observations", "point-agent usable ids", "target selection rule: nearest", "route to explain", "no detour"]),
        ("b2", ["first-person reasoning paragraph", "robot-body usable ids", "too-tight or blocked ids", "same-category candidate target ids", "tightest usable clearance", "goal display id"]),
        ("c", ["first-person reasoning paragraph", "user request", "same-type candidate target ids", "winner target id", "robot diameter: about 60 cm", "narrow-passage share"]),
    ],
)
def test_plain_planning_prompt_contains_task_contract(
    tmp_path: Path, task: str, expected_phrases: list[str]
) -> None:
    packet = oracle_cot.build_fact_packet(base_row(tmp_path, task))
    prompt = gpt_cot.plain_planning_user_prompt(packet)

    for phrase in expected_phrases:
        assert phrase in prompt.lower()
    assert "task analysis:" not in prompt.lower()
    assert "target judgment:" not in prompt.lower()
    assert "final json:" not in prompt.lower()
    assert "do not copy the evidence mechanically" in prompt.lower()
    assert "evidence:" in prompt.lower()
    assert "segment" in prompt.lower()


@pytest.mark.parametrize("api_endpoint", ["chat", "responses"])
def test_request_text_completion_uses_plain_planning_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api_endpoint: str
) -> None:
    packet = oracle_cot.build_fact_packet(base_row(tmp_path, "a2"))
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _DummyHTTPResponse(json.dumps({"output_text": "<think>\nReasoning.\n</think>\n[1, 3]"}))

    monkeypatch.setattr(gpt_cot.urllib.request, "urlopen", fake_urlopen)

    gpt_cot.request_text_completion(
        api_base="https://example.invalid",
        api_key="key",
        model="gpt-5.5",
        packet=packet,
        max_completion_tokens=64,
        timeout_s=1.0,
        temperature=0.0,
        top_p=1.0,
        api_endpoint=api_endpoint,
        compact_planning_prompt=False,
        minimal_planning_prompt=False,
        plain_planning_prompt=True,
        codex_request_headers=False,
    )

    body = captured["body"]
    if api_endpoint == "chat":
        assert isinstance(body, dict)
        assert body["messages"][1]["content"].startswith("Write one flowing first-person reasoning paragraph in <think> tags.")
    else:
        assert isinstance(body, dict)
        assert body["input"].startswith("Write one flowing first-person reasoning paragraph in <think> tags.")


@pytest.mark.parametrize("api_endpoint", ["chat", "responses"])
def test_request_text_completion_defaults_to_plain_planning_prompt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api_endpoint: str
) -> None:
    packet = oracle_cot.build_fact_packet(base_row(tmp_path, "a2"))
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return _DummyHTTPResponse(json.dumps({"output_text": "<think>\nReasoning.\n</think>\n[1, 3]"}))

    monkeypatch.setattr(gpt_cot.urllib.request, "urlopen", fake_urlopen)

    gpt_cot.request_text_completion(
        api_base="https://example.invalid",
        api_key="key",
        model="gpt-5.5",
        packet=packet,
        max_completion_tokens=64,
        timeout_s=1.0,
        temperature=0.0,
        top_p=1.0,
        api_endpoint=api_endpoint,
        compact_planning_prompt=False,
        minimal_planning_prompt=False,
        plain_planning_prompt=False,
        codex_request_headers=False,
    )

    body = captured["body"]
    assert isinstance(body, dict)
    if api_endpoint == "chat":
        assert body["messages"][1]["content"].startswith("Write one flowing first-person reasoning paragraph in <think> tags.")
    else:
        assert body["input"].startswith("Write one flowing first-person reasoning paragraph in <think> tags.")


def test_build_request_headers_can_mimic_codex_client() -> None:
    plain_headers = gpt_cot.build_request_headers("key", codex_request_headers=False)
    assert plain_headers == {
        "Authorization": "Bearer key",
        "Content-Type": "application/json",
    }

    codex_headers = gpt_cot.build_request_headers("key", codex_request_headers=True)
    assert codex_headers["Authorization"] == "Bearer key"
    assert codex_headers["originator"] == "codex_exec"
    assert codex_headers["user-agent"].startswith("codex_exec/0.133.0")
    assert codex_headers["Accept"] == "text/event-stream"
    metadata = json.loads(codex_headers["x-codex-turn-metadata"])
    assert metadata["session_id"] == codex_headers["session-id"]
    assert metadata["thread_id"] == codex_headers["thread-id"]
    assert metadata["thread_source"] == "user"
    assert codex_headers["x-codex-window-id"] == f"{metadata['session_id']}:0"


@pytest.mark.parametrize("task", ["a2", "b2", "c"])
def test_validate_gpt_cot_text_requires_reasoning_chain(tmp_path: Path, task: str) -> None:
    packet = oracle_cot.build_fact_packet(base_row(tmp_path, task))
    answer = json.dumps(packet["required_final_json"], ensure_ascii=True)
    path_facts = packet["facts"]["path_facts"]
    if task == "a2":
        text = (
            f"<think>\nI inspect the visible points and count the walkable ones. "
            f"I ground the target as chair at goal display ID {packet['facts']['target_grounding']['goal_display_id']}. "
            f"From start ID {path_facts['start_id']} I trace path {path_facts['shortest_path']} to the goal. "
            f"The geometry facts show path length {path_facts['path_length_m']}, detour ratio 1.0 and zero turns, and the segment 1->3 is collision-free, so the shortest path is the direct one. "
            f"I conclude with the final pointmass path {answer}.\n</think>\n{answer}"
        )
    elif task == "b2":
        target = packet["facts"]["target_grounding"]
        robot = packet["facts"]["robot_constraint"]
        text = (
            f"<think>\nI inspect the visible points and note the embodied-feasible ones versus the blocked ones. "
            f"The target is the table at goal display ID {target['goal_display_id']}, and the robot diameter {robot['robot_diameter_m']} with clearance {robot['min_clearance_m']} means I have to respect the embodied route. "
            f"The path from start ID {path_facts['start_id']} is {path_facts['shortest_path']}, with path length {path_facts['path_length_m']}, and the embodied facts show detour ratio 1.0 and zero turns, while the adjacent segment is collision-free, so the route is shortest without relying only on length. "
            f"I conclude with the final embodied path {answer}.\n</think>\n{answer}"
        )
    else:
        intent = packet["facts"]["intent_resolution"]
        cue = packet["facts"]["cue_resolution"]
        selection = packet["facts"]["target_selection"]
        robot = packet["facts"]["robot_constraint"]
        text = (
            f"<think>\nI read the request and cue, then compare the same-type candidate targets {selection['candidate_target_ids']}. "
            f"The winner target ID is {selection['winner_target_id']}, and the goal display ID is {selection['goal_display_id']}. "
            f"I inspect the visible points, note the embodied-feasible ones, and keep the robot diameter {robot['robot_diameter_m']} and clearance facts in mind. "
            f"From start ID {path_facts['start_id']} I follow the path {path_facts['shortest_path']}, and the route facts show path length {path_facts['path_length_m']}, detour ratio 1.0, and zero turns, while the route segments are collision-free, so the embodied route is shortest without leaning only on length.\n</think>\n{answer}"
        )
    qc = gpt_cot.validate_cot_text(text, packet)
    assert qc["ok"]


def test_validate_gpt_cot_text_accepts_natural_path_wording(tmp_path: Path) -> None:
    packet = oracle_cot.build_fact_packet(base_row(tmp_path, "a2"))
    answer = json.dumps(packet["required_final_json"], ensure_ascii=True)
    text = (
        "<think>\n"
        "I first notice that the scene has several visible point markers and that all visible IDs are usable under the pointmass walkable check, "
        "so there is no blocked point that would force me away from a simple route. "
        "From what I can see, the target to ground is the nearest chair, which corresponds to display ID 3, so I start from ID 1 and check how to reach that target. "
        "Because the 1->3 connection is clear and collision-free, the route can go straight from 1 to 3 without inserting any intermediate point, "
        "which means the path has no turns and no sideways detour. "
        "That direct shape matters: any alternative would have to bend through extra points, while the direct path stays aligned and remains about 1.25 meters long. "
        f"Therefore the shortest route is the straight no-detour path from start ID 1 to the goal at ID 3, so the final pointmass path is {answer}.\n"
        "</think>\n"
        f"{answer}"
    )

    qc = gpt_cot.validate_cot_text(text, packet)

    assert qc["ok"]
    assert qc["final_json"] == [1, 3]


def test_validate_gpt_cot_text_accepts_point_mass_and_starting_at(tmp_path: Path) -> None:
    packet = oracle_cot.build_fact_packet(base_row(tmp_path, "a2"))
    answer = json.dumps(packet["required_final_json"], ensure_ascii=True)
    text = (
        "<think>\n"
        "I inspect the visible numbered points and see that the nearest chair is the selected target at goal display ID 3. "
        "Starting at 1, the route is feasible for point-mass travel because the endpoint is walkable and no blocked point interrupts the direct move. "
        "The adjacent segment from 1 to 3 is collision-free, with the dense path also clear, so I do not need an intermediate waypoint. "
        "Because the path is straight, has no detour, has no turns, and is about 1.25 meters long, the shortest valid route is [1, 3].\n"
        "</think>\n"
        f"{answer}"
    )

    qc = gpt_cot.validate_cot_text(text, packet)

    assert qc["ok"]


def test_validate_gpt_cot_text_accepts_margin_as_clearance(tmp_path: Path) -> None:
    packet = oracle_cot.build_fact_packet(base_row(tmp_path, "b2"))
    answer = json.dumps(packet["required_final_json"], ensure_ascii=True)
    text = (
        "<think>\n"
        "I look at the visible numbered points, look for the nearest cushion, and resolve the target to the goal display ID 3. "
        "Starting at 1, the robot body has enough room at the start and a small but usable margin at the goal, while the only too-tight point is blocked out of this route, so the embodied move is still valid. "
        "The 1 to 3 connection is collision-free, and the dense check stays clear, so I can keep the path direct. "
        "Because the route has no detour and no turns and is about 1.25 meters long, the direct embodied route is the shortest valid one.\n"
        "</think>\n"
        f"{answer}"
    )

    qc = gpt_cot.validate_cot_text(text, packet)

    assert qc["ok"]


def test_validate_gpt_cot_text_requires_collision_free_segment_language(tmp_path: Path) -> None:
    packet = oracle_cot.build_fact_packet(base_row(tmp_path, "a2"))
    answer = json.dumps(packet["required_final_json"], ensure_ascii=True)
    text = (
        "<think>\n"
        "I inspect the visible points and see that the point-agent walkable list is open while no blocked point affects the route. "
        "The nearest chair is the target at goal display ID 3, so from start ID 1 I can go straight from 1 to 3. "
        "This direct route has no turns, no detour, and is about 1.25 m long, so it is the shortest pointmass path. "
        f"I therefore choose {answer}.\n"
        "</think>\n"
        f"{answer}"
    )

    qc = gpt_cot.validate_cot_text(text, packet)

    assert not qc["ok"]
    assert "missing_segment_collision_free_element" in qc["issues"]


def test_validate_gpt_cot_text_accepts_natural_implicit_single_candidate(tmp_path: Path) -> None:
    packet = oracle_cot.build_fact_packet(base_row(tmp_path, "c"))
    answer = json.dumps(packet["required_final_json"], ensure_ascii=True)
    text = (
        "<think>\n"
        "I interpret the request as looking for a sink, and in the visible scene there is only one sink-like target to choose from, "
        "so I do not have to compare multiple competing sinks before using the object at ID 3 as the goal. "
        "Starting from ID 1, I check that the robot body can fit through the visible points because clearance matters for embodied movement. "
        "The adjacent segment from 1 to 3 is collision-free and the denser trace is clear too, so the robot can move through the connection without hitting anything. "
        "Since the route has no detour, no turns, and is about 1.25 m long, it is the shortest valid embodied route to the resolved target. "
        f"I therefore follow {answer}.\n"
        "</think>\n"
        f"{answer}"
    )

    qc = gpt_cot.validate_cot_text(text, packet)

    assert qc["ok"]
    assert qc["final_json"] == [1, 3]


def test_path_segment_evidence_text_mentions_collision_free_edges() -> None:
    text = gpt_cot.segment_evidence_text(
        [
            {"from_id": 1, "to_id": 8, "collision_free_projection": True, "dense_collision_free_projection": True},
            {"from_id": 8, "to_id": 12, "collision_free_projection": True, "dense_collision_free_projection": True},
        ]
    )

    assert text is not None
    assert "1->8" in text
    assert "collision-free projection" in text


@pytest.mark.parametrize("task", ["a1", "b1"])
def test_validate_traversability_accepts_continuous_visible_range(tmp_path: Path, task: str) -> None:
    packet = oracle_cot.build_fact_packet(base_row(tmp_path, task))
    answer = json.dumps(packet["required_final_json"], ensure_ascii=True)
    if task == "a1":
        text = (
            "<think>\n"
            "I first look over the full visible candidate set from 1 through 3. "
            "For a point agent, the walkable IDs are [1, 2, 3], and the blocked IDs are []. "
            "Since every visible candidate is pointmass-walkable, I keep exactly [1, 2, 3].\n"
            "</think>\n"
            f"{answer}"
        )
    else:
        text = (
            "<think>\n"
            "I first look over the full visible candidate set from 1 through 3 for the robot body. "
            "The embodied-feasible IDs are [1, 3], and the too tight blocked IDs are [2]. "
            "Because only [1, 3] leave enough room with clearance for the robot, I keep exactly that feasible set.\n"
            "</think>\n"
            f"{answer}"
        )

    qc = gpt_cot.validate_cot_text(text, packet)

    assert qc["ok"]


def test_evaluate_generated_text_skip_validation_accepts_parse_only(tmp_path: Path) -> None:
    packet = oracle_cot.build_fact_packet(base_row(tmp_path, "a2"))
    text = "whatever this is\n[1, 3]"

    qc = gpt_cot.evaluate_generated_text(text, packet, skip_validation=True)

    assert qc["ok"]
    assert qc["final_json"] == [1, 3]
    assert qc["validation_skipped"] is True


def test_run_uses_shards_and_writes_per_shard_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packets = [
        {
            "question_id": f"q_{index}",
            "task": "a2",
            "required_final_json": [1, 3],
        }
        for index in range(4)
    ]

    def fake_selected_packets(args):
        return packets

    def fake_call_with_retries(args, api_base, api_key, packet):
        return {"output_text": f"<think>\nReasoning for {packet['question_id']}.\n</think>\n[1, 3]"}

    monkeypatch.setattr(gpt_cot.oracle, "resolve_api_base", lambda api_base, api_base_env: "https://example.invalid")
    monkeypatch.setattr(gpt_cot.oracle, "resolve_api_key", lambda api_key_env: "key")
    monkeypatch.setattr(gpt_cot, "selected_packets", fake_selected_packets)
    monkeypatch.setattr(gpt_cot, "call_with_retries", fake_call_with_retries)

    args = SimpleNamespace(
        allow_external_upload=True,
        shard_count=2,
        shard_index=1,
        api_base=None,
        api_base_env="OPENAI_BASE_URL",
        api_key_env="OPENAI_API_KEY",
        output_dir=str(tmp_path / "out"),
        resume=False,
        model="gpt-5.5",
        max_retries=0,
        timeout_s=1.0,
        temperature=0.0,
        top_p=1.0,
        api_endpoint="chat",
        compact_planning_prompt=False,
        minimal_planning_prompt=False,
        plain_planning_prompt=True,
        codex_request_headers=False,
        max_completion_tokens=64,
        request_sleep_s=0.0,
        progress_every=0,
        skip_validation=True,
        shard_size=500,
    )

    report = gpt_cot.run(args)

    accepted = sorted((tmp_path / "out" / "accepted" / "a2").glob("*.jsonl"))
    raw = sorted((tmp_path / "out" / "raw_responses" / "a2").glob("*.jsonl"))
    report_path = tmp_path / "out" / "qc_reports" / "shard_01_of_02.json"

    assert report["request_config"]["skip_validation"] is True
    assert report["request_config"]["shard_index"] == 1
    assert report["request_config"]["shard_count"] == 2
    assert report_path.exists()
    assert [path.name for path in accepted] == ["shard_00000.jsonl"]
    assert [path.name for path in raw] == ["shard_00000.jsonl"]

    accepted_rows = [json.loads(line) for line in accepted[0].read_text().splitlines() if line.strip()]
    assert [row["question_id"] for row in accepted_rows] == ["q_1", "q_3"]


def test_existing_accepted_qids_includes_reqc_rows(tmp_path: Path) -> None:
    accepted = tmp_path / "out" / "accepted" / "a2" / "shard_00000.jsonl"
    accepted_reqc = tmp_path / "out" / "accepted_reqc" / "c" / "shard_00000.jsonl"
    accepted.parent.mkdir(parents=True)
    accepted_reqc.parent.mkdir(parents=True)
    accepted.write_text(json.dumps({"question_id": "q_regular"}) + "\n")
    accepted_reqc.write_text(json.dumps({"question_id": "q_reqc"}) + "\n")

    assert gpt_cot.existing_accepted_qids(tmp_path / "out") == {"q_regular", "q_reqc"}
