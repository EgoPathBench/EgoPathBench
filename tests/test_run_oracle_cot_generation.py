from __future__ import annotations

import importlib.util
import http.client
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_oracle_cot_generation.py"
SPEC = importlib.util.spec_from_file_location("run_oracle_cot_generation", SCRIPT_PATH)
oracle_cot = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(oracle_cot)


def write_visible(tmp_path: Path) -> str:
    path = tmp_path / "visible_waypoints_b2.json"
    payload = {
        "visible_waypoints": [
            {
                "display_id": 1,
                "pointmass_walkable": True,
                "embodied_feasible": True,
                "point_clearance_m": 0.45,
                "embodied_clearance_margin_m": 0.2,
                "screen_xy_norm": [0.0, -1.0],
            },
            {
                "display_id": 2,
                "pointmass_walkable": True,
                "embodied_feasible": False,
                "point_clearance_m": 0.05,
                "embodied_clearance_margin_m": 0.0,
                "screen_xy_norm": [0.1, -0.7],
            },
            {
                "display_id": 3,
                "pointmass_walkable": True,
                "embodied_feasible": True,
                "point_clearance_m": 0.4,
                "embodied_clearance_margin_m": 0.15,
                "screen_xy_norm": [0.2, -0.4],
            },
        ]
    }
    path.write_text(json.dumps(payload))
    return str(path.resolve())


def base_row(tmp_path: Path, task: str) -> dict:
    visible_path = write_visible(tmp_path)
    gt = {
        "answer": [1, 3] if task in {"a2", "b2", "c"} else [1, 2, 3],
        "all_ids": [1, 2, 3],
        "walkable_ids": [1, 2, 3],
        "start_id": 1,
        "goal_id": 3,
        "goal_ids": [3],
        "target_id": 99,
        "target_category": "chair",
        "target_rule": "nearest",
        "robot_diameter_m": 0.6,
        "route_semantics": "embodied" if task in {"b2", "c"} else "pointmass",
        "canonical_sparse_length_m": 1.25,
        "navigation_embodiment_facts": {
            "path_length_embodied_m": 1.25 if task in {"b2", "c"} else None,
            "detour_ratio_embodied": 1.0 if task in {"b2", "c"} else None,
            "turn_count_embodied": 0 if task in {"b2", "c"} else None,
            "embodied_clearance_margin_min_m": 0.05,
            "narrow_passage_fraction": 0.3,
        },
        "navigation_geometry_facts": {
            "path_length_point_m": 1.25,
            "detour_ratio_point": 1.0,
            "turn_count_point": 0,
            "geometry_profile": "straight_short",
        },
        "routing_complexity": {
            "num_segments_sparse": 1 if task == "a2" else None,
            "point_path_projection_safe": True if task == "a2" else None,
            "point_dense_path_projection_safe": True if task == "a2" else None,
            "embodied_num_segments_sparse": 1 if task in {"b2", "c"} else None,
            "embodied_path_projection_safe": True if task in {"b2", "c"} else None,
            "embodied_dense_path_projection_safe": True if task in {"b2", "c"} else None,
            "embodied_min_clearance_m": 0.05 if task in {"b2", "c"} else None,
            "embodied_narrow_passage_fraction": 0.3 if task in {"b2", "c"} else None,
            "embodied_detour_ratio_sparse": 1.0 if task in {"b2", "c"} else None,
        },
        "instruction_selection": {
            "cohort_target_ids": [99, 100],
            "winner_target_ids": [99],
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
        "generated_question": "I want to sit near the table.",
    }
    if task == "b1":
        gt["answer"] = [1, 3]
        gt["walkable_ids"] = [1, 3]
    return {
        "question_id": f"q_{task}",
        "task": task,
        "user_prompt": "Return the path.",
        "system_prompt": "System",
        "image_path": "/tmp/image.png",
        "visible_waypoints_path": visible_path,
        "ground_truth": gt,
    }


def test_build_fact_packets_are_task_specific(tmp_path: Path) -> None:
    for task in oracle_cot.TASKS:
        packet = oracle_cot.build_fact_packet(base_row(tmp_path, task))
        assert packet["task"] == task
        assert packet["reasoning_type"] == oracle_cot.REASONING_TYPES[task]
        assert packet["candidate_display_ids"] == [1, 2, 3]
        assert packet["required_final_json"]

    a1 = oracle_cot.build_fact_packet(base_row(tmp_path, "a1"))
    assert "pointmass_walkable_ids" in a1["facts"]["candidate_space"]

    b2 = oracle_cot.build_fact_packet(base_row(tmp_path, "b2"))
    assert b2["facts"]["robot_constraint"]["robot_diameter_m"] == 0.6
    assert b2["facts"]["path_facts"]["shortest_path"] == [1, 3]
    assert b2["facts"]["path_facts"]["path_segments"][0]["collision_free_projection"] is True

    c = oracle_cot.build_fact_packet(base_row(tmp_path, "c"))
    assert c["facts"]["intent_resolution"]["intent_family"] == "sit_rest"
    assert c["facts"]["target_selection"]["goal_display_id"] == 3


def test_visible_point_summary_keeps_complete_id_lists() -> None:
    visible_waypoints = [
        {
            "display_id": index,
            "pointmass_walkable": index <= 12,
            "embodied_feasible": index % 2 == 0,
        }
        for index in range(1, 19)
    ]

    point_summary = oracle_cot.visible_point_summary(visible_waypoints, embodied=False)
    embodied_summary = oracle_cot.visible_point_summary(visible_waypoints, embodied=True)

    assert point_summary["feasible_display_ids"] == list(range(1, 13))
    assert point_summary["blocked_display_ids"] == list(range(13, 19))
    assert embodied_summary["feasible_display_ids"] == [2, 4, 6, 8, 10, 12, 14, 16, 18]
    assert embodied_summary["blocked_display_ids"] == [1, 3, 5, 7, 9, 11, 13, 15, 17]


def test_merge_row_with_gt_fills_missing_ground_truth_without_overwrite() -> None:
    row = {
        "question_id": "q_a2",
        "task": "a2",
        "image_path": "/tmp/vqa.png",
        "ground_truth": {"answer": [1, 2], "target_category": "chair"},
    }
    gt_row = {
        "question_id": "q_a2",
        "task": "a2",
        "image_path": "/tmp/gt.png",
        "start_id": 1,
        "goal_id": 2,
        "target_category": "table",
    }
    merged = oracle_cot.merge_row_with_gt(row, gt_row)
    assert merged["image_path"] == "/tmp/vqa.png"
    assert merged["ground_truth"]["answer"] == [1, 2]
    assert merged["ground_truth"]["target_category"] == "chair"
    assert merged["ground_truth"]["start_id"] == 1
    assert merged["ground_truth"]["goal_id"] == 2


def test_select_task_shard_partitions_each_task(tmp_path: Path) -> None:
    packets = []
    for task in oracle_cot.TASKS:
        for index in range(7):
            row = base_row(tmp_path, task)
            row["question_id"] = f"q_{task}_{index}"
            packets.append(oracle_cot.build_fact_packet(row))

    shards = [
        oracle_cot.select_task_shard(packets, shard_index=index, shard_count=3)
        for index in range(3)
    ]
    shard_qids = [{packet["question_id"] for packet in shard} for shard in shards]

    assert set().union(*shard_qids) == {packet["question_id"] for packet in packets}
    assert not (shard_qids[0] & shard_qids[1])
    assert not (shard_qids[0] & shard_qids[2])
    assert not (shard_qids[1] & shard_qids[2])
    assert all(any(packet["task"] == task for packet in shard) for shard in shards for task in oracle_cot.TASKS)
    assert all("_task_index_for_shard" in packet for shard in shards for packet in shard)
    assert "_task_index_for_shard" not in oracle_cot.packet_for_generation(shards[0][0])

    with pytest.raises(oracle_cot.OracleCotError):
        oracle_cot.select_task_shard(packets, shard_index=3, shard_count=3)


def test_validate_generated_rejects_mismatch_and_leakage(tmp_path: Path) -> None:
    packet = oracle_cot.build_fact_packet(base_row(tmp_path, "a2"))
    valid = {
        "question_id": "q_a2",
        "task": "a2",
        "reasoning_type": "pointmass_explicit_shortest_path",
        "reasoning_payload": {
            "target_grounding": {
                "target_rule": "nearest",
                "target_category": "chair",
                "goal_display_id": 3,
            },
            "path_facts": {
                "start_id": 1,
                "shortest_path": [1, 3],
                "path_length_m": 1.25,
                "route_semantics": "pointmass",
            },
            "shortest_path_explanation": "This is the supplied shortest legal route.",
        },
        "final_json": [1, 3],
    }
    assert oracle_cot.validate_generated(valid, packet)["ok"]

    invalid = {
        **valid,
        "reasoning_payload": {"note": "metadata says use this hash"},
        "final_json": [1, 4],
    }
    qc = oracle_cot.validate_generated(invalid, packet)
    assert not qc["ok"]
    assert "final_json_mismatch" in qc["issues"]
    assert any(issue.startswith("banned_leakage") for issue in qc["issues"])


def test_audit_sparse_oracle_path_without_direct_pairs(tmp_path: Path) -> None:
    packet = oracle_cot.build_fact_packet(base_row(tmp_path, "a2"))
    packet["audit_refs"]["direct_pairs_ref"] = None

    audit = oracle_cot.audit_path_edge_legality(packet, [1, 3])

    assert audit["ok"] is True
    assert audit["status"] == "checked_ok_oracle_sparse_path"
    assert audit["routing_mode"] == "oracle_sparse"


def test_validate_generated_requires_task_schema(tmp_path: Path) -> None:
    packet = oracle_cot.build_fact_packet(base_row(tmp_path, "c"))
    missing_cue_payload = {
        "question_id": "q_c",
        "task": "c",
        "reasoning_type": "implicit_goal_embodied_shortest_path",
        "reasoning_payload": {
            "intent_resolution": {
                "user_request": "I want to sit near the table.",
                "intent_family": "sit_rest",
                "resolved_target_category": "chair",
            },
            "target_selection": {"winner_target_id": 99, "goal_display_id": 3},
            "path_facts": {"start_id": 1, "shortest_path": [1, 3], "route_semantics": "embodied"},
        },
        "final_json": [1, 3],
    }
    qc = oracle_cot.validate_generated(missing_cue_payload, packet)
    assert not qc["ok"]
    assert "schema_missing:reasoning_payload.cue_resolution" in qc["issues"]


def test_sharegpt_rows_use_single_image_turn(tmp_path: Path) -> None:
    packet = oracle_cot.build_fact_packet(base_row(tmp_path, "b2"))
    direct = oracle_cot.build_direct_sft_row(packet)
    assert direct["messages"][0]["role"] == "user"
    assert direct["messages"][0]["content"].startswith("<image>")
    assert direct["messages"][1]["content"] == "[1, 3]"
    assert direct["images"] == ["/tmp/image.png"]

    parsed = {
        "question_id": "q_b2",
        "task": "b2",
        "reasoning_type": "embodied_explicit_shortest_path",
        "reasoning_payload": {"path_facts": {"shortest_path": [1, 3]}},
        "final_json": [1, 3],
    }
    cot = oracle_cot.build_cot_sft_row(packet, parsed)
    assistant = cot["messages"][1]["content"]
    assert assistant.startswith("<think>\n")
    assert "\n</think>\n[1, 3]" in assistant
    assert "Final JSON:" not in assistant
    assert not assistant.lstrip().startswith("{")
    assert "reasoning_payload" not in assistant
    assert "route" in assistant.lower()


def test_cot_rows_for_a1_b1_use_full_candidate_split(tmp_path: Path) -> None:
    a1_packet = oracle_cot.build_fact_packet(base_row(tmp_path, "a1"))
    a1_row = oracle_cot.build_cot_sft_row(a1_packet, oracle_cot.deterministic_oracle_parsed(a1_packet))
    a1_text = a1_row["messages"][1]["content"]
    assert "visible candidate IDs" in a1_text
    assert "[1, 2, 3]" in a1_text
    assert "walkable" in a1_text
    assert "blocked IDs" in a1_text

    b1_packet = oracle_cot.build_fact_packet(base_row(tmp_path, "b1"))
    b1_row = oracle_cot.build_cot_sft_row(b1_packet, oracle_cot.deterministic_oracle_parsed(b1_packet))
    b1_text = b1_row["messages"][1]["content"]
    assert "visible candidate IDs" in b1_text
    assert "[1, 2, 3]" in b1_text
    assert "[1, 3]" in b1_text
    assert "[2]" in b1_text
    assert "embodied-feasible" in b1_text
    assert "too tight" in b1_text


@pytest.mark.parametrize("task", oracle_cot.TASKS)
def test_oracle_cot_sft_assistant_is_natural_language_not_json(tmp_path: Path, task: str) -> None:
    packet = oracle_cot.build_fact_packet(base_row(tmp_path, task))
    parsed = oracle_cot.deterministic_oracle_parsed(packet)
    cot = oracle_cot.build_cot_sft_row(packet, parsed)
    assistant = cot["messages"][1]["content"]

    assert assistant.startswith("<think>\n")
    assert f"\n</think>\n{json.dumps(packet['required_final_json'], ensure_ascii=True)}" in assistant
    assert not assistant.lstrip().startswith("{")
    assert "question_id" not in assistant
    assert "reasoning_payload" not in assistant
    assert "Final JSON:" not in assistant
    for banned in oracle_cot.BANNED_OUTPUT_PATTERNS:
        assert banned not in assistant.lower()

    if task in {"a2", "b2", "c"}:
        assert "route" in assistant.lower()
        assert "collision-free" in assistant.lower()
        assert "without a collision" in assistant.lower()
        assert json.dumps(packet["required_final_json"], ensure_ascii=True) in assistant
    if task in {"b1", "b2", "c"}:
        assert "robot" in assistant.lower()
    if task == "c":
        assert "request" in assistant
        assert "cue" in assistant


def test_generation_packet_excludes_local_audit_paths(tmp_path: Path) -> None:
    packet = oracle_cot.build_fact_packet(base_row(tmp_path, "c"))
    sanitized = oracle_cot.packet_for_generation(packet)
    assert "image_path" not in sanitized
    assert "system_prompt" not in sanitized
    assert "source_refs" not in sanitized
    assert "audit_refs" not in sanitized
    assert "task_schema" not in sanitized
    assert "reasoning_type" not in sanitized
    assert sanitized["required_final_json"] == [1, 3]
    path_waypoints = sanitized["facts"]["path_facts"]["path_waypoints"]
    assert path_waypoints[0]["display_id"] == 1
    assert "embodied_clearance_margin_m" in path_waypoints[0]
    assert "screen_xy_norm" not in path_waypoints[0]


def test_manifest_recovers_generation_audit_from_raw_responses(tmp_path: Path) -> None:
    output_dir = tmp_path / "oracle"
    raw_path = output_dir / "raw_responses" / "a2" / "shard_00000.jsonl"
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text(
        json.dumps(
            {
                "question_id": "q_a2",
                "task": "a2",
                "model": "gpt-5.5",
                "api_base": "http://43.153.119.62:8317/v1",
                "request_mode": "batch",
            }
        )
        + "\n"
    )
    args = SimpleNamespace(
        release_root="release",
        split="train",
        stage="full",
        tasks="a2",
        seed=1,
        shard_size=500,
        generate=False,
        model="unused",
        temperature=0.0,
        top_p=1.0,
        max_completion_tokens=1800,
        api_key_env="OPENAI_API_KEY",
        api_base_env="OPENAI_BASE_URL",
        api_base=None,
        allow_external_upload=False,
    )

    manifest = oracle_cot.build_manifest(args, output_dir, [], {"data_dir": "data"})

    assert manifest["model"] == "gpt-5.5"
    assert manifest["api_base"] == "http://43.153.119.62:8317/v1"
    assert manifest["raw_response_audit"]["raw_response_rows"] == 1
    assert manifest["raw_response_audit"]["request_modes"] == {"batch": 1}
    assert "question text" in manifest["request_config"]["sent_fact_packet_includes"]
    assert "image_path" in manifest["request_config"]["sent_fact_packet_excludes"]


def test_generation_prompt_includes_task_payload_template(tmp_path: Path) -> None:
    a1_packet = oracle_cot.build_fact_packet(base_row(tmp_path, "a1"))
    a1_prompt = json.loads(oracle_cot.user_generation_prompt(a1_packet))
    a1_payload = a1_prompt["required_output_envelope"]["reasoning_payload"]
    assert "candidate_space" in a1_payload
    assert a1_payload["candidate_space"]["num_candidates"] == 3
    assert isinstance(a1_payload["candidate_space"]["positive_examples"], list)
    assert a1_payload["selection_rule"]
    assert "Do not replace reasoning_payload" in "\n".join(a1_prompt["hard_constraints"])

    b1_packet = oracle_cot.build_fact_packet(base_row(tmp_path, "b1"))
    b1_prompt = json.loads(oracle_cot.user_generation_prompt(b1_packet))
    b1_payload = b1_prompt["required_output_envelope"]["reasoning_payload"]
    assert b1_payload["robot_constraint"]["robot_diameter_m"] == 0.6
    assert "candidate_space" in b1_payload
    assert b1_payload["selection_rule"]

    b2_packet = oracle_cot.build_fact_packet(base_row(tmp_path, "b2"))
    b2_prompt = json.loads(oracle_cot.user_generation_prompt(b2_packet))
    b2_payload = b2_prompt["required_output_envelope"]["reasoning_payload"]
    assert b2_payload["target_grounding"]["goal_display_id"] == 3
    assert b2_payload["robot_constraint"]["robot_diameter_m"] == 0.6
    assert b2_payload["path_facts"]["shortest_path"] == [1, 3]
    assert "embodiment_explanation" in b2_payload


def test_batch_generation_prompt_and_parser_require_all_records(tmp_path: Path) -> None:
    packets = [oracle_cot.build_fact_packet(base_row(tmp_path, task)) for task in ("a1", "b2")]
    prompt = json.loads(oracle_cot.batch_generation_prompt(packets))

    assert "records" in prompt["required_output"]
    assert len(prompt["items"]) == 2
    assert prompt["items"][0]["required_output_envelope"]["question_id"] == "q_a1"
    assert any("top-level records array" in item for item in prompt["hard_constraints"])

    text = json.dumps(
        {
            "records": [
                {
                    "question_id": "q_a1",
                    "task": "a1",
                    "reasoning_type": "pointmass_traversability",
                    "reasoning_payload": {"candidate_space": {}, "selection_rule": "x"},
                    "final_json": [1, 2, 3],
                },
                {
                    "question_id": "q_b2",
                    "task": "b2",
                    "reasoning_type": "embodied_explicit_shortest_path",
                    "reasoning_payload": {"path_facts": {"shortest_path": [1, 3]}},
                    "final_json": [1, 3],
                },
            ]
        }
    )
    parsed = oracle_cot.parse_batch_records(text, ["q_a1", "q_b2"])
    assert sorted(parsed) == ["q_a1", "q_b2"]

    with pytest.raises(oracle_cot.OracleCotError, match="missing records array"):
        oracle_cot.parse_batch_records("{}", ["q_a1"])

    with pytest.raises(oracle_cot.OracleCotError, match="length mismatch"):
        oracle_cot.parse_batch_records(text, ["q_a1"])


def test_generate_requires_explicit_external_upload_ack() -> None:
    args = SimpleNamespace(
        generate=True,
        allow_external_upload=False,
        api_base="https://example.invalid/v1",
        api_base_env="OPENAI_BASE_URL",
    )
    with pytest.raises(oracle_cot.OracleCotError, match="refusing to send"):
        oracle_cot.require_external_upload_ack(args)

    args.allow_external_upload = True
    assert oracle_cot.require_external_upload_ack(args) is None


def test_full_external_generation_uses_conservative_retry_backoff() -> None:
    args = SimpleNamespace(
        stage="full",
        generate=True,
        fail_fast_on_api_error=True,
        max_retries=4,
        retry_sleep_s=70.0,
        retry_jitter_s=0.0,
    )

    assert [oracle_cot.retry_sleep_seconds(args, attempt) for attempt in range(4)] == [
        70.0,
        140.0,
        280.0,
        560.0,
    ]
    assert oracle_cot.final_api_error_cooldown_seconds(args) == 600.0
    assert oracle_cot.request_max_retries(args) == 1
    args.defer_max_retries = 2
    assert oracle_cot.request_max_retries(args) == 2

    args.stage = "validation"
    assert oracle_cot.retry_sleep_seconds(args, 3) == 280.0
    assert oracle_cot.final_api_error_cooldown_seconds(args) == 0.0
    assert oracle_cot.request_max_retries(args) == 4


def test_full_external_generation_request_gate_spaces_requests(monkeypatch, tmp_path: Path) -> None:
    now = {"value": 100.0}
    sleeps: list[float] = []

    def fake_time() -> float:
        return now["value"]

    def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now["value"] += seconds

    monkeypatch.setattr(oracle_cot.time, "time", fake_time)
    monkeypatch.setattr(oracle_cot.time, "sleep", fake_sleep)
    args = SimpleNamespace(
        stage="full",
        generate=True,
        fail_fast_on_api_error=True,
        output_dir=str(tmp_path),
        full_generation_request_interval_s=8.0,
    )

    oracle_cot.wait_for_full_generation_request_slot(args, shard_label="0/1")
    oracle_cot.wait_for_full_generation_request_slot(args, shard_label="0/1")

    assert sleeps == [8.0]
    gate_path = tmp_path / "full_generation_request_gate.lock"
    assert float(gate_path.read_text()) == 108.0


def test_full_external_generation_inflight_lock_serializes_requests(tmp_path: Path) -> None:
    args = SimpleNamespace(
        stage="full",
        generate=True,
        fail_fast_on_api_error=True,
        output_dir=str(tmp_path),
        full_generation_max_inflight=1,
    )
    lock_path = tmp_path / "full_generation_inflight.lock"

    with oracle_cot.full_generation_inflight_slot(args, shard_label="0/1"):
        assert lock_path.exists()
        with lock_path.open("a+") as f:
            with pytest.raises(BlockingIOError):
                oracle_cot.fcntl.flock(
                    f.fileno(),
                    oracle_cot.fcntl.LOCK_EX | oracle_cot.fcntl.LOCK_NB,
                )

    with lock_path.open("a+") as f:
        oracle_cot.fcntl.flock(f.fileno(), oracle_cot.fcntl.LOCK_EX | oracle_cot.fcntl.LOCK_NB)
        oracle_cot.fcntl.flock(f.fileno(), oracle_cot.fcntl.LOCK_UN)


def test_full_external_generation_inflight_lock_releases_after_exception(tmp_path: Path) -> None:
    args = SimpleNamespace(
        stage="full",
        generate=True,
        fail_fast_on_api_error=True,
        output_dir=str(tmp_path),
        full_generation_max_inflight=1,
    )
    lock_path = tmp_path / "full_generation_inflight.lock"

    with pytest.raises(RuntimeError):
        with oracle_cot.full_generation_inflight_slot(args, shard_label="0/1"):
            raise RuntimeError("boom")

    with lock_path.open("a+") as f:
        oracle_cot.fcntl.flock(f.fileno(), oracle_cot.fcntl.LOCK_EX | oracle_cot.fcntl.LOCK_NB)
        oracle_cot.fcntl.flock(f.fileno(), oracle_cot.fcntl.LOCK_UN)


def test_fail_fast_on_api_error_does_not_write_rejected(tmp_path: Path) -> None:
    packet = oracle_cot.build_fact_packet(base_row(tmp_path, "a2"))
    item = {"packet": packet, "qid": packet["question_id"], "task": packet["task"], "task_index": 0}
    args = SimpleNamespace(shard_size=500, fail_fast_on_api_error=True)

    with pytest.raises(oracle_cot.OracleCotError, match="generation failed"):
        oracle_cot.write_generated_record(
            output_dir=tmp_path / "oracle",
            args=args,
            item=item,
            api_base="https://example.invalid/v1",
            raw_record=None,
            parsed=None,
            error_message="HTTP Error 502: Bad Gateway",
        )

    assert not list((tmp_path / "oracle" / "rejected").glob("*/*.jsonl"))


def test_full_generation_defer_transient_api_error(tmp_path: Path) -> None:
    packet = oracle_cot.build_fact_packet(base_row(tmp_path, "a2"))
    item = {"packet": packet, "qid": packet["question_id"], "task": packet["task"], "task_index": 0}
    args = SimpleNamespace(
        shard_size=500,
        stage="full",
        generate=True,
        fail_fast_on_api_error=True,
        defer_transient_api_errors=True,
    )

    outcome = oracle_cot.write_generated_record(
        output_dir=tmp_path / "oracle",
        args=args,
        item=item,
        api_base="https://example.invalid/v1",
        raw_record=None,
        parsed=None,
        error_message="HTTP Error 502: Bad Gateway",
    )

    assert outcome is None
    assert not list((tmp_path / "oracle" / "rejected").glob("*/*.jsonl"))


def test_full_generation_deferred_errors_force_resume_exit() -> None:
    args = SimpleNamespace(
        stage="full",
        generate=True,
        fail_fast_on_api_error=True,
        defer_transient_api_errors=True,
    )

    with pytest.raises(oracle_cot.OracleCotError, match="deferred transient API errors"):
        oracle_cot.raise_if_deferred_api_errors(args, deferred_api_errors=2, shard_label="0/1")

    oracle_cot.raise_if_deferred_api_errors(args, deferred_api_errors=0, shard_label="0/1")
    args.stage = "validation"
    oracle_cot.raise_if_deferred_api_errors(args, deferred_api_errors=2, shard_label="0/1")


def test_full_generation_defer_can_be_disabled(monkeypatch, tmp_path: Path) -> None:
    packet = oracle_cot.build_fact_packet(base_row(tmp_path, "a2"))
    item = {"packet": packet, "qid": packet["question_id"], "task": packet["task"], "task_index": 0}
    args = SimpleNamespace(
        shard_size=500,
        stage="full",
        generate=True,
        fail_fast_on_api_error=True,
        defer_transient_api_errors=False,
        retry_sleep_s=1.0,
        retry_jitter_s=0.0,
    )
    sleeps: list[float] = []
    monkeypatch.setattr(oracle_cot.time, "sleep", lambda seconds: sleeps.append(seconds))

    with pytest.raises(oracle_cot.OracleCotError, match="generation failed"):
        oracle_cot.write_generated_record(
            output_dir=tmp_path / "oracle",
            args=args,
            item=item,
            api_base="https://example.invalid/v1",
            raw_record=None,
            parsed=None,
            error_message="HTTP Error 502: Bad Gateway",
        )

    assert not list((tmp_path / "oracle" / "rejected").glob("*/*.jsonl"))
    assert sleeps == [600.0]


def test_batch_generation_retries_remote_disconnect(monkeypatch, tmp_path: Path) -> None:
    packet = oracle_cot.build_fact_packet(base_row(tmp_path, "a2"))
    calls = {"count": 0}

    def fake_request_for_prompt(**_: object) -> dict:
        calls["count"] += 1
        if calls["count"] == 1:
            raise http.client.RemoteDisconnected("Remote end closed connection without response")
        return {
            "id": "ok",
            "usage": {"total_tokens": 1},
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "records": [
                                    {
                                        "question_id": "q_a2",
                                        "task": "a2",
                                        "reasoning_type": "pointmass_explicit_shortest_path",
                                        "reasoning_payload": {
                                            "target_grounding": {
                                                "target_rule": "nearest",
                                                "target_category": "chair",
                                                "goal_display_id": 3,
                                            },
                                            "path_facts": {
                                                "start_id": 1,
                                                "shortest_path": [1, 3],
                                                "path_length_m": 1.25,
                                                "route_semantics": "pointmass",
                                            },
                                            "shortest_path_explanation": "Supplied shortest legal route.",
                                        },
                                        "final_json": [1, 3],
                                    }
                                ]
                            }
                        )
                    }
                }
            ],
        }

    monkeypatch.setattr(oracle_cot, "chat_completion_request_for_prompt", fake_request_for_prompt)
    monkeypatch.setattr(oracle_cot.time, "sleep", lambda _: None)
    args = SimpleNamespace(
        batch_max_completion_tokens=None,
        max_completion_tokens=1800,
        timeout_s=120,
        max_retries=1,
        retry_sleep_s=0,
        retry_jitter_s=0,
        model="gpt-5.5",
        temperature=0.0,
        top_p=1.0,
    )

    raw, parsed, error = oracle_cot.request_batch_generated(
        args=args,
        api_base="http://example.test/v1",
        api_key="sk-test",
        items=[{"packet": packet, "qid": "q_a2", "task": "a2"}],
        shard_label="0/1",
    )

    assert error is None
    assert calls["count"] == 2
    assert raw is not None
    assert parsed is not None
    assert parsed["q_a2"]["final_json"] == [1, 3]


def test_partial_generation_refuses_default_current_sft_export() -> None:
    args = SimpleNamespace(
        generate=True,
        stage="pilot",
        allow_partial_current_sft_export=False,
    )
    with pytest.raises(oracle_cot.OracleCotError, match="partial pilot/validation SFT data"):
        oracle_cot.require_safe_sft_export_target(args, oracle_cot.DEFAULT_LF_DATA_DIR)

    args.allow_partial_current_sft_export = True
    assert oracle_cot.require_safe_sft_export_target(args, oracle_cot.DEFAULT_LF_DATA_DIR) is None

    args.allow_partial_current_sft_export = False
    assert oracle_cot.require_safe_sft_export_target(args, Path("results/oracle_cot_pilot_lf_data")) is None


def test_write_sft_data_exports_stratified_sanity_and_gate(tmp_path: Path) -> None:
    packets = []
    for task in oracle_cot.TASKS:
        for index in range(120):
            row = base_row(tmp_path, task)
            row["question_id"] = f"q_{task}_{index:03d}"
            packets.append(oracle_cot.build_fact_packet(row))

    report = oracle_cot.write_sft_data(
        packets=packets,
        output_dir=tmp_path / "oracle",
        data_dir=tmp_path / "lf_data",
        smoke_count=2,
    )
    assert report["direct_rows"] == 600
    assert report["cot_rows"] == 600
    assert report["direct_sanity500_rows"] == 500
    assert report["cot_sanity500_rows"] == 500
    assert report["direct_gate10pct_rows"] == 60
    assert report["cot_gate10pct_rows"] == 60
    assert report["cot_source"] == "deterministic_fact_packets"

    sanity_rows = json.loads((tmp_path / "lf_data" / "navbench_direct_sft_sanity500.json").read_text())
    sanity_counts = {}
    for row in sanity_rows:
        task = row["metadata"]["task"]
        sanity_counts[task] = sanity_counts.get(task, 0) + 1
    assert sanity_counts == {task: 100 for task in oracle_cot.TASKS}

    dataset_info = json.loads((tmp_path / "lf_data" / "dataset_info.json").read_text())
    assert "navbench_direct_sft_gate10pct" in dataset_info
    assert "navbench_oracle_cot_sft_sanity500" in dataset_info

    cot_rows = json.loads((tmp_path / "lf_data" / "navbench_oracle_cot_sft.json").read_text())
    cot_text = cot_rows[0]["messages"][1]["content"]
    assert cot_text.startswith("<think>\n")
    assert "\n</think>\n" in cot_text
    assert "Final JSON:" not in cot_text
    assert "reasoning_payload" not in cot_text
