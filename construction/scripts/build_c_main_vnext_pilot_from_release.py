#!/usr/bin/env python3
"""
Build a config-driven C-main-vNext pilot from an existing stable release.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


SPLITS = ("train", "val", "benchmark")

BASE_ONTOLOGY = {
    "chair": "sit_rest",
    "couch": "sit_rest",
    "stool": "sit_rest",
    "sink": "wash_clean",
    "basin": "wash_clean",
    "bin": "discard",
    "cabinet": "store_put_away",
    "table": "place_object",
    "desk": "place_object",
    "counter": "place_object",
}

BASE_INTENT_TEMPLATES = {
    "sit_rest": [
        "I want to sit down.",
        "I need a place to sit.",
        "I want to take a seat.",
    ],
    "wash_clean": [
        "I want to wash my hands.",
        "I need to rinse my hands.",
        "I want to wash up a bit.",
    ],
    "discard": [
        "I want to throw something away.",
        "I need to toss this out.",
        "I need somewhere to throw this away.",
    ],
    "store_put_away": [
        "I want to put this away.",
        "I need to store this.",
        "I need somewhere to put this away.",
    ],
    "place_object": [
        "I want to set this down.",
        "I need somewhere to place this.",
        "I want to put this down.",
    ],
}

DEFAULT_ALLOWED_CUES = ("none", "distance", "side", "ordinal")

CUE_SUFFIXES = {
    "nearer": "Go to the nearer one.",
    "farther": "Go to the farther one.",
    "closest": "Go to the closest one.",
    "farthest": "Go to the farthest one.",
    "left": "Go to the left one.",
    "right": "Go to the right one.",
    "first": "Go to the first one.",
    "second": "Go to the second one.",
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def build_routing_index(rows: list[dict]) -> dict[tuple[str, int, str], dict]:
    index: dict[tuple[str, int, str], dict] = {}
    for row in rows:
        key = (str(row["scene_id"]), int(row["view_id"]), str(row["routing_id"]))
        index[key] = row
    return index


def stable_index(text: str, size: int) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % size


def load_ontology_config(path: Path) -> dict:
    payload = json.loads(path.read_text())
    category_to_family = dict(BASE_ONTOLOGY)
    family_templates = {key: list(value) for key, value in BASE_INTENT_TEMPLATES.items()}
    family_allowed_cues = {key: tuple(DEFAULT_ALLOWED_CUES) for key in family_templates}
    family_meta: dict[str, dict] = {}

    for family, spec in (payload.get("families") or {}).items():
        status = str(spec.get("status", "recommended_for_pilot"))
        if not status.startswith("recommended"):
            continue
        allowed_categories = [str(cat).strip().lower() for cat in spec.get("allowed_categories", []) if str(cat).strip()]
        templates = [str(text).strip() for text in spec.get("question_templates", []) if str(text).strip()]
        residual_cues = [str(cue).strip() for cue in spec.get("residual_cues_allowed", []) if str(cue).strip()]
        if not allowed_categories or not templates:
            continue
        for category in allowed_categories:
            category_to_family[category] = family
        family_templates[family] = templates
        family_allowed_cues[family] = tuple(residual_cues or DEFAULT_ALLOWED_CUES)
        family_meta[family] = {
            "status": status,
            "allowed_categories": allowed_categories,
            "forbidden_categories": [str(cat).strip().lower() for cat in spec.get("forbidden_categories", []) if str(cat).strip()],
            "intent_labels": [str(label).strip() for label in spec.get("intent_labels", []) if str(label).strip()],
            "target_selection_policy": str(spec.get("target_selection_policy", "closed_family_single_target")),
            "tie_rule": str(spec.get("tie_rule", "")),
        }

    return {
        "payload": payload,
        "category_to_family": category_to_family,
        "family_templates": family_templates,
        "family_allowed_cues": family_allowed_cues,
        "family_meta": family_meta,
    }


def family_for_category(category: str | None, category_to_family: dict[str, str]) -> str | None:
    if category is None:
        return None
    return category_to_family.get(str(category).strip().lower())


def _candidate_sort_key(candidate: dict) -> tuple[float, float, int]:
    return (
        float(candidate["screen_x"]),
        float(candidate["distance_m"]),
        int(candidate["target_id"]),
    )


def _distance_sort_key(candidate: dict) -> tuple[float, float, int]:
    return (
        float(candidate["distance_m"]),
        float(candidate["screen_x"]),
        int(candidate["target_id"]),
    )


def cue_target(candidates: list[dict], cue_value: str) -> int | None:
    if cue_value in ("nearer", "farther"):
        if len(candidates) != 2:
            return None
        ordered = sorted(candidates, key=_distance_sort_key)
        return int(ordered[0 if cue_value == "nearer" else 1]["target_id"])
    if cue_value in ("closest", "farthest"):
        if not candidates:
            return None
        ordered = sorted(candidates, key=_distance_sort_key)
        return int(ordered[0 if cue_value == "closest" else -1]["target_id"])
    if cue_value in ("left", "right"):
        if not candidates:
            return None
        ordered = sorted(candidates, key=_candidate_sort_key)
        return int(ordered[0 if cue_value == "left" else -1]["target_id"])
    if cue_value in ("first", "second"):
        ordered = sorted(candidates, key=_candidate_sort_key)
        index = 0 if cue_value == "first" else 1
        if len(ordered) <= index:
            return None
        return int(ordered[index]["target_id"])
    return None


def distance_cue_value(target: dict, family_candidates: list[dict]) -> str | None:
    ordered = sorted(family_candidates, key=_distance_sort_key)
    rank = next((idx for idx, cand in enumerate(ordered) if int(cand["target_id"]) == int(target["target_id"])), None)
    if rank is None:
        return None
    if len(ordered) == 2:
        return "nearer" if rank == 0 else "farther"
    if rank == 0:
        return "closest"
    if rank == len(ordered) - 1:
        return "farthest"
    return None


def side_cue_value(target: dict, family_candidates: list[dict]) -> str | None:
    ordered = sorted(family_candidates, key=_candidate_sort_key)
    if not ordered:
        return None
    if int(ordered[0]["target_id"]) == int(target["target_id"]):
        return "left"
    if int(ordered[-1]["target_id"]) == int(target["target_id"]):
        return "right"
    return None


def ordinal_cue_value(target: dict, family_candidates: list[dict]) -> str | None:
    ordered = sorted(family_candidates, key=_candidate_sort_key)
    for idx, candidate in enumerate(ordered[:2]):
        if int(candidate["target_id"]) == int(target["target_id"]):
            return "first" if idx == 0 else "second"
    return None


def select_residual_cue(
    target: dict,
    admissible_candidates: list[dict],
    family_candidates: list[dict],
    *,
    allowed_cue_types: tuple[str, ...],
) -> tuple[str, str | None, str | None]:
    if len(family_candidates) == 1:
        if "none" in allowed_cue_types:
            return "none", None, None
        return "reject", None, "cue_policy_disallows_none"

    saw_explicit_reference = False
    cue_builders = (
        ("distance", distance_cue_value),
        ("side", side_cue_value),
        ("ordinal", ordinal_cue_value),
    )
    for cue_type, builder in cue_builders:
        if cue_type not in allowed_cue_types:
            continue
        cue_value = builder(target, family_candidates)
        if cue_value is None:
            continue
        family_target = cue_target(family_candidates, cue_value)
        if family_target != int(target["target_id"]):
            continue
        full_target = cue_target(admissible_candidates, cue_value)
        if full_target is not None:
            saw_explicit_reference = True
            continue
        return cue_type, cue_value, None

    if saw_explicit_reference:
        return "reject", None, "explicit_reference_dominant"
    return "reject", None, "multi_cue_required"


def build_generated_question(question_id: str, intent_family: str, cue_value: str | None, family_templates: dict[str, list[str]]) -> str:
    templates = family_templates[intent_family]
    base = templates[stable_index(question_id, len(templates))]
    if cue_value is None:
        return base
    return f"{base} {CUE_SUFFIXES[cue_value]}"


def build_user_prompt(start_id: int, generated_question: str) -> str:
    return (
        f"The image shows numbered candidate points. You are the robot, and your body diameter is 0.6 m. "
        f"Start at display ID {int(start_id)}. A person says: \"{generated_question}\" "
        "Return a JSON array of display IDs representing a valid collision-free path to the resolved target."
    )


def load_view_bundle(release_root: Path, scene_id: str, view_id: int) -> dict:
    path = release_root / "renders" / scene_id / f"view_{int(view_id)}" / "next_view_bundle_v3.json"
    with open(path) as f:
        return json.load(f)


def build_u_view(bundle: dict, category_to_family: dict[str, str]) -> list[dict]:
    visible_by_id = {
        int(obj["id"]): obj
        for obj in bundle.get("visible_objects", [])
        if isinstance(obj, dict) and "id" in obj
    }
    display_id_by_candidate = {
        str(candidate["candidate_id"]): int(candidate["display_id"])
        for candidate in bundle.get("classification_candidates", [])
        if isinstance(candidate, dict) and candidate.get("candidate_id") is not None and candidate.get("display_id") is not None
    }
    u_view: list[dict] = []
    for candidate in bundle.get("routing_candidates", []):
        target_id = candidate.get("target_id")
        if target_id is None:
            continue
        target_id = int(target_id)
        visible_object = visible_by_id.get(target_id)
        if not isinstance(visible_object, dict):
            continue
        canonical_category = str(candidate.get("canonical_category") or candidate.get("target_category")).strip().lower()
        intent_family = family_for_category(canonical_category, category_to_family)
        if intent_family is None:
            continue
        screen_x = visible_object.get("screen_x")
        screen_y = visible_object.get("screen_y")
        if not bool(visible_object.get("raycast_verified")):
            continue
        if screen_x is None or screen_y is None:
            continue
        # `screen_x/screen_y` use signed normalized device coordinates in [-1, 1].
        if not (-1.0 <= float(screen_x) <= 1.0 and -1.0 <= float(screen_y) <= 1.0):
            continue
        if candidate.get("goal_candidate_id") is None:
            continue
        if not candidate.get("optimal_path_world"):
            continue
        if not candidate.get("gt_path_waypoint_ids_pointmass"):
            continue
        navigation_validity = candidate.get("navigation_validity") or {}
        invariants = candidate.get("invariants") or {}
        if not bool(navigation_validity.get("eligible")):
            continue
        if not bool(invariants.get("start_in_forward_sector")):
            continue
        if not bool(invariants.get("goal_in_target_zone")):
            continue
        if not bool(invariants.get("embodied_collision_free")):
            continue
        goal_display_id = display_id_by_candidate.get(str(candidate.get("goal_candidate_id")))
        if goal_display_id is None:
            continue
        u_view.append(
            {
                "routing_id": str(candidate.get("routing_id")),
                "target_id": target_id,
                "target_category": str(candidate.get("target_category") or candidate.get("canonical_category")),
                "canonical_category": canonical_category,
                "intent_family": intent_family,
                "distance_m": float(candidate.get("distance_m", 0.0)),
                "screen_x": float(screen_x),
                "screen_y": float(screen_y),
                "goal_display_id": int(goal_display_id),
                "goal_candidate_id": str(candidate.get("goal_candidate_id")),
                "navigation_validity": navigation_validity,
            }
        )
    return sorted(u_view, key=lambda row: (row["routing_id"], row["target_id"]))


def _split_summary() -> dict:
    return {"input": 0, "kept": 0, "rejected": Counter(), "family_counts": Counter()}


def build_c_main_vnext_pilot_from_release(*, release_root: Path, source_release_dir: Path, output_dir: Path, ontology_config_path: Path) -> dict:
    release_root = release_root.resolve()
    source_release_dir = source_release_dir.resolve()
    output_dir = output_dir.resolve()
    ontology_config_path = ontology_config_path.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ontology = load_ontology_config(ontology_config_path)
    category_to_family = ontology["category_to_family"]
    family_templates = ontology["family_templates"]
    family_allowed_cues = ontology["family_allowed_cues"]
    family_meta = ontology["family_meta"]

    summary = {
        "schema_version": "c_main_vnext_pilot_summary_v1",
        "source_release_dir": str(source_release_dir),
        "ontology_config_path": str(ontology_config_path),
        "splits": {},
    }
    sidecars_root = output_dir / "sidecars"
    embodied_rows_by_key = build_routing_index(load_jsonl(release_root / "gt" / "gt_next_b2.jsonl"))

    for split in SPLITS:
        split_summary = _split_summary()
        source_rows = load_jsonl(source_release_dir / split / "gt" / "gt_next_c.jsonl")
        split_summary["input"] = len(source_rows)
        kept_rows: list[dict] = []
        kept_vqa: list[dict] = []
        cached_u_view: dict[tuple[str, int], list[dict]] = {}
        rows_by_view: dict[tuple[str, int], list[dict]] = defaultdict(list)

        for source_row in source_rows:
            scene_id = str(source_row["scene_id"])
            view_id = int(source_row["view_id"])
            key = (scene_id, view_id)
            if key not in cached_u_view:
                cached_u_view[key] = build_u_view(load_view_bundle(release_root, scene_id, view_id), category_to_family)
            u_view = cached_u_view[key]
            embodied_source = embodied_rows_by_key.get((scene_id, view_id, str(source_row["routing_id"])))
            if embodied_source is None:
                split_summary["rejected"]["missing_embodied_source"] += 1
                continue

            target = next((cand for cand in u_view if int(cand["target_id"]) == int(source_row["target_id"])), None)
            if target is None:
                split_summary["rejected"]["not_in_ontology"] += 1
                continue

            family_candidates = [cand for cand in u_view if cand["intent_family"] == target["intent_family"]]
            allowed_cue_types = tuple(family_allowed_cues.get(target["intent_family"], DEFAULT_ALLOWED_CUES))
            cue_type, cue_value, reject_reason = select_residual_cue(
                target,
                u_view,
                family_candidates,
                allowed_cue_types=allowed_cue_types,
            )
            if reject_reason is not None:
                split_summary["rejected"][reject_reason] += 1
                continue

            generated_question = build_generated_question(str(source_row["question_id"]), target["intent_family"], cue_value, family_templates)
            label = "C-S" if cue_type == "none" else "C-H"
            resolved_goal_display_id = int(embodied_source["path_ids"][-1])
            kept = {
                **source_row,
                "start_id": embodied_source["start_id"],
                "goal_id": embodied_source["goal_id"],
                "goal_ids": embodied_source.get("goal_ids"),
                "path_ids": embodied_source["path_ids"],
                "optimal_length_m": embodied_source["optimal_length_m"],
                "canonical_sparse_length_m": embodied_source["canonical_sparse_length_m"],
                "robot_diameter_m": embodied_source.get("robot_diameter_m"),
                "route_source_task": "b2",
                "route_semantics": "embodied",
                "split": split,
                "intent_family": target["intent_family"],
                "intent_strength": "strict" if cue_type == "none" else "hybrid",
                "current_reference_mode": str((source_row.get("navigation_reference_facts") or {}).get("reference_resolution_mode", "category_only")),
                "residual_cue_type": cue_type,
                "generated_question": generated_question,
                "resolved_target_id": int(target["target_id"]),
                "resolved_goal_display_id": resolved_goal_display_id,
                "candidate_inventory_ref": "",
                "label": label,
                "reject_reason": None,
                "derivation_trace": {
                    "stable_source": {
                        "question_id": source_row["question_id"],
                        "split": split,
                        "source": source_row["source"],
                        "scene_id": scene_id,
                        "view_id": view_id,
                        "routing_id": source_row["routing_id"],
                        "embodied_question_id": embodied_source["question_id"],
                    },
                    "candidate_inventory": {
                        "path": "",
                        "universe_size": len(u_view),
                        "family_compatible_size": len(family_candidates),
                        "evaluator_inventory_size": 0,
                    },
                    "ontology_rule": {
                        "target_category": target["canonical_category"],
                        "intent_family": target["intent_family"],
                        "target_selection_policy": family_meta.get(target["intent_family"], {}).get("target_selection_policy", "closed_family_single_target"),
                        "tie_rule": family_meta.get(target["intent_family"], {}).get("tie_rule", ""),
                    },
                    "cue_test": {
                        "current_reference_mode": str((source_row.get("navigation_reference_facts") or {}).get("reference_resolution_mode", "category_only")),
                        "residual_cue_type": cue_type,
                        "cue_resolver_universe": "U(view)",
                        "cue_surface": cue_value,
                        "cue_only_identifiable": False,
                        "allowed_residual_cues": list(allowed_cue_types),
                    },
                    "route_supervision": {
                        "route_source_task": "b2",
                        "route_semantics": "embodied",
                    },
                    "label_decision": {
                        "label": label,
                        "reject_reason": None,
                    },
                    "llm_metadata": {
                        "question_generator_version": "template_vnext_pilot_v1",
                    },
                },
            }
            rows_by_view[key].append(kept)

        final_rows: list[dict] = []
        for key, rows in rows_by_view.items():
            collisions: dict[int, list[dict]] = defaultdict(list)
            for row in rows:
                collisions[int(row["resolved_goal_display_id"])].append(row)
            collided_ids = {goal_id for goal_id, items in collisions.items() if len(items) > 1}
            if collided_ids:
                for row in rows:
                    if int(row["resolved_goal_display_id"]) in collided_ids:
                        split_summary["rejected"]["not_single_target_evaluable"] += 1
                rows = [row for row in rows if int(row["resolved_goal_display_id"]) not in collided_ids]
            final_rows.extend(rows)

        final_rows.sort(key=lambda row: str(row["question_id"]))
        retained_by_view: dict[tuple[str, int], list[dict]] = defaultdict(list)
        for row in final_rows:
            retained_by_view[(str(row["scene_id"]), int(row["view_id"]))].append(row)
            split_summary["family_counts"][row["intent_family"]] += 1

        for key, rows in retained_by_view.items():
            scene_id, view_id = key
            inventory_path = sidecars_root / scene_id / f"view_{view_id}" / "c_main_vnext_inventory.json"
            inventory_path.parent.mkdir(parents=True, exist_ok=True)
            u_view = cached_u_view[key]
            retained_target_ids = {int(row["resolved_target_id"]) for row in rows}
            e_view = [cand for cand in u_view if int(cand["target_id"]) in retained_target_ids]
            inventory = {
                "schema_version": "c_main_vnext_candidate_inventory_v1",
                "scene_id": scene_id,
                "view_id": view_id,
                "u_view": u_view,
                "e_view": e_view,
            }
            inventory_path.write_text(json.dumps(inventory, indent=2))
            for row in rows:
                row["candidate_inventory_ref"] = str(inventory_path)
                row["derivation_trace"]["candidate_inventory"]["path"] = str(inventory_path)
                row["derivation_trace"]["candidate_inventory"]["evaluator_inventory_size"] = len(e_view)
                vqa = {
                    "question_id": row["question_id"],
                    "source": row["source"],
                    "task": row["task"],
                    "tier_family": row.get("tier_family"),
                    "tier": row.get("tier"),
                    "scene_id": row["scene_id"],
                    "view_id": row["view_id"],
                    "routing_id": row["routing_id"],
                    "image_path": row["image_path"],
                    "visible_waypoints_path": row["visible_waypoints_path"],
                    "system_prompt": (
                        "You are an embodied navigation robot. A user gives you a short request that "
                        "implies the kind of object they want. Infer the target family from the request, "
                        "resolve the final grounded target in the scene, and plan a collision-free path "
                        "for your body."
                    ),
                    "user_prompt": build_user_prompt(int(row["start_id"]), row["generated_question"]),
                    "question_text": row["generated_question"],
                    "ground_truth": {
                        "answer": row["path_ids"],
                        "start_id": row["start_id"],
                        "goal_id": row["goal_id"],
                        "goal_ids": row.get("goal_ids"),
                        "path_ids": row["path_ids"],
                        "target_id": row["resolved_target_id"],
                        "resolved_goal_display_id": row["resolved_goal_display_id"],
                        "label": row["label"],
                        "intent_family": row["intent_family"],
                        "intent_strength": row["intent_strength"],
                        "residual_cue_type": row["residual_cue_type"],
                        "route_source_task": "b2",
                        "route_semantics": "embodied",
                        "candidate_inventory_ref": row["candidate_inventory_ref"],
                    },
                }
                kept_vqa.append(vqa)

        kept_rows = final_rows
        kept_vqa.sort(key=lambda row: str(row["question_id"]))

        split_summary["kept"] = len(kept_rows)
        summary["splits"][split] = {
            "input": split_summary["input"],
            "kept": split_summary["kept"],
            "family_counts": dict(sorted(split_summary["family_counts"].items())),
            "rejected": dict(sorted(split_summary["rejected"].items())),
        }

        write_jsonl(output_dir / split / "gt" / "gt_next_c.jsonl", kept_rows)
        write_jsonl(output_dir / split / "vqa" / "vqa_next_c.jsonl", kept_vqa)

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a config-driven C-main-vNext pilot from an existing stable release.")
    parser.add_argument("--release-root", required=True)
    parser.add_argument("--source-release-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ontology-config", required=True)
    args = parser.parse_args()

    summary = build_c_main_vnext_pilot_from_release(
        release_root=Path(args.release_root),
        source_release_dir=Path(args.source_release_dir),
        output_dir=Path(args.output_dir),
        ontology_config_path=Path(args.ontology_config),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
