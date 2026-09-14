from __future__ import annotations

import hashlib
import json
from pathlib import Path


SCENE_NAV_FACTS_FILENAME = "scene_nav_facts_v1.json"
SCENE_NAV_FACTS_SCHEMA = "scene_nav_facts_v1"
VIEW_BUNDLE_FILENAME = "next_view_bundle_v3.json"
VIEW_BUNDLE_SCHEMA = "next_view_bundle_v3"


def _stable_json_dumps(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def compute_facts_hash(payload: dict) -> str:
    return hashlib.sha256(_stable_json_dumps(payload).encode("utf-8")).hexdigest()


def _classification_candidate_id(view_id: int, index: int) -> str:
    return f"classification_v{int(view_id)}_c{int(index):03d}"


def _routing_candidate_id(view_id: int, index: int, routing: dict) -> str:
    routing_id = routing.get("routing_id")
    if routing_id:
        return str(routing_id)
    return f"routing_v{int(view_id)}_c{int(index):03d}"


def _normalize_classification_candidates(view_id: int, candidates: list[dict]) -> tuple[list[dict], dict[int, str]]:
    normalized: list[dict] = []
    waypoint_to_candidate_id: dict[int, str] = {}
    for idx, candidate in enumerate(candidates or [], start=1):
        normalized_candidate = dict(candidate)
        candidate_id = str(normalized_candidate.get("candidate_id") or _classification_candidate_id(view_id, idx))
        normalized_candidate["candidate_id"] = candidate_id
        normalized.append(normalized_candidate)
        waypoint_id = int(normalized_candidate.get("waypoint_id", -1))
        if waypoint_id >= 0:
            waypoint_to_candidate_id[waypoint_id] = candidate_id
    return normalized, waypoint_to_candidate_id


def _map_waypoint_ids_to_candidate_ids(waypoint_ids: list[int], waypoint_to_candidate_id: dict[int, str]) -> list[str]:
    candidate_ids: list[str] = []
    for waypoint_id in waypoint_ids or []:
        try:
            mapped = waypoint_to_candidate_id[int(waypoint_id)]
        except Exception:
            continue
        candidate_ids.append(mapped)
    return candidate_ids


def _normalize_routing_candidates(
    view_id: int,
    routing_candidates: list[dict],
    waypoint_to_candidate_id: dict[int, str],
) -> list[dict]:
    normalized: list[dict] = []
    for idx, routing in enumerate(routing_candidates or [], start=1):
        normalized_routing = dict(routing)
        normalized_routing["routing_id"] = _routing_candidate_id(view_id, idx, normalized_routing)
        gt_waypoint_ids_pointmass = [
            int(x) for x in normalized_routing.get("gt_path_waypoint_ids_pointmass", []) if int(x) >= 0
        ]
        gt_waypoint_ids_embodied = [
            int(x) for x in normalized_routing.get("gt_path_waypoint_ids_embodied", []) if int(x) >= 0
        ]
        normalized_routing["gt_path_candidate_ids_pointmass"] = _map_waypoint_ids_to_candidate_ids(
            gt_waypoint_ids_pointmass,
            waypoint_to_candidate_id,
        )
        normalized_routing["gt_path_candidate_ids_embodied"] = _map_waypoint_ids_to_candidate_ids(
            gt_waypoint_ids_embodied,
            waypoint_to_candidate_id,
        )
        start_anchor = normalized_routing.get("start_anchor") or {}
        goal_anchor = normalized_routing.get("goal_anchor") or {}
        start_waypoint_id = int(start_anchor.get("waypoint_id", -1))
        goal_waypoint_id = int(goal_anchor.get("waypoint_id", -1))
        normalized_routing["start_candidate_id"] = waypoint_to_candidate_id.get(start_waypoint_id)
        normalized_routing["goal_candidate_id"] = waypoint_to_candidate_id.get(goal_waypoint_id)
        normalized.append(normalized_routing)
    return normalized


def build_scene_nav_facts(scene_id: str, views: list[dict]) -> dict:
    normalized_views: list[dict] = []
    for raw_view in sorted(views or [], key=lambda item: int(item.get("view_id", 0))):
        view_id = int(raw_view.get("view_id", 0))
        classification_candidates, waypoint_to_candidate_id = _normalize_classification_candidates(
            view_id,
            list(raw_view.get("classification_candidates", [])),
        )
        routing_candidates = _normalize_routing_candidates(
            view_id,
            list(raw_view.get("routing_candidates", [])),
            waypoint_to_candidate_id,
        )
        normalized_views.append({
            "view_id": view_id,
            "camera": dict(raw_view.get("camera", {})),
            "visible_objects": list(raw_view.get("visible_objects", [])),
            "affordance_validity": dict(raw_view.get("affordance_validity", {})),
            "affordance_axes": dict(raw_view.get("affordance_axes", {})),
            "affordance_tier": raw_view.get("affordance_tier"),
            "affordance_complexity": dict(raw_view.get("affordance_complexity", {})),
            "classification_candidates": classification_candidates,
            "routing_candidates": routing_candidates,
            "rejection_reasons": list(raw_view.get("rejection_reasons", [])),
        })

    facts = {
        "schema_version": SCENE_NAV_FACTS_SCHEMA,
        "scene_id": str(scene_id),
        "views": normalized_views,
    }
    facts["facts_hash"] = compute_facts_hash(facts)
    return facts


def save_scene_nav_facts(scene_out: Path, facts: dict) -> Path:
    out_path = Path(scene_out) / SCENE_NAV_FACTS_FILENAME
    with open(out_path, "w") as f:
        json.dump(facts, f, indent=2)
    return out_path


def load_scene_nav_facts(scene_out: Path | str) -> dict:
    path = Path(scene_out)
    if path.is_dir():
        path = path / SCENE_NAV_FACTS_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"missing required {SCENE_NAV_FACTS_FILENAME}: {path}")
    with open(path) as f:
        facts = json.load(f)
    if facts.get("schema_version") != SCENE_NAV_FACTS_SCHEMA:
        raise ValueError(
            f"{SCENE_NAV_FACTS_FILENAME} must declare schema_version={SCENE_NAV_FACTS_SCHEMA}"
        )
    if not facts.get("facts_hash"):
        payload = dict(facts)
        facts_hash = compute_facts_hash({k: v for k, v in payload.items() if k != "facts_hash"})
        facts["facts_hash"] = facts_hash
    return facts


def get_view_facts(scene_facts: dict, view_id: int) -> dict:
    for view in scene_facts.get("views", []):
        if int(view.get("view_id", -1)) == int(view_id):
            return view
    raise KeyError(f"view_id={view_id} not found in {SCENE_NAV_FACTS_FILENAME}")


def build_view_bundle_from_scene_nav_facts(scene_facts: dict, view_id: int) -> dict:
    view = get_view_facts(scene_facts, view_id)
    return {
        "schema_version": VIEW_BUNDLE_SCHEMA,
        "facts_schema": SCENE_NAV_FACTS_SCHEMA,
        "facts_hash": scene_facts.get("facts_hash"),
        "scene_id": scene_facts.get("scene_id"),
        "view_id": int(view.get("view_id", 0)),
        "camera": dict(view.get("camera", {})),
        "visible_objects": list(view.get("visible_objects", [])),
        "affordance_validity": dict(view.get("affordance_validity", {})),
        "affordance_axes": dict(view.get("affordance_axes", {})),
        "affordance_tier": view.get("affordance_tier"),
        "affordance_complexity": dict(view.get("affordance_complexity", {})),
        "classification_candidates": list(view.get("classification_candidates", [])),
        "routing_candidates": list(view.get("routing_candidates", [])),
        "rejection_reasons": list(view.get("rejection_reasons", [])),
    }
