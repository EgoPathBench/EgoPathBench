#!/usr/bin/env python3
"""
Project scene-level trusted relations into the current H(view).

Release v1 is intentionally conservative:
- directional relations are disabled
- only supported, visible, anchor-promptable relations survive
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
CUE_REPAIR_PROTOCOL_PATH = ROOT_DIR / "configs" / "cue_repair_protocol_v1.json"


def _load_json(path: Path) -> dict:
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _normalized(text: object) -> str | None:
    if text is None:
        return None
    value = str(text).strip().lower()
    return value or None


def _compute_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def get_cue_repair_protocol(path: Path = CUE_REPAIR_PROTOCOL_PATH) -> dict:
    payload = _load_json(path)
    if payload.get("schema_version") != "cue_repair_protocol_v1":
        raise ValueError(f"unexpected cue repair protocol schema_version in {path}")
    return payload


def _bundle_visible_id_set(bundle: dict) -> set[int]:
    visible_ids: set[int] = set()
    for obj in list(bundle.get("visible_objects") or []):
        object_id = obj.get("id")
        if object_id is None:
            continue
        visible_ids.add(int(object_id))
    return visible_ids


def _target_supported_under_hview(hview: dict, object_id: int) -> bool:
    raw = (hview.get("target_supported_under_hview") or {}).get(str(object_id))
    return bool(raw)


def _anchor_promptable_under_hview(hview: dict, anchor_id: int) -> bool:
    if not _target_supported_under_hview(hview, anchor_id):
        return False
    confusers = list((hview.get("anchor_confusable_ids") or {}).get(str(anchor_id), []))
    return len(confusers) == 0


def _relation_margin(relation: dict) -> float:
    try:
        return float(relation.get("relation_margin", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _project_one_relation(*, relation: dict, hview: dict, bundle_visible_ids: set[int], cue_protocol: dict) -> dict:
    target_id = int(relation["target_id"])
    anchor_ids = [int(v) for v in list(relation.get("anchor_ids") or [])]
    relation_type = str(relation.get("relation_type") or "")
    disabled_relation_types = {
        _normalized(value)
        for value in list(cue_protocol.get("disabled_relation_types") or [])
        if _normalized(value) is not None
    }
    relation_type_norm = _normalized(relation_type)

    projection_status = "usable"
    rejected_projection_reason = None
    view_consistency_status = "stable"

    target_supported = _target_supported_under_hview(hview, target_id)
    if not target_supported:
        projection_status = "rejected"
        rejected_projection_reason = "target_not_supported_under_hview"
        view_consistency_status = "target_not_supported_under_hview"

    anchor_visible_and_supported = True
    anchor_promptable_under_hview = True
    for anchor_id in anchor_ids:
        if anchor_id not in bundle_visible_ids or not _target_supported_under_hview(hview, anchor_id):
            anchor_visible_and_supported = False
            anchor_promptable_under_hview = False
            if rejected_projection_reason is None:
                projection_status = "rejected"
                rejected_projection_reason = "anchor_not_supported_under_hview"
                view_consistency_status = "anchor_not_supported_under_hview"
            break
        if not _anchor_promptable_under_hview(hview, anchor_id):
            anchor_promptable_under_hview = False
            if rejected_projection_reason is None:
                projection_status = "rejected"
                rejected_projection_reason = "anchor_not_promptable_under_hview"
                view_consistency_status = "anchor_not_promptable_under_hview"

    if relation_type_norm in disabled_relation_types:
        projection_status = "rejected"
        rejected_projection_reason = "directional_relation_disabled_in_v1"
        view_consistency_status = "directional_relation_disabled_in_v1"
        anchor_promptable_under_hview = False if not anchor_visible_and_supported else anchor_promptable_under_hview

    return {
        "relation_candidate_id": relation.get("relation_candidate_id"),
        "target_id": target_id,
        "anchor_ids": anchor_ids,
        "anchor_categories": list(
            relation.get("anchor_categories")
            or ((relation.get("scene_relation_evidence") or {}).get("anchor_categories") or [])
        ),
        "relation_type": relation_type,
        "relation_frame": relation.get("relation_frame"),
        "projection_status": projection_status,
        "anchor_visible_and_supported": bool(anchor_visible_and_supported),
        "anchor_promptable_under_hview": bool(anchor_promptable_under_hview),
        "relation_stability_margin": _relation_margin(relation),
        "view_consistency_status": view_consistency_status,
        "rejected_projection_reason": rejected_projection_reason,
    }


def build_view_relation_projection(
    *,
    scene_id: str,
    view_id: int,
    scene_relation_graph: dict,
    hview_confusable_universe: dict,
    bundle: dict,
) -> dict:
    cue_protocol = get_cue_repair_protocol()
    bundle_visible_ids = _bundle_visible_id_set(bundle)
    projected_relations = [
        _project_one_relation(
            relation=relation,
            hview=hview_confusable_universe,
            bundle_visible_ids=bundle_visible_ids,
            cue_protocol=cue_protocol,
        )
        for relation in list(scene_relation_graph.get("relations") or [])
    ]
    payload = {
        "schema_version": "view_relation_projection_v1",
        "scene_id": scene_id,
        "view_id": int(view_id),
        "projected_relations": projected_relations,
        "cue_repair_protocol_version": cue_protocol["schema_version"],
        "cue_repair_protocol_hash": _compute_hash(cue_protocol),
    }
    payload["view_relation_projection_hash"] = _compute_hash(payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build projected scene relation sidecar for a single view.")
    parser.add_argument("--scene-relation-graph", required=True)
    parser.add_argument("--hview-confusable-universe", required=True)
    parser.add_argument("--view-bundle", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    scene_relation_graph = _load_json(Path(args.scene_relation_graph))
    hview_confusable_universe = _load_json(Path(args.hview_confusable_universe))
    bundle = _load_json(Path(args.view_bundle))

    scene_id = str(scene_relation_graph.get("scene_id") or hview_confusable_universe.get("scene_id"))
    view_id = int(hview_confusable_universe.get("view_id"))
    projection = build_view_relation_projection(
        scene_id=scene_id,
        view_id=view_id,
        scene_relation_graph=scene_relation_graph,
        hview_confusable_universe=hview_confusable_universe,
        bundle=bundle,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(projection, f, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
