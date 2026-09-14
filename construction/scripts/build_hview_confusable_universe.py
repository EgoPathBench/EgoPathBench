#!/usr/bin/env python3
"""
Build conservative human-confusable H(view) sidecars.

This builder intentionally fail-closes:
- only `supported` + `visible` objects can witness uniqueness
- audit visible object mismatches raise immediately
- cross-type / anchor confusers are conservative, thresholded, and config-driven
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from h_view_utils import (
    build_h_view,
    get_hview_confusable_universe_hash,
    get_hview_confusable_universe_protocol,
    get_schema_bundle_hash,
    load_proxy_group_map,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROXY_GROUPS_PATH = ROOT_DIR / "configs" / "c_intent_family_proxy_groups_v1.json"
VIEW_BUNDLE_FILENAME = "next_view_bundle_v3.json"


def _normalize(text: object) -> str | None:
    if text is None:
        return None
    value = str(text).strip().lower()
    return value or None


def _parse_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_json(path: Path) -> dict | list:
    with open(path) as f:
        return json.load(f)


def _supported_visible(obj: dict) -> bool:
    return obj.get("support_status") == "supported" and obj.get("visibility_status") == "visible"


def _object_label_set(obj: dict) -> set[str]:
    labels = set()
    for key in ("canonical_category", "category_family", "same_type_key"):
        label = _normalize(obj.get(key))
        if label is not None:
            labels.add(label)
    return labels


def _visible_object_signature_rows(objects: list[dict]) -> list[tuple[int, str | None]]:
    rows = []
    for obj in objects:
        obj_id = obj.get("id")
        if obj_id is None:
            continue
        rows.append(
            (
                int(obj_id),
                _normalize(obj.get("canonical_category")) or _normalize(obj.get("category")),
            )
        )
    return sorted(rows)


def validate_visible_objects_consistency(
    *,
    bundle_visible_objects: list[dict],
    audit_visible_objects: list[dict] | None,
    protocol: dict,
) -> tuple[bool, dict]:
    enabled = bool(
        ((protocol.get("consistency_check") or {}).get("bundle_visible_objects_must_match_audit_visible_objects"))
    )
    if audit_visible_objects is None:
        return True, {"enabled": enabled, "reason": "audit_visible_objects_missing"}
    bundle_rows = _visible_object_signature_rows(bundle_visible_objects)
    audit_rows = _visible_object_signature_rows(audit_visible_objects)
    passed = bundle_rows == audit_rows
    details = {
        "enabled": enabled,
        "bundle_rows": bundle_rows,
        "audit_rows": audit_rows,
    }
    if enabled and not passed:
        raise ValueError("visible_objects consistency check failed between bundle and audit visible_objects.json")
    return passed, details


def _confuser_labels_for_object(obj: dict, mapping: dict[str, list[str]]) -> set[str]:
    labels = _object_label_set(obj)
    confusers: set[str] = set()
    for label in labels:
        confusers.update(_normalize(item) for item in mapping.get(label, []) if _normalize(item) is not None)
    return confusers


def _linf_screen_distance(obj_a: dict, obj_b: dict) -> float | None:
    ax = _parse_float(((obj_a.get("screen_box_or_anchor") or {}).get("x")))
    ay = _parse_float(((obj_a.get("screen_box_or_anchor") or {}).get("y")))
    bx = _parse_float(((obj_b.get("screen_box_or_anchor") or {}).get("x")))
    by = _parse_float(((obj_b.get("screen_box_or_anchor") or {}).get("y")))
    if None in (ax, ay, bx, by):
        return None
    return max(abs(ax - bx), abs(ay - by))


def _depth_delta_m(bundle_visible_by_id: dict[int, dict], object_id_a: int, object_id_b: int) -> float | None:
    a = bundle_visible_by_id.get(object_id_a) or {}
    b = bundle_visible_by_id.get(object_id_b) or {}
    depth_a = _parse_float(a.get("depth"))
    depth_b = _parse_float(b.get("depth"))
    if None in (depth_a, depth_b):
        return None
    return abs(depth_a - depth_b)


def _size_ratio(bundle_visible_by_id: dict[int, dict], object_id_a: int, object_id_b: int) -> float | None:
    a = bundle_visible_by_id.get(object_id_a) or {}
    b = bundle_visible_by_id.get(object_id_b) or {}
    size_a = _parse_float(a.get("size"))
    size_b = _parse_float(b.get("size"))
    if size_a is None or size_b is None or size_a <= 0.0 or size_b <= 0.0:
        return None
    larger = max(size_a, size_b)
    smaller = min(size_a, size_b)
    if smaller <= 0.0:
        return None
    return larger / smaller


def _confusable_under_thresholds(
    *,
    obj_a: dict,
    obj_b: dict,
    bundle_visible_by_id: dict[int, dict],
    thresholds: dict,
) -> bool:
    linf_distance = _linf_screen_distance(obj_a, obj_b)
    if linf_distance is None or linf_distance > float(thresholds["max_screen_linf_distance"]):
        return False
    depth_delta = _depth_delta_m(bundle_visible_by_id, int(obj_a["object_id"]), int(obj_b["object_id"]))
    if depth_delta is not None and depth_delta > float(thresholds["max_depth_delta_m"]):
        return False
    ratio = _size_ratio(bundle_visible_by_id, int(obj_a["object_id"]), int(obj_b["object_id"]))
    if ratio is not None:
        if ratio < float(thresholds["min_size_ratio"]) or ratio > float(thresholds["max_size_ratio"]):
            return False
    return True


def _build_confuser_maps(
    *,
    objects: list[dict],
    bundle_visible_by_id: dict[int, dict],
    cross_type_confuser_map: dict[str, list[str]],
    anchor_confuser_map: dict[str, list[str]],
    thresholds: dict,
) -> tuple[dict[str, list[int]], dict[str, list[int]]]:
    supported_visible = [obj for obj in objects if _supported_visible(obj)]
    cross_type: dict[str, list[int]] = {}
    anchor_confusable: dict[str, list[int]] = {}

    for obj in objects:
        object_id = int(obj["object_id"])
        canonical_category = _normalize(obj.get("canonical_category"))
        cross_labels = _confuser_labels_for_object(obj, cross_type_confuser_map)
        anchor_labels = _confuser_labels_for_object(obj, anchor_confuser_map)
        cross_hits: list[int] = []
        anchor_hits: list[int] = []
        for other in supported_visible:
            other_id = int(other["object_id"])
            if other_id == object_id:
                continue
            other_labels = _object_label_set(other)
            if canonical_category is not None and _normalize(other.get("canonical_category")) == canonical_category:
                continue
            if not _confusable_under_thresholds(
                obj_a=obj,
                obj_b=other,
                bundle_visible_by_id=bundle_visible_by_id,
                thresholds=thresholds,
            ):
                continue
            if cross_labels & other_labels:
                cross_hits.append(other_id)
            if anchor_labels & other_labels:
                anchor_hits.append(other_id)
        cross_type[str(object_id)] = sorted(set(cross_hits))
        anchor_confusable[str(object_id)] = sorted(set(anchor_hits))
    return cross_type, anchor_confusable


def build_hview_confusable_universe(
    *,
    scene_id: str,
    view_id: int,
    bundle: dict,
    layout: list[dict],
    metadata: dict,
    proxy_group_map: dict[str, list[str]],
    audit_visible_objects: list[dict] | None = None,
    audit_assets: dict | None = None,
) -> tuple[dict, dict]:
    del layout
    del metadata
    protocol = get_hview_confusable_universe_protocol()
    same_type_mode = str(protocol.get("default_same_type_mode") or "family_proxy")
    h_view = build_h_view(
        bundle,
        routing_rows=[],
        proxy_group_map=proxy_group_map,
        same_type_mode=same_type_mode,
    )
    bundle_visible_objects = list(bundle.get("visible_objects") or [])
    visible_objects_consistency_pass, consistency_details = validate_visible_objects_consistency(
        bundle_visible_objects=bundle_visible_objects,
        audit_visible_objects=audit_visible_objects,
        protocol=protocol,
    )

    bundle_visible_by_id: dict[int, dict] = {}
    for visible in bundle_visible_objects:
        object_id = visible.get("id")
        if object_id is None:
            continue
        bundle_visible_by_id[int(object_id)] = dict(visible)

    objects = [dict(obj) for obj in h_view["objects"]]
    same_type_visible_ids: dict[str, list[int]] = {}
    target_supported_under_hview: dict[str, bool] = {}
    for obj in objects:
        object_id = int(obj["object_id"])
        same_type_key = _normalize(obj.get("same_type_key"))
        witness_ids = list((h_view.get("effective_uniqueness_witness_same_type_ids") or {}).get(same_type_key or "", []))
        same_type_visible_ids[str(object_id)] = sorted(int(v) for v in witness_ids)
        target_supported_under_hview[str(object_id)] = _supported_visible(obj)

    cross_type_confusable_ids, anchor_confusable_ids = _build_confuser_maps(
        objects=objects,
        bundle_visible_by_id=bundle_visible_by_id,
        cross_type_confuser_map=dict(protocol.get("cross_type_confuser_map") or {}),
        anchor_confuser_map=dict(protocol.get("anchor_confuser_map") or {}),
        thresholds=dict(protocol.get("confuser_thresholds") or {}),
    )
    effective_candidate_universe: dict[str, list[int]] = {}
    for obj in objects:
        object_id = int(obj["object_id"])
        key = str(object_id)
        effective_candidate_universe[key] = sorted(
            set(same_type_visible_ids.get(key, []))
            | set(cross_type_confusable_ids.get(key, []))
            | ({object_id} if target_supported_under_hview.get(key, False) else set())
        )

    audit_assets_payload = {
        "visible_objects_json": None if audit_assets is None else audit_assets.get("visible_objects_json"),
        "rgb_png": None if audit_assets is None else audit_assets.get("rgb_png"),
        "object_index_png": None if audit_assets is None else audit_assets.get("object_index_png"),
        "object_index_map_json": None if audit_assets is None else audit_assets.get("object_index_map_json"),
        "depth_png": None if audit_assets is None else audit_assets.get("depth_png"),
    }

    core_payload = {
        "schema_version": "h_view_confusable_universe_v1",
        "scene_id": scene_id,
        "view_id": int(view_id),
        "same_type_mode": same_type_mode,
        "objects": objects,
        "same_type_visible_ids": same_type_visible_ids,
        "cross_type_confusable_ids": cross_type_confusable_ids,
        "anchor_confusable_ids": anchor_confusable_ids,
        "target_supported_under_hview": target_supported_under_hview,
        "effective_candidate_universe": effective_candidate_universe,
        "support_status_summary": dict(h_view.get("support_status_summary") or {}),
        "visible_objects_consistency_pass": bool(visible_objects_consistency_pass),
        "audit_assets": audit_assets_payload,
        "hview_confusable_universe_version": protocol["schema_version"],
        "hview_confusable_universe_hash": get_hview_confusable_universe_hash(),
        "schema_bundle_hash": get_schema_bundle_hash(),
    }
    universe = {
        **core_payload,
        "confusable_universe_hash": json_hash(core_payload),
    }
    audit = {
        "scene_id": scene_id,
        "view_id": int(view_id),
        "visible_objects_consistency_pass": bool(visible_objects_consistency_pass),
        "visible_objects_consistency_details": consistency_details,
        "audit_assets": audit_assets_payload,
    }
    return universe, audit


def json_hash(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_optional_json_list(path: Path | None) -> list[dict] | None:
    if path is None or not path.exists():
        return None
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"expected JSON list at {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build H(view) confusable universe sidecars.")
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--render-scene-dir", required=True)
    parser.add_argument("--view-id", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--proxy-groups-json", default=str(DEFAULT_PROXY_GROUPS_PATH))
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    render_scene_dir = Path(args.render_scene_dir)
    view_id = int(args.view_id)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    view_dir = render_scene_dir / f"view_{view_id}"
    bundle_path = view_dir / VIEW_BUNDLE_FILENAME
    layout_path = scene_dir / "layout.json"
    metadata_path = scene_dir / "metadata.json"
    visible_objects_json_path = view_dir / "visible_objects.json"
    rgb_path = view_dir / "rgb.png"
    object_index_path = view_dir / "object_index.png"
    object_index_map_path = render_scene_dir / "object_index_map.json"
    depth_path = view_dir / "depth.png"

    bundle = _load_json(bundle_path)
    if not isinstance(bundle, dict):
        raise ValueError(f"expected bundle object: {bundle_path}")
    layout = _load_json(layout_path)
    metadata = _load_json(metadata_path)
    if not isinstance(layout, list):
        raise ValueError(f"expected layout list: {layout_path}")
    if not isinstance(metadata, dict):
        raise ValueError(f"expected metadata object: {metadata_path}")
    scene_id = str(metadata.get("scene_id") or scene_dir.name)

    universe, audit = build_hview_confusable_universe(
        scene_id=scene_id,
        view_id=view_id,
        bundle=bundle,
        layout=layout,
        metadata=metadata,
        proxy_group_map=load_proxy_group_map(Path(args.proxy_groups_json)),
        audit_visible_objects=_load_optional_json_list(visible_objects_json_path),
        audit_assets={
            "visible_objects_json": str(visible_objects_json_path) if visible_objects_json_path.exists() else None,
            "rgb_png": str(rgb_path) if rgb_path.exists() else None,
            "object_index_png": str(object_index_path) if object_index_path.exists() else None,
            "object_index_map_json": str(object_index_map_path) if object_index_map_path.exists() else None,
            "depth_png": str(depth_path) if depth_path.exists() else None,
        },
    )

    with open(output_dir / "h_view_confusable_universe.json", "w") as f:
        json.dump(universe, f, indent=2, ensure_ascii=True)
    with open(output_dir / "h_view_consistency_audit.json", "w") as f:
        json.dump(audit, f, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
