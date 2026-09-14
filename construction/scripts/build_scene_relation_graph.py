#!/usr/bin/env python3
"""
Build layout-derived trusted scene relation priors.

Release v1 intentionally restricts itself to non-direction-ambiguous relation types:
`near`, `on`, `inside`, and `under`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
CUE_REPAIR_PROTOCOL_PATH = ROOT_DIR / "configs" / "cue_repair_protocol_v1.json"
SCENE_RELATION_GRAPH_SCHEMA_VERSION = "scene_relation_graph_v1"
RELATION_GENERATION_SOURCE = "layout_bbox_pairwise_v1"
NEAR_EDGE_GAP_THRESHOLD_M = 0.60
SUPPORT_VERTICAL_GAP_THRESHOLD_M = 0.12
UNDER_VERTICAL_GAP_THRESHOLD_M = 0.60
XY_OVERLAP_MIN_RATIO = 0.25
INSIDE_MARGIN_TOLERANCE_M = 0.02


def _load_json(path: Path) -> dict | list:
    with open(path) as f:
        return json.load(f)


def load_cue_repair_protocol(path: Path = CUE_REPAIR_PROTOCOL_PATH) -> dict:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"invalid cue repair protocol: {path}")
    if payload.get("schema_version") != "cue_repair_protocol_v1":
        raise ValueError(f"unexpected cue repair protocol schema_version in {path}")
    return payload


def _bbox_ranges(obj: dict) -> dict[str, float]:
    bbox = obj.get("bbox")
    if not isinstance(bbox, list) or len(bbox) < 6:
        raise ValueError(f"object id={obj.get('id')} missing bbox[0:6]")
    cx, cy, cz, sx, sy, sz = [float(v) for v in bbox[:6]]
    return {
        "xmin": cx - sx / 2.0,
        "xmax": cx + sx / 2.0,
        "ymin": cy - sy / 2.0,
        "ymax": cy + sy / 2.0,
        "zmin": cz - sz / 2.0,
        "zmax": cz + sz / 2.0,
        "cx": cx,
        "cy": cy,
        "cz": cz,
        "sx": sx,
        "sy": sy,
        "sz": sz,
    }


def _xy_intersection(a: dict[str, float], b: dict[str, float]) -> tuple[float, float, float]:
    overlap_x = max(0.0, min(a["xmax"], b["xmax"]) - max(a["xmin"], b["xmin"]))
    overlap_y = max(0.0, min(a["ymax"], b["ymax"]) - max(a["ymin"], b["ymin"]))
    return overlap_x, overlap_y, overlap_x * overlap_y


def _xy_overlap_ratio(a: dict[str, float], b: dict[str, float]) -> float:
    _, _, area = _xy_intersection(a, b)
    base_area = min(a["sx"] * a["sy"], b["sx"] * b["sy"])
    if base_area <= 0.0:
        return 0.0
    return area / base_area


def _xy_edge_gap(a: dict[str, float], b: dict[str, float]) -> float:
    gap_x = max(0.0, a["xmin"] - b["xmax"], b["xmin"] - a["xmax"])
    gap_y = max(0.0, a["ymin"] - b["ymax"], b["ymin"] - a["ymax"])
    return (gap_x**2 + gap_y**2) ** 0.5


def _relation_uncertainty(margin: float) -> str:
    if margin >= 0.10:
        return "low"
    if margin >= 0.03:
        return "medium"
    return "high"


def _base_relation(
    *,
    scene_id: str,
    target: dict,
    anchor: dict,
    relation_type: str,
    relation_frame: str,
    relation_margin: float,
    scene_relation_evidence: dict,
    metadata: dict | None,
) -> dict:
    source_group_id = None if not isinstance(metadata, dict) else metadata.get("source_group_id")
    dataset = None if not isinstance(metadata, dict) else metadata.get("dataset")
    return {
        "relation_candidate_id": f"{scene_id}::{int(target['id'])}::{relation_type}::{int(anchor['id'])}",
        "target_id": int(target["id"]),
        "anchor_ids": [int(anchor["id"])],
        "relation_type": relation_type,
        "relation_frame": relation_frame,
        "scene_relation_evidence": scene_relation_evidence,
        "relation_margin": round(float(relation_margin), 6),
        "relation_uncertainty": _relation_uncertainty(float(relation_margin)),
        "anchor_generation_source": RELATION_GENERATION_SOURCE,
        "relation_source_provenance": {
            "source": RELATION_GENERATION_SOURCE,
            "source_group_id": source_group_id,
            "dataset": dataset,
        },
    }


def _iter_trusted_relations(scene_id: str, layout: list[dict], metadata: dict | None) -> list[dict]:
    relations: list[dict] = []
    annotated = []
    for obj in layout:
        if "id" not in obj:
            continue
        annotated.append((obj, _bbox_ranges(obj)))

    for target, target_box in annotated:
        for anchor, anchor_box in annotated:
            if int(target["id"]) == int(anchor["id"]):
                continue

            overlap_ratio = _xy_overlap_ratio(target_box, anchor_box)
            edge_gap = _xy_edge_gap(target_box, anchor_box)
            bottom_gap = abs(target_box["zmin"] - anchor_box["zmax"])
            under_gap = anchor_box["zmin"] - target_box["zmax"]

            if edge_gap <= NEAR_EDGE_GAP_THRESHOLD_M:
                relations.append(
                    _base_relation(
                        scene_id=scene_id,
                        target=target,
                        anchor=anchor,
                        relation_type="near",
                        relation_frame="scene_planar",
                        relation_margin=NEAR_EDGE_GAP_THRESHOLD_M - edge_gap,
                        scene_relation_evidence={
                            "target_category": target.get("category"),
                            "anchor_categories": [anchor.get("category")],
                            "xy_edge_gap_m": round(edge_gap, 6),
                            "xy_overlap_ratio": round(overlap_ratio, 6),
                        },
                        metadata=metadata,
                    )
                )

            if (
                target_box["cz"] > anchor_box["cz"]
                and overlap_ratio >= XY_OVERLAP_MIN_RATIO
                and bottom_gap <= SUPPORT_VERTICAL_GAP_THRESHOLD_M
            ):
                relations.append(
                    _base_relation(
                        scene_id=scene_id,
                        target=target,
                        anchor=anchor,
                        relation_type="on",
                        relation_frame="scene_gravity_aligned",
                        relation_margin=min(
                            SUPPORT_VERTICAL_GAP_THRESHOLD_M - bottom_gap,
                            overlap_ratio - XY_OVERLAP_MIN_RATIO,
                        ),
                        scene_relation_evidence={
                            "target_category": target.get("category"),
                            "anchor_categories": [anchor.get("category")],
                            "xy_overlap_ratio": round(overlap_ratio, 6),
                            "vertical_gap_m": round(bottom_gap, 6),
                        },
                        metadata=metadata,
                    )
                )

            inside_margins = (
                target_box["xmin"] - anchor_box["xmin"],
                anchor_box["xmax"] - target_box["xmax"],
                target_box["ymin"] - anchor_box["ymin"],
                anchor_box["ymax"] - target_box["ymax"],
                target_box["zmin"] - anchor_box["zmin"],
                anchor_box["zmax"] - target_box["zmax"],
            )
            if min(inside_margins) >= -INSIDE_MARGIN_TOLERANCE_M:
                relations.append(
                    _base_relation(
                        scene_id=scene_id,
                        target=target,
                        anchor=anchor,
                        relation_type="inside",
                        relation_frame="scene_gravity_aligned",
                        relation_margin=min(inside_margins) + INSIDE_MARGIN_TOLERANCE_M,
                        scene_relation_evidence={
                            "target_category": target.get("category"),
                            "anchor_categories": [anchor.get("category")],
                            "inside_margins_m": [round(float(v), 6) for v in inside_margins],
                        },
                        metadata=metadata,
                    )
                )

            if (
                target_box["cz"] < anchor_box["cz"]
                and overlap_ratio >= XY_OVERLAP_MIN_RATIO
                and under_gap >= 0.0
                and under_gap <= UNDER_VERTICAL_GAP_THRESHOLD_M
            ):
                relations.append(
                    _base_relation(
                        scene_id=scene_id,
                        target=target,
                        anchor=anchor,
                        relation_type="under",
                        relation_frame="scene_gravity_aligned",
                        relation_margin=min(
                            UNDER_VERTICAL_GAP_THRESHOLD_M - under_gap,
                            overlap_ratio - XY_OVERLAP_MIN_RATIO,
                        ),
                        scene_relation_evidence={
                            "target_category": target.get("category"),
                            "anchor_categories": [anchor.get("category")],
                            "xy_overlap_ratio": round(overlap_ratio, 6),
                            "vertical_clearance_m": round(under_gap, 6),
                        },
                        metadata=metadata,
                    )
                )

    relations.sort(
        key=lambda relation: (
            int(relation["target_id"]),
            str(relation["relation_type"]),
            tuple(int(v) for v in relation["anchor_ids"]),
        )
    )
    return relations


def build_scene_relation_graph(
    *,
    scene_id: str,
    layout: list[dict],
    metadata: dict | None = None,
    cue_protocol: dict | None = None,
) -> tuple[dict, dict]:
    protocol = cue_protocol or load_cue_repair_protocol()
    trusted_types = {
        str(v)
        for v in list(protocol.get("trusted_relation_types") or [])
        if isinstance(v, str) and v
    }
    disabled_types = {
        str(v)
        for v in list(protocol.get("disabled_relation_types") or [])
        if isinstance(v, str) and v
    }
    allowed_types = trusted_types - disabled_types

    relations = [
        relation
        for relation in _iter_trusted_relations(scene_id, layout, metadata)
        if str(relation["relation_type"]) in allowed_types
    ]

    graph = {
        "schema_version": SCENE_RELATION_GRAPH_SCHEMA_VERSION,
        "scene_id": scene_id,
        "relations": relations,
    }
    provenance = {
        "scene_id": scene_id,
        "relation_generation_source": RELATION_GENERATION_SOURCE,
        "cue_repair_protocol_version": protocol["schema_version"],
        "trusted_relation_types": sorted(allowed_types),
        "source_group_id": None if not isinstance(metadata, dict) else metadata.get("source_group_id"),
        "dataset": None if not isinstance(metadata, dict) else metadata.get("dataset"),
    }
    return graph, provenance


def _infer_scene_id(scene_dir: Path, metadata: dict | None) -> str:
    if isinstance(metadata, dict) and isinstance(metadata.get("scene_id"), str) and metadata["scene_id"].strip():
        return metadata["scene_id"].strip()
    return scene_dir.name


def main() -> None:
    parser = argparse.ArgumentParser(description="Build layout-based trusted scene relation graph.")
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layout_path = scene_dir / "layout.json"
    metadata_path = scene_dir / "metadata.json"
    if not layout_path.exists():
        raise FileNotFoundError(f"missing layout.json: {layout_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"missing metadata.json: {metadata_path}")

    layout = _load_json(layout_path)
    metadata = _load_json(metadata_path)
    if not isinstance(layout, list):
        raise ValueError(f"layout.json must contain a list: {layout_path}")
    if not isinstance(metadata, dict):
        raise ValueError(f"metadata.json must contain an object: {metadata_path}")

    scene_id = _infer_scene_id(scene_dir, metadata)
    graph, provenance = build_scene_relation_graph(
        scene_id=scene_id,
        layout=layout,
        metadata=metadata,
    )
    provenance = {
        **provenance,
        "layout_path": str(layout_path),
        "metadata_path": str(metadata_path),
    }

    with open(output_dir / "scene_relation_graph.json", "w") as f:
        json.dump(graph, f, indent=2, ensure_ascii=True)
    with open(output_dir / "scene_relation_provenance.json", "w") as f:
        json.dump(provenance, f, indent=2, ensure_ascii=True)


if __name__ == "__main__":
    main()
