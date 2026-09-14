#!/usr/bin/env python3
"""
Audit InternScenes real2sim assets for NavBench3D.

This script:
1) Scans scenes/*/layout.json and collects required model_uids.
2) Checks which model files already exist under asset_library.
3) Writes missing model_uids for precise mirror download.
4) Builds a small smoke scene list (10-50 by default) for fast validation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


def expected_model_paths(model_uid: str, asset_dir: Path) -> list[Path]:
    if model_uid.startswith("partnet_mobility/"):
        return [asset_dir / model_uid / "whole.glb"]
    return [
        asset_dir / f"{model_uid}.glb",
        asset_dir / model_uid / "whole.glb",
    ]


def load_layout(path: Path) -> list[dict]:
    with open(path, "r") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def collect_scenes(scenes_dir: Path) -> list[Path]:
    return sorted(
        d for d in scenes_dir.iterdir()
        if d.is_dir() and (d / "layout.json").exists()
    )


def collect_required_uids(layout: Iterable[dict]) -> set[str]:
    uids: set[str] = set()
    for obj in layout:
        uid = obj.get("model_uid", "")
        if isinstance(uid, str) and uid:
            uids.add(uid)
    return uids


def scene_has_render(scene_id: str, renders_dir: Path | None) -> bool:
    if renders_dir is None:
        return True
    return (renders_dir / scene_id / "cameras.json").exists()


def write_text_list(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for v in values:
            f.write(f"{v}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit InternScenes missing assets")
    parser.add_argument("--scenes-dir", required=True, help="Path to scenes/ directory")
    parser.add_argument("--asset-dir", required=True, help="Path to asset_library/ directory")
    parser.add_argument(
        "--output-dir",
        default="doc/internscene_bootstrap",
        help="Directory to write audit outputs",
    )
    parser.add_argument(
        "--smoke-size",
        type=int,
        default=20,
        help="Smoke list size (recommended 10-50)",
    )
    parser.add_argument(
        "--renders-dir",
        default=None,
        help="Optional renders root. If set, smoke list prefers scenes with cameras.json",
    )
    args = parser.parse_args()

    scenes_dir = Path(args.scenes_dir).resolve()
    asset_dir = Path(args.asset_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    renders_dir = Path(args.renders_dir).resolve() if args.renders_dir else None

    scene_dirs = collect_scenes(scenes_dir)
    if not scene_dirs:
        raise RuntimeError(f"No scenes found under {scenes_dir}")

    required_uids: set[str] = set()
    missing_uids: set[str] = set()
    scene_rows: list[dict] = []

    for scene_dir in scene_dirs:
        layout = load_layout(scene_dir / "layout.json")
        scene_uids = collect_required_uids(layout)
        required_uids.update(scene_uids)

        scene_missing: set[str] = set()
        for uid in scene_uids:
            if not any(p.exists() for p in expected_model_paths(uid, asset_dir)):
                scene_missing.add(uid)
                missing_uids.add(uid)

        scene_rows.append({
            "scene_id": scene_dir.name,
            "required_model_uids": len(scene_uids),
            "missing_model_uids": len(scene_missing),
            "has_render": scene_has_render(scene_dir.name, renders_dir),
        })

    # Prefer scenes with 0 missing assets and existing renders.
    sorted_rows = sorted(
        scene_rows,
        key=lambda r: (
            r["missing_model_uids"],
            0 if r["has_render"] else 1,
            r["scene_id"],
        ),
    )
    smoke_size = max(1, args.smoke_size)
    smoke_scene_ids = [r["scene_id"] for r in sorted_rows[:smoke_size]]

    output_dir.mkdir(parents=True, exist_ok=True)

    write_text_list(output_dir / "required_model_uids.txt", sorted(required_uids))
    write_text_list(output_dir / "missing_model_uids.txt", sorted(missing_uids))
    write_text_list(output_dir / "smoke_scenes.txt", smoke_scene_ids)

    with open(output_dir / "scene_missing_counts.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["scene_id", "required_model_uids", "missing_model_uids", "has_render"],
        )
        writer.writeheader()
        writer.writerows(sorted_rows)

    summary = {
        "scenes_total": len(scene_dirs),
        "required_model_uids_total": len(required_uids),
        "missing_model_uids_total": len(missing_uids),
        "asset_dir_exists": asset_dir.exists(),
        "smoke_size": smoke_size,
        "smoke_scenes_path": str(output_dir / "smoke_scenes.txt"),
    }
    with open(output_dir / "audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
