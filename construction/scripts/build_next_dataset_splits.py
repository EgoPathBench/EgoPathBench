#!/usr/bin/env python3
"""
Build release-grade scene split files for Next dataset generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_internscene_split_manifest import build_split_manifest


def infer_dataset_and_group(scene_id: str) -> tuple[str, str]:
    parts = [part for part in scene_id.split("__") if part]
    if len(parts) < 2:
        raise ValueError(f"cannot infer dataset/group from scene id: {scene_id}")
    dataset = parts[0]
    if dataset == "matterport3d" and len(parts) >= 3:
        return dataset, "__".join(parts[:2])
    return dataset, scene_id


def collect_scene_records(scenes_dir: Path) -> list[dict]:
    records: list[dict] = []
    for scene_dir in sorted(p for p in scenes_dir.iterdir() if p.is_dir()):
        metadata_path = scene_dir / "metadata.json"
        layout_path = scene_dir / "layout.json"
        if not metadata_path.exists() and not layout_path.exists():
            continue
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            dataset = str(metadata.get("dataset") or infer_dataset_and_group(scene_dir.name)[0])
            source_group_id = str(metadata.get("source_group_id") or infer_dataset_and_group(scene_dir.name)[1])
        else:
            dataset, source_group_id = infer_dataset_and_group(scene_dir.name)
        records.append(
            {
                "qualified_id": scene_dir.name,
                "dataset": dataset,
                "source_group_id": source_group_id,
            }
        )
    if not records:
        raise RuntimeError(f"no scene directories found under {scenes_dir}")
    return records


def build_next_dataset_splits(
    *,
    scenes_dir: Path,
    split_ratios: dict[str, float],
    seed: str,
) -> dict:
    records = collect_scene_records(scenes_dir)
    manifest = build_split_manifest(records, split_ratios=split_ratios, seed=seed)
    manifest["schema_version"] = "next_dataset_split_manifest_v1"
    return manifest


def write_split_outputs(output_dir: Path, manifest: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    for split_name, scene_ids in manifest.get("splits", {}).items():
        if not scene_ids:
            continue
        with open(output_dir / f"{split_name}_scenes.txt", "w") as f:
            for scene_id in scene_ids:
                f.write(f"{scene_id}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Next dataset scene splits.")
    parser.add_argument("--scenes-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", default="navbench3d-2026-03-21")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--benchmark-ratio", type=float, default=None)
    parser.add_argument("--test-ratio", type=float, default=None, help="Deprecated alias for --benchmark-ratio")
    args = parser.parse_args()
    benchmark_ratio = (
        float(args.benchmark_ratio)
        if args.benchmark_ratio is not None
        else float(args.test_ratio if args.test_ratio is not None else 0.1)
    )

    manifest = build_next_dataset_splits(
        scenes_dir=Path(args.scenes_dir).resolve(),
        split_ratios={
            "train": args.train_ratio,
            "val": args.val_ratio,
            "benchmark": benchmark_ratio,
        },
        seed=args.seed,
    )
    write_split_outputs(Path(args.output_dir).resolve(), manifest)
    print(json.dumps(manifest["counts_by_dataset"], indent=2))


if __name__ == "__main__":
    main()
