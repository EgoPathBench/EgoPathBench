#!/usr/bin/env python3
"""
Fix scenes/*/StructureMesh symlinks to local InternScenes Layout_info.

Expected scene id format:
  {dataset}__{scene_id}
where dataset is typically 3rscan or scannet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_scene_name(name: str) -> tuple[str, str] | None:
    if "__" not in name:
        return None
    dataset, sid = name.split("__", 1)
    if not dataset or not sid:
        return None
    return dataset, sid


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix StructureMesh symlinks")
    parser.add_argument("--scenes-dir", required=True)
    parser.add_argument("--layout-info-dir", required=True, help=".../internscenes_raw/Layout_info")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--report-json",
        default="doc/internscene_bootstrap/symlink_fix_report.json",
        help="Path to output report",
    )
    args = parser.parse_args()

    scenes_dir = Path(args.scenes_dir).resolve()
    layout_info_dir = Path(args.layout_info_dir).resolve()
    report_path = Path(args.report_json).resolve()

    fixed = 0
    already_ok = 0
    missing_target = 0
    skipped = 0
    broken_existing = 0

    details: list[dict] = []

    for scene_dir in sorted(d for d in scenes_dir.iterdir() if d.is_dir()):
        parsed = parse_scene_name(scene_dir.name)
        if parsed is None:
            skipped += 1
            continue
        dataset, sid = parsed
        target = layout_info_dir / dataset / sid / "StructureMesh"
        link = scene_dir / "StructureMesh"

        if not target.exists():
            missing_target += 1
            details.append({"scene_id": scene_dir.name, "status": "missing_target", "target": str(target)})
            continue

        if link.is_symlink():
            resolved = link.resolve(strict=False)
            if resolved == target.resolve():
                already_ok += 1
                continue
            if not link.exists():
                broken_existing += 1
            if not args.dry_run:
                link.unlink()
        elif link.exists():
            # Existing file/dir but not symlink
            if not args.dry_run:
                if link.is_dir():
                    skipped += 1
                    details.append({
                        "scene_id": scene_dir.name,
                        "status": "non_symlink_dir",
                        "path": str(link),
                    })
                    continue
                link.unlink()
        if not args.dry_run:
            link.symlink_to(target)
        fixed += 1

    report = {
        "scenes_dir": str(scenes_dir),
        "layout_info_dir": str(layout_info_dir),
        "dry_run": args.dry_run,
        "fixed": fixed,
        "already_ok": already_ok,
        "missing_target": missing_target,
        "broken_existing": broken_existing,
        "skipped": skipped,
        "details_sample": details[:200],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
