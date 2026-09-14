#!/usr/bin/env python3
"""Build a relocatable local release; this command does not upload anything.

Run evaluation with the generated dataset directory as the working directory.
Original question identities and evaluation labels are preserved.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def prepare(source: Path, output: Path) -> dict:
    source = source.resolve()
    output = output.resolve()
    manifest = json.loads((source / "release/release_manifest.json").read_text())
    if not manifest.get("formal_release"):
        raise ValueError("Expected a formal release")
    output.mkdir(parents=True, exist_ok=False)
    assets = {}
    counts = {}

    def rewrite(value, evalfix=False, key=None):
        if isinstance(value, dict):
            return {name: rewrite(item, evalfix, name) for name, item in value.items()}
        if isinstance(value, list):
            return [rewrite(item, evalfix) for item in value]
        if not isinstance(value, str):
            return value
        if value.startswith("/"):
            path = Path(value)
            if "task_outputs" not in path.parts:
                if key in ("image_path", "visible_waypoints_path", "direct_pairs_ref"):
                    raise ValueError(f"Unmapped runtime path: {value}")
                return value  # Historical construction provenance, not a runtime dependency.
            relative = Path(*path.parts[path.parts.index("task_outputs") + 1:])
            destination = Path("assets") / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            assets[destination] = path
            if path.name.startswith("visible_waypoints"):
                for graph in path.parent.parent.glob("waypoint_graph*.json"):
                    assets[Path("assets") / relative.parent.parent / graph.name] = graph
            return destination.as_posix()
        if value.startswith("sidecars/"):
            local = source / value
            if not local.exists() and evalfix:
                local = source / "benchmark_evalfix/benchmark" / value
            if not local.exists():
                raise FileNotFoundError(local)
            return local.relative_to(source).as_posix()
        return value

    for directory in ("release", "benchmark_evalfix", "sidecars"):
        for path in sorted((source / directory).rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".jsonl":
                n = 0
                with path.open() as src, target.open("w") as dst:
                    for line in src:
                        if line.strip():
                            row = rewrite(json.loads(line), directory == "benchmark_evalfix")
                            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                            n += 1
                counts[relative.as_posix()] = n
            elif path.suffix == ".json":
                payload = rewrite(json.loads(path.read_text()), directory == "benchmark_evalfix")
                target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            else:
                shutil.copy2(path, target)

    for split, details in manifest["splits"].items():
        for task, expected in details["question_counts_by_task"].items():
            for kind in ("vqa", "gt"):
                actual = counts[f"release/{split}/{kind}/{kind}_next_{task}.jsonl"]
                if actual != expected:
                    raise ValueError(f"{split}/{task}/{kind}: {actual} != {expected}")
    for destination, path in sorted(assets.items()):
        target = output / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    report = {
        "status": "local_candidate_not_published",
        "source_manifest_sha256": hashlib.sha256((source / "release/release_manifest.json").read_bytes()).hexdigest(),
        "counts": counts,
        "asset_files": len(assets),
        "asset_bytes": sum(path.stat().st_size for path in assets.values()),
        "path_contract": "Paths are relative to the dataset root; run evaluation from that directory.",
        "license_status": "Awaiting maintainer license selection and upstream redistribution review",
    }
    (output / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Formal published data directory")
    parser.add_argument("--output", type=Path, required=True, help="New local candidate directory")
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output), indent=2))
