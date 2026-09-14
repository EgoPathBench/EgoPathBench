#!/usr/bin/env python3
"""Package a prepared dataset into downloadable archives with SHA-256 checksums."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile


def package(source: Path, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    names = []
    for directory in ("release", "benchmark_evalfix", "sidecars", "training"):
        name = directory + ".tar.gz"
        with tarfile.open(output / name, "w:gz", compresslevel=3) as archive:
            archive.add(source / directory, arcname=directory)
        names.append(name)
        print("Packaged", name, flush=True)
    buckets = [[] for _ in range(32)]
    for scene in sorted((source / "assets").iterdir()):
        index = int(hashlib.sha256(scene.name.encode()).hexdigest(), 16) % len(buckets)
        buckets[index].append(scene)
    for index, scenes in enumerate(buckets):
        name = f"assets-{index:03d}.tar"
        with tarfile.open(output / name, "w") as archive:
            for scene in scenes:
                archive.add(scene, arcname=scene.relative_to(source))
        names.append(name)
        print("Packaged", name, flush=True)
    records = []
    for name in names:
        path = output / name
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        records.append({"file": name, "bytes": path.stat().st_size, "sha256": digest})
    (output / "archive_manifest.json").write_text(json.dumps(records, indent=2) + "\n")
    for name in ("README.md", "THIRD_PARTY_NOTICES.md", "preparation_report.json", "reference_route_validation.json"):
        shutil.copy2(source / name, output / name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    package(args.source, args.output)
