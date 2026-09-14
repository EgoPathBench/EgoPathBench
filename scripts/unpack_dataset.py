#!/usr/bin/env python3
"""Verify and extract downloaded EgoPathBench archives (Python 3.12+)."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def unpack(source: Path, output: Path):
    records = json.loads((source / "archive_manifest.json").read_text())
    for record in records:
        path = source / record["file"]
        if path.stat().st_size != record["bytes"]:
            raise ValueError(f"Incomplete archive: {path.name}")
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != record["sha256"]:
                raise ValueError(f"Checksum mismatch: {path.name}")
    output.mkdir(parents=True, exist_ok=False)
    for record in records:
        with tarfile.open(source / record["file"]) as archive:
            archive.extractall(output, filter="data")
        print("Extracted", record["file"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    unpack(args.source, args.output)
