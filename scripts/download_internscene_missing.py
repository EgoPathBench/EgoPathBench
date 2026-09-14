#!/usr/bin/env python3
"""
Download missing InternScenes real2sim assets via HuggingFace mirror.

Default behavior:
- Uses mirror endpoint: https://hf-mirror.com
- Downloads real2sim layout archive: Layout_info.tar.gz
- Tries precise model download from missing_model_uids.txt
- Optionally downloads archive bundles for unresolved prefixes as fallback

Notes:
- InternScenes asset files are gated. You need a HuggingFace token with
  accepted license access for this dataset (403 otherwise).
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
from shutil import copyfileobj
from pathlib import Path


def uid_to_repo_paths(uid: str) -> list[str]:
    # Most assets are flat .glb files; partnet can also be folder/whole.glb.
    return [
        f"asset_library/{uid}.glb",
        f"asset_library/{uid}/whole.glb",
    ]


def load_uid_list(path: Path) -> list[str]:
    uids = []
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith("#"):
                uids.append(s)
    return sorted(set(uids))


def resolve_token(cli_token: str | None) -> str | None:
    if cli_token:
        return cli_token
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")


def is_http_status(exc: Exception, code: int) -> bool:
    response = getattr(exc, "response", None)
    if response is None:
        return False
    return getattr(response, "status_code", None) == code


def is_not_found(exc: Exception) -> bool:
    text = str(exc).lower()
    return is_http_status(exc, 404) or "entry not found" in text or "not found" in text


def is_forbidden(exc: Exception) -> bool:
    text = str(exc).lower()
    return is_http_status(exc, 401) or is_http_status(exc, 403) or "forbidden" in text


def hf_download(
    repo_id: str,
    filename: str,
    local_dir: Path,
    token: str | None,
) -> Path:
    # Import after HF_ENDPOINT is set in main().
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=str(local_dir),
            token=token,
        )
    )


def list_prefix_files(
    repo_id: str,
    prefix: str,
    token: str | None,
) -> list[str]:
    from huggingface_hub import HfApi

    endpoint = os.environ.get("HF_ENDPOINT")
    api = HfApi(endpoint=endpoint) if endpoint else HfApi()
    path_in_repo = f"asset_library/{prefix}"
    entries = list(
        api.list_repo_tree(
            repo_id=repo_id,
            repo_type="dataset",
            path_in_repo=path_in_repo,
            recursive=False,
            token=token,
        )
    )
    return [e.path for e in entries if hasattr(e, "path")]


def extract_tar_gz(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=output_dir)


def normalize_partnet_tree(raw_dir: Path) -> dict[str, int]:
    """
    InternScenes partnet archive extracts to:
      asset_library/partnet_mobility/partnet_mobility/<id>/whole.glb
    while our pipeline expects:
      asset_library/partnet_mobility/<id>/whole.glb
    Create top-level symlinks for compatibility.
    """
    base = raw_dir / "asset_library" / "partnet_mobility"
    nested = base / "partnet_mobility"
    if not nested.is_dir():
        return {"created": 0, "updated": 0, "skipped": 0}

    created = 0
    updated = 0
    skipped = 0
    for src in sorted(d for d in nested.iterdir() if d.is_dir()):
        dst = base / src.name
        if dst.is_symlink():
            if dst.resolve() == src.resolve():
                skipped += 1
                continue
            dst.unlink()
            dst.symlink_to(src)
            updated += 1
            continue
        if dst.exists():
            # Replace empty placeholders if they exist.
            if dst.is_dir() and not any(dst.iterdir()):
                dst.rmdir()
                dst.symlink_to(src)
                updated += 1
            elif dst.is_file():
                dst.unlink()
                dst.symlink_to(src)
                updated += 1
            else:
                skipped += 1
            continue
        dst.symlink_to(src)
        created += 1
    return {"created": created, "updated": updated, "skipped": skipped}


def normalize_objaverse_tree(raw_dir: Path) -> dict[str, int]:
    """
    Objaverse archive may extract to:
      asset_library/objaverse/objaverse/<uid>/*
    while pipeline expects:
      asset_library/objaverse/<uid>/*
    Create top-level symlinks when nested layout exists.
    """
    base = raw_dir / "asset_library" / "objaverse"
    nested = base / "objaverse"
    if not nested.is_dir():
        return {"created": 0, "updated": 0, "skipped": 0}

    created = 0
    updated = 0
    skipped = 0

    # Case A: objaverse/<uid>.glb files directly in nested folder.
    glb_files = [p for p in nested.iterdir() if p.is_file() and p.suffix == ".glb"]
    for src in sorted(glb_files):
        dst = base / src.name
        if dst.is_symlink():
            if dst.resolve() == src.resolve():
                skipped += 1
                continue
            dst.unlink()
            dst.symlink_to(src)
            updated += 1
            continue
        if dst.exists():
            if dst.is_file() and dst.stat().st_size == 0:
                dst.unlink()
                dst.symlink_to(src)
                updated += 1
            else:
                skipped += 1
            continue
        dst.symlink_to(src)
        created += 1

    # Case B: objaverse/<uid>/... directory layout.
    for src in sorted(d for d in nested.iterdir() if d.is_dir()):
        dst = base / src.name
        if dst.is_symlink():
            if dst.resolve() == src.resolve():
                skipped += 1
                continue
            dst.unlink()
            dst.symlink_to(src)
            updated += 1
            continue
        if dst.exists():
            if dst.is_dir() and not any(dst.iterdir()):
                dst.rmdir()
                dst.symlink_to(src)
                updated += 1
            elif dst.is_file():
                dst.unlink()
                dst.symlink_to(src)
                updated += 1
            else:
                skipped += 1
            continue
        dst.symlink_to(src)
        created += 1

    return {"created": created, "updated": updated, "skipped": skipped}


def combine_split_archive(shards: list[Path], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as out_f:
        for shard in shards:
            with open(shard, "rb") as in_f:
                copyfileobj(in_f, out_f)
    return output_path


def download_layout_info(
    repo_id: str,
    raw_dir: Path,
    token: str | None,
    dry_run: bool,
    extract_layout_info: bool,
) -> str:
    archive_rel = "Layout_info.tar.gz"
    layout_root = raw_dir / "Layout_info"
    if dry_run:
        print(f"[dry-run] hf_hub_download: {archive_rel}")
        if extract_layout_info:
            print(f"[dry-run] extract: {archive_rel} -> {layout_root}")
        return str(raw_dir / archive_rel)

    try:
        archive_path = hf_download(
            repo_id=repo_id,
            filename=archive_rel,
            local_dir=raw_dir,
            token=token,
        )
    except Exception as exc:
        if is_forbidden(exc) or "gated" in str(exc).lower():
            raise PermissionError(
                "Layout_info download is gated. Provide --token (or HF_TOKEN/"
                "HUGGINGFACE_TOKEN) with accepted access to "
                "InternRobotics/InternScenes."
            ) from exc
        raise
    if extract_layout_info:
        # Safe to re-run; extraction is idempotent for existing files.
        extract_tar_gz(archive_path, raw_dir)
    return str(archive_path)


def precise_download_missing(
    repo_id: str,
    raw_dir: Path,
    uids: list[str],
    token: str | None,
    dry_run: bool,
) -> tuple[list[str], dict[str, int]]:
    unresolved: list[str] = []
    tried_paths = 0
    downloaded = 0

    for uid in uids:
        success = False
        for repo_path in uid_to_repo_paths(uid):
            tried_paths += 1
            local_path = raw_dir / repo_path
            if local_path.exists():
                success = True
                break
            if dry_run:
                print(f"[dry-run] hf_hub_download: {repo_path}")
                downloaded += 1
                success = True
                break
            try:
                hf_download(
                    repo_id=repo_id,
                    filename=repo_path,
                    local_dir=raw_dir,
                    token=token,
                )
                downloaded += 1
                success = True
                break
            except Exception as exc:
                if is_forbidden(exc):
                    raise PermissionError(
                        "403/401 while downloading gated InternScenes assets. "
                        "Provide --token (or HF_TOKEN/HUGGINGFACE_TOKEN) with "
                        "accepted access to InternRobotics/InternScenes."
                    ) from exc
                if is_not_found(exc):
                    continue
                continue
        if not success:
            unresolved.append(uid)

    stats = {
        "uids_total": len(uids),
        "repo_paths_tried": tried_paths,
        "downloads_triggered": downloaded,
        "unresolved_uids": len(unresolved),
    }
    return unresolved, stats


def fallback_download_by_prefix(
    repo_id: str,
    raw_dir: Path,
    unresolved: list[str],
    token: str | None,
    dry_run: bool,
) -> tuple[list[str], list[str], dict[str, dict[str, int]]]:
    prefixes = sorted({uid.split("/")[0] for uid in unresolved if "/" in uid})
    if not prefixes:
        return [], [], {}

    # Known bundle-style prefixes in InternScenes.
    bundle_map: dict[str, list[str]] = {
        "partnet_mobility": ["asset_library/partnet_mobility/partnet_mobility.tar.gz"],
    }

    # objaverse is sharded under *.tar.gz.00..NN
    if "objaverse" in prefixes:
        try:
            obj_files = list_prefix_files(repo_id, "objaverse", token)
            bundle_map["objaverse"] = [p for p in obj_files if ".tar.gz" in p]
        except Exception:
            bundle_map["objaverse"] = []

    download_paths: list[str] = []
    for prefix in prefixes:
        download_paths.extend(bundle_map.get(prefix, []))
    download_paths = sorted(set(download_paths))

    if dry_run:
        print("[dry-run] fallback prefixes:", prefixes)
        for repo_path in download_paths:
            print(f"[dry-run] hf_hub_download: {repo_path}")
        return prefixes, download_paths, {}

    normalization: dict[str, dict[str, int]] = {}
    downloaded_local_paths: list[Path] = []
    for repo_path in download_paths:
        local_path = hf_download(
            repo_id=repo_id,
            filename=repo_path,
            local_dir=raw_dir,
            token=token,
        )
        downloaded_local_paths.append(local_path)
        # Only partnet uses a single .tar.gz archive in current mapping.
        if repo_path.endswith("partnet_mobility/partnet_mobility.tar.gz"):
            extract_tar_gz(local_path, raw_dir / "asset_library" / "partnet_mobility")
            normalization["partnet_mobility"] = normalize_partnet_tree(raw_dir)

    if "objaverse" in prefixes:
        shards = sorted(
            p for p in downloaded_local_paths
            if p.name.startswith("objaverse.tar.gz.")
        )
        if shards:
            merged = combine_split_archive(
                shards=shards,
                output_path=raw_dir / "asset_library" / "objaverse" / "objaverse.tar.gz",
            )
            extract_tar_gz(merged, raw_dir / "asset_library" / "objaverse")
            normalization["objaverse"] = normalize_objaverse_tree(raw_dir)
    return prefixes, download_paths, normalization


def main() -> None:
    parser = argparse.ArgumentParser(description="Download missing InternScenes assets via mirror")
    parser.add_argument("--repo-id", default="InternRobotics/InternScenes")
    parser.add_argument("--raw-dir", required=True, help="Target internscenes_raw root")
    parser.add_argument(
        "--missing-uids-file",
        required=True,
        help="Path to missing_model_uids.txt generated by audit script",
    )
    parser.add_argument(
        "--hf-endpoint",
        default="https://hf-mirror.com",
        help="HF mirror endpoint (set empty string to keep current environment)",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HuggingFace token (or set HF_TOKEN / HUGGINGFACE_TOKEN)",
    )
    parser.add_argument("--skip-layout-info", action="store_true")
    parser.add_argument(
        "--no-extract-layout-info",
        action="store_true",
        help="Download Layout_info.tar.gz but do not extract",
    )
    parser.add_argument(
        "--fallback-threshold",
        type=int,
        default=200,
        help="Run prefix-level fallback only if unresolved >= this threshold",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--report-json",
        default="doc/internscene_bootstrap/download_report.json",
        help="Path to write download report JSON",
    )
    args = parser.parse_args()

    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint

    token = resolve_token(args.token)
    raw_dir = Path(args.raw_dir).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)

    missing_file = Path(args.missing_uids_file).resolve()
    uids = load_uid_list(missing_file)

    precise_stats = {
        "uids_total": len(uids),
        "repo_paths_tried": 0,
        "downloads_triggered": 0,
        "unresolved_uids": len(uids),
    }
    unresolved: list[str] = list(uids)
    fallback_prefixes: list[str] = []
    fallback_download_paths: list[str] = []
    fallback_normalization: dict[str, dict[str, int]] = {}
    layout_archive_path = ""
    error_message = ""
    auth_blocked = False

    try:
        if not args.skip_layout_info:
            print("Downloading Layout_info.tar.gz for real2sim...")
            layout_archive_path = download_layout_info(
                repo_id=args.repo_id,
                raw_dir=raw_dir,
                token=token,
                dry_run=args.dry_run,
                extract_layout_info=not args.no_extract_layout_info,
            )

        print(f"Precise downloading missing model_uids: {len(uids)}")
        unresolved, precise_stats = precise_download_missing(
            repo_id=args.repo_id,
            raw_dir=raw_dir,
            uids=uids,
            token=token,
            dry_run=args.dry_run,
        )

        if len(unresolved) >= args.fallback_threshold:
            print(
                f"Unresolved uids={len(unresolved)} >= fallback threshold={args.fallback_threshold}, "
                "triggering prefix-level fallback download."
            )
            fallback_prefixes, fallback_download_paths, fallback_normalization = fallback_download_by_prefix(
                repo_id=args.repo_id,
                raw_dir=raw_dir,
                unresolved=unresolved,
                token=token,
                dry_run=args.dry_run,
            )
    except PermissionError as exc:
        auth_blocked = True
        error_message = str(exc)
    except Exception as exc:  # Keep report available even on failure.
        error_message = f"{type(exc).__name__}: {exc}"
        lowered = str(exc).lower()
        if is_forbidden(exc) or "gated" in lowered or "restricted" in lowered:
            auth_blocked = True

    report = {
        "repo_id": args.repo_id,
        "hf_endpoint": os.environ.get("HF_ENDPOINT", ""),
        "raw_dir": str(raw_dir),
        "missing_uids_file": str(missing_file),
        "skip_layout_info": args.skip_layout_info,
        "extract_layout_info": not args.no_extract_layout_info,
        "layout_archive_path": layout_archive_path,
        "dry_run": args.dry_run,
        "token_present": bool(token),
        "auth_blocked": auth_blocked,
        "precise_download": precise_stats,
        "fallback_prefixes": fallback_prefixes,
        "fallback_download_paths": fallback_download_paths,
        "fallback_normalization": fallback_normalization,
        "unresolved_uids_sample": unresolved[:200],
    }
    if error_message:
        report["error"] = error_message

    report_path = Path(args.report_json).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    if error_message:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
