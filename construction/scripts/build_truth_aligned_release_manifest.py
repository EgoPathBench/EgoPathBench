#!/usr/bin/env python3
"""
Build a canonical current/archive layout for truth-aligned releases.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


TASKS = ("a1", "b1", "a2", "b2", "c")
CANONICAL_SUBSETS = {
    "benchmark_truthaligned_v1": "benchmark",
    "val_truthaligned_v1": "val",
    "train_truthaligned_v1": "train",
}
INHERITED_AUX_SUBSETS = {
    "train_weakaux_legacy_v1": "train",
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _copy_tree(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"missing required release subset: {source}")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite existing release subset: {target}")
    shutil.copytree(source, target)


def _copy_file(source: Path, target: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"missing required file: {source}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copy2(source, target)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _rewrite_sidecar_payload_paths(value: object, *, source_sidecars_root: Path, target_sidecars_root: Path) -> object:
    if isinstance(value, dict):
        return {
            key: _rewrite_sidecar_payload_paths(
                child,
                source_sidecars_root=source_sidecars_root,
                target_sidecars_root=target_sidecars_root,
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [
            _rewrite_sidecar_payload_paths(
                child,
                source_sidecars_root=source_sidecars_root,
                target_sidecars_root=target_sidecars_root,
            )
            for child in value
        ]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return value
        path = Path(text)
        if not path.is_absolute():
            return value
        try:
            suffix = path.resolve().relative_to(source_sidecars_root.resolve())
        except ValueError:
            sidecar_parts = list(path.parts)
            sidecar_indexes = [idx for idx, part in enumerate(sidecar_parts) if part == "sidecars"]
            for idx in reversed(sidecar_indexes):
                suffix = Path(*sidecar_parts[idx + 1 :])
                if not suffix.parts:
                    continue
                candidate = target_sidecars_root / suffix
                if candidate.exists() or suffix.name in {"h_view.json", "c_main_inventory.json"}:
                    return str(candidate)
            return value
        return str(target_sidecars_root / suffix)
    return value


def _rewrite_sidecar_path(
    value: object,
    *,
    source_sidecars_root: Path,
    target_sidecars_root: Path,
    strict: bool = False,
) -> object:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return value
    source_path = Path(text)
    resolved_source_root = source_sidecars_root.resolve()
    resolved_source_path = source_path.resolve()
    if strict and not source_path.exists():
        raise FileNotFoundError(f"missing c sidecar file: {source_path}")
    try:
        suffix = resolved_source_path.relative_to(resolved_source_root)
    except ValueError:
        if strict:
            raise ValueError(f"c sidecar path must stay under {source_sidecars_root}, got: {text}")
        return value
    return str(target_sidecars_root / suffix)


def _rewrite_gt_row_sidecars(row: dict, *, source_sidecars_root: Path, target_sidecars_root: Path) -> dict:
    updated = dict(row)
    for key in ("candidate_inventory_ref", "h_view_ref"):
        if key in updated:
            updated[key] = _rewrite_sidecar_path(
                updated.get(key),
                source_sidecars_root=source_sidecars_root,
                target_sidecars_root=target_sidecars_root,
                strict=True,
            )
    derivation_trace = dict(updated.get("derivation_trace") or {})
    candidate_inventory = dict(derivation_trace.get("candidate_inventory") or {})
    if "path" in candidate_inventory:
        candidate_inventory["path"] = _rewrite_sidecar_path(
            candidate_inventory.get("path"),
            source_sidecars_root=source_sidecars_root,
            target_sidecars_root=target_sidecars_root,
            strict=True,
        )
        derivation_trace["candidate_inventory"] = candidate_inventory
    h_view = dict(derivation_trace.get("h_view") or {})
    if "path" in h_view:
        h_view["path"] = _rewrite_sidecar_path(
            h_view.get("path"),
            source_sidecars_root=source_sidecars_root,
            target_sidecars_root=target_sidecars_root,
            strict=True,
        )
        derivation_trace["h_view"] = h_view
    if derivation_trace:
        updated["derivation_trace"] = derivation_trace
    return updated


def _rewrite_vqa_row_sidecars(row: dict, *, source_sidecars_root: Path, target_sidecars_root: Path) -> dict:
    updated = dict(row)
    ground_truth = dict(updated.get("ground_truth") or {})
    if "candidate_inventory_ref" in ground_truth:
        ground_truth["candidate_inventory_ref"] = _rewrite_sidecar_path(
            ground_truth.get("candidate_inventory_ref"),
            source_sidecars_root=source_sidecars_root,
            target_sidecars_root=target_sidecars_root,
            strict=True,
        )
        updated["ground_truth"] = ground_truth
    return updated


def _copy_and_rewrite_task_output_path(value: object, *, target_assets_root: Path) -> object:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return value
    source_path = Path(text)
    if not source_path.is_absolute():
        return value
    parts = list(source_path.parts)
    if "task_outputs" not in parts:
        return value
    idx = parts.index("task_outputs")
    rel_path = Path(*parts[idx + 1 :])
    target_path = target_assets_root / "task_outputs" / rel_path
    _copy_file(source_path, target_path)
    return str(target_path)


def _rewrite_row_task_output_refs(row: dict, *, target_assets_root: Path) -> dict:
    updated = dict(row)
    for key in ("image_path", "visible_waypoints_path"):
        if key in updated:
            updated[key] = _copy_and_rewrite_task_output_path(
                updated.get(key),
                target_assets_root=target_assets_root,
            )
    return updated


def _localize_subset_task_output_refs(*, subset_dir: Path) -> None:
    target_assets_root = subset_dir / "assets"
    for task in TASKS:
        gt_path = subset_dir / "gt" / f"gt_next_{task}.jsonl"
        if gt_path.exists():
            gt_rows = [
                _rewrite_row_task_output_refs(row, target_assets_root=target_assets_root)
                for row in load_jsonl(gt_path)
            ]
            write_jsonl(gt_path, gt_rows)
        vqa_path = subset_dir / "vqa" / f"vqa_next_{task}.jsonl"
        if vqa_path.exists():
            vqa_rows = [
                _rewrite_row_task_output_refs(row, target_assets_root=target_assets_root)
                for row in load_jsonl(vqa_path)
            ]
            write_jsonl(vqa_path, vqa_rows)


def _rewrite_c_sidecars_into_current(*, subset_dir: Path, repaired_subset_dir: Path) -> None:
    gt_path = subset_dir / "gt" / "gt_next_c.jsonl"
    vqa_path = subset_dir / "vqa" / "vqa_next_c.jsonl"
    if not gt_path.exists() and not vqa_path.exists():
        return
    source_sidecars_root = repaired_subset_dir / "sidecars"
    target_sidecars_root = subset_dir / "sidecars"
    if gt_path.exists():
        gt_rows = [
            _rewrite_gt_row_sidecars(
                row,
                source_sidecars_root=source_sidecars_root,
                target_sidecars_root=target_sidecars_root,
            )
            for row in load_jsonl(gt_path)
        ]
        write_jsonl(gt_path, gt_rows)
    if vqa_path.exists():
        vqa_rows = [
            _rewrite_vqa_row_sidecars(
                row,
                source_sidecars_root=source_sidecars_root,
                target_sidecars_root=target_sidecars_root,
            )
            for row in load_jsonl(vqa_path)
        ]
        write_jsonl(vqa_path, vqa_rows)
    for inventory_path in (subset_dir / "sidecars").rglob("c_main_inventory.json"):
        payload = json.loads(inventory_path.read_text())
        rewritten = _rewrite_sidecar_payload_paths(
            payload,
            source_sidecars_root=source_sidecars_root,
            target_sidecars_root=target_sidecars_root,
        )
        inventory_path.write_text(json.dumps(rewritten, ensure_ascii=True) + "\n")


def _prune_subset_rows_by_scene(*, subset_dir: Path, blocked_scenes: set[str]) -> set[str]:
    kept_scenes: set[str] = set()
    if not blocked_scenes:
        for task in TASKS:
            for row in load_jsonl(subset_dir / "gt" / f"gt_next_{task}.jsonl"):
                scene_id = str(row.get("scene_id") or "").strip()
                if scene_id:
                    kept_scenes.add(scene_id)
        return kept_scenes

    for task in TASKS:
        gt_path = subset_dir / "gt" / f"gt_next_{task}.jsonl"
        vqa_path = subset_dir / "vqa" / f"vqa_next_{task}.jsonl"
        gt_rows = load_jsonl(gt_path)
        if gt_rows:
            filtered_gt = []
            kept_qids: set[str] = set()
            for row in gt_rows:
                scene_id = str(row.get("scene_id") or "").strip()
                if scene_id and scene_id in blocked_scenes:
                    continue
                filtered_gt.append(row)
                question_id = str(row.get("question_id") or "").strip()
                if question_id:
                    kept_qids.add(question_id)
                if scene_id:
                    kept_scenes.add(scene_id)
            write_jsonl(gt_path, filtered_gt)
            if vqa_path.exists():
                filtered_vqa = []
                for row in load_jsonl(vqa_path):
                    qid = str(row.get("question_id") or "").strip()
                    scene_id = str(row.get("scene_id") or "").strip()
                    if qid and qid in kept_qids:
                        filtered_vqa.append(row)
                    elif not kept_qids and (not scene_id or scene_id not in blocked_scenes):
                        filtered_vqa.append(row)
                write_jsonl(vqa_path, filtered_vqa)
        elif vqa_path.exists():
            filtered_vqa = []
            for row in load_jsonl(vqa_path):
                scene_id = str(row.get("scene_id") or "").strip()
                if scene_id and scene_id in blocked_scenes:
                    continue
                filtered_vqa.append(row)
                if scene_id:
                    kept_scenes.add(scene_id)
            write_jsonl(vqa_path, filtered_vqa)
    return kept_scenes


def _row_truth_bucket(row: dict) -> str:
    prompt_truth_tier = str(row.get("prompt_truth_tier") or "").strip().lower()
    human_visible_truth_status = str(row.get("human_visible_truth_status") or "").strip().lower()
    if prompt_truth_tier == "reject":
        return "uncertain"
    if human_visible_truth_status in {"uncertain", "contradicted"}:
        return "uncertain"
    return "truth_aligned"


def summarize_subset(subset_dir: Path) -> dict:
    question_counts_by_task: dict[str, int] = {}
    uncertain = 0
    truth_aligned = 0
    total = 0
    for task in TASKS:
        rows = load_jsonl(subset_dir / "gt" / f"gt_next_{task}.jsonl")
        if not rows:
            continue
        question_counts_by_task[task] = len(rows)
        total += len(rows)
        for row in rows:
            if _row_truth_bucket(row) == "uncertain":
                uncertain += 1
            else:
                truth_aligned += 1
    return {
        "questions_total": total,
        "question_counts_by_task": dict(sorted(question_counts_by_task.items())),
        "uncertain": uncertain,
        "truth_aligned": truth_aligned,
    }


def summarize_auxiliary_subset(subset_dir: Path) -> dict:
    question_counts_by_task: dict[str, int] = {}
    total = 0
    for task in TASKS:
        rows = load_jsonl(subset_dir / "gt" / f"gt_next_{task}.jsonl")
        if not rows:
            continue
        question_counts_by_task[task] = len(rows)
        total += len(rows)
    return {
        "questions_total": total,
        "question_counts_by_task": dict(sorted(question_counts_by_task.items())),
        "legacy_auxiliary": total,
        "uncertain": 0,
        "truth_aligned": 0,
        "protocol_family": "legacy_auxiliary_v1",
    }


def build_truth_aligned_release_manifest(
    *,
    output_root: Path,
    release_version: str,
    supersedes: list[str] | None = None,
    archived_versions: list[dict] | None = None,
    inherited_subsets: list[dict] | None = None,
    replaced_subsets: list[dict] | None = None,
    split_counts: dict[str, dict] | None = None,
    protocol_family: str = "truth_aligned_v1",
) -> dict:
    current_root = output_root / "current"
    manifest = {
        "release_version": release_version,
        "protocol_family": protocol_family,
        "is_current": True,
        "supersedes": supersedes or [],
        "archived_versions": archived_versions or [],
        "canonical_paths": {
            subset_name: str(current_root / subset_name)
            for subset_name in (
                "benchmark_truthaligned_v1",
                "val_truthaligned_v1",
                "train_truthaligned_v1",
            )
        },
        "auxiliary_paths": {
            subset_name: str(output_root / "auxiliary" / subset_name)
            for subset_name in ("train_weakaux_legacy_v1",)
        },
        "inherited_subsets": inherited_subsets or [],
        "replaced_subsets": replaced_subsets or [],
        "split_counts": split_counts or {},
    }
    current_root.mkdir(parents=True, exist_ok=True)
    with open(current_root / "release_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def build_truth_aligned_release_package(
    *,
    legacy_release_dir: Path,
    repaired_release_dir: Path,
    output_root: Path,
    release_version: str,
    supersedes: str | None = None,
) -> dict:
    output_root = output_root.resolve()
    repaired_manifest_path = repaired_release_dir / "release_manifest.json"
    if repaired_manifest_path.exists():
        with open(repaired_manifest_path, "r") as f:
            repaired_manifest = json.load(f)
        if bool(repaired_manifest.get("preview_only")):
            raise ValueError(
                f"repaired release is preview_only and cannot be canonicalized: {repaired_manifest_path}"
            )
        vqa_build_mode = str(repaired_manifest.get("vqa_build_mode") or "").strip()
        if vqa_build_mode and vqa_build_mode != "rebuild_from_gt":
            raise ValueError(
                f"repaired release uses non-canonical vqa_build_mode={vqa_build_mode}: {repaired_manifest_path}"
            )

    current_root = output_root / "current"
    auxiliary_root = output_root / "auxiliary"
    archive_root = output_root / "archive"
    current_root.mkdir(parents=True, exist_ok=True)
    auxiliary_root.mkdir(parents=True, exist_ok=True)
    archive_root.mkdir(parents=True, exist_ok=True)

    if any(current_root.iterdir()):
        raise FileExistsError(f"current release root is not empty: {current_root}")

    archived_version = supersedes or legacy_release_dir.name
    archived_version_root = archive_root / archived_version
    archived_version_root.mkdir(parents=True, exist_ok=True)

    inherited_subsets: list[dict] = []
    replaced_subsets: list[dict] = []
    split_counts: dict[str, dict] = {}

    for subset_name, legacy_split in INHERITED_AUX_SUBSETS.items():
        source_subset = legacy_release_dir / legacy_split
        target_subset = auxiliary_root / subset_name
        _copy_tree(source_subset, target_subset)
        split_counts[subset_name] = summarize_auxiliary_subset(target_subset)
        inherited_subsets.append(
            {
                "subset_name": subset_name,
                "source_split": legacy_split,
                "source_path": str(source_subset),
                "target_path": str(target_subset),
                "policy": "legacy_inherited_as_auxiliary",
            }
        )

    for subset_name, repaired_split in CANONICAL_SUBSETS.items():
        repaired_subset = repaired_release_dir / repaired_split
        current_subset = current_root / subset_name
        _copy_tree(repaired_subset, current_subset)
        _localize_subset_task_output_refs(subset_dir=current_subset)
        _rewrite_c_sidecars_into_current(subset_dir=current_subset, repaired_subset_dir=repaired_subset)

        legacy_subset = legacy_release_dir / repaired_split
        archived_subset = archived_version_root / repaired_split
        _copy_tree(legacy_subset, archived_subset)
        replaced_subsets.append(
            {
                "subset_name": subset_name,
                "legacy_split": repaired_split,
                "legacy_path": str(legacy_subset),
                "archived_path": str(archived_subset),
                "replacement_path": str(current_subset),
            }
        )

    occupied_scenes: set[str] = set()
    for subset_name in ("benchmark_truthaligned_v1", "val_truthaligned_v1", "train_truthaligned_v1"):
        current_subset = current_root / subset_name
        kept_scenes = _prune_subset_rows_by_scene(
            subset_dir=current_subset,
            blocked_scenes=occupied_scenes,
        )
        occupied_scenes.update(kept_scenes)
        split_counts[subset_name] = summarize_subset(current_subset)

    manifest = build_truth_aligned_release_manifest(
        output_root=output_root,
        release_version=release_version,
        supersedes=[archived_version],
        archived_versions=[
            {
                "version": archived_version,
                "path": str(archived_version_root),
            }
        ],
        inherited_subsets=inherited_subsets,
        replaced_subsets=replaced_subsets,
        split_counts=split_counts,
    )
    with open(output_root / "release_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build truth-aligned current/archive release layout.")
    parser.add_argument("--legacy-release-dir", required=True)
    parser.add_argument("--repaired-release-dir", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--release-version", required=True)
    parser.add_argument("--supersedes", default=None)
    args = parser.parse_args()

    manifest = build_truth_aligned_release_package(
        legacy_release_dir=Path(args.legacy_release_dir).resolve(),
        repaired_release_dir=Path(args.repaired_release_dir).resolve(),
        output_root=Path(args.output_root).resolve(),
        release_version=args.release_version,
        supersedes=args.supersedes,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
