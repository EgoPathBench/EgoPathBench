#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CONFIG="${CONFIG:-configs/internscene_local.yaml}"
SCENES_DIR="${SCENES_DIR:-/mnt/data/zhaoyang/navbench3d-data/scenes_v3_4src_relaxed_refilter_post_objaverse_20260325_124913}"
ASSET_BASE="${ASSET_BASE:-/mnt/data/zhaoyang/navbench3d-data/internscenes_raw/asset_library}"
COMPOSED_DIR="${COMPOSED_DIR:-/mnt/data/zhaoyang/navbench3d-data/composed}"
RUN_ROOT="${RUN_ROOT:-/mnt/data/zhaoyang/navbench3d-data/tmp/next_release_full_8219_$(date +%Y%m%d_%H%M%S)}"
COMPOSE_WORKERS="${COMPOSE_WORKERS:-32}"
GPU_IDS_CSV="${GPU_IDS_CSV:-0,1}"
RENDER_PROCS_PER_GPU="${RENDER_PROCS_PER_GPU:-2}"
BLENDER_BIN="${BLENDER_BIN:-/usr/local/bin/blender}"
NAVBENCH_PY_SITE="${NAVBENCH_PY_SITE:-/mnt/data/zhaoyang/navbench3d-data/py311_site}"

mkdir -p "${RUN_ROOT}" "${RUN_ROOT}/logs" "${RUN_ROOT}/lists"

IFS=',' read -r -a GPU_IDS <<< "${GPU_IDS_CSV}"
TOTAL_RENDER_WORKERS=$(( ${#GPU_IDS[@]} * RENDER_PROCS_PER_GPU ))
if [[ "${TOTAL_RENDER_WORKERS}" -le 0 ]]; then
  echo "TOTAL_RENDER_WORKERS must be > 0" >&2
  exit 1
fi

if [[ -z "${NAVBENCH_PY_SITE}" ]]; then
  echo "NAVBENCH_PY_SITE must not be empty" >&2
  exit 1
fi
if [[ ! -d "${NAVBENCH_PY_SITE}" ]]; then
  echo "NAVBENCH_PY_SITE does not exist: ${NAVBENCH_PY_SITE}" >&2
  exit 1
fi

echo "[1/8] Build split manifest -> ${RUN_ROOT}/splits"
conda run -n navbench3d python scripts/build_next_dataset_splits.py \
  --scenes-dir "${SCENES_DIR}" \
  --output-dir "${RUN_ROOT}/splits" \
  --train-ratio 0.8 \
  --val-ratio 0.1 \
  --benchmark-ratio 0.1

echo "[2/8] Compose all scenes into ${COMPOSED_DIR}"
conda run -n navbench3d python scripts/compose_scene.py \
  --scenes-dir "${SCENES_DIR}" \
  --asset-base "${ASSET_BASE}" \
  --output-dir "${COMPOSED_DIR}" \
  --workers "${COMPOSE_WORKERS}" \
  2>&1 | tee "${RUN_ROOT}/logs/compose.log"

echo "[3/8] Build render shard lists (${TOTAL_RENDER_WORKERS} workers)"
python - <<'PY' "${SCENES_DIR}" "${RUN_ROOT}/lists" "${TOTAL_RENDER_WORKERS}" "${RUN_ROOT}/renders"
from pathlib import Path
import json
import sys

scenes_dir = Path(sys.argv[1])
lists_dir = Path(sys.argv[2])
worker_count = int(sys.argv[3])
renders_dir = Path(sys.argv[4])
lists_dir.mkdir(parents=True, exist_ok=True)

all_scene_ids = sorted(
    p.name for p in scenes_dir.iterdir()
    if p.is_dir() and (p / "layout.json").exists()
)

def is_render_complete(scene_id: str) -> bool:
    cameras_path = renders_dir / scene_id / "cameras.json"
    if not cameras_path.exists():
        return False
    try:
        payload = json.loads(cameras_path.read_text())
    except Exception:
        return False
    return isinstance(payload, list)

scene_ids = [scene_id for scene_id in all_scene_ids if not is_render_complete(scene_id)]
shards = [[] for _ in range(worker_count)]
for idx, scene_id in enumerate(scene_ids):
    shards[idx % worker_count].append(scene_id)
for worker_idx, shard in enumerate(shards):
    with open(lists_dir / f"render_shard_{worker_idx:02d}.txt", "w") as f:
        for scene_id in shard:
            f.write(scene_id + "\n")
print({
    "scene_count_total": len(all_scene_ids),
    "scene_count_pending": len(scene_ids),
    "scene_count_completed": len(all_scene_ids) - len(scene_ids),
    "worker_count": worker_count,
})
PY

echo "[4/8] Render all scenes with pre-composed GLBs"
render_pids=()
for worker_idx in $(seq 0 $(( TOTAL_RENDER_WORKERS - 1 ))); do
  gpu_slot=$(( worker_idx % ${#GPU_IDS[@]} ))
  gpu_id="${GPU_IDS[${gpu_slot}]}"
  shard_path="${RUN_ROOT}/lists/render_shard_$(printf '%02d' "${worker_idx}").txt"
  log_path="${RUN_ROOT}/logs/render_worker_$(printf '%02d' "${worker_idx}").log"
  (
    while IFS= read -r scene_id; do
      [[ -z "${scene_id}" ]] && continue
      CUDA_VISIBLE_DEVICES="${gpu_id}" NAVBENCH_PY_SITE="${NAVBENCH_PY_SITE}" "${BLENDER_BIN}" \
        --background \
        --python-use-system-env \
        --python-exit-code 11 \
        --python scripts/render_scenes.py -- \
        --config "${CONFIG}" \
        --scenes-dir "${SCENES_DIR}" \
        --output-dir "${RUN_ROOT}/renders" \
        --scene-id "${scene_id}" \
        --composed-dir "${COMPOSED_DIR}" \
        --funnel-audit-out "${RUN_ROOT}/funnel_audit"
    done < "${shard_path}"
  ) > "${log_path}" 2>&1 &
  render_pids+=("$!")
done

for pid in "${render_pids[@]}"; do
  wait "${pid}"
done

echo "[5/8] Build task outputs"
conda run -n navbench3d python scripts/build_task_outputs.py \
  --config "${CONFIG}" \
  --scenes-dir "${SCENES_DIR}" \
  --renders-dir "${RUN_ROOT}/renders" \
  --output-dir "${RUN_ROOT}/task_outputs" \
  2>&1 | tee "${RUN_ROOT}/logs/task_outputs.log"

echo "[6/8] Build GT"
conda run -n navbench3d python scripts/build_navigation_gt_next.py \
  --config "${CONFIG}" \
  --scenes-dir "${SCENES_DIR}" \
  --renders-dir "${RUN_ROOT}/renders" \
  --task-outputs-dir "${RUN_ROOT}/task_outputs" \
  --output-dir "${RUN_ROOT}/gt" \
  2>&1 | tee "${RUN_ROOT}/logs/gt.log"

echo "[7/8] Build VQA"
conda run -n navbench3d python scripts/build_vqa_questions_next.py \
  --gt-dir "${RUN_ROOT}/gt" \
  --output-dir "${RUN_ROOT}/vqa" \
  2>&1 | tee "${RUN_ROOT}/logs/vqa.log"

echo "[8/8] Export train/val/benchmark release datasets"
conda run -n navbench3d python scripts/build_next_release_datasets.py \
  --gt-dir "${RUN_ROOT}/gt" \
  --vqa-dir "${RUN_ROOT}/vqa" \
  --split-manifest "${RUN_ROOT}/splits/split_manifest.json" \
  --output-dir "${RUN_ROOT}/release" \
  2>&1 | tee "${RUN_ROOT}/logs/release.log"

echo "Full release pipeline finished: ${RUN_ROOT}"
