#!/usr/bin/env bash
# Run compose + render smoke set for InternScenes real2sim.
#
# Example:
#   bash scripts/run_internscene_smoke.sh \
#     --config configs/internscene_local.yaml \
#     --data-root /mnt/data/zhaoyang/navbench3d-data \
#     --smoke-list doc/internscene_bootstrap/smoke_scenes.txt \
#     --limit 20

set -euo pipefail

CONFIG="configs/internscene_local.yaml"
DATA_ROOT="/mnt/data/zhaoyang/navbench3d-data"
SMOKE_LIST="doc/internscene_bootstrap/smoke_scenes.txt"
LIMIT=20
BLENDER_BIN="${BLENDER:-}"
RENDER_OUT=""
SCENES_DIR=""
SKIP_COMPOSE=0
SKIP_RENDER=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --smoke-list) SMOKE_LIST="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --scenes-dir) SCENES_DIR="$2"; shift 2 ;;
    --blender) BLENDER_BIN="$2"; shift 2 ;;
    --render-out) RENDER_OUT="$2"; shift 2 ;;
    --skip-compose) SKIP_COMPOSE=1; shift 1 ;;
    --skip-render) SKIP_RENDER=1; shift 1 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

if [[ -z "${SCENES_DIR}" ]]; then
  SCENES_DIR="${DATA_ROOT}/scenes"
fi
ASSET_BASE="${DATA_ROOT}/internscenes_raw/asset_library"
COMPOSED_DIR="${DATA_ROOT}/composed"
if [[ -z "${RENDER_OUT}" ]]; then
  RENDER_OUT="${DATA_ROOT}/renders_smoke_next"
fi

mkdir -p "${COMPOSED_DIR}" "${RENDER_OUT}"

if [[ ! -f "${SMOKE_LIST}" ]]; then
  echo "Smoke list not found: ${SMOKE_LIST}"
  exit 1
fi

if [[ ! -d "${ASSET_BASE}" ]]; then
  echo "Asset base missing: ${ASSET_BASE}"
  exit 1
fi

mapfile -t SCENES < <(grep -v '^\s*$' "${SMOKE_LIST}" | head -n "${LIMIT}")
echo "Smoke scenes: ${#SCENES[@]}"

if [[ "${SKIP_COMPOSE}" -eq 0 ]]; then
  for sid in "${SCENES[@]}"; do
    scene_dir="${SCENES_DIR}/${sid}"
    out_glb="${COMPOSED_DIR}/${sid}.glb"
    if [[ ! -d "${scene_dir}" ]]; then
      echo "[compose] SKIP ${sid} (scene missing)"
      continue
    fi
    if [[ -f "${out_glb}" ]]; then
      echo "[compose] SKIP ${sid} (already exists)"
      continue
    fi
    echo "[compose] ${sid}"
    python scripts/compose_scene.py \
      --scene-dir "${scene_dir}" \
      --asset-base "${ASSET_BASE}" \
      --output "${out_glb}"
  done
else
  echo "Skip compose stage (using direct asset load)."
fi

if [[ -z "${BLENDER_BIN}" ]]; then
  echo "BLENDER is not set. Skipping render stage."
  echo "Set BLENDER env or pass --blender /path/to/blender to enable rendering."
  exit 0
fi

if [[ "${SKIP_RENDER}" -eq 1 ]]; then
  echo "Skip render stage (compose-only run)."
  exit 0
fi

for sid in "${SCENES[@]}"; do
  echo "[render] ${sid}"
  if [[ "${SKIP_COMPOSE}" -eq 1 ]]; then
    "${BLENDER_BIN}" --background --python-use-system-env \
      --python scripts/render_scenes.py \
      -- \
      --config "${CONFIG}" \
      --scenes-dir "${SCENES_DIR}" \
      --output-dir "${RENDER_OUT}" \
      --scene-id "${sid}"
  else
    "${BLENDER_BIN}" --background --python-use-system-env \
      --python scripts/render_scenes.py \
      -- \
      --config "${CONFIG}" \
      --scenes-dir "${SCENES_DIR}" \
      --output-dir "${RENDER_OUT}" \
      --composed-dir "${COMPOSED_DIR}" \
      --scene-id "${sid}"
  fi
done

echo "Smoke run completed."
