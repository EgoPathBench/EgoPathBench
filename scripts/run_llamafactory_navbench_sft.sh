#!/usr/bin/env bash
set -euo pipefail

VARIANT="${1:-}"
PHASE="${2:-}"
SEED="${3:-42}"

if [[ -z "${VARIANT}" || -z "${PHASE}" ]]; then
  echo "Usage: $0 {direct|cot} {smoke|sanity500|gate10pct|full} [seed]" >&2
  exit 2
fi

REPO_ROOT="${NAVBENCH_REPO_ROOT:-/home/zhaoyang/Projects/navbench3d-code}"
LLAMAFACTORY_ROOT="${LLAMAFACTORY_ROOT:-/mnt/data/wuchanglin/LLaMA-Factory}"
CONFIG_ROOT="${REPO_ROOT}/configs/llamafactory"

case "${VARIANT}:${PHASE}" in
  direct:smoke)
    CONFIG="${CONFIG_ROOT}/navbench_qwen35_4b_direct_smoke_lora.yaml"
    ;;
  cot:smoke)
    CONFIG="${CONFIG_ROOT}/navbench_qwen35_4b_oracle_cot_smoke_lora.yaml"
    ;;
  direct:sanity500)
    CONFIG="${CONFIG_ROOT}/navbench_qwen35_4b_direct_sanity500_lora.yaml"
    ;;
  cot:sanity500)
    CONFIG="${CONFIG_ROOT}/navbench_qwen35_4b_oracle_cot_sanity500_lora.yaml"
    ;;
  direct:gate10pct)
    CONFIG="${CONFIG_ROOT}/navbench_qwen35_4b_direct_gate10pct_lora.yaml"
    ;;
  cot:gate10pct)
    CONFIG="${CONFIG_ROOT}/navbench_qwen35_4b_oracle_cot_gate10pct_lora.yaml"
    ;;
  direct:full)
    CONFIG="${CONFIG_ROOT}/navbench_qwen35_4b_direct_full_lora.yaml"
    ;;
  cot:full)
    CONFIG="${CONFIG_ROOT}/navbench_qwen35_4b_oracle_cot_full_lora.yaml"
    ;;
  *)
    echo "Unknown variant/phase: ${VARIANT}/${PHASE}" >&2
    exit 2
    ;;
esac

if [[ ! -f "${CONFIG}" ]]; then
  echo "Missing config: ${CONFIG}" >&2
  exit 2
fi
if [[ ! -d "${LLAMAFACTORY_ROOT}" ]]; then
  echo "Missing LLaMA-Factory root: ${LLAMAFACTORY_ROOT}" >&2
  exit 2
fi

TMP_CONFIG="$(mktemp "/tmp/navbench_lf_${VARIANT}_${PHASE}_${SEED}_XXXX.yaml")"
sed \
  -e "s/seed: 42/seed: ${SEED}/" \
  -e "s/_seed42/_seed${SEED}/g" \
  "${CONFIG}" > "${TMP_CONFIG}"

if [[ -n "${NAVBENCH_RESUME_FROM_CHECKPOINT:-}" ]]; then
  if [[ ! -d "${NAVBENCH_RESUME_FROM_CHECKPOINT}" ]]; then
    echo "Missing resume checkpoint: ${NAVBENCH_RESUME_FROM_CHECKPOINT}" >&2
    exit 2
  fi
  sed -i \
    -e "s|^resume_from_checkpoint:.*|resume_from_checkpoint: ${NAVBENCH_RESUME_FROM_CHECKPOINT}|" \
    "${TMP_CONFIG}"
fi

export PYTHONPATH="${REPO_ROOT}/tools/llamafactory_compat:${LLAMAFACTORY_ROOT}/src:${PYTHONPATH:-}"
export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/navbench3d_mplconfig}"
mkdir -p "${MPLCONFIGDIR}"
cd "${LLAMAFACTORY_ROOT}"

if command -v llamafactory-cli >/dev/null 2>&1; then
  FORCE_TORCHRUN=1 llamafactory-cli train "${TMP_CONFIG}"
else
  FORCE_TORCHRUN=1 python -m llamafactory.cli train "${TMP_CONFIG}"
fi
