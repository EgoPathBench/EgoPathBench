#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/zhaoyang/.conda/envs/navbench-lf/bin/python}"
LLAMAFACTORY_ROOT="${LLAMAFACTORY_ROOT:-/mnt/data/wuchanglin/LLaMA-Factory}"
BASE_RUN_DIR="${BASE_RUN_DIR:-results/llamafactory_runs/qwen35_4b/v4seg_full_new_gpt_cot_lora_seed42}"
CONT_RUN_DIR="${CONT_RUN_DIR:-results/llamafactory_runs/qwen35_4b/v4seg_full_new_gpt_cot_lora_seed42_cont4}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/llamafactory/navbench_qwen35_4b_v4seg_full_new_gpt_cot_lora_cont4.yaml}"
LOG_DIR="${LOG_DIR:-results/llamafactory_runs/qwen35_4b/logs}"
EVAL_LOG_DIR="${EVAL_LOG_DIR:-results/llamafactory_eval/qwen35_4b/logs}"
SEARCH_REPORT="${SEARCH_REPORT:-results/llamafactory_eval/qwen35_4b/v4seg_full_new_gpt_cot_lora_ckpt_search_report.json}"
DIRECT_SUMMARY="${DIRECT_SUMMARY:-results/llamafactory_eval/qwen35_4b/direct_full_seed42_benchmark_max8192/summary.json}"
TARGET_PLANNING_SPL="${TARGET_PLANNING_SPL:-0.0324}"

mkdir -p "${LOG_DIR}" "${EVAL_LOG_DIR}"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

run_training_if_needed() {
  if [[ -f "${CONT_RUN_DIR}/adapter_model.safetensors" && -f "${CONT_RUN_DIR}/trainer_state.json" ]]; then
    log "training_skip_existing cont_run_dir=${CONT_RUN_DIR}"
    return 0
  fi

  log "training_start config=${TRAIN_CONFIG}"
  (
    cd "${LLAMAFACTORY_ROOT}"
    export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
    export PYTHONPATH="${ROOT_DIR}/tools/llamafactory_compat:${LLAMAFACTORY_ROOT}/src:${PYTHONPATH:-}"
    export DISABLE_VERSION_CHECK="${DISABLE_VERSION_CHECK:-1}"
    export PYTHONNOUSERSITE=1
    export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/navbench3d_mplconfig}"
    mkdir -p "${MPLCONFIGDIR}"
    FORCE_TORCHRUN=1 NNODES=1 NPROC_PER_NODE=2 \
      "${PYTHON_BIN}" -m llamafactory.cli train "${ROOT_DIR}/${TRAIN_CONFIG}"
  ) 2>&1 | tee -a "${ROOT_DIR}/${LOG_DIR}/v4seg_full_new_gpt_cot_lora_seed42_cont4_train.log"
  log "training_done cont_run_dir=${CONT_RUN_DIR}"
}

checkpoint_step() {
  local path="$1"
  basename "${path}" | sed -E 's/^checkpoint-//'
}

candidate_checkpoints() {
  find "${BASE_RUN_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V
  find "${CONT_RUN_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V
  if [[ -f "${CONT_RUN_DIR}/adapter_model.safetensors" ]]; then
    printf '%s\n' "${CONT_RUN_DIR}"
  fi
}

run_eval_for_checkpoint() {
  local ckpt="$1"
  local gpu="$2"
  local tag
  local run_name
  if [[ "${ckpt}" == "${CONT_RUN_DIR}" ]]; then
    tag="cont4_final"
  elif [[ "${ckpt}" == "${BASE_RUN_DIR}"/checkpoint-* ]]; then
    tag="base2_checkpoint_$(checkpoint_step "${ckpt}")"
  else
    tag="cont4_checkpoint_$(checkpoint_step "${ckpt}")"
  fi
  run_name="v4seg_full_new_gpt_cot_lora_${tag}_benchmark_max8192"
  log "eval_start gpu=${gpu} run_name=${run_name} ckpt=${ckpt}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" scripts/run_llamafactory_navbench_eval.py \
    --run-name "${run_name}" \
    --split benchmark \
    --adapter-name-or-path "${ROOT_DIR}/${ckpt}" \
    --max-new-tokens 8192 \
    --temperature 0.0 \
    --top-p 1.0 \
    --resume \
    2>&1 | tee -a "${EVAL_LOG_DIR}/${run_name}.log"
  "${PYTHON_BIN}" scripts/postprocess_llamafactory_cot_predictions.py \
    --source-run-name "${run_name}" \
    --output-run-name "${run_name}" \
    --split benchmark \
    --overwrite \
    2>&1 | tee -a "${EVAL_LOG_DIR}/${run_name}_postprocess.log"
  log "eval_done run_name=${run_name}"
}

run_evals() {
  mapfile -t ckpts < <(candidate_checkpoints)
  local idx=0
  for ckpt in "${ckpts[@]}"; do
    [[ -d "${ckpt}" ]] || continue
    run_eval_for_checkpoint "${ckpt}" "$((idx % 2))" &
    idx=$((idx + 1))
    if (( idx % 2 == 0 )); then
      wait
    fi
  done
  wait
}

write_report() {
  "${PYTHON_BIN}" - "${DIRECT_SUMMARY}" "${TARGET_PLANNING_SPL}" "${SEARCH_REPORT}" <<'PY'
import json
import sys
from pathlib import Path

direct_summary = Path(sys.argv[1])
target = float(sys.argv[2])
report_path = Path(sys.argv[3])
root = Path("results/llamafactory_eval/qwen35_4b")

records = []
for summary_path in sorted(root.glob("v4seg_full_new_gpt_cot_lora_*_benchmark_max8192/summary_final_json_extracted.json")):
    data = json.loads(summary_path.read_text())
    aggregate = data.get("aggregate", {})
    planning = aggregate.get("planning_mean", {})
    classification = aggregate.get("classification_mean", {})
    run_name = summary_path.parent.name
    records.append({
        "run_name": run_name,
        "summary": str(summary_path),
        "planning_spl_all": planning.get("spl_all"),
        "planning_success_rate": planning.get("success_rate"),
        "planning_valid_path_rate": planning.get("valid_path_rate"),
        "classification_f1": classification.get("f1"),
        "classification_balanced_accuracy": classification.get("balanced_accuracy"),
        "beats_direct_planning_spl": (
            planning.get("spl_all") is not None and float(planning["spl_all"]) > target
        ),
    })

records.sort(key=lambda row: (row["planning_spl_all"] is not None, row["planning_spl_all"] or -1), reverse=True)
direct = json.loads(direct_summary.read_text()) if direct_summary.exists() else {}
payload = {
    "direct_summary": str(direct_summary),
    "direct_planning_mean": direct.get("aggregate", {}).get("planning_mean", {}),
    "target_planning_spl_all": target,
    "best": records[0] if records else None,
    "records": records,
}
report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True))
PY
}

run_training_if_needed
run_evals
write_report
