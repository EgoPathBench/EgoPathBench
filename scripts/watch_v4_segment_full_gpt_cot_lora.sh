#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUTPUT_DIR="${OUTPUT_DIR:-results/oracle_cot_gpt_text_v4_segment_full_20260616}"
LF_DATA_DIR="${LF_DATA_DIR:-results/lf_data_v4_segment_full_20260616}"
LOG_PATH="${LOG_PATH:-results/llamafactory_runs/qwen35_4b/logs/v4seg_full_gpt_cot_lora_pipeline.log}"
TARGET_ROWS="${TARGET_ROWS:-31852}"
SLEEP_S="${SLEEP_S:-300}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhaoyang/.conda/envs/navbench-lf/bin/python}"
GEN_SESSION_PREFIX="${GEN_SESSION_PREFIX:-navbench_v4seg_full_gpt_cot}"
TRAIN_SESSION="${TRAIN_SESSION:-navbench_v4seg_full_gpt_cot_lora_train}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/llamafactory/navbench_qwen35_4b_v4seg_full_new_gpt_cot_lora.yaml}"
TRAIN_DIR="${TRAIN_DIR:-results/llamafactory_runs/qwen35_4b/v4seg_full_new_gpt_cot_lora_seed42}"
TRAIN_LOG="${TRAIN_LOG:-results/llamafactory_runs/qwen35_4b/logs/v4seg_full_new_gpt_cot_lora_seed42_train.log}"
SHARD_COUNT="${SHARD_COUNT:-10}"

mkdir -p "$(dirname "${LOG_PATH}")" "$(dirname "${TRAIN_LOG}")"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOG_PATH}"
}

count_jsonl_rows() {
  local dir="$1"
  if [[ ! -d "${dir}" ]]; then
    printf '0'
    return 0
  fi
  find "${dir}" -name '*.jsonl' -print0 2>/dev/null | xargs -0 cat 2>/dev/null | wc -l | tr -d ' '
}

read_progress() {
  "${PYTHON_BIN}" - "${OUTPUT_DIR}" <<'PY'
from pathlib import Path
import collections
import json
import sys

base = Path(sys.argv[1])
target = {"a1": 6483, "b1": 6483, "a2": 7843, "b2": 7843, "c": 3200}
seen = {}
raw_count = 0
rejected_count = 0
for path in sorted((base / "accepted").glob("*/*.jsonl")):
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        qid = str(row.get("question_id") or "")
        if qid:
            seen[qid] = row
for path in sorted((base / "raw_responses").glob("*/*.jsonl")):
    raw_count += sum(1 for line in path.read_text(errors="replace").splitlines() if line.strip())
for path in sorted((base / "rejected").glob("*/*.jsonl")):
    rejected_count += sum(1 for line in path.read_text(errors="replace").splitlines() if line.strip())
by_task = collections.Counter(str(row.get("task")) for row in seen.values())
remaining = {task: target[task] - by_task.get(task, 0) for task in target}
print(json.dumps({
    "accepted_unique": len(seen),
    "accepted_by_task": dict(sorted(by_task.items())),
    "remaining_by_task": remaining,
    "raw_rows": raw_count,
    "rejected_rows": rejected_count,
}, ensure_ascii=True, sort_keys=True))
PY
}

json_value() {
  local payload="$1"
  local key="$2"
  "${PYTHON_BIN}" - "$payload" "$key" <<'PY'
import json
import sys
payload = json.loads(sys.argv[1])
print(payload.get(sys.argv[2], ""))
PY
}

generation_sessions_running() {
  local count
  count="$(tmux ls 2>/dev/null | grep -c "^${GEN_SESSION_PREFIX}_s" || true)"
  [[ "${count}" -gt 0 ]]
}

restart_generation_if_needed() {
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    log "generation_restart_skip OPENAI_API_KEY_missing"
    return 0
  fi
  if generation_sessions_running; then
    return 0
  fi
  log "generation_restart_start prefix=${GEN_SESSION_PREFIX}"
  OUTPUT_DIR="${OUTPUT_DIR}" \
  LOG_DIR="${OUTPUT_DIR}/logs_resume_$(date -u +%Y%m%d_%H%M%S)" \
  SESSION_PREFIX="${GEN_SESSION_PREFIX}" \
  SHARD_COUNT="${SHARD_COUNT}" \
  MAX_PER_TASK=0 \
  MAX_COMPLETION_TOKENS="${MAX_COMPLETION_TOKENS:-1800}" \
  TEMPERATURE=0.0 \
  TOP_P=1.0 \
  API_BASE="${OPENAI_BASE_URL:-http://43.153.119.62:8317/v1}" \
  MODEL="${MODEL:-gpt-5.5}" \
  TIMEOUT_S="${TIMEOUT_S:-180}" \
  MAX_RETRIES="${MAX_RETRIES:-2}" \
  RETRY_SLEEP_S="${RETRY_SLEEP_S:-10}" \
  RETRY_JITTER_S="${RETRY_JITTER_S:-3}" \
  PROGRESS_EVERY=25 \
    bash scripts/run_gpt_oracle_cot_text_full_worker_grid.sh >> "${LOG_PATH}" 2>&1
}

repair_and_export_if_ready() {
  local accepted_rows
  local repaired_rows
  local cot_rows
  accepted_rows="$(count_jsonl_rows "${OUTPUT_DIR}/accepted")"
  repaired_rows="$(count_jsonl_rows "${OUTPUT_DIR}/accepted_format_repaired")"
  cot_rows="0"
  if [[ -f "${LF_DATA_DIR}/export_report.json" ]]; then
    cot_rows="$("${PYTHON_BIN}" - "${LF_DATA_DIR}/export_report.json" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
payload = json.loads(path.read_text()) if path.exists() else {}
print(payload.get("cot_rows", 0))
PY
)"
  fi

  if [[ "${cot_rows}" == "${TARGET_ROWS}" ]]; then
    log "export_skip_existing cot_rows=${cot_rows}"
    return 0
  fi
  if [[ "${accepted_rows}" != "${TARGET_ROWS}" ]]; then
    log "export_wait accepted_rows=${accepted_rows} target=${TARGET_ROWS}"
    return 1
  fi

  log "format_repair_start accepted_rows=${accepted_rows}"
  "${PYTHON_BIN}" scripts/repair_gpt_oracle_cot_text_format.py \
    --input-dir "${OUTPUT_DIR}/accepted" \
    --output-dir "${OUTPUT_DIR}/accepted_format_repaired" \
    --report-path "${OUTPUT_DIR}/accepted_format_repair_report.json" \
    --failed-path "${OUTPUT_DIR}/accepted_format_repair_failed.jsonl" \
    --overwrite >> "${LOG_PATH}" 2>&1
  repaired_rows="$(count_jsonl_rows "${OUTPUT_DIR}/accepted_format_repaired")"
  log "format_repair_done repaired_rows=${repaired_rows}"
  if [[ "${repaired_rows}" != "${TARGET_ROWS}" ]]; then
    log "format_repair_bad_rows repaired_rows=${repaired_rows} target=${TARGET_ROWS}"
    return 1
  fi

  log "export_lf_start"
  "${PYTHON_BIN}" scripts/export_v4_segment_full_lf_data.py \
    --cot-dir "${OUTPUT_DIR}/accepted_format_repaired" \
    --output-dir "${LF_DATA_DIR}" \
    --require-full \
    --expected-rows "${TARGET_ROWS}" >> "${LOG_PATH}" 2>&1
  cot_rows="$("${PYTHON_BIN}" - "${LF_DATA_DIR}/export_report.json" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
print(payload.get("cot_rows", 0))
PY
)"
  log "export_lf_done cot_rows=${cot_rows}"
  [[ "${cot_rows}" == "${TARGET_ROWS}" ]]
}

start_training_if_ready() {
  if [[ ! -f "${LF_DATA_DIR}/export_report.json" ]]; then
    return 0
  fi
  local cot_rows
  cot_rows="$("${PYTHON_BIN}" - "${LF_DATA_DIR}/export_report.json" <<'PY'
import json, sys
from pathlib import Path
payload = json.loads(Path(sys.argv[1]).read_text())
print(payload.get("cot_rows", 0))
PY
)"
  if [[ "${cot_rows}" != "${TARGET_ROWS}" ]]; then
    log "train_wait cot_rows=${cot_rows} target=${TARGET_ROWS}"
    return 0
  fi
  if [[ -f "${TRAIN_DIR}/adapter_model.safetensors" && -f "${TRAIN_DIR}/trainer_state.json" ]]; then
    log "train_done_exists train_dir=${TRAIN_DIR}"
    return 2
  fi
  if tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; then
    log "train_session_exists session=${TRAIN_SESSION}"
    return 0
  fi
  log "train_start session=${TRAIN_SESSION} config=${TRAIN_CONFIG}"
  tmux new-session -d -s "${TRAIN_SESSION}" \
    "cd '${ROOT_DIR}' && export PATH=/home/zhaoyang/.conda/envs/navbench-lf/bin:\$PATH && export PYTHONPATH='${ROOT_DIR}/tools/llamafactory_compat:/mnt/data/wuchanglin/LLaMA-Factory/src:'\"\${PYTHONPATH:-}\" && export DISABLE_VERSION_CHECK=1 && export PYTHONNOUSERSITE=1 && export CUDA_VISIBLE_DEVICES='\${CUDA_VISIBLE_DEVICES:-0,1}' && cd /mnt/data/wuchanglin/LLaMA-Factory && FORCE_TORCHRUN=1 python -m llamafactory.cli train '${ROOT_DIR}/${TRAIN_CONFIG}' >> '${ROOT_DIR}/${TRAIN_LOG}' 2>&1"
  return 0
}

log "watch_start output_dir=${OUTPUT_DIR} lf_data_dir=${LF_DATA_DIR} target=${TARGET_ROWS}"
while true; do
  progress="$(read_progress)"
  accepted_unique="$(json_value "${progress}" accepted_unique)"
  log "progress ${progress}"
  if [[ "${accepted_unique}" != "${TARGET_ROWS}" ]]; then
    restart_generation_if_needed
  fi
  repair_and_export_if_ready || true
  status=0
  start_training_if_ready || status=$?
  if [[ "${status}" == "2" ]]; then
    log "watch_complete train_dir=${TRAIN_DIR}"
    exit 0
  fi
  sleep "${SLEEP_S}"
done
