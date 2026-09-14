#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUTPUT_DIR="${OUTPUT_DIR:-results/oracle_cot_full_v2_20260608}"
LF_DATA_DIR="${LF_DATA_DIR:-results/oracle_cot_llamafactory_data_current}"
LOG_PATH="${LOG_PATH:-results/llamafactory_runs/qwen35_4b/logs/oracle_cot_full_pipeline_watch.log}"
TARGET_ROWS="${TARGET_ROWS:-31852}"
SLEEP_S="${SLEEP_S:-600}"
PYTHON_BIN="${PYTHON_BIN:-/home/zhaoyang/.conda/envs/navbench-lf/bin/python}"
TRAIN_SESSION="${TRAIN_SESSION:-navbench_cot_full_train_seed42}"
EVAL_SESSION="${EVAL_SESSION:-navbench_cot_full_eval_seed42}"
BENCH_SESSION="${BENCH_SESSION:-navbench_cot_full_benchmark_seed42}"
ACCEPT_SESSION="${ACCEPT_SESSION:-navbench_cot_full_acceptance_seed42}"
TRAIN_DIR="${TRAIN_DIR:-results/llamafactory_runs/qwen35_4b/oracle_cot_full_seed42}"
VAL_SUMMARY="${VAL_SUMMARY:-results/llamafactory_eval/qwen35_4b/oracle_cot_full_seed42_val/summary.json}"
BENCH_SUMMARY="${BENCH_SUMMARY:-results/llamafactory_eval/qwen35_4b/oracle_cot_full_seed42_benchmark/summary.json}"
ACCEPTANCE_AUDIT="${ACCEPTANCE_AUDIT:-results/llamafactory_eval/qwen35_4b/oracle_cot_full_seed42_acceptance_audit.json}"

mkdir -p "$(dirname "${LOG_PATH}")"

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${LOG_PATH}"
}

read_progress() {
  "${PYTHON_BIN}" - "${OUTPUT_DIR}" <<'PY'
from pathlib import Path
import json, collections
import sys
base = Path(sys.argv[1])
target = {"a1": 6483, "b1": 6483, "a2": 7843, "b2": 7843, "c": 3200}
accepted = {}
all_rows = []
rejected = {}
for path in sorted((base / "parsed").glob("*/*.jsonl")):
    for line in path.read_text(errors="replace").splitlines():
        if line.strip():
            row = json.loads(line)
            all_rows.append(row)
            accepted[str(row.get("question_id"))] = row
for path in sorted((base / "rejected").glob("*/*.jsonl")):
    for line in path.read_text(errors="replace").splitlines():
        if line.strip():
            row = json.loads(line)
            rejected[str(row.get("question_id"))] = row
by_task = collections.Counter(row.get("task") for row in accepted.values())
pending = {qid: row for qid, row in rejected.items() if qid not in accepted}
qc_bad = sum(1 for row in all_rows if not (row.get("qc") or {}).get("ok"))
print(json.dumps({
    "accepted_unique": len(accepted),
    "accepted_all_rows": len(all_rows),
    "duplicates": len(all_rows) - len(accepted),
    "accepted_by_task": dict(sorted(by_task.items())),
    "remaining_by_task": {task: target[task] - by_task.get(task, 0) for task in target},
    "pending_rejected": len(pending),
    "qc_bad_rows": qc_bad,
}, ensure_ascii=True, sort_keys=True))
PY
}

json_value() {
  local payload="$1"
  local key="$2"
  "${PYTHON_BIN}" - "$payload" "$key" <<'PY'
import json, sys
payload = json.loads(sys.argv[1])
print(payload.get(sys.argv[2], ""))
PY
}

count_json_array() {
  local path="$1"
  "${PYTHON_BIN}" - "$path" <<'PY'
import json, sys
from pathlib import Path
path = Path(sys.argv[1])
if not path.exists():
    print(-1)
else:
    payload = json.loads(path.read_text())
    print(len(payload) if isinstance(payload, list) else -1)
PY
}

read_sft_status() {
  "${PYTHON_BIN}" - "${LF_DATA_DIR}" <<'PY'
import json, sys
from pathlib import Path
base = Path(sys.argv[1])

def load_rows(name):
    path = base / name
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    return payload if isinstance(payload, list) else []

direct = load_rows("navbench_direct_sft.json")
cot = load_rows("navbench_oracle_cot_sft.json")
cot_texts = [
    str(((row.get("messages") or [{}, {}])[1]).get("content", ""))
    for row in cot
    if isinstance(row, dict)
]
print(json.dumps({
    "direct_rows": len(direct),
    "cot_rows": len(cot),
    "cot_json_start_rows": sum(1 for text in cot_texts if text.lstrip().startswith("{")),
    "cot_reasoning_payload_rows": sum(1 for text in cot_texts if "reasoning_payload" in text),
    "cot_final_json_marker_rows": sum(1 for text in cot_texts if "Final JSON:" in text),
    "cot_open_think_rows": sum(1 for text in cot_texts if text.startswith("<think>")),
    "cot_close_think_rows": sum(1 for text in cot_texts if "</think>" in text),
}, ensure_ascii=True, sort_keys=True))
PY
}

materialize_full_sft() {
  local existing_cot_rows
  local existing_direct_rows
  existing_cot_rows="$(count_json_array "${LF_DATA_DIR}/navbench_oracle_cot_sft.json")"
  existing_direct_rows="$(count_json_array "${LF_DATA_DIR}/navbench_direct_sft.json")"
  if [[ "${existing_cot_rows}" == "${TARGET_ROWS}" && "${existing_direct_rows}" == "${TARGET_ROWS}" ]]; then
    log "materialize_full_sft_skip_existing cot_rows=${existing_cot_rows} direct_rows=${existing_direct_rows}"
    return 0
  fi

  log "materialize_full_sft_start"
  "${PYTHON_BIN}" scripts/run_oracle_cot_generation.py \
    --stage full \
    --tasks a1,b1,a2,b2,c \
    --output-dir "${OUTPUT_DIR}" \
    --llamafactory-data-dir "${LF_DATA_DIR}" \
    --skip-fact-packet-write

  local cot_rows
  local direct_rows
  cot_rows="$(count_json_array "${LF_DATA_DIR}/navbench_oracle_cot_sft.json")"
  direct_rows="$(count_json_array "${LF_DATA_DIR}/navbench_direct_sft.json")"
  log "materialize_full_sft_done cot_rows=${cot_rows} direct_rows=${direct_rows}"
  if [[ "${cot_rows}" != "${TARGET_ROWS}" || "${direct_rows}" != "${TARGET_ROWS}" ]]; then
    log "materialize_full_sft_bad_rows cot_rows=${cot_rows} direct_rows=${direct_rows} target=${TARGET_ROWS}"
    return 1
  fi
}

start_training_if_needed() {
  if [[ -f "${TRAIN_DIR}/adapter_model.safetensors" && -f "${TRAIN_DIR}/trainer_state.json" ]]; then
    log "train_done_exists ${TRAIN_DIR}"
    return 0
  fi
  if tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; then
    log "train_session_exists ${TRAIN_SESSION}"
    return 0
  fi
  log "train_start session=${TRAIN_SESSION}"
  tmux new-session -d -s "${TRAIN_SESSION}" \
    "cd '${ROOT_DIR}' && conda run -n navbench-lf bash scripts/run_llamafactory_navbench_sft.sh cot full 42 >> results/llamafactory_runs/qwen35_4b/logs/oracle_cot_full_seed42_train.log 2>&1"
}

start_eval_if_needed() {
  if [[ ! -f "${TRAIN_DIR}/adapter_model.safetensors" ]]; then
    log "eval_wait_checkpoint_missing"
    return 0
  fi
  if [[ ! -f "${VAL_SUMMARY}" ]]; then
    if ! tmux has-session -t "${EVAL_SESSION}" 2>/dev/null; then
      log "val_eval_start session=${EVAL_SESSION}"
      tmux new-session -d -s "${EVAL_SESSION}" \
        "cd '${ROOT_DIR}' && conda run -n navbench-lf python scripts/run_llamafactory_navbench_eval.py --run-name oracle_cot_full_seed42_val --split val --adapter-name-or-path '${ROOT_DIR}/${TRAIN_DIR}' --max-new-tokens 512 --temperature 0.0 --top-p 1.0 --resume >> results/llamafactory_eval/qwen35_4b/oracle_cot_full_seed42_val.log 2>&1"
    fi
    return 0
  fi
  if [[ ! -f "${BENCH_SUMMARY}" ]]; then
    if ! tmux has-session -t "${BENCH_SESSION}" 2>/dev/null; then
      log "benchmark_eval_start session=${BENCH_SESSION}"
      tmux new-session -d -s "${BENCH_SESSION}" \
        "cd '${ROOT_DIR}' && conda run -n navbench-lf python scripts/run_llamafactory_navbench_eval.py --run-name oracle_cot_full_seed42_benchmark --split benchmark --adapter-name-or-path '${ROOT_DIR}/${TRAIN_DIR}' --max-new-tokens 512 --temperature 0.0 --top-p 1.0 --resume >> results/llamafactory_eval/qwen35_4b/oracle_cot_full_seed42_benchmark.log 2>&1"
    fi
    return 0
  fi
  if [[ ! -f "${ACCEPTANCE_AUDIT}" ]]; then
    if ! tmux has-session -t "${ACCEPT_SESSION}" 2>/dev/null; then
      log "acceptance_audit_start session=${ACCEPT_SESSION}"
      tmux new-session -d -s "${ACCEPT_SESSION}" \
        "cd '${ROOT_DIR}' && conda run -n navbench-lf python scripts/audit_llamafactory_navbench_sft.py --output '${ACCEPTANCE_AUDIT}' >> results/llamafactory_eval/qwen35_4b/oracle_cot_full_seed42_acceptance.log 2>&1"
    fi
    return 0
  fi
  log "pipeline_complete acceptance=${ACCEPTANCE_AUDIT}"
  return 2
}

log "pipeline_watch_start output_dir=${OUTPUT_DIR} target=${TARGET_ROWS}"
while true; do
  progress="$(read_progress)"
  log "progress ${progress}"

  materialize_full_sft
  sft_status="$(read_sft_status)"
  cot_rows="$(json_value "${sft_status}" cot_rows)"
  direct_rows="$(json_value "${sft_status}" direct_rows)"
  cot_json_start_rows="$(json_value "${sft_status}" cot_json_start_rows)"
  cot_reasoning_payload_rows="$(json_value "${sft_status}" cot_reasoning_payload_rows)"
  cot_final_json_marker_rows="$(json_value "${sft_status}" cot_final_json_marker_rows)"
  cot_open_think_rows="$(json_value "${sft_status}" cot_open_think_rows)"
  cot_close_think_rows="$(json_value "${sft_status}" cot_close_think_rows)"
  log "sft_status ${sft_status}"

  if [[ "${cot_rows}" == "${TARGET_ROWS}" \
      && "${direct_rows}" == "${TARGET_ROWS}" \
      && "${cot_json_start_rows}" == "0" \
      && "${cot_reasoning_payload_rows}" == "0" \
      && "${cot_final_json_marker_rows}" == "0" \
      && "${cot_open_think_rows}" == "${TARGET_ROWS}" \
      && "${cot_close_think_rows}" == "${TARGET_ROWS}" ]]; then
    start_training_if_needed
    status=0
    start_eval_if_needed || status=$?
    if [[ "${status:-0}" == "2" ]]; then
      exit 0
    fi
  fi

  sleep "${SLEEP_S}"
done
