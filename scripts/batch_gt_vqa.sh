#!/bin/bash
# NavBench3D: Batch GT + VQA generation
# Runs GT generation first, then VQA generation, with logging and progress tracking.
# Usage: bash scripts/batch_gt_vqa.sh 2>&1 | tee /datadisk/NavBench3D/logs/batch_gt_vqa.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIG="configs/default.yaml"
SCENES_DIR="/datadisk/NavBench3D/scenes"
RENDERS_DIR="/datadisk/NavBench3D/renders"
GT_DIR="/datadisk/NavBench3D/gt_paths"
VQA_DIR="/datadisk/NavBench3D/vqa_questions"
LOG_DIR="/datadisk/NavBench3D/logs"

mkdir -p "$LOG_DIR"

echo "=========================================="
echo "NavBench3D Batch GT + VQA Generation"
echo "Started: $(date)"
echo "=========================================="

# ---- Phase 1: GT Generation ----
echo ""
echo "=== Phase 1: Navigation GT Generation ==="
echo "Start: $(date)"
GT_START=$(date +%s)

python3 scripts/build_navigation_gt.py \
    --config "$CONFIG" \
    --scenes-dir "$SCENES_DIR" \
    --renders-dir "$RENDERS_DIR" \
    --output-dir "$GT_DIR" \
    2>&1 | tee "$LOG_DIR/gt_generation.log"

GT_END=$(date +%s)
GT_ELAPSED=$(( GT_END - GT_START ))
GT_MIN=$(( GT_ELAPSED / 60 ))
GT_SEC=$(( GT_ELAPSED % 60 ))
echo ""
echo "GT Generation completed in ${GT_MIN}m ${GT_SEC}s"

# Count GT results
GT_COUNT=$(find "$GT_DIR" -name "navigation_gt.json" | wc -l)
PAIR_COUNT=$(python3 -c "
import json, os
total = 0
for sid in os.listdir('$GT_DIR'):
    f = os.path.join('$GT_DIR', sid, 'navigation_gt.json')
    if os.path.exists(f):
        total += len(json.load(open(f)).get('navigation_pairs', []))
print(total)
")
echo "GT Results: $GT_COUNT scenes, $PAIR_COUNT navigation pairs"

# ---- Phase 2: VQA Generation ----
echo ""
echo "=== Phase 2: VQA Question Generation ==="
echo "Start: $(date)"
VQA_START=$(date +%s)

python3 scripts/build_vqa_questions.py \
    --config "$CONFIG" \
    --scenes-dir "$SCENES_DIR" \
    --renders-dir "$RENDERS_DIR" \
    --gt-dir "$GT_DIR" \
    --output-dir "$VQA_DIR" \
    2>&1 | tee "$LOG_DIR/vqa_generation.log"

VQA_END=$(date +%s)
VQA_ELAPSED=$(( VQA_END - VQA_START ))
VQA_MIN=$(( VQA_ELAPSED / 60 ))
VQA_SEC=$(( VQA_ELAPSED % 60 ))
echo ""
echo "VQA Generation completed in ${VQA_MIN}m ${VQA_SEC}s"

# Count VQA results
VQA_COUNT=$(find "$VQA_DIR" -name "questions.json" | wc -l)
Q_COUNT=$(python3 -c "
import json, os
total = 0
for sid in os.listdir('$VQA_DIR'):
    f = os.path.join('$VQA_DIR', sid, 'questions.json')
    if os.path.exists(f):
        total += len(json.load(open(f)))
print(total)
")
echo "VQA Results: $VQA_COUNT scenes, $Q_COUNT questions"

# ---- Summary ----
TOTAL_ELAPSED=$(( VQA_END - GT_START ))
TOTAL_MIN=$(( TOTAL_ELAPSED / 60 ))
TOTAL_SEC=$(( TOTAL_ELAPSED % 60 ))

echo ""
echo "=========================================="
echo "=== Batch GT + VQA Complete ==="
echo "GT:  $GT_COUNT scenes, $PAIR_COUNT pairs (${GT_MIN}m ${GT_SEC}s)"
echo "VQA: $VQA_COUNT scenes, $Q_COUNT questions (${VQA_MIN}m ${VQA_SEC}s)"
echo "Total time: ${TOTAL_MIN}m ${TOTAL_SEC}s"
echo "Finished: $(date)"
echo "=========================================="
