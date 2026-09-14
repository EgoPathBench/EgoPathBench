#!/bin/bash
# =============================================================
# NavBench3D - Batch AI Re-rendering with FLUX.2-dev
# =============================================================
#
# Renders all valid scenes (~2922 views) using FLUX.2-dev (32B).
# Estimated time: ~73 hours (90s/view × 2922 views).
#
# Usage:
#   nohup bash scripts/batch_ai_rerender.sh > logs/ai_rerender.log 2>&1 &
#
# Monitor:
#   tail -f logs/ai_rerender.log
#   grep -c "PASS\|FAIL" logs/ai_rerender.log   # count completed views
#   grep "ERROR" logs/ai_rerender.log            # check for errors
#
# Resume after interruption (safe — skips already processed scenes):
#   nohup bash scripts/batch_ai_rerender.sh > logs/ai_rerender_resume.log 2>&1 &
#
# =============================================================

set -euo pipefail

# ── Environment ──
unset HF_TOKEN
export HF_HOME=/datadisk/NavBench3D/hf_cache
export CUDA_VISIBLE_DEVICES=0

# Activate conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate navbench3d

cd /home/v-yangzhao4/projects/NavBench3D

# ── Config ──
RENDERS_DIR="/datadisk/NavBench3D/renders"
OUTPUT_DIR="/datadisk/NavBench3D/ai_renders"
CONFIG="configs/default.yaml"
BACKEND="flux2"
LOG_DIR="logs"

mkdir -p "$LOG_DIR" "$OUTPUT_DIR"

echo "=============================================="
echo "NavBench3D - Batch AI Re-rendering"
echo "=============================================="
echo "Backend:    FLUX.2-dev (32B)"
echo "Renders:    $RENDERS_DIR"
echo "Output:     $OUTPUT_DIR"
echo "Config:     $CONFIG"
echo "Started:    $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
echo ""

# Count total scenes/views
TOTAL_SCENES=$(ls -d "$RENDERS_DIR"/*/view_0/rgb.png 2>/dev/null | wc -l)
TOTAL_VIEWS=$(ls -d "$RENDERS_DIR"/*/view_*/rgb.png 2>/dev/null | wc -l)
echo "Total: $TOTAL_SCENES scenes, $TOTAL_VIEWS views"
echo ""

# ── Run ──
python3 scripts/ai_rerender.py \
    --config "$CONFIG" \
    --renders-dir "$RENDERS_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --backend "$BACKEND" \
    --skip-existing

EXITCODE=$?

echo ""
echo "=============================================="
echo "Batch AI Re-rendering Complete"
echo "Exit code: $EXITCODE"
echo "Finished:  $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

# ── Summary ──
if [ -d "$OUTPUT_DIR" ]; then
    DONE=$(find "$OUTPUT_DIR" -name "rerender_results.json" | wc -l)
    VIEWS_DONE=$(find "$OUTPUT_DIR" -name "rgb_photorealistic.png" | wc -l)
    echo "Processed: $DONE scenes, $VIEWS_DONE views"
    
    # Quick pass/fail count
    PASSES=$(grep -r '"passes": true' "$OUTPUT_DIR"/*/rerender_results.json 2>/dev/null | wc -l || echo 0)
    FAILS=$(grep -r '"passes": false' "$OUTPUT_DIR"/*/rerender_results.json 2>/dev/null | wc -l || echo 0)
    echo "Pass: $PASSES views, Fail: $FAILS views"
fi
