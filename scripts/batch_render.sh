#!/bin/bash
# NavBench3D - Batch Rendering Script
# Renders all navigable scenes using Blender in background mode.
# Features: skip completed, error logging, progress tracking, resume support.
#
# Usage:
#   bash scripts/batch_render.sh                    # render all
#   bash scripts/batch_render.sh --start 100        # start from scene 100
#   bash scripts/batch_render.sh --limit 50         # render only 50 scenes

set -uo pipefail

# ===== Config (override via env or CLI) =====
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
BLENDER="${BLENDER:-/usr/local/bin/blender}"
CONFIG="${CONFIG:-configs/internscene_local.yaml}"
SCENES_DIR="${SCENES_DIR:-/mnt/data/zhaoyang/navbench3d-data/scenes_v2}"
COMPOSED_DIR="${COMPOSED_DIR:-/mnt/data/zhaoyang/navbench3d-data/composed}"
RENDERS_DIR="${RENDERS_DIR:-/mnt/data/zhaoyang/navbench3d-data/renders_full_v1}"
LOG_DIR="${LOG_DIR:-/mnt/data/zhaoyang/navbench3d-data/logs/batch_render_full_v1}"
EXTRA_PYTHONPATH="${EXTRA_PYTHONPATH:-}"

# CLI args
START_IDX=0
LIMIT=999999

# Parse named args
while [[ $# -gt 0 ]]; do
    case $1 in
        --start) START_IDX="$2"; shift 2 ;;
        --limit) LIMIT="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        --scenes-dir) SCENES_DIR="$2"; shift 2 ;;
        --composed-dir) COMPOSED_DIR="$2"; shift 2 ;;
        --renders-dir) RENDERS_DIR="$2"; shift 2 ;;
        --log-dir) LOG_DIR="$2"; shift 2 ;;
        --blender) BLENDER="$2"; shift 2 ;;
        --extra-pythonpath) EXTRA_PYTHONPATH="$2"; shift 2 ;;
        *) shift ;;
    esac
done

mkdir -p "$LOG_DIR" "$RENDERS_DIR"

# ===== Gather scenes =====
mapfile -t SCENE_LIST < <(ls -d "$SCENES_DIR"/*/ | sort)
TOTAL=${#SCENE_LIST[@]}
echo "=== NavBench3D Batch Rendering ==="
echo "Total scenes: $TOTAL"
echo "Start index: $START_IDX"
echo "Limit: $LIMIT"
echo ""

# ===== Stats =====
RENDERED=0
SKIPPED=0
FAILED=0
START_TIME=$(date +%s)
FAIL_LOG="$LOG_DIR/failed_scenes.txt"
> "$FAIL_LOG"

for i in $(seq "$START_IDX" $((TOTAL - 1))); do
    if [[ $RENDERED -ge $LIMIT ]]; then
        echo "Reached limit ($LIMIT). Stopping."
        break
    fi

    SCENE_PATH="${SCENE_LIST[$i]}"
    SCENE_ID=$(basename "$SCENE_PATH")

    # Skip if already rendered (cameras.json exists)
    if [[ -f "$RENDERS_DIR/$SCENE_ID/cameras.json" ]]; then
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    # Skip if no composed GLB
    GLB_FILE="$COMPOSED_DIR/${SCENE_ID}.glb"
    if [[ ! -f "$GLB_FILE" ]]; then
        echo "[$((i+1))/$TOTAL] SKIP $SCENE_ID (no GLB)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi

    IDX_DISPLAY=$((i + 1))
    ELAPSED=$(($(date +%s) - START_TIME))
    if [[ $RENDERED -gt 0 ]]; then
        AVG=$((ELAPSED / RENDERED))
        ETA=$(( (TOTAL - i) * AVG ))
        ETA_H=$((ETA / 3600))
        ETA_M=$(( (ETA % 3600) / 60 ))
        echo "[$IDX_DISPLAY/$TOTAL] Rendering $SCENE_ID  (done=$RENDERED, skip=$SKIPPED, fail=$FAILED, ETA=${ETA_H}h${ETA_M}m)"
    else
        echo "[$IDX_DISPLAY/$TOTAL] Rendering $SCENE_ID"
    fi

    # Run Blender with timeout (15 min per scene max)
    SCENE_LOG="$LOG_DIR/${SCENE_ID}.log"
    if [[ -n "$EXTRA_PYTHONPATH" ]]; then
        NAVBENCH_PY_SITE="${EXTRA_PYTHONPATH}" \
        PYTHONPATH="${EXTRA_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}" \
        timeout 900 "$BLENDER" --background --python-use-system-env \
            --python "$PROJECT_DIR/scripts/render_scenes.py" \
            -- \
            --config "$PROJECT_DIR/$CONFIG" \
            --scenes-dir "$SCENES_DIR" \
            --output-dir "$RENDERS_DIR" \
            --composed-dir "$COMPOSED_DIR" \
            --extra-pythonpath "$EXTRA_PYTHONPATH" \
            --scene-id "$SCENE_ID" \
            > "$SCENE_LOG" 2>&1
    else
        timeout 900 "$BLENDER" --background --python-use-system-env \
        --python "$PROJECT_DIR/scripts/render_scenes.py" \
        -- \
        --config "$PROJECT_DIR/$CONFIG" \
        --scenes-dir "$SCENES_DIR" \
        --output-dir "$RENDERS_DIR" \
        --composed-dir "$COMPOSED_DIR" \
        --scene-id "$SCENE_ID" \
        > "$SCENE_LOG" 2>&1
    fi

    EXIT_CODE=$?
    if [[ $EXIT_CODE -eq 0 ]] && [[ -f "$RENDERS_DIR/$SCENE_ID/cameras.json" ]]; then
        RENDERED=$((RENDERED + 1))
        # Clean up log on success to save space
        rm -f "$SCENE_LOG"
    else
        FAILED=$((FAILED + 1))
        echo "  FAILED (exit=$EXIT_CODE) — see $SCENE_LOG"
        echo "$SCENE_ID exit=$EXIT_CODE" >> "$FAIL_LOG"
    fi
done

# ===== Summary =====
ELAPSED=$(($(date +%s) - START_TIME))
HOURS=$((ELAPSED / 3600))
MINS=$(( (ELAPSED % 3600) / 60 ))

echo ""
echo "=== Batch Rendering Complete ==="
echo "Rendered: $RENDERED"
echo "Skipped:  $SKIPPED"
echo "Failed:   $FAILED"
echo "Time:     ${HOURS}h ${MINS}m"
echo "Failed scenes: $FAIL_LOG"
