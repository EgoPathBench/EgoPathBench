#!/bin/bash
# NavBench3D - Full Pipeline Runner
# Run the complete pipeline from data download to evaluation.
#
# Usage:
#   bash scripts/run_pipeline.sh [step]
#   Steps: download, render, rerender, gt, vqa, benchmark, evaluate, all

set -e

CONFIG="configs/default.yaml"
SCENES_DIR="data/scenes"
RENDERS_DIR="data/renders"
AI_RENDERS_DIR="data/ai_renders"
GT_DIR="data/gt_paths"
VQA_DIR="data/vqa_questions"
RESULTS_DIR="data/results"

STEP="${1:-all}"

echo "============================================"
echo "NavBench3D Pipeline"
echo "Step: $STEP"
echo "============================================"

# Step 1: Download and filter scenes
if [[ "$STEP" == "download" || "$STEP" == "all" ]]; then
    echo ""
    echo "[Step 1] Downloading and filtering InternScenes..."
    python scripts/download_and_filter.py \
        --config "$CONFIG" \
        --download \
        --output-dir "$SCENES_DIR"
    echo "[Step 1] Done."
fi

# Step 2: Render scenes with Blender
if [[ "$STEP" == "render" || "$STEP" == "all" ]]; then
    echo ""
    echo "[Step 2] Rendering scenes with Blender..."
    echo "  Generating render commands..."
    python scripts/render_scenes.py \
        --config "$CONFIG" \
        --scenes-dir "$SCENES_DIR" \
        --output-dir "$RENDERS_DIR" \
        --dry-run
    echo ""
    echo "  To render, run each command with Blender installed."
    echo "  Or run directly:"
    echo "  blender --background --python scripts/render_scenes.py -- --config $CONFIG --scenes-dir $SCENES_DIR"
    echo "[Step 2] Done (commands generated)."
fi

# Step 3: AI Re-rendering
if [[ "$STEP" == "rerender" || "$STEP" == "all" ]]; then
    echo ""
    echo "[Step 3] AI Re-rendering with FLUX.2-dev..."
    python scripts/ai_rerender.py \
        --config "$CONFIG" \
        --renders-dir "$RENDERS_DIR" \
        --output-dir "$AI_RENDERS_DIR" \
        --skip-existing
    echo "[Step 3] Done."
fi

# Step 4: Build navigation ground truth
if [[ "$STEP" == "gt" || "$STEP" == "all" ]]; then
    echo ""
    echo "[Step 4] Building navigation ground truth..."
    python scripts/build_navigation_gt.py \
        --config "$CONFIG" \
        --scenes-dir "$SCENES_DIR" \
        --renders-dir "$RENDERS_DIR" \
        --output-dir "$GT_DIR"
    echo "[Step 4] Done."
fi

# Step 5: Build VQA questions
if [[ "$STEP" == "vqa" || "$STEP" == "all" ]]; then
    echo ""
    echo "[Step 5] Building VQA question sets..."
    python scripts/build_vqa_questions.py \
        --config "$CONFIG" \
        --scenes-dir "$SCENES_DIR" \
        --renders-dir "$RENDERS_DIR" \
        --ai-renders-dir "$AI_RENDERS_DIR" \
        --gt-dir "$GT_DIR" \
        --output-dir "$VQA_DIR"
    echo "[Step 5] Done."
fi

# Step 6: Run benchmark (all models)
if [[ "$STEP" == "benchmark" || "$STEP" == "all" ]]; then
    echo ""
    echo "[Step 6] Running benchmark evaluation..."
    QUESTIONS="$VQA_DIR/all_questions.json"

    if [ ! -f "$QUESTIONS" ]; then
        echo "  ERROR: Questions file not found: $QUESTIONS"
        echo "  Run the 'vqa' step first."
        exit 1
    fi

    # Run each model
    for MODEL in "gpt-4o" "claude-3.5-sonnet" "gemini-2.0-flash" "qwen-vl-max"; do
        echo ""
        echo "  Running model: $MODEL"
        python scripts/run_benchmark.py \
            --config "$CONFIG" \
            --questions "$QUESTIONS" \
            --model "$MODEL" \
            --output "$RESULTS_DIR/${MODEL}_predictions.json" \
            2>&1 | tee "$RESULTS_DIR/${MODEL}_log.txt"
    done
    echo "[Step 6] Done."
fi

# Step 7: Evaluate predictions
if [[ "$STEP" == "evaluate" || "$STEP" == "all" ]]; then
    echo ""
    echo "[Step 7] Evaluating predictions..."
    QUESTIONS="$VQA_DIR/all_questions.json"

    for PRED_FILE in "$RESULTS_DIR"/*_predictions.json; do
        if [ -f "$PRED_FILE" ]; then
            MODEL_NAME=$(basename "$PRED_FILE" _predictions.json)
            echo ""
            echo "  Evaluating: $MODEL_NAME"
            python scripts/evaluate.py \
                --predictions "$PRED_FILE" \
                --gt "$QUESTIONS" \
                --scenes-dir "$SCENES_DIR" \
                --output "$RESULTS_DIR/${MODEL_NAME}_evaluation.json"
        fi
    done
    echo "[Step 7] Done."
fi

echo ""
echo "============================================"
echo "Pipeline complete!"
echo "============================================"
