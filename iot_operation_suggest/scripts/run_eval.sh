#!/bin/bash

# Model Evaluation Launch Script
# Run model evaluation standalone

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Default parameters
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/outputs/sft}"

# Evaluation data configuration
EVAL_DATA_FILE="${EVAL_DATA_FILE:-${PROJECT_ROOT}/corpus/eval_data.jsonl}"
MAX_SAMPLES="${MAX_SAMPLES:-500}"

# Generation configuration
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
TEMPERATURE="${TEMPERATURE:-0.1}"

# Output configuration
OUTPUT_DIR="${OUTPUT_DIR:-${MODEL_PATH}/eval_results}"

echo "=========================================="
echo "Model Evaluation Configuration"
echo "=========================================="
echo "Model path: $MODEL_PATH"
echo ""
echo "Evaluation Data Configuration:"
echo "  Data file: $EVAL_DATA_FILE"
echo "  Max samples: $MAX_SAMPLES"
echo ""
echo "Generation Configuration:"
echo "  Max new tokens: $MAX_NEW_TOKENS"
echo "  Temperature: $TEMPERATURE"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run evaluation
python -m src.evaluation.evaluator \
    --model_path "$MODEL_PATH" \
    --eval_data_file "$EVAL_DATA_FILE" \
    --max_samples "$MAX_SAMPLES" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --output_dir "$OUTPUT_DIR" \
    --bf16 \
    --seed 42

echo "=========================================="
echo "Evaluation completed!"
echo "Results saved to: $OUTPUT_DIR"
echo "=========================================="
