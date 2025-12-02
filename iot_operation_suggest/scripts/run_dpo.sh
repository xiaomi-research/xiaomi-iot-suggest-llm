#!/bin/bash

# DPO Training Launch Script
# Supports random sampling from merged single data file, automatic evaluation after training
# Supports LoRA efficient fine-tuning (enabled by default)

# Project root directory
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# Default parameters
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"

# LoRA configuration (DPO enables LoRA by default)
USE_LORA="${USE_LORA:-true}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj}"

# Training data configuration
TRAIN_DATA_FILE="${TRAIN_DATA_FILE:-${PROJECT_ROOT}/corpus/train_data.jsonl}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"  # -1 means use all data
VAL_RATIO="${VAL_RATIO:-0.1}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-10000}"

# Evaluation data configuration
EVAL_DATA_FILE="${EVAL_DATA_FILE:-${PROJECT_ROOT}/corpus/eval_data.jsonl}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-6000}"

# Training configuration
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/dpo}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LEARNING_RATE="${LEARNING_RATE:-5e-6}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
BETA="${BETA:-0.1}"

echo "=========================================="
echo "DPO Training Configuration"
echo "=========================================="
echo "Model: $MODEL_NAME_OR_PATH"
echo ""
echo "LoRA Configuration:"
echo "  Use LoRA: $USE_LORA"
if [ "$USE_LORA" = "true" ]; then
    echo "  LoRA r: $LORA_R"
    echo "  LoRA alpha: $LORA_ALPHA"
    echo "  LoRA dropout: $LORA_DROPOUT"
    echo "  Target modules: $LORA_TARGET_MODULES"
fi
echo ""
echo "Training Data Configuration:"
echo "  Data file: $TRAIN_DATA_FILE"
echo "  Max samples: $TRAIN_MAX_SAMPLES (-1 means all)"
echo "  Validation ratio: $VAL_RATIO"
echo "  Max validation samples: $VAL_MAX_SAMPLES"
echo ""
echo "Evaluation Data Configuration:"
echo "  Data file: $EVAL_DATA_FILE"
echo "  Max samples: $EVAL_MAX_SAMPLES"
echo ""
echo "Training Configuration:"
echo "  Output directory: $OUTPUT_DIR"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Gradient accumulation: $GRAD_ACCUM"
echo "  Learning rate: $LEARNING_RATE"
echo "  Max sequence length: $MAX_LENGTH"
echo "  Beta: $BETA"
echo "=========================================="

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Build LoRA arguments
LORA_ARGS=""
if [ "$USE_LORA" = "true" ]; then
    LORA_ARGS="--lora_r $LORA_R --lora_alpha $LORA_ALPHA --lora_dropout $LORA_DROPOUT --lora_target_modules $LORA_TARGET_MODULES"
else
    LORA_ARGS="--no_lora"
fi

# Run training
python -m src.training.dpo_trainer \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    $LORA_ARGS \
    --train_data_file "$TRAIN_DATA_FILE" \
    --train_max_samples "$TRAIN_MAX_SAMPLES" \
    --val_ratio "$VAL_RATIO" \
    --val_max_samples "$VAL_MAX_SAMPLES" \
    --eval_data_file "$EVAL_DATA_FILE" \
    --eval_max_samples "$EVAL_MAX_SAMPLES" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --learning_rate "$LEARNING_RATE" \
    --max_length "$MAX_LENGTH" \
    --beta "$BETA" \
    --bf16 \
    --seed 42

echo "=========================================="
echo "Training and evaluation completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "Evaluation results: $OUTPUT_DIR/eval_results"
echo "=========================================="
