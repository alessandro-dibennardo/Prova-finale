#!/bin/bash

set -euo pipefail

# Configuration --------------------------------------------------------------
# Override these variables via the environment when invoking the script, e.g.:
#   DATASET_PATH=/path/to/dev.json MODEL_PATH=./model/extractor/qwen \
#   CUDA_VISIBLE_DEVICES=0 bash extract.sh

: "${SWIFT_CMD:=swift}"
: "${MODEL_PATH:=./model/extractor/please_put_models_here}"
: "${DATASET_PATH:?Set DATASET_PATH to the SWIFT-format dataset to extract entities from.}"
: "${OUTPUT_DIR:=./model/extractor/infer_result}"
: "${SHOW_SAMPLES:=5}"

mkdir -p "${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
"${SWIFT_CMD}" infer \
    --ckpt_dir "${MODEL_PATH}" \
    --custom_val_dataset_path "${DATASET_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --show_dataset_sample "${SHOW_SAMPLES}"
