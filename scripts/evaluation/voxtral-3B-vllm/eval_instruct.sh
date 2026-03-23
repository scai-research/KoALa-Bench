#!/bin/bash

# ============================================================
# Instruct Evaluation Script — Voxtral Mini 3B (vLLM)
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-${SCRIPT_DIR}}"
while [ ! -f "${BASE_PATH}/prompts.yaml" ] && [ "${BASE_PATH}" != "/" ]; do
    BASE_PATH="$(dirname "${BASE_PATH}")"
done
MODEL_PATH="mistralai/Voxtral-Mini-3B-2507"
OUTPUT_BASE="${BASE_PATH}/results_real/Instruct"
TENSOR_PARALLEL_SIZE=1

BACKEND="voxtral3b_vllm"

BENCHMARKS=(
    "vicuna_clean"
    "alpaca_clean"
    "openhermes_clean"
    "kudge_clean"
    "vicuna_other"
    "alpaca_other"
    "openhermes_other"
    "kudge_other"
)

RUN_INFERENCE=true
MAX_SAMPLES=""
BATCH_SIZE=4
GPT_MODEL="gpt-4o-mini"
OPENAI_API_KEY_FILE="${BASE_PATH}/openai_key.txt"
PROMPT_FILE="${BASE_PATH}/prompts.yaml"

# ============================================================
export PYTHONPATH="${BASE_PATH}/src:${PYTHONPATH}"
echo "[BACKEND] ${BACKEND} (vLLM)"

for BENCH in "${BENCHMARKS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running Instruct: ${BENCH}"
    echo "=========================================="

    INPUT_JSONL="${BASE_PATH}/Instruct/${BENCH}.jsonl"
    OUTPUT_DIR="${OUTPUT_BASE}/${BACKEND}/${BENCH}"

    if [ ! -f "${INPUT_JSONL}" ]; then
        echo "[SKIP] JSONL not found: ${INPUT_JSONL}"
        continue
    fi

    OPTS=""
    OPTS+=" --original_jsonl ${INPUT_JSONL}"
    OPTS+=" --output_dir ${OUTPUT_DIR}"
    OPTS+=" --backend ${BACKEND}"
    OPTS+=" --base_dir ${BASE_PATH}"
    OPTS+=" --gpt_model ${GPT_MODEL}"
    OPTS+=" --prompt-file ${PROMPT_FILE}"
    OPTS+=" --tensor_parallel_size ${TENSOR_PARALLEL_SIZE}"
    if [ -f "${OPENAI_API_KEY_FILE}" ]; then
        OPTS+=" --openai_api_key_file ${OPENAI_API_KEY_FILE}"
    fi
    if [ "${RUN_INFERENCE}" = true ]; then
        OPTS+=" --run_inference"
        OPTS+=" --batch_size ${BATCH_SIZE}"
        if [ -n "${MODEL_PATH}" ]; then
            OPTS+=" --model_path ${MODEL_PATH}"
        fi
    fi

    if [ -n "${MAX_SAMPLES}" ]; then
        OPTS+=" --max_samples ${MAX_SAMPLES}"
    fi

    CMD="python3 ${BASE_PATH}/Instruct/evaluate_instruct.py ${OPTS}"
    echo "Running: ${CMD}"
    mkdir -p "${OUTPUT_DIR}"
    eval ${CMD}
done

echo ""
echo "All Instruct benchmarks (Voxtral Mini 3B vLLM) completed!"
