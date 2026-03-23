#!/bin/bash

# ============================================================
# SQA (Spoken QA) Evaluation Script — Qwen2-Audio (vLLM)
# ============================================================

# paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-${SCRIPT_DIR}}"
while [ ! -f "${BASE_PATH}/prompts.yaml" ] && [ "${BASE_PATH}" != "/" ]; do
    BASE_PATH="$(dirname "${BASE_PATH}")"
done
OUTPUT_BASE="${BASE_PATH}/results_real/SQA"

# model
BACKEND="qwen_vllm"
MODEL_PATH=""

# benchmarks (주석 처리로 선택)
BENCHMARKS=(
    "click_clean"
    "kobest_boolq_clean"
    "click_other"
    "kobest_boolq_other"
)

# options — 프롬프트: prompts.yaml 에서 4개 변형 로드
PROMPT_FILE="${BASE_PATH}/prompts.yaml"
MAX_SAMPLES=""        # 빈 값이면 전체
SAVE_GENERATION=true
MAX_NEW_TOKENS=64

# ============================================================
# Run
# ============================================================
export PYTHONPATH="${BASE_PATH}/src:${PYTHONPATH}"
echo "[BACKEND] ${BACKEND}"

for BENCH in "${BENCHMARKS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running SQA: ${BENCH}"
    echo "=========================================="

    INPUT_JSONL="${BASE_PATH}/SQA/${BENCH}.jsonl"
    OUTPUT_DIR="${OUTPUT_BASE}/${BACKEND}/${BENCH}"

    if [ ! -f "${INPUT_JSONL}" ]; then
        echo "[SKIP] JSONL not found: ${INPUT_JSONL}"
        continue
    fi

    OPTS=""
    OPTS+=" --jsonl ${INPUT_JSONL}"
    OPTS+=" --output_dir ${OUTPUT_DIR}"
    OPTS+=" --backend ${BACKEND}"
    OPTS+=" --base_dir ${BASE_PATH}"
    OPTS+=" --max_new_tokens ${MAX_NEW_TOKENS}"
    OPTS+=" --prompt-file ${PROMPT_FILE}"

    if [ -n "${MODEL_PATH}" ]; then
        OPTS+=" --model_path ${MODEL_PATH}"
    fi

    if [ -n "${MAX_SAMPLES}" ]; then
        OPTS+=" --max_samples ${MAX_SAMPLES}"
    fi

    if [ "${SAVE_GENERATION}" = true ]; then
        OPTS+=" --save_generation"
    fi

    CMD="python3 ${BASE_PATH}/SQA/evaluate_sqa.py ${OPTS}"
    echo "Running: ${CMD}"
    mkdir -p ${OUTPUT_DIR}
    eval ${CMD}
done

echo ""
echo "All SQA benchmarks (Qwen2-Audio vLLM) completed!"
