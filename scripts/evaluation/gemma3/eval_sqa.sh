#!/bin/bash

# ============================================================
# SQA (Spoken QA) Evaluation Script — Gemma-3n (vLLM)
# ============================================================

# paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-${SCRIPT_DIR}}"
while [ ! -f "${BASE_PATH}/prompts.yaml" ] && [ "${BASE_PATH}" != "/" ]; do
    BASE_PATH="$(dirname "${BASE_PATH}")"
done
OUTPUT_BASE="${BASE_PATH}/results_real/SQA"

# model
BACKEND="gemma3n_vllm"
MODEL_PATH="google/gemma-3n-E4B-it"  # 또는 google/gemma-3n-E4B-it

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
BATCH_SIZE=8          # vLLM 배치 크기 (vLLM이 내부적으로 최적화)
SAVE_GENERATION=true
MAX_NEW_TOKENS=64
# Gemma-3n + vLLM에서 TP=2 사용 시 shape 불일치 → 단일 GPU
TENSOR_PARALLEL_SIZE=1

# ============================================================
# Run
# ============================================================
export PYTHONPATH="${BASE_PATH}/src:${PYTHONPATH}"
echo "[BACKEND] ${BACKEND} (vLLM)"
echo "[MODEL] ${MODEL_PATH}"
echo "[TENSOR_PARALLEL] ${TENSOR_PARALLEL_SIZE}"

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
    OPTS+=" --batch_size ${BATCH_SIZE}"
    OPTS+=" --tensor_parallel_size ${TENSOR_PARALLEL_SIZE}"
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
echo "All SQA benchmarks (Gemma-3n vLLM) completed!"
