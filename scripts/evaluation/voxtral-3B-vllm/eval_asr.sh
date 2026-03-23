#!/bin/bash

# ============================================================
# ASR Evaluation Script — Voxtral Mini 3B (vLLM)
# ============================================================

# paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-${SCRIPT_DIR}}"
while [ ! -f "${BASE_PATH}/prompts.yaml" ] && [ "${BASE_PATH}" != "/" ]; do
    BASE_PATH="$(dirname "${BASE_PATH}")"
done
OUTPUT_BASE="${BASE_PATH}/results_real/ASR"
ASR_DATA_PATH="${BASE_PATH}/ASR"

# model
BACKEND="voxtral3b_vllm"
MODEL_PATH="mistralai/Voxtral-Mini-3B-2507"
TENSOR_PARALLEL_SIZE=1

# benchmarks (주석 처리로 선택)
BENCHMARKS=(
    "common_voice_korea_clean"
    "common_voice_korea_other"
    "ksponspeech_eval_clean"
    "ksponspeech_eval_other"
    "zeroth_korean_test_clean"
    "zeroth_korean_test_other"
)

# options
PROMPT_FILE="${BASE_PATH}/prompts.yaml"
MAX_SAMPLES=""  # 빈 값이면 전체
BATCH_SIZE=1    # vLLM이 내부 최적화

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
    echo "Running ASR: ${BENCH}"
    echo "=========================================="

    INPUT_JSONL="${ASR_DATA_PATH}/${BENCH}.jsonl"
    OUTPUT_DIR="${OUTPUT_BASE}/${BACKEND}/${BENCH}"

    if [ ! -f "${INPUT_JSONL}" ]; then
        echo "[SKIP] JSONL not found: ${INPUT_JSONL}"
        continue
    fi

    OPTS=""
    OPTS+=" --input ${INPUT_JSONL}"
    OPTS+=" --output-dir ${OUTPUT_DIR}"
    OPTS+=" --backend ${BACKEND}"
    OPTS+=" --prompt-file ${PROMPT_FILE}"
    OPTS+=" --batch-size ${BATCH_SIZE}"
    OPTS+=" --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}"

    if [ -n "${MODEL_PATH}" ]; then
        OPTS+=" --model ${MODEL_PATH}"
    fi

    if [ -n "${MAX_SAMPLES}" ]; then
        OPTS+=" --max-samples ${MAX_SAMPLES}"
    fi

    CMD="python3 ${BASE_PATH}/ASR/run_asr_evaluation.py ${OPTS}"
    echo "Running: ${CMD}"
    mkdir -p "${OUTPUT_DIR}"
    eval ${CMD}
done

echo ""
echo "All ASR benchmarks (Voxtral Mini 3B vLLM) completed!"
