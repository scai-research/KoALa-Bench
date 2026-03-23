#!/bin/bash

# ============================================================
# ASR Evaluation Script — Qwen3-Omni (vLLM)
# ============================================================

# paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-${SCRIPT_DIR}}"
while [ ! -f "${BASE_PATH}/prompts.yaml" ] && [ "${BASE_PATH}" != "/" ]; do
    BASE_PATH="$(dirname "${BASE_PATH}")"
done
OUTPUT_BASE="${BASE_PATH}/results_real/ASR"

# model
BACKEND="qwen3_vllm"
MODEL_PATH="Qwen/Qwen3-Omni-30B-A3B-Instruct"

# JSONL이 있는 위치
ASR_DATA_PATH="${BASE_PATH}/ASR"

BENCHMARKS=(
    "common_voice_korea_clean"
    "common_voice_korea_other"
    "ksponspeech_eval_clean"
    "ksponspeech_eval_other"
    "zeroth_korean_test_clean"
    "zeroth_korean_test_other"
)

# options — 프롬프트: prompts.yaml 에서 4개 변형 로드
PROMPT_FILE="${BASE_PATH}/prompts.yaml"
MAX_SAMPLES=""  # 빈 값이면 전체
BATCH_SIZE=8     # vLLM 배치 크기
TENSOR_PARALLEL_SIZE=1  # GPU 수

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
    mkdir -p ${OUTPUT_DIR}
    eval ${CMD}
done

echo ""
echo "All ASR benchmarks (Qwen3-Omni vLLM) completed!"
