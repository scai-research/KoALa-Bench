#!/bin/bash

# ============================================================
# Translation Evaluation Script — Gemma-3n (vLLM)
# ============================================================

# paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-${SCRIPT_DIR}}"
while [ ! -f "${BASE_PATH}/prompts.yaml" ] && [ "${BASE_PATH}" != "/" ]; do
    BASE_PATH="$(dirname "${BASE_PATH}")"
done
MODEL_PATH="google/gemma-3n-E4B-it"  # 또는 google/gemma-3n-E4B-it
OUTPUT_BASE="${BASE_PATH}/results_real/Translation"

# JSONL이 있는 위치
TRANSLATION_DATA_PATH="${BASE_PATH}/Translation"

# model (vLLM 백엔드)
BACKEND="gemma3n_vllm"

# benchmarks (주석 처리로 선택)
BENCHMARKS=(
    "etri_tst-COMMON_clean"
    "etri_tst-HE_clean"
)

# options — 프롬프트: prompts.yaml 에서 4개 변형 로드
PROMPT_FILE="${BASE_PATH}/prompts.yaml"
MAX_SAMPLES=""          # 빈 값이면 전체
TOKENIZE_METHOD="character"  # character, space, morpheme
GT_FIELD="answer_ko"
BATCH_SIZE=8            # vLLM 배치 크기 (vLLM이 내부적으로 최적화)
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
    echo "Running Translation: ${BENCH}"
    echo "=========================================="
    
    INPUT_JSONL="${TRANSLATION_DATA_PATH}/${BENCH}.jsonl"
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
    OPTS+=" --tokenize ${TOKENIZE_METHOD}"
    OPTS+=" --gt-field ${GT_FIELD}"
    OPTS+=" --batch-size ${BATCH_SIZE}"
    OPTS+=" --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}"
    
    if [ -n "${MODEL_PATH}" ]; then
        OPTS+=" --model ${MODEL_PATH}"
    fi
    
    if [ -n "${MAX_SAMPLES}" ]; then
        OPTS+=" --max-samples ${MAX_SAMPLES}"
    fi
    
    CMD="python3 ${BASE_PATH}/Translation/run_translation_evaluation.py ${OPTS}"
    echo "Running: ${CMD}"
    mkdir -p ${OUTPUT_DIR}
    eval ${CMD}
done

echo ""
echo "All Translation benchmarks (Gemma-3n vLLM) completed!"
