#!/bin/bash
# ============================================================
# LSQA Evaluation — Gemini Flash (gemini_flash)
# ============================================================
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-${SCRIPT_DIR}}"
while [ ! -f "${BASE_PATH}/prompts.yaml" ] && [ "${BASE_PATH}" != "/" ]; do
    BASE_PATH="$(dirname "${BASE_PATH}")"
done
GEMINI_API_KEY_FILE="${BASE_PATH}/gemini_key.txt"
[ -f "${GEMINI_API_KEY_FILE}" ] && export GEMINI_API_KEY="$(head -1 "${GEMINI_API_KEY_FILE}" | tr -d '\r\n')"

EVAL_SCRIPT="${BASE_PATH}/PA-QA/evaluate_lsqa.py"
OUTPUT_BASE="${BASE_PATH}/results_real/LSQA"
LSQA_DIR="${BASE_PATH}/PA-QA"
BACKEND="gemini_flash"
MODEL_PATH="gemini-2.5-flash-lite"
TENSOR_PARALLEL_SIZE=1

AUDIO_BASE_CLEAN="${BASE_PATH}/audio"
AUDIO_BASE_NOISE="${BASE_PATH}"

BENCHMARKS=(
    "mctest_clean"
    "mctest_other"
)

SAVE_GENERATION=true
MAX_NEW_TOKENS=256
MAX_SAMPLES="${MAX_SAMPLES:-}"
# 프롬프트 v1만 사용 (여러 개 쓰려면 PROMPT_NAME 비우기)
PROMPT_FILE="${BASE_PATH}/prompts.yaml"
PROMPT_NAME="v1"

export PYTHONPATH="${BASE_PATH}/src:${PYTHONPATH}"
echo "[LSQA] BACKEND=${BACKEND} MODEL=${MODEL_PATH}"

for BENCH in "${BENCHMARKS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running LSQA: ${BENCH}"
    echo "=========================================="

    INPUT_JSONL="${LSQA_DIR}/${BENCH}.jsonl"
    OUTPUT_DIR="${OUTPUT_BASE}/${BACKEND}/${BENCH}"

    if [ ! -f "${INPUT_JSONL}" ]; then
        echo "[SKIP] JSONL not found: ${INPUT_JSONL}"
        continue
    fi

    if [[ "${BENCH}" == *"_noise"* ]]; then
        AUDIO_BASE="${AUDIO_BASE_NOISE}"
    else
        AUDIO_BASE="${AUDIO_BASE_CLEAN}"
    fi

    CMD="python3 ${EVAL_SCRIPT}"
    CMD+=" --jsonl ${INPUT_JSONL}"
    CMD+=" --output_dir ${OUTPUT_DIR}"
    CMD+=" --backend ${BACKEND}"
    CMD+=" --model_path ${MODEL_PATH}"
    CMD+=" --base_dir ${AUDIO_BASE}"
    CMD+=" --tensor_parallel_size ${TENSOR_PARALLEL_SIZE}"
    CMD+=" --max_new_tokens ${MAX_NEW_TOKENS}"
    [ -n "${MAX_SAMPLES}" ] && CMD+=" --max_samples ${MAX_SAMPLES}"
    [ "${SAVE_GENERATION}" = true ] && CMD+=" --save_generation"
    [ -n "${PROMPT_FILE}" ] && CMD+=" --prompt-file ${PROMPT_FILE}"
    [ -n "${PROMPT_NAME}" ] && CMD+=" --prompt-name ${PROMPT_NAME}"

    echo "JSONL:    ${INPUT_JSONL}"
    echo "출력:     ${OUTPUT_DIR}"
    mkdir -p "${OUTPUT_DIR}"
    eval ${CMD}
done

echo ""
echo "LSQA (gemini-flash) 완료!"
