#!/bin/bash
# ============================================================
# SQA Evaluation — Gemini Flash (gemini_flash)
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-${SCRIPT_DIR}}"
while [ ! -f "${BASE_PATH}/prompts.yaml" ] && [ "${BASE_PATH}" != "/" ]; do
    BASE_PATH="$(dirname "${BASE_PATH}")"
done
GEMINI_API_KEY_FILE="${BASE_PATH}/gemini_key.txt"
[ -f "${GEMINI_API_KEY_FILE}" ] && export GEMINI_API_KEY="$(head -1 "${GEMINI_API_KEY_FILE}" | tr -d '\r\n')"

OUTPUT_BASE="${BASE_PATH}/results_real/SQA"
BACKEND="gemini_flash"
MODEL_PATH="gemini-2.5-flash-lite"

BENCHMARKS=(
    "click_clean"
    "kobest_boolq_clean"
    "click_other"
    "kobest_boolq_other"
)

# 프롬프트 v1만 사용 (여러 개 쓰려면 PROMPT_NAME 비우기)
PROMPT_FILE="${BASE_PATH}/prompts.yaml"
PROMPT_NAME="v1"
MAX_SAMPLES="${MAX_SAMPLES:-}"
BATCH_SIZE=1
SAVE_GENERATION=true
MAX_NEW_TOKENS=64
TENSOR_PARALLEL_SIZE=1

export PYTHONPATH="${BASE_PATH}/src:${PYTHONPATH}"
echo "[BACKEND] ${BACKEND} (model=${MODEL_PATH})"

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

    OPTS=" --jsonl ${INPUT_JSONL} --output_dir ${OUTPUT_DIR} --backend ${BACKEND}"
    OPTS+=" --base_dir ${BASE_PATH} --max_new_tokens ${MAX_NEW_TOKENS}"
    OPTS+=" --batch_size ${BATCH_SIZE} --tensor_parallel_size ${TENSOR_PARALLEL_SIZE}"
    OPTS+=" --prompt-file ${PROMPT_FILE}"
    [ -n "${PROMPT_NAME}" ] && OPTS+=" --prompt-name ${PROMPT_NAME}"
    [ -n "${MODEL_PATH}" ] && OPTS+=" --model_path ${MODEL_PATH}"
    [ -n "${MAX_SAMPLES}" ] && OPTS+=" --max_samples ${MAX_SAMPLES}"
    [ "${SAVE_GENERATION}" = true ] && OPTS+=" --save_generation"

    mkdir -p "${OUTPUT_DIR}"
    echo "Running: python3 ${BASE_PATH}/SQA/evaluate_sqa.py ${OPTS}"
    python3 ${BASE_PATH}/SQA/evaluate_sqa.py ${OPTS}
done

echo ""
echo "SQA (gemini-flash) completed!"
