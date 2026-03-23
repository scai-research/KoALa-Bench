#!/bin/bash
# ============================================================
# Translation Evaluation — Gemini Flash (gemini_flash)
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-${SCRIPT_DIR}}"
while [ ! -f "${BASE_PATH}/prompts.yaml" ] && [ "${BASE_PATH}" != "/" ]; do
    BASE_PATH="$(dirname "${BASE_PATH}")"
done
GEMINI_API_KEY_FILE="${BASE_PATH}/gemini_key.txt"
[ -f "${GEMINI_API_KEY_FILE}" ] && export GEMINI_API_KEY="$(head -1 "${GEMINI_API_KEY_FILE}" | tr -d '\r\n')"

MODEL_PATH="gemini-2.5-flash-lite"
OUTPUT_BASE="${BASE_PATH}/results_real/Translation"
TRANSLATION_DATA_PATH="${BASE_PATH}/Translation"
BACKEND="gemini_flash"

BENCHMARKS=(
    "etri_tst-COMMON_clean"
    "etri_tst-HE_clean"
)

# 프롬프트 하나만 사용 (v1). 여러 개 쓰려면 PROMPT_NAME 비우기 (prompt_file의 translation 전부 실행)
PROMPT_FILE="${BASE_PATH}/prompts.yaml"
PROMPT_NAME="v1"
MAX_SAMPLES="${MAX_SAMPLES:-}"
TOKENIZE_METHOD="character"
GT_FIELD="answer_ko"
BATCH_SIZE=1
TENSOR_PARALLEL_SIZE=1

export PYTHONPATH="${BASE_PATH}/src:${PYTHONPATH}"
echo "[BACKEND] ${BACKEND} (model=${MODEL_PATH})"

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

    OPTS=" --input ${INPUT_JSONL} --output-dir ${OUTPUT_DIR} --backend ${BACKEND}"
    OPTS+=" --prompt-file ${PROMPT_FILE} --tokenize ${TOKENIZE_METHOD}"
    [ -n "${PROMPT_NAME}" ] && OPTS+=" --prompt-name ${PROMPT_NAME}"
    OPTS+=" --gt-field ${GT_FIELD} --batch-size ${BATCH_SIZE}"
    OPTS+=" --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}"
    [ -n "${MODEL_PATH}" ] && OPTS+=" --model ${MODEL_PATH}"
    [ -n "${MAX_SAMPLES}" ] && OPTS+=" --max-samples ${MAX_SAMPLES}"

    mkdir -p "${OUTPUT_DIR}"
    echo "Running: python3 ${BASE_PATH}/Translation/run_translation_evaluation.py ${OPTS}"
    python3 ${BASE_PATH}/Translation/run_translation_evaluation.py ${OPTS}
done

echo ""
echo "Translation (gemini-flash) completed!"
