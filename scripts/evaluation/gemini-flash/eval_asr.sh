#!/bin/bash
# ============================================================
# ASR Evaluation — Gemini Flash (gemini_flash)
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-${SCRIPT_DIR}}"
while [ ! -f "${BASE_PATH}/prompts.yaml" ] && [ "${BASE_PATH}" != "/" ]; do
    BASE_PATH="$(dirname "${BASE_PATH}")"
done
GEMINI_API_KEY_FILE="${BASE_PATH}/gemini_key.txt"
[ -f "${GEMINI_API_KEY_FILE}" ] && export GEMINI_API_KEY="$(head -1 "${GEMINI_API_KEY_FILE}" | tr -d '\r\n')"

OUTPUT_BASE="${BASE_PATH}/results_real/ASR"
ASR_DATA_PATH="${BASE_PATH}/ASR"
BACKEND="gemini_flash"
MODEL_PATH="gemini-2.5-flash-lite"

BENCHMARKS=(
    "common_voice_korea_clean"
    "common_voice_korea_other"
    "ksponspeech_eval_clean"
    "ksponspeech_eval_other"
    "zeroth_korean_test_clean"
    "zeroth_korean_test_other"
)

PROMPT_FILE="${BASE_PATH}/prompts.yaml"
PROMPT_NAME="${PROMPT_NAME:-v2}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
BATCH_SIZE=1
TENSOR_PARALLEL_SIZE=1

export PYTHONPATH="${BASE_PATH}/src:${PYTHONPATH}"
echo "[BACKEND] ${BACKEND} (model=${MODEL_PATH})"

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

    OPTS=" --input ${INPUT_JSONL} --output-dir ${OUTPUT_DIR} --backend ${BACKEND}"
    OPTS+=" --prompt-file ${PROMPT_FILE} --prompt-name ${PROMPT_NAME}"
    OPTS+=" --batch-size ${BATCH_SIZE} --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}"
    [ -n "${MODEL_PATH}" ] && OPTS+=" --model ${MODEL_PATH}"
    [ -n "${MAX_SAMPLES}" ] && OPTS+=" --max-samples ${MAX_SAMPLES}"

    mkdir -p "${OUTPUT_DIR}"
    echo "Running: python3 ${BASE_PATH}/ASR/run_asr_evaluation.py ${OPTS}"
    python3 ${BASE_PATH}/ASR/run_asr_evaluation.py ${OPTS}
done

echo ""
echo "ASR (gemini-flash) completed!"
