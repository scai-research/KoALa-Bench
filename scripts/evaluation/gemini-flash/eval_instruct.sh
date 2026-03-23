#!/bin/bash
# ============================================================
# Instruct Evaluation — Gemini Flash (gemini_flash)
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-${SCRIPT_DIR}}"
while [ ! -f "${BASE_PATH}/prompts.yaml" ] && [ "${BASE_PATH}" != "/" ]; do
    BASE_PATH="$(dirname "${BASE_PATH}")"
done
GEMINI_API_KEY_FILE="${BASE_PATH}/gemini_key.txt"
[ -f "${GEMINI_API_KEY_FILE}" ] && export GEMINI_API_KEY="$(head -1 "${GEMINI_API_KEY_FILE}" | tr -d '\r\n')"

OUTPUT_BASE="${BASE_PATH}/results_real/Instruct"
BACKEND="gemini_flash"
MODEL_PATH="gemini-2.5-flash-lite"

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
# 프롬프트 v1만 사용 (여러 개 쓰려면 PROMPT_NAME 비우기)
PROMPT_FILE="${BASE_PATH}/prompts.yaml"
PROMPT_NAME="v1"
MAX_SAMPLES="${MAX_SAMPLES:-}"
BATCH_SIZE=1
TENSOR_PARALLEL_SIZE=1
GPT_MODEL="gpt-4o-mini"
OPENAI_API_KEY_FILE="${BASE_PATH}/openai_key.txt"

export PYTHONPATH="${BASE_PATH}/src:${PYTHONPATH}"
echo "[BACKEND] ${BACKEND} (model=${MODEL_PATH})"

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

    OPTS=" --original_jsonl ${INPUT_JSONL} --output_dir ${OUTPUT_DIR}"
    OPTS+=" --backend ${BACKEND} --base_dir ${BASE_PATH}"
    OPTS+=" --prompt-file ${PROMPT_FILE} --gpt_model ${GPT_MODEL}"
    [ -n "${PROMPT_NAME}" ] && OPTS+=" --prompt-name ${PROMPT_NAME}"
    [ -f "${OPENAI_API_KEY_FILE}" ] && OPTS+=" --openai_api_key_file ${OPENAI_API_KEY_FILE}"
    if [ "${RUN_INFERENCE}" = true ]; then
        OPTS+=" --run_inference --batch_size ${BATCH_SIZE} --tensor_parallel_size ${TENSOR_PARALLEL_SIZE}"
        [ -n "${MODEL_PATH}" ] && OPTS+=" --model_path ${MODEL_PATH}"
    fi
    [ -n "${MAX_SAMPLES}" ] && OPTS+=" --max_samples ${MAX_SAMPLES}"

    mkdir -p "${OUTPUT_DIR}"
    echo "Running: python3 ${BASE_PATH}/Instruct/evaluate_instruct.py ${OPTS}"
    python3 ${BASE_PATH}/Instruct/evaluate_instruct.py ${OPTS}
done

echo ""
echo "Instruct (gemini-flash) completed!"
