#!/bin/bash
# ============================================================
# SQA Evaluation — GPT-4o-mini Realtime (OpenAI Realtime API)
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-${SCRIPT_DIR}}"
while [ ! -f "${BASE_PATH}/prompts.yaml" ] && [ "${BASE_PATH}" != "/" ]; do
    BASE_PATH="$(dirname "${BASE_PATH}")"
done
OPENAI_API_KEY_FILE="${BASE_PATH}/openai_key.txt"
export OPENAI_API_KEY_FILE
[ -f "${OPENAI_API_KEY_FILE}" ] && export OPENAI_API_KEY="$(head -1 "${OPENAI_API_KEY_FILE}" | tr -d '\r\n')"

OUTPUT_BASE="${BASE_PATH}/results_real/SQA"
BACKEND="gpt_realtime_mini"
MODEL_PATH="gpt-audio-mini"
BENCHMARKS=("click_clean" "kobest_boolq_clean" "click_other" "kobest_boolq_other")
PROMPT_FILE="${BASE_PATH}/prompts.yaml"
PROMPT_NAME="${PROMPT_NAME:-v1}"
MAX_SAMPLES=""
BATCH_SIZE=1
TENSOR_PARALLEL_SIZE=1
SAVE_GENERATION=true
MAX_NEW_TOKENS=64
export PYTHONPATH="${BASE_PATH}/src:${PYTHONPATH}"
echo "[BACKEND] ${BACKEND} (Realtime API)"
echo "[MODEL] ${MODEL_PATH}"
for BENCH in "${BENCHMARKS[@]}"; do
    echo "Running SQA: ${BENCH}"
    INPUT_JSONL="${BASE_PATH}/SQA/${BENCH}.jsonl"
    OUTPUT_DIR="${OUTPUT_BASE}/${BACKEND}/${BENCH}"
    [ ! -f "${INPUT_JSONL}" ] && echo "[SKIP] ${INPUT_JSONL}" && continue
    OPTS=" --jsonl ${INPUT_JSONL} --output_dir ${OUTPUT_DIR} --backend ${BACKEND} --base_dir ${BASE_PATH} --max_new_tokens ${MAX_NEW_TOKENS} --batch_size ${BATCH_SIZE} --tensor_parallel_size ${TENSOR_PARALLEL_SIZE} --prompt-file ${PROMPT_FILE} --prompt-name ${PROMPT_NAME}"
    [ -n "${MODEL_PATH}" ] && OPTS+=" --model_path ${MODEL_PATH}"
    [ -n "${MAX_SAMPLES}" ] && OPTS+=" --max_samples ${MAX_SAMPLES}"
    [ "${SAVE_GENERATION}" = true ] && OPTS+=" --save_generation"
    mkdir -p "${OUTPUT_DIR}"
    python3 ${BASE_PATH}/SQA/evaluate_sqa.py ${OPTS}
done
echo "SQA (gpt-realtime-mini) completed!"
