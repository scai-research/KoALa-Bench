#!/bin/bash
# ============================================================
# Instruct Evaluation — GPT-4o-mini Realtime (OpenAI Realtime API)
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-${SCRIPT_DIR}}"
while [ ! -f "${BASE_PATH}/prompts.yaml" ] && [ "${BASE_PATH}" != "/" ]; do
    BASE_PATH="$(dirname "${BASE_PATH}")"
done
OPENAI_API_KEY_FILE="${BASE_PATH}/openai_key.txt"
export OPENAI_API_KEY_FILE
[ -f "${OPENAI_API_KEY_FILE}" ] && export OPENAI_API_KEY="$(head -1 "${OPENAI_API_KEY_FILE}" | tr -d '\r\n')"

OUTPUT_BASE="${BASE_PATH}/results_real/Instruct"
BACKEND="gpt_realtime_mini"
MODEL_PATH="gpt-audio-mini"
BENCHMARKS=("vicuna_clean" "alpaca_clean" "openhermes_clean" "kudge_clean" "vicuna_other" "alpaca_other" "openhermes_other" "kudge_other")
RUN_INFERENCE=true
PROMPT_FILE="${BASE_PATH}/prompts.yaml"
PROMPT_NAME="${PROMPT_NAME:-v1}"
MAX_SAMPLES=""
BATCH_SIZE=1
TENSOR_PARALLEL_SIZE=1
GPT_MODEL="gpt-4o-mini"
export PYTHONPATH="${BASE_PATH}/src:${PYTHONPATH}"
echo "[BACKEND] ${BACKEND} (Realtime API)"
echo "[MODEL] ${MODEL_PATH}"
for BENCH in "${BENCHMARKS[@]}"; do
    echo "Running Instruct: ${BENCH}"
    INPUT_JSONL="${BASE_PATH}/Instruct/${BENCH}.jsonl"
    OUTPUT_DIR="${OUTPUT_BASE}/${BACKEND}/${BENCH}"
    [ ! -f "${INPUT_JSONL}" ] && echo "[SKIP] ${INPUT_JSONL}" && continue
    OPTS=" --original_jsonl ${INPUT_JSONL} --output_dir ${OUTPUT_DIR} --backend ${BACKEND} --base_dir ${BASE_PATH} --prompt-file ${PROMPT_FILE} --prompt-name ${PROMPT_NAME} --gpt_model ${GPT_MODEL}"
    [ -f "${OPENAI_API_KEY_FILE}" ] && OPTS+=" --openai_api_key_file ${OPENAI_API_KEY_FILE}"
    [ "${RUN_INFERENCE}" = true ] && OPTS+=" --run_inference --batch_size ${BATCH_SIZE} --tensor_parallel_size ${TENSOR_PARALLEL_SIZE}" && [ -n "${MODEL_PATH}" ] && OPTS+=" --model_path ${MODEL_PATH}"
    [ -n "${MAX_SAMPLES}" ] && OPTS+=" --max_samples ${MAX_SAMPLES}"
    mkdir -p "${OUTPUT_DIR}"
    python3 ${BASE_PATH}/Instruct/evaluate_instruct.py ${OPTS}
done
echo "Instruct (gpt-realtime-mini) completed!"
