#!/bin/bash
# ============================================================
# Translation Evaluation — GPT-4o-mini Realtime (OpenAI Realtime API)
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-${SCRIPT_DIR}}"
while [ ! -f "${BASE_PATH}/prompts.yaml" ] && [ "${BASE_PATH}" != "/" ]; do
    BASE_PATH="$(dirname "${BASE_PATH}")"
done
OPENAI_API_KEY_FILE="${BASE_PATH}/openai_key.txt"
export OPENAI_API_KEY_FILE
[ -f "${OPENAI_API_KEY_FILE}" ] && export OPENAI_API_KEY="$(head -1 "${OPENAI_API_KEY_FILE}" | tr -d '\r\n')"

MODEL_PATH="gpt-audio-mini"
OUTPUT_BASE="${BASE_PATH}/results_real/Translation"
# JSONL: qwen3-omni 등과 동일하게 hslim/Translation/output 사용
TRANSLATION_DATA_PATH="${BASE_PATH}/Translation"
BACKEND="gpt_realtime_mini"
BENCHMARKS=("etri_tst-COMMON_clean" "etri_tst-HE_clean")
PROMPT_FILE="${BASE_PATH}/prompts.yaml"
PROMPT_NAME="${PROMPT_NAME:-v1}"
MAX_SAMPLES=""
TOKENIZE_METHOD="character"
GT_FIELD="answer_ko"
BATCH_SIZE=1
TENSOR_PARALLEL_SIZE=1
export PYTHONPATH="${BASE_PATH}/src:${PYTHONPATH}"
echo "[BACKEND] ${BACKEND} (Realtime API)"
echo "[MODEL] ${MODEL_PATH}"
for BENCH in "${BENCHMARKS[@]}"; do
    echo "Running Translation: ${BENCH}"
    INPUT_JSONL="${TRANSLATION_DATA_PATH}/${BENCH}.jsonl"
    OUTPUT_DIR="${OUTPUT_BASE}/${BACKEND}/${BENCH}"
    [ ! -f "${INPUT_JSONL}" ] && echo "[SKIP] ${INPUT_JSONL}" && continue
    OPTS=" --input ${INPUT_JSONL} --output-dir ${OUTPUT_DIR} --backend ${BACKEND} --prompt-file ${PROMPT_FILE} --prompt-name ${PROMPT_NAME} --tokenize ${TOKENIZE_METHOD} --gt-field ${GT_FIELD} --batch-size ${BATCH_SIZE} --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}"
    [ -n "${MODEL_PATH}" ] && OPTS+=" --model ${MODEL_PATH}"
    [ -n "${MAX_SAMPLES}" ] && OPTS+=" --max-samples ${MAX_SAMPLES}"
    mkdir -p "${OUTPUT_DIR}"
    python3 ${BASE_PATH}/Translation/run_translation_evaluation.py ${OPTS}
done
echo "Translation (gpt-realtime-mini) completed!"
