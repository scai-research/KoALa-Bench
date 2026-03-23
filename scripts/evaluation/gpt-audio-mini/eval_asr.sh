#!/bin/bash
# ============================================================
# ASR Evaluation — gpt-audio-mini (Chat audio input, batch, 저렴)
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-${SCRIPT_DIR}}"
while [ ! -f "${BASE_PATH}/prompts.yaml" ] && [ "${BASE_PATH}" != "/" ]; do
    BASE_PATH="$(dirname "${BASE_PATH}")"
done
# gpt_realtime_mini: 키 파일 경로 (변경 시 여기만 수정)
OPENAI_API_KEY_FILE="${BASE_PATH}/openai_key.txt"
export OPENAI_API_KEY_FILE
[ -f "${OPENAI_API_KEY_FILE}" ] && export OPENAI_API_KEY="$(head -1 "${OPENAI_API_KEY_FILE}" | tr -d '\r\n')"

OUTPUT_BASE="${BASE_PATH}/results_real/ASR"
BACKEND="gpt_realtime_mini"
MODEL_PATH="gpt-audio-mini"
ASR_DATA_PATH="${BASE_PATH}/ASR"
BENCHMARKS=("common_voice_korea_clean" "common_voice_korea_other" "ksponspeech_eval_clean" "ksponspeech_eval_other" "zeroth_korean_test_clean" "zeroth_korean_test_other")
PROMPT_FILE="${BASE_PATH}/prompts.yaml"
# API 비용: prompt-file 쓸 때 4개 전부 도는 대신 하나만 (prompts.yaml 의 name 과 일치)
PROMPT_NAME="${PROMPT_NAME:-v2}"
MAX_SAMPLES=""
BATCH_SIZE=1
TENSOR_PARALLEL_SIZE=1
export PYTHONPATH="${BASE_PATH}/src:${PYTHONPATH}"
echo "[BACKEND] ${BACKEND} (model=${MODEL_PATH})"
for BENCH in "${BENCHMARKS[@]}"; do
    echo "Running ASR: ${BENCH}"
    INPUT_JSONL="${ASR_DATA_PATH}/${BENCH}.jsonl"
    OUTPUT_DIR="${OUTPUT_BASE}/${BACKEND}/${BENCH}"
    [ ! -f "${INPUT_JSONL}" ] && echo "[SKIP] ${INPUT_JSONL}" && continue
    OPTS=" --input ${INPUT_JSONL} --output-dir ${OUTPUT_DIR} --backend ${BACKEND} --prompt-file ${PROMPT_FILE} --prompt-name ${PROMPT_NAME} --batch-size ${BATCH_SIZE} --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}"
    [ -n "${MODEL_PATH}" ] && OPTS+=" --model ${MODEL_PATH}"
    [ -n "${MAX_SAMPLES}" ] && OPTS+=" --max-samples ${MAX_SAMPLES}"
    mkdir -p "${OUTPUT_DIR}"
    python3 ${BASE_PATH}/ASR/run_asr_evaluation.py ${OPTS}
done
echo "ASR (gpt-realtime-mini) completed!"
