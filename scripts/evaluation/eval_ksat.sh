#!/bin/bash
# ============================================================
# K-SAT (수능 듣기) 평가 — 범용 쉘 스크립트
#
# 사용법:
#   BACKEND=qwen MODEL_PATH=Qwen/Qwen2-Audio-7B-Instruct bash eval_ksat.sh
#   BACKEND=gemini_flash MODEL_PATH=gemini-2.5-flash-lite bash eval_ksat.sh
#   BACKEND=voxtral3b MODEL_PATH=mistralai/Voxtral-Mini-3B-2507 bash eval_ksat.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-${SCRIPT_DIR}}"
while [ ! -f "${BASE_PATH}/prompts.yaml" ] && [ "${BASE_PATH}" != "/" ]; do
    BASE_PATH="$(dirname "${BASE_PATH}")"
done
SQA_DIR="${BASE_PATH}/SQA"
OUTPUT_BASE="${BASE_PATH}/results_real/K-SAT"

BACKEND="${BACKEND:-qwen}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2-Audio-7B-Instruct}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

# SQA와 동일 4개 프롬프트 사용. Gemini는 v1만 쓰려면 PROMPT_NAME=v1
PROMPT_FILE="${BASE_PATH}/prompts.yaml"
PROMPT_NAME="${PROMPT_NAME:-}"   # 비우면 4개 전부 실행 후 best 채택. v1 이면 해당만

# K-SAT JSONL 파일 (clean + noise)
BENCHMARKS=(
    "KCSAT_clean"
    "KCSAT_noise"
)

# Gemini API 키
GEMINI_API_KEY_FILE="${BASE_PATH}/gemini_key.txt"
[ -f "${GEMINI_API_KEY_FILE}" ] && export GEMINI_API_KEY="$(head -1 "${GEMINI_API_KEY_FILE}" | tr -d '\r\n')"
# OpenAI API 키
OPENAI_API_KEY_FILE="${BASE_PATH}/openai_key.txt"
[ -f "${OPENAI_API_KEY_FILE}" ] && export OPENAI_API_KEY="$(head -1 "${OPENAI_API_KEY_FILE}" | tr -d '\r\n')"

# ============================================================
export PYTHONPATH="${BASE_PATH}/src:${PYTHONPATH}"
echo "[K-SAT] BACKEND=${BACKEND}  MODEL=${MODEL_PATH}"

for BENCH in "${BENCHMARKS[@]}"; do
    INPUT_JSONL="${SQA_DIR}/${BENCH}.jsonl"
    OUTPUT_DIR="${OUTPUT_BASE}/${BACKEND}/${BENCH}"

    echo ""
    echo "=========================================="
    echo "K-SAT: ${BENCH}"
    echo "=========================================="
    [ ! -f "${INPUT_JSONL}" ] && echo "[SKIP] ${INPUT_JSONL} not found" && continue

    OPTS=" --jsonl ${INPUT_JSONL} --output_dir ${OUTPUT_DIR}"
    OPTS+=" --backend ${BACKEND} --model_path ${MODEL_PATH}"
    OPTS+=" --tensor_parallel_size ${TENSOR_PARALLEL_SIZE}"
    OPTS+=" --prompt-file ${PROMPT_FILE}"
    [ -n "${PROMPT_NAME}" ] && OPTS+=" --prompt-name ${PROMPT_NAME}"
    [ -n "${MAX_SAMPLES}" ] && OPTS+=" --max_samples ${MAX_SAMPLES}"

    mkdir -p "${OUTPUT_DIR}"
    python3 "${SQA_DIR}/evaluate_ksat.py" ${OPTS}
done

echo ""
echo "K-SAT (${BACKEND}) done."
