#!/bin/bash
# ============================================================
# LSQA (Long Speech QA) — Voxtral Mini 3B
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-${SCRIPT_DIR}}"
while [ ! -f "${BASE_PATH}/prompts.yaml" ] && [ "${BASE_PATH}" != "/" ]; do
    BASE_PATH="$(dirname "${BASE_PATH}")"
done
EVAL_SCRIPT="${BASE_PATH}/PA-QA/evaluate_lsqa.py"
OUTPUT_BASE="${BASE_PATH}/results_real/LSQA"
LSQA_DIR="${BASE_PATH}/PA-QA"

BACKEND="voxtral3b"
MODEL_PATH="mistralai/Voxtral-Mini-3B-2507"
TENSOR_PARALLEL_SIZE=1

AUDIO_BASE_CLEAN="${BASE_PATH}/audio"
AUDIO_BASE_NOISE="${BASE_PATH}"

BENCHMARKS=(
    "mctest_clean"
    "mctest_other"
)

SAVE_GENERATION=true
MAX_NEW_TOKENS=64

PROMPT_FILE="${BASE_PATH}/prompts.yaml"
# PROMPT_NAME 미설정 → 4개 프롬프트(v1~v4) 모두 사용

# ============================================================
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

    [ "${SAVE_GENERATION}" = true ] && CMD+=" --save_generation"
    [ -n "${PROMPT_FILE}" ] && CMD+=" --prompt-file ${PROMPT_FILE}"

    echo "JSONL:    ${INPUT_JSONL}"
    echo "출력:     ${OUTPUT_DIR}"
    echo "base_dir: ${AUDIO_BASE}"
    mkdir -p "${OUTPUT_DIR}"
    eval ${CMD}
done

echo ""
echo "LSQA (Voxtral Mini 3B) 완료!"
