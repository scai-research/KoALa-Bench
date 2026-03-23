#!/bin/bash
# ============================================================
# K-disentQA Evaluation — Gemini Flash (gemini_flash)
# ============================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-${SCRIPT_DIR}}"
while [ ! -f "${BASE_PATH}/prompts.yaml" ] && [ "${BASE_PATH}" != "/" ]; do
    BASE_PATH="$(dirname "${BASE_PATH}")"
done
GEMINI_API_KEY_FILE="${BASE_PATH}/gemini_key.txt"
[ -f "${GEMINI_API_KEY_FILE}" ] && export GEMINI_API_KEY="$(head -1 "${GEMINI_API_KEY_FILE}" | tr -d '\r\n')"

KDISENTQA_DIR="${BASE_PATH}/SCA-QA"
AUDIO_BASE="${BASE_PATH}/audio/k-disentqa"
AUDIO_NOISE_BASE="${BASE_PATH}/audio_noise/k-disentqa"
OUTPUT_BASE="${BASE_PATH}/results_real/SCA-QA"
BACKEND="gemini_flash"
MODEL_PATH="gemini-2.5-flash-lite"
TENSOR_PARALLEL_SIZE=1

# clean/other JSONL 사용 (SCA-QA/*_clean|*_other.jsonl)
BENCHMARKS=(
    "history_after_chosun_other"
    "history_after_chosun_clean"
    "history_before_chosun_clean"
    "history_before_chosun_other"
    "k-sports_clean"
    "k-sports_other"
    "kpop_clean"
    "kpop_other"
)
MAX_SAMPLES="${MAX_SAMPLES:-}"
# 프롬프트 v1만 사용 (여러 개 쓰려면 PROMPT_NAME 비우기)
PROMPT_FILE="${BASE_PATH}/prompts.yaml"
PROMPT_NAME="v1"

export PYTHONPATH="${BASE_PATH}/src:${PYTHONPATH}"
echo "[K-disentQA] BACKEND=${BACKEND} MODEL=${MODEL_PATH}"

for BENCH in "${BENCHMARKS[@]}"; do
    echo ""
    echo "=========================================="
    echo "K-disentQA: ${BENCH}"
    echo "=========================================="

    INPUT_JSONL="${KDISENTQA_DIR}/${BENCH}.jsonl"
    OUTPUT_DIR="${OUTPUT_BASE}/${BACKEND}/${BENCH}"
    [ ! -f "${INPUT_JSONL}" ] && echo "[SKIP] ${INPUT_JSONL} not found" && continue

    BENCH_FOR_AUDIO="${BENCH}"
    [[ "${BENCH}" == *_clean ]] && BENCH_FOR_AUDIO="${BENCH%_clean}"
    if [[ "${BENCH_FOR_AUDIO}" == *_other ]]; then
        BASE_BENCH="${BENCH_FOR_AUDIO%_other}"
        SPEECH_DIR="${AUDIO_NOISE_BASE}/${BASE_BENCH}_final_noise"
        AUDIO_SUFFIX="_tts_noisy.wav"
    else
        if [[ "${BENCH_FOR_AUDIO}" == *_tts ]]; then
            BASE_FOR_AUDIO="${BENCH_FOR_AUDIO%_tts}"
            SPEECH_DIR="${AUDIO_BASE}/${BASE_FOR_AUDIO}_final"
        else
            SPEECH_DIR="${AUDIO_BASE}/${BENCH_FOR_AUDIO}_final"
        fi
        AUDIO_SUFFIX="_tts.wav"
    fi

    OPTS=" --jsonl ${INPUT_JSONL} --output_dir ${OUTPUT_DIR} --speech_output_dir ${SPEECH_DIR}"
    OPTS+=" --audio_suffix ${AUDIO_SUFFIX}"
    OPTS+=" --backend ${BACKEND} --model_path ${MODEL_PATH} --tensor_parallel_size ${TENSOR_PARALLEL_SIZE}"
    OPTS+=" --prompt-file ${PROMPT_FILE}"
    [ -n "${PROMPT_NAME}" ] && OPTS+=" --prompt-name ${PROMPT_NAME}"
    [ -n "${MAX_SAMPLES}" ] && OPTS+=" --max_samples ${MAX_SAMPLES}"

    mkdir -p "${OUTPUT_DIR}"
    python3 "${KDISENTQA_DIR}/evaluate.py" ${OPTS}
done
echo "K-disentQA (gemini-flash) done."
