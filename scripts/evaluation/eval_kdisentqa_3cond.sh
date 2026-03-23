#!/bin/bash

# ============================================================
# K-disentQA 3-조건 평가 (text_only / speech+new / speech+original)
#
# original_raw 가 있는 4개 벤치마크만 대상:
#   history_after_chosun_clean
#   history_before_chosun_clean
#   k-sports_clean
#   kpop_clean
#
# 사용법:
#   BACKEND=qwen MODEL_PATH=Qwen/Qwen2-Audio-7B-Instruct bash eval_kdisentqa_3cond.sh
#   BACKEND=voxtral3b MODEL_PATH=mistralai/Voxtral-Mini-3B-2507 bash eval_kdisentqa_3cond.sh
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-${SCRIPT_DIR}}"
while [ ! -f "${BASE_PATH}/prompts.yaml" ] && [ "${BASE_PATH}" != "/" ]; do
    BASE_PATH="$(dirname "${BASE_PATH}")"
done
KDISENTQA_DIR="${BASE_PATH}/SCA-QA"
AUDIO_NEW_BASE="${BASE_PATH}/audio/k-disentqa"
AUDIO_ORIG_BASE="${BASE_PATH}/audio/k-disentqa-original"
OUTPUT_BASE="${BASE_PATH}/results_real/SCA-QA_3cond"

BACKEND="${BACKEND:-qwen}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2-Audio-7B-Instruct}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
PROMPT_FILE="${BASE_PATH}/prompts.yaml"
PROMPT_NAME="${PROMPT_NAME:-v1}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

# 벤치마크 → (JSONL, new_context 오디오 디렉토리, original_context 오디오 디렉토리)
declare -A BENCH_JSONL
declare -A BENCH_NEW_DIR
declare -A BENCH_ORIG_DIR

BENCH_JSONL[history_after_chosun]="history_after_chosun_clean"
BENCH_NEW_DIR[history_after_chosun]="${AUDIO_NEW_BASE}/history_after_chosun_final"
BENCH_ORIG_DIR[history_after_chosun]="${AUDIO_ORIG_BASE}/history_after_chosun_final"

BENCH_JSONL[history_before_chosun]="history_before_chosun_clean"
BENCH_NEW_DIR[history_before_chosun]="${AUDIO_NEW_BASE}/history_before_chosun_final"
BENCH_ORIG_DIR[history_before_chosun]="${AUDIO_ORIG_BASE}/history_before_chosun_final"

BENCH_JSONL[k-sports]="k-sports_clean"
BENCH_NEW_DIR[k-sports]="${AUDIO_NEW_BASE}/k-sports_final"
BENCH_ORIG_DIR[k-sports]="${AUDIO_ORIG_BASE}/k-sports_final"

BENCH_JSONL[kpop]="kpop_clean"
BENCH_NEW_DIR[kpop]="${AUDIO_NEW_BASE}/kpop_final"
BENCH_ORIG_DIR[kpop]="${AUDIO_ORIG_BASE}/kpop_final"

BENCH_KEYS=("history_after_chosun" "history_before_chosun" "k-sports" "kpop")

# ============================================================
export PYTHONPATH="${BASE_PATH}/src:${PYTHONPATH}"
echo "[K-disentQA 3-cond] BACKEND=${BACKEND}  MODEL=${MODEL_PATH}"

for KEY in "${BENCH_KEYS[@]}"; do
    BENCH="${BENCH_JSONL[$KEY]}"
    INPUT_JSONL="${KDISENTQA_DIR}/${BENCH}.jsonl"
    OUTPUT_DIR="${OUTPUT_BASE}/${BACKEND}/${BENCH}"
    SPEECH_DIR="${BENCH_NEW_DIR[$KEY]}"
    ORIG_DIR="${BENCH_ORIG_DIR[$KEY]}"

    echo ""
    echo "=========================================="
    echo "K-disentQA 3-cond: ${BENCH}"
    echo "  new_context  : ${SPEECH_DIR}"
    echo "  orig_context : ${ORIG_DIR}"
    echo "=========================================="

    [ ! -f "${INPUT_JSONL}" ] && echo "[SKIP] ${INPUT_JSONL} not found" && continue

    OPTS=" --jsonl ${INPUT_JSONL}"
    OPTS+=" --output_dir ${OUTPUT_DIR}"
    OPTS+=" --speech-dir ${SPEECH_DIR}"
    OPTS+=" --original-speech-dir ${ORIG_DIR}"
    OPTS+=" --backend ${BACKEND} --model_path ${MODEL_PATH}"
    OPTS+=" --tensor_parallel_size ${TENSOR_PARALLEL_SIZE}"
    OPTS+=" --prompt-file ${PROMPT_FILE}"
    [ -n "${PROMPT_NAME}" ] && OPTS+=" --prompt-name ${PROMPT_NAME}"
    [ -n "${MAX_SAMPLES}" ] && OPTS+=" --max_samples ${MAX_SAMPLES}"

    mkdir -p "${OUTPUT_DIR}"
    python3 "${KDISENTQA_DIR}/evaluate_with_original.py" ${OPTS}
done

echo ""
echo "K-disentQA 3-cond (${BACKEND}) done."
