#!/bin/bash

# ============================================================
# Instruct Evaluation Script — Gemma-3n (vLLM)
# ============================================================

# paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-${SCRIPT_DIR}}"
while [ ! -f "${BASE_PATH}/prompts.yaml" ] && [ "${BASE_PATH}" != "/" ]; do
    BASE_PATH="$(dirname "${BASE_PATH}")"
done
OUTPUT_BASE="${BASE_PATH}/results_real/Instruct"

# model
BACKEND="gemma3n_vllm"
MODEL_PATH="google/gemma-3n-E4B-it"  # 또는 google/gemma-3n-E4B-it

# benchmarks (주석 처리로 선택)
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

# inference options
RUN_INFERENCE=true

# evaluation options — 프롬프트: prompts.yaml 에서 4개 변형 로드
PROMPT_FILE="${BASE_PATH}/prompts.yaml"
MAX_SAMPLES=""
BATCH_SIZE=8          # vLLM 배치 크기 (vLLM이 내부적으로 최적화)
# Gemma-3n + vLLM에서 TP=2 사용 시 shape 불일치 → 단일 GPU
TENSOR_PARALLEL_SIZE=1
GPT_MODEL="gpt-4o-mini"

# GPT API (키 파일 사용, 없으면 환경변수 OPENAI_API_KEY)
OPENAI_API_KEY_FILE="${BASE_PATH}/openai_key.txt"

# ============================================================
# Run
# ============================================================
export PYTHONPATH="${BASE_PATH}/src:${PYTHONPATH}"
echo "[BACKEND] ${BACKEND} (vLLM)"
echo "[MODEL] ${MODEL_PATH}"
echo "[TENSOR_PARALLEL] ${TENSOR_PARALLEL_SIZE}"

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

    OPTS=""
    OPTS+=" --original_jsonl ${INPUT_JSONL}"
    OPTS+=" --output_dir ${OUTPUT_DIR}"
    OPTS+=" --backend ${BACKEND}"
    OPTS+=" --base_dir ${BASE_PATH}"
    OPTS+=" --prompt-file ${PROMPT_FILE}"
    OPTS+=" --gpt_model ${GPT_MODEL}"
    if [ -f "${OPENAI_API_KEY_FILE}" ]; then
        OPTS+=" --openai_api_key_file ${OPENAI_API_KEY_FILE}"
    fi
    if [ "${RUN_INFERENCE}" = true ]; then
        OPTS+=" --run_inference"
        OPTS+=" --batch_size ${BATCH_SIZE}"
        OPTS+=" --tensor_parallel_size ${TENSOR_PARALLEL_SIZE}"
        if [ -n "${MODEL_PATH}" ]; then
            OPTS+=" --model_path ${MODEL_PATH}"
        fi
    fi

    if [ -n "${MAX_SAMPLES}" ]; then
        OPTS+=" --max_samples ${MAX_SAMPLES}"
    fi

    CMD="python3 ${BASE_PATH}/Instruct/evaluate_instruct.py ${OPTS}"
    echo "Running: ${CMD}"
    mkdir -p ${OUTPUT_DIR}
    eval ${CMD}
done

echo ""
echo "All Instruct benchmarks (Gemma-3n vLLM) completed!"
