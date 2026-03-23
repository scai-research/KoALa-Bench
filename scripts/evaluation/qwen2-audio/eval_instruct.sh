#!/bin/bash

# ============================================================
# Instruct Evaluation Script (GPT Score)
# ============================================================

# paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-${SCRIPT_DIR}}"
while [ ! -f "${BASE_PATH}/prompts.yaml" ] && [ "${BASE_PATH}" != "/" ]; do
    BASE_PATH="$(dirname "${BASE_PATH}")"
done
MODEL_PATH="Qwen/Qwen2-Audio-7B-Instruct"
OUTPUT_BASE="${BASE_PATH}/results_real/Instruct"

# model (transformers + flash_attn 배치: qwen, qwen3)
BACKEND="qwen"

# benchmarks (주석 처리로 선택)
# audio 경로: audio/<벤치마크이름>/<index>.wav
# qwen 미진행 벤치마크만: vicuna_noisy, vicuna_other, alpaca_other, openhermes_other, kudge_other
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
RUN_INFERENCE=true  # true면 추론 후 평가, false면 기존 prediction 파일로 평가

# evaluation options
MAX_SAMPLES=""  # 빈 값이면 전체
BATCH_SIZE=8    
GPT_MODEL="gpt-4o-mini"

# GPT API (생성 답변 평가용). 키 파일 또는 환경변수 OPENAI_API_KEY 사용.
OPENAI_API_KEY_FILE="${BASE_PATH}/openai_key.txt"

# options — 프롬프트: prompts.yaml 에서 4개 변형 로드
PROMPT_FILE="${BASE_PATH}/prompts.yaml"

# ============================================================
# Run
# ============================================================
export PYTHONPATH="${BASE_PATH}/src:${PYTHONPATH}"
echo "[BACKEND] ${BACKEND}"

for BENCH in "${BENCHMARKS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running Instruct: ${BENCH}"
    echo "=========================================="
    
    INPUT_JSONL="${BASE_PATH}/Instruct/${BENCH}.jsonl"
    OUTPUT_DIR="${OUTPUT_BASE}/${BACKEND}/${BENCH}"
    
    OPTS=""
    OPTS+=" --original_jsonl ${INPUT_JSONL}"
    OPTS+=" --output_dir ${OUTPUT_DIR}"
    OPTS+=" --backend ${BACKEND}"
    OPTS+=" --base_dir ${BASE_PATH}"
    OPTS+=" --gpt_model ${GPT_MODEL}"
    OPTS+=" --prompt-file ${PROMPT_FILE}"
    if [ -f "${OPENAI_API_KEY_FILE}" ]; then
        OPTS+=" --openai_api_key_file ${OPENAI_API_KEY_FILE}"
    fi
    if [ "${RUN_INFERENCE}" = true ]; then
        OPTS+=" --run_inference"
        OPTS+=" --batch_size ${BATCH_SIZE}"
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
echo "All Instruct benchmarks completed!"
