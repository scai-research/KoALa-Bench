#!/bin/bash

# ============================================================
# ASR Evaluation Script
# ============================================================

# paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-${SCRIPT_DIR}}"
while [ ! -f "${BASE_PATH}/prompts.yaml" ] && [ "${BASE_PATH}" != "/" ]; do
    BASE_PATH="$(dirname "${BASE_PATH}")"
done
# HF 캐시 경로 사용 시 API 호출 없이 로드 (인터넷 불필요). 비우면 백엔드 기본 모델 ID 사용.
#MODEL_PATH="Qwen/Qwen2-Audio-7B-Instruct"
OUTPUT_BASE="${BASE_PATH}/results_real/ASR"

# JSONL이 있는 위치
ASR_DATA_PATH="${BASE_PATH}/ASR"

# model: qwen (Qwen2-Audio), qwen3 (Qwen3-Omni HF), qwen3_vllm (Qwen3-Omni vLLM)
BACKEND="qwen"
MODEL_PATH=""

# benchmarks (주석 처리로 선택)
BENCHMARKS=(
    "common_voice_korea_clean"
    "common_voice_korea_other"
    "ksponspeech_eval_clean"
    "ksponspeech_eval_other"
    "zeroth_korean_test_clean"
    "zeroth_korean_test_other"
)

# options — 프롬프트: prompts.yaml 에서 4개 변형 로드
PROMPT_FILE="${BASE_PATH}/prompts.yaml"
MAX_SAMPLES=""  # 빈 값이면 전체
BATCH_SIZE=8    # flash_attn 배치 크기 (GPU 메모리에 맞게 조절)

# ============================================================
# Run
# ============================================================
export PYTHONPATH="${BASE_PATH}/src:${PYTHONPATH}"
echo "[BACKEND] ${BACKEND}"

for BENCH in "${BENCHMARKS[@]}"; do
    echo ""
    echo "=========================================="
    echo "Running ASR: ${BENCH}"
    echo "=========================================="
    
    INPUT_JSONL="${ASR_DATA_PATH}/${BENCH}.jsonl"
    OUTPUT_DIR="${OUTPUT_BASE}/${BACKEND}/${BENCH}"
    
    if [ ! -f "${INPUT_JSONL}" ]; then
        echo "[SKIP] JSONL not found: ${INPUT_JSONL}"
        continue
    fi
    
    OPTS=""
    OPTS+=" --input ${INPUT_JSONL}"
    OPTS+=" --output-dir ${OUTPUT_DIR}"
    OPTS+=" --backend ${BACKEND}"
    OPTS+=" --prompt-file ${PROMPT_FILE}"
    OPTS+=" --batch-size ${BATCH_SIZE}"
    
    if [ -n "${MODEL_PATH}" ]; then
        OPTS+=" --model ${MODEL_PATH}"
    fi
    
    if [ -n "${MAX_SAMPLES}" ]; then
        OPTS+=" --max-samples ${MAX_SAMPLES}"
    fi
    
    CMD="python3 ${BASE_PATH}/ASR/run_asr_evaluation.py ${OPTS}"
    echo "Running: ${CMD}"
    mkdir -p ${OUTPUT_DIR}
    eval ${CMD}
done

echo ""
echo "All ASR benchmarks completed!"
