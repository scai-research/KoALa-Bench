#!/bin/bash

# ============================================================
# ASR Evaluation Script — Gemma-3n (vLLM)
# ============================================================

# paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-${SCRIPT_DIR}}"
while [ ! -f "${BASE_PATH}/prompts.yaml" ] && [ "${BASE_PATH}" != "/" ]; do
    BASE_PATH="$(dirname "${BASE_PATH}")"
done
OUTPUT_BASE="${BASE_PATH}/results_real/ASR"

# JSONL이 있는 위치
ASR_DATA_PATH="${BASE_PATH}/ASR"

# model
BACKEND="gemma3n_vllm"
MODEL_PATH="google/gemma-3n-E4B-it"  # 또는 google/gemma-3n-E4B-it

# benchmarks (주석 처리로 선택)
BENCHMARKS=(
    "common_voice_korea_clean"
    # "common_voice_korea_other"
    # "ksponspeech_eval_clean"
    # "ksponspeech_eval_other"
    # "zeroth_korean_test_clean"
    # "zeroth_korean_test_other"
)

# options — 프롬프트: prompts.yaml 에서 4개 변형 로드
PROMPT_FILE="${BASE_PATH}/prompts.yaml"
# v1만 테스트하려면: PROMPT_NAME="v1"  (비우면 4개 모두 실행)
PROMPT_NAME="v4"
MAX_SAMPLES="10"  # 빈 값이면 전체
BATCH_SIZE=1     # vLLM 배치 크기 (vLLM이 내부적으로 최적화)
# Gemma-3n + vLLM에서 TP=2 사용 시 mat1/mat2 shape 불일치 발생 → 단일 GPU 사용
TENSOR_PARALLEL_SIZE=1

# ============================================================
# Run
# ============================================================
export PYTHONPATH="${BASE_PATH}/src:${PYTHONPATH}"
echo "[BACKEND] ${BACKEND} (vLLM)"
echo "[MODEL] ${MODEL_PATH}"
echo "[TENSOR_PARALLEL] ${TENSOR_PARALLEL_SIZE}"
echo "[PROMPT_FILE] ${PROMPT_FILE}"
echo "[PROMPT_NAME] ${PROMPT_NAME:-'(all)'}"

# 디버깅: 사용할 ASR 프롬프트 내용 출력
if [ -f "${PROMPT_FILE}" ]; then
    echo ""
    echo "────────── ASR 프롬프트 (사용 대상) ──────────"
    python3 -c "
import yaml, sys
with open('${PROMPT_FILE}', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)
prompts = cfg.get('asr', [])
name_filter = '${PROMPT_NAME}'
if name_filter:
    prompts = [p for p in prompts if p.get('name') == name_filter]
for p in prompts:
    print(f\"  name: {p.get('name', '?')}\")
    print(f\"  prompt: {repr(p.get('prompt', ''))}\")
    print()
"
    echo "─────────────────────────────────────────────"
fi

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
    [ -n "${PROMPT_NAME}" ] && OPTS+=" --prompt-name ${PROMPT_NAME}"
    OPTS+=" --batch-size ${BATCH_SIZE}"
    OPTS+=" --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}"

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

    # 개별 예측 결과 출력
    PRED_FILE=$(find ${OUTPUT_DIR} -name "*_results.jsonl" -o -name "*_predictions.jsonl" | head -1)
    if [ -n "${PRED_FILE}" ] && [ -f "${PRED_FILE}" ]; then
        echo ""
        echo "────────── 예측 결과 ──────────"
        python3 -c "
import json, sys
with open('${PRED_FILE}') as f:
    for line in f:
        r = json.loads(line)
        print(f\"[{r.get('index','?')}] CER={r.get('cer',0):.4f}\")
        print(f\"  GT:   {r.get('gt_normalized','')}\")
        print(f\"  PRED: {r.get('pred_normalized','')}\")
        print()
"
        echo "───────────────────────────────"
    fi
done

echo ""
echo "All ASR benchmarks (Gemma-3n vLLM) completed!"
