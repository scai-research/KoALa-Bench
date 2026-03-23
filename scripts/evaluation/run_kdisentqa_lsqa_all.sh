#!/bin/bash
# ============================================================
# 5개 모델 K-disentQA + LSQA 일괄 실행 (qwen2, qwen3-omni, gemma3n, gpt, voxtral)
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-${SCRIPT_DIR}}"
while [ ! -f "${BASE_PATH}/prompts.yaml" ] && [ "${BASE_PATH}" != "/" ]; do
    BASE_PATH="$(dirname "${BASE_PATH}")"
done
cd "${BASE_PATH}"

export PYTHONPATH="${BASE_PATH}/src:${PYTHONPATH}"
[ -f "${BASE_PATH}/openai_key.txt" ] && export OPENAI_API_KEY="$(head -1 "${BASE_PATH}/openai_key.txt" | tr -d '\r\n')"

echo "============================================"
echo "run_kdisentqa_lsqa_all: 5 models × (K-disentQA + LSQA)"
echo "============================================"

bash scripts/evaluation/qwen2-audio/eval_kdisentqa.sh && \
bash scripts/evaluation/qwen2-audio/eval_lsqa.sh && \
bash scripts/evaluation/qwen3-omni/eval_kdisentqa_vllm.sh && \
bash scripts/evaluation/qwen3-omni/eval_lsqa_vllm.sh && \
bash scripts/evaluation/gemma3/eval_kdisentqa.sh && \
bash scripts/evaluation/gemma3/eval_lsqa.sh && \
bash scripts/evaluation/gpt-audio-mini/eval_kdisentqa.sh && \
bash scripts/evaluation/gpt-audio-mini/eval_lsqa.sh && \
bash scripts/evaluation/voxtral-3B/eval_kdisentqa.sh && \
bash scripts/evaluation/voxtral-3B/eval_lsqa.sh

echo ""
echo "============================================"
echo "run_kdisentqa_lsqa_all 완료"
echo "============================================"
