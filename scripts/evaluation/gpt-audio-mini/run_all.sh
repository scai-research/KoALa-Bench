#!/bin/bash
# ============================================================
# GPT Realtime Mini — 전체 태스크 순차 평가
#   ASR → SQA → LSQA → Translation → Instruct → K-disentQA
# OpenAI API 기반이라 GPU 불필요 (필요 시 CUDA_VISIBLE_DEVICES 주석 해제)
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-${SCRIPT_DIR}}"
while [ ! -f "${BASE_PATH}/prompts.yaml" ] && [ "${BASE_PATH}" != "/" ]; do
    BASE_PATH="$(dirname "${BASE_PATH}")"
done
export PYTHONPATH="${BASE_PATH}/src:${PYTHONPATH}"
# 하위 eval_*.sh 에서도 각자 설정하지만, run_all 단독 실행 시에도 동작하도록 export
OPENAI_API_KEY_FILE="${BASE_PATH}/openai_key.txt"
export OPENAI_API_KEY_FILE
[ -f "${OPENAI_API_KEY_FILE}" ] && export OPENAI_API_KEY="$(head -1 "${OPENAI_API_KEY_FILE}" | tr -d '\r\n')"
# export CUDA_VISIBLE_DEVICES=3,4

echo "============================================"
echo "gpt-realtime-mini run_all (BACKEND=gpt_realtime_mini)"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "============================================"

cd "${SCRIPT_DIR}"

run() {
    local name="$1"
    echo ""
    echo "----------------------------------------"
    echo ">>> ${name}"
    echo "----------------------------------------"
    bash "${name}"
}

#run eval_asr.sh
#run eval_sqa.sh
#run eval_lsqa.sh
#run eval_instruct.sh
run eval_translation.sh
#run eval_kdisentqa.sh

echo ""
echo "============================================"
echo "gpt-realtime-mini run_all 완료"
echo "결과: ${BASE_PATH}/results_real/{ASR,SQA,LSQA,Translation,Instruct,K-disentQA}/gpt_realtime_mini/"
echo "============================================"
