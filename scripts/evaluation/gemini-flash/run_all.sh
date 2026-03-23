#!/bin/bash
# ============================================================
# Run All Evaluation Tasks — Gemini Flash
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-${SCRIPT_DIR}}"
while [ ! -f "${BASE_PATH}/prompts.yaml" ] && [ "${BASE_PATH}" != "/" ]; do
    BASE_PATH="$(dirname "${BASE_PATH}")"
done
GEMINI_API_KEY_FILE="${BASE_PATH}/gemini_key.txt"
[ -f "${GEMINI_API_KEY_FILE}" ] && export GEMINI_API_KEY="$(head -1 "${GEMINI_API_KEY_FILE}" | tr -d '\r\n')"
export PYTHONPATH="${BASE_PATH}/src:${PYTHONPATH}"

echo "=========================================="
echo "Starting All Evaluations (gemini-flash)"
echo "=========================================="

run() {
    local name="$1"
    echo ""
    echo "------------------------------------------"
    echo ">>> ${name}"
    echo "------------------------------------------"
    bash "${SCRIPT_DIR}/${name}"
}

run eval_asr.sh
run eval_sqa.sh
run eval_lsqa.sh
run eval_kdisentqa.sh
run eval_instruct.sh
run eval_translation.sh

echo ""
echo "=========================================="
echo "All Evaluations (gemini-flash) Complete!"
echo "결과: ${BASE_PATH}/results_real/{ASR,SQA,LSQA,K-disentQA,Instruct,Translation}/gemini_flash/"
echo "=========================================="
