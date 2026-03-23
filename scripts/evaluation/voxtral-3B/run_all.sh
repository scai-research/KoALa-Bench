#!/bin/bash

# ============================================================
# Run All Evaluation Tasks — Voxtral Mini 3B
#   ASR → SQA → LSQA → Translation → Instruct → K-disentQA
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_PATH="${BASE_PATH:-${SCRIPT_DIR}}"
while [ ! -f "${BASE_PATH}/prompts.yaml" ] && [ "${BASE_PATH}" != "/" ]; do
    BASE_PATH="$(dirname "${BASE_PATH}")"
done
export PYTHONPATH="${BASE_PATH}/src:${PYTHONPATH}"

echo "=========================================="
echo "Starting All Evaluations (Voxtral Mini 3B)"
echo "SCRIPT_DIR=${SCRIPT_DIR}"
echo "=========================================="

cd "${SCRIPT_DIR}"

run() {
    local name="$1"
    echo ""
    echo "----------------------------------------"
    echo ">>> ${name}"
    echo "----------------------------------------"
    bash "${name}"
}

run eval_asr.sh
run eval_sqa.sh
run eval_lsqa.sh
run eval_translation.sh
run eval_instruct.sh
#run eval_kdisentqa.sh

echo ""
echo "=========================================="
echo "All Evaluations (Voxtral Mini 3B) Complete!"
echo "Results: ${BASE_PATH}/results_real/{ASR,SQA,LSQA,Translation,Instruct,K-disentQA}/voxtral3b/"
echo "=========================================="

