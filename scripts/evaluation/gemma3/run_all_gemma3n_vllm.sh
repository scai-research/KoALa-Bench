#!/bin/bash

# ============================================================
# Run All Evaluation Tasks — Gemma-3n (vLLM)
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "Starting All Evaluations (Gemma-3n vLLM)"
echo "=========================================="

# ASR
echo ""
echo "[1/4] Running ASR Evaluation..."
bash ${SCRIPT_DIR}/eval_asr.sh

# SQA
echo ""
echo "[2/4] Running SQA Evaluation..."
bash ${SCRIPT_DIR}/eval_sqa.sh

# Instruct
echo ""
echo "[3/4] Running Instruct Evaluation..."
bash ${SCRIPT_DIR}/eval_instruct.sh

# Translation
echo ""
echo "[4/4] Running Translation Evaluation..."
bash ${SCRIPT_DIR}/eval_translation.sh

echo ""
echo "=========================================="
echo "All Evaluations (Gemma-3n vLLM) Complete!"
echo "=========================================="
