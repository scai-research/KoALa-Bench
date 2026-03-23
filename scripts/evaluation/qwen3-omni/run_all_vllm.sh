#!/bin/bash

# ============================================================
# Run All Evaluation Tasks — Qwen3-Omni (vLLM)
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=========================================="
echo "Starting All Evaluations (Qwen3-Omni vLLM)"
echo "=========================================="

# # ASR
# echo ""
# echo "[1/4] Running ASR Evaluation..."
# bash ${SCRIPT_DIR}/eval_asr_vllm.sh

# SQA
echo ""
echo "[2/4] Running SQA Evaluation..."
bash ${SCRIPT_DIR}/eval_sqa_vllm.sh

# Instruct
echo ""
echo "[3/4] Running Instruct Evaluation..."
bash ${SCRIPT_DIR}/eval_instruct_vllm.sh

# Translation
echo ""
echo "[4/4] Running Translation Evaluation..."
bash ${SCRIPT_DIR}/eval_translation_vllm.sh

echo ""
echo "=========================================="
echo "All Evaluations (Qwen3-Omni vLLM) Complete!"
echo "=========================================="
