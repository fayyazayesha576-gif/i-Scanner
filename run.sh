#!/usr/bin/env bash
# ─────────────────────────────────────────────
#  iScanner — Local Run Script
# ─────────────────────────────────────────────
set -e

echo ""
echo "  ╔══════════════════════════════════╗"
echo "  ║       iScanner Backend           ║"
echo "  ╚══════════════════════════════════╝"
echo ""

# Optional: override model file paths via env vars
export MODEL_PATH="${MODEL_PATH:-attention_fusion_best.pth}"
export SCALER_PATH="${SCALER_PATH:-meta_scaler.pkl}"
export THRESHOLD_PATH="${THRESHOLD_PATH:-threshold.npy}"

echo "  Model     : $MODEL_PATH"
echo "  Scaler    : $SCALER_PATH"
echo "  Threshold : $THRESHOLD_PATH"
echo ""

cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
