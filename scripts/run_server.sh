#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Example: GPU_ID=2 bash scripts/run_server.sh smoke
# Run setup_gpu.sh first. Uses only the selected GPU; never stops other processes.
export CUDA_VISIBLE_DEVICES="${GPU_ID:-2}"
export TOKENIZERS_PARALLELISM=false
PYTHON="${PYTHON:-.venv/bin/python}"
MODE="${1:-smoke}"
MODEL="${MODEL:-Qwen/Qwen2.5-3B-Instruct}"
REVISION_ARGS=()
if [[ -n "${MODEL_REVISION:-}" ]]; then
  REVISION_ARGS=(--revision "$MODEL_REVISION")
fi

case "$MODE" in
  prepare)
    if [[ ! -f data/raw/metadata.csv ]]; then
      "$PYTHON" -m wattbot unpack --archive "${DATASET_ARCHIVE:-WattBot2026.zip}"
    fi
    "$PYTHON" -m wattbot profile
    "$PYTHON" -m wattbot split
    if [[ ! -f data/processed/chunks.jsonl ]]; then
      if "$PYTHON" -m wattbot download; then
        :
      elif [[ "${ALLOW_PARTIAL_CORPUS:-0}" != 1 ]]; then
        echo 'Some PDFs failed. Inspect artifacts/download_manifest.json, or explicitly set ALLOW_PARTIAL_CORPUS=1.' >&2
        exit 2
      fi
      "$PYTHON" -m wattbot ingest
    fi
    "$PYTHON" -m wattbot retrieve --context-k 5
    "$PYTHON" -m wattbot retrieval-eval
    ;;
  smoke)
    "$PYTHON" -m wattbot generate --limit 5 --model "$MODEL" "${REVISION_ARGS[@]}" --output-dir runs/gpu_smoke
    "$PYTHON" -m wattbot validate --questions runs/gpu_smoke/questions.csv --submission runs/gpu_smoke/predictions.csv
    ;;
  train)
    "$PYTHON" -m wattbot generate --model "$MODEL" "${REVISION_ARGS[@]}" --output-dir runs/gpu_train
    "$PYTHON" -m wattbot evaluate --submission runs/gpu_train/predictions.csv --output-dir runs/gpu_train/evaluation
    ;;
  hybrid)
    # Embedding and reranking on CPU leave the selected GPU available for generation.
    "$PYTHON" -m wattbot embed --model BAAI/bge-small-en-v1.5 --device cpu
    "$PYTHON" -m wattbot retrieve --dense-dir artifacts/dense --context-k 5 --output runs/hybrid/train_retrieval.jsonl
    "$PYTHON" -m wattbot retrieval-eval --retrieval runs/hybrid/train_retrieval.jsonl --output runs/hybrid/retrieval_metrics.json
    ;;
  hybrid-train)
    "$PYTHON" -m wattbot generate --model "$MODEL" "${REVISION_ARGS[@]}" --retrieval runs/hybrid/train_retrieval.jsonl --output-dir runs/gpu_hybrid_train
    "$PYTHON" -m wattbot evaluate --submission runs/gpu_hybrid_train/predictions.csv --output-dir runs/gpu_hybrid_train/evaluation
    ;;
  test)
    "$PYTHON" -m wattbot retrieve --questions data/raw/test_Q.csv --context-k 5 --output runs/bm25/test_retrieval.jsonl
    "$PYTHON" -m wattbot generate --questions data/raw/test_Q.csv --retrieval runs/bm25/test_retrieval.jsonl --model "$MODEL" "${REVISION_ARGS[@]}" --output-dir runs/gpu_test
    "$PYTHON" -m wattbot validate --submission runs/gpu_test/predictions.csv
    ;;
  *)
    echo "Usage: GPU_ID=2 bash scripts/run_server.sh {prepare|smoke|train|hybrid|hybrid-train|test}"
    exit 2
    ;;
esac
