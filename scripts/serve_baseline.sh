#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-google/gemma-4-31B-it}"

vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 32768 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.95 \
  --generation-config vllm \
  --language-model-only \
  --no-enable-prefix-caching
