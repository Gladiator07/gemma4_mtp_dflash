#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-google/gemma-4-31B-it}"
DRAFT_MODEL="${DRAFT_MODEL:-z-lab/gemma-4-31B-it-DFlash}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-15}"

vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 32768 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.95 \
  --generation-config vllm \
  --trust-remote-code \
  --language-model-only \
  --no-enable-prefix-caching \
  --speculative-config '{"method":"dflash","model":"'"$DRAFT_MODEL"'","num_speculative_tokens":'"$NUM_SPECULATIVE_TOKENS"',"attention_backend":"flash_attn"}' \
  --attention-backend triton_attn
