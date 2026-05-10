#!/usr/bin/env bash
set -euo pipefail

# Run SPEED-Bench qualitative benchmarks against an already-running DFlash vLLM
# server. This intentionally mirrors run_mtp_benchmarks.sh so DFlash results are
# comparable row-for-row with the MTP arrangement.
#
# Required env:
#   MODEL    served model id, must match `vllm serve`
#   VARIANT  result label, distinguishes runs in the same dir
#
# Optional env (recorded into result metadata only):
#   DRAFT_MODEL              model passed to the server's --speculative-config
#   NUM_SPECULATIVE_TOKENS   gamma value passed to the server's --speculative-config
#   ATTENTION_BACKEND        server --attention-backend value
#   SPEC_ATTENTION_BACKEND   speculative-config attention_backend value
#
# Optional mode:
#   SMOKE=1  run only rag/writing at c16 with 16 prompts, saved under artifacts/smoke

: "${MODEL:?Set MODEL, for example: MODEL=google/gemma-4-31B-it}"
: "${VARIANT:?Set VARIANT, for example: VARIANT=gemma4_31b_dflash}"

BASE_URL="http://127.0.0.1:8000"
DATASET_PATH="data/speed_bench"
OUTPUT_LEN=4096
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton_attn}"
SPEC_ATTENTION_BACKEND="${SPEC_ATTENTION_BACKEND:-flash_attn}"

if [[ "${SMOKE:-0}" == "1" ]]; then
  MODE="smoke"
  RESULT_ROOT="artifacts/smoke"
  CATEGORIES=(rag writing)
  CONCURRENCIES=(16)
  NUM_PROMPTS=16
else
  MODE="full"
  RESULT_ROOT="artifacts/speed_bench"
  CATEGORIES=(
    coding
    humanities
    math
    multilingual
    qa
    rag
    reasoning
    roleplay
    stem
    summarization
    writing
  )
  CONCURRENCIES=(1 2 4 8 16)
  NUM_PROMPTS=-1
fi

if [[ ! -d "$DATASET_PATH" ]]; then
  echo "Dataset path not found: $DATASET_PATH" >&2
  echo "Prepare SPEED-Bench first." >&2
  exit 1
fi

echo "model: $MODEL"
echo "variant: $VARIANT"
echo "mode: $MODE"
echo "base_url: $BASE_URL"
echo "dataset_path: $DATASET_PATH"
echo "categories: ${CATEGORIES[*]}"
echo "concurrencies: ${CONCURRENCIES[*]}"
echo "num_prompts: $NUM_PROMPTS"
echo "output_len: $OUTPUT_LEN"
echo "result_root: $RESULT_ROOT"
echo "attention_backend: $ATTENTION_BACKEND"
echo "spec_attention_backend: $SPEC_ATTENTION_BACKEND"
if [[ -n "${DRAFT_MODEL:-}" ]]; then
  echo "draft_model: $DRAFT_MODEL"
fi
if [[ -n "${NUM_SPECULATIVE_TOKENS:-}" ]]; then
  echo "num_speculative_tokens: $NUM_SPECULATIVE_TOKENS"
fi

echo "running benchmarks..."
for category in "${CATEGORIES[@]}"; do
  for concurrency in "${CONCURRENCIES[@]}"; do
    result_dir="$RESULT_ROOT"
    result_file="${VARIANT}_${category}_c${concurrency}.json"
    mkdir -p "$result_dir"

    echo
    echo "=== ${VARIANT} / ${category} / c${concurrency} ==="

    metadata=(
      "variant=$VARIANT"
      "mode=$MODE"
      "subset=qualitative"
      "category=$category"
      "concurrency=$concurrency"
      "num_prompts=$NUM_PROMPTS"
      "output_cap=$OUTPUT_LEN"
      "temperature=0"
      "speculative_method=dflash"
      "attention_backend=$ATTENTION_BACKEND"
      "spec_attention_backend=$SPEC_ATTENTION_BACKEND"
    )
    if [[ -n "${DRAFT_MODEL:-}" ]]; then
      metadata+=("draft_model=$DRAFT_MODEL")
    fi
    if [[ -n "${NUM_SPECULATIVE_TOKENS:-}" ]]; then
      metadata+=("num_speculative_tokens=$NUM_SPECULATIVE_TOKENS")
    fi
    # Keep the same vLLM bench serve protocol as MTP: backend=openai and
    # endpoint=/v1/completions are the defaults. SPEED-Bench already applies
    # Gemma's chat template before sending prompts.
    vllm bench serve \
      --base-url "$BASE_URL" \
      --model "$MODEL" \
      --dataset-name speed_bench \
      --dataset-path "$DATASET_PATH" \
      --speed-bench-dataset-subset qualitative \
      --speed-bench-category "$category" \
      --speed-bench-output-len "$OUTPUT_LEN" \
      --num-prompts "$NUM_PROMPTS" \
      --no-oversample \
      --disable-shuffle \
      --temperature 0 \
      --request-rate inf \
      --max-concurrency "$concurrency" \
      --save-result \
      --save-detailed \
      --result-dir "$result_dir" \
      --result-filename "$result_file" \
      --metadata "${metadata[@]}"
  done
done
