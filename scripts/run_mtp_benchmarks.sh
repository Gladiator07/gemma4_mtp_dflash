#!/usr/bin/env bash
set -euo pipefail

# Run SPEED-Bench qualitative benchmarks against an already-running vLLM server.
# Start the baseline or MTP server separately; this script only sends benchmark
# requests and saves the results.
#
# Required env:
#   MODEL    served model id, must match `vllm serve` (e.g. google/gemma-4-31B-it)
#   VARIANT  result label, distinguishes runs in the same dir
#            (e.g. gemma4_31b_baseline, gemma4_31b_mtp)
#
# Optional env (recorded into result metadata only):
#   DRAFT_MODEL              draft model id passed to the server's --speculative-config
#   NUM_SPECULATIVE_TOKENS   gamma value passed to the server's --speculative-config
#
# Optional mode:
#   SMOKE=1  run only rag/writing at c16 with 16 prompts, saved under artifacts/smoke
#
# Full artifacts land under:
#   artifacts/speed_bench/<VARIANT>_<category>_c<concurrency>.json
# Smoke artifacts land under:
#   artifacts/smoke/<VARIANT>_<category>_c16.json

: "${MODEL:?Set MODEL, for example: MODEL=google/gemma-4-31B-it}"
: "${VARIANT:?Set VARIANT, for example: VARIANT=gemma4_31b_baseline}"

BASE_URL="http://127.0.0.1:8000"
DATASET_PATH="data/speed_bench"
OUTPUT_LEN=4096

if [[ "${SMOKE:-0}" == "1" ]]; then
  MODE="smoke"
  RESULT_ROOT="artifacts/smoke"
  CATEGORIES=(rag writing)
  CONCURRENCIES=(16)
  NUM_PROMPTS=16
else
  MODE="full"
  RESULT_ROOT="artifacts/speed_bench"
  # All 11 SPEED-Bench qualitative categories.
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
  # -1 = use every prompt in the category. Combined with
  # --no-oversample/--disable-shuffle below, this gives the same prompt
  # sequences across variants for a fair comparison.
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
    )
    if [[ -n "${DRAFT_MODEL:-}" ]]; then
      metadata+=("draft_model=$DRAFT_MODEL")
    fi
    if [[ -n "${NUM_SPECULATIVE_TOKENS:-}" ]]; then
      metadata+=("num_speculative_tokens=$NUM_SPECULATIVE_TOKENS")
    fi

    # vLLM bench serve defaults to backend=openai and endpoint=/v1/completions.
    # Keep that default: the SPEED-Bench loader already applies Gemma's chat
    # template before sending, so chat-completions would double-template.
    # --temperature 0: greedy decoding keeps the run deterministic.
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
