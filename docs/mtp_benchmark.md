# Gemma 4 MTP Benchmark Handoff

This document is the human hand-off for running the Gemma 4 baseline vs MTP benchmark on one JarvisLabs H100. It explains what to run, in what order, where the results land, and what to check before trusting the numbers.

The short version: start one vLLM server, run `scripts/run_mtp_benchmarks.sh` against it, stop that server, start the next server, and run the same script again.

## What We Are Measuring

We are comparing two serving modes for `google/gemma-4-31B-it`:

- baseline decoding, with no draft model
- MTP speculative decoding, using `google/gemma-4-31B-it-assistant`

Both runs must use the same H100, same dataset, same server limits, same output cap, same sampling, and same benchmark client command. The only intended difference is whether the server was started with MTP enabled.

## Fixed Setup

Use these settings for both baseline and MTP.

- Hardware: one JarvisLabs H100 80GB
- Project root on the H100: `/home/gemma4_exps`
- Python environment: `/home/gemma4_exps/.venv`
- Hugging Face cache: `/home/gemma4_exps/hf_home`
- Target model: `google/gemma-4-31B-it`
- MTP draft model: `google/gemma-4-31B-it-assistant`
- Dataset: SPEED-Bench qualitative
- Context window: `--max-model-len 32768`
- Scheduler token cap: `--max-num-batched-tokens 4096`
- Max output cap: `4096`
- Sampling: `temperature=0`
- Prefix caching: disabled
- Text-only mode: `--language-model-only`
- Full sweep concurrency: `1, 2, 4, 8, 16`

`4096` appears in two places and they mean different things:

- `--max-num-batched-tokens 4096` is the vLLM scheduler token cap on the server.
- `--speed-bench-output-len 4096` is the benchmark max output cap.

The server is configured with a 32K context window. Each generation may produce up to 4096 output tokens.

## One-Time Machine Setup

Start from the H100 shell.

```bash
cd /home/gemma4_exps

uv venv --python python3 .venv
source .venv/bin/activate

uv pip install vllm --torch-backend=auto

uv pip install -U hf_transfer datasets pandas numpy tiktoken huggingface_hub
```

The official stable install is the first thing to try. If the MTP server fails while loading the assistant model with an error about `Gemma4AssistantConfig`, upgrade that same venv to a vLLM nightly build and retry the server:

```bash
uv pip install -U vllm --pre \
  --extra-index-url https://wheels.vllm.ai/nightly/cu130 \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  --index-strategy unsafe-best-match
```

Set the cache location before downloading models or running vLLM.

```bash
export HF_HOME=/home/gemma4_exps/hf_home
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_DEEP_GEMM_WARMUP=skip
mkdir -p "$HF_HOME"
```

Download the prepared SPEED-Bench qualitative dataset from Hugging Face.

```bash
mkdir -p data/speed_bench

hf download Gladiator/speed_bench_consolidated qualitative.jsonl \
  --repo-type dataset \
  --local-dir data/speed_bench
```

The qualitative dataset should contain these 11 categories:

```text
coding humanities math multilingual qa rag reasoning roleplay stem summarization writing
```

On the current prepared dataset, that is one `qualitative.jsonl` file with 880 rows: 80 rows per category.

## How To Run

Use one tmux session so the run is easy to watch. A simple layout is:

- `server`: start and stop the vLLM server here
- `bench`: run the benchmark commands here

Create that layout once:

```bash
tmux new -d -s gemma4_mtp -n server
tmux new-window -t gemma4_mtp -n bench
tmux attach -t gemma4_mtp
```

The benchmark script does not start or stop vLLM. It only sends requests to the server already listening on `http://127.0.0.1:8000`.

The script relies on vLLM bench defaults for the client protocol:

- backend defaults to `openai`
- endpoint defaults to `/v1/completions`

That is intentional. vLLM's SPEED-Bench loader already applies Gemma's chat template before sending the prompt. Using chat-completions here would wrap the already-rendered prompt again and make the token counts unfair.

## Step 1: Start The Baseline Server

Run this in the `server` window.

```bash
cd /home/gemma4_exps
source .venv/bin/activate

export HF_HOME=/home/gemma4_exps/hf_home
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_DEEP_GEMM_WARMUP=skip
mkdir -p "$HF_HOME"

vllm serve google/gemma-4-31B-it \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 32768 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.95 \
  --generation-config vllm \
  --language-model-only \
  --no-enable-prefix-caching
```

Wait until the server is fully ready before running the benchmark.

## Step 2: Run Baseline Full Sweep

Run this in the `bench` window.

```bash
cd /home/gemma4_exps
source .venv/bin/activate

MODEL=google/gemma-4-31B-it \
VARIANT=gemma4_31b_baseline \
./scripts/run_mtp_benchmarks.sh
```

This runs all 11 categories at concurrency `1, 2, 4, 8, 16`.

It writes 55 JSON files under:

```text
artifacts/speed_bench/
```

Example filenames:

```text
artifacts/speed_bench/gemma4_31b_baseline_coding_c1.json
artifacts/speed_bench/gemma4_31b_baseline_rag_c8.json
artifacts/speed_bench/gemma4_31b_baseline_writing_c16.json
```

When the baseline sweep finishes, stop the baseline server before starting MTP.

## Step 3: Start The MTP Server

Run this in the `server` window.

```bash
cd /home/gemma4_exps
source .venv/bin/activate

export HF_HOME=/home/gemma4_exps/hf_home
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_DEEP_GEMM_WARMUP=skip
export NUM_SPECULATIVE_TOKENS=8
mkdir -p "$HF_HOME"

vllm serve google/gemma-4-31B-it \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 32768 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.95 \
  --generation-config vllm \
  --language-model-only \
  --no-enable-prefix-caching \
  --speculative-config '{"model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":'"$NUM_SPECULATIVE_TOKENS"'}'
```

Use `NUM_SPECULATIVE_TOKENS=8` for the main MTP comparison.

## Step 4: Run MTP Full Sweep

Run this in the `bench` window.

```bash
cd /home/gemma4_exps
source .venv/bin/activate

MODEL=google/gemma-4-31B-it \
VARIANT=gemma4_31b_mtp \
DRAFT_MODEL=google/gemma-4-31B-it-assistant \
NUM_SPECULATIVE_TOKENS=8 \
./scripts/run_mtp_benchmarks.sh
```

This also writes 55 JSON files under:

```text
artifacts/speed_bench/
```

Example filenames:

```text
artifacts/speed_bench/gemma4_31b_mtp_coding_c1.json
artifacts/speed_bench/gemma4_31b_mtp_rag_c8.json
artifacts/speed_bench/gemma4_31b_mtp_writing_c16.json
```

## What The Script Runs

The script loops over categories and concurrency values, then runs one `vllm bench serve` command per cell.

```text
categories: all 11 qualitative categories
concurrency: 1 2 4 8 16
prompts per category: all prompts in that category
result root: artifacts/speed_bench
```

Each saved JSON contains both summary and detailed data because the script uses `--save-result` and `--save-detailed`. vLLM stores those detailed fields in the same JSON file, not in a separate sidecar file.

The script also adds simple top-level labels through `--metadata`, such as:

```text
variant
mode
subset
category
concurrency
num_prompts
output_cap
temperature
draft_model
num_speculative_tokens
```

vLLM saves these labels as top-level JSON keys.

## Final Checks Before Trusting The Run

Do not stop at “the command finished.” Check the artifacts.

For every JSON:

- `failed` should be 0
- `completed` should match the planned number of prompts
- token counts should be present
- latency arrays should be present
- `generated_texts` should not be empty

For a few representative files:

- read a couple of generated outputs manually
- reject the run if outputs are empty, repetitive, corrupted, or full of chat-template markers
- compare baseline and MTP input token counts for the same category/concurrency

For MTP files:

- confirm acceptance fields are present
- recompute acceptance rate and acceptance length for a few files
- make sure the per-position acceptance array has 8 values for the main run

The basic MTP sanity formulas are:

```text
acceptance_rate = accepted_tokens / draft_tokens
acceptance_length = 1 + accepted_tokens / num_drafts
```

After the run, keep the H100 paused rather than destroyed if follow-up analysis will need the model cache or result artifacts.

## Useful Links

- vLLM Gemma 4 recipe: https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html
- Gemma 4 MTP docs: https://ai.google.dev/gemma/docs/mtp/overview
- Gemma 4 31B instruct model: https://huggingface.co/google/gemma-4-31B-it
- Gemma 4 31B assistant model: https://huggingface.co/google/gemma-4-31B-it-assistant
- SPEED-Bench dataset: https://huggingface.co/datasets/nvidia/SPEED-Bench
- vLLM PR #41745: https://github.com/vllm-project/vllm/pull/41745
