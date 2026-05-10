# Gemma 4 DFlash Benchmark Handoff

This document is the hand-off for running Gemma 4 DFlash on the same benchmark rig used for the MTP runs. Keep the benchmark-critical settings matched to MTP: same H100, same dataset, same client command, same scheduler cap, same output cap, same sampling, same concurrency sweep, and same artifact layout.

The DFlash-specific differences are the draft model, the DFlash speculative method, and the mixed attention setup: target `triton_attn`, draft `flash_attn`.

## Latest Validated State

Validated on May 10, 2026 on the renamed JarvisLabs H100 machine:

- Machine name: `gemma4_mtp_dflash`
- Current running machine id: `407972`
- Project root on the H100: `/home/gemma4_exps`
- Existing MTP environment: `/home/gemma4_exps/.venv`
- New DFlash environment: `/home/gemma4_exps/.venv-dflash-pr41703-a20847b`
- vLLM install: `0.20.2rc1.dev191+ga20847b5c.precompiled`
- vLLM PR head: `a20847b5cc74dc0b471d715dde3ee7fa17c99eb8`
- Hugging Face cache: `/home/gemma4_exps/hf_home`

The latest PR head no longer needs the old `--hf-overrides '{"text_config":{"use_bidirectional_attention":null}}'` workaround. Both 31B and 26B-A4B DFlash servers started successfully without that flag.

## What We Are Measuring

We are measuring DFlash speculative decoding for the two Gemma 4 targets used in this project:

| Target model | DFlash draft model | Result variant |
|---|---|---|
| `google/gemma-4-31B-it` | `z-lab/gemma-4-31B-it-DFlash` | `gemma4_31b_dflash_nt15` |
| `google/gemma-4-26B-A4B-it` | `z-lab/gemma-4-26B-A4B-it-DFlash` | `gemma4_26b_a4b_dflash_nt15` |

DFlash should be compared against the matching baseline and MTP results for the same target model.

The intended serving difference from MTP is only the speculative path:

- MTP uses the Gemma assistant draft model.
- DFlash uses the z-lab DFlash draft model.
- DFlash uses `num_speculative_tokens=15`, matching the z-lab and vLLM PR command.
- DFlash uses target `triton_attn` and draft `flash_attn`.

Everything else should stay matched to the MTP benchmark.

## Fixed Setup

Use these settings for both DFlash target models.

- Hardware: one JarvisLabs H100 80GB
- Project root on the H100: `/home/gemma4_exps`
- Python environment: `/home/gemma4_exps/.venv-dflash-pr41703-a20847b`
- Hugging Face cache: `/home/gemma4_exps/hf_home`
- Dataset: SPEED-Bench qualitative
- Context window: `--max-model-len 32768`
- Scheduler token cap: `--max-num-batched-tokens 4096`
- Max concurrent sequences: `--max-num-seqs 16`
- Max output cap: `4096`
- Sampling: `temperature=0`
- Prefix caching: disabled
- Text-only mode: `--language-model-only`
- DFlash target attention backend: `--attention-backend triton_attn`
- DFlash draft attention backend: `"attention_backend": "flash_attn"`
- DFlash speculative tokens: `num_speculative_tokens=15`
- Full sweep concurrency: `1, 2, 4, 8, 16`

`4096` appears in two places and they mean different things:

- `--max-num-batched-tokens 4096` is the vLLM scheduler token cap on the server.
- `--speed-bench-output-len 4096` is the benchmark max output cap.

The z-lab example command uses `--max-num-batched-tokens 32768`. Do not use that value for the comparison run. The MTP benchmark used `4096`, so DFlash also uses `4096`.

`--max-num-seqs 16` matches the largest benchmark client concurrency. Keep it with `--max-num-batched-tokens 4096` and `num_speculative_tokens=15`; both DFlash target models failed startup without this cap in the validated setup.

## PR Fix That Removed The Override

Older DFlash runs needed this workaround:

```bash
--hf-overrides '{"text_config":{"use_bidirectional_attention":null}}'
```

That was needed because the DFlash draft attention inherited Gemma 4's multimodal-prefix attention setting from the target model, and the draft `flash_attn` backend rejected it.

The latest PR head fixes this inside vLLM. DFlash draft attention now opts out of the target model's multimodal-prefix mask requirement. For this benchmark, run without `hf-overrides`.

Keep `--language-model-only`. It matches the MTP benchmark and keeps the serving path text-only.

## One-Time DFlash Environment Setup

Start from the H100 shell.

First check whether the DFlash environment already exists and is already pinned to the expected vLLM PR head. If this check passes, reuse the environment and skip the create/install commands below.

```bash
cd /home/gemma4_exps

if [[ -d .venv-dflash-pr41703-a20847b ]]; then
  source .venv-dflash-pr41703-a20847b/bin/activate
  python - <<'PY'
import importlib.metadata as md

print("vllm", md.version("vllm"))
print(md.distribution("vllm").read_text("direct_url.json"))
PY
else
  echo "DFlash env not found; create it with the setup commands below."
fi
```

```bash
cd /home/gemma4_exps

uv venv --python 3.11 --prompt dflash-pr41703-a20847b .venv-dflash-pr41703-a20847b
source .venv-dflash-pr41703-a20847b/bin/activate

VLLM_USE_PRECOMPILED=1 uv pip install --torch-backend=auto \
  "vllm @ git+https://github.com/vllm-project/vllm.git@a20847b5cc74dc0b471d715dde3ee7fa17c99eb8"

uv pip install -U hf_transfer datasets pandas numpy tiktoken huggingface_hub
```

Set the cache and CUDA environment before downloading models or running vLLM.

```bash
export HF_HOME=/home/gemma4_exps/hf_home
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_DEEP_GEMM_WARMUP=skip

export CUDA_HOME=/usr/local/cuda
export CUDAToolkit_ROOT=/usr/local/cuda
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

mkdir -p "$HF_HOME"
```

On the validated H100 machine, the host CUDA toolkit is available at `/usr/local/cuda`, and `nvcc` is at `/usr/local/cuda/bin/nvcc`. If `CUDA_HOME` is set, point it to this host toolkit path. Do not point `CUDA_HOME` at the Python package directory under `.venv-dflash-pr41703-a20847b/lib/python3.11/site-packages/nvidia/cu13`; that directory contains Python-packaged CUDA runtime libraries, not the full CUDA toolkit.

Download the prepared SPEED-Bench qualitative dataset if it is not already present.

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
tmux new -d -s gemma4_dflash -n server
tmux new-window -t gemma4_dflash -n bench
tmux attach -t gemma4_dflash
```

The benchmark script does not start or stop vLLM. It only sends requests to the server already listening on `http://127.0.0.1:8000`.

The script relies on vLLM bench defaults for the client protocol:

- backend defaults to `openai`
- endpoint defaults to `/v1/completions`

That is intentional. vLLM's SPEED-Bench loader already applies Gemma's chat template before sending the prompt. Using chat-completions here would wrap the already-rendered prompt again and make the token counts unfair.

## Start The 31B DFlash Server

Run this in the server shell.

```bash
cd /home/gemma4_exps
source .venv-dflash-pr41703-a20847b/bin/activate

export HF_HOME=/home/gemma4_exps/hf_home
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_DEEP_GEMM_WARMUP=skip

export CUDA_HOME=/usr/local/cuda
export CUDAToolkit_ROOT=/usr/local/cuda
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

vllm serve google/gemma-4-31B-it \
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
  --speculative-config '{"method": "dflash", "model": "z-lab/gemma-4-31B-it-DFlash", "num_speculative_tokens": 15, "attention_backend": "flash_attn"}' \
  --attention-backend triton_attn
```

Wait until `/v1/models` responds before running the benchmark.

## Start The 26B-A4B DFlash Server

Stop the 31B server first, then run this in the server shell.

```bash
cd /home/gemma4_exps
source .venv-dflash-pr41703-a20847b/bin/activate

export HF_HOME=/home/gemma4_exps/hf_home
export HF_HUB_ENABLE_HF_TRANSFER=1
export VLLM_DEEP_GEMM_WARMUP=skip

export CUDA_HOME=/usr/local/cuda
export CUDAToolkit_ROOT=/usr/local/cuda
export PATH="/usr/local/cuda/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

vllm serve google/gemma-4-26B-A4B-it \
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
  --speculative-config '{"method": "dflash", "model": "z-lab/gemma-4-26B-A4B-it-DFlash", "num_speculative_tokens": 15, "attention_backend": "flash_attn"}' \
  --attention-backend triton_attn
```

Wait until `/v1/models` responds before running the benchmark.

## Smoke Test Command

After the server is ready, run smoke mode once for the model that is currently served. Smoke mode runs `rag` and `writing` at concurrency `16` with `16` prompts each.

```bash
cd /home/gemma4_exps
source .venv-dflash-pr41703-a20847b/bin/activate

SMOKE=1 \
MODEL=google/gemma-4-31B-it \
VARIANT=gemma4_31b_dflash_nt15 \
DRAFT_MODEL=z-lab/gemma-4-31B-it-DFlash \
NUM_SPECULATIVE_TOKENS=15 \
ATTENTION_BACKEND=triton_attn \
SPEC_ATTENTION_BACKEND=flash_attn \
HF_OVERRIDES=none \
PR_HEAD=a20847b5cc74dc0b471d715dde3ee7fa17c99eb8 \
./scripts/run_dflash_benchmarks.sh
```

For 26B-A4B, use the same smoke command shape with these model values:

```bash
MODEL=google/gemma-4-26B-A4B-it
VARIANT=gemma4_26b_a4b_dflash_nt15
DRAFT_MODEL=z-lab/gemma-4-26B-A4B-it-DFlash
```

## Full Sweep Command

After smoke passes, use `scripts/run_dflash_benchmarks.sh` for the full run. It mirrors the MTP benchmark client: same SPEED-Bench qualitative categories, same concurrency sweep, same output cap, same request protocol, and same artifact shape.

31B:

```bash
cd /home/gemma4_exps
source .venv-dflash-pr41703-a20847b/bin/activate

MODEL=google/gemma-4-31B-it \
VARIANT=gemma4_31b_dflash_nt15 \
DRAFT_MODEL=z-lab/gemma-4-31B-it-DFlash \
NUM_SPECULATIVE_TOKENS=15 \
ATTENTION_BACKEND=triton_attn \
SPEC_ATTENTION_BACKEND=flash_attn \
HF_OVERRIDES=none \
PR_HEAD=a20847b5cc74dc0b471d715dde3ee7fa17c99eb8 \
./scripts/run_dflash_benchmarks.sh
```

26B-A4B:

```bash
cd /home/gemma4_exps
source .venv-dflash-pr41703-a20847b/bin/activate

MODEL=google/gemma-4-26B-A4B-it \
VARIANT=gemma4_26b_a4b_dflash_nt15 \
DRAFT_MODEL=z-lab/gemma-4-26B-A4B-it-DFlash \
NUM_SPECULATIVE_TOKENS=15 \
ATTENTION_BACKEND=triton_attn \
SPEC_ATTENTION_BACKEND=flash_attn \
HF_OVERRIDES=none \
PR_HEAD=a20847b5cc74dc0b471d715dde3ee7fa17c99eb8 \
./scripts/run_dflash_benchmarks.sh
```

Full results go flat into:

```text
artifacts/speed_bench/
```

The variant names include model, method, and speculative depth. PR head and `hf_overrides=none` are recorded as JSON metadata, not encoded into the filename.

## What The Script Runs

The script loops over categories and concurrency values, then runs one `vllm bench serve` command per cell.

```text
categories: all 11 qualitative categories
concurrency: 1 2 4 8 16
prompts per category: all prompts in that category
result root: artifacts/speed_bench
```

Each full DFlash variant writes 55 JSON files. Both DFlash variants together write 110 JSON files.

Example filenames:

```text
artifacts/speed_bench/gemma4_31b_dflash_nt15_coding_c1.json
artifacts/speed_bench/gemma4_31b_dflash_nt15_rag_c8.json
artifacts/speed_bench/gemma4_26b_a4b_dflash_nt15_writing_c16.json
```

Each saved JSON contains both summary and detailed data because the script uses `--save-result` and `--save-detailed`. vLLM stores those detailed fields in the same JSON file, not in a separate sidecar file.

The script also adds top-level labels through `--metadata`, such as:

```text
variant
mode
subset
category
concurrency
num_prompts
output_cap
temperature
speculative_method
attention_backend
spec_attention_backend
draft_model
num_speculative_tokens
hf_overrides
pr_head
```

vLLM saves these labels as top-level JSON keys.

## Smoke Results From Latest PR Head

These smoke results used `num_prompts=16`, category `rag` and `writing`, concurrency `16`, `--speed-bench-output-len 4096`, `--max-num-batched-tokens 4096`, and no `hf-overrides`.

| Model | Category | Completed | Failed | Output tok/s | Mean TPOT ms | Mean TTFT ms | Acceptance rate | Acceptance length |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 31B DFlash | rag | 16 | 0 | 200.50 | 11.86 | 3401.94 | 27.99% | 5.20 |
| 31B DFlash | writing | 16 | 0 | 246.34 | 33.47 | 8422.47 | 11.97% | 2.80 |
| 26B-A4B DFlash | rag | 16 | 0 | 281.68 | 16.66 | 2622.03 | 23.96% | 4.59 |
| 26B-A4B DFlash | writing | 16 | 0 | 989.67 | 8.66 | 756.62 | 10.89% | 2.63 |

Validated smoke files from the setup run:

```text
artifacts/smoke/gemma4_31b_dflash_pr41703_a20847b_no_hfoverride_rag_c16.json
artifacts/smoke/gemma4_31b_dflash_pr41703_a20847b_no_hfoverride_writing_c16.json
artifacts/smoke/gemma4_26b_a4b_dflash_pr41703_a20847b_no_hfoverride_rag_c16.json
artifacts/smoke/gemma4_26b_a4b_dflash_pr41703_a20847b_no_hfoverride_writing_c16.json
```

Manual output spot-checks looked normal for all four smoke files. The outputs were non-empty and did not show obvious looping, corruption, or chat-template leakage.

## Final Checks Before Trusting A Full Run

Do not stop at "the command finished." Check the artifacts.

For every JSON:

- `failed` should be 0
- `completed` should match the planned number of prompts
- token counts should be present
- latency arrays should be present
- `generated_texts` should not be empty

For a few representative files:

- read a couple of generated outputs manually
- reject the run if outputs are empty, repetitive, corrupted, or full of chat-template markers
- compare DFlash and MTP input token counts for the same target model, category, and concurrency
- confirm the DFlash files use `num_speculative_tokens=15`
- confirm the DFlash files use `attention_backend=triton_attn`
- confirm the DFlash files use `spec_attention_backend=flash_attn`
- confirm the DFlash files use `hf_overrides=none`

For DFlash files:

- confirm acceptance fields are present
- recompute acceptance rate and acceptance length for a few files
- the per-position acceptance array should correspond to 15 draft positions for the `nt15` run

The basic speculative-decoding sanity formulas are:

```text
acceptance_rate = accepted_tokens / draft_tokens
acceptance_length = 1 + accepted_tokens / num_drafts
```

After both full sweeps finish, compare DFlash against the fastest matching MTP variant for each target model.

## Useful Links

- z-lab DFlash repo: https://github.com/z-lab/dflash
- vLLM Gemma 4 recipe: https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html
- vLLM Gemma 4 DFlash PR: https://github.com/vllm-project/vllm/pull/41703
- vLLM PR head validated here: https://github.com/vllm-project/vllm/commit/a20847b5cc74dc0b471d715dde3ee7fa17c99eb8
- Gemma 4 31B DFlash draft: https://huggingface.co/z-lab/gemma-4-31B-it-DFlash
- Gemma 4 26B-A4B DFlash draft: https://huggingface.co/z-lab/gemma-4-26B-A4B-it-DFlash
- Gemma 4 31B instruct model: https://huggingface.co/google/gemma-4-31B-it
- Gemma 4 26B-A4B instruct model: https://huggingface.co/google/gemma-4-26B-A4B-it
- SPEED-Bench dataset: https://huggingface.co/datasets/nvidia/SPEED-Bench
