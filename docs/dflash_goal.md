# Codex Goal: Full Gemma 4 DFlash Benchmark

Run the full Gemma 4 DFlash benchmark suite end to end, matching the existing baseline and MTP benchmark arrangement.

## Important Constraint

Use only the existing DFlash benchmark script:

```text
/home/gemma4_exps/scripts/run_dflash_benchmarks.sh
```

Do not create a new benchmark script. Do not modify this script unless there is a major blocker that makes the benchmark impossible to run. If you hit such a blocker, document exactly what failed, why changing the script is necessary, make the smallest possible change, and continue. This is a constrained benchmark, so do not experiment with alternate benchmark commands or helper scripts.

The benchmark script only sends benchmark requests. It does not start vLLM. You must start the correct DFlash vLLM server before running the script.

`VARIANT` is required. It is not just a display label: the script uses it in the saved JSON filenames. Because results are saved flat in one directory, use the exact variant names below so files do not collide and are easy to compare.

Allowed `VARIANT` values for this suite:

```text
gemma4_31b_dflash_nt15
gemma4_26b_a4b_dflash_nt15
```

Use `HF_OVERRIDES=none` and `PR_HEAD=a20847b5cc74dc0b471d715dde3ee7fa17c99eb8` when running the benchmark script so the saved JSON records the validated DFlash setup.

Do not delete, move, overwrite, or modify any existing MTP benchmark JSON files on the benchmark machine. Do not delete or modify the existing MTP environment at `/home/gemma4_exps/.venv`. DFlash must use only `/home/gemma4_exps/.venv-dflash-pr41703-a20847b`.

## Workspace

- Start from the benchmark workspace.
- Use `docs/dflash_benchmark.md` as the detailed benchmark contract.
- H100 machine: JarvisLabs machine `407972`, named `gemma4_mtp_dflash`. If the id changed after a pause/resume, locate the existing machine by name instead of creating a new one.
- H100 project root: `/home/gemma4_exps`.
- Use the DFlash venv: `/home/gemma4_exps/.venv-dflash-pr41703-a20847b`.
- Do not touch the existing MTP venv: `/home/gemma4_exps/.venv`.
- Use the existing dataset: `/home/gemma4_exps/data/speed_bench`.
- Use the existing Hugging Face cache: `/home/gemma4_exps/hf_home`.

## Execution Rules

- Run variants sequentially on the same H100.
- For each variant, start the correct DFlash vLLM server first.
- Wait until `http://127.0.0.1:8000/v1/models` responds.
- Run smoke mode first with `SMOKE=1`.
- Inspect the smoke JSON files before the full run.
- Then run `/home/gemma4_exps/scripts/run_dflash_benchmarks.sh` from `/home/gemma4_exps` without `SMOKE=1`.
- Watch the run continuously. Use roughly a 60 second sleep/poll cadence.
- Keep logs under `/home/gemma4_exps/artifacts/logs`.
- Stop the server cleanly before starting the next variant.
- Keep the run observable with tmux if possible. A good session name is `gemma4_dflash` with `server` and `bench` windows.

## Server Command

Both DFlash servers must use these benchmark-critical settings:

```text
--max-model-len 32768
--max-num-batched-tokens 4096
--max-num-seqs 16
--gpu-memory-utilization 0.95
--generation-config vllm
--trust-remote-code
--language-model-only
--no-enable-prefix-caching
--attention-backend triton_attn
```

Do not use `--hf-overrides`. The validated PR head fixed the old multimodal-prefix attention issue without that workaround.

### Dense DFlash Server

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

### MoE DFlash Server

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

## Suite

### 1. Dense DFlash nt15

```bash
MODEL=google/gemma-4-31B-it
VARIANT=gemma4_31b_dflash_nt15
DRAFT_MODEL=z-lab/gemma-4-31B-it-DFlash
NUM_SPECULATIVE_TOKENS=15
ATTENTION_BACKEND=triton_attn
SPEC_ATTENTION_BACKEND=flash_attn
HF_OVERRIDES=none
PR_HEAD=a20847b5cc74dc0b471d715dde3ee7fa17c99eb8
```

### 2. MoE DFlash nt15

```bash
MODEL=google/gemma-4-26B-A4B-it
VARIANT=gemma4_26b_a4b_dflash_nt15
DRAFT_MODEL=z-lab/gemma-4-26B-A4B-it-DFlash
NUM_SPECULATIVE_TOKENS=15
ATTENTION_BACKEND=triton_attn
SPEC_ATTENTION_BACKEND=flash_attn
HF_OVERRIDES=none
PR_HEAD=a20847b5cc74dc0b471d715dde3ee7fa17c99eb8
```

For each variant, pass these values as environment variables on the same command line, or export them before invoking the script. Run the same env once with `SMOKE=1`, verify smoke output, then run again without `SMOKE=1` for the full sweep.

## Per-Variant Expectations

- Smoke output per variant: 2 JSON files under `/home/gemma4_exps/artifacts/smoke`.
- Full output per variant: 55 JSON files under `/home/gemma4_exps/artifacts/speed_bench`.
- Expected total full output: 110 JSON files.
- Results are saved flat, not nested by variant or category.
- Filename pattern: `<VARIANT>_<category>_c<concurrency>.json`.
- DFlash full-run filenames must start only with `gemma4_31b_dflash_nt15_` or `gemma4_26b_a4b_dflash_nt15_`.
- Existing MTP/baseline files such as `gemma4_31b_mtp_nt8_*`, `gemma4_31b_mtp_nt16_*`, `gemma4_26b_a4b_mtp_nt8_*`, `gemma4_26b_a4b_mtp_nt16_*`, and `gemma4_*_baseline_*` are read-only comparison inputs.
- The script should run all 11 SPEED-Bench qualitative categories.
- The script should run concurrencies `1, 2, 4, 8, 16`.

Example full output filenames:

```text
/home/gemma4_exps/artifacts/speed_bench/gemma4_31b_dflash_nt15_coding_c1.json
/home/gemma4_exps/artifacts/speed_bench/gemma4_31b_dflash_nt15_rag_c8.json
/home/gemma4_exps/artifacts/speed_bench/gemma4_26b_a4b_dflash_nt15_writing_c16.json
/home/gemma4_exps/artifacts/speed_bench/gemma4_26b_a4b_dflash_nt15_reasoning_c4.json
```

## Validation After Each Variant

- Confirm exactly 2 smoke JSON files exist for that variant.
- Confirm exactly 55 full JSON files exist for that variant.
- For every file, verify `completed`, `failed`, `generated_texts`, token counts, TTFT, TPOT, and ITL fields.
- Verify speculative decoding fields exist.
- Verify `num_speculative_tokens=15`.
- Verify `speculative_method=dflash`.
- Verify `attention_backend=triton_attn`.
- Verify `spec_attention_backend=flash_attn`.
- Verify `hf_overrides=none`.
- Verify `pr_head=a20847b5cc74dc0b471d715dde3ee7fa17c99eb8`.
- Per-position acceptance array length should be 15.
- Spot-check a few `generated_texts` manually for empty, corrupt, repetitive, or chat-template-looking output.

## Final Deliverable

- Write `/home/gemma4_exps/artifacts/speed_bench/dflash_benchmark_run_summary.md`.
- Include completed variants, file counts, failures if any, headline throughput, DFlash acceptance rate/length, and caveats.
- Compare DFlash against the matching baseline and the fastest matching MTP variant for each target model.
- Copy or summarize the final result back into the benchmark workspace if possible.
- Do not destroy the H100.
- If the full suite is complete and no process is running, pause the H100 to stop billing and clearly report that it was paused.

## Mental Model

DFlash uses the same benchmark client shape as MTP. The difference is the server:

- MTP server: Gemma assistant draft model with MTP speculative decoding.
- DFlash server: z-lab DFlash draft model with DFlash speculative decoding.

The script is only the benchmark client runner.
