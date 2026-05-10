# Codex Goal: Full Gemma 4 Baseline vs MTP Benchmark

Run the full Gemma 4 baseline vs MTP benchmark suite end to end.

## Important Constraint

Use only the existing benchmark script:

```text
/home/gemma4_exps/scripts/run_mtp_benchmarks.sh
```

Do not create a new benchmark script. Do not modify this script unless there is
a major blocker that makes the benchmark impossible to run. If you hit such a
blocker, document exactly what failed, why changing the script is necessary,
make the smallest possible change, and continue. This is a constrained
benchmark, so do not experiment with alternate benchmark commands or helper
scripts.

The benchmark script supports both baseline and MTP:

- Baseline: set `MODEL` and `VARIANT` only. Do not set `DRAFT_MODEL` or
  `NUM_SPECULATIVE_TOKENS`.
- MTP: set `MODEL`, `VARIANT`, `DRAFT_MODEL`, and
  `NUM_SPECULATIVE_TOKENS`.
- The script only sends benchmark requests. It does not start vLLM.
- You must start the correct vLLM server before running the script.

`VARIANT` is required. It is not just a display label: the script uses it in the
saved JSON filenames. Because results are saved flat in one directory, use the
exact variant names below so files do not collide and are easy to compare.

Allowed `VARIANT` values for this suite:

```text
gemma4_31b_baseline
gemma4_26b_a4b_baseline
gemma4_31b_mtp_nt8
gemma4_26b_a4b_mtp_nt8
gemma4_31b_mtp_nt16
gemma4_26b_a4b_mtp_nt16
```

## Workspace

- Start from the benchmark workspace.
- Use `benchmark_design.md` as the benchmark contract.
- H100 machine: JarvisLabs machine `407701`, named `gemma4_mtp`.
- H100 project root: `/home/gemma4_exps`.
- Use the existing venv: `/home/gemma4_exps/.venv`.
- Use the existing dataset: `/home/gemma4_exps/data/speed_bench`.

## Execution Rules

- Run variants sequentially on the same H100.
- For each variant, start the correct vLLM server first.
- Wait until `http://127.0.0.1:8000/v1/models` responds.
- Then run `/home/gemma4_exps/scripts/run_mtp_benchmarks.sh` from
  `/home/gemma4_exps`.
- Do not set `SMOKE=1`. This must be full mode.
- Watch the run continuously. Use roughly a 60 second sleep/poll cadence.
- Keep logs under `/home/gemma4_exps/artifacts/logs`.
- Stop the server cleanly before starting the next variant.
- Prefer the existing tmux session `gemma4_mtp` with windows for `server` and
  `bench` so I can inspect the machine. If tmux gets in the way, you may avoid
  tmux and use `jl exec` or `jl run`, but keep the run observable and well
  logged.

## Suite

### 1. Dense Baseline

```bash
MODEL=google/gemma-4-31B-it
VARIANT=gemma4_31b_baseline
```

Do not set `DRAFT_MODEL`. Do not set `NUM_SPECULATIVE_TOKENS`.

### 2. MoE Baseline

```bash
MODEL=google/gemma-4-26B-A4B-it
VARIANT=gemma4_26b_a4b_baseline
```

Do not set `DRAFT_MODEL`. Do not set `NUM_SPECULATIVE_TOKENS`.

### 3. Dense MTP nt8

```bash
MODEL=google/gemma-4-31B-it
VARIANT=gemma4_31b_mtp_nt8
DRAFT_MODEL=google/gemma-4-31B-it-assistant
NUM_SPECULATIVE_TOKENS=8
```

### 4. MoE MTP nt8

```bash
MODEL=google/gemma-4-26B-A4B-it
VARIANT=gemma4_26b_a4b_mtp_nt8
DRAFT_MODEL=google/gemma-4-26B-A4B-it-assistant
NUM_SPECULATIVE_TOKENS=8
```

### 5. Dense MTP nt16

```bash
MODEL=google/gemma-4-31B-it
VARIANT=gemma4_31b_mtp_nt16
DRAFT_MODEL=google/gemma-4-31B-it-assistant
NUM_SPECULATIVE_TOKENS=16
```

### 6. MoE MTP nt16

```bash
MODEL=google/gemma-4-26B-A4B-it
VARIANT=gemma4_26b_a4b_mtp_nt16
DRAFT_MODEL=google/gemma-4-26B-A4B-it-assistant
NUM_SPECULATIVE_TOKENS=16
```

## Per-Variant Expectations

- The script should run all 11 SPEED-Bench qualitative categories.
- The script should run concurrencies `1, 2, 4, 8, 16`.
- Expected output per variant: 55 JSON files.
- Expected total output: 330 JSON files.
- Output directory: `/home/gemma4_exps/artifacts/speed_bench`.
- Results are saved flat, not nested by variant or category.
- Filename pattern: `<VARIANT>_<category>_c<concurrency>.json`.

Example output filenames:

```text
/home/gemma4_exps/artifacts/speed_bench/gemma4_31b_baseline_coding_c1.json
/home/gemma4_exps/artifacts/speed_bench/gemma4_31b_baseline_rag_c8.json
/home/gemma4_exps/artifacts/speed_bench/gemma4_26b_a4b_baseline_writing_c16.json
/home/gemma4_exps/artifacts/speed_bench/gemma4_31b_mtp_nt8_coding_c1.json
/home/gemma4_exps/artifacts/speed_bench/gemma4_26b_a4b_mtp_nt8_rag_c8.json
/home/gemma4_exps/artifacts/speed_bench/gemma4_31b_mtp_nt16_writing_c16.json
/home/gemma4_exps/artifacts/speed_bench/gemma4_26b_a4b_mtp_nt16_reasoning_c4.json
```

## Validation After Each Variant

- Confirm exactly 55 JSON files exist for that variant.
- For every file, verify `completed`, `failed`, `generated_texts`, token counts,
  TTFT, TPOT, and ITL fields.
- For baseline files, speculative decoding fields may be absent. That is
  expected.
- For MTP files, verify speculative decoding fields exist.
- For nt8, per-position acceptance array length must be 8.
- For nt16, per-position acceptance array length must be 16.
- Spot-check a few `generated_texts` manually for empty, corrupt, repetitive, or
  chat-template-looking output.

## Final Deliverable

- Write `/home/gemma4_exps/artifacts/speed_bench/benchmark_run_summary.md`.
- Include completed variants, file counts, failures if any, headline throughput,
  MTP acceptance rate/length, and caveats.
- Copy or summarize the final result back into the benchmark workspace if possible.
- Do not destroy the H100.
- If the full suite is complete and no process is running, pause the H100 to
  stop billing and clearly report that it was paused.

## Mental Model

Baseline and MTP use the same benchmark script. The difference is the server:

- Baseline server: start vLLM without `--speculative-config`.
- MTP server: start vLLM with `--speculative-config`.

The script is only the benchmark client runner.
