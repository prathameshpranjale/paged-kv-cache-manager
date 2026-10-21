# NOTES

## Status
M0 to M5 done and verified on WSL2 Ubuntu 24.04 (g++ 13.3, CMake 3.28): `./scripts/test.sh` passes (Debug, ASan+UBSan, TSan on the
concurrency tests); `./scripts/bench.sh` regenerates `results/`. M6 (GPU) not started.

## Decisions and differences from the spec
- **`fork()` added** to `KVManager`. With "only full blocks are shared" a shared block is never written, so the spec's COW rule
  would never fire. `fork` shares everything including a partial last block (parallel sampling / beam search), which is what makes
  COW testable. The simulator does not use `fork`, so benchmark COW counts are 0 (reported in the README).
- `admit_with_prefix` clamps `shared_tokens` to `min(shared, parent.tokens, prompt)` and shares `floor(that / B)` blocks.
  `blocks_needed_to_admit(parent, shared, prompt)` is the single source of truth for fit checks.
- **Prefix sharing needs a live parent** (a running request of the same prefix group that already holds the prefix). Real systems
  (SGLang, vLLM prefix caching) keep prefix blocks cached after the owner finishes; this simulator does not, so its sharing gain
  is a lower bound for that policy.
- Step model: admission = prefill in that step, first token in the next step. A request completes at the end of the step that
  produces its last token (`finish = step + 1`). Latency = finish - arrival; queue delay = first admission - arrival.
- Preemption = recompute only (swap to CPU not implemented). Victim = most recently admitted running request (may be the
  request that failed to grow). Preempted requests go to the front of the queue (FCFS order); SPF ignores queue position.
- The priority-class policy (optional in the spec) is not implemented.
- Utilization counts a shared block once (fill = max over its holders). Metrics means are over "active" steps (>= 1 running).
- Workload sampler: own splitmix64 PRNG and hand-written Uniform / capped-Pareto(alpha 1.5) / exponential samplers instead of
  `<random>` distributions, because those are not specified to give identical streams across standard libraries.
- Debug builds validate every KVManager operation (`set_check_each_op`, default on without NDEBUG); the simulator validates every
  step when `SimConfig::validate_each_step` is set (tests do, benchmarks do not).

## Problems hit
- No compiler on the Windows side; installed `g++ cmake make` in the existing WSL Ubuntu (apt) to build here.
- TSan aborted at start-up with `unexpected memory mapping` (high-entropy ASLR on WSL2). Fixed by running it under `setarch -R`.
- First benchmark loads left scenarios A to C with zero queueing (FCFS and SPF identical, nothing to compare). Raised arrival
  rates in A and C; B stays unloaded on purpose and is labelled "memory not binding".
- Initial SPF test used an impossible request (needed more blocks than the pool); the simulator correctly rejected it.
- Plots were made with a Windows Python that has matplotlib; WSL's python lacks it, so `bench.sh` skips plots there.

## Hand-off to the GPU machine (M6)
- Copy the whole folder; the CPU code has no dependencies besides CMake and a C++17 compiler.
- Put the kernel in `cuda/`: one query token per sequence, K/V gathered through `BlockTable.blocks` (int32 block ids, block size B),
  reference = gather blocks into contiguous K/V and run plain attention; test several B (8, 16, 32) and lengths including
  `tokens % B == 0`, and a shared block between two sequences. Tolerance for fp16 about 1e-2 relative (state what you use).
- Use `KVManager` + `generate_workload` to produce realistic block tables to feed the kernel.
- Everything reported so far is simulator output; no GPU number exists yet. Do not copy any until measured.

## Real-model check (added after M5)
- Route A: `python/kvpage_py.cpp` (pybind11, built with `-DKVPAGE_PYTHON=ON`) wraps the real `KVManager`; `python/paged_model.py` runs Qwen2 layers with
  K/V in `[layer][block][offset][kv_head][head_dim]` tensors and gathers through the block table. The harness (not the manager) copies block data on COW,
  detected by the last block id changing on an append into a partial block.
- Environment: WSL venv `~/kvenv` (torch 2.14 CPU, transformers 5.19, pybind11 3.1); model cached in `~/.cache/huggingface` (about 1 GB).
- First run passed 20/20; the negative control (COW without data copy) fails as it should.
- Limits: fp32 only, one request per forward (no batching), no GPU, 5 prompts, one model. The same harness can be reused as the reference side for M6.

## Added after the first model check
- `KVManager::stats()` is incremental (per-block fill counters); `recompute_stats()` is the slow reference and `validate()` compares both.
  Simulation went from 35 ms to 12 ms for scenario A.
- Parallel sampling: `Request::fork_of`, `WorkloadConfig::samples_per_request`, `SimConfig::fork_sharing`. Extra samples wait for their leader's first
  admission and are then forked (all blocks shared, COW on first append). A preempted sample becomes an independent request (loses sharing).
  Contiguous baselines treat samples as independent requests.
- CPU attention references (`paged_attention.h`), pybind exports, `python/attention_check.py` (vs torch SDPA).
- CUDA sources are only compile-checked (clang, no nvcc/GPU here): the pip `nvidia-cuda-nvcc-cu12` package has no `nvcc` binary, so
  `scripts/cuda_compile_check.sh` builds a fake CUDA root from pip pieces (runtime + cccl headers, libdevice, ptxas), stubs `cudaConfigureCall`
  (clang 18 does not know CUDA 12.9 headers) and a `curand` header. That proves it parses and generates PTX, nothing about running behaviour.
- Model check grew from 20 to 50 checks (block size 1 and 8, random prefix sharing, batched decode). Batched decode is decode-only; prefill is per request.
- Git Bash on this machine rewrites `/mnt/...` arguments passed to `wsl`; run scripts as `wsl -d Ubuntu -- bash -lc 'cd /mnt/e/AAAAProject && ...'`.
- Still not done: GPU run of M6, swap-to-CPU preemption, priority classes, persistent prefix cache, clang/-Werror builds, git history.

## Second round (all remaining items)
- **Prefix cache** (`KVManager::cache_prefix`, `admit_with_cached_prefix`, `evictable_blocks`): one reference per cached full block; LRU eviction inside
  `admit_*` and `append_token` before an allocation fails. `evictable_blocks()` must be *exact* (blocks no request references, minus the blocks the
  requester is about to share): an earlier conservative version was caught by the randomized test when two entries shared blocks.
  Simulator end-of-run calls `clear_cache()` before the leak check. Cached blocks count as allocated and as stored (they are full), so utilization is unaffected.
- **Priority policy**: highest class first (FCFS inside a class); if the top waiting request does not fit, strictly lower-priority running requests are preempted one
  by one (newest, lowest first). Memory-pressure victims are also lowest-class first. Contiguous baselines support the ordering only.
- **Swap preemption**: modelled as a CPU pool of blocks (`swap_capacity_blocks`); swapped requests re-enter via normal admission without sharing and without
  recompute; when the pool is full a preemption falls back to recompute. `prefill_tokens_per_step` / `swap_blocks_per_step` add a stall (the request holds memory
  but does not generate) so the two modes can be compared; **those rates are assumptions**. With rates of 0 both modes make identical scheduling decisions (tested).
- **Benchmark**: now multi-seed (`--seeds`, default 5) with mean ± std dev, scenarios F to I added; the old single-seed README numbers are superseded.
- **Tooling**: `-DKVPAGE_WERROR=ON` (g++ and clang++ clean with -Wshadow), `-DKVPAGE_COVERAGE=ON`, libFuzzer target `fuzz/fuzz_kv.cpp` (a 91 s run found nothing).
- **Python**: dtype-aware engine (fp32 / bf16 / fp16), Llama support (the rotary helper is shared), `decode_batch`, layout-invariance test, workload comparison.
  For reduced precision the HF comparison is informational (different kernels give rounding noise); the criterion is bit-exactness against the same engine with one big block.
- **Environment trouble, for whoever repeats this**: the WSL VM lives on C:, which had 0 bytes free after a 2 GB download, producing I/O errors; models above 1 GB now
  go to `E:\hf_cache` via `HF_HOME=/mnt/e/hf_cache`. fp32 TinyLlama (4.4 GB weights) crashed the 7.8 GB VM twice (swap lives on C:), so it was run in bf16.
  Do not leave long jobs in the background through `wsl.exe` here: the session was killed and the job with it. Run them in the foreground (10 min limit) with output to a file.
- **Git**: repository initialised; author identity is repo-local (see `git log`), change it with `git config user.name/user.email` if needed. No `LICENSE` file was added
  (the owner's choice); `THIRD_PARTY.md` lists credits and model licences.
- Still not done: running the CUDA kernel on a GPU, optimizing it, profiling, comparison with SDPA on a GPU; models above 1.1B; fp32 TinyLlama.
