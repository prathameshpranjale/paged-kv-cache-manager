# GPU checklist (M6)

Everything below the line "On the GPU machine" has **not been run**: this project was developed on a CPU-only machine.
What *was* verified before the move is listed first, so a failure on the GPU points at the right place.

## Verified on the CPU machine
- `kvp::reference_attention` (gather through the block table, plain double-precision softmax) equals PyTorch
  `scaled_dot_product_attention` to about 1e-7 (`python/attention_check.py`, 5 shape sets incl. GQA 14/2 and block size 1).
- `kvp::paged_attention_cpu(partitions = P)` implements the **same algorithm as the CUDA kernel** (in-place reads through the
  block table, per-warp online softmax, merge of partial results, empty partitions) and matches the reference to <1e-5
  for block sizes 1/4/8/16/32, lengths 1, B-1, B, B+1, 2B, 3B+1, more partitions than tokens, shared blocks, GQA ratios
  (`tests/test_attention.cpp`, 300+ random cases under ASan/UBSan).
- `cuda/paged_attention.cu` and `cuda/test_paged_attention.cu` **compile** (clang 18, sm_75, syntax + PTX generation,
  `scripts/cuda_compile_check.sh`). They have never executed.
- Block tables used by the test come from the real `KVManager` (including prefix-shared blocks).

## On the GPU machine (to do)
1. Prerequisites: NVIDIA driver, CUDA toolkit 11.8+ (12.x preferred), CMake 3.18+, a C++17 host compiler.
   `nvidia-smi` and `nvcc --version` must work. Nsight (optional) needs counter access: root on a cloud VM usually works,
   free notebook services usually block it.
2. Build and run the correctness test (sm_75 = T4; use 80 for A100, 86 for A10/3090, 89 for L4/4090):
   ```
   cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DKVPAGE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
   cmake --build build-cuda -j --target kvpage_cuda_test
   ./build-cuda/kvpage_cuda_test --no-bench      # correctness first
   ```
   Expect `ALL PASSED`. The tolerance is 1e-2 of max |reference| (fp16 inputs/outputs); typical error should be
   around 1e-3. If a case fails, compare with `kvp::paged_attention_cpu` for the same shape first: if the CPU model of the
   algorithm passes (it does) the bug is in the CUDA translation (indexing, shuffle, shared-memory merge).
3. Also run the CPU side on that machine: `./scripts/test.sh fast` (confirms the toolchain builds the library).
4. Benchmark: `./build-cuda/kvpage_cuda_test` prints ms/launch, unique KV bytes, GB/s and % of the device's peak memory
   bandwidth for Llama-3-8B and Qwen2.5-0.5B head shapes. Then `python cuda/bench_torch_sdpa.py` for the PyTorch SDPA
   baseline on the same shapes (needs torch with CUDA). Record GPU model, driver, CUDA version, clocks, command lines.
5. Profile (Nsight Systems for launch gaps, Nsight Compute for the kernel): check achieved DRAM throughput vs peak, occupancy,
   and warp stalls. Decode attention should be memory-bound; if achieved bandwidth is far below peak, the likely causes are
   below. State the bound you find (memory vs latency vs issue), with the profiler numbers.
6. Optimize one change at a time, re-running step 2 after each:
   - **GQA reuse:** version 1 launches one block per *query* head, so the `num_heads / kv_heads` heads of a group each re-read
     the same K/V (served from L2 at best). Process the whole group in one block and read K/V once.
   - **Vectorized loads:** read 8 bytes / 16 bytes per lane (`half2`/`float2`/`uint4`) instead of 2-byte scalar loads;
     for head_dim 64 a lane currently loads two halves at a time.
   - **More parallelism for short batches:** with few sequences, `num_heads * num_seqs` blocks cannot fill the GPU; split a long
     context across blocks (flash-decoding style: partial (max, sum, acc) per split, then a small reduction kernel).
   - **Prefetch / larger tiles:** process several tokens per warp iteration to hide latency; keep K and V loads in flight.
   - **Block-size effects:** compare block sizes 8/16/32; small blocks add table lookups, large blocks waste memory.
7. Report: achieved bandwidth vs peak, speed relative to PyTorch SDPA, correctness table. Say what was not measured.

## Honest status line to keep in the README until step 2 passes
"The CUDA kernel is written and compiles but has not been run; correctness is established for its CPU model only."
