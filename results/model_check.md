# Model check: real LLM with a paged KV cache vs the stock contiguous cache

- model: `Qwen/Qwen2.5-0.5B-Instruct` (fp32, CPU, torch 2.14.1+cpu), greedy decoding
- paged side: C++ `KVManager` via pybind11 (`build-py/kvpage_py`); K/V tensors stored per block, attention gathers through the block table
- reference: HF `model(..., past_key_values=DynamicCache)` one token at a time
- pass criteria: identical token ids and max |logit difference| < 0.001
- command: `python python/model_check.py --model Qwen/Qwen2.5-0.5B-Instruct --dtype fp32`; total 300s; 54/54 checks passed

- timing: decode of 4 requests x 11 steps on CPU: one request at a time 5.67s, batched 1.96s (2.9x)
- timing: workload (8 requests x 24 new tokens, KV 24.0 KiB/token): KV memory, reserve 128 tokens per request: 24.0 MiB; contiguous exact length: 17.3 MiB (sum of all final lengths); paged peak (block 16): 15.0 MiB
- timing: workload wall-clock on CPU: stock HF one request at a time 25.9s, paged engine with batched decode 8.4s (pure-PyTorch gather; this measures harness overhead, not a kernel)

| Check | Result | Detail |
|---|---|---|
| equivalence B=1 prompt_len=5 | PASS | top1_agreement=1.00 max_logit_diff=3.24e-05 steps=16 |
| equivalence B=1 prompt_len=16 | PASS | top1_agreement=1.00 max_logit_diff=1.87e-05 steps=16 |
| equivalence B=1 prompt_len=32 | PASS | top1_agreement=1.00 max_logit_diff=1.76e-05 steps=16 |
| equivalence B=1 prompt_len=47 | PASS | top1_agreement=1.00 max_logit_diff=2.21e-05 steps=16 |
| equivalence B=1 prompt_len=64 | PASS | top1_agreement=1.00 max_logit_diff=1.67e-05 steps=16 |
| equivalence B=1 prompt_len=100 | PASS | top1_agreement=1.00 max_logit_diff=1.93e-05 steps=16 |
| equivalence B=4 prompt_len=5 | PASS | top1_agreement=1.00 max_logit_diff=3.24e-05 steps=16 |
| equivalence B=4 prompt_len=16 | PASS | top1_agreement=1.00 max_logit_diff=1.87e-05 steps=16 |
| equivalence B=4 prompt_len=32 | PASS | top1_agreement=1.00 max_logit_diff=1.76e-05 steps=16 |
| equivalence B=4 prompt_len=47 | PASS | top1_agreement=1.00 max_logit_diff=2.21e-05 steps=16 |
| equivalence B=4 prompt_len=64 | PASS | top1_agreement=1.00 max_logit_diff=1.67e-05 steps=16 |
| equivalence B=4 prompt_len=100 | PASS | top1_agreement=1.00 max_logit_diff=1.93e-05 steps=16 |
| equivalence B=8 prompt_len=5 | PASS | top1_agreement=1.00 max_logit_diff=3.24e-05 steps=16 |
| equivalence B=8 prompt_len=16 | PASS | top1_agreement=1.00 max_logit_diff=1.87e-05 steps=16 |
| equivalence B=8 prompt_len=32 | PASS | top1_agreement=1.00 max_logit_diff=1.76e-05 steps=16 |
| equivalence B=8 prompt_len=47 | PASS | top1_agreement=1.00 max_logit_diff=2.21e-05 steps=16 |
| equivalence B=8 prompt_len=64 | PASS | top1_agreement=1.00 max_logit_diff=1.67e-05 steps=16 |
| equivalence B=8 prompt_len=100 | PASS | top1_agreement=1.00 max_logit_diff=1.93e-05 steps=16 |
| equivalence B=16 prompt_len=5 | PASS | top1_agreement=1.00 max_logit_diff=3.24e-05 steps=16 |
| equivalence B=16 prompt_len=16 | PASS | top1_agreement=1.00 max_logit_diff=1.87e-05 steps=16 |
| equivalence B=16 prompt_len=32 | PASS | top1_agreement=1.00 max_logit_diff=1.76e-05 steps=16 |
| equivalence B=16 prompt_len=47 | PASS | top1_agreement=1.00 max_logit_diff=2.21e-05 steps=16 |
| equivalence B=16 prompt_len=64 | PASS | top1_agreement=1.00 max_logit_diff=1.67e-05 steps=16 |
| equivalence B=16 prompt_len=100 | PASS | top1_agreement=1.00 max_logit_diff=1.93e-05 steps=16 |
| equivalence B=32 prompt_len=5 | PASS | top1_agreement=1.00 max_logit_diff=3.24e-05 steps=16 |
| equivalence B=32 prompt_len=16 | PASS | top1_agreement=1.00 max_logit_diff=1.87e-05 steps=16 |
| equivalence B=32 prompt_len=32 | PASS | top1_agreement=1.00 max_logit_diff=1.76e-05 steps=16 |
| equivalence B=32 prompt_len=47 | PASS | top1_agreement=1.00 max_logit_diff=2.21e-05 steps=16 |
| equivalence B=32 prompt_len=64 | PASS | top1_agreement=1.00 max_logit_diff=1.67e-05 steps=16 |
| equivalence B=32 prompt_len=100 | PASS | top1_agreement=1.00 max_logit_diff=1.93e-05 steps=16 |
| equivalence B=8 prompt_len=23@7 | PASS | top1_agreement=1.00 max_logit_diff=1.84e-05 steps=12 |
| equivalence B=8 prompt_len=58@31 | PASS | top1_agreement=1.00 max_logit_diff=1.38e-05 steps=12 |
| equivalence B=8 prompt_len=81@3 | PASS | top1_agreement=1.00 max_logit_diff=1.99e-05 steps=12 |
| equivalence B=16 prompt_len=23@7 | PASS | top1_agreement=1.00 max_logit_diff=1.84e-05 steps=12 |
| equivalence B=16 prompt_len=58@31 | PASS | top1_agreement=1.00 max_logit_diff=1.38e-05 steps=12 |
| equivalence B=16 prompt_len=81@3 | PASS | top1_agreement=1.00 max_logit_diff=1.99e-05 steps=12 |
| layout invariance B=1 (scattered blocks, interleaved decode) vs B=256 | PASS | bit_exact=True max_diff=0.0e+00 physical_blocks_non_monotonic=True |
| layout invariance B=3 (scattered blocks, interleaved decode) vs B=256 | PASS | bit_exact=True max_diff=0.0e+00 physical_blocks_non_monotonic=True |
| layout invariance B=16 (scattered blocks, interleaved decode) vs B=256 | PASS | bit_exact=True max_diff=0.0e+00 physical_blocks_non_monotonic=True |
| prefix sharing (2 shared blocks, interleaved decode) | PASS | parent_eq=True child_eq=True max_logit_diff=1.74e-05 blocks_shared=True prefill_tokens_skipped=32 |
| prefix sharing: no leak after free | PASS | free=64/64 |
| random prefix sharing seed=0 B=8 parent=68 shared=54 | PASS | parent_eq=True child_eq=True max_logit_diff=1.76e-05 prefill_tokens_skipped=48 (expected 48) |
| random prefix sharing seed=1 B=4 parent=56 shared=55 | PASS | parent_eq=True child_eq=True max_logit_diff=5.15e-05 prefill_tokens_skipped=52 (expected 52) |
| random prefix sharing seed=2 B=4 parent=25 shared=3 | PASS | parent_eq=True child_eq=True max_logit_diff=1.79e-05 prefill_tokens_skipped=0 (expected 0) |
| random prefix sharing seed=3 B=4 parent=57 shared=35 | PASS | parent_eq=True child_eq=True max_logit_diff=1.88e-05 prefill_tokens_skipped=32 (expected 32) |
| random prefix sharing seed=4 B=4 parent=39 shared=7 | PASS | parent_eq=True child_eq=True max_logit_diff=1.54e-05 prefill_tokens_skipped=4 (expected 4) |
| random prefix sharing seed=5 B=16 parent=36 shared=23 | PASS | parent_eq=True child_eq=True max_logit_diff=1.76e-05 prefill_tokens_skipped=16 (expected 16) |
| random prefix sharing seed=6 B=16 parent=25 shared=16 | PASS | parent_eq=True child_eq=True max_logit_diff=1.79e-05 prefill_tokens_skipped=16 (expected 16) |
| random prefix sharing seed=7 B=8 parent=29 shared=13 | PASS | parent_eq=True child_eq=True max_logit_diff=1.86e-05 prefill_tokens_skipped=8 (expected 8) |
| copy-on-write (fork at partial block, divergent continuations) | PASS | parent_eq=True child_eq=True max_logit_diff=1.84e-05 cow_copies=1 |
| copy-on-write: no leak after free | PASS |  |
| batched decode (6 requests, mixed lengths, shared prefix + fork/COW in one batch) | PASS | tokens_equal=True max_logit_diff=2.29e-05 cow_copies=1 blocks_saved_at_end=4 |
| real workload: 8 requests, shared system prompt, continuous batching | PASS | all_tokens_equal=True prefill_tokens_computed=403/547 peak_blocks=40 peak_blocks_saved_by_sharing=9 |
| preemption + recompute (pool too small for all requests) | PASS | all_tokens_equal_to_reference=True requests=5 preemptions=1 recomputed_tokens=46 peak_blocks=16/16 free_at_end=16 |
