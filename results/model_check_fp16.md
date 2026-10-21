# Model check: real LLM with a paged KV cache vs the stock contiguous cache

- model: `Qwen/Qwen2.5-0.5B-Instruct` (fp16, CPU, torch 2.14.1+cpu), greedy decoding
- paged side: C++ `KVManager` via pybind11 (`build-py/kvpage_py`); K/V tensors stored per block, attention gathers through the block table
- reference: HF `model(..., past_key_values=DynamicCache)` one token at a time
- pass criteria: paged == same engine with one big block, bit-exact (torch.equal); the difference to HF is reported as the numeric noise floor
- command: `python python/model_check.py --model Qwen/Qwen2.5-0.5B-Instruct --dtype fp16 --quick`; total 49s; 9/9 checks passed

- timing: fp16: sharing / COW / batching / preemption checks need exact HF token agreement and run in fp32 only

| Check | Result | Detail |
|---|---|---|
| equivalence B=4 prompt_len=5 | PASS | paged_equals_contiguous_layout_bit_exact=True; vs HF (noise floor, informational): max_logit_diff=0.09 top1_agreement=1.00 (teacher forced, 16 steps) |
| equivalence B=4 prompt_len=32 | PASS | paged_equals_contiguous_layout_bit_exact=True; vs HF (noise floor, informational): max_logit_diff=0.08 top1_agreement=1.00 (teacher forced, 16 steps) |
| equivalence B=4 prompt_len=47 | PASS | paged_equals_contiguous_layout_bit_exact=True; vs HF (noise floor, informational): max_logit_diff=0.05 top1_agreement=1.00 (teacher forced, 16 steps) |
| equivalence B=16 prompt_len=5 | PASS | paged_equals_contiguous_layout_bit_exact=True; vs HF (noise floor, informational): max_logit_diff=0.09 top1_agreement=1.00 (teacher forced, 16 steps) |
| equivalence B=16 prompt_len=32 | PASS | paged_equals_contiguous_layout_bit_exact=True; vs HF (noise floor, informational): max_logit_diff=0.08 top1_agreement=1.00 (teacher forced, 16 steps) |
| equivalence B=16 prompt_len=47 | PASS | paged_equals_contiguous_layout_bit_exact=True; vs HF (noise floor, informational): max_logit_diff=0.05 top1_agreement=1.00 (teacher forced, 16 steps) |
| layout invariance B=1 (scattered blocks, interleaved decode) vs B=256 | PASS | bit_exact=True max_diff=0.0e+00 physical_blocks_non_monotonic=True |
| layout invariance B=3 (scattered blocks, interleaved decode) vs B=256 | PASS | bit_exact=True max_diff=0.0e+00 physical_blocks_non_monotonic=True |
| layout invariance B=16 (scattered blocks, interleaved decode) vs B=256 | PASS | bit_exact=True max_diff=0.0e+00 physical_blocks_non_monotonic=True |
