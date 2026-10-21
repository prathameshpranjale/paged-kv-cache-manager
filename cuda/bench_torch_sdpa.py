"""Baseline for the spec's M6 comparison: PyTorch scaled_dot_product_attention on the same decode shapes
as cuda/test_paged_attention.cu's benchmark (contiguous K/V, one query token per sequence, fp16).

Run on the GPU machine:  python cuda/bench_torch_sdpa.py
Prints the same columns (ms/launch, unique KV MB, GB/s, % of peak) so rows can be compared directly.
This is a dense contiguous baseline, i.e. what you get WITHOUT paging; the paged kernel should be
within a small factor of it, and the point of paging is memory capacity, not beating this number.
"""
import torch

assert torch.cuda.is_available(), "needs a CUDA GPU"
dev = torch.device("cuda")
p = torch.cuda.get_device_properties(0)
print(f"device: {p.name}")


def bench(num_seqs, seq_len, nh, kvh, hd, iters=50):
    q = torch.randn(num_seqs, nh, 1, hd, device=dev, dtype=torch.float16)
    k = torch.randn(num_seqs, kvh, seq_len, hd, device=dev, dtype=torch.float16)
    v = torch.randn_like(k)
    rep = nh // kvh
    f = lambda: torch.nn.functional.scaled_dot_product_attention(
        q, k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1))
    # GQA expansion inside f() adds extra traffic; also time the expanded-in-advance variant (MHA-style baseline)
    ke, ve = k.repeat_interleave(rep, 1), v.repeat_interleave(rep, 1)
    g = lambda: torch.nn.functional.scaled_dot_product_attention(q, ke, ve)
    out = []
    for name, fn in (("sdpa (expands GQA each call)", f), ("sdpa (pre-expanded K/V)", g)):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(iters):
            fn()
        b.record()
        torch.cuda.synchronize()
        ms = a.elapsed_time(b) / iters
        unique = num_seqs * seq_len * kvh * hd * 2 * 2
        out.append((name, ms, unique / 1e6, unique / (ms * 1e-3) / 1e9))
    return out


print("| seqs | seq_len | heads q/kv/dim | variant | ms/launch | unique KV MB | GB/s |\n|---|---|---|---|---|---|---|")
for cfg in [(64, 2048, 32, 8, 128), (16, 4096, 32, 8, 128), (64, 2048, 14, 2, 64)]:
    for name, ms, mb, gbps in bench(*cfg):
        print(f"| {cfg[0]} | {cfg[1]} | {cfg[2]}/{cfg[3]}/{cfg[4]} | {name} | {ms:.3f} | {mb:.1f} | {gbps:.1f} |")
print("Divide GB/s by your GPU's peak memory bandwidth (T4: 320 GB/s) for % of peak.")
