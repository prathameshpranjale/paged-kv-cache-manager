"""Cross-check the C++ attention references against PyTorch's scaled_dot_product_attention.
Run inside the venv after building build-py. Prints a table; exit code 1 on any mismatch."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build-py"))
import numpy as np
import torch
import kvpage_py as kp

rng = np.random.default_rng(0)
bad = 0
print("| num_heads/kv_heads | head_dim | block | seq_lens | C++ ref vs torch | C++ paged(P=4) vs torch |")
print("|---|---|---|---|---|---|")
for nh, kvh, hd, B, lens in [(8, 2, 64, 16, [1, 15, 16, 17, 100]), (14, 2, 64, 8, [33, 64, 65]),
                             (8, 8, 128, 32, [5, 128, 129]), (4, 1, 32, 1, [3, 9]), (16, 4, 128, 16, [200, 7])]:
    ns = len(lens)
    mb = max((n + B - 1) // B for n in lens)
    nblocks = ns * mb + 3
    K = rng.standard_normal((nblocks, B, kvh, hd), dtype=np.float32)
    V = rng.standard_normal((nblocks, B, kvh, hd), dtype=np.float32)
    Q = rng.standard_normal((ns, nh, hd), dtype=np.float32)
    perm = rng.permutation(nblocks)  # scattered physical blocks, as a real allocator would give
    tables = np.zeros((ns, mb), dtype=np.int32)
    for i, n in enumerate(lens):
        for j in range((n + B - 1) // B):
            tables[i, j] = perm[i * mb + j]
    sl = np.array(lens, dtype=np.int32)
    ref = kp.reference_attention(nh, kvh, hd, B, Q, K, V, tables, sl)
    pag = kp.paged_attention_cpu(nh, kvh, hd, B, Q, K, V, tables, sl, 4)
    # torch: gather contiguous K/V per sequence, GQA via repeat_interleave, SDPA with one query token
    outs = []
    for i, n in enumerate(lens):
        blocks = torch.from_numpy(K)[torch.from_numpy(tables[i, : (n + B - 1) // B]).long()]
        k = blocks.reshape(-1, kvh, hd)[:n].permute(1, 0, 2).repeat_interleave(nh // kvh, 0)
        vb = torch.from_numpy(V)[torch.from_numpy(tables[i, : (n + B - 1) // B]).long()]
        v = vb.reshape(-1, kvh, hd)[:n].permute(1, 0, 2).repeat_interleave(nh // kvh, 0)
        q = torch.from_numpy(Q[i])[:, None, :]  # [nh, 1, hd]
        outs.append(torch.nn.functional.scaled_dot_product_attention(q, k, v)[:, 0])
    tref = torch.stack(outs).numpy()
    d1, d2 = float(np.abs(ref - tref).max()), float(np.abs(pag - tref).max())
    ok = d1 < 1e-5 and d2 < 1e-5
    bad += not ok
    print(f"| {nh}/{kvh} | {hd} | {B} | {lens} | {d1:.2e} | {d2:.2e} | {'' if ok else 'FAIL'}")
print("ALL OK" if not bad else f"{bad} MISMATCHES")
sys.exit(1 if bad else 0)
