"""End-to-end check: a real LLM running with its KV cache in blocks managed by the C++ KVManager
must produce the same logits/tokens as the stock contiguous HF KV cache.

Usage (inside WSL venv): python python/model_check.py [--model Qwen/Qwen2.5-0.5B-Instruct] [--out results/model_check.md]
Needs build-py/kvpage_py*.so (cmake -DKVPAGE_PYTHON=ON).
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "build-py"))
sys.path.insert(0, os.path.dirname(__file__))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from paged_model import PagedEngine

TOL = 1e-3  # max abs logit difference accepted (fp32, logits are O(10)); set from --dtype in main()
STRICT = True  # fp32: free-running greedy decoding must give identical tokens; reduced precision: teacher forced
TOLS = {"fp32": 1e-3, "fp16": 0.0, "bf16": 0.0}  # reduced precision: no HF tolerance, see test_equivalence
DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
rows = []  # (test, passed, detail)


def record(name, ok, detail):
    rows.append((name, ok, detail))
    print(("[ OK ] " if ok else "[FAIL] ") + name + "  " + detail, flush=True)


@torch.no_grad()
def ref_generate(model, prompt, n_new, forced_first=None):
    """Stock HF contiguous-cache greedy decoding. Returns (tokens, logits that produced each token)."""
    out = model(torch.tensor([prompt]), use_cache=True)
    pkv, logits = out.past_key_values, [out.logits[0, -1]]
    nxt = int(logits[0].argmax()) if forced_first is None else forced_first
    toks = []
    for i in range(n_new):
        toks.append(nxt)
        if i == n_new - 1:
            break
        out = model(torch.tensor([[nxt]]), past_key_values=pkv, use_cache=True)
        logits.append(out.logits[0, -1])
        nxt = int(logits[-1].argmax())
    return toks, logits


def paged_generate(eng, rid, seq, n_new, logits_out, first=None, parent=None, shared=0):
    """Prefill + greedy decode of one request on the paged engine."""
    lg = eng.prefill(rid, seq, parent, shared)
    assert lg is not None
    logits_out.append(lg)
    nxt = int(lg.argmax()) if first is None else first
    toks = [nxt]
    while len(toks) < n_new:
        lg = eng.decode(rid, toks[-1])
        assert lg is not None
        logits_out.append(lg)
        toks.append(int(lg.argmax()))
    return toks


def maxdiff(a, b):
    return max(float((x - y).abs().max()) for x, y in zip(a, b))


def test_equivalence(model, prompts, block_sizes, n_new):
    for B in block_sizes:
        for plen, prompt in prompts.items():
            ref_t, ref_l = ref_generate(model, prompt, n_new)
            eng = PagedEngine(model, num_blocks=160, block_size=B)
            got_l = []
            if STRICT:
                got_t = paged_generate(eng, 1, prompt, n_new, got_l)
                agree = float(got_t == ref_t)
            else:
                # Reduced precision: HF and this engine use different attention kernels, so they differ by rounding noise
                # that has nothing to do with paging. The pass criterion is therefore paging itself: the paged engine
                # must equal this same engine with one big block (B=256) bit for bit, both teacher forced along HF's tokens.
                lg = eng.prefill(1, prompt)
                got_l.append(lg)
                for t in ref_t[:-1]:
                    got_l.append(eng.decode(1, t))
                got_t = [int(x.argmax()) for x in got_l]
                agree = sum(a == b for a, b in zip(got_t, ref_t)) / len(ref_t)
                big = PagedEngine(model, num_blocks=2, block_size=256)
                bl = [big.prefill(1, prompt)]
                for t in ref_t[:-1]:
                    bl.append(big.decode(1, t))
                exact = all(torch.equal(a, b) for a, b in zip(bl, got_l))
            d = maxdiff(ref_l, got_l)
            if STRICT:
                ok = d < TOL and eng.kv.validate() == "" and agree == 1.0
            else:
                ok = exact and eng.kv.validate() == ""
            eng.free(1)
            ok = ok and eng.kv.free_blocks() == 160
            record(f"equivalence B={B} prompt_len={plen}", ok,
                   (f"top1_agreement={agree:.2f} max_logit_diff={d:.2e} steps={n_new}" if STRICT else
                    f"paged_equals_contiguous_layout_bit_exact={exact}; vs HF (noise floor, informational): "
                    f"max_logit_diff={d:.2f} top1_agreement={agree:.2f} (teacher forced, {n_new} steps)"))


def test_prefix_sharing(model, base, B=16, n_new=16):
    parent_prompt = base[:40]
    child_prompt = base[:32] + base[60:80]  # shares the first 32 tokens = 2 full blocks
    pt, pl = ref_generate(model, parent_prompt, n_new)
    ct, cl = ref_generate(model, child_prompt, n_new)
    eng = PagedEngine(model, num_blocks=64, block_size=B)
    gp, gc = [], []
    lg = eng.prefill(1, parent_prompt)
    gp.append(lg)
    p_toks = [int(lg.argmax())]
    lg = eng.prefill(2, child_prompt, parent=1, shared_tokens=32)
    gc.append(lg)
    c_toks = [int(lg.argmax())]
    shared_ok = (eng.kv.block_table(1)[:2] == eng.kv.block_table(2)[:2]
                 and all(eng.kv.refcount(b) == 2 for b in eng.kv.block_table(1)[:2])
                 and eng.kv.stats().blocks_saved == 2)
    saved = len(parent_prompt) + len(child_prompt) - eng.prefill_tokens_computed
    for _ in range(n_new - 1):  # interleave the two requests
        lg = eng.decode(1, p_toks[-1]); gp.append(lg); p_toks.append(int(lg.argmax()))
        lg = eng.decode(2, c_toks[-1]); gc.append(lg); c_toks.append(int(lg.argmax()))
    d = max(maxdiff(pl, gp), maxdiff(cl, gc))
    ok = p_toks == pt and c_toks == ct and d < TOL and shared_ok and saved == 32 and eng.kv.validate() == ""
    record("prefix sharing (2 shared blocks, interleaved decode)", ok,
           f"parent_eq={p_toks == pt} child_eq={c_toks == ct} max_logit_diff={d:.2e} "
           f"blocks_shared={shared_ok} prefill_tokens_skipped={saved}")
    eng.free(1); eng.free(2)
    record("prefix sharing: no leak after free", eng.kv.free_blocks() == 64, f"free={eng.kv.free_blocks()}/64")


def test_cow(model, base, B=16, n_new=14):
    prompt = base[:37]  # 2 full blocks + 5 tokens: last block is partial
    with torch.no_grad():
        first = model(torch.tensor([prompt])).logits[0, -1]
    top2 = [int(t) for t in first.topk(2).indices]
    pt, pl = ref_generate(model, prompt, n_new)  # parent takes the best token
    ct, cl = ref_generate(model, prompt, n_new, forced_first=top2[1])  # child takes the runner-up
    eng = PagedEngine(model, num_blocks=32, block_size=B)
    gp, gc = [], []
    lg = eng.prefill(1, prompt)
    gp.append(lg); gc.append(lg)
    p_toks, c_toks = [top2[0]], [top2[1]]
    eng.fork(2, 1)  # child shares ALL blocks, including the partial one
    shared_partial = eng.kv.refcount(eng.kv.block_table(1)[-1]) == 2
    for _ in range(n_new - 1):
        lg = eng.decode(2, c_toks[-1]); gc.append(lg); c_toks.append(int(lg.argmax()))  # child writes first -> COW
        lg = eng.decode(1, p_toks[-1]); gp.append(lg); p_toks.append(int(lg.argmax()))
    d = max(maxdiff(pl, gp), maxdiff(cl, gc))
    st = eng.kv.stats()
    ok = (p_toks == pt and c_toks == ct and d < TOL and shared_partial and st.cow_copies == 1
          and eng.cow_data_copies == 1 and top2[0] != top2[1])
    record("copy-on-write (fork at partial block, divergent continuations)", ok,
           f"parent_eq={p_toks == pt} child_eq={c_toks == ct} max_logit_diff={d:.2e} cow_copies={st.cow_copies}")
    eng.free(1); eng.free(2)
    record("copy-on-write: no leak after free", eng.kv.free_blocks() == 32 and eng.kv.validate() == "", "")


def test_preemption(model, prompts, B=8, num_blocks=16, out_len=20):
    """Several requests in a pool too small for all of them; the newest is preempted and recomputed."""
    refs = {rid: ref_generate(model, p, out_len)[0] for rid, p in prompts.items()}
    eng = PagedEngine(model, num_blocks=num_blocks, block_size=B)
    seqs = {rid: list(p) for rid, p in prompts.items()}   # prompt + generated so far
    gen = {rid: 0 for rid in prompts}
    waiting = list(prompts)
    running, done = [], {}
    preemptions = recomputed = 0
    peak_blocks = 0
    steps = 0
    while len(done) < len(prompts):
        steps += 1
        assert steps < 5000, "livelock"
        preempted = False
        for rid in list(running):
            if rid not in running:
                continue
            while True:
                lg = eng.decode(rid, seqs[rid][-1])
                if lg is not None:
                    break
                victim = running[-1]
                eng.free(victim); running.remove(victim); waiting.insert(0, victim)
                preemptions += 1; recomputed += len(seqs[victim]); preempted = True
                if victim == rid:
                    break
            if rid not in running:
                continue
            seqs[rid].append(int(lg.argmax())); gen[rid] += 1
            if gen[rid] == out_len:
                eng.free(rid); running.remove(rid); done[rid] = seqs[rid][len(prompts[rid]):]
        while waiting and not preempted:
            rid = waiting[0]
            reserve = 0 if not running else 1
            if eng.kv.free_blocks() - eng.kv.blocks_needed_to_admit(-1, 0, len(seqs[rid])) < reserve:
                break
            lg = eng.prefill(rid, seqs[rid])
            waiting.pop(0); running.append(rid)
            seqs[rid].append(int(lg.argmax())); gen[rid] += 1  # prefill yields the next token
            if gen[rid] == out_len:
                eng.free(rid); running.remove(rid); done[rid] = seqs[rid][len(prompts[rid]):]
        peak_blocks = max(peak_blocks, num_blocks - eng.kv.free_blocks())
        assert eng.kv.validate() == ""
    eq = all(done[r] == refs[r] for r in prompts)
    ok = eq and preemptions >= 1 and eng.kv.free_blocks() == num_blocks
    record("preemption + recompute (pool too small for all requests)", ok,
           f"all_tokens_equal_to_reference={eq} requests={len(prompts)} preemptions={preemptions} "
           f"recomputed_tokens={recomputed} peak_blocks={peak_blocks}/{num_blocks} free_at_end={eng.kv.free_blocks()}")


def test_random_prefix_sharing(model, base, seeds=range(8), n_new=8):
    """Random parent/child pairs: random block size, random shared length (often not a block multiple)."""
    import random
    for seed in seeds:
        rnd = random.Random(seed)
        B = rnd.choice([4, 8, 16])
        plen = rnd.randint(20, 70)
        shared = rnd.randint(1, plen)
        suffix = rnd.randint(3, 25)
        off = rnd.randint(0, 20)
        parent_prompt = base[:plen]
        child_prompt = base[:shared] + base[80 + off: 80 + off + suffix]
        pt, pl = ref_generate(model, parent_prompt, n_new)
        ct, cl = ref_generate(model, child_prompt, n_new)
        eng = PagedEngine(model, num_blocks=96, block_size=B)
        gp, gc = [], []
        lg = eng.prefill(1, parent_prompt); gp.append(lg); p_toks = [int(lg.argmax())]
        lg = eng.prefill(2, child_prompt, parent=1, shared_tokens=shared); gc.append(lg); c_toks = [int(lg.argmax())]
        expect_skip = min(shared, plen, len(child_prompt)) // B * B
        skipped = plen + len(child_prompt) - eng.prefill_tokens_computed
        for _ in range(n_new - 1):
            lg = eng.decode(1, p_toks[-1]); gp.append(lg); p_toks.append(int(lg.argmax()))
            lg = eng.decode(2, c_toks[-1]); gc.append(lg); c_toks.append(int(lg.argmax()))
        d = max(maxdiff(pl, gp), maxdiff(cl, gc))
        ok = p_toks == pt and c_toks == ct and d < TOL and skipped == expect_skip and eng.kv.validate() == ""
        eng.free(1); eng.free(2)
        ok = ok and eng.kv.free_blocks() == 96
        record(f"random prefix sharing seed={seed} B={B} parent={plen} shared={shared}", ok,
               f"parent_eq={p_toks == pt} child_eq={c_toks == ct} max_logit_diff={d:.2e} "
               f"prefill_tokens_skipped={skipped} (expected {expect_skip})")


info = []  # extra lines for the report (timings)


def test_batched_decode(model, base, B=16, n_new=12):
    """Several requests (different lengths, one shared-prefix child, one fork) decoded in ONE batched forward."""
    prompts = {1: base[:21], 2: base[:48], 3: base[:33], 4: base[5:70]}
    prompts[5] = base[:32] + base[90:100]  # shares 2 full blocks with request 2
    refs = {r: ref_generate(model, p, n_new) for r, p in prompts.items()}
    fork_ref = ref_generate(model, prompts[3], n_new)  # request 6 = fork of 3: same prompt, same greedy path
    eng = PagedEngine(model, num_blocks=96, block_size=B)
    logits = {r: [] for r in list(prompts) + [6]}
    toks = {}
    for r in (1, 2, 3, 4):
        lg = eng.prefill(r, prompts[r]); logits[r].append(lg); toks[r] = [int(lg.argmax())]
    lg = eng.prefill(5, prompts[5], parent=2, shared_tokens=32); logits[5].append(lg); toks[5] = [int(lg.argmax())]
    eng.fork(6, 3); logits[6].append(logits[3][0]); toks[6] = [toks[3][0]]  # fork shares the partial block
    rids = [1, 2, 3, 4, 5, 6]
    for _ in range(n_new - 1):
        out = eng.decode_batch(rids, [toks[r][-1] for r in rids])
        for r, lg in zip(rids, out):
            logits[r].append(lg); toks[r].append(int(lg.argmax()))
    d = max(max(maxdiff(refs[r][1], logits[r]) for r in prompts), maxdiff(fork_ref[1], logits[6]))
    eq = all(toks[r] == refs[r][0] for r in prompts) and toks[6] == fork_ref[0]
    st = eng.kv.stats()
    ok = eq and d < TOL and eng.kv.validate() == "" and st.cow_copies >= 1 and eng.cow_data_copies >= 1
    record("batched decode (6 requests, mixed lengths, shared prefix + fork/COW in one batch)", ok,
           f"tokens_equal={eq} max_logit_diff={d:.2e} cow_copies={st.cow_copies} blocks_saved_at_end={st.blocks_saved}")
    # speed info: the same 4 requests decoded one at a time vs batched
    def fresh():
        e = PagedEngine(model, num_blocks=96, block_size=B)
        first = {r: int(e.prefill(r, prompts[r]).argmax()) for r in (1, 2, 3, 4)}
        return e, first
    e1, last = fresh()
    t0 = time.time()
    for _ in range(n_new - 1):
        for r in (1, 2, 3, 4):
            last[r] = int(e1.decode(r, last[r]).argmax())
    t_seq = time.time() - t0
    e2, first = fresh()
    lastb = [first[r] for r in (1, 2, 3, 4)]
    t0 = time.time()
    for _ in range(n_new - 1):
        lastb = [int(x.argmax()) for x in e2.decode_batch([1, 2, 3, 4], lastb)]
    t_b4 = time.time() - t0
    info.append(f"decode of 4 requests x {n_new - 1} steps on CPU: one request at a time {t_seq:.2f}s, "
                f"batched {t_b4:.2f}s ({t_seq / t_b4:.1f}x)")


def scatter_pool(eng, n_blocks, seed=0):
    """Fragment the free list: allocate n single-block requests, free them in shuffled order, so later
    requests receive scattered, non-monotonic physical block ids."""
    import random
    rids = list(range(1000, 1000 + n_blocks))
    for r in rids:
        assert eng.kv.admit(r, 1)
    random.Random(seed).shuffle(rids)
    for r in rids:
        eng.kv.free(r)


def test_layout_invariance(model, base, n_new=10):
    """Same model, same inputs: the result must not depend on block size or on where blocks live.
    Reference = block size 256 (one block per sequence, contiguous). Others use scattered blocks and
    interleaved decoding. Compared with torch.equal: bit-exact, no tolerance."""
    prompts = {1: base[:37], 2: base[:64], 3: base[20:75]}
    ref = {}
    for r, p in prompts.items():
        e = PagedEngine(model, num_blocks=2, block_size=256)
        lg = e.prefill(1, p)
        logits, toks = [lg], [int(lg.argmax())]
        for _ in range(n_new - 1):
            lg = e.decode(1, toks[-1]); logits.append(lg); toks.append(int(lg.argmax()))
        ref[r] = (logits, toks)
    for B in (1, 3, 16):
        eng = PagedEngine(model, num_blocks=(200 if B > 1 else 400), block_size=B)
        scatter_pool(eng, 60 if B > 1 else 120, seed=B)
        got = {r: [eng.prefill(r, prompts[r])] for r in prompts}
        for step in range(n_new - 1):  # interleaved decoding, so physical blocks of requests interleave
            for r in prompts:
                got[r].append(eng.decode(r, ref[r][1][step]))
        exact = all(torch.equal(a, b) for r in prompts for a, b in zip(ref[r][0], got[r]))
        d = max(maxdiff(ref[r][0], got[r]) for r in prompts)
        tables = [eng.kv.block_table(r) for r in prompts]
        scattered = any(t != sorted(t) for t in tables)
        ok = exact and eng.kv.validate() == "" and scattered
        record(f"layout invariance B={B} (scattered blocks, interleaved decode) vs B=256", ok,
               f"bit_exact={exact} max_diff={d:.1e} physical_blocks_non_monotonic={scattered}")


def test_workload_compare(model, base, B=16, n_new=24, max_len=128):
    """8 requests (4 share a 48-token system prompt), stock HF contiguous cache one at a time vs the paged engine with
    continuous batching. Reports real KV memory and wall-clock on this CPU. Tokens must match."""
    system = base[:48]
    prompts = {}
    for i in range(8):
        tail = base[60 + 7 * i: 60 + 7 * i + 12 + 6 * i]
        prompts[i] = (system + tail) if i < 4 else base[100 + 5 * i: 100 + 5 * i + 25 + 8 * i]
    # stock HF, sequential
    t0 = time.time()
    ref = {i: ref_generate(model, p, n_new)[0] for i, p in prompts.items()}
    t_hf = time.time() - t0
    # paged: admit all, share the system prompt, batched decode
    eng = PagedEngine(model, num_blocks=200, block_size=B)
    t0 = time.time()
    toks, peak_blocks, peak_saved = {}, 0, 0
    for i, p in prompts.items():
        parent = 0 if (i < 4 and i > 0) else None
        lg = eng.prefill(i, p, parent=parent, shared_tokens=48 if parent is not None else 0)
        toks[i] = [int(lg.argmax())]
        st = eng.kv.stats()
        peak_blocks, peak_saved = max(peak_blocks, st.allocated_blocks), max(peak_saved, st.blocks_saved)
    active = list(prompts)
    while active:
        out = eng.decode_batch(active, [toks[i][-1] for i in active])
        for i, lg in zip(active, out):
            toks[i].append(int(lg.argmax()))
        for i in [i for i in active if len(toks[i]) == n_new]:
            eng.free(i); active.remove(i)
        st = eng.kv.stats()
        peak_blocks, peak_saved = max(peak_blocks, st.allocated_blocks), max(peak_saved, st.blocks_saved)
    t_paged = time.time() - t0
    eq = {i: toks[i] == ref[i] for i in prompts}
    bpt = eng.kv_bytes_per_token()
    total_tokens = sum(len(p) + n_new for p in prompts.values())
    exact_bytes = total_tokens * bpt
    maxlen_bytes = len(prompts) * max_len * bpt
    paged_bytes = peak_blocks * B * bpt
    computed = eng.prefill_tokens_computed
    prompt_tokens = sum(len(p) for p in prompts.values())
    ok = all(eq.values()) and eng.kv.free_blocks() == 200
    record("real workload: 8 requests, shared system prompt, continuous batching", ok,
           f"all_tokens_equal={all(eq.values())} prefill_tokens_computed={computed}/{prompt_tokens} "
           f"peak_blocks={peak_blocks} peak_blocks_saved_by_sharing={peak_saved}")
    info.append(f"workload ({len(prompts)} requests x {n_new} new tokens, KV {bpt / 1024:.1f} KiB/token): "
                f"KV memory, reserve {max_len} tokens per request: {maxlen_bytes / 2**20:.1f} MiB; "
                f"contiguous exact length: {exact_bytes / 2**20:.1f} MiB (sum of all final lengths); "
                f"paged peak (block {B}): {paged_bytes / 2**20:.1f} MiB")
    info.append(f"workload wall-clock on CPU: stock HF one request at a time {t_hf:.1f}s, paged engine with batched decode "
                f"{t_paged:.1f}s (pure-PyTorch gather; this measures harness overhead, not a kernel)")


def main():
    global TOL, STRICT
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--out", default="results/model_check.md")
    ap.add_argument("--dtype", default="fp32", choices=sorted(DTYPES))
    ap.add_argument("--quick", action="store_true", help="fewer prompts / block sizes / seeds (for large models)")
    args = ap.parse_args()
    TOL, STRICT = TOLS[args.dtype], args.dtype == "fp32"
    torch.manual_seed(0)
    torch.set_num_threads(os.cpu_count() or 1)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=DTYPES[args.dtype]).eval()
    text = ("Paged attention splits the key value cache of a language model into fixed size blocks and keeps a "
            "block table for every request, just like an operating system maps virtual pages to physical frames. "
            "Requests that share a system prompt can share the blocks that hold it, and a request that is evicted "
            "under memory pressure can be recomputed later from its tokens. ") * 3
    base = tok(text)["input_ids"]
    assert len(base) > 100
    t0 = time.time()

    sizes, lens = ((4, 16), (5, 32, 47)) if args.quick else ((1, 4, 8, 16, 32), (5, 16, 32, 47, 64, 100))
    prompts = {n: base[:n] for n in lens}  # includes exact multiples of 16 and of 32
    test_equivalence(model, prompts, block_sizes=sizes, n_new=16)
    if not args.quick:
        test_equivalence(model, {f"{n}@{o}": base[o:o + n] for n, o in ((23, 7), (58, 31), (81, 3))},
                         block_sizes=(8, 16), n_new=12)  # prompts that start mid-text
    test_layout_invariance(model, base)
    if STRICT:  # structure tests compare tokens with HF exactly: only meaningful where HF agrees to ~1e-5 (fp32)
        test_prefix_sharing(model, base)
        test_random_prefix_sharing(model, base, seeds=range(3 if args.quick else 8))
        test_cow(model, base)
        test_batched_decode(model, base)
    else:
        info.append(f"{args.dtype}: sharing / COW / batching / preemption checks need exact HF token agreement and run in fp32 only")
    test_workload_compare(model, base) if STRICT else None
    pre_prompts = {i: base[10 * i: 10 * i + 20 + 5 * i] for i in range(5)}
    if STRICT:
        test_preemption(model, pre_prompts)

    passed = sum(1 for r in rows if r[1])
    print(f"\n{passed}/{len(rows)} checks passed in {time.time() - t0:.0f}s")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("# Model check: real LLM with a paged KV cache vs the stock contiguous cache\n\n")
        f.write(f"- model: `{args.model}` ({args.dtype}, CPU, torch {torch.__version__}), greedy decoding\n")
        f.write(f"- paged side: C++ `KVManager` via pybind11 (`build-py/kvpage_py`); K/V tensors stored per block, "
                f"attention gathers through the block table\n")
        f.write(f"- reference: HF `model(..., past_key_values=DynamicCache)` one token at a time\n")
        f.write(f"- pass criteria: " + ("identical token ids and max |logit difference| < %g" % TOL if STRICT else "paged == same engine with one big block, bit-exact (torch.equal); the difference to HF is reported as the numeric noise floor" % ()) + "\n")
        f.write(f"- command: `python python/model_check.py --model {args.model} --dtype {args.dtype}{' --quick' if args.quick else ''}`; total {time.time() - t0:.0f}s; "
                f"{passed}/{len(rows)} checks passed\n\n")
        for line in info:
            f.write(f"- timing: {line}\n")
        f.write("\n| Check | Result | Detail |\n|---|---|---|\n")
        for name, ok, detail in rows:
            f.write(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |\n")
    sys.exit(0 if passed == len(rows) else 1)


if __name__ == "__main__":
    main()
