# Project: a paged KV-cache manager and batching scheduler in C++

A self-contained specification. It can be handed to a person or a coding assistant on another machine (including one with an NVIDIA GPU) without any other context. Read "How to use this document" first.

## How to use this document

1. Build the milestones **in order** (M0 to M6). M0 to M5 need no GPU and no AI model; only M6 needs a GPU.
2. After each milestone: build, run the tests, and report the real output. Do not claim a result that was not run.
3. Every number in the README must come from a run you did, with the command and machine noted.
4. Keep a short `NOTES.md` of decisions, problems and anything that differs from this spec.

## 1. What this is, in plain words

A large language model (LLM) writes text one token (word piece) at a time. To pick the next token it needs the **keys and values** (K and V) of every earlier token. Instead of recomputing them at every step, the system saves them: this saved data is the **KV cache**.

The KV cache is large (about 0.5 MiB per token for a 7B model with full multi-head attention, see section 3), and a server handles many requests at once whose final lengths are unknown. Reserving the maximum length for every request wastes most of the memory. **PagedAttention** (the idea behind vLLM) fixes this with the same trick operating systems use for RAM: split the cache into small fixed-size **blocks**, give a request a new block only when it needs one, and keep a **block table** per request that maps its token positions to physical blocks. Blocks can be shared between requests (copy-on-write) and requests can be paused under memory pressure.

This project re-implements that memory manager and the scheduler around it, as an educational C++ project, and measures how much memory it saves compared with contiguous reservation.

**This is not a novel invention.** vLLM (paged KV cache), SGLang (prefix sharing with a radix tree) and TensorRT-LLM have production versions. Credit them: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention" (SOSP 2023). State in the README that this is a learning implementation.

## 2. Goals and non-goals

**Goals**
- A correct block allocator, block tables, prefix sharing with copy-on-write, and preemption.
- A scheduler with continuous batching and at least two policies, compared on a simulated workload.
- Strong tests: invariants, randomized stress, sanitizers, a concurrency test.
- Honest measurements of memory utilization, throughput and latency, versus a contiguous-reservation baseline.
- (M6, optional) a GPU decode-attention kernel that reads K/V through the block table, verified against a reference.

**Non-goals**
- Beating vLLM, a full LLM runtime, training, or multi-GPU.
- Network serving, tokenization, or sampling quality (token ids are enough).

## 3. Background numbers (use for sizing and for the README)

KV cache bytes per token = `2 (K and V) x layers x kv_heads x head_dim x bytes_per_value`.

| Model (published configs; verify on each model's page) | Layers | KV heads | Head dim | fp16 bytes/token | 4096 tokens |
|---|---|---|---|---|---|
| Llama-2-7B (multi-head) | 32 | 32 | 128 | 524,288 (0.5 MiB) | 2 GiB |
| Llama-3-8B (grouped-query) | 32 | 8 | 128 | 131,072 (128 KiB) | 0.5 GiB |

With a block of `B` tokens, one block holds `B x bytes/token` bytes (B = 16 and Llama-3-8B: 2 MiB per block).

## 4. Environment

| Milestones | Requirement |
|---|---|
| M0 to M5 | Linux (or WSL), a C++17 compiler (g++ 9+ or clang), CMake 3.16+, 2 to 4 GB RAM. No GPU. Sanitizers (`-fsanitize=address,undefined` and `-fsanitize=thread`). Python 3 + matplotlib only if you want plots. |
| M6 (optional) | NVIDIA GPU, CUDA toolkit 11.8+ (12.x preferred), CMake CUDA support. A 16 GB T4 (for example an AWS g4dn.xlarge) is enough for a 1 to 1.5B-parameter model in fp16. Nsight tools are optional but useful (they need counter access: root on a cloud VM usually works; free notebook services often block it). |

For M6 only, a small open model for correctness checks (check each licence before use or publication): TinyLlama 1.1B, or a small Qwen2.5 model. Do not use any employer's models, data or code.

## 5. Design

### 5.1 Modules

```
include/ and src/
  block_allocator   fixed-size blocks, free list, reference counts
  block_table       per-request mapping: logical block index -> physical block id
  kv_manager        allocate/extend/fork/free requests; copy-on-write; accounting
  scheduler         waiting/running/preempted queues, policies, continuous batching
  workload          seeded request generator (arrival, prompt length, output length, shared prefixes)
  simulator         step-based engine that drives scheduler + kv_manager, collects metrics
  metrics           utilization, waste, throughput, latency percentiles, preemptions, COW copies
tests/              unit, property/randomized, concurrency
bench/              baseline (contiguous) vs paged, policy comparison
cuda/ (M6)          decode attention kernel + reference + correctness test
```

### 5.2 Core API (sketch; adjust as needed)

```cpp
using BlockId = int32_t;
constexpr BlockId kNoBlock = -1;

class BlockAllocator {
public:
    BlockAllocator(int num_blocks);
    BlockId allocate();                 // kNoBlock if none free; sets refcount = 1
    void    retain(BlockId b);          // refcount++ (sharing)
    void    release(BlockId b);         // refcount--; returns to free list at 0
    int     refcount(BlockId b) const;
    int     free_blocks() const;
    int     total_blocks() const;
};

class KVManager {
public:
    KVManager(int num_blocks, int block_size);
    bool can_admit(int prompt_tokens) const;
    bool admit(RequestId r, int prompt_tokens);                 // allocate ceil(prompt/B) blocks
    bool admit_with_prefix(RequestId r, RequestId parent,       // share the parent's FULL prefix blocks
                           int shared_tokens, int prompt_tokens);
    bool append_token(RequestId r);                             // may allocate a block; COW if last block is shared
    void free(RequestId r);                                     // release all blocks
    const BlockTable& table(RequestId r) const;
    Stats stats() const;                                        // used tokens, allocated blocks, internal waste, COW copies
};
```

Rules to implement:
- A request with `n` tokens owns `ceil(n / B)` blocks; only the last block may be partly filled.
- **Prefix sharing:** only completely filled blocks are shared (`retain`). A partial block is private.
- **Copy-on-write:** when appending to a request whose last block has `refcount > 1`, allocate a new block, record a copy (count it), `release` the old one, and continue.
- `append_token` returns false when no block is available; the scheduler then preempts.

### 5.3 Scheduler (continuous batching)

Each simulation step:
1. Every running request generates one token (`append_token`). Finished requests are freed immediately.
2. If an `append_token` fails, **preempt** a victim (default: the most recently admitted running request): free its blocks and put it at the front of the waiting queue to **recompute** later (its already-generated tokens become part of its new prompt). Optional: swap to a CPU pool instead (count the transfers).
3. Admit waiting requests while `can_admit` holds, keeping a small watermark of free blocks so running requests can keep growing.

Implement at least these policies, selectable at run time:
- **FCFS** (first come, first served admission).
- **Shortest-prompt-first** (admit the shortest waiting prompt that fits).
- (optional) **Priority classes** with preemption of lower classes.

### 5.4 Baseline for comparison

**Contiguous reservation:** each admitted request reserves `max_len` token slots up front and holds them until it finishes. Same workload, same total memory budget. Report how many requests it can run concurrently versus paged.

## 6. Workload and metrics

Workload generator (seeded, so runs are reproducible): requests with arrival step, prompt length and output length drawn from configurable distributions (for example uniform and a long-tailed one), a configurable fraction of requests sharing a common system-prompt prefix, and a memory budget in blocks.

Metrics to record per run:
- **Memory utilization** = tokens actually stored / token slots allocated (paged and contiguous).
- **Internal waste** (unused slots in last blocks) and **external waste** (should be zero for paged).
- **Concurrent requests** sustained, and **tokens generated per step** (throughput proxy).
- Request **latency** (arrival to completion) p50 and p95, queueing delay.
- **Preemptions** and recomputed tokens; **COW copies** and **blocks saved by prefix sharing**.

State clearly in the README that the simulator measures memory management and scheduling, **not GPU speed**.

## 7. Testing requirements

1. **Invariants** (check after every operation in debug builds and after every simulation step):
   - Every block is either on the free list or has `refcount >= 1`, never both; `free_blocks + referenced_blocks == total_blocks`.
   - Each request's `ceil(tokens/B)` equals its block-table size.
   - Sum of reference counts equals the total number of block-table entries across all requests.
   - No leak: after all requests finish, every block is free.
2. **Unit tests:** allocator exhaustion and reuse, admit/append/free, boundary cases (`tokens % B == 0`), prefix sharing, copy-on-write, preemption and recompute.
3. **Randomized / property tests:** thousands of seeded random operation sequences with the invariants checked each step (shrink to a failing seed and print it).
4. **Sanitizers:** run the whole suite under AddressSanitizer + UndefinedBehaviorSanitizer; run the concurrency test under ThreadSanitizer.
5. **Concurrency test:** a thread-safe request intake queue (bounded, mutex + condition variable or lock-free for one producer and one consumer) feeding the scheduler thread; multiple producer threads; no lost or duplicated requests.
6. **Determinism:** the same seed gives identical metrics.

## 8. Milestones and acceptance criteria

| # | Milestone | Done when |
|---|---|---|
| M0 | Project skeleton: CMake, test runner, sanitizer build options, CI script | `cmake --build` and the test command work, sanitizer builds work |
| M1 | `BlockAllocator` + tests | allocation, release, refcounts, exhaustion pass; invariants hold |
| M2 | `BlockTable` + `KVManager` (admit, append, free) | per-request block counts correct incl. boundary cases; no leaks |
| M3 | Prefix sharing + copy-on-write | shared blocks counted once; COW triggers exactly when expected; randomized tests pass |
| M4 | Scheduler + simulator + metrics (FCFS and one more policy), preemption/recompute | simulation runs end to end, deterministic by seed, invariants hold every step |
| M5 | Benchmarks: paged vs contiguous baseline, policy comparison; README with tables and plots; ThreadSanitizer concurrency test | results table produced by a script; commands to reproduce documented |
| M6 (optional, GPU) | Decode-attention kernel reading K/V through the block table | matches a plain reference within tolerance (fp16: about 1e-2 relative, state yours) for several block sizes and lengths, then benchmarked against PyTorch SDPA on the same shapes |

M6 details: reference = straightforward attention over a contiguous K/V built by gathering the blocks. Kernel = one query token per sequence, K/V gathered through the block table, softmax over the sequence, weighted sum of V. Start simple (one thread block per head per sequence), verify, then optimize (vectorized loads, shared-memory reduction) and profile (Nsight Systems/Compute) to explain the bound (decode attention is normally memory-bound). Report achieved memory bandwidth versus the GPU's peak.

## 9. Deliverables

- Repository with the layout above, `README.md`, `NOTES.md`, a one-command build and test (`./scripts/test.sh`), and a one-command benchmark (`./scripts/bench.sh`) that regenerates the results table.
- README sections: what and why, credit and limits, design (with a diagram of block tables and blocks), how to build and run, results (tables with the exact command, machine, seed), lessons learned.
- A results table of the form: memory utilization, concurrent requests, p50/p95 latency, preemptions, COW copies for each policy and for the contiguous baseline. **Fill from real runs only.**

## 10. Rules

- Correctness first: tests before optimization. A finished, well-tested manager beats a half-working kernel.
- No claim without a run: do not invent speed-ups, savings or latency numbers. If something was not run, say so.
- Credit the sources: the PagedAttention paper and vLLM, and anything you adapted (with licences).
- No employer IP: no company models, data or code.
- Keep it small enough to explain every line in an interview.

## 11. What each part teaches (for the project owner)

| Idea | Where it shows up |
|---|---|
| Paging and page tables | block tables map logical positions to physical blocks |
| Allocation, free lists, fragmentation | the block allocator; internal waste versus contiguous reservation |
| Copy-on-write | prefix sharing and `append_token` on a shared last block |
| Page replacement / swapping | preemption with recompute (or swap) under memory pressure |
| Scheduling | continuous batching and the admission policies |
| Concurrency | the intake queue, invariants under threads, ThreadSanitizer |
| GPU memory-bound kernels (M6) | decode attention bandwidth analysis |
