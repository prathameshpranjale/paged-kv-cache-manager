#pragma once
#include <vector>

#include "kvpage/metrics.h"
#include "kvpage/workload.h"

namespace kvp {

// FCFS: head of the queue only. ShortestPromptFirst: shortest prompt that fits.
// Priority: highest priority first (FCFS within a class); a waiting request may preempt strictly lower-priority
// running requests, and memory-pressure victims are chosen lowest priority first.
enum class Policy { FCFS, ShortestPromptFirst, Priority };
// What happens to a preempted request: recompute its context later (prefill), or swap its blocks to a CPU pool.
enum class PreemptMode { Recompute, Swap };
const char* policy_name(Policy p);

struct SimConfig {
    int num_blocks = 1024;
    int block_size = 16;
    Policy policy = Policy::FCFS;
    int watermark_blocks = 2;  // free blocks kept back so running requests can grow
    int max_running = 0;       // 0: unlimited
    PreemptMode preempt_mode = PreemptMode::Recompute;
    int swap_capacity_blocks = 1 << 30;  // CPU pool size; when full a preemption falls back to recompute
    // Cost model (0 = free). A request that is (re)admitted does not generate for this many steps:
    int prefill_tokens_per_step = 0;  // recompute/prefill speed: stall = ceil(tokens to compute / rate)
    int swap_blocks_per_step = 0;     // PCIe speed: swap-in stall = ceil(blocks / rate)
    bool priority_preemption = true;  // Policy::Priority: waiting request preempts lower-priority running ones
    bool prefix_cache = false;  // keep full prefix blocks after their owner finishes (reclaimed when memory is needed)
    bool prefix_sharing = true;
    bool fork_sharing = true;  // parallel samples fork the leader (shared blocks + COW); false: independent requests
    bool validate_each_step = false;  // run every invariant check after every step
    int64_t max_steps = 50000000;     // livelock guard (throws)
};

// Paged KV cache + continuous batching. Throws std::invalid_argument if some
// request can never fit, std::logic_error on an invariant violation.
Metrics run_paged(const std::vector<Request>& requests, const SimConfig& cfg);

struct ContigConfig {
    int64_t total_slots = 16384;  // same memory budget as the paged run
    int reserve_len = 0;          // slots reserved per request; 0: exactly prompt+output (oracle)
    Policy policy = Policy::FCFS;
    int max_running = 0;
    int64_t max_steps = 50000000;
};

// Baseline: each admitted request reserves `reserve_len` slots until it finishes.
Metrics run_contiguous(const std::vector<Request>& requests, const ContigConfig& cfg);

}  // namespace kvp
