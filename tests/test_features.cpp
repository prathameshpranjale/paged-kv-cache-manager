// Persistent prefix cache, priority policy, swap-vs-recompute preemption.
#include <vector>

#include "kvpage/kv_manager.h"
#include "kvpage/simulator.h"
#include "kvpage/workload.h"
#include "testing.h"

using namespace kvp;

namespace {
Request make(RequestId id, int arrival, int prompt, int output, int priority = 0) {
    Request r;
    r.id = id;
    r.arrival = arrival;
    r.prompt_len = prompt;
    r.output_len = output;
    r.priority = priority;
    return r;
}
const RequestResult& result_of(const Metrics& m, RequestId id) {
    for (const auto& r : m.results)
        if (r.id == id) return r;
    static RequestResult none;
    CHECK(false);
    return none;
}
}  // namespace

// ------------------------------------------------------------------ prefix cache (manager level)

TEST(cache_prefix_outlives_its_owner) {
    KVManager kv(10, 16);
    CHECK(kv.admit(1, 40));                    // 2 full blocks + 8 tokens
    CHECK(kv.cache_prefix(7, 1, 40));          // caches the 2 full blocks (32 tokens)
    CHECK_EQ(kv.cached_prefix_tokens(7), 32);
    CHECK(!kv.cache_prefix(7, 1, 40));         // key exists
    kv.free(1);
    CHECK_EQ(kv.free_blocks(), 8);             // the 2 cached blocks stay allocated
    CHECK_EQ(kv.stats().cached_blocks, 2);
    CHECK_EQ(kv.evictable_blocks(), 2);
    CHECK_EQ(kv.validate(), std::string());
    CHECK_EQ(kv.blocks_needed_to_admit_cached(7, 32, 50), 4 - 2);
    CHECK(kv.admit_with_cached_prefix(2, 7, 32, 50));
    CHECK_EQ(kv.free_blocks(), 6);             // only 2 new blocks
    CHECK_EQ(kv.table(2).blocks.size(), 4u);
    CHECK_EQ(kv.evictable_blocks(), 0);        // now shared with a running request
    CHECK_EQ(kv.validate(), std::string());
    kv.free(2);
    kv.clear_cache();
    CHECK_EQ(kv.free_blocks(), 10);
    CHECK_EQ(kv.validate(), std::string());
}

TEST(cache_too_short_prefix_is_not_cached) {
    KVManager kv(4, 16);
    CHECK(kv.admit(1, 15));
    CHECK(!kv.cache_prefix(1, 1, 15));  // no completely filled block
    CHECK(!kv.has_cached_prefix(1));
}

TEST(cache_is_reclaimed_when_admission_needs_memory) {
    KVManager kv(4, 16);
    CHECK(kv.admit(1, 32));
    CHECK(kv.cache_prefix(0, 1, 32));
    kv.free(1);
    CHECK_EQ(kv.free_blocks(), 2);
    CHECK(kv.can_admit(64));                   // available = free + evictable
    CHECK(kv.admit(2, 64));                    // evicts the cache entry to make room
    CHECK(!kv.has_cached_prefix(0));
    CHECK_EQ(kv.stats().cache_evictions, 1);
    CHECK_EQ(kv.free_blocks(), 0);
    CHECK_EQ(kv.validate(), std::string());
}

TEST(cache_is_reclaimed_when_append_needs_a_block) {
    KVManager kv(3, 16);
    CHECK(kv.admit(1, 32));
    CHECK(kv.cache_prefix(0, 1, 32));
    kv.free(1);                                // cache holds 2, 1 free
    CHECK(kv.admit(2, 16));                    // takes the last free block
    CHECK_EQ(kv.free_blocks(), 0);
    CHECK(kv.append_token(2));                 // needs a block: reclaims the cache
    CHECK_EQ(kv.stats().cache_evictions, 1);
    CHECK_EQ(kv.table(2).blocks.size(), 2u);
    CHECK_EQ(kv.validate(), std::string());
}

TEST(cache_does_not_evict_the_entry_being_shared) {
    KVManager kv(3, 16);
    CHECK(kv.admit(1, 32));
    CHECK(kv.cache_prefix(0, 1, 32));
    kv.free(1);                                // cache: 2 blocks, free: 1
    CHECK(kv.admit_with_cached_prefix(2, 0, 32, 48));  // shares 2, takes the free one
    CHECK(kv.has_cached_prefix(0));
    CHECK_EQ(kv.stats().cache_evictions, 0);
    CHECK_EQ(kv.validate(), std::string());
    // not enough memory: nothing may change
    KVManager small(3, 16);
    CHECK(small.admit(1, 32));
    CHECK(small.cache_prefix(0, 1, 32));
    small.free(1);
    CHECK(!small.admit_with_cached_prefix(2, 0, 32, 80));  // needs 5 - 2 = 3 new blocks, only 1 free
    CHECK(!small.contains(2));
    CHECK(small.has_cached_prefix(0));
    CHECK_EQ(small.allocator().refcount(0), 1);
    CHECK_EQ(small.free_blocks(), 1);
    CHECK_EQ(small.validate(), std::string());
}

TEST(cache_lru_order) {
    KVManager kv(6, 16);
    CHECK(kv.admit(1, 16));
    CHECK(kv.admit(2, 16));
    CHECK(kv.cache_prefix(1, 1, 16));
    CHECK(kv.cache_prefix(2, 2, 16));
    kv.free(1);
    kv.free(2);
    CHECK(kv.admit_with_cached_prefix(3, 1, 16, 16));  // touches entry 1: entry 2 is now the older one
    kv.free(3);
    CHECK(kv.admit(4, 16 * 4));                // needs 4 blocks, 4 free: no eviction
    CHECK_EQ(kv.stats().cache_evictions, 0);
    CHECK(kv.admit(5, 16));                    // needs 1: must evict the LRU entry (2), keep entry 1
    CHECK(!kv.has_cached_prefix(2));
    CHECK(kv.has_cached_prefix(1));
    CHECK_EQ(kv.validate(), std::string());
}

// ------------------------------------------------------------------ priority policy

TEST(priority_request_preempts_lower_priority) {
    // 12 blocks of 4. r0 (low) holds 10-11 blocks; r1 (high) needs 2 and does not fit without preemption.
    std::vector<Request> reqs = {make(0, 0, 40, 8, 0), make(1, 2, 8, 4, 1)};
    SimConfig c;
    c.num_blocks = 12;
    c.block_size = 4;
    c.watermark_blocks = 0;
    c.validate_each_step = true;
    c.policy = Policy::FCFS;
    Metrics f = run_paged(reqs, c);
    c.policy = Policy::Priority;
    Metrics p = run_paged(reqs, c);
    CHECK_EQ(p.requests_completed, 2);
    CHECK_EQ(result_of(p, 1).first_admit, 2);              // admitted on arrival by displacing r0
    CHECK(result_of(f, 1).first_admit > 2);                // FCFS makes it wait for r0
    CHECK(result_of(p, 0).preemptions >= 1);
    CHECK_EQ(result_of(p, 1).preemptions, 0);
    CHECK(result_of(p, 1).finish < result_of(f, 1).finish);
    CHECK_EQ(p.final_free_slots, p.capacity_slots);
    c.priority_preemption = false;                         // ordering only: no displacement
    Metrics np = run_paged(reqs, c);
    CHECK(result_of(np, 1).first_admit > 2);
}

TEST(priority_policy_helps_the_high_class_under_load) {
    WorkloadConfig w;
    w.seed = 5;
    w.num_requests = 400;
    w.mean_interarrival = 1.0;
    w.prompt_min = 32; w.prompt_max = 256;
    w.output_min = 32; w.output_max = 256;
    w.high_priority_fraction = 0.25;
    w.max_seq_len = 1024;
    auto reqs = generate_workload(w);
    SimConfig c;
    c.num_blocks = 300;
    c.block_size = 16;
    c.validate_each_step = true;
    c.policy = Policy::FCFS;
    Metrics f = run_paged(reqs, c);
    c.policy = Policy::Priority;
    Metrics p = run_paged(reqs, c);
    CHECK_EQ(p.requests_completed, 400);
    CHECK(p.queue_mean_high < f.queue_mean_high);          // high class waits less than under FCFS
    CHECK(p.queue_mean_high < p.queue_mean_low);           // and less than the low class
    CHECK(p.lat_p95_high > 0 && p.lat_p95_low > 0);
}

// ------------------------------------------------------------------ swap vs recompute

namespace {
std::vector<Request> tight_workload() {
    WorkloadConfig w;
    w.seed = 9;
    w.num_requests = 60;
    w.mean_interarrival = 0.5;
    w.prompt_min = 20; w.prompt_max = 120;
    w.output_min = 20; w.output_max = 120;
    w.max_seq_len = 300;
    return generate_workload(w);
}
}  // namespace

TEST(swap_preemption_moves_blocks_instead_of_recomputing) {
    auto reqs = tight_workload();
    SimConfig c;
    c.num_blocks = 40;
    c.block_size = 16;
    c.validate_each_step = true;
    Metrics rec = run_paged(reqs, c);
    c.preempt_mode = PreemptMode::Swap;
    Metrics swp = run_paged(reqs, c);
    CHECK(rec.preemptions > 0);
    CHECK(rec.recomputed_tokens > 0);
    CHECK_EQ(rec.swap_out_blocks, 0);
    CHECK(swp.preemptions > 0);
    CHECK_EQ(swp.recomputed_tokens, 0);
    CHECK(swp.swap_out_blocks > 0);
    CHECK_EQ(swp.swap_out_blocks, swp.swap_in_blocks);     // everything swapped out came back
    CHECK_EQ(swp.requests_completed, 60);
    CHECK_EQ(swp.final_free_slots, swp.capacity_slots);
    // with an instantaneous cost model both modes make identical scheduling decisions
    CHECK_EQ(swp.preemptions, rec.preemptions);
    CHECK_EQ(swp.total_steps, rec.total_steps);
}

TEST(swap_falls_back_to_recompute_when_cpu_pool_is_full) {
    auto reqs = tight_workload();
    SimConfig c;
    c.num_blocks = 40;
    c.block_size = 16;
    c.preempt_mode = PreemptMode::Swap;
    c.swap_capacity_blocks = 0;
    c.validate_each_step = true;
    Metrics m = run_paged(reqs, c);
    CHECK_EQ(m.swap_out_blocks, 0);
    CHECK(m.recomputed_tokens > 0);
    c.swap_capacity_blocks = 8;  // small pool: some swaps, some fallbacks
    Metrics s = run_paged(reqs, c);
    CHECK(s.swap_out_blocks > 0);
    CHECK(s.recomputed_tokens > 0);
    CHECK_EQ(s.swap_out_blocks, s.swap_in_blocks);
}

TEST(cost_model_stalls_slow_things_down) {
    auto reqs = tight_workload();
    SimConfig c;
    c.num_blocks = 40;
    c.block_size = 16;
    c.validate_each_step = true;
    Metrics free_cost = run_paged(reqs, c);
    c.prefill_tokens_per_step = 64;  // recompute is slow
    Metrics rec = run_paged(reqs, c);
    c.preempt_mode = PreemptMode::Swap;
    c.swap_blocks_per_step = 32;     // PCIe is fast relative to compute
    Metrics swp = run_paged(reqs, c);
    CHECK(rec.total_steps >= free_cost.total_steps);
    CHECK(swp.total_steps >= free_cost.total_steps);
    CHECK_EQ(rec.requests_completed, 60);
    CHECK_EQ(swp.requests_completed, 60);
}

// ------------------------------------------------------------------ persistent prefix cache (simulator level)

TEST(sim_prefix_cache_hits_when_requests_do_not_overlap) {
    WorkloadConfig w;
    w.seed = 3;
    w.num_requests = 12;
    w.mean_interarrival = 100.0;  // each request finishes before the next arrives
    w.prompt_min = 16; w.prompt_max = 40;
    w.output_min = 8; w.output_max = 20;
    w.shared_prefix_fraction = 1.0;
    w.prefix_len = 128;
    w.max_seq_len = 400;
    auto reqs = generate_workload(w);
    SimConfig c;
    c.num_blocks = 100;
    c.block_size = 16;
    c.validate_each_step = true;
    c.prefix_cache = false;
    Metrics off = run_paged(reqs, c);
    c.prefix_cache = true;
    Metrics on = run_paged(reqs, c);
    CHECK_EQ(off.cache_hits, 0);
    CHECK_EQ(off.prefill_tokens_saved, 0);       // live-parent sharing finds no live parent
    CHECK(on.cache_hits >= 8);                   // later requests of the same group hit the cache
    CHECK(on.prefill_tokens_saved >= 8 * 128);
    CHECK(on.prefill_tokens_computed < off.prefill_tokens_computed);
    CHECK_EQ(on.final_free_slots, on.capacity_slots);  // the cache is not counted as a leak
}

TEST(sim_prefix_cache_yields_memory_to_running_requests) {
    WorkloadConfig w;
    w.seed = 4;
    w.num_requests = 80;
    w.mean_interarrival = 0.7;
    w.prompt_min = 16; w.prompt_max = 100;
    w.output_min = 16; w.output_max = 100;
    w.shared_prefix_fraction = 0.8;
    w.prefix_len = 64;
    w.num_prefix_groups = 6;
    w.max_seq_len = 300;
    auto reqs = generate_workload(w);
    SimConfig c;
    c.num_blocks = 60;  // tight: the cache must give blocks back
    c.block_size = 16;
    c.validate_each_step = true;
    c.prefix_cache = true;
    Metrics m = run_paged(reqs, c);
    CHECK_EQ(m.requests_completed, 80);
    CHECK(m.cache_evictions > 0);
    CHECK_EQ(m.final_free_slots, m.capacity_slots);
}

TEST(sim_all_new_options_together_are_deterministic) {
    WorkloadConfig w;
    w.seed = 17;
    w.num_requests = 60;
    w.mean_interarrival = 0.5;
    w.prompt_min = 16; w.prompt_max = 120;
    w.output_min = 8; w.output_max = 80;
    w.shared_prefix_fraction = 0.5;
    w.prefix_len = 48;
    w.num_prefix_groups = 3;
    w.high_priority_fraction = 0.3;
    w.samples_per_request = 2;
    w.max_seq_len = 300;
    auto reqs = generate_workload(w);
    SimConfig c;
    c.num_blocks = 50;
    c.block_size = 16;
    c.policy = Policy::Priority;
    c.preempt_mode = PreemptMode::Swap;
    c.swap_capacity_blocks = 30;
    c.prefill_tokens_per_step = 100;
    c.swap_blocks_per_step = 16;
    c.prefix_cache = true;
    c.validate_each_step = true;
    Metrics a = run_paged(reqs, c);
    Metrics b = run_paged(reqs, c);
    CHECK_EQ(a.summary(), b.summary());
    CHECK_EQ(a.requests_completed, static_cast<int>(reqs.size()));
    c.validate_each_step = false;
    CHECK_EQ(run_paged(reqs, c).summary(), a.summary());
}
