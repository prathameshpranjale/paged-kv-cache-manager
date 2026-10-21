#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "kvpage/types.h"

namespace kvp {

struct RequestResult {
    RequestId id = 0;
    int arrival = 0;
    int first_admit = -1;
    int finish = -1;  // step at whose end the last token exists
    int preemptions = 0;
    int priority = 0;
};

struct Metrics {
    std::string label;
    int64_t capacity_slots = 0;
    int64_t final_free_slots = 0;  // must equal capacity_slots (no leak)
    int requests_completed = 0;
    int64_t total_steps = 0;
    int64_t tokens_generated = 0;
    double tokens_per_step = 0;  // over all steps
    // Means below are over "active" steps (>= 1 running request).
    double mean_utilization = 0;  // sum(stored slots) / sum(allocated slots)
    double mean_waste_slots = 0;  // allocated - stored, per step
    double mean_concurrent = 0;
    int peak_concurrent = 0;
    double lat_p50 = 0, lat_p95 = 0;       // arrival -> completion, in steps
    double queue_mean = 0, queue_p95 = 0;  // arrival -> first admission
    int64_t preemptions = 0;
    int64_t recomputed_tokens = 0;
    int64_t cow_copies = 0;
    int64_t swap_out_blocks = 0, swap_in_blocks = 0;  // blocks moved to / from the CPU pool
    int64_t cache_hits = 0, cache_evictions = 0;      // persistent prefix cache
    int64_t prefill_tokens_computed = 0, prefill_tokens_saved = 0;  // by admissions (saved: shared prefix blocks)
    double lat_p95_high = 0, lat_p95_low = 0;         // by priority class (0 if the class is empty)
    double queue_mean_high = 0, queue_mean_low = 0;
    double blocks_saved_mean = 0;
    int64_t blocks_saved_peak = 0;
    std::vector<RequestResult> results;

    std::string summary() const;  // full-precision text, used for determinism checks
};

// Nearest-rank percentile, p in [0,100]. Empty input gives 0.
double percentile(std::vector<double> v, double p);

struct StepSample {
    int running = 0;
    int64_t stored = 0;        // token slots holding data
    int64_t allocated = 0;     // token slots reserved or allocated
    int64_t saved_blocks = 0;  // blocks avoided by prefix sharing
    int generated = 0;         // tokens produced this step
};

class MetricsCollector {
public:
    void on_step(const StepSample& s);
    void on_preempt(int tokens_lost) {
        ++preemptions_;
        recomputed_ += tokens_lost;
    }
    void on_complete(const RequestResult& r) { results_.push_back(r); }
    void on_swap(int64_t out_blocks, int64_t in_blocks) {
        swap_out_ += out_blocks;
        swap_in_ += in_blocks;
    }
    void on_cache_hit() { ++cache_hits_; }
    void on_prefill(int64_t computed, int64_t saved) {
        prefill_computed_ += computed;
        prefill_saved_ += saved;
    }
    Metrics finish(const std::string& label, int64_t total_steps, int64_t capacity_slots,
                   int64_t final_free_slots, int64_t cow_copies, int64_t cache_evictions = 0);

private:
    int64_t active_ = 0, generated_ = 0;
    double stored_sum_ = 0, alloc_sum_ = 0, running_sum_ = 0, saved_sum_ = 0;
    int peak_ = 0;
    int64_t saved_peak_ = 0;
    int64_t preemptions_ = 0, recomputed_ = 0, swap_out_ = 0, swap_in_ = 0, cache_hits_ = 0, prefill_computed_ = 0, prefill_saved_ = 0;
    std::vector<RequestResult> results_;
};

}  // namespace kvp
