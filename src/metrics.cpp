#include "kvpage/metrics.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

namespace kvp {

double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    size_t rank = static_cast<size_t>(std::ceil(p / 100.0 * static_cast<double>(v.size())));
    rank = std::min(std::max<size_t>(rank, 1), v.size());
    return v[rank - 1];
}

void MetricsCollector::on_step(const StepSample& s) {
    generated_ += s.generated;
    if (s.running <= 0) return;
    ++active_;
    stored_sum_ += static_cast<double>(s.stored);
    alloc_sum_ += static_cast<double>(s.allocated);
    running_sum_ += s.running;
    saved_sum_ += static_cast<double>(s.saved_blocks);
    peak_ = std::max(peak_, s.running);
    saved_peak_ = std::max(saved_peak_, s.saved_blocks);
}

Metrics MetricsCollector::finish(const std::string& label, int64_t total_steps, int64_t capacity_slots,
                                 int64_t final_free_slots, int64_t cow_copies,
                                 int64_t cache_evictions) {
    Metrics m;
    m.label = label;
    m.capacity_slots = capacity_slots;
    m.final_free_slots = final_free_slots;
    m.requests_completed = static_cast<int>(results_.size());
    m.total_steps = total_steps;
    m.tokens_generated = generated_;
    m.tokens_per_step = total_steps ? static_cast<double>(generated_) / static_cast<double>(total_steps) : 0;
    if (active_ > 0) {
        double a = static_cast<double>(active_);
        m.mean_utilization = alloc_sum_ > 0 ? stored_sum_ / alloc_sum_ : 0;
        m.mean_waste_slots = (alloc_sum_ - stored_sum_) / a;
        m.mean_concurrent = running_sum_ / a;
        m.blocks_saved_mean = saved_sum_ / a;
    }
    m.peak_concurrent = peak_;
    m.blocks_saved_peak = saved_peak_;
    m.preemptions = preemptions_;
    m.recomputed_tokens = recomputed_;
    m.cow_copies = cow_copies;
    m.swap_out_blocks = swap_out_;
    m.swap_in_blocks = swap_in_;
    m.cache_hits = cache_hits_;
    m.prefill_tokens_computed = prefill_computed_;
    m.prefill_tokens_saved = prefill_saved_;
    m.cache_evictions = cache_evictions;

    std::sort(results_.begin(), results_.end(),
              [](const RequestResult& a, const RequestResult& b) { return a.id < b.id; });
    std::vector<double> lat, queue, lat_hi, lat_lo;
    double q_hi = 0, q_lo = 0;
    for (const auto& r : results_) {
        lat.push_back(r.finish - r.arrival);
        queue.push_back(r.first_admit - r.arrival);
        (r.priority > 0 ? lat_hi : lat_lo).push_back(r.finish - r.arrival);
        (r.priority > 0 ? q_hi : q_lo) += r.first_admit - r.arrival;
    }
    m.lat_p95_high = percentile(lat_hi, 95);
    m.lat_p95_low = percentile(lat_lo, 95);
    if (!lat_hi.empty()) m.queue_mean_high = q_hi / static_cast<double>(lat_hi.size());
    if (!lat_lo.empty()) m.queue_mean_low = q_lo / static_cast<double>(lat_lo.size());
    m.lat_p50 = percentile(lat, 50);
    m.lat_p95 = percentile(lat, 95);
    m.queue_p95 = percentile(queue, 95);
    if (!queue.empty()) {
        double s = 0;
        for (double q : queue) s += q;
        m.queue_mean = s / static_cast<double>(queue.size());
    }
    m.results = results_;
    return m;
}

std::string Metrics::summary() const {
    char buf[1024];
    std::snprintf(buf, sizeof buf,
                  "%s cap=%lld free=%lld done=%d steps=%lld gen=%lld tps=%.9g util=%.9g waste=%.9g "
                  "conc=%.9g peak=%d p50=%.9g p95=%.9g qmean=%.9g q95=%.9g pre=%lld rec=%lld cow=%lld "
                  "saved=%.9g/%lld swap=%lld/%lld cache=%lld/%lld hi=%.9g/%.9g lo=%.9g/%.9g prefill=%lld/%lld",
                  label.c_str(), (long long)capacity_slots, (long long)final_free_slots, requests_completed,
                  (long long)total_steps, (long long)tokens_generated, tokens_per_step, mean_utilization,
                  mean_waste_slots, mean_concurrent, peak_concurrent, lat_p50, lat_p95, queue_mean, queue_p95,
                  (long long)preemptions, (long long)recomputed_tokens, (long long)cow_copies,
                  blocks_saved_mean, (long long)blocks_saved_peak, (long long)swap_out_blocks, (long long)swap_in_blocks,
                  (long long)cache_hits, (long long)cache_evictions, lat_p95_high, queue_mean_high, lat_p95_low,
                  queue_mean_low, (long long)prefill_tokens_computed, (long long)prefill_tokens_saved);
    std::string s = buf;
    for (const auto& r : results)
        s += "\n" + std::to_string(r.id) + ":" + std::to_string(r.first_admit) + ":" +
             std::to_string(r.finish) + ":" + std::to_string(r.preemptions);
    return s;
}

}  // namespace kvp
