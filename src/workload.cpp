#include "kvpage/workload.h"

#include <algorithm>
#include <cmath>

namespace kvp {
namespace {

// splitmix64: tiny, fast, identical on every platform.
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    double unit() { return static_cast<double>(next() >> 11) * (1.0 / 9007199254740992.0); }  // [0,1)
    int range(int lo, int hi) {  // inclusive
        return lo + static_cast<int>(next() % static_cast<uint64_t>(hi - lo + 1));
    }
};

int sample(Rng& rng, Dist d, int lo, int hi) {
    if (d == Dist::Uniform) return rng.range(lo, hi);
    double x = lo / std::pow(1.0 - rng.unit(), 1.0 / 1.5);  // Pareto, alpha = 1.5
    return static_cast<int>(std::min<double>(x, hi));
}

}  // namespace

std::vector<Request> generate_workload(const WorkloadConfig& c) {
    Rng rng(c.seed);
    std::vector<Request> out;
    out.reserve(c.num_requests);
    double t = 0;
    RequestId next_id = 0;
    for (int i = 0; i < c.num_requests; ++i) {
        Request r;
        r.id = next_id++;
        r.arrival = static_cast<int>(t);
        t += -c.mean_interarrival * std::log(1.0 - rng.unit());

        int suffix = sample(rng, c.prompt_dist, c.prompt_min, c.prompt_max);
        r.output_len = sample(rng, c.output_dist, c.output_min, c.output_max);
        bool shared = c.shared_prefix_fraction > 0 && rng.unit() < c.shared_prefix_fraction;
        if (c.high_priority_fraction > 0 && rng.unit() < c.high_priority_fraction) r.priority = 1;
        if (shared) {
            r.prefix_group = rng.range(0, std::max(c.num_prefix_groups, 1) - 1);
            r.prefix_len = c.prefix_len;
        }
        r.prompt_len = r.prefix_len + suffix;
        if (r.prompt_len + r.output_len > c.max_seq_len) {
            r.output_len = std::max(1, c.max_seq_len - r.prompt_len);
            r.prompt_len = std::min(r.prompt_len, c.max_seq_len - r.output_len);
            r.prefix_len = std::min(r.prefix_len, r.prompt_len);
        }
        out.push_back(r);
        for (int s = 1; s < c.samples_per_request; ++s) {  // extra samples of the same prompt
            Request ch = r;
            ch.id = next_id++;
            ch.fork_of = r.id;
            ch.output_len = std::max(1, std::min(sample(rng, c.output_dist, c.output_min, c.output_max),
                                                 c.max_seq_len - r.prompt_len));
            out.push_back(ch);
        }
    }
    return out;  // arrivals are non-decreasing by construction
}

}  // namespace kvp
