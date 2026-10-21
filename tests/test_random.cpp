// Randomized / property tests. Every sequence is reproducible from its seed; on a
// failure the sequence is shortened to the smallest failing prefix before reporting.
#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "kvpage/kv_manager.h"
#include "kvpage/simulator.h"
#include "kvpage/workload.h"
#include "testing.h"

using namespace kvp;

namespace {

struct Lcg {  // small deterministic generator, independent of the library's
    uint64_t s;
    explicit Lcg(uint64_t seed) : s(seed * 2862933555777941757ull + 3037000493ull) {}
    uint32_t next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<uint32_t>(s >> 33);
    }
    int range(int lo, int hi) { return lo + static_cast<int>(next() % static_cast<uint32_t>(hi - lo + 1)); }
};

// Runs `nops` random operations against a KVManager and a trivial shadow model.
// Returns "" on success, otherwise a description of the first mismatch.
std::string run_kv_sequence(uint64_t seed, int nops) {
    Lcg rng(seed);
    const int B = rng.range(1, 8);
    const int total = rng.range(4, 40);
    KVManager kv(total, B);
    kv.set_check_each_op(false);  // we validate explicitly after each op
    std::map<RequestId, int> model;  // id -> tokens
    RequestId next_id = 0;

    auto pick_live = [&]() -> RequestId {
        if (model.empty()) return kNoRequest;
        auto it = model.begin();
        std::advance(it, rng.range(0, static_cast<int>(model.size()) - 1));
        return it->first;
    };

    for (int op = 0; op < nops; ++op) {
        int c = rng.range(0, 99);
        std::string what;
        if (c < 20) {
            int prompt = rng.range(0, 5 * B);
            bool expect = kv.available_blocks() >= ceil_div(prompt, B);
            RequestId id = next_id++;
            bool got = kv.admit(id, prompt);
            what = "admit";
            if (got != expect) return "op " + std::to_string(op) + " admit result mismatch";
            if (got) model[id] = prompt;
        } else if (c < 35) {
            RequestId parent = pick_live();
            if (parent == kNoRequest) continue;
            int shared = rng.range(0, model[parent] + B), prompt = rng.range(0, 6 * B);
            int k = std::min({shared, model[parent], prompt}) / B;
            bool expect = kv.available_blocks() >= ceil_div(prompt, B) - k;
            RequestId id = next_id++;
            bool got = kv.admit_with_prefix(id, parent, shared, prompt);
            what = "admit_with_prefix";
            if (got != expect) return "op " + std::to_string(op) + " admit_with_prefix result mismatch";
            if (got) model[id] = prompt;
        } else if (c < 40) {
            RequestId parent = pick_live();
            if (parent == kNoRequest) continue;
            RequestId id = next_id++;
            kv.fork(id, parent);
            model[id] = model[parent];
            what = "fork";
        } else if (c < 46) {
            RequestId owner = pick_live();
            if (owner == kNoRequest) continue;
            int key = rng.range(0, 3);
            int tokens = rng.range(0, 4 * B);
            bool expect = !kv.has_cached_prefix(key) && std::min(tokens, model[owner]) / B > 0;
            bool got = kv.cache_prefix(key, owner, tokens);
            what = "cache_prefix";
            if (got != expect) return "op " + std::to_string(op) + " cache_prefix result mismatch";
        } else if (c < 54) {
            int key = rng.range(0, 3);
            int shared = rng.range(0, 4 * B), prompt = rng.range(0, 6 * B);
            int cached = kv.cached_prefix_tokens(key);
            int k = std::max(0, std::min({shared, cached, prompt})) / B;
            int need = ceil_div(prompt, B) - k;
            bool expect = kv.free_blocks() + kv.evictable_blocks(key, k) >= need;
            RequestId id = next_id++;
            bool got = kv.admit_with_cached_prefix(id, key, shared, prompt);
            what = "admit_with_cached_prefix";
            if (got != expect) return "op " + std::to_string(op) + " admit_with_cached_prefix result mismatch";
            if (got) model[id] = prompt;
        } else if (c < 84) {
            RequestId id = pick_live();
            if (id == kNoRequest) continue;
            const BlockTable& t = kv.table(id);
            bool need_block = model[id] % B == 0 || kv.allocator().refcount(t.blocks.back()) > 1;
            bool expect = !need_block || kv.available_blocks() >= 1;
            bool got = kv.append_token(id);
            what = "append";
            if (got != expect) return "op " + std::to_string(op) + " append result mismatch";
            if (got) ++model[id];
        } else {
            RequestId id = pick_live();
            if (id == kNoRequest) continue;
            if (rng.range(0, 29) == 0) {
                kv.clear_cache();
                what = "clear_cache";
            } else {
                kv.free(id);
                model.erase(id);
                what = "free";
            }
        }
        std::string err = kv.validate();
        if (!err.empty()) return "op " + std::to_string(op) + " (" + what + "): " + err;
        for (const auto& [id, tokens] : model)
            if (kv.table(id).tokens != tokens) return "op " + std::to_string(op) + ": token count drifted";
        if (kv.stats().requests != static_cast<int>(model.size())) return "request count drifted";
    }
    for (const auto& [id, tokens] : model) kv.free(id);
    kv.clear_cache();
    if (kv.free_blocks() != total) return "leak after freeing everything";
    return kv.validate();
}

}  // namespace

TEST(random_kv_manager_sequences) {
    const int kSeeds = 3000, kOps = 300;
    int bad = 0;
    for (uint64_t seed = 1; seed <= kSeeds && bad < 3; ++seed) {
        std::string err = run_kv_sequence(seed, kOps);
        if (err.empty()) continue;
        ++bad;
        int n = 1;  // shrink: smallest prefix of the same seed that still fails
        while (run_kv_sequence(seed, n).empty()) ++n;
        std::cerr << "  FAILING SEED " << seed << " (shrunk to " << n << " ops): "
                  << run_kv_sequence(seed, n) << "\n";
    }
    CHECK_EQ(bad, 0);
}

TEST(random_scheduler_runs_keep_invariants) {
    int preempting_runs = 0;
    for (uint64_t seed = 1; seed <= 150; ++seed) {
        Lcg rng(seed);
        WorkloadConfig w;
        w.seed = seed;
        w.num_requests = rng.range(5, 40);
        w.mean_interarrival = 0.3 * rng.range(1, 10);
        w.prompt_dist = rng.range(0, 1) ? Dist::LongTail : Dist::Uniform;
        w.output_dist = rng.range(0, 1) ? Dist::LongTail : Dist::Uniform;
        w.prompt_min = 2;
        w.prompt_max = 60;
        w.output_min = 1;
        w.output_max = 40;
        w.shared_prefix_fraction = rng.range(0, 1) ? 0.6 : 0.0;
        w.prefix_len = rng.range(4, 40);
        w.num_prefix_groups = rng.range(1, 3);
        w.max_seq_len = 120;
        w.samples_per_request = rng.range(1, 3);
        w.high_priority_fraction = rng.range(0, 1) ? 0.3 : 0.0;
        auto reqs = generate_workload(w);

        SimConfig c;
        c.block_size = 1 << rng.range(1, 4);  // 2..16
        c.num_blocks = ceil_div(w.max_seq_len, c.block_size) + rng.range(0, 30);
        switch (rng.range(0, 2)) {
            case 0: c.policy = Policy::FCFS; break;
            case 1: c.policy = Policy::ShortestPromptFirst; break;
            default: c.policy = Policy::Priority; break;
        }
        c.priority_preemption = rng.range(0, 3) != 0;
        c.preempt_mode = rng.range(0, 1) ? PreemptMode::Swap : PreemptMode::Recompute;
        c.swap_capacity_blocks = rng.range(0, 1) ? (1 << 20) : rng.range(0, 20);
        c.prefill_tokens_per_step = rng.range(0, 1) ? 0 : rng.range(8, 64);
        c.swap_blocks_per_step = rng.range(0, 1) ? 0 : rng.range(1, 8);
        c.prefix_cache = rng.range(0, 1) == 1;
        c.watermark_blocks = rng.range(0, 3);
        c.max_running = rng.range(0, 1) ? 0 : rng.range(1, 6);
        c.prefix_sharing = rng.range(0, 3) != 0;
        c.fork_sharing = rng.range(0, 3) != 0;
        c.validate_each_step = true;
        Metrics m;
        try {
            m = run_paged(reqs, c);
        } catch (const std::exception& e) {
            CHECK(false);
            std::cerr << "  FAILING SEED " << seed << ": " << e.what() << "\n";
            continue;
        }
        int64_t want = 0;
        for (const auto& r : reqs) want += r.output_len;
        if (m.requests_completed != static_cast<int>(reqs.size()) || m.tokens_generated != want ||
            m.final_free_slots != m.capacity_slots) {
            CHECK(false);
            std::cerr << "  FAILING SEED " << seed << ": done=" << m.requests_completed << "/" << reqs.size()
                      << " gen=" << m.tokens_generated << "/" << want << " free=" << m.final_free_slots
                      << "/" << m.capacity_slots << "\n";
        }
        if (m.preemptions > 0) ++preempting_runs;
    }
    CHECK(preempting_runs > 0);  // the generator must actually reach the preemption path
    std::cout << "    (" << preempting_runs << " of 150 runs preempted)\n";
}
