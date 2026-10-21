// CPU micro-benchmarks of the C++ code itself (not the simulated metrics).
// Usage: kvpage_cpu_bench [--reps N] [--csv path]   Reports the median of N repetitions.
// Build in Release; in Debug the per-op invariant checks dominate and the numbers are meaningless.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "kvpage/kv_manager.h"
#include "kvpage/simulator.h"
#include "kvpage/workload.h"

using namespace kvp;
using Clock = std::chrono::steady_clock;

namespace {

volatile int64_t g_sink;  // keeps the optimizer from deleting the work

struct Result {
    std::string name;
    double ops;       // operations per repetition
    double median_s;  // median seconds per repetition
    double min_s, max_s;
};

// `fn` performs one repetition and returns the number of operations it did.
Result measure(const std::string& name, int reps, const std::function<int64_t()>& fn) {
    std::vector<double> t;
    int64_t ops = 0;
    fn();  // warm-up
    for (int i = 0; i < reps; ++i) {
        auto a = Clock::now();
        ops = fn();
        auto b = Clock::now();
        t.push_back(std::chrono::duration<double>(b - a).count());
    }
    std::sort(t.begin(), t.end());
    return {name, static_cast<double>(ops), t[t.size() / 2], t.front(), t.back()};
}

}  // namespace

int main(int argc, char** argv) {
    int reps = 9;
    std::string csv;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--reps") && i + 1 < argc) reps = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) csv = argv[++i];
    }
    std::vector<Result> res;

    // 1. Raw allocator: allocate everything, release everything.
    res.push_back(measure("allocator: allocate+release, 1M blocks x 3 rounds", reps, [] {
        BlockAllocator a(1 << 20);
        std::vector<BlockId> ids(1 << 20);
        int64_t ops = 0;
        for (int r = 0; r < 3; ++r) {
            for (auto& b : ids) b = a.allocate();
            for (auto b : ids) a.release(b);
            ops += 2 << 20;
        }
        g_sink = a.free_blocks();
        return ops;
    }));

    // 2. KVManager decode path: 256 requests each appending 4096 tokens (B=16).
    res.push_back(measure("kv_manager: append_token, 256 reqs x 4096 tokens, B=16", reps, [] {
        KVManager kv(256 * 256 + 16, 16);
        kv.set_check_each_op(false);
        for (int r = 0; r < 256; ++r) kv.admit(r, 0);
        int64_t ops = 0;
        for (int t = 0; t < 4096; ++t)
            for (int r = 0; r < 256; ++r) ops += kv.append_token(r);
        for (int r = 0; r < 256; ++r) kv.free(r);
        g_sink = ops;
        return ops;
    }));

    // 3. Admit / free churn with a shared prefix.
    res.push_back(measure("kv_manager: admit_with_prefix+free, 200k requests", reps, [] {
        KVManager kv(4096, 16);
        kv.set_check_each_op(false);
        kv.admit(-2, 512);  // parent holding a 512-token (32-block) prefix
        int64_t ops = 0;
        for (int i = 0; i < 200000; ++i) {
            ops += kv.admit_with_prefix(i, -2, 512, 700);
            kv.free(i);
            ops += 1;
        }
        g_sink = ops;
        return ops;
    }));

    // 4. KVManager::stats() (called once per simulated step).
    res.push_back(measure("kv_manager: stats(), 200 live requests x 20k calls", reps, [] {
        KVManager kv(200 * 40, 16);
        kv.set_check_each_op(false);
        for (int r = 0; r < 200; ++r) kv.admit(r, 500);
        int64_t acc = 0;
        for (int i = 0; i < 20000; ++i) acc += kv.stats().used_tokens;
        g_sink = acc;
        return int64_t{20000};
    }));

    // 5. Whole simulator, scenario A of the benchmark (1500 requests), per step and per token.
    WorkloadConfig w;
    w.prompt_min = 32; w.prompt_max = 512;
    w.output_min = 32; w.output_max = 512;
    w.mean_interarrival = 2.0;
    w.max_seq_len = 2048;
    w.num_requests = 1500;
    w.seed = 1;
    auto reqs = generate_workload(w);
    SimConfig sc;
    sc.num_blocks = 4096;
    int64_t steps = 0, tokens = 0;
    res.push_back(measure("simulator: scenario A paged-FCFS (ops = simulated steps)", reps, [&] {
        Metrics m = run_paged(reqs, sc);
        steps = m.total_steps;
        tokens = m.tokens_generated;
        return m.total_steps;
    }));
    Result tok = res.back();
    tok.name = "simulator: scenario A paged-FCFS (ops = generated tokens)";
    tok.ops = static_cast<double>(tokens);
    res.push_back(tok);
    sc.validate_each_step = true;
    sc.num_blocks = 4096;
    res.push_back(measure("simulator: same, with validate_each_step (ops = steps)", std::max(3, reps / 3), [&] {
        Metrics m = run_paged(reqs, sc);
        return m.total_steps;
    }));
    ContigConfig cc;
    cc.total_slots = 4096 * 16;
    cc.reserve_len = 2048;
    res.push_back(measure("simulator: scenario A contig-maxlen (ops = simulated steps)", reps, [&] {
        return run_contiguous(reqs, cc).total_steps;
    }));
    res.push_back(measure("workload: generate 100k requests", reps, [] {
        WorkloadConfig g;
        g.num_requests = 100000;
        g.shared_prefix_fraction = 0.5;
        g.max_seq_len = 2048;
        g.prompt_dist = Dist::LongTail;
        g_sink = static_cast<int64_t>(generate_workload(g).size());
        return int64_t{100000};
    }));

    std::printf("reps=%d (median; min/max in CSV), simulator A: %lld steps, %lld tokens\n\n", reps,
                (long long)steps, (long long)tokens);
    std::printf("| Benchmark | Ops / rep | Median ms / rep | Mops/s | ns/op |\n|---|---|---|---|---|\n");
    for (const auto& r : res)
        std::printf("| %s | %.0f | %.2f | %.2f | %.1f |\n", r.name.c_str(), r.ops, r.median_s * 1e3,
                    r.ops / r.median_s / 1e6, r.median_s / r.ops * 1e9);
    if (!csv.empty()) {
        std::ofstream f(csv);
        f << "benchmark,ops,median_s,min_s,max_s\n";
        for (const auto& r : res)
            f << '"' << r.name << "\"," << r.ops << ',' << r.median_s << ',' << r.min_s << ',' << r.max_s << '\n';
    }
    return 0;
}
