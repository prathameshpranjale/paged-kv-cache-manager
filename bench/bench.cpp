// Paged KV cache vs contiguous reservation; policies; prefix cache; priority classes; swap vs recompute.
// Usage: kvpage_bench [--seeds N] [--seed0 S] [--requests N] [--csv summary.csv] [--per-seed-csv raw.csv]
// Each scenario is run for seeds S..S+N-1 (a different workload per seed); tables show mean +- sample std dev.
// The simulator measures memory management and scheduling, NOT GPU speed: one "step" is one decode iteration
// of the whole batch. Cost-model rates (scenarios G and I) are ASSUMPTIONS, not measurements.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "kvpage/simulator.h"
#include "kvpage/workload.h"

using namespace kvp;

namespace {

using Runner = std::function<Metrics(const std::vector<Request>&)>;

struct Variant {
    std::string name;
    Runner run;
};

struct Scenario {
    std::string name;
    std::string note;
    WorkloadConfig w;
    std::vector<Variant> variants;
    bool priority_columns = false;  // show per-class latency / queueing
    bool cost_columns = false;      // show swap / prefill columns
};

struct Stat {
    std::vector<double> v;
    void add(double x) { v.push_back(x); }
    double mean() const {
        double s = 0;
        for (double x : v) s += x;
        return v.empty() ? 0 : s / static_cast<double>(v.size());
    }
    double sd() const {
        if (v.size() < 2) return 0;
        double m = mean(), s = 0;
        for (double x : v) s += (x - m) * (x - m);
        return std::sqrt(s / static_cast<double>(v.size() - 1));
    }
    std::string pm(const char* fmt = "%.1f") const {
        char a[64], b[64], out[160];
        std::snprintf(a, sizeof a, fmt, mean());
        std::snprintf(b, sizeof b, fmt, sd());
        std::snprintf(out, sizeof out, "%s ± %s", a, b);
        return out;
    }
};

struct Agg {
    std::string label;
    Stat util, conc, peak, tps, p50, p95, queue, preempt, recomputed, cow, saved_mean, saved_peak;
    Stat swap_out, cache_hits, cache_evict, prefill_computed, prefill_saved;
    Stat p95_hi, p95_lo, q_hi, q_lo;
};

SimConfig paged(int blocks, Policy p = Policy::FCFS) {
    SimConfig c;
    c.num_blocks = blocks;
    c.block_size = 16;
    c.policy = p;
    return c;
}

Runner paged_runner(SimConfig c, const std::string& label = "") {
    return [c, label](const std::vector<Request>& r) {
        Metrics m = run_paged(r, c);
        if (!label.empty()) m.label = label;
        return m;
    };
}

Runner contig_runner(int blocks, int reserve_len) {
    return [blocks, reserve_len](const std::vector<Request>& r) {
        ContigConfig cc;
        cc.total_slots = static_cast<int64_t>(blocks) * 16;
        cc.reserve_len = reserve_len;
        return run_contiguous(r, cc);
    };
}

}  // namespace

int main(int argc, char** argv) {
    int seeds = 5, requests = 1500;
    uint64_t seed0 = 1;
    std::string csv_path, raw_path;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--seeds") && i + 1 < argc) seeds = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--seed0") && i + 1 < argc) seed0 = std::strtoull(argv[++i], nullptr, 10);
        else if (!std::strcmp(argv[i], "--requests") && i + 1 < argc) requests = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--csv") && i + 1 < argc) csv_path = argv[++i];
        else if (!std::strcmp(argv[i], "--per-seed-csv") && i + 1 < argc) raw_path = argv[++i];
        else {
            std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
            return 2;
        }
    }
    const int kBlocks = 4096;

    std::vector<Scenario> sc;
    auto contig_pair = [&](Scenario& s, int blocks) {
        s.variants.push_back({"contig-maxlen", contig_runner(blocks, s.w.max_seq_len)});
        s.variants.push_back({"contig-oracle", contig_runner(blocks, 0)});
    };
    {
        Scenario s;
        s.name = "A uniform lengths, memory-bound";
        s.w.prompt_min = 32; s.w.prompt_max = 512; s.w.output_min = 32; s.w.output_max = 512;
        s.w.mean_interarrival = 2.0; s.w.max_seq_len = 2048;
        s.variants.push_back({"paged-FCFS", paged_runner(paged(kBlocks))});
        s.variants.push_back({"paged-SPF", paged_runner(paged(kBlocks, Policy::ShortestPromptFirst))});
        contig_pair(s, kBlocks);
        sc.push_back(s);
    }
    {
        Scenario s;
        s.name = "B long-tailed lengths, memory not binding";
        s.w.prompt_dist = Dist::LongTail; s.w.prompt_min = 32; s.w.prompt_max = 1536;
        s.w.output_dist = Dist::LongTail; s.w.output_min = 16; s.w.output_max = 512;
        s.w.mean_interarrival = 0.25; s.w.max_seq_len = 2048;
        s.variants.push_back({"paged-FCFS", paged_runner(paged(kBlocks))});
        contig_pair(s, kBlocks);
        sc.push_back(s);
    }
    {
        Scenario s;
        s.name = "C 70% share a 512-token system prompt";
        s.w.prompt_min = 32; s.w.prompt_max = 256; s.w.output_min = 32; s.w.output_max = 256;
        s.w.shared_prefix_fraction = 0.7; s.w.prefix_len = 512;
        s.w.mean_interarrival = 1.2; s.w.max_seq_len = 2048;
        s.variants.push_back({"paged-FCFS", paged_runner(paged(kBlocks))});
        SimConfig ns = paged(kBlocks);
        ns.prefix_sharing = false;
        s.variants.push_back({"paged-FCFS-noshare", paged_runner(ns)});
        contig_pair(s, kBlocks);
        sc.push_back(s);
    }
    {
        Scenario s;
        s.name = "D tight memory, overload";
        s.note = "arrivals far above service rate: queues grow all run, latencies measure the backlog";
        s.w.prompt_min = 32; s.w.prompt_max = 512; s.w.output_min = 32; s.w.output_max = 512;
        s.w.mean_interarrival = 1.0; s.w.max_seq_len = 1024;
        s.variants.push_back({"paged-FCFS", paged_runner(paged(768))});
        s.variants.push_back({"paged-SPF", paged_runner(paged(768, Policy::ShortestPromptFirst))});
        contig_pair(s, 768);
        sc.push_back(s);
    }
    {
        Scenario s;
        s.name = "E parallel sampling, 4 samples per prompt";
        s.w.prompt_min = 32; s.w.prompt_max = 512; s.w.output_min = 32; s.w.output_max = 256;
        s.w.samples_per_request = 4; s.w.mean_interarrival = 4.0; s.w.max_seq_len = 2048;
        s.variants.push_back({"paged-FCFS", paged_runner(paged(kBlocks))});
        SimConfig nf = paged(kBlocks);
        nf.fork_sharing = false;
        s.variants.push_back({"paged-FCFS-nofork", paged_runner(nf)});
        contig_pair(s, kBlocks);
        sc.push_back(s);
    }
    {
        Scenario s;
        s.name = "F steady load (below saturation)";
        s.note = "arrival rate below what paged can serve: latencies are steady-state, not backlog";
        s.w.prompt_min = 32; s.w.prompt_max = 512; s.w.output_min = 32; s.w.output_max = 512;
        s.w.mean_interarrival = 4.0; s.w.max_seq_len = 2048;
        s.variants.push_back({"paged-FCFS", paged_runner(paged(kBlocks))});
        s.variants.push_back({"paged-SPF", paged_runner(paged(kBlocks, Policy::ShortestPromptFirst))});
        contig_pair(s, kBlocks);
        sc.push_back(s);
    }
    {
        Scenario s;
        s.name = "G prefix cache: 16 system prompts of 512 tokens, light load";
        s.note = "prefill modelled at 256 tokens/step (assumption); cache reclaimed under memory pressure";
        s.cost_columns = true;
        s.w.prompt_min = 32; s.w.prompt_max = 256; s.w.output_min = 32; s.w.output_max = 128;
        s.w.shared_prefix_fraction = 0.9; s.w.prefix_len = 512; s.w.num_prefix_groups = 16;
        s.w.mean_interarrival = 12.0; s.w.max_seq_len = 2048;
        SimConfig base = paged(kBlocks);
        base.prefill_tokens_per_step = 256;
        SimConfig none = base; none.prefix_sharing = false;
        SimConfig live = base;
        SimConfig cache = base; cache.prefix_cache = true;
        s.variants.push_back({"paged no sharing", paged_runner(none, "paged-noshare")});
        s.variants.push_back({"paged live-parent sharing", paged_runner(live, "paged-live")});
        s.variants.push_back({"paged live + cache", paged_runner(cache, "paged-live+cache")});
        sc.push_back(s);
    }
    {
        Scenario s;
        s.name = "H priority classes: 25% high priority, tight memory";
        s.priority_columns = true;
        s.w.prompt_min = 32; s.w.prompt_max = 512; s.w.output_min = 32; s.w.output_max = 256;
        s.w.high_priority_fraction = 0.25; s.w.mean_interarrival = 2.5; s.w.max_seq_len = 1024;
        const int blocks = 640;
        s.variants.push_back({"FCFS", paged_runner(paged(blocks))});
        s.variants.push_back({"SPF", paged_runner(paged(blocks, Policy::ShortestPromptFirst))});
        SimConfig po = paged(blocks, Policy::Priority);
        po.priority_preemption = false;
        s.variants.push_back({"priority (order only)", paged_runner(po, "PRIO-order")});
        s.variants.push_back({"priority + preemption", paged_runner(paged(blocks, Policy::Priority), "PRIO-preempt")});
        sc.push_back(s);
    }
    {
        Scenario s;
        s.name = "I swap vs recompute, tight memory";
        s.note = "assumed rates: prefill 256 tokens/step (fast) or 16 (slow), swap 64 blocks/step; step = one decode iteration";
        s.cost_columns = true;
        s.w.prompt_min = 32; s.w.prompt_max = 512; s.w.output_min = 32; s.w.output_max = 512;
        s.w.mean_interarrival = 2.0; s.w.max_seq_len = 1024;
        const int blocks = 768;
        for (int prefill : {256, 16}) {
            SimConfig r = paged(blocks);
            r.prefill_tokens_per_step = prefill;
            SimConfig w = r;
            w.preempt_mode = PreemptMode::Swap;
            w.swap_blocks_per_step = 64;
            s.variants.push_back({"recompute, prefill " + std::to_string(prefill) + " tok/step",
                                  paged_runner(r, "recompute-" + std::to_string(prefill))});
            s.variants.push_back({"swap, prefill " + std::to_string(prefill) + " tok/step",
                                  paged_runner(w, "swap-" + std::to_string(prefill))});
        }
        sc.push_back(s);
    }

    std::ofstream raw;
    if (!raw_path.empty()) {
        raw.open(raw_path);
        raw << "scenario,system,seed,util,mean_conc,peak_conc,tok_per_step,p50,p95,queue_mean,preempt,recomputed,cow,"
               "saved_mean,saved_peak,swap_out,cache_hits,cache_evictions,prefill_computed,prefill_saved,"
               "p95_high,p95_low,queue_high,queue_low\n";
    }
    std::printf("seeds=%d (first %llu) requests/prompts per run=%d block_size=16; cells are mean ± sample std dev over seeds\n",
                seeds, (unsigned long long)seed0, requests);

    struct Out { std::string scenario; std::vector<Agg> aggs; };
    std::vector<Out> outs;
    for (auto& s : sc) {
        std::vector<Agg> aggs(s.variants.size());
        for (int k = 0; k < seeds; ++k) {
            s.w.seed = seed0 + static_cast<uint64_t>(k);
            s.w.num_requests = requests;
            auto reqs = generate_workload(s.w);
            for (size_t v = 0; v < s.variants.size(); ++v) {
                Metrics m = s.variants[v].run(reqs);
                if (m.final_free_slots != m.capacity_slots || m.requests_completed != static_cast<int>(reqs.size())) {
                    std::fprintf(stderr, "sanity check failed: %s / %s seed %llu\n", s.name.c_str(),
                                 s.variants[v].name.c_str(), (unsigned long long)s.w.seed);
                    return 1;
                }
                Agg& a = aggs[v];
                a.label = s.variants[v].name;
                a.util.add(100 * m.mean_utilization); a.conc.add(m.mean_concurrent); a.peak.add(m.peak_concurrent);
                a.tps.add(m.tokens_per_step); a.p50.add(m.lat_p50); a.p95.add(m.lat_p95); a.queue.add(m.queue_mean);
                a.preempt.add(static_cast<double>(m.preemptions)); a.recomputed.add(static_cast<double>(m.recomputed_tokens));
                a.cow.add(static_cast<double>(m.cow_copies)); a.saved_mean.add(m.blocks_saved_mean);
                a.saved_peak.add(static_cast<double>(m.blocks_saved_peak)); a.swap_out.add(static_cast<double>(m.swap_out_blocks));
                a.cache_hits.add(static_cast<double>(m.cache_hits)); a.cache_evict.add(static_cast<double>(m.cache_evictions));
                a.prefill_computed.add(static_cast<double>(m.prefill_tokens_computed));
                a.prefill_saved.add(static_cast<double>(m.prefill_tokens_saved));
                a.p95_hi.add(m.lat_p95_high); a.p95_lo.add(m.lat_p95_low); a.q_hi.add(m.queue_mean_high); a.q_lo.add(m.queue_mean_low);
                if (raw.is_open())
                    raw << '"' << s.name << "\"," << s.variants[v].name << ',' << s.w.seed << ',' << 100 * m.mean_utilization << ','
                        << m.mean_concurrent << ',' << m.peak_concurrent << ',' << m.tokens_per_step << ',' << m.lat_p50 << ','
                        << m.lat_p95 << ',' << m.queue_mean << ',' << m.preemptions << ',' << m.recomputed_tokens << ','
                        << m.cow_copies << ',' << m.blocks_saved_mean << ',' << m.blocks_saved_peak << ',' << m.swap_out_blocks
                        << ',' << m.cache_hits << ',' << m.cache_evictions << ',' << m.prefill_tokens_computed << ','
                        << m.prefill_tokens_saved << ',' << m.lat_p95_high << ',' << m.lat_p95_low << ',' << m.queue_mean_high
                        << ',' << m.queue_mean_low << '\n';
            }
        }
        outs.push_back({s.name, aggs});

        std::printf("\n### %s\n", s.name.c_str());
        if (!s.note.empty()) std::printf("_%s_\n", s.note.c_str());
        std::printf("\n| System | Util %% | Mean conc | Peak conc | Tok/step | p50 lat | p95 lat | Mean queue | Preempt | Recomputed tok | COW | Blocks saved (mean / peak) |\n"
                    "|---|---|---|---|---|---|---|---|---|---|---|---|\n");
        for (const Agg& a : aggs)
            std::printf("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %.0f / %.0f |\n", a.label.c_str(),
                        a.util.pm().c_str(), a.conc.pm().c_str(), a.peak.pm("%.0f").c_str(), a.tps.pm().c_str(),
                        a.p50.pm("%.0f").c_str(), a.p95.pm("%.0f").c_str(), a.queue.pm().c_str(), a.preempt.pm("%.0f").c_str(),
                        a.recomputed.pm("%.0f").c_str(), a.cow.pm("%.0f").c_str(), a.saved_mean.mean(), a.saved_peak.mean());
        if (s.priority_columns) {
            std::printf("\n| System | p95 latency high | p95 latency low | Mean queue high | Mean queue low |\n|---|---|---|---|---|\n");
            for (const Agg& a : aggs)
                std::printf("| %s | %s | %s | %s | %s |\n", a.label.c_str(), a.p95_hi.pm("%.0f").c_str(), a.p95_lo.pm("%.0f").c_str(),
                            a.q_hi.pm().c_str(), a.q_lo.pm().c_str());
        }
        if (s.cost_columns) {
            std::printf("\n| System | Swapped-out blocks | Cache hits | Cache evictions | Prefill tokens computed | Prefill tokens saved |\n|---|---|---|---|---|---|\n");
            for (const Agg& a : aggs)
                std::printf("| %s | %.0f | %.0f | %.0f | %.0f | %.0f |\n", a.label.c_str(), a.swap_out.mean(), a.cache_hits.mean(),
                            a.cache_evict.mean(), a.prefill_computed.mean(), a.prefill_saved.mean());
        }
    }

    if (!csv_path.empty()) {
        std::ofstream f(csv_path);
        f << "scenario,system,util,util_sd,mean_conc,mean_conc_sd,tok_per_step,p50,p95,p95_sd,queue_mean,preempt,recomputed,cow,saved_mean,saved_peak\n";
        for (const Out& o : outs)
            for (const Agg& a : o.aggs)
                f << '"' << o.scenario << "\"," << a.label << ',' << a.util.mean() / 100 << ',' << a.util.sd() / 100 << ','
                  << a.conc.mean() << ',' << a.conc.sd() << ',' << a.tps.mean() << ',' << a.p50.mean() << ',' << a.p95.mean()
                  << ',' << a.p95.sd() << ',' << a.queue.mean() << ',' << a.preempt.mean() << ',' << a.recomputed.mean() << ','
                  << a.cow.mean() << ',' << a.saved_mean.mean() << ',' << a.saved_peak.mean() << '\n';
    }
    return 0;
}
