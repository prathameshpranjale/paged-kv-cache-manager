#include <algorithm>

#include "kvpage/simulator.h"
#include "kvpage/workload.h"
#include "testing.h"

using namespace kvp;

namespace {
Request make(RequestId id, int arrival, int prompt, int output) {
    Request r;
    r.id = id;
    r.arrival = arrival;
    r.prompt_len = prompt;
    r.output_len = output;
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

TEST(sched_single_request_timeline) {
    SimConfig c;
    c.num_blocks = 8;
    c.block_size = 4;
    c.validate_each_step = true;
    Metrics m = run_paged({make(0, 0, 5, 3)}, c);
    CHECK_EQ(m.requests_completed, 1);
    CHECK_EQ(result_of(m, 0).first_admit, 0);  // admitted (prefill) in step 0
    CHECK_EQ(result_of(m, 0).finish, 4);       // tokens in steps 1, 2, 3 -> done at end of step 3
    CHECK_EQ(m.tokens_generated, 3);
    CHECK_EQ(m.final_free_slots, m.capacity_slots);
}

TEST(sched_preemption_and_recompute) {
    // 6 blocks of 4 = 24 slots. Two requests of 8 prompt + 12 output each need 5 blocks
    // at the end, so both cannot finish side by side: the newer one must be preempted.
    SimConfig c;
    c.num_blocks = 6;
    c.block_size = 4;
    c.watermark_blocks = 0;
    c.validate_each_step = true;
    Metrics m = run_paged({make(0, 0, 8, 12), make(1, 0, 8, 12)}, c);
    CHECK_EQ(m.requests_completed, 2);
    CHECK(m.preemptions >= 1);
    CHECK(m.recomputed_tokens > 0);
    CHECK_EQ(m.tokens_generated, 24);  // each output token is produced exactly once
    CHECK_EQ(m.final_free_slots, m.capacity_slots);
    CHECK_EQ(result_of(m, 0).preemptions, 0);  // the oldest request is never the victim here
    CHECK(result_of(m, 1).preemptions >= 1);
}

TEST(sched_request_that_can_never_fit_is_rejected) {
    SimConfig c;
    c.num_blocks = 2;
    c.block_size = 4;
    CHECK_THROWS(run_paged({make(0, 0, 8, 8)}, c));
    ContigConfig cc;
    cc.total_slots = 10;
    cc.reserve_len = 16;
    CHECK_THROWS(run_contiguous({make(0, 0, 4, 4)}, cc));
}

TEST(sched_spf_avoids_head_of_line_blocking) {
    // 10 blocks of 4. r0 holds 6-9 blocks for a long time; r1 (big) cannot fit behind it;
    // r2 (tiny) can. FCFS makes r2 wait behind r1, SPF admits r2 at once.
    std::vector<Request> reqs = {make(0, 0, 24, 12), make(1, 1, 20, 4), make(2, 2, 3, 4)};
    SimConfig c;
    c.num_blocks = 10;
    c.block_size = 4;
    c.watermark_blocks = 0;
    c.validate_each_step = true;
    c.policy = Policy::FCFS;
    Metrics f = run_paged(reqs, c);
    c.policy = Policy::ShortestPromptFirst;
    Metrics s = run_paged(reqs, c);
    CHECK_EQ(f.requests_completed, 3);
    CHECK_EQ(s.requests_completed, 3);
    CHECK_EQ(result_of(s, 2).first_admit, 2);
    CHECK(result_of(f, 2).first_admit > result_of(s, 2).first_admit);
}

TEST(sched_prefix_sharing_saves_blocks) {
    WorkloadConfig w;
    w.seed = 7;
    w.num_requests = 40;
    w.mean_interarrival = 0.5;
    w.shared_prefix_fraction = 1.0;
    w.prefix_len = 64;
    w.max_seq_len = 512;
    auto reqs = generate_workload(w);
    SimConfig c;
    c.num_blocks = 2000;
    c.block_size = 16;
    c.validate_each_step = true;
    c.prefix_sharing = true;
    Metrics on = run_paged(reqs, c);
    c.prefix_sharing = false;
    Metrics off = run_paged(reqs, c);
    CHECK(on.blocks_saved_peak > 0);
    CHECK_EQ(off.blocks_saved_peak, 0);
    CHECK(on.mean_waste_slots >= 0);
    CHECK_EQ(on.requests_completed, 40);
}

TEST(sched_contiguous_maxlen_limits_concurrency) {
    WorkloadConfig w;
    w.seed = 3;
    w.num_requests = 60;
    w.mean_interarrival = 0.1;  // heavy load
    w.prompt_max = 100;
    w.output_max = 100;
    w.max_seq_len = 256;
    auto reqs = generate_workload(w);
    ContigConfig cc;
    cc.total_slots = 1024;
    cc.reserve_len = 256;
    Metrics m = run_contiguous(reqs, cc);
    CHECK_EQ(m.requests_completed, 60);
    CHECK_EQ(m.peak_concurrent, 4);  // 1024 / 256
    CHECK_EQ(m.final_free_slots, m.capacity_slots);
    CHECK(m.mean_utilization < 1.0);

    SimConfig pc;
    pc.num_blocks = 64;  // same 1024 slots
    pc.block_size = 16;
    Metrics p = run_paged(reqs, pc);
    CHECK(p.peak_concurrent > m.peak_concurrent);
    CHECK(p.mean_utilization > m.mean_utilization);
}

TEST(determinism_same_seed_same_metrics) {
    WorkloadConfig w;
    w.seed = 99;
    w.num_requests = 120;
    w.mean_interarrival = 0.4;
    w.prompt_dist = Dist::LongTail;
    w.shared_prefix_fraction = 0.5;
    w.max_seq_len = 400;
    SimConfig c;
    c.num_blocks = 120;
    c.block_size = 16;
    c.validate_each_step = true;
    Metrics a = run_paged(generate_workload(w), c);
    Metrics b = run_paged(generate_workload(w), c);
    CHECK_EQ(a.summary(), b.summary());
    w.seed = 100;
    Metrics d = run_paged(generate_workload(w), c);
    CHECK(a.summary() != d.summary());
    c.validate_each_step = false;  // checking must not change behaviour
    CHECK_EQ(run_paged(generate_workload(w), c).summary(), d.summary());
}

TEST(workload_respects_limits_and_is_sorted) {
    WorkloadConfig w;
    w.seed = 5;
    w.num_requests = 2000;
    w.prompt_dist = Dist::LongTail;
    w.output_dist = Dist::LongTail;
    w.shared_prefix_fraction = 0.5;
    w.max_seq_len = 300;
    auto r = generate_workload(w);
    CHECK_EQ(r.size(), 2000u);
    int shared = 0;
    for (size_t i = 0; i < r.size(); ++i) {
        CHECK(r[i].prompt_len + r[i].output_len <= 300);
        CHECK(r[i].output_len >= 1 && r[i].prompt_len >= 1);
        CHECK(r[i].prefix_len <= r[i].prompt_len);
        if (i) CHECK(r[i].arrival >= r[i - 1].arrival);
        if (r[i].prefix_group >= 0) ++shared;
    }
    CHECK(shared > 800 && shared < 1200);  // about half
}

TEST(metrics_percentile_nearest_rank) {
    std::vector<double> v = {5, 1, 4, 2, 3, 10, 9, 8, 7, 6};
    CHECK_EQ(percentile(v, 50), 5.0);
    CHECK_EQ(percentile(v, 95), 10.0);
    CHECK_EQ(percentile(v, 0), 1.0);
    CHECK_EQ(percentile({}, 50), 0.0);
}

TEST(sched_parallel_sampling_forks_and_cows) {
    WorkloadConfig w;
    w.seed = 21;
    w.num_requests = 30;
    w.mean_interarrival = 1.0;
    w.prompt_min = 20;
    w.prompt_max = 90;  // mostly not block aligned
    w.output_min = 10;
    w.output_max = 60;
    w.samples_per_request = 4;
    w.max_seq_len = 300;
    auto reqs = generate_workload(w);
    CHECK_EQ(reqs.size(), 120u);
    SimConfig c;
    c.num_blocks = 400;
    c.block_size = 16;
    c.validate_each_step = true;
    Metrics shared = run_paged(reqs, c);
    c.fork_sharing = false;
    Metrics indep = run_paged(reqs, c);
    CHECK_EQ(shared.requests_completed, 120);
    CHECK_EQ(indep.requests_completed, 120);
    CHECK_EQ(shared.final_free_slots, shared.capacity_slots);
    CHECK(shared.cow_copies > 0);       // forks hit copy-on-write
    CHECK(shared.blocks_saved_peak > 0);
    CHECK_EQ(indep.cow_copies, 0);
    CHECK_EQ(indep.blocks_saved_peak, 0);
    CHECK(shared.mean_utilization > 0.0);
}
