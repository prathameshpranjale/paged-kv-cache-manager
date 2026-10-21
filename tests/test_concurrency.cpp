// Run under ThreadSanitizer (scripts/test.sh does): cmake -DSANITIZE=tsan.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>

#include "kvpage/bounded_queue.h"
#include "kvpage/simulator.h"
#include "kvpage/workload.h"
#include "testing.h"

using namespace kvp;

namespace {

// P producer threads push disjoint slices of the workload into a bounded queue; one
// scheduler thread drains it. Returns what the consumer received.
std::vector<Request> run_pipeline(const std::vector<Request>& all, int producers, size_t capacity) {
    BoundedQueue<Request> q(capacity);
    std::vector<Request> received;
    std::thread consumer([&] {
        Request r;
        while (q.pop(r)) received.push_back(r);
    });
    std::vector<std::thread> ps;
    for (int p = 0; p < producers; ++p)
        ps.emplace_back([&, p] {
            for (size_t i = static_cast<size_t>(p); i < all.size(); i += static_cast<size_t>(producers))
                q.push(all[i]);
        });
    for (auto& t : ps) t.join();
    q.close();
    consumer.join();
    return received;
}

void check_exactly_once(const std::vector<Request>& all, std::vector<Request> got) {
    CHECK_EQ(got.size(), all.size());
    std::sort(got.begin(), got.end(), [](const Request& a, const Request& b) { return a.id < b.id; });
    for (size_t i = 0; i < got.size() && i < all.size(); ++i) CHECK_EQ(got[i].id, all[i].id);  // no loss, no dup
}

}  // namespace

TEST(concurrency_no_lost_or_duplicated_requests) {
    WorkloadConfig w;
    w.seed = 11;
    w.num_requests = 4000;
    auto all = generate_workload(w);
    for (int producers : {1, 2, 4, 8})
        for (size_t cap : {size_t{1}, size_t{4}, size_t{64}}) check_exactly_once(all, run_pipeline(all, producers, cap));
}

TEST(concurrency_pipeline_feeds_scheduler_deterministically) {
    WorkloadConfig w;
    w.seed = 12;
    w.num_requests = 300;
    w.mean_interarrival = 0.5;
    w.shared_prefix_fraction = 0.4;
    w.max_seq_len = 400;
    auto all = generate_workload(w);
    SimConfig c;
    c.num_blocks = 150;
    c.block_size = 16;
    Metrics reference = run_paged(all, c);
    for (int round = 0; round < 5; ++round) {
        auto got = run_pipeline(all, 4, 8);  // arrival order differs run to run
        check_exactly_once(all, got);
        Metrics m = run_paged(got, c);  // the simulator orders by (arrival, id) itself
        CHECK_EQ(m.summary(), reference.summary());
    }
}

TEST(concurrency_close_unblocks_waiting_threads) {
    BoundedQueue<int> q(1);
    CHECK(q.push(1));
    std::atomic<bool> producer_returned{false}, consumer_returned{false};
    std::thread blocked_producer([&] {
        bool ok = q.push(2);  // blocks: queue is full
        producer_returned = !ok;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    q.close();
    blocked_producer.join();
    CHECK(producer_returned.load());  // returned false once closed
    int v = 0;
    CHECK(q.pop(v));  // the queued item is still delivered
    CHECK_EQ(v, 1);
    CHECK(!q.pop(v));  // then closed + drained
    std::thread blocked_consumer([&] {
        BoundedQueue<int> empty(2);
        std::thread closer([&] { empty.close(); });
        int x;
        consumer_returned = !empty.pop(x);
        closer.join();
    });
    blocked_consumer.join();
    CHECK(consumer_returned.load());
}
