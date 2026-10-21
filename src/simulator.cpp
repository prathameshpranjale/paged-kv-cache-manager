#include "kvpage/simulator.h"

#include <algorithm>
#include <climits>
#include <deque>
#include <map>
#include <stdexcept>
#include <string>

#include "kvpage/kv_manager.h"

namespace kvp {

const char* policy_name(Policy p) {
    switch (p) {
        case Policy::FCFS: return "FCFS";
        case Policy::ShortestPromptFirst: return "SPF";
        case Policy::Priority: return "PRIO";
    }
    return "?";
}

namespace {

struct State {
    int generated = 0;  // output tokens produced so far
    int first_admit = -1;
    int preemptions = 0;
    bool running = false;
    int stall = 0;            // steps left before it may generate (prefill / swap-in cost)
    bool swapped = false;     // its blocks sit in the CPU pool
    int swapped_blocks = 0;
};

std::vector<Request> sorted_copy(const std::vector<Request>& in) {
    std::vector<Request> r = in;
    std::stable_sort(r.begin(), r.end(), [](const Request& a, const Request& b) {
        return a.arrival != b.arrival ? a.arrival < b.arrival : a.id < b.id;
    });
    for (size_t i = 1; i < r.size(); ++i)
        if (r[i].id == r[i - 1].id) throw std::invalid_argument("duplicate request id");
    return r;
}

// Position in `waiting` of the highest-priority request (first one wins ties), or -1.
template <class Prio>
int top_priority_pos(const std::deque<int>& waiting, Prio prio) {
    int best = -1;
    for (int p = 0; p < static_cast<int>(waiting.size()); ++p)
        if (best < 0 || prio(waiting[p]) > prio(waiting[best])) best = p;
    return best;
}

// Position in `waiting` of the request to admit next, or -1.
// FCFS: only the head may go (head-of-line blocking). SPF: shortest prompt that fits.
// Priority: the highest-priority request, if it fits (otherwise nobody: no priority inversion).
template <class PromptLen, class Prio, class Fits>
int pick(const std::deque<int>& waiting, Policy policy, PromptLen prompt_len, Prio prio, Fits fits) {
    if (waiting.empty()) return -1;
    if (policy == Policy::FCFS) return fits(waiting.front()) ? 0 : -1;
    if (policy == Policy::Priority) {
        int p = top_priority_pos(waiting, prio);
        return fits(waiting[p]) ? p : -1;
    }
    int best = -1;
    for (int p = 0; p < static_cast<int>(waiting.size()); ++p) {
        int i = waiting[p];
        if (!fits(i)) continue;
        if (best < 0 || prompt_len(i) < prompt_len(waiting[best]) ||
            (prompt_len(i) == prompt_len(waiting[best]) && i < waiting[best]))
            best = p;
    }
    return best;
}

int stall_for(int units, int rate) { return rate > 0 ? ceil_div(units, rate) : 0; }

}  // namespace

Metrics run_paged(const std::vector<Request>& input, const SimConfig& cfg) {
    std::vector<Request> reqs = sorted_copy(input);
    if (!cfg.fork_sharing)
        for (auto& r : reqs) r.fork_of = kNoRequest;  // samples become independent requests
    std::map<RequestId, int> index_of;
    for (int i = 0; i < static_cast<int>(reqs.size()); ++i) index_of[reqs[i].id] = i;
    const int n = static_cast<int>(reqs.size());
    const int B = cfg.block_size;
    for (const auto& r : reqs)
        if (r.output_len < 1 || ceil_div(r.prompt_len + r.output_len, B) > cfg.num_blocks)
            throw std::invalid_argument("request " + std::to_string(r.id) + " can never fit");

    KVManager kv(cfg.num_blocks, B);
    kv.set_check_each_op(cfg.validate_each_step);
    MetricsCollector mc;
    std::vector<State> st(n);
    std::vector<std::vector<int>> children(n);  // samples waiting for their leader's admission
    std::deque<int> waiting;
    std::vector<int> running;  // admission order; the back is the newest
    int next = 0, done = 0;
    int64_t step = 0;
    int64_t cpu_used = 0;  // blocks currently in the CPU swap pool

    auto prompt_now = [&](int i) { return reqs[i].prompt_len + st[i].generated; };
    auto prio = [&](int i) { return reqs[i].priority; };

    // Victim for memory pressure: newest running request (Priority: lowest class first, newest within it).
    // `limit`: only requests with priority < limit may be chosen (INT_MAX = any). Returns -1 if none.
    auto pick_victim = [&](int limit) -> int {
        if (running.empty()) return -1;
        if (cfg.policy != Policy::Priority) return limit == INT_MAX ? running.back() : -1;
        int best = -1;
        for (int k = static_cast<int>(running.size()) - 1; k >= 0; --k)
            if (best < 0 || prio(running[k]) < prio(best)) best = running[k];
        return prio(best) < limit ? best : -1;
    };

    auto preempt = [&](int i) {
        const int blocks = static_cast<int>(kv.table(reqs[i].id).blocks.size());
        if (cfg.preempt_mode == PreemptMode::Swap && cpu_used + blocks <= cfg.swap_capacity_blocks) {
            st[i].swapped = true;  // keep its data on the CPU; no recompute needed later
            st[i].swapped_blocks = blocks;
            cpu_used += blocks;
            mc.on_preempt(0);
            mc.on_swap(blocks, 0);
        } else {
            mc.on_preempt(prompt_now(i));  // everything it held must be recomputed
        }
        kv.free(reqs[i].id);
        st[i].running = false;
        ++st[i].preemptions;
        running.erase(std::find(running.begin(), running.end(), i));
        waiting.push_front(i);
    };

    for (; done < n; ++step) {
        if (step > cfg.max_steps) throw std::logic_error("simulation exceeded max_steps (livelock?)");
        while (next < n && reqs[next].arrival <= step) {
            int i = next++;
            if (reqs[i].fork_of != kNoRequest) children[index_of.at(reqs[i].fork_of)].push_back(i);
            else waiting.push_back(i);  // extra samples wait for their leader to be admitted
        }

        // 1. Every running request generates one token; finished ones are freed at once.
        bool preempted = false;
        int generated_now = 0;
        const std::vector<int> snapshot = running;
        for (int i : snapshot) {
            if (!st[i].running) continue;  // preempted earlier in this step
            if (st[i].stall > 0) {         // still prefilling / swapping in
                --st[i].stall;
                continue;
            }
            while (!kv.append_token(reqs[i].id)) {
                // 2. Out of blocks: preempt a victim.
                int victim = pick_victim(INT_MAX);
                preempt(victim);
                preempted = true;
                if (victim == i) break;
            }
            if (!st[i].running) continue;
            ++st[i].generated;
            ++generated_now;
            if (st[i].generated == reqs[i].output_len) {
                kv.free(reqs[i].id);
                st[i].running = false;
                running.erase(std::find(running.begin(), running.end(), i));
                RequestResult rr;
                rr.id = reqs[i].id;
                rr.arrival = reqs[i].arrival;
                rr.first_admit = st[i].first_admit;
                rr.finish = static_cast<int>(step) + 1;
                rr.preemptions = st[i].preemptions;
                rr.priority = reqs[i].priority;
                mc.on_complete(rr);
                ++done;
            }
        }

        // 3. Admit waiting requests while they fit (not in a step that preempted: avoids thrash).
        while (!preempted && !waiting.empty()) {
            if (cfg.max_running > 0 && static_cast<int>(running.size()) >= cfg.max_running) break;
            // One live parent per prefix group: a running request that already holds the prefix.
            std::map<int, RequestId> parent_of;
            if (cfg.prefix_sharing)
                for (int j : running) {
                    const Request& rj = reqs[j];
                    if (rj.prefix_group >= 0 && !parent_of.count(rj.prefix_group) &&
                        kv.table(rj.id).tokens >= rj.prefix_len)
                        parent_of[rj.prefix_group] = rj.id;
                }
            const int reserve = running.empty() ? 0 : cfg.watermark_blocks;
            auto parent_for = [&](int i) -> RequestId {
                if (st[i].swapped || reqs[i].prefix_group < 0) return kNoRequest;
                auto it = parent_of.find(reqs[i].prefix_group);
                return it == parent_of.end() ? kNoRequest : it->second;
            };
            // Persistent prefix cache entry usable by request i (key = its prefix group), or -1.
            auto cache_key = [&](int i) -> int {
                if (!cfg.prefix_cache || !cfg.prefix_sharing || st[i].swapped || reqs[i].prefix_group < 0) return -1;
                return kv.has_cached_prefix(reqs[i].prefix_group) ? reqs[i].prefix_group : -1;
            };
            auto need_of = [&](int i) {
                int k = cache_key(i);
                return k >= 0 ? kv.blocks_needed_to_admit_cached(k, reqs[i].prefix_len, prompt_now(i))
                              : kv.blocks_needed_to_admit(parent_for(i), reqs[i].prefix_len, prompt_now(i));
            };
            auto fits = [&](int i) {
                // reclaimable cache blocks count as available, except the entry this request would share
                int k = cache_key(i);
                int shared = k >= 0 ? kv.shared_blocks_with_cached(k, reqs[i].prefix_len, prompt_now(i)) : 0;
                return kv.free_blocks() + kv.evictable_blocks(k, shared) - need_of(i) >= reserve;
            };
            int p = pick(waiting, cfg.policy, prompt_now, prio, fits);
            if (p < 0) {
                // Priority: the top waiting request may displace strictly lower-priority running requests.
                if (cfg.policy == Policy::Priority && cfg.priority_preemption) {
                    int top = top_priority_pos(waiting, prio);
                    int v = pick_victim(prio(waiting[top]));
                    if (v >= 0) {
                        preempt(v);
                        continue;  // re-evaluate with the freed memory (parents may have changed)
                    }
                }
                break;
            }
            int i = waiting[p];
            waiting.erase(waiting.begin() + p);
            const int key = cache_key(i);
            const int need = need_of(i);
            const int shared_tokens = (ceil_div(prompt_now(i), B) - need) * B;
            bool ok = key >= 0 ? kv.admit_with_cached_prefix(reqs[i].id, key, reqs[i].prefix_len, prompt_now(i))
                               : kv.admit_with_prefix(reqs[i].id, parent_for(i), reqs[i].prefix_len, prompt_now(i));
            if (!ok) throw std::logic_error("admission failed after fits() succeeded");
            if (key >= 0 && shared_tokens > 0) mc.on_cache_hit();
            mc.on_prefill(st[i].swapped ? 0 : prompt_now(i) - shared_tokens, st[i].swapped ? 0 : shared_tokens);
            if (st[i].swapped) {  // swap-in: transfer instead of recompute
                cpu_used -= st[i].swapped_blocks;
                mc.on_swap(0, st[i].swapped_blocks);
                st[i].stall = stall_for(st[i].swapped_blocks, cfg.swap_blocks_per_step);
                st[i].swapped = false;
                st[i].swapped_blocks = 0;
            } else {
                st[i].stall = stall_for(prompt_now(i) - shared_tokens, cfg.prefill_tokens_per_step);
            }
            st[i].running = true;
            if (st[i].first_admit < 0) st[i].first_admit = static_cast<int>(step);
            running.push_back(i);
            if (cfg.prefix_cache && cfg.prefix_sharing && reqs[i].prefix_group >= 0 &&
                reqs[i].prefix_len >= B && !kv.has_cached_prefix(reqs[i].prefix_group))
                kv.cache_prefix(reqs[i].prefix_group, reqs[i].id, reqs[i].prefix_len);
            for (int c : children[i]) {  // parallel samples: fork the leader, sharing every block
                kv.fork(reqs[c].id, reqs[i].id);
                st[c].running = true;
                st[c].stall = st[i].stall;  // they wait for the same prefill
                st[c].first_admit = static_cast<int>(step);
                running.push_back(c);
            }
            children[i].clear();
        }

        KVStats ks = kv.stats();
        StepSample s;
        s.running = static_cast<int>(running.size());
        s.stored = ks.used_tokens;
        s.allocated = ks.allocated_slots;
        s.saved_blocks = ks.blocks_saved;
        s.generated = generated_now;
        mc.on_step(s);

        if (cfg.validate_each_step) {
            std::string err = kv.validate();
            if (!err.empty()) throw std::logic_error("step " + std::to_string(step) + ": " + err);
            if (ks.requests != s.running) throw std::logic_error("running set != KV requests");
            for (int j : running)
                if (kv.table(reqs[j].id).tokens != reqs[j].prompt_len + st[j].generated)
                    throw std::logic_error("token count mismatch for request " + std::to_string(reqs[j].id));
            if (cpu_used < 0) throw std::logic_error("negative CPU pool usage");
        }
    }

    const int64_t evictions = kv.stats().cache_evictions;
    const int64_t cow = kv.stats().cow_copies;
    kv.clear_cache();  // cached prefixes are not leaks: release them before the end-of-run check
    if (cpu_used != 0) throw std::logic_error("CPU swap pool not empty at the end");
    int64_t cap = static_cast<int64_t>(cfg.num_blocks) * B;
    int64_t free_slots = static_cast<int64_t>(kv.free_blocks()) * B;
    std::string label = std::string("paged-") + policy_name(cfg.policy);
    return mc.finish(label, step, cap, free_slots, cow, evictions);
}

Metrics run_contiguous(const std::vector<Request>& input, const ContigConfig& cfg) {
    const std::vector<Request> reqs = sorted_copy(input);
    const int n = static_cast<int>(reqs.size());
    auto reserve_of = [&](int i) -> int64_t {
        return cfg.reserve_len > 0 ? cfg.reserve_len : reqs[i].prompt_len + reqs[i].output_len;
    };
    for (int i = 0; i < n; ++i)
        if (reqs[i].output_len < 1 || reserve_of(i) < reqs[i].prompt_len + reqs[i].output_len ||
            reserve_of(i) > cfg.total_slots)
            throw std::invalid_argument("request " + std::to_string(reqs[i].id) + " cannot be reserved");

    MetricsCollector mc;
    std::vector<State> st(n);
    std::deque<int> waiting;
    std::vector<int> running;
    int64_t free_slots = cfg.total_slots;
    int next = 0, done = 0;
    int64_t step = 0;

    for (; done < n; ++step) {
        if (step > cfg.max_steps) throw std::logic_error("simulation exceeded max_steps");
        while (next < n && reqs[next].arrival <= step) waiting.push_back(next++);

        int generated_now = 0;
        for (size_t k = 0; k < running.size();) {
            int i = running[k];
            ++st[i].generated;
            ++generated_now;
            if (st[i].generated == reqs[i].output_len) {
                free_slots += reserve_of(i);
                st[i].running = false;
                running.erase(running.begin() + static_cast<std::ptrdiff_t>(k));
                RequestResult rr;
                rr.id = reqs[i].id;
                rr.arrival = reqs[i].arrival;
                rr.first_admit = st[i].first_admit;
                rr.finish = static_cast<int>(step) + 1;
                rr.priority = reqs[i].priority;
                mc.on_complete(rr);
                ++done;
            } else {
                ++k;
            }
        }

        while (!waiting.empty()) {
            if (cfg.max_running > 0 && static_cast<int>(running.size()) >= cfg.max_running) break;
            auto fits = [&](int i) { return free_slots >= reserve_of(i); };
            auto plen = [&](int i) { return reqs[i].prompt_len; };
            auto prio = [&](int i) { return reqs[i].priority; };
            int p = pick(waiting, cfg.policy, plen, prio, fits);
            if (p < 0) break;
            int i = waiting[p];
            waiting.erase(waiting.begin() + p);
            free_slots -= reserve_of(i);
            st[i].running = true;
            st[i].first_admit = static_cast<int>(step);
            running.push_back(i);
        }

        StepSample s;
        s.running = static_cast<int>(running.size());
        for (int j : running) s.stored += reqs[j].prompt_len + st[j].generated;
        s.allocated = cfg.total_slots - free_slots;
        s.generated = generated_now;
        mc.on_step(s);
    }

    std::string label = cfg.reserve_len > 0 ? "contig-maxlen-" : "contig-oracle-";
    label += policy_name(cfg.policy);
    return mc.finish(label, step, cfg.total_slots, free_slots, 0);
}

}  // namespace kvp
