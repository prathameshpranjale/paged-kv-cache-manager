#pragma once
#include <cstdint>
#include <vector>

#include "kvpage/types.h"

namespace kvp {

struct Request {
    RequestId id = 0;
    int arrival = 0;        // step at which the request arrives
    int prompt_len = 0;     // includes the shared prefix, if any
    int output_len = 1;     // tokens to generate (>= 1)
    int prefix_group = -1;  // -1: no shared prefix
    int prefix_len = 0;     // leading prompt tokens shared by the group
    RequestId fork_of = kNoRequest;  // parallel sampling: another sample of this leader request
    int priority = 0;       // higher is more important (Policy::Priority)
};

enum class Dist { Uniform, LongTail };  // LongTail: capped Pareto (alpha 1.5)

struct WorkloadConfig {
    uint64_t seed = 1;
    int num_requests = 500;
    double mean_interarrival = 1.0;  // steps, exponential gaps
    Dist prompt_dist = Dist::Uniform;
    int prompt_min = 32, prompt_max = 256;  // length of the non-shared part
    Dist output_dist = Dist::Uniform;
    int output_min = 16, output_max = 256;
    double shared_prefix_fraction = 0.0;
    int prefix_len = 128;
    int num_prefix_groups = 1;
    int max_seq_len = 1024;  // prompt + output never exceeds this
    double high_priority_fraction = 0.0;  // these requests get priority 1, the rest 0
    int samples_per_request = 1;  // >1: parallel sampling; extra samples share the leader prompt (fork_of)
};

// Deterministic for a given config (own PRNG and samplers, no <random> distributions).
// Requests are returned sorted by (arrival, id).
std::vector<Request> generate_workload(const WorkloadConfig& cfg);

}  // namespace kvp
