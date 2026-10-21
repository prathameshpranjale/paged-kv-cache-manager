// CPU reference attention vs the kernel-shaped algorithm, on block tables produced by the real KVManager.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "kvpage/kv_manager.h"
#include "kvpage/paged_attention.h"
#include "testing.h"

using namespace kvp;

namespace {

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
    uint32_t next() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<uint32_t>(s >> 33);
    }
    float uniform() { return static_cast<float>(next() % 20001) / 10000.f - 1.f; }  // [-1, 1]
    int range(int lo, int hi) { return lo + static_cast<int>(next() % static_cast<uint32_t>(hi - lo + 1)); }
};

// Builds random sequences through a real KVManager (some share prefix blocks) and returns the max
// |reference - paged_attention_cpu(partitions)| over all outputs.
float run_case(uint64_t seed, const AttnShape& s, int partitions, std::vector<int> lens, bool share) {
    Rng rng(seed);
    const int ns = static_cast<int>(lens.size());
    int max_blocks = 0;
    for (int n : lens) max_blocks = std::max(max_blocks, ceil_div(n, s.block_size));
    KVManager kv(ns * max_blocks + 4, s.block_size);
    for (int i = 0; i < ns; ++i) {
        bool ok = (share && i > 0 && lens[i] >= s.block_size)
                      ? kv.admit_with_prefix(i, 0, std::min(lens[0], lens[i]), lens[i])  // shares leading full blocks
                      : kv.admit(i, lens[i]);
        CHECK(ok);
    }
    const size_t kv_elems = static_cast<size_t>(kv.total_blocks()) * s.block_size * s.kv_heads * s.head_dim;
    std::vector<float> K(kv_elems), V(kv_elems), Q(static_cast<size_t>(ns) * s.num_heads * s.head_dim);
    for (auto& x : K) x = rng.uniform();
    for (auto& x : V) x = rng.uniform();
    for (auto& x : Q) x = rng.uniform() * 2.f;
    std::vector<int32_t> tables(static_cast<size_t>(ns) * max_blocks, 0);  // padding points at block 0
    for (int i = 0; i < ns; ++i) {
        const auto& b = kv.table(i).blocks;
        for (size_t j = 0; j < b.size(); ++j) tables[static_cast<size_t>(i) * max_blocks + j] = b[j];
    }
    std::vector<int32_t> sl(lens.begin(), lens.end());
    std::vector<float> ref(Q.size()), got(Q.size());
    reference_attention(s, ns, max_blocks, Q.data(), K.data(), V.data(), tables.data(), sl.data(), ref.data());
    paged_attention_cpu(s, ns, max_blocks, Q.data(), K.data(), V.data(), tables.data(), sl.data(), got.data(),
                        partitions);
    float worst = 0;
    for (size_t i = 0; i < ref.size(); ++i) worst = std::max(worst, std::fabs(ref[i] - got[i]));
    return worst;
}

}  // namespace

TEST(attention_matches_reference_over_block_sizes_and_lengths) {
    for (int B : {1, 4, 8, 16, 32}) {
        AttnShape s{8, 2, 64, B};
        // lengths around block boundaries: 1, B-1, B, B+1, 2B, 3B+1 (those that are >= 1) and a few random
        std::vector<int> lens;
        for (int n : {1, B - 1, B, B + 1, 2 * B, 3 * B + 1, 97, 130})
            if (n >= 1) lens.push_back(n);
        for (int partitions : {1, 4, 8}) {
            float d = run_case(static_cast<uint64_t>(B * 100 + partitions), s, partitions, lens, false);
            CHECK(d < 1e-5f);
            if (!(d < 1e-5f)) std::cerr << "  B=" << B << " partitions=" << partitions << " diff=" << d << "\n";
        }
    }
}

TEST(attention_handles_more_partitions_than_tokens_and_single_token) {
    AttnShape s{4, 4, 32, 16};
    CHECK(run_case(5, s, 8, {1}, false) < 1e-5f);   // 7 of 8 partitions are empty
    CHECK(run_case(6, s, 32, {2, 3, 5}, false) < 1e-5f);
}

TEST(attention_with_shared_blocks_and_gqa_ratios) {
    for (auto hk : {std::pair<int, int>{8, 8}, {8, 2}, {8, 1}, {14, 2}}) {
        AttnShape s{hk.first, hk.second, 64, 16};
        float d = run_case(77, s, 4, {70, 70, 55, 33, 16}, true);  // sequences 1.. share leader blocks
        CHECK(d < 1e-5f);
    }
}

TEST(attention_random_cases) {
    for (uint64_t seed = 1; seed <= 60; ++seed) {
        Rng r(seed);
        AttnShape s{8, 2, 32 * r.range(1, 4), 1 << r.range(0, 5)};
        std::vector<int> lens(static_cast<size_t>(r.range(1, 6)));
        for (auto& n : lens) n = r.range(1, 200);
        float d = run_case(seed, s, r.range(1, 8), lens, r.range(0, 1) == 1);
        CHECK(d < 2e-5f);
        if (!(d < 2e-5f)) std::cerr << "  FAILING SEED " << seed << " diff=" << d << "\n";
    }
}

TEST(attention_rejects_bad_input) {
    AttnShape s{8, 3, 64, 16};  // 8 % 3 != 0
    float x = 0;
    int32_t t = 0, n = 1;
    CHECK_THROWS(reference_attention(s, 1, 1, &x, &x, &x, &t, &n, &x));
    AttnShape ok{8, 2, 64, 16};
    int32_t zero = 0;
    CHECK_THROWS(reference_attention(ok, 1, 1, &x, &x, &x, &t, &zero, &x));
}
