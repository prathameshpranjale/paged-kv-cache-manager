#include "kvpage/paged_attention.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace kvp {
namespace {

void check(const AttnShape& s, int num_seqs, int max_blocks, const int32_t* seq_lens) {
    if (s.kv_heads <= 0 || s.num_heads % s.kv_heads != 0) throw std::invalid_argument("num_heads % kv_heads != 0");
    for (int i = 0; i < num_seqs; ++i)
        if (seq_lens[i] < 1 || (seq_lens[i] + s.block_size - 1) / s.block_size > max_blocks)
            throw std::invalid_argument("bad seq_len");
}

inline size_t kv_index(const AttnShape& s, int block, int off, int kvh) {
    return ((static_cast<size_t>(block) * s.block_size + off) * s.kv_heads + kvh) * s.head_dim;
}

}  // namespace

void reference_attention(const AttnShape& s, int num_seqs, int max_blocks, const float* q, const float* k_cache,
                         const float* v_cache, const int32_t* block_tables, const int32_t* seq_lens, float* out) {
    check(s, num_seqs, max_blocks, seq_lens);
    const int group = s.num_heads / s.kv_heads;
    const double scale = 1.0 / std::sqrt(static_cast<double>(s.head_dim));
    for (int i = 0; i < num_seqs; ++i) {
        const int n = seq_lens[i];
        // gather contiguous K/V: [n][kv_heads][head_dim]
        std::vector<float> K(static_cast<size_t>(n) * s.kv_heads * s.head_dim), V(K.size());
        for (int t = 0; t < n; ++t) {
            int blk = block_tables[static_cast<size_t>(i) * max_blocks + t / s.block_size];
            for (int h = 0; h < s.kv_heads; ++h) {
                size_t src = kv_index(s, blk, t % s.block_size, h);
                size_t dst = (static_cast<size_t>(t) * s.kv_heads + h) * s.head_dim;
                std::copy(k_cache + src, k_cache + src + s.head_dim, K.begin() + static_cast<std::ptrdiff_t>(dst));
                std::copy(v_cache + src, v_cache + src + s.head_dim, V.begin() + static_cast<std::ptrdiff_t>(dst));
            }
        }
        for (int h = 0; h < s.num_heads; ++h) {
            const int kvh = h / group;
            const float* qh = q + (static_cast<size_t>(i) * s.num_heads + h) * s.head_dim;
            std::vector<double> sc(n);
            double mx = -std::numeric_limits<double>::infinity();
            for (int t = 0; t < n; ++t) {
                const float* kt = &K[(static_cast<size_t>(t) * s.kv_heads + kvh) * s.head_dim];
                double d = 0;
                for (int e = 0; e < s.head_dim; ++e) d += static_cast<double>(qh[e]) * kt[e];
                sc[t] = d * scale;
                mx = std::max(mx, sc[t]);
            }
            double sum = 0;
            for (int t = 0; t < n; ++t) sum += (sc[t] = std::exp(sc[t] - mx));
            float* oh = out + (static_cast<size_t>(i) * s.num_heads + h) * s.head_dim;
            for (int e = 0; e < s.head_dim; ++e) {
                double acc = 0;
                for (int t = 0; t < n; ++t)
                    acc += sc[t] * V[(static_cast<size_t>(t) * s.kv_heads + kvh) * s.head_dim + e];
                oh[e] = static_cast<float>(acc / sum);
            }
        }
    }
}

void paged_attention_cpu(const AttnShape& s, int num_seqs, int max_blocks, const float* q, const float* k_cache,
                         const float* v_cache, const int32_t* block_tables, const int32_t* seq_lens, float* out,
                         int partitions) {
    check(s, num_seqs, max_blocks, seq_lens);
    if (partitions < 1) throw std::invalid_argument("partitions < 1");
    const int group = s.num_heads / s.kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(s.head_dim));
    const float neg_inf = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < num_seqs; ++i) {
        const int n = seq_lens[i];
        const int32_t* table = block_tables + static_cast<size_t>(i) * max_blocks;
        for (int h = 0; h < s.num_heads; ++h) {
            const int kvh = h / group;
            const float* qh = q + (static_cast<size_t>(i) * s.num_heads + h) * s.head_dim;
            std::vector<float> m(partitions, neg_inf), l(partitions, 0.f);
            std::vector<float> acc(static_cast<size_t>(partitions) * s.head_dim, 0.f);
            for (int p = 0; p < partitions; ++p)
                for (int t = p; t < n; t += partitions) {  // this "warp" visits tokens p, p+P, ...
                    size_t base = kv_index(s, table[t / s.block_size], t % s.block_size, kvh);
                    float d = 0;
                    for (int e = 0; e < s.head_dim; ++e) d += qh[e] * k_cache[base + e];
                    d *= scale;
                    float m_new = std::max(m[p], d);
                    float corr = std::exp(m[p] - m_new);  // exp(-inf) == 0 on the first token
                    float w = std::exp(d - m_new);
                    l[p] = l[p] * corr + w;
                    for (int e = 0; e < s.head_dim; ++e)
                        acc[static_cast<size_t>(p) * s.head_dim + e] =
                            acc[static_cast<size_t>(p) * s.head_dim + e] * corr + w * v_cache[base + e];
                    m[p] = m_new;
                }
            // merge partitions; empty ones (m = -inf) contribute nothing
            float M = neg_inf;
            for (int p = 0; p < partitions; ++p) M = std::max(M, m[p]);
            float L = 0;
            std::vector<float> o(s.head_dim, 0.f);
            for (int p = 0; p < partitions; ++p) {
                if (m[p] == neg_inf) continue;
                float c = std::exp(m[p] - M);
                L += l[p] * c;
                for (int e = 0; e < s.head_dim; ++e) o[e] += acc[static_cast<size_t>(p) * s.head_dim + e] * c;
            }
            float* oh = out + (static_cast<size_t>(i) * s.num_heads + h) * s.head_dim;
            for (int e = 0; e < s.head_dim; ++e) oh[e] = o[e] / L;
        }
    }
}

}  // namespace kvp
