// Version 1: simple and verifiable. Not yet optimized (see NOTES.md / cuda/GPU_CHECKLIST.md for next steps).
#include <cmath>

#include "paged_attention.cuh"

namespace {

constexpr int kMaxWarps = 8;

template <int HD>
__global__ void paged_attention_kernel(const __half* __restrict__ q, const __half* __restrict__ k_cache,
                                       const __half* __restrict__ v_cache, const int* __restrict__ block_tables,
                                       const int* __restrict__ seq_lens, __half* __restrict__ out, int num_heads,
                                       int kv_heads, int block_size, int max_blocks, float scale) {
    static_assert(HD % 32 == 0, "head_dim must be a multiple of 32");
    constexpr int EPL = HD / 32;  // elements per lane: lane owns dims [lane*EPL, lane*EPL + EPL)
    const int head = blockIdx.x;
    const int seq = blockIdx.y;
    const int kvh = head / (num_heads / kv_heads);
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int nwarps = blockDim.x / 32;

    const int n = seq_lens[seq];
    const int* table = block_tables + static_cast<size_t>(seq) * max_blocks;

    float qf[EPL];
    const __half* qh = q + (static_cast<size_t>(seq) * num_heads + head) * HD + lane * EPL;
#pragma unroll
    for (int i = 0; i < EPL; ++i) qf[i] = __half2float(qh[i]);

    float m = -INFINITY, l = 0.f;
    float acc[EPL];
#pragma unroll
    for (int i = 0; i < EPL; ++i) acc[i] = 0.f;

    // This warp visits tokens warp, warp + nwarps, ... (all lanes cooperate on one token at a time).
    for (int t = warp; t < n; t += nwarps) {
        const int block = table[t / block_size];
        const size_t base =
            ((static_cast<size_t>(block) * block_size + (t % block_size)) * kv_heads + kvh) * HD + lane * EPL;
        float d = 0.f;
#pragma unroll
        for (int i = 0; i < EPL; ++i) d += qf[i] * __half2float(k_cache[base + i]);
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) d += __shfl_xor_sync(0xffffffffu, d, off);  // every lane gets the dot
        d *= scale;
        const float m_new = fmaxf(m, d);
        const float corr = expf(m - m_new);  // m = -inf on the first token: expf(-inf) = 0
        const float w = expf(d - m_new);
        l = l * corr + w;
#pragma unroll
        for (int i = 0; i < EPL; ++i) acc[i] = acc[i] * corr + w * __half2float(v_cache[base + i]);
        m = m_new;
    }

    // Merge the warps' partial results.
    __shared__ float s_m[kMaxWarps], s_l[kMaxWarps];
    __shared__ float s_acc[kMaxWarps][HD];
    if (lane == 0) {
        s_m[warp] = m;
        s_l[warp] = l;
    }
#pragma unroll
    for (int i = 0; i < EPL; ++i) s_acc[warp][lane * EPL + i] = acc[i];
    __syncthreads();

    if (warp == 0) {
        float M = -INFINITY;
        for (int w = 0; w < nwarps; ++w) M = fmaxf(M, s_m[w]);
        float L = 0.f;
        float o[EPL];
#pragma unroll
        for (int i = 0; i < EPL; ++i) o[i] = 0.f;
        for (int w = 0; w < nwarps; ++w) {
            if (s_m[w] == -INFINITY) continue;  // this warp saw no tokens
            const float c = expf(s_m[w] - M);
            L += s_l[w] * c;
#pragma unroll
            for (int i = 0; i < EPL; ++i) o[i] += s_acc[w][lane * EPL + i] * c;
        }
        __half* oh = out + (static_cast<size_t>(seq) * num_heads + head) * HD + lane * EPL;
#pragma unroll
        for (int i = 0; i < EPL; ++i) oh[i] = __float2half(o[i] / L);
    }
}

template <int HD>
void launch(const __half* q, const __half* k, const __half* v, const int* tables, const int* lens, __half* out,
            int num_seqs, int num_heads, int kv_heads, int block_size, int max_blocks, int warps,
            cudaStream_t stream) {
    dim3 grid(num_heads, num_seqs);
    paged_attention_kernel<HD><<<grid, warps * 32, 0, stream>>>(q, k, v, tables, lens, out, num_heads, kv_heads,
                                                               block_size, max_blocks,
                                                               1.0f / sqrtf(static_cast<float>(HD)));
}

}  // namespace

cudaError_t launch_paged_attention(const __half* q, const __half* k_cache, const __half* v_cache,
                                   const int* block_tables, const int* seq_lens, __half* out, int num_seqs,
                                   int num_heads, int kv_heads, int head_dim, int block_size, int max_blocks,
                                   int warps, cudaStream_t stream) {
    if (warps < 1 || warps > kMaxWarps || kv_heads <= 0 || num_heads % kv_heads != 0 || num_seqs <= 0)
        return cudaErrorInvalidValue;
    switch (head_dim) {
        case 32:  launch<32>(q, k_cache, v_cache, block_tables, seq_lens, out, num_seqs, num_heads, kv_heads, block_size, max_blocks, warps, stream); break;
        case 64:  launch<64>(q, k_cache, v_cache, block_tables, seq_lens, out, num_seqs, num_heads, kv_heads, block_size, max_blocks, warps, stream); break;
        case 96:  launch<96>(q, k_cache, v_cache, block_tables, seq_lens, out, num_seqs, num_heads, kv_heads, block_size, max_blocks, warps, stream); break;
        case 128: launch<128>(q, k_cache, v_cache, block_tables, seq_lens, out, num_seqs, num_heads, kv_heads, block_size, max_blocks, warps, stream); break;
        case 256: launch<256>(q, k_cache, v_cache, block_tables, seq_lens, out, num_seqs, num_heads, kv_heads, block_size, max_blocks, warps, stream); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}
