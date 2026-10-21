// Correctness test + bandwidth benchmark for the paged decode-attention kernel.
// Block tables come from the real kvp::KVManager; the expected output from kvp::reference_attention
// (gather into contiguous K/V, plain softmax, double precision) computed on the SAME fp16-rounded inputs.
//
// Usage: kvpage_cuda_test [--no-bench]
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "kvpage/kv_manager.h"
#include "kvpage/paged_attention.h"
#include "paged_attention.cuh"

#define CUDA_OK(x)                                                                         \
    do {                                                                                   \
        cudaError_t e_ = (x);                                                              \
        if (e_ != cudaSuccess) {                                                           \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(2);                                                                  \
        }                                                                                  \
    } while (0)

using namespace kvp;

namespace {

constexpr double kTolerance = 1e-2;  // max |diff| / max |reference| (fp16 inputs and output; state in README)

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
    uint32_t next() { s = s * 6364136223846793005ull + 1442695040888963407ull; return static_cast<uint32_t>(s >> 33); }
    float uniform() { return static_cast<float>(next() % 20001) / 10000.f - 1.f; }
};

struct Case {
    int nh, kvh, hd, B;
    std::vector<int> lens;
    bool share;  // later sequences share the first sequence's full blocks
};

// Fills inputs, runs the kernel with `warps` warps, returns max |got - ref| / max |ref|.
double run_case(const Case& c, int warps, uint64_t seed) {
    Rng rng(seed);
    AttnShape s{c.nh, c.kvh, c.hd, c.B};
    const int ns = static_cast<int>(c.lens.size());
    int max_blocks = 0;
    for (int n : c.lens) max_blocks = std::max(max_blocks, ceil_div(n, c.B));
    KVManager kv(ns * max_blocks + 4, c.B);
    for (int i = 0; i < ns; ++i) {
        bool ok = (c.share && i > 0 && c.lens[i] >= c.B) ? kv.admit_with_prefix(i, 0, std::min(c.lens[0], c.lens[i]), c.lens[i])
                                                           : kv.admit(i, c.lens[i]);
        if (!ok) { std::fprintf(stderr, "KVManager admit failed\n"); std::exit(2); }
    }
    const size_t kv_elems = static_cast<size_t>(kv.total_blocks()) * c.B * c.kvh * c.hd;
    const size_t q_elems = static_cast<size_t>(ns) * c.nh * c.hd;
    std::vector<__half> hk(kv_elems), hv(kv_elems), hq(q_elems);
    std::vector<float> fk(kv_elems), fv(kv_elems), fq(q_elems);  // fp16-rounded copies for the reference
    for (size_t i = 0; i < kv_elems; ++i) { hk[i] = __float2half(rng.uniform()); fk[i] = __half2float(hk[i]); }
    for (size_t i = 0; i < kv_elems; ++i) { hv[i] = __float2half(rng.uniform()); fv[i] = __half2float(hv[i]); }
    for (size_t i = 0; i < q_elems; ++i) { hq[i] = __float2half(rng.uniform() * 2.f); fq[i] = __half2float(hq[i]); }
    std::vector<int32_t> tables(static_cast<size_t>(ns) * max_blocks, 0);
    for (int i = 0; i < ns; ++i) {
        const auto& b = kv.table(i).blocks;
        for (size_t j = 0; j < b.size(); ++j) tables[static_cast<size_t>(i) * max_blocks + j] = b[j];
    }
    std::vector<int32_t> lens(c.lens.begin(), c.lens.end());
    std::vector<float> ref(q_elems);
    reference_attention(s, ns, max_blocks, fq.data(), fk.data(), fv.data(), tables.data(), lens.data(), ref.data());

    __half *dq, *dk, *dv, *dout;
    int *dt, *dl;
    CUDA_OK(cudaMalloc(&dq, q_elems * sizeof(__half)));
    CUDA_OK(cudaMalloc(&dout, q_elems * sizeof(__half)));
    CUDA_OK(cudaMalloc(&dk, kv_elems * sizeof(__half)));
    CUDA_OK(cudaMalloc(&dv, kv_elems * sizeof(__half)));
    CUDA_OK(cudaMalloc(&dt, tables.size() * sizeof(int)));
    CUDA_OK(cudaMalloc(&dl, lens.size() * sizeof(int)));
    CUDA_OK(cudaMemcpy(dq, hq.data(), q_elems * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dk, hk.data(), kv_elems * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dv, hv.data(), kv_elems * sizeof(__half), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dt, tables.data(), tables.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dl, lens.data(), lens.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemset(dout, 0xFF, q_elems * sizeof(__half)));  // NaN pattern: unwritten outputs are caught
    CUDA_OK(launch_paged_attention(dq, dk, dv, dt, dl, dout, ns, c.nh, c.kvh, c.hd, c.B, max_blocks, warps));
    CUDA_OK(cudaDeviceSynchronize());
    std::vector<__half> hout(q_elems);
    CUDA_OK(cudaMemcpy(hout.data(), dout, q_elems * sizeof(__half), cudaMemcpyDeviceToHost));
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout); cudaFree(dt); cudaFree(dl);

    double max_ref = 1e-9, max_diff = 0;
    for (size_t i = 0; i < q_elems; ++i) max_ref = std::max(max_ref, static_cast<double>(std::fabs(ref[i])));
    for (size_t i = 0; i < q_elems; ++i) {
        double g = __half2float(hout[i]);
        if (!(std::fabs(g) < 1e30)) return 1e9;  // NaN / inf
        max_diff = std::max(max_diff, std::fabs(g - ref[i]));
    }
    return max_diff / max_ref;
}

void benchmark(int num_seqs, int seq_len, int nh, int kvh, int hd, int B, int warps) {
    const int max_blocks = ceil_div(seq_len, B);
    const int num_blocks = num_seqs * max_blocks;
    const size_t kv_elems = static_cast<size_t>(num_blocks) * B * kvh * hd;
    const size_t q_elems = static_cast<size_t>(num_seqs) * nh * hd;
    std::vector<int> tables(static_cast<size_t>(num_seqs) * max_blocks);
    for (int i = 0; i < num_seqs; ++i)
        for (int j = 0; j < max_blocks; ++j)  // scatter blocks: interleave sequences like a real allocator would
            tables[static_cast<size_t>(i) * max_blocks + j] = j * num_seqs + i;
    std::vector<int> lens(num_seqs, seq_len);
    __half *dq, *dk, *dv, *dout;
    int *dt, *dl;
    CUDA_OK(cudaMalloc(&dq, q_elems * 2));
    CUDA_OK(cudaMalloc(&dout, q_elems * 2));
    CUDA_OK(cudaMalloc(&dk, kv_elems * 2));
    CUDA_OK(cudaMalloc(&dv, kv_elems * 2));
    CUDA_OK(cudaMalloc(&dt, tables.size() * 4));
    CUDA_OK(cudaMalloc(&dl, lens.size() * 4));
    CUDA_OK(cudaMemset(dq, 0x3C, q_elems * 2));
    CUDA_OK(cudaMemset(dk, 0x3C, kv_elems * 2));
    CUDA_OK(cudaMemset(dv, 0x3C, kv_elems * 2));
    CUDA_OK(cudaMemcpy(dt, tables.data(), tables.size() * 4, cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(dl, lens.data(), lens.size() * 4, cudaMemcpyHostToDevice));
    for (int i = 0; i < 5; ++i)
        CUDA_OK(launch_paged_attention(dq, dk, dv, dt, dl, dout, num_seqs, nh, kvh, hd, B, max_blocks, warps));
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    const int iters = 50;
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i)
        launch_paged_attention(dq, dk, dv, dt, dl, dout, num_seqs, nh, kvh, hd, B, max_blocks, warps);
    cudaEventRecord(b);
    CUDA_OK(cudaEventSynchronize(b));
    float ms = 0;
    cudaEventElapsedTime(&ms, a, b);
    ms /= iters;
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);
    const double unique_bytes = static_cast<double>(num_seqs) * seq_len * kvh * hd * 2 /*K,V*/ * 2 /*fp16*/;
    const double gbps = unique_bytes / (ms * 1e-3) / 1e9;
    int mem_khz = 0, bus = 0;
    cudaDeviceGetAttribute(&mem_khz, cudaDevAttrMemoryClockRate, 0);
    cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, 0);
    const double peak = 2.0 * mem_khz * 1e3 * (bus / 8.0) / 1e9;  // DDR: 2 transfers per clock
    std::printf("| %d | %d | %d/%d/%d | %d | %d | %.3f | %.1f | %.1f | %.0f%% |\n", num_seqs, seq_len, nh, kvh, hd, B,
                warps, ms, unique_bytes / 1e6, gbps, 100.0 * gbps / peak);
    std::printf("device: %s, memory peak %.0f GB/s (clock %d kHz, bus %d bit)\n", p.name, peak, mem_khz, bus);
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dout); cudaFree(dt); cudaFree(dl);
}

}  // namespace

int main(int argc, char** argv) {
    bool bench = !(argc > 1 && !std::strcmp(argv[1], "--no-bench"));
    int dev = 0;
    CUDA_OK(cudaGetDevice(&dev));
    cudaDeviceProp p;
    CUDA_OK(cudaGetDeviceProperties(&p, dev));
    std::printf("device %d: %s (sm_%d%d)\n", dev, p.name, p.major, p.minor);

    std::vector<Case> cases;
    for (int B : {1, 8, 16, 32}) {
        std::vector<int> lens = {1, B - 1 > 0 ? B - 1 : 1, B, B + 1, 2 * B, 3 * B + 1, 97, 130};
        cases.push_back({8, 2, 64, B, lens, false});
        cases.push_back({8, 2, 128, B, lens, false});
        cases.push_back({32, 8, 128, B, {70, 70, 55, 33, 16}, true});  // shared prefix blocks, Llama-3-8B heads
    }
    cases.push_back({14, 2, 64, 16, {33, 64, 65, 300}, false});  // Qwen2.5-0.5B heads
    cases.push_back({4, 4, 32, 16, {1, 2, 3}, false});           // fewer tokens than warps
    cases.push_back({8, 8, 256, 16, {100, 17}, false});

    int failed = 0;
    std::printf("| case | warps | max rel diff | result |\n|---|---|---|---|\n");
    uint64_t seed = 1;
    for (const Case& c : cases)
        for (int warps : {1, 4, 8}) {
            double d = run_case(c, warps, seed++);
            bool ok = d < kTolerance;
            failed += !ok;
            std::printf("| nh=%d kvh=%d hd=%d B=%d lens=%zu%s | %d | %.2e | %s |\n", c.nh, c.kvh, c.hd, c.B,
                        c.lens.size(), c.share ? " shared" : "", warps, d, ok ? "PASS" : "FAIL");
        }
    std::printf("\n%s (%d failing, tolerance %.0e relative to max |reference|)\n", failed ? "FAILED" : "ALL PASSED",
                failed, kTolerance);
    if (bench && !failed) {
        std::printf("\n| seqs | seq_len | heads q/kv/dim | block | warps | ms/launch | unique KV MB | GB/s | %% of peak |\n"
                    "|---|---|---|---|---|---|---|---|---|\n");
        benchmark(64, 2048, 32, 8, 128, 16, 4);   // Llama-3-8B decode, 64 sequences of 2048 tokens
        benchmark(64, 2048, 32, 8, 128, 16, 8);
        benchmark(16, 4096, 32, 8, 128, 16, 4);
        benchmark(64, 2048, 14, 2, 64, 16, 4);    // Qwen2.5-0.5B shapes
    }
    return failed ? 1 : 0;
}
