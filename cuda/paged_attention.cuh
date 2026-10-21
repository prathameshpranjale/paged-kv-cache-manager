#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// Decode attention over a paged KV cache (one query token per sequence), fp16 data, fp32 math.
// Layouts are identical to include/kvpage/paged_attention.h:
//   q, out        [num_seqs][num_heads][head_dim]                  __half
//   k_cache,
//   v_cache       [num_blocks][block_size][kv_heads][head_dim]     __half
//   block_tables  [num_seqs][max_blocks]                           int32
//   seq_lens      [num_seqs]                                       int32 (>= 1)
// head_dim must be one of 32, 64, 96, 128, 256. num_heads % kv_heads == 0.
// One thread block per (query head, sequence); `warps` (1..8) warps split the context's tokens
// round-robin, each with its own online-softmax state, merged through shared memory (this is exactly
// what kvp::paged_attention_cpu(partitions = warps) models on the CPU).
cudaError_t launch_paged_attention(const __half* q, const __half* k_cache, const __half* v_cache,
                                   const int* block_tables, const int* seq_lens, __half* out, int num_seqs,
                                   int num_heads, int kv_heads, int head_dim, int block_size, int max_blocks,
                                   int warps, cudaStream_t stream = 0);
