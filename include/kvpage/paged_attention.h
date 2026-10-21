#pragma once
#include <cstdint>

namespace kvp {

// Decode attention over a paged KV cache: ONE query token per sequence.
//
// Layouts (row-major, contiguous):
//   q, out        [num_seqs][num_heads][head_dim]
//   k_cache,
//   v_cache       [num_blocks][block_size][kv_heads][head_dim]
//   block_tables  [num_seqs][max_blocks]   (int32 physical block ids; entries past a sequence are ignored)
//   seq_lens      [num_seqs]               (tokens in the context, >= 1)
// Grouped-query attention: query head h uses kv head h / (num_heads / kv_heads). scale = 1/sqrt(head_dim).
// The same layout is used by cuda/paged_attention.cu so both can be compared directly.
struct AttnShape {
    int num_heads = 8;
    int kv_heads = 2;
    int head_dim = 64;
    int block_size = 16;
};

// Reference: gather each sequence's K/V into contiguous buffers through its block table, then plain
// softmax attention in double precision. This is what the spec calls the "plain reference".
void reference_attention(const AttnShape& s, int num_seqs, int max_blocks, const float* q,
                         const float* k_cache, const float* v_cache, const int32_t* block_tables,
                         const int32_t* seq_lens, float* out);

// Same math as the GPU kernel: K/V read in place through the block table, single pass, online softmax.
// The context is split over `partitions` interleaved token groups (the GPU's warps), each keeping its own
// running (max, sum, accumulator), merged at the end. Float accumulation. Lets us validate the algorithm
// (including empty partitions, partial last blocks, shared blocks) on CPU before any GPU run.
void paged_attention_cpu(const AttnShape& s, int num_seqs, int max_blocks, const float* q,
                         const float* k_cache, const float* v_cache, const int32_t* block_tables,
                         const int32_t* seq_lens, float* out, int partitions);

}  // namespace kvp
