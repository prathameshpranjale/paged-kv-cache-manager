#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "kvpage/block_allocator.h"
#include "kvpage/block_table.h"

namespace kvp {

struct KVStats {
    int requests = 0;
    int64_t used_tokens = 0;       // token slots actually filled, shared blocks counted once
    int64_t allocated_blocks = 0;  // physical blocks in use
    int64_t allocated_slots = 0;   // allocated_blocks * block_size
    int64_t internal_waste = 0;    // allocated_slots - used_tokens (unused tail slots)
    int64_t logical_blocks = 0;    // sum of block-table sizes
    int64_t blocks_saved = 0;      // logical_blocks - allocated_blocks (sharing)
    int64_t cow_copies = 0;        // cumulative
    int64_t cached_blocks = 0;     // blocks held by the persistent prefix cache
    int64_t cache_evictions = 0;   // cumulative
};

class KVManager {
public:
    KVManager(int num_blocks, int block_size);

    // New blocks an admission would take: ceil(prompt/B) minus the full blocks
    // that would be shared with `parent` (pass kNoRequest for none).
    int blocks_needed_to_admit(RequestId parent, int shared_tokens, int prompt_tokens) const;
    bool can_admit(int prompt_tokens) const;

    bool admit(RequestId r, int prompt_tokens);  // ceil(prompt/B) private blocks
    // Share the parent's completely filled prefix blocks (never a partial one),
    // allocate the rest privately. Nothing changes on failure.
    bool admit_with_prefix(RequestId r, RequestId parent, int shared_tokens, int prompt_tokens);
    // Share ALL of the parent's blocks, including a partial last one (like
    // parallel sampling). The first append into that shared partial block copies it.
    bool fork(RequestId r, RequestId parent);
    // Add one token. May allocate a block, or copy-on-write a shared partial last
    // block. Returns false (state unchanged) if no block is available.
    bool append_token(RequestId r);
    void free(RequestId r);

    bool contains(RequestId r) const { return tables_.count(r) != 0; }
    const BlockTable& table(RequestId r) const;
    KVStats stats() const;            // O(1): maintained incrementally
    KVStats recompute_stats() const;  // O(total blocks): from scratch; validate() checks both agree

    int block_size() const { return block_size_; }
    int free_blocks() const { return alloc_.free_blocks(); }
    int total_blocks() const { return alloc_.total_blocks(); }
    const BlockAllocator& allocator() const { return alloc_; }

    // ---- persistent prefix cache: full prefix blocks that outlive the request that created them ----
    // The cache holds one reference per block. Entries are reclaimed (least recently used first)
    // whenever an allocation would otherwise fail, so cached blocks never block admission.
    bool cache_prefix(int key, RequestId owner, int tokens);  // false if the key exists or < 1 full block
    bool has_cached_prefix(int key) const { return cache_.count(key) != 0; }
    int cached_prefix_tokens(int key) const;                  // 0 if absent
    int blocks_needed_to_admit_cached(int key, int shared_tokens, int prompt_tokens) const;
    bool admit_with_cached_prefix(RequestId r, int key, int shared_tokens, int prompt_tokens);
    // Blocks that evicting the whole cache would free right now (held by the cache only). When `key` is given,
    // the first `shared_blocks` blocks of that entry are excluded: a request about to share them keeps them alive.
    int evictable_blocks(int key = -1, int shared_blocks = 0) const;
    // Full blocks an admission with prefix `key` would share (clamped like admit_with_cached_prefix does).
    int shared_blocks_with_cached(int key, int shared_tokens, int prompt_tokens) const;
    int available_blocks() const { return free_blocks() + evictable_blocks(); }
    void clear_cache();

    // Empty string if every invariant of section 7 holds.
    std::string validate() const;
    // When on, every mutating call validates and throws std::logic_error on a
    // violation. Defaults to on in builds without NDEBUG.
    void set_check_each_op(bool on) { check_each_op_ = on; }

private:
    struct CacheEntry {
        std::vector<BlockId> blocks;
        int tokens = 0;
        int64_t last_used = 0;
    };
    void after_op() const;
    void release_block(BlockId b);
    void evict_entry(std::map<int, CacheEntry>::iterator it);
    bool ensure_free(int n);
    int cache_only_blocks() const;
    std::unordered_map<BlockId, int> cache_ref_counts() const;

    int block_size_;
    BlockAllocator alloc_;
    std::map<RequestId, BlockTable> tables_;  // ordered: deterministic iteration
    int64_t cow_copies_ = 0;
    std::map<int, CacheEntry> cache_;
    int64_t clock_ = 0;
    int64_t cache_evictions_ = 0;
    std::vector<int> fill_;        // filled token slots of each physical block (0 when free)
    int64_t used_tokens_ = 0;      // sum of fill_
    int64_t logical_blocks_ = 0;   // sum of block-table sizes
#ifdef NDEBUG
    bool check_each_op_ = false;
#else
    bool check_each_op_ = true;
#endif
};

}  // namespace kvp
