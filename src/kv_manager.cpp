#include "kvpage/kv_manager.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>

namespace kvp {

KVManager::KVManager(int num_blocks, int block_size) : block_size_(block_size), alloc_(num_blocks) {
    if (block_size <= 0) throw std::invalid_argument("block_size must be > 0");
    fill_.assign(static_cast<size_t>(num_blocks), 0);
}

const BlockTable& KVManager::table(RequestId r) const {
    auto it = tables_.find(r);
    if (it == tables_.end()) throw std::logic_error("unknown request " + std::to_string(r));
    return it->second;
}

void KVManager::after_op() const {
    if (!check_each_op_) return;
    std::string err = validate();
    if (!err.empty()) throw std::logic_error("invariant violated: " + err);
}

// Drop one reference; a block that becomes free no longer counts any filled slots.
void KVManager::release_block(BlockId b) {
    alloc_.release(b);
    if (alloc_.refcount(b) == 0) {
        used_tokens_ -= fill_[b];
        fill_[b] = 0;
    }
}

// ---------------------------------------------------------------- persistent prefix cache

void KVManager::evict_entry(std::map<int, CacheEntry>::iterator it) {
    for (BlockId b : it->second.blocks) release_block(b);
    cache_.erase(it);
    ++cache_evictions_;
}

// Number of cache references on each block (a block can be held by several entries).
std::unordered_map<BlockId, int> KVManager::cache_ref_counts() const {
    std::unordered_map<BlockId, int> refs;
    for (const auto& [key, e] : cache_)
        for (BlockId b : e.blocks) ++refs[b];
    return refs;
}

int KVManager::shared_blocks_with_cached(int key, int shared_tokens, int prompt_tokens) const {
    auto it = cache_.find(key);
    if (it == cache_.end()) return 0;
    return std::max(0, std::min({shared_tokens, it->second.tokens, prompt_tokens})) / block_size_;
}

// Blocks that evicting every cache entry would free: held by the cache only (no request table).
// A requester that is about to share the first `shared_blocks` blocks of entry `key` keeps those alive.
int KVManager::evictable_blocks(int key, int shared_blocks) const {
    auto refs = cache_ref_counts();
    if (key >= 0) {
        auto it = cache_.find(key);
        if (it != cache_.end())
            for (int i = 0; i < shared_blocks && i < static_cast<int>(it->second.blocks.size()); ++i)
                refs.erase(it->second.blocks[i]);
    }
    int n = 0;
    for (const auto& [b, c] : refs)
        if (alloc_.refcount(b) == c) ++n;
    return n;
}

// Evict least-recently-used entries until `n` blocks are free. An entry is eligible when it holds at least
// one block that no request references; evicting all eligible entries frees exactly evictable_blocks().
bool KVManager::ensure_free(int n) {
    while (alloc_.free_blocks() < n) {
        auto refs = cache_ref_counts();
        auto victim = cache_.end();
        for (auto it = cache_.begin(); it != cache_.end(); ++it) {
            bool eligible = std::any_of(it->second.blocks.begin(), it->second.blocks.end(),
                                        [&](BlockId b) { return alloc_.refcount(b) == refs[b]; });
            if (eligible && (victim == cache_.end() || it->second.last_used < victim->second.last_used)) victim = it;
        }
        if (victim == cache_.end()) return false;
        evict_entry(victim);
    }
    return true;
}

bool KVManager::cache_prefix(int key, RequestId owner, int tokens) {
    if (cache_.count(key)) return false;
    const BlockTable& t = table(owner);
    int k = std::max(0, std::min(tokens, t.tokens)) / block_size_;  // only completely filled blocks
    if (k == 0) return false;
    CacheEntry e;
    e.blocks.assign(t.blocks.begin(), t.blocks.begin() + k);
    for (BlockId b : e.blocks) alloc_.retain(b);
    e.tokens = k * block_size_;
    e.last_used = ++clock_;
    cache_.emplace(key, std::move(e));
    after_op();
    return true;
}

int KVManager::cached_prefix_tokens(int key) const {
    auto it = cache_.find(key);
    return it == cache_.end() ? 0 : it->second.tokens;
}

int KVManager::blocks_needed_to_admit_cached(int key, int shared_tokens, int prompt_tokens) const {
    return ceil_div(prompt_tokens, block_size_) - shared_blocks_with_cached(key, shared_tokens, prompt_tokens);
}

bool KVManager::admit_with_cached_prefix(RequestId r, int key, int shared_tokens, int prompt_tokens) {
    if (tables_.count(r)) throw std::logic_error("request already admitted");
    if (prompt_tokens < 0) throw std::invalid_argument("negative prompt");
    auto it = cache_.find(key);
    BlockTable t;
    t.tokens = prompt_tokens;
    int shared_blocks = 0;
    if (it != cache_.end()) {
        it->second.last_used = ++clock_;
        shared_blocks = shared_blocks_with_cached(key, shared_tokens, prompt_tokens);
        for (int i = 0; i < shared_blocks; ++i) {  // retain first so eviction below cannot take them
            alloc_.retain(it->second.blocks[i]);
            t.blocks.push_back(it->second.blocks[i]);
        }
    }
    int need = ceil_div(prompt_tokens, block_size_) - shared_blocks;
    if (!ensure_free(need)) {
        for (BlockId b : t.blocks) release_block(b);
        return false;
    }
    for (int i = 0; i < need; ++i) {
        BlockId b = alloc_.allocate();
        int f = std::min(block_size_, prompt_tokens - (shared_blocks + i) * block_size_);
        fill_[b] = f;
        used_tokens_ += f;
        t.blocks.push_back(b);
    }
    logical_blocks_ += static_cast<int64_t>(t.blocks.size());
    tables_.emplace(r, std::move(t));
    after_op();
    return true;
}

void KVManager::clear_cache() {
    while (!cache_.empty()) evict_entry(cache_.begin());
    cache_evictions_ = 0;
    after_op();
}

// ---------------------------------------------------------------- requests

int KVManager::blocks_needed_to_admit(RequestId parent, int shared_tokens, int prompt_tokens) const {
    int shared_blocks = 0;
    if (parent != kNoRequest) {
        int s = std::min({shared_tokens, table(parent).tokens, prompt_tokens});
        shared_blocks = std::max(s, 0) / block_size_;  // only completely filled blocks
    }
    return ceil_div(prompt_tokens, block_size_) - shared_blocks;
}

bool KVManager::can_admit(int prompt_tokens) const {
    return available_blocks() >= blocks_needed_to_admit(kNoRequest, 0, prompt_tokens);
}

bool KVManager::admit(RequestId r, int prompt_tokens) {
    return admit_with_prefix(r, kNoRequest, 0, prompt_tokens);
}

bool KVManager::admit_with_prefix(RequestId r, RequestId parent, int shared_tokens, int prompt_tokens) {
    if (tables_.count(r)) throw std::logic_error("request already admitted");
    if (prompt_tokens < 0) throw std::invalid_argument("negative prompt");
    int need = blocks_needed_to_admit(parent, shared_tokens, prompt_tokens);
    int shared_blocks = ceil_div(prompt_tokens, block_size_) - need;
    BlockTable t;
    t.tokens = prompt_tokens;
    for (int i = 0; i < shared_blocks; ++i) {  // retain first: reclaiming cache below must not free them
        BlockId b = table(parent).blocks[i];
        alloc_.retain(b);
        t.blocks.push_back(b);
    }
    if (!ensure_free(need)) {
        for (BlockId b : t.blocks) release_block(b);
        return false;
    }
    for (int i = 0; i < need; ++i) {
        BlockId b = alloc_.allocate();  // cannot fail: ensured
        int f = std::min(block_size_, prompt_tokens - (shared_blocks + i) * block_size_);
        fill_[b] = f;
        used_tokens_ += f;
        t.blocks.push_back(b);
    }
    logical_blocks_ += static_cast<int64_t>(t.blocks.size());
    tables_.emplace(r, std::move(t));
    after_op();
    return true;
}

bool KVManager::fork(RequestId r, RequestId parent) {
    if (tables_.count(r)) throw std::logic_error("request already admitted");
    BlockTable t = table(parent);  // copy
    for (BlockId b : t.blocks) alloc_.retain(b);
    logical_blocks_ += static_cast<int64_t>(t.blocks.size());
    tables_.emplace(r, std::move(t));
    after_op();
    return true;
}

bool KVManager::append_token(RequestId r) {
    auto it = tables_.find(r);
    if (it == tables_.end()) throw std::logic_error("unknown request " + std::to_string(r));
    BlockTable& t = it->second;
    const bool partial = t.tokens % block_size_ != 0;
    // A new block is needed when the last one is full (or absent) or shared (copy-on-write). Reclaim the
    // cache first; that can also drop the last block's refcount, so decide the case afterwards.
    if ((!partial || alloc_.refcount(t.blocks.back()) > 1) && alloc_.free_blocks() < 1) ensure_free(1);
    if (!partial) {
        BlockId b = alloc_.allocate();
        if (b == kNoBlock) return false;
        t.blocks.push_back(b);
        fill_[b] = 1;
        ++used_tokens_;
        ++logical_blocks_;
    } else if (alloc_.refcount(t.blocks.back()) > 1) {
        // Partial last block shared with another request: copy-on-write.
        BlockId b = alloc_.allocate();
        if (b == kNoBlock) return false;
        release_block(t.blocks.back());  // other holders keep the old block (and its fill)
        t.blocks.back() = b;
        fill_[b] = t.tokens % block_size_ + 1;
        used_tokens_ += fill_[b];
        ++cow_copies_;
    } else {
        ++fill_[t.blocks.back()];
        ++used_tokens_;
    }
    ++t.tokens;
    after_op();
    return true;
}

void KVManager::free(RequestId r) {
    auto it = tables_.find(r);
    if (it == tables_.end()) throw std::logic_error("unknown request " + std::to_string(r));
    for (BlockId b : it->second.blocks) release_block(b);
    logical_blocks_ -= static_cast<int64_t>(it->second.blocks.size());
    tables_.erase(it);
    after_op();
}

// ---------------------------------------------------------------- statistics and validation

// Distinct blocks referenced by the cache but by no request table (they do not count as "saved by sharing").
int KVManager::cache_only_blocks() const {
    if (cache_.empty()) return 0;
    return evictable_blocks();
}

KVStats KVManager::stats() const {
    KVStats s;
    s.requests = static_cast<int>(tables_.size());
    s.used_tokens = used_tokens_;
    s.logical_blocks = logical_blocks_;
    s.allocated_blocks = alloc_.referenced_blocks();
    s.allocated_slots = s.allocated_blocks * block_size_;
    s.internal_waste = s.allocated_slots - s.used_tokens;
    s.blocks_saved = s.logical_blocks - (s.allocated_blocks - cache_only_blocks());
    s.cow_copies = cow_copies_;
    s.cache_evictions = cache_evictions_;
    for (const auto& [key, e] : cache_) s.cached_blocks += static_cast<int64_t>(e.blocks.size());
    return s;
}

KVStats KVManager::recompute_stats() const {
    KVStats s;
    s.requests = static_cast<int>(tables_.size());
    std::vector<int> fill(alloc_.total_blocks(), 0);
    for (const auto& [id, t] : tables_) {
        s.logical_blocks += static_cast<int64_t>(t.blocks.size());
        for (size_t i = 0; i < t.blocks.size(); ++i) {
            int f = std::min(block_size_, t.tokens - static_cast<int>(i) * block_size_);
            fill[t.blocks[i]] = std::max(fill[t.blocks[i]], f);
        }
    }
    for (const auto& [key, e] : cache_)
        for (BlockId b : e.blocks) {
            fill[b] = std::max(fill[b], block_size_);  // cached blocks are completely filled
            ++s.cached_blocks;
        }
    for (int f : fill) s.used_tokens += f;
    s.allocated_blocks = alloc_.referenced_blocks();
    s.allocated_slots = s.allocated_blocks * block_size_;
    s.internal_waste = s.allocated_slots - s.used_tokens;
    s.blocks_saved = s.logical_blocks - (s.allocated_blocks - cache_only_blocks());
    s.cow_copies = cow_copies_;
    s.cache_evictions = cache_evictions_;
    return s;
}

std::string KVManager::validate() const {
    std::string err = alloc_.validate();
    if (!err.empty()) return err;
    std::vector<int> refs(alloc_.total_blocks(), 0);
    int64_t entries = 0;
    for (const auto& [id, t] : tables_) {
        if (static_cast<int>(t.blocks.size()) != ceil_div(t.tokens, block_size_))
            return "request " + std::to_string(id) + ": block-table size != ceil(tokens/B)";
        for (BlockId b : t.blocks) {
            if (b < 0 || b >= alloc_.total_blocks()) return "table holds an out-of-range block";
            ++refs[b];
            ++entries;
        }
    }
    for (const auto& [key, e] : cache_) {
        if (e.tokens != static_cast<int>(e.blocks.size()) * block_size_)
            return "cache entry " + std::to_string(key) + ": tokens != blocks * block_size";
        for (BlockId b : e.blocks) {
            if (b < 0 || b >= alloc_.total_blocks()) return "cache holds an out-of-range block";
            if (fill_[b] != block_size_) return "cached block is not completely filled";
            ++refs[b];
            ++entries;
        }
    }
    int64_t sum = 0;
    for (int b = 0; b < alloc_.total_blocks(); ++b) {
        if (refs[b] != alloc_.refcount(b))
            return "block " + std::to_string(b) + ": refcount != number of table/cache references";
        sum += alloc_.refcount(b);
    }
    if (sum != entries) return "sum of refcounts != number of block-table and cache entries";
    KVStats a = stats(), b = recompute_stats();
    if (a.used_tokens != b.used_tokens || a.logical_blocks != b.logical_blocks ||
        a.blocks_saved != b.blocks_saved || a.internal_waste != b.internal_waste || a.cached_blocks != b.cached_blocks)
        return "incremental stats differ from recomputed stats";
    return "";
}

}  // namespace kvp
