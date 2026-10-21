#include "kvpage/block_allocator.h"

#include <stdexcept>

namespace kvp {

BlockAllocator::BlockAllocator(int num_blocks) {
    if (num_blocks < 0) throw std::invalid_argument("num_blocks < 0");
    refs_.assign(num_blocks, 0);
    free_.reserve(num_blocks);
    for (int b = num_blocks - 1; b >= 0; --b) free_.push_back(b);  // pop gives 0, 1, 2, ...
}

void BlockAllocator::check_id(BlockId b) const {
    if (b < 0 || b >= total_blocks()) throw std::logic_error("block id out of range");
}

BlockId BlockAllocator::allocate() {
    if (free_.empty()) return kNoBlock;
    BlockId b = free_.back();
    free_.pop_back();
    refs_[b] = 1;
    return b;
}

void BlockAllocator::retain(BlockId b) {
    check_id(b);
    if (refs_[b] <= 0) throw std::logic_error("retain of a free block");
    ++refs_[b];
}

void BlockAllocator::release(BlockId b) {
    check_id(b);
    if (refs_[b] <= 0) throw std::logic_error("release of a free block");
    if (--refs_[b] == 0) free_.push_back(b);
}

int BlockAllocator::refcount(BlockId b) const {
    check_id(b);
    return refs_[b];
}

std::string BlockAllocator::validate() const {
    std::vector<char> on_free(refs_.size(), 0);
    for (BlockId b : free_) {
        if (b < 0 || b >= total_blocks()) return "free list holds an out-of-range id";
        if (on_free[b]) return "block " + std::to_string(b) + " is on the free list twice";
        on_free[b] = 1;
        if (refs_[b] != 0) return "block " + std::to_string(b) + " is free but has refcount > 0";
    }
    int referenced = 0;
    for (int b = 0; b < total_blocks(); ++b) {
        if (refs_[b] < 0) return "negative refcount on block " + std::to_string(b);
        if (refs_[b] == 0 && !on_free[b])
            return "block " + std::to_string(b) + " has refcount 0 but is not on the free list";
        if (refs_[b] > 0) ++referenced;
    }
    if (free_blocks() + referenced != total_blocks()) return "free + referenced != total";
    return "";
}

}  // namespace kvp
