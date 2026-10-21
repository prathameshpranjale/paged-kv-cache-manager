#pragma once
#include <string>
#include <vector>

#include "kvpage/types.h"

namespace kvp {

// Fixed pool of equally sized blocks with a free list and reference counts.
// Misuse (bad id, retain/release of a free block) throws std::logic_error.
class BlockAllocator {
public:
    explicit BlockAllocator(int num_blocks);

    BlockId allocate();           // kNoBlock if none free; sets refcount = 1
    void retain(BlockId b);       // refcount++ (sharing)
    void release(BlockId b);      // refcount--; back to the free list at 0
    int refcount(BlockId b) const;
    int free_blocks() const { return static_cast<int>(free_.size()); }
    int total_blocks() const { return static_cast<int>(refs_.size()); }
    int referenced_blocks() const { return total_blocks() - free_blocks(); }

    // Empty string if consistent, otherwise a description of the first problem.
    std::string validate() const;

private:
    void check_id(BlockId b) const;

    std::vector<int> refs_;
    std::vector<BlockId> free_;  // used as a stack; lowest id is allocated first
};

}  // namespace kvp
