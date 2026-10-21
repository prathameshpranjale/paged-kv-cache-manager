#pragma once
#include <vector>

#include "kvpage/types.h"

namespace kvp {

// Per-request mapping: logical block index -> physical block id.
// Invariant: blocks.size() == ceil(tokens / block_size).
struct BlockTable {
    std::vector<BlockId> blocks;
    int tokens = 0;
};

}  // namespace kvp
