#include <set>

#include "kvpage/block_allocator.h"
#include "testing.h"

using namespace kvp;

TEST(allocator_exhaustion_and_reuse) {
    BlockAllocator a(4);
    CHECK_EQ(a.free_blocks(), 4);
    std::set<BlockId> got;
    for (int i = 0; i < 4; ++i) {
        BlockId b = a.allocate();
        CHECK(b != kNoBlock);
        got.insert(b);
        CHECK_EQ(a.refcount(b), 1);
    }
    CHECK_EQ(got.size(), 4u);  // all distinct
    CHECK_EQ(a.free_blocks(), 0);
    CHECK_EQ(a.allocate(), kNoBlock);  // exhausted
    CHECK_EQ(a.validate(), std::string());
    a.release(2);
    CHECK_EQ(a.free_blocks(), 1);
    CHECK_EQ(a.allocate(), 2);  // the released block is reused
    for (BlockId b = 0; b < 4; ++b) a.release(b);
    CHECK_EQ(a.free_blocks(), 4);
    CHECK_EQ(a.validate(), std::string());
}

TEST(allocator_allocates_lowest_ids_first) {
    BlockAllocator a(3);
    CHECK_EQ(a.allocate(), 0);
    CHECK_EQ(a.allocate(), 1);
    CHECK_EQ(a.allocate(), 2);
}

TEST(allocator_refcount_sharing) {
    BlockAllocator a(2);
    BlockId b = a.allocate();
    a.retain(b);
    a.retain(b);
    CHECK_EQ(a.refcount(b), 3);
    a.release(b);
    a.release(b);
    CHECK_EQ(a.free_blocks(), 1);  // still referenced once
    CHECK_EQ(a.refcount(b), 1);
    a.release(b);
    CHECK_EQ(a.free_blocks(), 2);
    CHECK_EQ(a.refcount(b), 0);
    CHECK_EQ(a.validate(), std::string());
}

TEST(allocator_misuse_throws) {
    BlockAllocator a(2);
    CHECK_THROWS(a.release(0));  // free block
    CHECK_THROWS(a.retain(1));   // free block
    CHECK_THROWS(a.refcount(2));
    CHECK_THROWS(a.release(-1));
    BlockId b = a.allocate();
    a.release(b);
    CHECK_THROWS(a.release(b));  // double free
    CHECK_THROWS(BlockAllocator(-1));
}

TEST(allocator_zero_blocks) {
    BlockAllocator a(0);
    CHECK_EQ(a.allocate(), kNoBlock);
    CHECK_EQ(a.validate(), std::string());
}
