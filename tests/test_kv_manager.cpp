#include "kvpage/kv_manager.h"
#include "testing.h"

using namespace kvp;

TEST(kv_block_counts_at_boundaries) {
    const int B = 4;
    for (int tokens = 0; tokens <= 3 * B + 1; ++tokens) {
        KVManager kv(16, B);
        CHECK(kv.admit(1, tokens));
        CHECK_EQ(static_cast<int>(kv.table(1).blocks.size()), ceil_div(tokens, B));
        CHECK_EQ(kv.table(1).tokens, tokens);
        CHECK_EQ(kv.free_blocks(), 16 - ceil_div(tokens, B));
        CHECK_EQ(kv.validate(), std::string());
    }
}

TEST(kv_append_allocates_only_at_block_boundary) {
    const int B = 4;
    KVManager kv(8, B);
    CHECK(kv.admit(1, 0));
    for (int t = 1; t <= 3 * B; ++t) {
        int before = kv.free_blocks();
        CHECK(kv.append_token(1));
        bool boundary = (t - 1) % B == 0;  // first token of a new block
        CHECK_EQ(before - kv.free_blocks(), boundary ? 1 : 0);
        CHECK_EQ(static_cast<int>(kv.table(1).blocks.size()), ceil_div(t, B));
    }
}

TEST(kv_only_last_block_partial) {
    KVManager kv(8, 4);
    CHECK(kv.admit(1, 9));  // 2 full blocks + 1 token
    KVStats s = kv.stats();
    CHECK_EQ(s.used_tokens, 9);
    CHECK_EQ(s.allocated_slots, 12);
    CHECK_EQ(s.internal_waste, 3);
}

TEST(kv_free_returns_everything_no_leak) {
    KVManager kv(10, 4);
    CHECK(kv.admit(1, 7));
    CHECK(kv.admit(2, 12));
    for (int i = 0; i < 5; ++i) CHECK(kv.append_token(1));
    kv.free(1);
    kv.free(2);
    CHECK_EQ(kv.free_blocks(), 10);
    CHECK_EQ(kv.stats().requests, 0);
    CHECK_EQ(kv.validate(), std::string());
}

TEST(kv_admit_fails_cleanly_when_full) {
    KVManager kv(3, 4);
    CHECK(kv.admit(1, 8));        // 2 blocks
    CHECK(!kv.can_admit(8));      // needs 2, only 1 free
    CHECK(!kv.admit(2, 8));
    CHECK(!kv.contains(2));       // no side effects
    CHECK_EQ(kv.free_blocks(), 1);
    CHECK(kv.can_admit(4));
    CHECK(kv.admit(2, 4));
    CHECK_EQ(kv.free_blocks(), 0);
}

TEST(kv_append_fails_cleanly_when_full) {
    KVManager kv(2, 4);
    CHECK(kv.admit(1, 8));
    CHECK(!kv.append_token(1));  // needs a third block
    CHECK_EQ(kv.table(1).tokens, 8);
    CHECK_EQ(kv.free_blocks(), 0);
    kv.free(1);
    CHECK(kv.admit(2, 3));
    CHECK(kv.append_token(2));   // fills the last block, no new block
    CHECK(kv.append_token(2));   // needs the second block
    CHECK_EQ(kv.validate(), std::string());
}

TEST(kv_unknown_and_duplicate_requests_throw) {
    KVManager kv(4, 4);
    CHECK_THROWS(kv.append_token(7));
    CHECK_THROWS(kv.free(7));
    CHECK_THROWS(kv.table(7));
    CHECK(kv.admit(1, 4));
    CHECK_THROWS(kv.admit(1, 4));
    CHECK_THROWS(KVManager(4, 0));
}

TEST(kv_validate_catches_corruption) {
    // The invariant checker must actually be able to fail: break a refcount behind
    // the manager's back and expect a complaint.
    KVManager kv(4, 4);
    CHECK(kv.admit(1, 8));
    BlockId b = kv.table(1).blocks[0];
    const_cast<BlockAllocator&>(kv.allocator()).retain(b);
    CHECK(!kv.validate().empty());
}
