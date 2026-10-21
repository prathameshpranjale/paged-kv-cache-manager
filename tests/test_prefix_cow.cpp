#include "kvpage/kv_manager.h"
#include "testing.h"

using namespace kvp;

TEST(prefix_shares_only_full_blocks) {
    const int B = 16;
    KVManager kv(20, B);
    CHECK(kv.admit(1, 40));  // blocks: 16 + 16 + 8 (partial)
    CHECK_EQ(kv.blocks_needed_to_admit(1, 40, 60), ceil_div(60, B) - 2);
    CHECK(kv.admit_with_prefix(2, 1, 40, 60));  // shares 2 full blocks, never the partial one
    const auto& p = kv.table(1);
    const auto& c = kv.table(2);
    CHECK_EQ(c.blocks[0], p.blocks[0]);
    CHECK_EQ(c.blocks[1], p.blocks[1]);
    CHECK(c.blocks[2] != p.blocks[2]);  // partial block is private
    CHECK_EQ(kv.allocator().refcount(p.blocks[0]), 2);
    CHECK_EQ(kv.allocator().refcount(p.blocks[2]), 1);
    // shared blocks counted once: 3 (parent) + 2 new for the child
    CHECK_EQ(kv.total_blocks() - kv.free_blocks(), 5);
    KVStats s = kv.stats();
    CHECK_EQ(s.blocks_saved, 2);
    CHECK_EQ(s.logical_blocks, 3 + 4);
    CHECK_EQ(kv.validate(), std::string());
}

TEST(prefix_shared_tokens_clamped_to_parent_and_prompt) {
    KVManager kv(20, 4);
    CHECK(kv.admit(1, 10));                       // 2 full blocks + partial
    CHECK(kv.admit_with_prefix(2, 1, 100, 6));    // asks for more than parent/prompt have
    CHECK_EQ(kv.table(2).blocks[0], kv.table(1).blocks[0]);
    CHECK(kv.table(2).blocks[1] != kv.table(1).blocks[1]);  // block 1 would be partial for the child
    CHECK_EQ(kv.validate(), std::string());
}

TEST(prefix_admit_failure_has_no_side_effects) {
    KVManager kv(4, 4);
    CHECK(kv.admit(1, 12));  // 3 blocks, 1 free
    CHECK(!kv.admit_with_prefix(2, 1, 12, 24));  // shares 3, needs 3 more
    CHECK(!kv.contains(2));
    CHECK_EQ(kv.allocator().refcount(kv.table(1).blocks[0]), 1);
    CHECK_EQ(kv.free_blocks(), 1);
}

TEST(cow_triggers_when_appending_into_shared_partial_block) {
    KVManager kv(10, 4);
    CHECK(kv.admit(1, 6));  // block0 full, block1 has 2 of 4
    CHECK(kv.fork(2, 1));   // shares both blocks, including the partial one
    BlockId shared = kv.table(1).blocks[1];
    CHECK_EQ(kv.allocator().refcount(shared), 2);
    int free_before = kv.free_blocks();
    CHECK(kv.append_token(2));  // must copy the partial block
    CHECK_EQ(kv.stats().cow_copies, 1);
    CHECK(kv.table(2).blocks[1] != shared);
    CHECK_EQ(kv.table(1).blocks[1], shared);
    CHECK_EQ(kv.allocator().refcount(shared), 1);
    CHECK_EQ(kv.free_blocks(), free_before - 1);
    CHECK(kv.append_token(2));  // block is private now: no further copy
    CHECK(kv.append_token(1));  // parent's block is private too
    CHECK_EQ(kv.stats().cow_copies, 1);
    CHECK_EQ(kv.validate(), std::string());
}

TEST(cow_not_triggered_when_last_block_full) {
    KVManager kv(10, 4);
    CHECK(kv.admit(1, 8));  // both blocks full
    CHECK(kv.fork(2, 1));
    CHECK(kv.append_token(2));  // needs a new block, no copy
    CHECK_EQ(kv.stats().cow_copies, 0);
    CHECK_EQ(kv.table(2).blocks.size(), 3u);
    CHECK_EQ(kv.table(2).blocks[0], kv.table(1).blocks[0]);
    CHECK_EQ(kv.table(2).blocks[1], kv.table(1).blocks[1]);
}

TEST(cow_fails_cleanly_without_free_block) {
    KVManager kv(2, 4);
    CHECK(kv.admit(1, 6));  // uses both blocks
    CHECK(kv.fork(2, 1));
    CHECK(!kv.append_token(2));  // cannot copy: no free block
    CHECK_EQ(kv.stats().cow_copies, 0);
    CHECK_EQ(kv.table(2).tokens, 6);
    CHECK_EQ(kv.allocator().refcount(kv.table(1).blocks[1]), 2);
    CHECK_EQ(kv.validate(), std::string());
}

TEST(prefix_freeing_parent_keeps_shared_blocks_alive) {
    KVManager kv(10, 4);
    CHECK(kv.admit(1, 8));
    CHECK(kv.admit_with_prefix(2, 1, 8, 10));
    BlockId b0 = kv.table(1).blocks[0];
    kv.free(1);
    CHECK_EQ(kv.allocator().refcount(b0), 1);  // child still holds it
    CHECK_EQ(kv.validate(), std::string());
    kv.free(2);
    CHECK_EQ(kv.free_blocks(), 10);
}

TEST(prefix_many_children_share_one_copy) {
    KVManager kv(64, 8);
    CHECK(kv.admit(0, 32));  // 4 full blocks
    for (int i = 1; i <= 10; ++i) CHECK(kv.admit_with_prefix(i, 0, 32, 40));
    CHECK_EQ(kv.total_blocks() - kv.free_blocks(), 4 + 10);  // 4 shared + 1 private each
    CHECK_EQ(kv.allocator().refcount(kv.table(0).blocks[0]), 11);
    CHECK_EQ(kv.stats().blocks_saved, 40);
    for (int i = 0; i <= 10; ++i) kv.free(i);
    CHECK_EQ(kv.free_blocks(), 64);
}
