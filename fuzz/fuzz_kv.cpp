// libFuzzer target: interprets the input as a sequence of KVManager operations. Every operation validates
// all invariants (the manager is built without NDEBUG, so check_each_op is on); a violation throws and
// the fuzzer reports a crash. Build: cmake -DCMAKE_CXX_COMPILER=clang++ -DKVPAGE_FUZZ=ON, run ./fuzz_kv corpus/
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "kvpage/kv_manager.h"

using namespace kvp;

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    if (size < 4) return 0;
    const int B = 1 + data[0] % 8;
    const int total = 2 + data[1] % 30;
    KVManager kv(total, B);
    std::vector<RequestId> live;
    RequestId next_id = 0;
    auto pick = [&](uint8_t x) { return live[x % live.size()]; };

    for (size_t i = 2; i + 2 < size; i += 3) {
        const uint8_t op = data[i] % 9, a = data[i + 1], b = data[i + 2];
        switch (op) {
            case 0:
                if (kv.admit(next_id, a % (6 * B + 1))) live.push_back(next_id);
                ++next_id;
                break;
            case 1:
                if (live.empty()) break;
                if (kv.admit_with_prefix(next_id, pick(a), b % (5 * B + 1), a % (6 * B + 1))) live.push_back(next_id);
                ++next_id;
                break;
            case 2:
                if (live.empty()) break;
                kv.fork(next_id, pick(a));
                live.push_back(next_id++);
                break;
            case 3:
            case 4:
                if (!live.empty()) kv.append_token(pick(a));
                break;
            case 5:
                if (live.empty()) break;
                {
                    RequestId r = pick(a);
                    kv.free(r);
                    live.erase(std::find(live.begin(), live.end(), r));
                }
                break;
            case 6:
                if (!live.empty()) kv.cache_prefix(b % 3, pick(a), a % (5 * B + 1));
                break;
            case 7:
                if (kv.admit_with_cached_prefix(next_id, b % 3, a % (5 * B + 1), b % (6 * B + 1))) live.push_back(next_id);
                ++next_id;
                break;
            default:
                if (a % 8 == 0) kv.clear_cache();
                break;
        }
    }
    for (RequestId r : live) kv.free(r);
    kv.clear_cache();
    if (kv.free_blocks() != total) __builtin_trap();  // leak
    return 0;
}
