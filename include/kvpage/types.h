#pragma once
#include <cstdint>

namespace kvp {

using BlockId = int32_t;
using RequestId = int64_t;
constexpr BlockId kNoBlock = -1;
constexpr RequestId kNoRequest = -1;

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

}  // namespace kvp
