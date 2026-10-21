#!/usr/bin/env bash
# Fuzz KVManager with libFuzzer + ASan/UBSan (needs clang). Usage: ./scripts/fuzz.sh [seconds]
set -euo pipefail
cd "$(dirname "$0")/.."
SECS=${1:-60}
cmake -S . -B build-fuzz -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug -DKVPAGE_FUZZ=ON >/dev/null
cmake --build build-fuzz -j"$(nproc)" --target fuzz_kv 2>&1 | grep -E "error|warning" || true
mkdir -p fuzz/corpus
build-fuzz/fuzz_kv -max_total_time="$SECS" -print_final_stats=1 fuzz/corpus 2>&1 | tail -15
