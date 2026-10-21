#!/usr/bin/env bash
# Build and test with every available compiler using -Werror (warnings: -Wall -Wextra -Wpedantic -Wshadow).
set -euo pipefail
cd "$(dirname "$0")/.."
for cxx in g++ clang++; do
    if ! command -v "$cxx" >/dev/null; then echo "skipping $cxx (not installed)"; continue; fi
    dir="build-werror-$cxx"
    echo "=== $cxx $($cxx --version | head -1) ==="
    cmake -S . -B "$dir" -DCMAKE_CXX_COMPILER="$cxx" -DCMAKE_BUILD_TYPE=Debug -DKVPAGE_WERROR=ON >/dev/null
    cmake --build "$dir" -j"$(nproc)" --target kvpage_tests kvpage_bench kvpage_cpu_bench 2>&1 | grep -E "error|warning" || true
    "$dir/kvpage_tests" | tail -1
done
