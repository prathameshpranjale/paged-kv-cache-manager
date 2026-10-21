#!/usr/bin/env bash
# CI entry point: full build + all tests including sanitizers, then a small benchmark smoke run.
set -euo pipefail
cd "$(dirname "$0")/.."
./scripts/test.sh
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build-release -j"$(nproc 2>/dev/null || echo 2)" --target kvpage_bench
build-release/kvpage_bench --seeds 2 --requests 200 >/dev/null
echo "CI OK"
