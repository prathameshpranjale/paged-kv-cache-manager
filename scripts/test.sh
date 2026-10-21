#!/usr/bin/env bash
# One-command build + test: Debug (invariants on), ASan+UBSan full suite, TSan concurrency test.
# Usage: ./scripts/test.sh [fast]   ("fast" skips the sanitizer builds)
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=$(nproc 2>/dev/null || echo 2)

build_and_test() {  # dir sanitizer filter
    local dir=$1 san=$2 filter=${3:-}
    echo "=== $dir (SANITIZE=$san) ==="
    cmake -S . -B "$dir" -DCMAKE_BUILD_TYPE=Debug -DSANITIZE="$san" >/dev/null
    cmake --build "$dir" -j"$JOBS" --target kvpage_tests
    local run=("$dir/kvpage_tests")
    if [ -n "$filter" ]; then run+=(--filter "$filter"); fi
    if [ "$san" = tsan ]; then
        # TSan aborts under high-entropy ASLR (common on WSL2 / new kernels): disable ASLR for it.
        setarch "$(uname -m)" -R "${run[@]}" || "${run[@]}"
    else
        "${run[@]}"
    fi
}

build_and_test build none
if [ "${1:-}" != "fast" ]; then
    build_and_test build-asan asan
    build_and_test build-tsan tsan concurrency
fi
echo "ALL TESTS PASSED"
