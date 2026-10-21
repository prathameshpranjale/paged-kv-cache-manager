#!/usr/bin/env bash
# CPU timing benchmark of the code itself (Release). Writes results/cpu_benchmark.md and .csv.
# Usage: ./scripts/cpu_bench.sh [--reps N]
set -euo pipefail
cd "$(dirname "$0")/.."
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build-release -j"$(nproc 2>/dev/null || echo 2)" --target kvpage_cpu_bench
mkdir -p results
{
    echo "# CPU benchmark (timing of the C++ code, not simulated metrics)"
    echo
    echo "- command: \`./scripts/cpu_bench.sh $*\`"
    echo "- date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "- machine: $(uname -sr), $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo unknown cpu), $(nproc) threads"
    echo "- compiler: $(c++ --version | head -1), CMAKE_BUILD_TYPE=Release (-O3 -DNDEBUG), single thread"
    echo
    build-release/kvpage_cpu_bench "$@" --csv results/cpu_benchmark.csv
} > results/cpu_benchmark.md
cat results/cpu_benchmark.md
