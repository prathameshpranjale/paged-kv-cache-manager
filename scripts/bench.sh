#!/usr/bin/env bash
# One-command benchmark: builds Release, runs the scenarios, writes results/results.md (+ .csv, + plot if matplotlib exists).
# Usage: ./scripts/bench.sh [--seed N] [--requests N]
set -euo pipefail
cd "$(dirname "$0")/.."
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build-release -j"$(nproc 2>/dev/null || echo 2)" --target kvpage_bench
mkdir -p results
ARGS="$*"
{
    echo "# Benchmark results"
    echo
    echo "- command: \`./scripts/bench.sh ${ARGS}\` (runs \`build-release/kvpage_bench ${ARGS}\`)"
    echo "- date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "- machine: $(uname -sr), $(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo unknown cpu), $(nproc) threads"
    echo "- compiler: $(c++ --version | head -1)"
    echo
    # shellcheck disable=SC2086
    build-release/kvpage_bench $ARGS --csv results/results.csv --per-seed-csv results/results_per_seed.csv
} > results/results.md
cat results/results.md
if python3 -c "import matplotlib" 2>/dev/null; then
    python3 scripts/plot.py results/results.csv results
else
    echo "(matplotlib not found: skipping plots; run scripts/plot.py where it is installed)"
fi
