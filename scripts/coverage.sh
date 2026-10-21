#!/usr/bin/env bash
# Line coverage of src/ and include/ from the test suite (gcc + gcov). Prints a per-file summary.
set -euo pipefail
cd "$(dirname "$0")/.."
cmake -S . -B build-cov -DCMAKE_BUILD_TYPE=Debug -DKVPAGE_COVERAGE=ON >/dev/null
cmake --build build-cov -j"$(nproc)" --target kvpage_tests 2>&1 | grep -E "error" || true
find build-cov -name '*.gcda' -delete
build-cov/kvpage_tests | tail -1
cd build-cov/CMakeFiles/kvpage.dir/src
echo "--- line coverage (library sources only) ---"
for f in *.gcda; do
    gcov -n "$f" 2>/dev/null | awk -v f="$f" '/^File .*\/src\//{file=$2} /^Lines executed/ && file!="" {print file, $0; file=""}'
done | sed "s|'||g; s|/mnt/e/AAAAProject/||"
