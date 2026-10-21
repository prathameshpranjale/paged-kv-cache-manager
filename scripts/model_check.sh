#!/usr/bin/env bash
# Real-LLM checks (CPU): paged KV cache driven by the C++ KVManager vs the stock HF cache.
# One-time setup:  python3 -m venv ~/kvenv && . ~/kvenv/bin/activate &&
#   pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install transformers pybind11 numpy
# Usage: ./scripts/model_check.sh [--model NAME] [--dtype fp32|bf16|fp16] [--quick] [--out FILE]
#   default: Qwen/Qwen2.5-0.5B-Instruct fp32 (about 5 min, needs ~3 GB RAM). Set HF_HOME to cache models on another drive.
#   Memory: fp32 TinyLlama (4.4 GB) crashed a 7.8 GB WSL VM; use --dtype bf16 --quick for models above 1B.
set -euo pipefail
cd "$(dirname "$0")/.."
. "${KVENV:-$HOME/kvenv}/bin/activate"
cmake -S . -B build-py -DCMAKE_BUILD_TYPE=Release -DKVPAGE_PYTHON=ON \
    -Dpybind11_DIR="$(python -m pybind11 --cmakedir)" >/dev/null
cmake --build build-py --target kvpage_py -j"$(nproc)"
python python/attention_check.py          # C++ attention references vs torch SDPA
python python/model_check.py "$@"
python python/negative_control.py         # must report that the harness detects a deliberately broken COW
