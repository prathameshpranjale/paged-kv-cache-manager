# Third-party material and licences

This repository contains no code copied from other projects. It re-implements an idea and uses external tools and models only
at test time. The project's own code is under the MIT licence (see `LICENSE`; copyright line "kvpage authors", change it to your
name or organisation if you prefer). The third-party items below keep their own licences.

## Ideas credited
- Kwon et al., *Efficient Memory Management for Large Language Model Serving with PagedAttention*, SOSP 2023, and its
  implementation **vLLM** (Apache-2.0): block tables, paged KV cache, copy-on-write, preemption by recompute or swap.
- **SGLang** (RadixAttention, Apache-2.0): sharing prompt prefixes across requests (here: a simple persistent prefix cache).
- TensorRT-LLM and others have production versions of the same ideas.

## Models downloaded for the real-model checks (not distributed here)
| Model | Licence (from its Hugging Face model card) |
|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` | Apache-2.0 |
| `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Apache-2.0 |

Check each model card again before redistributing any weights or outputs. No employer models, data or code are used.

## Software used at build / test time (installed separately, not vendored)
CMake, g++ / clang++, pybind11 (BSD-3-Clause), PyTorch (BSD-3-Clause), Hugging Face transformers (Apache-2.0),
NVIDIA CUDA headers from pip wheels (only for the compile check; NVIDIA licence), matplotlib (for plots).
