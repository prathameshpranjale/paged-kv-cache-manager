# CPU benchmark (timing of the C++ code, not simulated metrics)

- command: `./scripts/cpu_bench.sh `
- date: 2026-10-08T09:02:18Z
- machine: Linux 6.6.87.1-microsoft-standard-WSL2, Intel(R) Core(TM) i5-9400 CPU @ 2.90GHz, 6 threads
- compiler: c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0, CMAKE_BUILD_TYPE=Release (-O3 -DNDEBUG), single thread

reps=9 (median; min/max in CSV), simulator A: 3478 steps, 395279 tokens

| Benchmark | Ops / rep | Median ms / rep | Mops/s | ns/op |
|---|---|---|---|---|
| allocator: allocate+release, 1M blocks x 3 rounds | 6291456 | 20.05 | 313.75 | 3.2 |
| kv_manager: append_token, 256 reqs x 4096 tokens, B=16 | 1048576 | 19.99 | 52.46 | 19.1 |
| kv_manager: admit_with_prefix+free, 200k requests | 400000 | 112.15 | 3.57 | 280.4 |
| kv_manager: stats(), 200 live requests x 20k calls | 20000 | 0.19 | 107.07 | 9.3 |
| simulator: scenario A paged-FCFS (ops = simulated steps) | 3478 | 13.35 | 0.26 | 3837.3 |
| simulator: scenario A paged-FCFS (ops = generated tokens) | 395279 | 13.35 | 29.62 | 33.8 |
| simulator: same, with validate_each_step (ops = steps) | 3478 | 13293.46 | 0.00 | 3822156.7 |
| simulator: scenario A contig-maxlen (ops = simulated steps) | 12629 | 1.45 | 8.70 | 114.9 |
| workload: generate 100k requests | 100000 | 5.28 | 18.94 | 52.8 |
