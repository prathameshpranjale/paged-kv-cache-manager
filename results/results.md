# Benchmark results

- command: `./scripts/bench.sh ` (runs `build-release/kvpage_bench `)
- date: 2026-10-08T08:29:10Z
- machine: Linux 6.6.87.1-microsoft-standard-WSL2, Intel(R) Core(TM) i5-9400 CPU @ 2.90GHz, 6 threads
- compiler: c++ (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0

seeds=5 (first 1) requests/prompts per run=1500 block_size=16; cells are mean ± sample std dev over seeds

### A uniform lengths, memory-bound

| System | Util % | Mean conc | Peak conc | Tok/step | p50 lat | p95 lat | Mean queue | Preempt | Recomputed tok | COW | Blocks saved (mean / peak) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| paged-FCFS | 98.3 ± 0.0 | 115.0 ± 1.7 | 153 ± 1 | 114.9 ± 1.7 | 274 ± 10 | 494 ± 3 | 4.8 ± 4.1 | 210 ± 115 | 58151 ± 29775 | 0 ± 0 | 0 / 0 |
| paged-SPF | 98.3 ± 0.0 | 115.2 ± 1.8 | 152 ± 1 | 115.1 ± 1.8 | 273 ± 10 | 493 ± 2 | 2.6 ± 2.2 | 196 ± 110 | 57913 ± 29872 | 0 ± 0 | 0 / 0 |
| contig-maxlen | 21.7 ± 0.1 | 31.2 ± 0.1 | 32 ± 0 | 31.2 ± 0.1 | 4913 ± 173 | 9271 ± 177 | 4662.5 ± 141.0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 / 0 |
| contig-oracle | 72.1 ± 0.2 | 95.3 ± 0.8 | 115 ± 4 | 95.3 ± 0.8 | 612 ± 76 | 1042 ± 76 | 344.1 ± 61.3 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 / 0 |

### B long-tailed lengths, memory not binding

| System | Util % | Mean conc | Peak conc | Tok/step | p50 lat | p95 lat | Mean queue | Preempt | Recomputed tok | COW | Blocks saved (mean / peak) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| paged-FCFS | 94.9 ± 0.2 | 73.7 ± 2.3 | 181 ± 10 | 73.6 ± 2.3 | 26 ± 0 | 115 ± 14 | 0.0 ± 0.0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 / 0 |
| contig-maxlen | 6.8 ± 0.2 | 28.1 ± 1.1 | 32 ± 0 | 28.1 ± 1.1 | 739 ± 34 | 1453 ± 55 | 711.1 ± 25.0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 / 0 |
| contig-oracle | 71.1 ± 2.1 | 73.7 ± 2.3 | 181 ± 10 | 73.6 ± 2.3 | 26 ± 0 | 115 ± 14 | 0.0 ± 0.0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 / 0 |

### C 70% share a 512-token system prompt

| System | Util % | Mean conc | Peak conc | Tok/step | p50 lat | p95 lat | Mean queue | Preempt | Recomputed tok | COW | Blocks saved (mean / peak) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| paged-FCFS | 96.9 ± 0.0 | 106.7 ± 2.5 | 147 ± 5 | 106.7 ± 2.5 | 145 ± 3 | 246 ± 2 | 0.0 ± 0.0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 2376 / 3360 |
| paged-FCFS-noshare | 98.7 ± 0.0 | 97.7 ± 1.6 | 121 ± 1 | 97.6 ± 1.6 | 225 ± 25 | 371 ± 26 | 80.3 ± 21.5 | 161 ± 25 | 83873 ± 12302 | 0 ± 0 | 0 / 0 |
| contig-maxlen | 28.8 ± 0.4 | 31.3 ± 0.1 | 32 ± 0 | 31.3 ± 0.1 | 2535 ± 105 | 4777 ± 90 | 2399.7 ± 74.2 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 / 0 |
| contig-oracle | 87.2 ± 0.2 | 87.9 ± 1.0 | 105 ± 3 | 87.9 ± 1.0 | 343 ± 48 | 583 ± 40 | 197.1 ± 33.9 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 / 0 |

### D tight memory, overload
_arrivals far above service rate: queues grow all run, latencies measure the backlog_

| System | Util % | Mean conc | Peak conc | Tok/step | p50 lat | p95 lat | Mean queue | Preempt | Recomputed tok | COW | Blocks saved (mean / peak) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| paged-FCFS | 98.3 ± 0.0 | 26.4 ± 0.2 | 39 ± 3 | 26.3 ± 0.2 | 6862 ± 233 | 12984 ± 187 | 6605.5 ± 165.1 | 1131 ± 47 | 323715 ± 9198 | 0 ± 0 | 0 / 0 |
| paged-SPF | 98.3 ± 0.0 | 26.5 ± 0.2 | 65 ± 4 | 26.4 ± 0.2 | 4927 ± 175 | 13372 ± 163 | 5053.0 ± 89.2 | 1468 ± 42 | 332512 ± 9141 | 0 ± 0 | 0 / 0 |
| contig-maxlen | 43.3 ± 0.3 | 11.9 ± 0.0 | 12 ± 0 | 11.9 ± 0.0 | 16133 ± 485 | 30700 ± 513 | 15911.6 ± 384.4 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 / 0 |
| contig-oracle | 72.1 ± 0.2 | 19.2 ± 0.1 | 24 ± 1 | 19.2 ± 0.1 | 9710 ± 349 | 18401 ± 302 | 9466.9 ± 258.0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 / 0 |

### E parallel sampling, 4 samples per prompt

| System | Util % | Mean conc | Peak conc | Tok/step | p50 lat | p95 lat | Mean queue | Preempt | Recomputed tok | COW | Blocks saved (mean / peak) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| paged-FCFS | 96.2 ± 0.0 | 137.2 ± 0.9 | 217 ± 12 | 137.2 ± 0.9 | 145 ± 2 | 246 ± 1 | 0.0 ± 0.0 | 0 ± 0 | 0 ± 0 | 4202 ± 28 | 1441 / 2507 |
| paged-FCFS-nofork | 98.0 ± 0.0 | 137.2 ± 0.9 | 197 ± 4 | 137.2 ± 0.9 | 147 ± 3 | 248 ± 2 | 2.0 ± 1.6 | 111 ± 46 | 32149 ± 12581 | 0 ± 0 | 0 / 0 |
| contig-maxlen | 17.5 ± 0.2 | 31.8 ± 0.0 | 32 ± 0 | 31.8 ± 0.0 | 10416 ± 57 | 20001 ± 144 | 10360.1 ± 82.6 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 / 0 |
| contig-oracle | 80.5 ± 0.2 | 135.4 ± 2.1 | 164 ± 3 | 135.4 ± 2.1 | 199 ± 19 | 355 ± 54 | 55.7 ± 21.7 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 / 0 |

### F steady load (below saturation)
_arrival rate below what paged can serve: latencies are steady-state, not backlog_

| System | Util % | Mean conc | Peak conc | Tok/step | p50 lat | p95 lat | Mean queue | Preempt | Recomputed tok | COW | Blocks saved (mean / peak) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| paged-FCFS | 98.3 ± 0.0 | 62.0 ± 1.4 | 89 ± 2 | 62.0 ± 1.4 | 269 ± 8 | 489 ± 1 | 0.0 ± 0.0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 / 0 |
| paged-SPF | 98.3 ± 0.0 | 62.0 ± 1.4 | 89 ± 2 | 62.0 ± 1.4 | 269 ± 8 | 489 ± 1 | 0.0 ± 0.0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 / 0 |
| contig-maxlen | 21.7 ± 0.1 | 31.1 ± 0.1 | 32 ± 0 | 31.1 ± 0.1 | 3442 ± 220 | 6424 ± 189 | 3186.1 ± 164.0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 / 0 |
| contig-oracle | 72.1 ± 0.2 | 62.0 ± 1.4 | 89 ± 2 | 62.0 ± 1.4 | 269 ± 8 | 489 ± 1 | 0.0 ± 0.0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 / 0 |

### G prefix cache: 16 system prompts of 512 tokens, light load
_prefill modelled at 256 tokens/step (assumption); cache reclaimed under memory pressure_

| System | Util % | Mean conc | Peak conc | Tok/step | p50 lat | p95 lat | Mean queue | Preempt | Recomputed tok | COW | Blocks saved (mean / peak) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| paged no sharing | 98.9 ± 0.0 | 7.0 ± 0.1 | 18 ± 2 | 6.7 ± 0.1 | 83 ± 1 | 127 ± 0 | 0.0 ± 0.0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 0 / 0 |
| paged live-parent sharing | 98.7 ± 0.0 | 6.9 ± 0.1 | 18 ± 2 | 6.7 ± 0.1 | 82 ± 1 | 127 ± 1 | 0.0 ± 0.0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 33 / 211 |
| paged live + cache | 99.5 ± 0.0 | 6.8 ± 0.1 | 18 ± 2 | 6.7 ± 0.1 | 81 ± 1 | 126 ± 1 | 0.0 ± 0.0 | 0 ± 0 | 0 ± 0 | 0 ± 0 | 32 / 205 |

| System | Swapped-out blocks | Cache hits | Cache evictions | Prefill tokens computed | Prefill tokens saved |
|---|---|---|---|---|---|
| paged no sharing | 0 | 0 | 0 | 912458 | 0 |
| paged live-parent sharing | 0 | 0 | 0 | 693117 | 219341 |
| paged live + cache | 0 | 1343 | 0 | 224945 | 687514 |

### H priority classes: 25% high priority, tight memory

| System | Util % | Mean conc | Peak conc | Tok/step | p50 lat | p95 lat | Mean queue | Preempt | Recomputed tok | COW | Blocks saved (mean / peak) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| FCFS | 97.9 ± 0.0 | 27.0 ± 0.2 | 36 ± 1 | 26.9 ± 0.2 | 2099 ± 105 | 3953 ± 111 | 1951.5 ± 83.9 | 550 ± 8 | 150146 ± 3986 | 0 ± 0 | 0 / 0 |
| SPF | 97.9 ± 0.0 | 27.1 ± 0.2 | 43 ± 1 | 27.0 ± 0.2 | 231 ± 8 | 5671 ± 68 | 1298.9 ± 61.5 | 610 ± 20 | 154193 ± 6001 | 0 ± 0 | 0 / 0 |
| priority (order only) | 97.9 ± 0.0 | 27.0 ± 0.2 | 36 ± 1 | 26.9 ± 0.2 | 2718 ± 192 | 4013 ± 97 | 1952.5 ± 85.1 | 530 ± 12 | 153572 ± 1390 | 0 ± 0 | 0 / 0 |
| priority + preemption | 97.9 ± 0.0 | 27.0 ± 0.2 | 36 ± 2 | 26.9 ± 0.2 | 2723 ± 194 | 4014 ± 99 | 1952.9 ± 86.6 | 815 ± 24 | 238682 ± 7229 | 0 ± 0 | 0 / 0 |

| System | p95 latency high | p95 latency low | Mean queue high | Mean queue low |
|---|---|---|---|---|
| FCFS | 3942 ± 117 | 3955 ± 113 | 1942.0 ± 100.3 | 1955.1 ± 86.1 |
| SPF | 5640 ± 256 | 5695 ± 114 | 1278.9 ± 98.6 | 1305.1 ± 77.1 |
| priority (order only) | 253 ± 2 | 4057 ± 103 | 6.5 ± 1.0 | 2646.5 ± 121.5 |
| priority + preemption | 248 ± 2 | 4058 ± 107 | 0.1 ± 0.0 | 2649.4 ± 123.6 |

### I swap vs recompute, tight memory
_assumed rates: prefill 256 tokens/step (fast) or 16 (slow), swap 64 blocks/step; step = one decode iteration_

| System | Util % | Mean conc | Peak conc | Tok/step | p50 lat | p95 lat | Mean queue | Preempt | Recomputed tok | COW | Blocks saved (mean / peak) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| recompute, prefill 256 tok/step | 98.3 ± 0.0 | 26.4 ± 0.2 | 38 ± 2 | 26.1 ± 0.1 | 6180 ± 244 | 11679 ± 184 | 5921.0 ± 173.1 | 1135 ± 36 | 328515 ± 5638 | 0 ± 0 | 0 / 0 |
| swap, prefill 256 tok/step | 98.3 ± 0.0 | 26.4 ± 0.1 | 38 ± 2 | 26.1 ± 0.1 | 6171 ± 240 | 11656 ± 180 | 5910.9 ± 172.2 | 1138 ± 38 | 0 ± 0 | 0 ± 0 | 0 / 0 |
| recompute, prefill 16 tok/step | 98.3 ± 0.0 | 26.9 ± 0.2 | 39 ± 2 | 24.6 ± 0.2 | 6656 ± 238 | 12577 ± 177 | 6377.4 ± 172.0 | 1181 ± 52 | 332059 ± 8424 | 0 ± 0 | 0 / 0 |
| swap, prefill 16 tok/step | 98.3 ± 0.0 | 26.7 ± 0.1 | 39 ± 2 | 25.3 ± 0.2 | 6422 ± 237 | 12117 ± 174 | 6150.2 ± 169.3 | 1164 ± 36 | 0 ± 0 | 0 ± 0 | 0 / 0 |

| System | Swapped-out blocks | Cache hits | Cache evictions | Prefill tokens computed | Prefill tokens saved |
|---|---|---|---|---|---|
| recompute, prefill 256 tok/step | 0 | 0 | 0 | 738588 | 0 |
| swap, prefill 256 tok/step | 21099 | 0 | 0 | 410073 | 0 |
| recompute, prefill 16 tok/step | 0 | 0 | 0 | 742132 | 0 |
| swap, prefill 16 tok/step | 21185 | 0 | 0 | 410073 | 0 |
