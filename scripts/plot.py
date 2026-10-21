#!/usr/bin/env python3
"""Plot results/results.csv (summary written by kvpage_bench --csv: mean and std dev over seeds). Needs matplotlib."""
import csv
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "results/results.csv"
outdir = sys.argv[2] if len(sys.argv) > 2 else "results"

rows = list(csv.DictReader(open(src)))
core = ["paged-FCFS", "paged-SPF", "contig-maxlen", "contig-oracle"]
scenarios = [s for s in OrderedDict.fromkeys(r["scenario"] for r in rows)
             if s[0] in "ABCDEF"]


def grouped_bars(metric, sd_metric, ylabel, fname, scale=1.0, log=False):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    w = 0.8 / len(core)
    for k, sysname in enumerate(core):
        ys, es = [], []
        for sc in scenarios:
            m = [r for r in rows if r["scenario"] == sc and r["system"] == sysname]
            ys.append(float(m[0][metric]) * scale if m else 0)
            es.append(float(m[0][sd_metric]) * scale if m and sd_metric else 0)
        ax.bar([i + k * w for i in range(len(scenarios))], ys, w, yerr=es if sd_metric else None, capsize=2, label=sysname)
    ax.set_xticks([i + 0.4 - w / 2 for i in range(len(scenarios))])
    ax.set_xticklabels([s.split(" ")[0] for s in scenarios])
    ax.set_xlabel("scenario (see results.md); error bars: std dev over seeds")
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{outdir}/{fname}", dpi=130)
    plt.close(fig)


grouped_bars("util", "util_sd", "memory utilization (%)", "utilization.png", 100)
grouped_bars("mean_conc", "mean_conc_sd", "mean concurrent requests", "concurrency.png")
grouped_bars("p95", "p95_sd", "p95 latency (steps, log scale)", "latency_p95.png", log=True)
print("wrote plots to", outdir)
