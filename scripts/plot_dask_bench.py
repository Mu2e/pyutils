#!/usr/bin/env python3
"""
Plotting utilities for benchmark results produced by
`scripts/benchmark_dask.py`.

Generates:
- throughput (files/sec) vs `n_workers` (hue=`threads_per_worker` and `processes`)
- heatmap of throughput/time with n_workers x threads_per_worker
- speedup and efficiency vs total cores
- per-file duration distributions (if available)

Requires: pandas, matplotlib

Usage:
  python3 scripts/plot_dask_bench.py --summary benchmark_dask_results.csv --out-dir bench_out
"""

import argparse
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def plot_throughput_vs_n_workers(df, out_path):
    """Plot files/sec vs n_workers, grouped by threads_per_worker and processes."""
    plt.figure(figsize=(10, 6))
    
    # Group by threads_per_worker and processes
    for (tpw, proc), grp in df.groupby(["threads_per_worker", "processes"]):
        grp_sorted = grp.sort_values("n_workers")
        label = f"tpw={int(tpw)}, proc={proc}"
        plt.plot(grp_sorted["n_workers"], grp_sorted["files_per_sec"], marker="o", label=label)
    
    plt.xlabel("n_workers")
    plt.ylabel("files / s")
    plt.title("Throughput vs n_workers")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Wrote {out_path}")


def plot_throughput_vs_total_cores(df, out_path):
    """Plot files/sec vs total_cores (n_workers * threads_per_worker)."""
    df_plot = df.copy()
    df_plot["total_cores"] = df_plot["n_workers"] * df_plot["threads_per_worker"]
    df_plot["label"] = df_plot.apply(
        lambda r: f"proc={r['processes']}", axis=1
    )
    
    plt.figure(figsize=(10, 6))
    for proc, grp in df_plot.groupby("processes"):
        grp_sorted = grp.sort_values("total_cores")
        label = f"processes={proc}"
        plt.plot(grp_sorted["total_cores"], grp_sorted["files_per_sec"], marker="o", label=label)
    
    plt.xlabel("total cores (n_workers × threads_per_worker)")
    plt.ylabel("files / s")
    plt.title("Throughput vs total cores")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Wrote {out_path}")


def plot_heatmap_throughput(df, out_path, processes_val=False):
    """Create heatmap of throughput with n_workers (rows) x threads_per_worker (columns)."""
    df_proc = df[df["processes"] == processes_val].copy()
    
    if df_proc.empty:
        print(f"No data for processes={processes_val}")
        return
    
    pivot = df_proc.pivot_table(
        index="n_workers",
        columns="threads_per_worker",
        values="files_per_sec",
        aggfunc="mean"
    )
    
    if pivot.empty:
        return
    
    plt.figure(figsize=(8, 6))
    im = plt.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(im, label="files / s")
    plt.yticks(range(len(pivot.index)), [str(int(x)) for x in pivot.index])
    plt.xticks(range(len(pivot.columns)), [str(int(x)) for x in pivot.columns])
    plt.xlabel("threads_per_worker")
    plt.ylabel("n_workers")
    plt.title(f"Throughput heatmap (processes={processes_val})")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Wrote {out_path}")


def plot_heatmap_total_time(df, out_path, processes_val=False):
    """Create heatmap of total_time with n_workers (rows) x threads_per_worker (columns)."""
    df_proc = df[df["processes"] == processes_val].copy()
    
    if df_proc.empty:
        print(f"No data for processes={processes_val}")
        return
    
    pivot = df_proc.pivot_table(
        index="n_workers",
        columns="threads_per_worker",
        values="total_time",
        aggfunc="mean"
    )
    
    if pivot.empty:
        return
    
    plt.figure(figsize=(8, 6))
    im = plt.imshow(pivot.values, aspect="auto", origin="lower", cmap="RdYlGn_r")
    plt.colorbar(im, label="total time (s)")
    plt.yticks(range(len(pivot.index)), [str(int(x)) for x in pivot.index])
    plt.xticks(range(len(pivot.columns)), [str(int(x)) for x in pivot.columns])
    plt.xlabel("threads_per_worker")
    plt.ylabel("n_workers")
    plt.title(f"Total time heatmap (processes={processes_val})")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Wrote {out_path}")


def plot_speedup_and_efficiency(df, out_path_speedup, out_path_efficiency):
    """Compute and plot speedup and efficiency vs total_cores."""
    df_plot = df.copy()
    df_plot["total_cores"] = df_plot["n_workers"] * df_plot["threads_per_worker"]
    
    # Find baseline: minimum total_time (fastest single-threaded config)
    baseline = df_plot[df_plot["total_cores"] == 1]["total_time"]
    if baseline.empty:
        baseline = df_plot["total_time"].min()
    else:
        baseline = baseline.min()
    
    df_plot["speedup"] = baseline / df_plot["total_time"]
    df_plot["efficiency"] = df_plot["speedup"] / df_plot["total_cores"]
    
    # Speedup plot
    plt.figure(figsize=(10, 6))
    for proc, grp in df_plot.groupby("processes"):
        grp_sorted = grp.sort_values("total_cores")
        label = f"processes={proc}"
        plt.plot(grp_sorted["total_cores"], grp_sorted["speedup"], marker="o", label=label)
    
    plt.xlabel("total cores")
    plt.ylabel("speedup")
    plt.title("Speedup vs total cores")
    plt.legend()
    plt.grid(True, alpha=0.3)
    # Add diagonal line for perfect scaling
    max_cores = df_plot["total_cores"].max()
    plt.plot([1, max_cores], [1, max_cores], "k--", alpha=0.3, label="perfect scaling")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path_speedup)
    plt.close()
    print(f"Wrote {out_path_speedup}")
    
    # Efficiency plot
    plt.figure(figsize=(10, 6))
    for proc, grp in df_plot.groupby("processes"):
        grp_sorted = grp.sort_values("total_cores")
        label = f"processes={proc}"
        plt.plot(grp_sorted["total_cores"], grp_sorted["efficiency"], marker="o", label=label)
    
    plt.xlabel("total cores")
    plt.ylabel("efficiency (speedup / cores)")
    plt.title("Efficiency vs total cores")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1.0, color="k", linestyle="--", alpha=0.3, label="perfect scaling")
    plt.tight_layout()
    plt.savefig(out_path_efficiency)
    plt.close()
    print(f"Wrote {out_path_efficiency}")


def plot_total_time_comparison(df, out_path):
    """Bar plot comparing total_time across configurations."""
    df_plot = df.copy()
    df_plot["total_cores"] = df_plot["n_workers"] * df_plot["threads_per_worker"]
    df_plot["config"] = df_plot.apply(
        lambda r: f"nw={int(r['n_workers'])},tpw={int(r['threads_per_worker'])},p={str(r['processes'])[0]}",
        axis=1
    )
    
    plt.figure(figsize=(14, 6))
    x = range(len(df_plot))
    colors = ["blue" if p else "orange" for p in df_plot["processes"]]
    plt.bar(x, df_plot["total_time"], color=colors, alpha=0.7)
    plt.xticks(x, df_plot["config"], rotation=45, ha="right", fontsize=9)
    plt.ylabel("total time (s)")
    plt.title("Total time by configuration")
    plt.grid(True, alpha=0.3, axis="y")
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor="blue", alpha=0.7, label="processes=False"),
                       Patch(facecolor="orange", alpha=0.7, label="processes=True")]
    plt.legend(handles=legend_elements)
    
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot benchmark results from benchmark_dask.py"
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="bench_out/dask/benchmark_dask_results.csv",
        help="Summary CSV from benchmark_dask.py",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="bench_out/dask",
        help="Directory where plots will be written",
    )
    args = parser.parse_args()

    if not os.path.exists(args.summary):
        print(f"Summary CSV not found: {args.summary}")
        return

    df = pd.read_csv(args.summary)
    
    # Ensure numeric types
    for col in ["n_workers", "threads_per_worker", "total_time", "files_per_sec"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    plots_dir = os.path.join(args.out_dir, "plots")
    ensure_dir(plots_dir)

    # Throughput vs n_workers
    out1 = os.path.join(plots_dir, "throughput_vs_n_workers.png")
    plot_throughput_vs_n_workers(df, out1)

    # Throughput vs total cores
    out2 = os.path.join(plots_dir, "throughput_vs_total_cores.png")
    plot_throughput_vs_total_cores(df, out2)

    # Heatmaps for processes=False
    out_h1_false = os.path.join(plots_dir, "heatmap_throughput_processes_false.png")
    plot_heatmap_throughput(df, out_h1_false, processes_val=False)

    out_h2_false = os.path.join(plots_dir, "heatmap_total_time_processes_false.png")
    plot_heatmap_total_time(df, out_h2_false, processes_val=False)

    # Heatmaps for processes=True (if data exists)
    if (df["processes"] == True).any():
        out_h1_true = os.path.join(plots_dir, "heatmap_throughput_processes_true.png")
        plot_heatmap_throughput(df, out_h1_true, processes_val=True)

        out_h2_true = os.path.join(plots_dir, "heatmap_total_time_processes_true.png")
        plot_heatmap_total_time(df, out_h2_true, processes_val=True)

    # Speedup and efficiency
    out_speed = os.path.join(plots_dir, "speedup_vs_cores.png")
    out_eff = os.path.join(plots_dir, "efficiency_vs_cores.png")
    plot_speedup_and_efficiency(df, out_speed, out_eff)

    # Total time comparison
    out_time = os.path.join(plots_dir, "total_time_comparison.png")
    plot_total_time_comparison(df, out_time)

    print(f"\nAll plots written to {plots_dir}")


if __name__ == "__main__":
    main()
