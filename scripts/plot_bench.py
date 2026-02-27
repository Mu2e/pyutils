#!/usr/bin/env python3
"""
Plotting utilities for benchmark results produced by
`scripts/benchmark_pyprocess.py`.

Generates:
- throughput (files/sec) vs `max_workers` (hue=`use_processes`)
- latency percentiles (median, p90, p99) vs `max_workers`
- per-file duration histogram for a chosen per-file CSV

Requires: pandas, matplotlib, seaborn

Usage:
  python3 scripts/plot_bench.py --summary bench_summary.csv --out-dir bench_out
"""

import argparse
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def plot_throughput(df, out_path):
    plt.figure(figsize=(7, 4))
    # Plot lines per `use_processes` value if present, otherwise single line
    if "use_processes" in df.columns:
        for val in sorted(df["use_processes"].unique()):
            sub = df[df["use_processes"] == val].sort_values("max_workers")
            plt.plot(sub["max_workers"], sub["files_per_sec"], marker="o", label=str(val))
        plt.legend(title="use_processes")
    else:
        df_sorted = df.sort_values("max_workers")
        plt.plot(df_sorted["max_workers"], df_sorted["files_per_sec"], marker="o")
    plt.xlabel("max_workers")
    plt.ylabel("files / s")
    plt.title("Throughput vs max_workers")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_latency_percentiles(df, out_path):
    plt.figure(figsize=(7, 4))
    df_sorted = df.sort_values("max_workers")
    for label in ["median_duration", "p90_duration", "p99_duration"]:
        if label in df_sorted.columns:
            plt.plot(df_sorted["max_workers"], df_sorted[label], marker="o", label=label)
    plt.xlabel("max_workers")
    plt.ylabel("duration (s)")
    plt.title("Latency percentiles vs max_workers")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_perfile_hist(perfile_csv, out_path):
    df = pd.read_csv(perfile_csv)
    if "duration" not in df.columns:
        print(f"Per-file CSV {perfile_csv} has no 'duration' column")
        return
    plt.figure(figsize=(7, 4))
    data = df["duration"].dropna()
    plt.hist(data, bins=60, density=False, alpha=0.7)
    # simple KDE-like smoothing using a line from a gaussian filter if available
    try:
        from scipy.ndimage import gaussian_filter1d
        import numpy as np
        counts, bins = np.histogram(data, bins=60)
        centers = 0.5 * (bins[1:] + bins[:-1])
        smooth = gaussian_filter1d(counts.astype(float), sigma=1.0)
        # scale smooth to match histogram scale
        plt.plot(centers, smooth, color="C1")
    except Exception:
        # scipy not available; skip smoothing
        pass
    plt.xlabel("per-file duration (s)")
    plt.title(f"Per-file duration distribution ({os.path.basename(perfile_csv)})")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_heatmap(df, value_col, out_path, index_col="max_workers", column_col="use_processes", aggfunc="mean"):
    # Pivot and plot heatmap (matplotlib imshow)
    if index_col not in df.columns or column_col not in df.columns:
        print(f"Cannot create heatmap: missing {index_col} or {column_col} in summary")
        return
    pivot = df.pivot_table(index=index_col, columns=column_col, values=value_col, aggfunc=aggfunc)
    if pivot.empty:
        print("Pivot table empty for heatmap")
        return
    plt.figure(figsize=(8, 5))
    im = plt.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(im, label=value_col)
    plt.yticks(range(len(pivot.index)), [str(x) for x in pivot.index])
    plt.xticks(range(len(pivot.columns)), [str(x) for x in pivot.columns])
    plt.xlabel(column_col)
    plt.ylabel(index_col)
    plt.title(f"{value_col} heatmap ({index_col} x {column_col})")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_speedup_and_efficiency(df, out_path_speedup, out_path_efficiency, cores_col="max_workers"):
    # Compute baseline as avg total_time where max_workers == 1 if present, else min
    if "total_time" not in df.columns:
        print("No total_time column to compute speedup")
        return
    df2 = df.copy()
    df2[cores_col] = pd.to_numeric(df2[cores_col], errors="coerce")
    # baseline per source_mode & location if present
    groups = []
    if "source_mode" in df2.columns and "location" in df2.columns:
        group_cols = ["source_mode", "location"]
    elif "source_mode" in df2.columns:
        group_cols = ["source_mode"]
    else:
        group_cols = []

    if group_cols:
        for name, g in df2.groupby(group_cols):
            try:
                base = g[g[cores_col] == 1]["total_time"].mean()
                if pd.isna(base):
                    base = g["total_time"].min()
            except Exception:
                base = g["total_time"].min()
            g = g.copy()
            g["speedup"] = base / g["total_time"]
            g["total_cores"] = g[cores_col]
            groups.append(g)
        df_speed = pd.concat(groups, ignore_index=True)
    else:
        try:
            base = df2[df2[cores_col] == 1]["total_time"].mean()
            if pd.isna(base):
                base = df2["total_time"].min()
        except Exception:
            base = df2["total_time"].min()
        df_speed = df2.copy()
        df_speed["speedup"] = base / df_speed["total_time"]
        df_speed["total_cores"] = df_speed[cores_col]

    # Speedup plot
    plt.figure(figsize=(7, 4))
    for key, grp in df_speed.groupby("use_processes") if "use_processes" in df_speed.columns else [(None, df_speed)]:
        plt.plot(grp[cores_col], grp["speedup"], marker="o", label=str(key))
    plt.xlabel("total cores")
    plt.ylabel("speedup")
    plt.title("Speedup vs cores")
    if "use_processes" in df_speed.columns:
        plt.legend(title="use_processes")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path_speedup)
    plt.close()

    # Efficiency = speedup / total_cores
    df_speed["efficiency"] = df_speed["speedup"] / df_speed["total_cores"].replace(0, pd.NA)
    plt.figure(figsize=(7, 4))
    for key, grp in df_speed.groupby("use_processes") if "use_processes" in df_speed.columns else [(None, df_speed)]:
        plt.plot(grp[cores_col], grp["efficiency"], marker="o", label=str(key))
    plt.xlabel("total cores")
    plt.ylabel("efficiency (speedup / cores)")
    plt.title("Efficiency vs cores")
    if "use_processes" in df_speed.columns:
        plt.legend(title="use_processes")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path_efficiency)
    plt.close()


def plot_failure_rate(df, out_path):
    if "failed" not in df.columns or "files_processed" not in df.columns:
        print("No failure/files_processed columns to plot failure rate")
        return
    df2 = df.copy()
    df2["failed_pct"] = df2["failed"] / (df2["failed"] + df2["files_processed"]).replace(0, pd.NA)
    x = range(len(df2))
    labels = [f"mw={int(r['max_workers'])}" if not pd.isna(r.get('max_workers')) else str(i) for i, r in df2.iterrows()]
    plt.figure(figsize=(8, 4))
    plt.bar(x, df2["failed"], label="failed")
    plt.bar(x, df2["files_processed"], bottom=df2["failed"], label="successful")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel("files")
    plt.title("Failed vs successful files per config")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_box_and_cdfs_from_perfiles(out_dir, out_prefix):
    # Collect per-file CSVs and group them by config parsed from filename
    pattern = os.path.join(out_dir, "*_perfile.csv")
    files = glob.glob(pattern)
    if not files:
        print("No per-file CSVs found for box/CDF plots")
        return
    groups = {}
    for fpath in files:
        # filename like mw=4_proc=False_branches=reduced_file_list_perfile.csv
        name = os.path.basename(fpath)
        parts = name.split("_")
        # try to extract mw and proc
        mw = next((p.split("=")[1] for p in parts if p.startswith("mw=")), None)
        proc = next((p.split("=")[1] for p in parts if p.startswith("proc=")), None)
        key = f"mw={mw}_proc={proc}"
        try:
            df = pd.read_csv(fpath)
            durations = df["duration"].dropna().values
        except Exception:
            continue
        groups[key] = durations

    # Boxplot
    keys = sorted(groups.keys())
    data = [groups[k] for k in keys]
    plt.figure(figsize=(max(6, len(keys)*0.6), 5))
    plt.boxplot(data, labels=keys, showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("per-file duration (s)")
    plt.title("Per-config per-file duration boxplot")
    plt.tight_layout()
    out_box = os.path.join(out_dir, f"{out_prefix}_boxplot.png")
    plt.savefig(out_box)
    plt.close()

    # CDFs
    import numpy as _np
    plt.figure(figsize=(7, 5))
    for k in keys:
        arr = _np.sort(groups[k])
        cdf = _np.arange(1, len(arr)+1) / len(arr)
        plt.plot(arr, cdf, label=k)
    plt.xlabel("duration (s)")
    plt.ylabel("CDF")
    plt.title("Per-config CDF of per-file durations")
    plt.legend(fontsize="small")
    plt.tight_layout()
    out_cdf = os.path.join(out_dir, f"{out_prefix}_cdfs.png")
    plt.savefig(out_cdf)
    plt.close()


def plot_perfile_timeline(perfile_csv, out_path):
    if not os.path.exists(perfile_csv):
        print(f"Per-file CSV not found: {perfile_csv}")
        return
    df = pd.read_csv(perfile_csv)
    if "duration" not in df.columns:
        print(f"Per-file CSV {perfile_csv} has no 'duration' column")
        return
    durations = df["duration"].fillna(0).values
    plt.figure(figsize=(10, 4))
    plt.plot(range(len(durations)), durations, marker=".")
    plt.xlabel("file index")
    plt.ylabel("duration (s)")
    plt.title(f"Per-file durations over file index ({os.path.basename(perfile_csv)})")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=str, default="bench_out/pyprocess/benchmark_results.csv", help="Summary CSV (from benchmark script)")
    parser.add_argument("--out-dir", type=str, default="bench_out/pyprocess", help="Directory containing per-file CSVs and where plots will be written")
    parser.add_argument("--plots-dir", type=str, default=None, help="Directory to write plots (defaults to <out-dir>/plots)")
    parser.add_argument("--perfile", type=str, default=None, help="Optional explicit per-file CSV to plot")
    args = parser.parse_args()

    if not os.path.exists(args.summary):
        print(f"Summary CSV not found: {args.summary}")
        return

    df = pd.read_csv(args.summary)
    # Normalize types
    for col in ["max_workers", "files_per_sec", "mean_duration"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    plots_dir = args.plots_dir or os.path.join(args.out_dir, "plots")
    ensure_dir(plots_dir)

    # Throughput plot
    out1 = os.path.join(plots_dir, "throughput_max_workers.png")
    plot_throughput(df, out1)
    print(f"Wrote {out1}")

    # Latency percentiles
    out2 = os.path.join(plots_dir, "latency_percentiles.png")
    plot_latency_percentiles(df, out2)
    print(f"Wrote {out2}")

    # Per-file histogram: select provided or pick a per-file CSV from out-dir
    perfile_csv = args.perfile
    if perfile_csv is None:
        candidates = glob.glob(os.path.join(args.out_dir, "*_perfile.csv"))
        candidates.sort(key=os.path.getmtime, reverse=True)
        if candidates:
            perfile_csv = candidates[0]

    if perfile_csv and os.path.exists(perfile_csv):
        out3 = os.path.join(plots_dir, "perfile_duration_hist.png")
        plot_perfile_hist(perfile_csv, out3)
        print(f"Wrote {out3}")
    else:
        print("No per-file CSV found to plot per-file histogram")

    # Additional plots
    # Heatmaps for files_per_sec and mean_duration (if available)
    if "files_per_sec" in df.columns:
        out_h1 = os.path.join(plots_dir, "heatmap_files_per_sec.png")
        plot_heatmap(df, "files_per_sec", out_h1)
        print(f"Wrote {out_h1}")
    if "mean_duration" in df.columns:
        out_h2 = os.path.join(plots_dir, "heatmap_mean_duration.png")
        plot_heatmap(df, "mean_duration", out_h2)
        print(f"Wrote {out_h2}")

    # Speedup and efficiency
    out_speed = os.path.join(plots_dir, "speedup_vs_cores.png")
    out_eff = os.path.join(plots_dir, "efficiency_vs_cores.png")
    plot_speedup_and_efficiency(df, out_speed, out_eff)
    print(f"Wrote {out_speed}")
    print(f"Wrote {out_eff}")

    # Failure rate
    out_fail = os.path.join(plots_dir, "failure_rate.png")
    plot_failure_rate(df, out_fail)
    print(f"Wrote {out_fail}")

    # Boxplot and CDFs from per-file CSVs (if any)
    plot_box_and_cdfs_from_perfiles(args.out_dir, "perfile")
    print(f"Wrote boxplot/CDFs to {args.out_dir}")

    # Per-file timeline for the picked perfile CSV
    if perfile_csv and os.path.exists(perfile_csv):
        out_tl = os.path.join(plots_dir, "perfile_timeline.png")
        plot_perfile_timeline(perfile_csv, out_tl)
        print(f"Wrote {out_tl}")


if __name__ == "__main__":
    main()
