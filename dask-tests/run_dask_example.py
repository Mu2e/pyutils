from pyutils.dask_integration import process_files_with_dask
from pyutils.pyprocess import Processor
import logging
import time
import tempfile
import os
from typing import List, Dict, Tuple, Optional
import awkward as ak

logging.basicConfig(level=logging.INFO)


def run_processor(file_list: List[str], branches: Dict, max_workers: Optional[int] = None, use_processes: bool = False) -> Tuple[Optional[ak.Array], float]:
    """Run the existing Processor.process_data using a temporary file-list and return (result, elapsed_seconds)."""
    # Write file list to a temporary file so Processor can read it via file_list_path
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        for p in file_list:
            f.write(p + "\n")
        tmp_path = f.name

    try:
        proc = Processor()
        t0 = time.perf_counter()
        result = proc.process_data(file_list_path=tmp_path, branches=branches, max_workers=max_workers, use_processes=use_processes)
        elapsed = time.perf_counter() - t0
        return result, elapsed
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


def run_dask(file_list: List[str], branches: Dict, n_workers: int = 4, threads_per_worker: int = 1, processes: bool = False, show_progress: bool = True) -> Tuple[Optional[ak.Array], float]:
    """Run the Dask-based wrapper and return (result, elapsed_seconds)."""
    t0 = time.perf_counter()
    result = process_files_with_dask(
        file_list=file_list,
        branches=branches,
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=processes,
        show_progress=show_progress,
    )
    elapsed = time.perf_counter() - t0
    return result, elapsed


def compare_results(a: Optional[ak.Array], b: Optional[ak.Array], show_items: int = 2) -> None:
    """Compare two awkward arrays (or None). Prints summary comparison.

    Comparison is shallow: checks for None, lengths, top-level fields, and prints first items.
    """
    if a is None and b is None:
        print("Both results are None")
        return
    if a is None:
        print("Processor result is None; Dask returned data")
        return
    if b is None:
        print("Dask result is None; Processor returned data")
        return

    try:
        la = len(a)
    except Exception:
        la = None
    try:
        lb = len(b)
    except Exception:
        lb = None

    print(f"Lengths: Processor={la}, Dask={lb}")

    try:
        fa = ak.fields(a)
    except Exception:
        fa = None
    try:
        fb = ak.fields(b)
    except Exception:
        fb = None

    print(f"Top-level fields: Processor={fa}, Dask={fb}")

    # Print first few items for manual inspection
    def _show_first(arr, n):
        try:
            lst = ak.to_list(arr[:n])
            return lst
        except Exception:
            return None

    print("First items from Processor:", _show_first(a, show_items))
    print("First items from Dask:", _show_first(b, show_items))


if __name__ == "__main__":
    # Adjust these paths/branches for your environment
    file_list = [
        "/Users/sophie/pyutils-dev/nts.mu2e.ensembleMDS3aMix1BBTriggered.MDC2025-001.001430_00000552.root",
        "/Users/sophie/pyutils-dev/nts.mu2e.ensembleMDS3aMix1BBTriggered.MDC2025-001.001430_00002561.root",
        "/Users/sophie/pyutils-dev/nts.mu2e.ensembleMDS3aMix1BBTriggered.MDC2025-001.001430_00000974.root",
    ]

    branches = {
        "trk": [
            "trk.nactive",
            "trk.pdg",
            "trk.status",
            "trkqual.valid",
            "trkqual.result",
            "trkpid.valid",
            "trkpid.result",
        ]
    }

    # Quick pre-parse: if user only requested plotting, handle it and exit
    import argparse
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--plot", action="store_true")
    pre_parser.add_argument("--output", type=str, default=None)
    pre_args, _ = pre_parser.parse_known_args()
    if pre_args.plot:
        csv_path = pre_args.output if pre_args.output is not None else None
        if csv_path is None:
            import glob
            candidates = glob.glob("benchmark_results_*.csv")
            candidates.sort(key=os.path.getmtime, reverse=True)
            if candidates:
                csv_path = candidates[0]
            elif os.path.exists("test.csv"):
                csv_path = "test.csv"
        if csv_path is None:
            print("No CSV provided and no benchmark_results_*.csv or test.csv found. Use --output <csv> to specify file.")
            exit(1)

        # Minimal plotting (avoid importing heavy libs at module import time)
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
        except Exception:
            print("Plotting requires pandas and matplotlib. Install them with: pip install pandas matplotlib")
            exit(1)

        df = pd.read_csv(csv_path)
        df["n_workers"] = pd.to_numeric(df.get("n_workers"), errors="coerce")
        df["threads_per_worker"] = pd.to_numeric(df.get("threads_per_worker"), errors="coerce")
        df["max_workers"] = pd.to_numeric(df.get("max_workers"), errors="coerce")
        df["avg_time_s"] = pd.to_numeric(df.get("avg_time_s"), errors="coerce")

        def total_cores(row):
            if pd.notna(row.get("n_workers")) and pd.notna(row.get("threads_per_worker")):
                return int(row["n_workers"] * row["threads_per_worker"])
            if pd.notna(row.get("max_workers")):
                return int(row["max_workers"])
            return None

        df["total_cores"] = df.apply(total_cores, axis=1)
        baseline_rows = df[(df['mode'] == "processor") & (df.max_workers == 1)]
        if not baseline_rows.empty:
            baseline = float(baseline_rows["avg_time_s"].iloc[0])
        else:
            baseline = float(df["avg_time_s"].max())
        df["speedup"] = baseline / df["avg_time_s"]

        out_prefix = os.path.splitext(os.path.basename(csv_path))[0]
        plots_dir = os.path.join(os.getcwd(), "benchmark_plots")
        os.makedirs(plots_dir, exist_ok=True)
        df_plot = df.dropna(subset=["total_cores", "avg_time_s"]).sort_values("total_cores")

        fig, ax = plt.subplots()
        ax.bar(df_plot["total_cores"].astype(str), df_plot["avg_time_s"])
        ax.set_xlabel("Total cores")
        ax.set_ylabel("Avg time (s)")
        ax.set_title("Average runtime vs total cores")
        fig.tight_layout()
        out1 = os.path.join(plots_dir, f"{out_prefix}_avg_time.png")
        fig.savefig(out1)
        plt.close(fig)
        print(f"Wrote {out1}")

        fig, ax = plt.subplots()
        ax.plot(df_plot["total_cores"], df_plot["speedup"], marker="o")
        ax.set_xlabel("Total cores")
        ax.set_ylabel("Speedup (baseline/time)")
        ax.set_title("Speedup vs total cores")
        fig.tight_layout()
        out2 = os.path.join(plots_dir, f"{out_prefix}_speedup.png")
        fig.savefig(out2)
        plt.close(fig)
        print(f"Wrote {out2}")

        dask_df = df[df['mode'] == "dask"].dropna(subset=["n_workers", "threads_per_worker", "avg_time_s"])
        if not dask_df.empty:
            pivot = dask_df.pivot_table(index="n_workers", columns="threads_per_worker", values="avg_time_s")
            fig, ax = plt.subplots()
            cax = ax.imshow(pivot.values, aspect="auto", origin="lower")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([str(int(x)) for x in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([str(int(x)) for x in pivot.index])
            ax.set_xlabel("threads_per_worker")
            ax.set_ylabel("n_workers")
            ax.set_title("Dask avg time heatmap")
            fig.colorbar(cax, label="avg_time_s")
            out3 = os.path.join(plots_dir, f"{out_prefix}_heatmap.png")
            fig.tight_layout()
            fig.savefig(out3)
            plt.close(fig)
            print(f"Wrote {out3}")
        # exit after plotting when using pre-parser
        exit(0)

    print("Running Processor (local concurrent.futures)...")
    p_res, p_time = run_processor(file_list, branches, max_workers=4, use_processes=False)
    print(f"Processor time: {p_time:.3f}s")

    print("Running Dask wrapper...")
    d_res, d_time = run_dask(file_list, branches, n_workers=4, threads_per_worker=1, processes=False, show_progress=False)
    print(f"Dask time: {d_time:.3f}s")

    print("Comparing results...")
    compare_results(p_res, d_res, show_items=2)

    # Benchmark mode via CLI
    import argparse

    parser = argparse.ArgumentParser(description="Run Processor vs Dask example/benchmark")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark sweep")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat each config this many times and report average")
    parser.add_argument("--dry-run", action="store_true", help="Print benchmark plan and exit")
    parser.add_argument("--output", type=str, default=None, help="CSV path to write benchmark results (default: timestamped file)")
    parser.add_argument("--plot", action="store_true", help="Generate plots from existing CSV and exit")
    args = parser.parse_args()

    def plot_benchmarks(csv_path: str) -> None:
        try:
            import pandas as pd
            import matplotlib.pyplot as plt
        except Exception as e:
            print("Plotting requires pandas and matplotlib. Install them with: pip install pandas matplotlib")
            print(f"Import error: {e}")
            return

        if not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            return

        df = pd.read_csv(csv_path)
        # Normalize numeric columns
        df["n_workers"] = pd.to_numeric(df.get("n_workers"), errors="coerce")
        df["threads_per_worker"] = pd.to_numeric(df.get("threads_per_worker"), errors="coerce")
        df["max_workers"] = pd.to_numeric(df.get("max_workers"), errors="coerce")
        df["avg_time_s"] = pd.to_numeric(df.get("avg_time_s"), errors="coerce")

        # total_cores: for dask use n_workers*threads_per_worker, else use max_workers
        def total_cores(row):
            if pd.notna(row.get("n_workers")) and pd.notna(row.get("threads_per_worker")):
                return int(row["n_workers"] * row["threads_per_worker"])
            if pd.notna(row.get("max_workers")):
                return int(row["max_workers"])
            return None

        df["total_cores"] = df.apply(total_cores, axis=1)

        # baseline: processor with max_workers==1 if available
        baseline_rows = df[(df['mode'] == "processor") & (df.max_workers == 1)]
        if not baseline_rows.empty:
            baseline = float(baseline_rows["avg_time_s"].iloc[0])
        else:
            baseline = float(df["avg_time_s"].max())

        df["speedup"] = baseline / df["avg_time_s"]

        out_prefix = os.path.splitext(os.path.basename(csv_path))[0]
        plots_dir = os.path.join(os.getcwd(), "benchmark_plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Avg time vs total cores
        fig, ax = plt.subplots()
        df_plot = df.dropna(subset=["total_cores", "avg_time_s"]).sort_values("total_cores")
        ax.bar(df_plot["total_cores"].astype(str), df_plot["avg_time_s"])
        ax.set_xlabel("Total cores")
        ax.set_ylabel("Avg time (s)")
        ax.set_title("Average runtime vs total cores")
        fig.tight_layout()
        out1 = os.path.join(plots_dir, f"{out_prefix}_avg_time.png")
        fig.savefig(out1)
        plt.close(fig)
        print(f"Wrote {out1}")

        # Speedup vs total cores
        fig, ax = plt.subplots()
        ax.plot(df_plot["total_cores"], df_plot["speedup"], marker="o")
        ax.set_xlabel("Total cores")
        ax.set_ylabel("Speedup (baseline/time)")
        ax.set_title("Speedup vs total cores")
        fig.tight_layout()
        out2 = os.path.join(plots_dir, f"{out_prefix}_speedup.png")
        fig.savefig(out2)
        plt.close(fig)
        print(f"Wrote {out2}")

        # Heatmap for Dask configs (n_workers x threads_per_worker)
        dask_df = df[df['mode'] == "dask"].dropna(subset=["n_workers", "threads_per_worker", "avg_time_s"])
        if not dask_df.empty:
            pivot = dask_df.pivot_table(index="n_workers", columns="threads_per_worker", values="avg_time_s")
            fig, ax = plt.subplots()
            cax = ax.imshow(pivot.values, aspect="auto", origin="lower")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([str(int(x)) for x in pivot.columns])
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([str(int(x)) for x in pivot.index])
            ax.set_xlabel("threads_per_worker")
            ax.set_ylabel("n_workers")
            ax.set_title("Dask avg time heatmap")
            fig.colorbar(cax, label="avg_time_s")
            out3 = os.path.join(plots_dir, f"{out_prefix}_heatmap.png")
            fig.tight_layout()
            fig.savefig(out3)
            plt.close(fig)
            print(f"Wrote {out3}")

    # Handle plotting separately
    if args.plot:
        csv_path = args.output if args.output is not None else None
        if csv_path is None:
            # try to find most recent benchmark_results_*.csv
            import glob
            candidates = glob.glob("benchmark_results_*.csv")
            candidates.sort(key=os.path.getmtime, reverse=True)
            if candidates:
                csv_path = candidates[0]
            else:
                # allow user to point at test.csv if present
                if os.path.exists("test.csv"):
                    csv_path = "test.csv"
                else:
                    print("No CSV provided and no benchmark_results_*.csv found. Use --output <csv> to specify file.")
                    csv_path = None

        if csv_path:
            plot_benchmarks(csv_path)
        # After plotting, exit
        exit(0)

    if args.benchmark:
        # Define a small grid of Dask configs to sweep
        dask_grid = [
            (1, 1),
            (2, 1),
            (4, 1),
            (2, 2),
        ]

        print("Benchmark plan:")
        for (nw, tp) in dask_grid:
            print(f"  Dask: n_workers={nw}, threads_per_worker={tp}")
        print(f"Processor: max_workers=1..4 (threads)")

        if args.dry_run:
            print("Dry run requested; exiting without executing benchmarks")
        else:
            # Run processor baseline (single run per worker setting)
            proc_results = {}
            for mw in [1, 2, 4]:
                print(f"Running Processor with max_workers={mw}...", flush=True)
                times = []
                for _ in range(args.repeat):
                    _, t = run_processor(file_list, branches, max_workers=mw, use_processes=False)
                    times.append(t)
                proc_results[mw] = sum(times) / len(times)
                print(f"  avg time: {proc_results[mw]:.3f}s")

            # Run Dask grid
            dask_results = {}
            for (nw, tp) in dask_grid:
                print(f"Running Dask n_workers={nw}, threads_per_worker={tp}...", flush=True)
                times = []
                for _ in range(args.repeat):
                    _, t = run_dask(file_list, branches, n_workers=nw, threads_per_worker=tp, processes=False, show_progress=False)
                    times.append(t)
                avg = sum(times) / len(times)
                dask_results[(nw, tp)] = avg
                print(f"  avg time: {avg:.3f}s")

            # Summary
            print("\nBenchmark summary:")
            print("Processor (avg seconds):")
            for k, v in proc_results.items():
                print(f"  max_workers={k}: {v:.3f}s")
            print("Dask (avg seconds):")
            for k, v in dask_results.items():
                print(f"  n_workers={k[0]}, threads_per_worker={k[1]}: {v:.3f}s")

            # Write CSV results
            import csv
            from datetime import datetime

            out_path = args.output
            if out_path is None:
                stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
                out_path = f"benchmark_results_{stamp}.csv"

            print(f"Writing benchmark CSV to {out_path}")
            with open(out_path, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=["mode", "n_workers", "threads_per_worker", "max_workers", "avg_time_s", "repeat_count"])
                writer.writeheader()
                # Processor rows
                for k, v in proc_results.items():
                    writer.writerow({
                        "mode": "processor",
                        "n_workers": "",
                        "threads_per_worker": "",
                        "max_workers": k,
                        "avg_time_s": f"{v:.6f}",
                        "repeat_count": args.repeat,
                    })
                # Dask rows
                for k, v in dask_results.items():
                    writer.writerow({
                        "mode": "dask",
                        "n_workers": k[0],
                        "threads_per_worker": k[1],
                        "max_workers": "",
                        "avg_time_s": f"{v:.6f}",
                        "repeat_count": args.repeat,
                    })

