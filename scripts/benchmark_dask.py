#!/usr/bin/env python3
"""
Benchmarking script for `pyutils.dask_integration.process_files_with_dask`.

Features:
- Sweeps `n_workers` and `threads_per_worker` values
- Toggles `processes` True/False
- Supports comparing full vs reduced `branches`
- Uses `file_list_path` (local list)
- Records per-file durations and summary metrics to CSV

Usage examples:
  python3 scripts/benchmark_dask.py --file-list files.txt --out results.csv

Note: This script uses the `process_files_with_dask` from the `pyutils` package
so run it from the repository root where `pyutils` is importable.
"""

import argparse
import csv
import json
import os
import time
from statistics import mean, median

from pyutils.dask_integration import process_files_with_dask


def percentile(sorted_list, p):
    """Calculate percentile of a sorted list."""
    if not sorted_list:
        return None
    k = (len(sorted_list) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_list) - 1)
    if f == c:
        return sorted_list[int(k)]
    d0 = sorted_list[f] * (c - k)
    d1 = sorted_list[c] * (k - f)
    return d0 + d1


def run_single_config(file_list, branches, n_workers, threads_per_worker, processes, out_dir, location="local", use_remote=True, verbosity=0):
    """Run a single dask configuration and record timing."""
    pfx = f"nw={n_workers}_tpw={threads_per_worker}_proc={processes}_loc={location}_remote={use_remote}_branches={'reduced' if branches else 'full'}"
    out_prefix = os.path.join(out_dir, pfx)
    os.makedirs(out_dir, exist_ok=True)

    # Run the dask processing with timing
    t0 = time.perf_counter()
    result = process_files_with_dask(
        file_list=file_list,
        branches=branches,
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        processes=processes,
        use_remote=use_remote,
        location=location,
        show_progress=False,
    )
    t1 = time.perf_counter()

    total_time = t1 - t0

    # Try to infer total event count
    try:
        if result is None:
            n_events = 0
        else:
            n_events = len(result)
    except Exception:
        try:
            # If dict-like (branches dict), try first field
            if isinstance(result, dict):
                n_events = len(next(iter(result.values())))
            else:
                n_events = None
        except Exception:
            n_events = None

    files_processed = len(file_list)
    files_per_sec = files_processed / total_time if total_time > 0 else None

    summary = {
        "n_workers": n_workers,
        "threads_per_worker": threads_per_worker,
        "processes": processes,
        "location": location,
        "use_remote": use_remote,
        "branches_mode": "reduced" if branches else "full",
        "files_processed": files_processed,
        "total_time": total_time,
        "files_per_sec": files_per_sec,
        "total_events": n_events if isinstance(n_events, int) else 0,
    }

    if verbosity > 0:
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Files/sec: {files_per_sec:.2f}")
        print(f"  Total events: {n_events}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark dask integration with various worker configurations"
    )
    parser.add_argument(
        "--file-list",
        dest="file_list",
        required=True,
        help="Path to a newline-separated local file list",
    )
    parser.add_argument(
        "--reduced-branches",
        dest="reduced_branches",
        help="Comma-separated reduced branch list (e.g. trk.nactive,trk.pdg)",
        default=None,
    )
    parser.add_argument(
        "--out",
        dest="out",
        help="Summary CSV output path",
        default="bench_out/dask/benchmark_dask_results.csv",
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        help="Directory to store per-run outputs",
        default="bench_out/dask",
    )
    parser.add_argument(
        "--n-workers-list",
        dest="nw_list",
        help="Comma-separated n_workers values; default auto",
        default=None,
    )
    parser.add_argument(
        "--threads-per-worker-list",
        dest="tpw_list",
        help="Comma-separated threads_per_worker values; default: 1",
        default=None,
    )
    parser.add_argument(
        "--location",
        dest="location",
        help="Location/source for files (e.g., tape, cache, local)",
        default="local",
    )
    parser.add_argument(
        "--location-list",
        dest="location_list",
        help="Comma-separated list of locations to sweep (overrides --location)",
        default=None,
    )
    parser.add_argument(
        "--use-remote",
        dest="use_remote",
        action="store_true",
        help="Enable remote file access",
    )
    parser.add_argument(
        "--verbosity",
        dest="verbosity",
        type=int,
        default=1,
    )
    args = parser.parse_args()

    # Read and count files
    with open(args.file_list, "r") as f:
        file_lines = [l.strip() for l in f if l.strip()]
    n_files = len(file_lines)
    print(f"Loaded {n_files} files from {args.file_list}")

    # Determine default n_workers if not specified
    cpu_cnt = os.cpu_count() or 1
    if args.nw_list:
        nw_values = [int(x) for x in args.nw_list.split(",")]
    else:
        # Default sweep: 1, 2, 4, 8, 16, 32, cpu_count, and min(2*cpu_count, n_files)
        nw_values = [1, 2, 4, 8, 16, 32, cpu_cnt, min(2 * cpu_cnt, max(1, n_files))]
    nw_values = sorted(set(nw_values))

    # Determine threads_per_worker values
    if args.tpw_list:
        tpw_values = [int(x) for x in args.tpw_list.split(",")]
    else:
        tpw_values = [1]
    tpw_values = sorted(set(tpw_values))

    # Always use a reduced grouped `branches` dict
    if args.reduced_branches:
        lst = [b.strip() for b in args.reduced_branches.split(",") if b.strip()]
        reduced_branches = {"trk": lst} if lst else None
    else:
        reduced_branches = {
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

    # Prepare location sweep
    if args.location_list:
        locations = [l.strip() for l in args.location_list.split(",") if l.strip()]
    else:
        locations = [args.location]

    # Build sweep combinations
    branch_options = [reduced_branches]

    combos = []
    for location in locations:
        for processes in (False, True):
            for branches in branch_options:
                for nw in nw_values:
                    for tpw in tpw_values:
                        # Skip unreasonable combinations
                        total_cores = nw * tpw
                        if total_cores > 2 * cpu_cnt:
                            continue
                        combos.append((nw, tpw, processes, branches, location))

    print(f"Will run {len(combos)} configurations")
    print(f"CPU count: {cpu_cnt}")
    print(f"n_workers values: {nw_values}")
    print(f"threads_per_worker values: {tpw_values}")
    print(f"processes options: False, True")
    print(f"locations: {locations}")
    print()

    # Run combos and collect summaries
    summaries = []
    for i, (nw, tpw, processes, branches, location) in enumerate(combos, 1):
        config_str = f"nw={nw}, tpw={tpw}, proc={processes}, loc={location}, remote={args.use_remote}, branches={'reduced' if branches else 'full'}"
        print(f"[{i}/{len(combos)}] Running: {config_str}")
        try:
            s = run_single_config(
                file_list=file_lines,
                branches=branches,
                n_workers=nw,
                threads_per_worker=tpw,
                processes=processes,
                location=location,
                use_remote=args.use_remote,
                out_dir=args.out_dir,
                verbosity=args.verbosity,
            )
            if s:
                summaries.append(s)
        except Exception as e:
            print(f"  Error: {e}")

    # Write summary CSV
    if summaries:
        keys = list(summaries[0].keys())
        with open(args.out, "w", newline="") as outcsv:
            writer = csv.DictWriter(outcsv, fieldnames=keys)
            writer.writeheader()
            for row in summaries:
                writer.writerow(row)
        print(f"\nDone. Wrote summary to {args.out}")
    else:
        print("\nNo results to write.")


if __name__ == "__main__":
    main()
