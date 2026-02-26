#!/usr/bin/env python3
"""
Benchmarking script for `pyutils.pyprocess.Processor`.

Features:
- Sweeps `max_workers` values
- Toggles `use_processes` True/False
- Supports comparing full vs reduced `branches`
- Can run with `file_list_path` (local list) or `defname` (SAM)
- Records per-file durations and summary metrics to CSV

Usage examples:
  python3 scripts/benchmark_pyprocess.py --file-list files.txt --out results.csv

Note: This script uses the `Processor` and `_worker_func` from the `pyutils` package
so run it from the repository root where `pyutils` is importable.
"""

import argparse
import csv
import functools
import json
import os
import time
from statistics import mean, median

from pyutils.pyprocess import Processor, _worker_func


def timing_wrapper(file_name, branches, tree_path, use_remote, location, schema, verbosity):
    """Top-level timing wrapper (picklable) that calls internal _worker_func."""
    t0 = time.perf_counter()
    result = _worker_func(
        file_name=file_name,
        branches=branches,
        tree_path=tree_path,
        use_remote=use_remote,
        location=location,
        schema=schema,
        verbosity=verbosity,
    )
    t1 = time.perf_counter()

    # Try to infer event count
    try:
        n_events = len(result)
    except Exception:
        try:
            # If awkward array or dict-like, try first field
            n_events = len(next(iter(result.values())))
        except Exception:
            n_events = None

    return {"file": file_name, "duration": (t1 - t0), "n_events": n_events, "result": result}


# Global config used by `bench_worker` so it has a single-argument signature
BENCH_CONFIG = {}


def bench_worker(file_name):
    """Single-argument module-level worker that calls timing_wrapper using global BENCH_CONFIG.

    This satisfies `Processor`'s requirement that `custom_worker_func` takes exactly one
    argument and is picklable for multiprocessing.
    """
    return timing_wrapper(file_name, **BENCH_CONFIG)


def percentile(sorted_list, p):
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


def run_single_config(file_source_args, branches, max_workers, use_processes, out_dir, tree_path, use_remote, location, schema, verbosity, worker_verbosity):
    pfx = f"mw={max_workers}_proc={use_processes}_branches={'reduced' if branches else 'full'}_{file_source_args.get('mode') or 'unknown'}"
    out_prefix = os.path.join(out_dir, pfx)
    os.makedirs(out_dir, exist_ok=True)

    processor = Processor(tree_path=tree_path, use_remote=use_remote, location=location, schema=schema, verbosity=verbosity, worker_verbosity=worker_verbosity)

    # Populate global BENCH_CONFIG used by the module-level `bench_worker`
    global BENCH_CONFIG
    BENCH_CONFIG = dict(
        branches=branches,
        tree_path=tree_path,
        use_remote=use_remote,
        location=location,
        schema=schema,
        verbosity=worker_verbosity,
    )
    t0 = time.perf_counter()
    results = processor.process_data(
        file_name=None,
        file_list_path=file_source_args.get("file_list_path"),
        defname=file_source_args.get("defname"),
        branches=branches,
        max_workers=max_workers,
        custom_worker_func=bench_worker,
        use_processes=use_processes,
    )
    t1 = time.perf_counter()

    # results: list of dicts returned by timing_wrapper (or None entries)
    durations = []
    failed = 0
    total_events = 0
    per_file_rows = []

    if results is None:
        # No results
        return None

    for r in results:
        if not r:
            failed += 1
            continue
        dur = r.get("duration")
        durations.append(dur)
        n_ev = r.get("n_events")
        if isinstance(n_ev, int):
            total_events += n_ev
        per_file_rows.append({"file": r.get("file"), "duration": dur, "n_events": n_ev})

    durations_sorted = sorted(durations)
    total_time = t1 - t0
    files_processed = len(durations)
    files_per_sec = files_processed / total_time if total_time > 0 else None

    summary = {
        "max_workers": max_workers,
        "use_processes": use_processes,
        "branches_mode": "reduced" if branches else "full",
        "source_mode": file_source_args.get("mode"),
        "files_processed": files_processed,
        "failed": failed,
        "total_time": total_time,
        "files_per_sec": files_per_sec,
        "mean_duration": mean(durations) if durations else None,
        "median_duration": median(durations) if durations else None,
        "p90_duration": percentile(durations_sorted, 90),
        "p99_duration": percentile(durations_sorted, 99),
        "total_events": total_events,
    }

    # Write per-file CSV
    per_file_csv = out_prefix + "_perfile.csv"
    with open(per_file_csv, "w", newline="") as pf:
        writer = csv.DictWriter(pf, fieldnames=["file", "duration", "n_events"])
        writer.writeheader()
        for row in per_file_rows:
            writer.writerow(row)

    return summary


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file-list", dest="file_list", help="Path to a newline-separated local file list")
    group.add_argument("--defname", dest="defname", help="SAM definition name (defname)")
    parser.add_argument("--reduced-branches", dest="reduced_branches", help="Comma-separated reduced branch list (e.g. Edep,track)", default=None)
    parser.add_argument("--out", dest="out", help="Summary CSV output path", default="benchmark_results.csv")
    parser.add_argument("--out-dir", dest="out_dir", help="Directory to store per-run outputs", default="benchmark_out")
    parser.add_argument("--max-workers-list", dest="mw_list", help="Comma-separated max_workers values; default auto", default=None)
    parser.add_argument("--tree-path", dest="tree_path", default="EventNtuple/ntuple")
    parser.add_argument("--use-remote", dest="use_remote", action="store_true")
    parser.add_argument("--location", dest="location", default="tape")
    parser.add_argument("--location-list", dest="location_list", help="Comma-separated list of locations to sweep (overrides --location)", default=None)
    parser.add_argument("--schema", dest="schema", default="root")
    parser.add_argument("--verbosity", dest="verbosity", type=int, default=1)
    parser.add_argument("--worker-verbosity", dest="worker_verbosity", type=int, default=0)
    args = parser.parse_args()

    # Prepare file source args
    if args.file_list:
        file_source_args = {"file_list_path": args.file_list, "mode": "file_list"}
        # count files
        with open(args.file_list, "r") as f:
            file_lines = [l.strip() for l in f if l.strip()]
        n_files = len(file_lines)
    else:
        file_source_args = {"defname": args.defname, "mode": "defname"}
        # n_files unknown until query; use a conservative guess for sweep
        n_files = 100

    cpu_cnt = os.cpu_count() or 1
    # default sweep
    if args.mw_list:
        mw_values = [int(x) for x in args.mw_list.split(",")]
    else:
        mw_values = [1, 2, 4, 8, 16, 32, cpu_cnt, min(2 * cpu_cnt, max(1, n_files))]
    # unique and sorted
    mw_values = sorted(set(mw_values))

    # Always use a reduced grouped `branches` dict. If the user supplies
    # `--reduced-branches` use those fields; otherwise default to the
    # `trk` group from dask-tests/run_dask_example.py.
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

    # Sweep combinations - only the reduced branch set is used
    branch_options = [reduced_branches]

    combos = []
    for location in locations:
        for use_processes in (False, True):
            for branches in branch_options:
                for mw in mw_values:
                    combos.append((mw, use_processes, branches, location))

    # Run combos and collect summaries
    summaries = []
    for mw, use_processes, branches, location in combos:
        # If the user specifies location 'local' treat it as local files (no remote mdh lookup)
        per_run_use_remote = False if location == "local" else args.use_remote
        print(f"Running: max_workers={mw}, use_processes={use_processes}, branches={'reduced' if branches else 'full'}, location={location}, use_remote={per_run_use_remote}")
        try:
            s = run_single_config(
                file_source_args=file_source_args,
                branches=branches,
                max_workers=mw,
                use_processes=use_processes,
                out_dir=args.out_dir,
                tree_path=args.tree_path,
                use_remote=per_run_use_remote,
                location=location,
                schema=args.schema,
                verbosity=args.verbosity,
                worker_verbosity=args.worker_verbosity,
            )
            if s:
                summaries.append(s)
        except Exception as e:
            print(f"Error running config mw={mw} proc={use_processes} branches={'reduced' if branches else 'full'}: {e}")

    # Write summary CSV
    if summaries:
        keys = list(summaries[0].keys())
        with open(args.out, "w", newline="") as outcsv:
            writer = csv.DictWriter(outcsv, fieldnames=keys)
            writer.writeheader()
            for row in summaries:
                writer.writerow(row)

    print(f"Done. Wrote summary to {args.out} (per-file outputs in {args.out_dir})")


if __name__ == "__main__":
    main()
