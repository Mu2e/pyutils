#!/usr/bin/env python3
"""
Benchmarking script for DaskProcessor with pyutils analysis pipeline.

This script benchmarks the complete pyutils analysis workflow using DaskProcessor:
- Data processing with various worker configurations
- Selection cuts
- Vector operations
- Histogram creation

Features:
- Sweeps n_workers and threads_per_worker values
- Toggles processes True/False
- Records execution times and event statistics to CSV
- Generates performance summary

Usage:
  python3 scripts/benchmark_pydask.py --file-list MDS3a.txt --out bench_out/pydask_results.csv

Requirements:
  - MDS3a.txt or other file list with ROOT file paths
  - Mu2e environment initialized with pyutils installed
"""

import argparse
import csv
import os
import time
from statistics import mean, median

from pyutils.pydask import DaskProcessor
from pyutils.pyselect import Select
from pyutils.pyvector import Vector
from pyutils.pyplot import Plot
from pyutils.pylogger import Logger
import awkward as ak


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


def run_analysis(file_list_path, n_workers, threads_per_worker, processes, logger):
    """
    Run complete pyutils analysis pipeline with DaskProcessor.
    
    Returns dict with timing and statistics.
    """
    
    branches = ["trksegs"]
    
    # Initialize processor
    processor = DaskProcessor(
        tree_path="EventNtuple/ntuple",
        use_remote=True,
        location="disk",
        verbosity=0,
        worker_verbosity=0
    )
    
    # Time the complete analysis
    t0 = time.perf_counter()
    
    try:
        # 1. Process data
        data = processor.process_data(
            file_list_path=file_list_path,
            branches=branches,
            n_workers=n_workers,
            threads_per_worker=threads_per_worker,
            processes=processes,
            show_progress=False
        )
        
        if data is None or len(data) == 0:
            return {
                "n_workers": n_workers,
                "threads_per_worker": threads_per_worker,
                "processes": processes,
                "total_time": time.perf_counter() - t0,
                "total_events": 0,
                "selected_events": 0,
                "efficiency": 0,
                "status": "failed"
            }
        
        total_events = len(data)
        
        # 2. Apply selection cuts
        selector = Select(verbosity=0)
        at_trkent = selector.select_surface(data=data, surface_name="TT_Front")
        trkent = data[at_trkent]
        selected_events = len(trkent)
        efficiency = selected_events / total_events if total_events > 0 else 0
        
        # 3. Compute vector operations
        vector = Vector(verbosity=0)
        mom_mag = vector.get_mag(branch=trkent["trksegs"], vector_name="mom")
        
        # 4. Create plots
        plotter = Plot()
        time_flat = ak.flatten(trkent["trksegs"]["time"], axis=None)
        mom_mag_flat = ak.flatten(mom_mag, axis=None)
        
        # Create placeholder plots (no actual output files)
        # plotter.plot_1D(...) - skipped for benchmark
        
        t1 = time.perf_counter()
        total_time = t1 - t0
        
        return {
            "n_workers": n_workers,
            "threads_per_worker": threads_per_worker,
            "processes": processes,
            "total_time": total_time,
            "total_events": total_events,
            "selected_events": selected_events,
            "efficiency": efficiency,
            "status": "success"
        }
        
    except Exception as e:
        logger.log(f"Analysis failed: {e}", "error")
        return {
            "n_workers": n_workers,
            "threads_per_worker": threads_per_worker,
            "processes": processes,
            "total_time": time.perf_counter() - t0,
            "total_events": 0,
            "selected_events": 0,
            "efficiency": 0,
            "status": f"failed: {str(e)}"
        }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark DaskProcessor with pyutils analysis pipeline"
    )
    parser.add_argument(
        "--file-list",
        dest="file_list",
        required=True,
        help="Path to file list (one ROOT file path per line)"
    )
    parser.add_argument(
        "--out",
        dest="out",
        help="Summary CSV output path",
        default="bench_out/pydask/benchmark_pydask_results.csv"
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        help="Directory for output files",
        default="bench_out/pydask"
    )
    parser.add_argument(
        "--n-workers-list",
        dest="nw_list",
        help="Comma-separated n_workers values (default: auto-sweep)",
        default=None
    )
    parser.add_argument(
        "--threads-per-worker-list",
        dest="tpw_list",
        help="Comma-separated threads_per_worker values (default: 1)",
        default=None
    )
    parser.add_argument(
        "--verbosity",
        dest="verbosity",
        type=int,
        default=1
    )
    args = parser.parse_args()
    
    # Initialize logger
    logger = Logger(print_prefix="[benchmark_pydask]", verbosity=args.verbosity)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Read file list
    with open(args.file_list, 'r') as f:
        file_lines = [l.strip() for l in f if l.strip()]
    
    n_files = len(file_lines)
    logger.log(f"Loaded {n_files} files from {args.file_list}", "success")
    
    # Determine n_workers sweep
    cpu_cnt = os.cpu_count() or 1
    if args.nw_list:
        nw_values = [int(x) for x in args.nw_list.split(",")]
    else:
        # Default sweep: powers of 2 up to CPU count and some larger values
        nw_values = [1, 2, 4, 8, 16, 32, cpu_cnt, min(2 * cpu_cnt, max(1, n_files))]
    nw_values = sorted(set(nw_values))
    
    # Determine threads_per_worker sweep
    if args.tpw_list:
        tpw_values = [int(x) for x in args.tpw_list.split(",")]
    else:
        tpw_values = [1]
    tpw_values = sorted(set(tpw_values))
    
    # Prepare benchmark configurations
    configs = []
    for nw in nw_values:
        for tpw in tpw_values:
            for use_processes in [False, True]:
                configs.append({
                    'n_workers': nw,
                    'threads_per_worker': tpw,
                    'processes': use_processes
                })
    
    logger.log(f"Running {len(configs)} configurations", "info")
    logger.log(f"Configurations:", "info")
    logger.log(f"  n_workers: {nw_values}", "info")
    logger.log(f"  threads_per_worker: {tpw_values}", "info")
    logger.log(f"  processes: [False, True]", "info")
    
    # Run benchmarks
    results = []
    for i, config in enumerate(configs, 1):
        logger.log(f"\n[{i}/{len(configs)}] Running: nw={config['n_workers']}, tpw={config['threads_per_worker']}, proc={config['processes']}", "info")
        
        result = run_analysis(
            file_list_path=args.file_list,
            n_workers=config['n_workers'],
            threads_per_worker=config['threads_per_worker'],
            processes=config['processes'],
            logger=logger
        )
        
        results.append(result)
        
        logger.log(f"  Time: {result['total_time']:.2f}s | Events: {result['total_events']} | Selected: {result['selected_events']} | Status: {result['status']}", "info")
    
    # Write results to CSV
    if results:
        fieldnames = [
            'n_workers',
            'threads_per_worker',
            'processes',
            'total_time',
            'total_events',
            'selected_events',
            'efficiency',
            'status'
        ]
        
        with open(args.out, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        logger.log(f"\nResults written to: {args.out}", "success")
        
        # Print summary statistics
        successful = [r for r in results if r['status'] == 'success']
        if successful:
            times = [r['total_time'] for r in successful]
            logger.log("\nPerformance Summary:", "success")
            logger.log(f"  Total runs: {len(results)} ({len(successful)} successful)", "info")
            logger.log(f"  Min time: {min(times):.2f}s", "info")
            logger.log(f"  Max time: {max(times):.2f}s", "info")
            logger.log(f"  Mean time: {mean(times):.2f}s", "info")
            logger.log(f"  Median time: {median(times):.2f}s", "info")
            
            # Find best configuration
            best = min(successful, key=lambda x: x['total_time'])
            logger.log(f"\nBest configuration: nw={best['n_workers']}, tpw={best['threads_per_worker']}, proc={best['processes']}", "success")
            logger.log(f"  Time: {best['total_time']:.2f}s", "info")
    
    logger.log("\nBenchmark complete!", "success")


if __name__ == "__main__":
    main()
