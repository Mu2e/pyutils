"""Dask integration helpers for pyutils

Provides a minimal wrapper to run the existing file-worker pipeline
with Dask (local or distributed). This is intended as a minimal, drop-in
example showing how to replace the concurrent.futures-based parallelism
with Dask delayed / distributed.

Example usage:
    from pyutils.dask_integration import process_files_with_dask

    # local cluster
    results = process_files_with_dask(
        file_list=my_files,
        branches=['branch1','branch2'],
        n_workers=4,
    )

    # remote scheduler
    results = process_files_with_dask(
        file_list=my_files,
        branches=['branch1'],
        scheduler_address='tcp://scheduler:8786'
    )
"""
from __future__ import annotations

from typing import List, Optional, Tuple
import logging
import awkward as ak

import dask
from dask import delayed
from dask.distributed import Client, progress

# Import the module-level worker function from pyprocess
from .pyprocess import _worker_func

LOGGER = logging.getLogger("pyutils.dask")


def process_files_with_dask(
    file_list: List[str],
    branches,
    tree_path: str = "EventNtuple/ntuple",
    use_remote: bool = False,
    location: str = "local",
    schema: str = "root",
    n_workers: Optional[int] = None,
    threads_per_worker: int = 1,
    processes: bool = False,
    scheduler_address: Optional[str] = None,
    show_progress: bool = True,
) -> Optional[ak.Array]:
    """Process many files using Dask and return concatenated awkward array.

    This function mirrors Processor.process_data semantics but uses Dask.

    If `scheduler_address` is provided, it will connect to the remote
    Dask scheduler; otherwise it will start a local `Client` which will be
    closed before returning.
    """

    if not file_list:
        LOGGER.error("Empty file_list provided to process_files_with_dask")
        return None

    client: Optional[Client] = None
    created_client = False
    try:
        if scheduler_address:
            client = Client(scheduler_address)
            LOGGER.info(f"Connected to Dask scheduler at {scheduler_address}")
        else:
            # Create a local client; let Dask decide sensible defaults if None
            client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes)
            created_client = True
            LOGGER.info(f"Started local Dask client: {client}")

        # Build delayed tasks that call the existing module-level worker
        tasks = [
            delayed(_worker_func)(
                file_name,
                branches=branches,
                tree_path=tree_path,
                use_remote=use_remote,
                location=location,
                schema=schema,
                verbosity=0,
            )
            for file_name in file_list
        ]

        # Submit tasks to the cluster
        futures = client.compute(tasks)

        if show_progress:
            try:
                progress(futures)
            except Exception:
                # progress widgets may fail in some terminals; ignore
                pass

        results = client.gather(futures)

        # Filter out None results and concatenate with awkward
        results = [r for r in results if r is not None]
        if not results:
            LOGGER.warning("Dask returned no successful results")
            return None

        concatenated = ak.concatenate(results)
        LOGGER.info(f"Concatenated {len(results)} arrays into {len(concatenated)} events")
        return concatenated

    finally:
        if created_client and client is not None:
            client.close()
