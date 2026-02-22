"""Dask-based Processor wrapper for pyutils

Provides a `DaskProcessor` class that mirrors the public API of
`Processor` in `pyprocess.py` but runs the per-file worker function using
Dask (local or distributed). This file intentionally leaves the original
`pyprocess.py` unchanged and reuses its module-level `_worker_func`.

Usage:
    from pyutils.pydask import DaskProcessor

    dp = DaskProcessor()
    arr = dp.process_data(file_list_path="files.txt", branches=..., n_workers=4)

This is a lightweight adapter intended to be a drop-in alternative to
`Processor.process_data` for users who want to run on a Dask cluster.
"""
from __future__ import annotations

from typing import List, Optional, Dict, Tuple
import logging
import awkward as ak
from dask import delayed
from dask.distributed import Client, progress

from .pyprocess import _worker_func, Processor

LOGGER = logging.getLogger("pyutils.pydask")


class DaskProcessor:
    """Processor-like class that uses Dask for parallel file processing.

    Constructor arguments mirror `Processor` where relevant.
    """

    def __init__(
        self,
        tree_path: str = "EventNtuple/ntuple",
        use_remote: bool = False,
        location: str = "tape",
        schema: str = "root",
        verbosity: int = 1,
        worker_verbosity: int = 0,
    ):
        # Reuse Processor for file-list utilities and logging
        self._base = Processor(
            tree_path=tree_path,
            use_remote=use_remote,
            location=location,
            schema=schema,
            verbosity=verbosity,
            worker_verbosity=worker_verbosity,
        )

    def get_file_list(self, defname=None, file_list_path=None):
        return self._base.get_file_list(defname=defname, file_list_path=file_list_path)

    def process_data(
        self,
        file_name: Optional[str] = None,
        file_list_path: Optional[str] = None,
        defname: Optional[str] = None,
        branches: Optional[Dict] = None,
        n_workers: Optional[int] = None,
        threads_per_worker: int = 1,
        processes: bool = False,
        scheduler_address: Optional[str] = None,
        show_progress: bool = True,
        retries: int = 0,
        custom_worker_func=None,
    ) -> Optional[ak.Array]:
        """Process files using Dask. Mirrors Processor.process_data semantics.

        Either provide a single `file_name` or one of `file_list_path`/`defname`.
        If `scheduler_address` is provided, connects to that cluster; otherwise
        starts a local `Client` which is closed before returning.
        """

        # Validate input sources
        file_sources = sum(x is not None for x in [file_name, defname, file_list_path])
        if file_sources != 1:
            self._base.logger.log("Please provide exactly one of 'file_name', 'file_list_path', or defname'", "error")
            return None

        # Validate custom worker
        if custom_worker_func is not None:
            if not callable(custom_worker_func):
                self._base.logger.log("custom_worker_func is not callable", "error")
                return None

        # Single-file shortcut
        if file_name:
            if custom_worker_func is None:
                worker = lambda fname: _worker_func(
                    fname,
                    branches=branches,
                    tree_path=self._base.tree_path,
                    use_remote=self._base.use_remote,
                    location=self._base.location,
                    schema=self._base.schema,
                    verbosity=self._base.worker_verbosity,
                )
            else:
                worker = custom_worker_func

            try:
                result = worker(file_name)
                return result
            except Exception as e:
                self._base.logger.log(f"Error processing {file_name}: {e}", "error")
                return None

        # Prepare file list
        file_list = self.get_file_list(defname=defname, file_list_path=file_list_path)
        if not file_list:
            self._base.logger.log("Results list has length zero", "warning")
            return None

        # Choose worker function
        if custom_worker_func is None:
            # use module-level _worker_func as in pyprocess
            def _wrap(fname):
                return _worker_func(
                    fname,
                    branches=branches,
                    tree_path=self._base.tree_path,
                    use_remote=self._base.use_remote,
                    location=self._base.location,
                    schema=self._base.schema,
                    verbosity=self._base.worker_verbosity,
                )

            worker_func = _wrap
        else:
            worker_func = custom_worker_func

        client: Optional[Client] = None
        created_client = False
        try:
            if scheduler_address:
                client = Client(scheduler_address)
                LOGGER.info(f"Connected to Dask scheduler at {scheduler_address}")
            else:
                client = Client(n_workers=n_workers, threads_per_worker=threads_per_worker, processes=processes)
                created_client = True
                LOGGER.info(f"Started local Dask client: {client}")

            # Create delayed tasks
            tasks = [delayed(worker_func)(fname) for fname in file_list]

            futures = client.compute(tasks, retries=retries)

            if show_progress:
                try:
                    progress(futures)
                except Exception:
                    pass

            results = client.gather(futures)

            results = [r for r in results if r is not None]
            if not results:
                self._base.logger.log("Dask returned no successful results", "warning")
                return None

            concatenated = ak.concatenate(results)
            self._base.logger.log(f"Returning concatenated array containing {len(concatenated)} events", "success")
            return concatenated

        finally:
            if created_client and client is not None:
                client.close()


__all__ = ["DaskProcessor"]
