#! /usr/bin/env python
import os
import subprocess
import gc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import awkward as ak
import inspect
import tqdm
import functools 
import time
import random

from . import _env_manager
from .pyimport import Importer
from .pylogger import Logger

# def _worker_func(file_name, branches, tree_path, use_remote, location, schema, verbosity):
def _worker_func(file_name, branches, tree_path, verbosity):
    """Module-level worker function for processing files, safe for threads and processes."""
    # Random stagger to avoid hammering I/O
    time.sleep(random.uniform(0.05, 0.20))  # 50-200 ms delay
    # Create Importer and pass arguments
    importer = Importer(
        file_name=file_name,
        branches=branches,
        tree_path=tree_path,
        verbosity=verbosity
    )

    return importer.import_branches()
        
class Processor:
    """Interface for processing files or datasets"""
    
    def __init__(self, tree_path="EventNtuple/ntuple", use_remote=False, location="tape", schema="root", verbosity=1, worker_verbosity=0):
        """Initialise the processor

        Args:
            tree_path (str, opt): Path to the Ntuple in file directory. Defaults to "EventNtuple/ntuple"
            use_remote (bool, opt): If not using local files. Defaults to False. 
            location (str, opt): Remote file location. Options are tape (default), disk, scratch, or nersc. 
            schema (str, opt): Remote file XRootD schema. Options are root (default), http, path, dcap, or samFile.
            verbosity (int, opt): Level of output detail (0: errors only, 1: info, warnings, 2: max). Defaults to 1.
            worker_verbosity (int, opt): Verbosity for work processes. Defaults to 0. Level of output detail (0: errors only, 1: info, warnings, 2: max)
        """
        self.tree_path = tree_path
        self.use_remote = use_remote
        self.location = location
        self.schema = schema
        self.verbosity = verbosity
        self.worker_verbosity = worker_verbosity

        self.logger = Logger( # Start logger
            print_prefix = "[pyprocess]", 
            verbosity = verbosity
        )
        
        if self.use_remote: #  Ensure mdh environment 
            _env_manager.ensure_environment()

        # tqdm progress bar format and styling
        self.bar_format = "{desc}: {percentage:3.0f}%|{bar:30}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"

        # Print out optional args 
        confirm_str = f"Initialised Processor:\n\tpath = '{self.tree_path}'\n\tuse_remote = {self.use_remote}"
        if use_remote:
            confirm_str += f"\n\tlocation = {self.location}\n\tschema = {self.schema}"
        confirm_str += f"\n\tverbosity={self.verbosity}"

        self.logger.log(confirm_str, "info")

    def get_file_list(self, file_name=None, defname=None, file_list_path=None, batch_size=50, stagger=(0.05, 0.20), timeout=30):
        """Utility to get a list of files from a single file, SAM definition, a text file, with remote prestaging.
    
        Args:
            file_name: File name 
            defname: SAM definition name 
            file_list_path: Path to a plain text file containing file paths
            batch_size: Number of files per mdh batch call
            stagger: Tuple of min/max sleep time between batches
            timeout: Timeout per mdh call (seconds)
            
        Returns:
            List of file paths (local or resolved remote URLs)
        """
    
        help_message = f"""
        Common causes of empty file list:
            - Authentication issues (try running getToken)
            - Invalid SAM definition (defname='{defname}')
            - The files are not staged
            - Incorrect file location (location='{self.location}')
            - use_remote is not True (remote='{self.use_remote}')
        """

        # Handle single files 
        # This is important, since process_functions are designed to call process_data with single files
        if file_name:
            file_list = [file_name]
            # If the file is accessible, return immediately 
            if self.use_remote and file_name.startswith(("root://", "http://", "dcap://")):
                return file_list
            elif os.path.exists(file_name):
                return file_list
            else:
                self.logger.log(f"File does not exist: {file_name}", "warning")
                return []    
        
        # Load file list from file
        elif file_list_path:
            if not os.path.exists(file_list_path):
                self.logger.log(f"File list path does not exist: {file_list_path}", "error")
                return []
            self.logger.log(f"Loading file list from {file_list_path}", "info")
            with open(file_list_path, "r") as f:
                file_list = [line.strip() for line in f if line.strip()]
    
        # Load file list from SAM definition
        elif defname:
            self.logger.log(f"Loading file list for SAM definition: {defname}", "max")
            try:
                commands = f"samweb list-files 'defname: {defname} with availability anylocation' | sort -V 2>/dev/null"
                output = subprocess.check_output(commands, shell=True, universal_newlines=True, stderr=subprocess.DEVNULL)
                file_list = [line for line in output.splitlines() if line]
            except Exception as e:
                self.logger.log(f"Exception while getting file list for {defname}: {e}", "error")
                return []
    
        else:
            self.logger.log("Error: Either 'file_name', 'defname' or 'file_list_path' must be provided", "error")
            return []
    
        if len(file_list) == 0:
            self.logger.log(f"File list has length zero{help_message}", "warning")
            return []
            
        ##### REMOTE FILE HANDLING ##### 
        # Prestage remote URLs in batches using threads
        if self.use_remote:
            resolved_files = []
            n_files = len(file_list)
        
            if len(file_list) == 0:
                self.logger.log(f"File list is empty{help_message}", "warning")
                return []
        
            # If there's only one file, resolve it directly without threads
            if file_name:
                self.logger.log("Single file detected, resolving directly", "info")
                try:
                    proc = subprocess.run(
                        ["mdh", "print-url", "-l", self.location, "-s", self.schema, "-"],
                        input=file_list[0],
                        capture_output=True,
                        text=True,
                        timeout=timeout
                    )
                    if proc.returncode != 0:
                        self.logger.log(f"Error prestaging single file: {proc.stderr}", "warning")
                        return []
                    file_list = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
                except subprocess.TimeoutExpired:
                    self.logger.log("Timeout resolving single file", "warning")
                    return []
                except Exception as e:
                    self.logger.log(f"Exception prestaging single file: {e}", "warning")
                    return []
                return file_list  # skip the rest
        
            # Otherwise, handle batches with threads
            prestage_threads = min(n_files // batch_size + 1, 2 * os.cpu_count(), 50)
        
            # Function to resolve one batch
            def resolve_batch(batch_idx, batch_files):
                # Optional stagger
                time.sleep(random.uniform(*stagger))
                try:
                    proc = subprocess.run(
                        ["mdh", "print-url", "-l", self.location, "-s", self.schema, "-"],
                        input="\n".join(batch_files),
                        capture_output=True,
                        text=True,
                        timeout=timeout
                    )
                    if proc.returncode != 0:
                        self.logger.log(f"Error resolving batch {batch_idx}: {proc.stderr}", "warning")
                        return []
                    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]
                except subprocess.TimeoutExpired:
                    self.logger.log(f"Timeout resolving batch {batch_idx}", "warning")
                    return []
                except Exception as e:
                    self.logger.log(f"Exception resolving batch {batch_idx}: {e}", "warning")
                    return []
        
            # Create batches
            batches = [(i // batch_size, file_list[i:i + batch_size])
                       for i in range(0, n_files, batch_size)]

            self.logger.log(f"Resolving remote file list in batches:\n\t{n_files} files, {len(batches)} batches, {prestage_threads} threads", "info")
            # Threaded execution with progress bar
            with ThreadPoolExecutor(max_workers=prestage_threads) as executor:
                futures = {executor.submit(resolve_batch, idx, batch): idx for idx, batch in batches}
        
                for future in tqdm.tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Resolving",
                    unit="batch",
                    ncols=150,
                    bar_format=self.bar_format,
                    colour="cyan"
                ):
                    urls = future.result()
                    resolved_files.extend(urls)
        
            file_list = resolved_files
        
        # Final logging
        if len(file_list) > 0:
            self.logger.log(f"Successfully loaded file list\n\tCount: {len(file_list)} files", "success")
        else:
            self.logger.log(f"File list has length {len(file_list)}{help_message}", "warning")
        
        return file_list
        
    def _process_files_parallel(self, file_list, worker_func, max_workers=None, use_processes=False):
        """Internal function to parallelise file operations with given a process function
        
        Args:
            file_list: List of files to process
            worker_func: Function to call for each file (must accept file name as first argument)
            max_workers: Maximum number of worker threads
            use_processes (bool, optional): Use process pool rather than thread pool 
        Returns:
            List of results from each processed file
        """
        
        if not file_list:
            self.logger.log("Error: Empty file list provided", "error")
            return None

        # If single file, skip multiprocessing
        if len(file_list) == 1:
            result = worker_func(file_list[0])
            return [result]
        
        if max_workers is None:
            # Return a sensible default for max threads
            max_workers = min(len(file_list), os.cpu_count()) # FIXME: sensible for threads, maybe not processes
            
        ExecutorClass = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        executor_type = "processes" if use_processes else "threads"
        
        self.logger.log(f"Starting processing on {len(file_list)} files with {max_workers} {executor_type}", "info")

        # Store results in a list
        results = []  

        # For tracking progress
        total_files = len(file_list)
        completed_files = 0 
        failed_files = 0

        with tqdm.tqdm(
            total=total_files, 
            desc="Processing",
            unit="file",
            bar_format=self.bar_format,
            colour="green",
            ncols=150 
        ) as pbar:
            
            # Start thread pool executor
            with ExecutorClass(max_workers=max_workers) as executor:
                # Create futures for each file processing task 
                futures = {
                    executor.submit(worker_func, file_name): file_name
                    for file_name in file_list
                }
                # Process results as they complete
                for future in as_completed(futures):
                    file_name = futures[future]
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                            completed_files += 1
                        else: 
                            failed_files += 1
                        
                        # Extract just the base filename for cleaner output
                        base_file_name = file_name.split('/')[-1]
                        
                    except Exception as e:
                        self.logger.log(f"Error processing {file_name}:\n{e}", "error")
                        # Increment failed files on exception
                        failed_files += 1
                        # Redraw progress bar
                        pbar.refresh()
                        # Propagete
                        raise e

                    finally:
                        # Always update the progress bar, regardless of success or failure
                        pbar.update(1)
                        # Update postfix with stats
                        pbar.set_postfix({
                            "successful": completed_files, 
                            "failed": failed_files 
                        })
                        # Safety cleanup
                        gc.collect()

                # More safety cleanup
                del futures
        
        # Return the results
        return results
            
    def process_data(self, file_name=None, file_list_path=None, defname=None, branches=None, max_workers=None, custom_worker_func=None, use_processes=False, prestage_remote=True):
        """Process the data 
        
        Args:
            file_name: File name 
            defname: SAM definition name
            file_list_path: Path to file list 
            branches: Flat list or grouped dict of branches to import
            max_workers: Maximum number of parallel workers
            custom_worker_func: Optional custom processing function for each file 
            use_processes: Whether to use processes rather than threads
            prestage_remote (bool, opt): Prestage file list with XRootD paths
            
        Returns:
            - If custom_worker_func is None: a concatenated awkward array with imported data from all files
            - If custom_worker_func is not None: a list of outputs from the custom process
        """

        # Check that we have one type of file argument 
        file_sources = sum(x is not None for x in [file_name, defname, file_list_path])
        if file_sources != 1: 
            self.logger.log(f"Please provide exactly one of 'file_name', 'file_list_path', or defname'", "error")
            return None

        # Validate custom_worker_func if provided
        if custom_worker_func is not None:
            # Check if it's callable
            if not callable(custom_worker_func):
                self.logger.log(f"custom_worker_func is not callable", "error")
                return None
                
            # Check function signature
            sig = inspect.signature(custom_worker_func)
            if len(sig.parameters) != 1:
                self.logger.log(f"custom_worker_func must take exactly one argument (file_name)", "error")
                return None

        # Set up process function
        if custom_worker_func is None: # Then use the default function
            worker_func = functools.partial(
                _worker_func,  # Module-level function
                branches=branches,
                tree_path=self.tree_path,
                verbosity=0 if file_name is None else self.worker_verbosity # multifile only
            )
        else: # Use the custom process function  
            worker_func = custom_worker_func

        # Prepare file list
        file_list = self.get_file_list(
            file_name=file_name,
            file_list_path=file_list_path, 
            defname=defname
        )

        # For single files, skip parallelisation
        if file_name: 
            result = worker_func(file_list[0]) # Run the process
            self.logger.log(f"Completed process on {file_name}", "success")
            return result 

        # Process and return list of results 
        results = self._process_files_parallel(
            file_list,
            worker_func,
            max_workers=max_workers,
            use_processes=use_processes
        )

        if len(results) == 0:
            self.logger.log(f"Results list has length zero", "warning")

        if custom_worker_func is None:
            # Concatenate the arrays
            results = ak.concatenate(results)
            if results is not None:
                self.logger.log(f"Returning concatenated array containing {len(results)} events", "success")
                self.logger.log(f"Array structure:", "max")
                if self.verbosity > 1:
                    results.type.show()
            else:
                self.logger.log(f"Concatenated array is None (failed to import branches)", "error")
                return None
        else: 
            self.logger.log(f"Returning {len(results)} results", "info")

        return results

# -----------------------------------------------------------------------
# Template for creating a custom processors with the Processor framework
# -----------------------------------------------------------------------

class Skeleton:
    """Template class for creating a custom analysis processor
    
    This template demonstrates how to create a class to run
    custom analysis jobs with the Processor framework 
    
    Using Skeleton:
        - Pass it as an argument to your custom Processor class
        - Customise the __init__ method with your configuration
        - Implement your processing logic in the process_file method
        - Add any additional helper methods you need
        - Override methods as needed

    Note: If using multiprocessing (not threading), the processor class must NOT be nested inside another object!
    """
    
    def __init__(self, verbosity=1):
        """Initialise your file processor with configuration parameters
        
        Customise this method to include parameters specific to your analysis.

        Args: 
            verbosity (int, opt): Level of output detail (0: errors only, 1: info, 2: debug, 3: max)
        
        """
        # File source configuration
        self.file_list_path = None  # Path to a text file with file paths
        self.defname = None         # SAM definition name
        self.file_name = None       # Single file to process
        
        # Data import configuration
        self.branches = []          # List of branches to import
        self.tree_path = "EventNtuple/ntuple"  # Path to tree name in file directory
        self.use_remote = False     # Whether to use remote file access
        self.location = "tape"      # File location (tape, disk, scratch, nersc)
        self.schema = "root"        # URL schema for remote files
        
        # Processing configuration
        self.max_workers = None     # Number of parallel workers (None=auto)
        self.use_processes = False  # Whether to use processes rather than threads 
        self.verbosity = verbosity
        self.worker_verbosity = 0   # Verbosity of worker function
        # Analysis-specific configuration
        # Add your own parameters here!
        # self.param1 = value1
        # self.param2 = value2

        # Start logger
        self.logger = Logger( 
            print_prefix = "[Skeleton]", 
            verbosity = verbosity
        )

        self.logger.log("Skeleton init", "info")
        
        
    def process_file(self, file_name):
        """Process a single file
        
        This is the core method that will be called for each file.
        Implement your file processing logic here.
        
        Args:
            file_name: Name of the file to process
            
        Returns:
            Any data structure representing the processed result
        """
        try:
            # Create a fresh Processor for this file
            local_processor = Processor(
                tree_path=self.tree_path,
                use_remote=self.use_remote,
                location=self.location,
                schema=self.schema,
                verbosity=self.worker_verbosity 
            )
            
            # Import the data
            data = local_processor.process_data(
                file_name=file_name,
                branches=self.branches
            )
            
            # Check if import was successful
            if data is None:
                self.logger.log(f"Failed to import data from {file_name}", "error")
                return None
                
            # Example processing - REPLACE WITH YOUR OWN LOGIC
            # This is just a placeholder that returns the file name and a count
            result = {
                "file_name": file_name,
                "event_count": len(data[self.branches[0]]) if self.branches else 0
            }

            return result
            
        except Exception as e:
            self.logger.log(f"Error processing {file_name}: {e}", "error")
            raise e # propagate exception
            
    def execute(self):
        """Run the processor on the configured files
        
        Returns:
            Combined results from all processed files
        """

        self.logger.log(f"Starting analysis", "info")
        
        # Input validation
        file_sources = sum(x is not None for x in [self.file_name, self.file_list_path, self.defname])
        if file_sources != 1:
            self.logger.log(f"Please provide exactly one of 'file_name', 'file_list_path', or defname'", "error")
            return None
            
        # Initialise the processor
        processor = Processor(
            tree_path=self.tree_path,
            use_remote=self.use_remote,
            location=self.location,
            schema=self.schema,
            verbosity=self.verbosity
        )
        
        # Process the data
        results = processor.process_data(
            file_name=self.file_name, # If it's just one file it will skip the process function
            file_list_path=self.file_list_path,
            defname=self.defname,
            branches=self.branches,
            max_workers=self.max_workers,
            custom_worker_func=self.process_file,
            use_processes=self.use_processes 
        )

        # Postprocess
        results = self.postprocess(results)

        self.logger.log(f"Analysis complete", "success")
            
        return results

    def postprocess(self, results): 
        """Run post processing on the results list 
        Placeholder method intended to be overridden
        """
        return results
        