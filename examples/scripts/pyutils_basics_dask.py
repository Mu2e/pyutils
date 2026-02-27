"""
pyutils basics with Dask
=========================

This script demonstrates the core pyutils functionality using DaskProcessor
to handle multiple files in parallel. It shows:

1. Importing data from multiple files using DaskProcessor
2. Applying selection cuts
3. Inspecting data
4. Performing vector operations
5. Creating plots from aggregated data

Key differences from standard pyutils_basics:
- Uses DaskProcessor instead of Processor for parallel file handling
- Works with multiple files instead of a single file
- Scales across cores/clusters using Dask
"""

import awkward as ak
import tempfile
import os

# ============================================================================
# 1. Setting up your environment
# ============================================================================

print("=" * 70)
print("pyutils Basics with Dask")
print("=" * 70)

# Import the pyutils modules
from pyutils.pydask import DaskProcessor
from pyutils.pyselect import Select
from pyutils.pyprint import Print
from pyutils.pyvector import Vector
from pyutils.pyplot import Plot
from pyutils.pylogger import Logger

# Initialize logger
logger = Logger(print_prefix="[pyutils_dask]", verbosity=2)

# ============================================================================
# 2. Processing Data with DaskProcessor
# ============================================================================

logger.log("Step 1: Initializing DaskProcessor", "info")

# Initialize DaskProcessor with appropriate settings
processor = DaskProcessor(
    tree_path="EventNtuple/ntuple",
    use_remote=True,           # Access files from remote location
    location="disk",           # Read from disk (persistent dataset)
    verbosity=1,
    worker_verbosity=0
)

logger.log("DaskProcessor initialized", "success")

# Define the branches we want
branches = ["trksegs"]

logger.log("Branches to extract: " + str(branches), "info")

# For demonstration, show the API for multi-file processing
logger.log("\nMulti-File Processing Example:", "info")
logger.log("""
To process multiple files, use a file list with DaskProcessor:

1. Prepare a file list (one file path per line):
   The repository includes MDS3a.txt with persistent dataset files

2. Process with DaskProcessor:

   processor = DaskProcessor(
       tree_path="EventNtuple/ntuple",
       use_remote=True,       # Access remote persistent datasets
       location="disk"        # Read from disk storage
   )
   
   data = processor.process_data(
       file_list_path="MDS3a.txt",
       branches=branches,
       n_workers=4,           # Use 4 parallel workers
       show_progress=True     # Show progress bar
   )

Advantages over Processor:
- Scales across multiple cores automatically
- Easy to add more files without code changes
- Built-in progress monitoring
- Can connect to remote Dask clusters
""", "info")

# Create a sample file list for demonstration
logger.log("\nCreating sample file list for demonstration...", "info")

# Use the MDS3a.txt file list provided in the repository
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
file_list_path = os.path.join(script_dir, "../../MDS3a.txt")

# Verify file exists
if os.path.exists(file_list_path):
    with open(file_list_path, 'r') as f:
        sample_files = [line.strip() for line in f if line.strip()]
    
    logger.log(f"Loaded file list from: {file_list_path}", "success")
    logger.log(f"Total files in list: {len(sample_files)}", "info")
    logger.log(f"First file: {sample_files[0][-80:]}", "info")
else:
    logger.log(f"File list not found at: {file_list_path}", "warning")
    logger.log("Using empty file list for demo", "info")
    sample_files = []
    file_list_path = None

# ============================================================================
# 3. Processing Single File (Demo)
# ============================================================================

logger.log("\n" + "=" * 70, "info")
logger.log("Step 2: Processing Multiple Files with DaskProcessor", "info")
logger.log("=" * 70, "info")

logger.log(f"Processing {len(sample_files)} files from MDS3a.txt with DaskProcessor...", "info")

try:
    # Use DaskProcessor to load and process multiple files in parallel
    if sample_files and file_list_path:
        logger.log("Using DaskProcessor with multi-file processing", "info")
        
        data = processor.process_data(
            file_list_path=file_list_path,
            branches=branches,
            n_workers=4,              # Use 4 parallel workers
            show_progress=True        # Show progress bar
        )
    
    else:
        logger.log("File list not available, cannot process multiple files", "warning")
        logger.log("Skipping analysis", "info")
        raise Exception("MDS3a.txt file list required for this example")
    
    logger.log("Data aggregation complete", "success")
    logger.log(f"Total events from all files: {len(data)}", "info")
    
    # ========================================================================
    # 4. Applying Selection Cuts
    # ========================================================================
    
    logger.log("\n" + "=" * 70, "info")
    logger.log("Step 3: Applying Selection Cuts", "info")
    logger.log("=" * 70, "info")
    
    selector = Select(verbosity=1)
    
    logger.log("Selecting track segments at tracker entrance (TT_Front)...", "info")
    
    # Create a mask to select track segments at the tracker entrance
    at_trkent = selector.select_surface(
        data=data,
        surface_name="TT_Front"  # Tracker entrance
    )
    
    # Add the mask to the data array
    data["at_trkent"] = at_trkent
    
    # Apply the mask
    trkent = data[at_trkent]
    
    logger.log(f"Selected {len(trkent)} events at tracker entrance", "success")
    
    # ========================================================================
    # 5. Inspecting Your Data
    # ========================================================================
    
    logger.log("\n" + "=" * 70, "info")
    logger.log("Step 4: Inspecting Data", "info")
    logger.log("=" * 70, "info")
    
    printer = Print(verbose=False)
    
    logger.log("Data structure at tracker entrance:", "info")
    printer.print_n_events(trkent, n_events=1)
    
    # ========================================================================
    # 6. Performing Vector Operations
    # ========================================================================
    
    logger.log("\n" + "=" * 70, "info")
    logger.log("Step 5: Performing Vector Operations", "info")
    logger.log("=" * 70, "info")
    
    vector = Vector(verbosity=1)
    
    logger.log("Computing momentum magnitude...", "info")
    
    mom_mag = vector.get_mag(
        branch=trkent["trksegs"],
        vector_name="mom"
    )
    
    logger.log("Momentum magnitude computed", "success")
    
    # ========================================================================
    # 7. Creating Plots
    # ========================================================================
    
    logger.log("\n" + "=" * 70, "info")
    logger.log("Step 6: Creating Plots", "info")
    logger.log("=" * 70, "info")
    
    plotter = Plot()
    
    # Flatten arrays for plotting
    time_flat = ak.flatten(trkent["trksegs"]["time"], axis=None)
    mom_mag_flat = ak.flatten(mom_mag, axis=None)
    
    logger.log(f"Time values to plot: {len(time_flat)}", "info")
    logger.log(f"Momentum values to plot: {len(mom_mag_flat)}", "info")
    
    # 1D Histogram: Time distribution
    logger.log("Creating 1D histogram of time distribution...", "info")
    
    plotter.plot_1D(
        time_flat,
        nbins=100,
        xmin=450,
        xmax=1695,
        title="Time at Tracker Entrance (Dask Example)",
        xlabel="Fit time at Trk Ent [ns]",
        ylabel="Events per bin",
        out_path='h1_time_dask.png',
        stat_box=True,
        error_bars=True
    )
    
    logger.log("1D histogram created: h1_time_dask.png", "success")
    
    # 2D Histogram: Momentum vs. Time
    logger.log("Creating 2D histogram of momentum vs. time...", "info")
    
    plotter.plot_2D(
        x=mom_mag_flat,
        y=time_flat,
        nbins_x=100,
        xmin=85,
        xmax=115,
        nbins_y=100,
        ymin=450,
        ymax=1650,
        title="Momentum vs. Time at Tracker Entrance (Dask Example)",
        xlabel="Fit mom at Trk Ent [MeV/c]",
        ylabel="Fit time at Trk Ent [ns]",
        out_path='h2_timevmom_dask.png'
    )
    
    logger.log("2D histogram created: h2_timevmom_dask.png", "success")
    
    # ========================================================================
    # Summary
    # ========================================================================
    
    logger.log("\n" + "=" * 70, "info")
    logger.log("SUMMARY", "info")
    logger.log("=" * 70, "info")
    
    summary = f"""
Multi-File Processing with DaskProcessor:
    - Total files processed: {len(sample_files)}
    - Total events aggregated: {len(data)}
    - Events at tracker entrance: {len(trkent)}
    - Momentum range: {ak.min(mom_mag_flat):.2f} - {ak.max(mom_mag_flat):.2f} MeV/c
    - Time range: {ak.min(time_flat):.0f} - {ak.max(time_flat):.0f} ns

Output plots created:
    - h1_time_dask.png (1D time distribution)
    - h2_timevmom_dask.png (2D momentum vs time)

Key advantages of DaskProcessor:
✓ Process multiple files in parallel
✓ Automatic load balancing across cores
✓ Progress tracking with show_progress=True
✓ Easy scaling to remote clusters
✓ Same output format as standard Processor
✓ Built-in error resilience with retries
"""
    logger.log(summary, "info")
    
except FileNotFoundError:
    logger.log("Sample data file not found. The script demonstrates the API.", "warning")
    logger.log("\nTo run with real data:", "info")
    logger.log("1. Edit the sample_files list with actual file paths", "info")
    logger.log("2. Set appropriate file_list_path for multiple files", "info")
    logger.log("3. Run: processor.process_data(file_list_path=..., branches=...)", "info")
except Exception as e:
    logger.log(f"Error during processing: {e}", "error")
    logger.log("Check that Mu2e environment is properly initialized", "warning")

logger.log("\nScript completed!", "success")
