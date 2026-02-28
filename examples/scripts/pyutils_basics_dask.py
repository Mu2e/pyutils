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
logger.log("\nDaskProcessor with SAM Definition:", "info")
logger.log("""
DaskProcessor can retrieve files from SAM definitions for parallel processing:

- SAM definition: "nts.mu2e.ensembleMDS3aMix1BBTriggered.MDC2025-001.root"
- Queries files dynamically from the data catalog
- DaskProcessor creates a local cluster to process files in parallel

Advantages:
- Works with SAM definitions for dynamic file queries
- Can also use file_list_path for static file lists
- Scales across multiple cores automatically
- Built-in progress monitoring
- Can connect to remote Dask clusters on EAF
""", "info")

# ============================================================================
# 3. Processing Multiple Files with SAM Definition
# ============================================================================

logger.log("\n" + "=" * 70, "info")
logger.log("Step 2: Retrieving File List with get_file_list()", "info")
logger.log("=" * 70, "info")

# Use get_file_list() to retrieve files from SAM definition
logger.log("Retrieving file list from SAM definition...", "info")

file_list = processor.get_file_list(
    defname="nts.mu2e.ensembleMDS3aMix1BBTriggered.MDC2025-001.root"
)

if file_list and len(file_list) > 0:
    logger.log(f"Retrieved {len(file_list)} files from SAM definition", "success")
    logger.log(f"First file: {file_list[0][-80:]}", "info")
    
    # Save to temporary file for use with process_data
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
        for f in file_list:
            tmp.write(f + '\n')
        temp_file_list = tmp.name
    
    logger.log(f"Saved file list to temporary file: {temp_file_list}", "info")
    
    # ====================================================================
    # 4. Processing Multiple Files with DaskProcessor
    # ====================================================================
    
    logger.log("\n" + "=" * 70, "info")
    logger.log("Step 3: Processing Multiple Files with DaskProcessor", "info")
    logger.log("=" * 70, "info")
    
    logger.log("Starting parallel processing with DaskProcessor...", "info")
    
    try:
        data = processor.process_data(
            file_list_path=temp_file_list,
            branches=branches,
            n_workers=4,              # Use 4 parallel workers
            show_progress=True        # Show progress bar
        )
        
        logger.log("Data aggregation complete", "success")
        logger.log(f"Total events from all files: {len(data)}", "info")
        
        # Clean up temp file
        import os
        os.unlink(temp_file_list)
        logger.log("Cleaned up temporary file", "info")
        
        # ====================================================================
        # 5. Applying Selection Cuts
        # ====================================================================
        
        logger.log("\n" + "=" * 70, "info")
        logger.log("Step 4: Applying Selection Cuts", "info")
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
        
        # ====================================================================
        # 6. Inspecting Your Data
        # ====================================================================
        
        logger.log("\n" + "=" * 70, "info")
        logger.log("Step 5: Inspecting Data", "info")
        logger.log("=" * 70, "info")
        
        printer = Print(verbose=False)
        
        logger.log("Data structure at tracker entrance:", "info")
        printer.print_n_events(trkent, n_events=1)
        
        # ====================================================================
        # 7. Performing Vector Operations
        # ====================================================================
        
        logger.log("\n" + "=" * 70, "info")
        logger.log("Step 6: Performing Vector Operations", "info")
        logger.log("=" * 70, "info")
        
        vector = Vector(verbosity=1)
        
        logger.log("Computing momentum magnitude...", "info")
        
        mom_mag = vector.get_mag(
            branch=trkent["trksegs"],
            vector_name="mom"
        )
        
        logger.log("Momentum magnitude computed", "success")
        
        # ====================================================================
        # 8. Creating Plots
        # ====================================================================
        
        logger.log("\n" + "=" * 70, "info")
        logger.log("Step 7: Creating Plots", "info")
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
        
        # ====================================================================
        # Summary
        # ====================================================================
        
        logger.log("\n" + "=" * 70, "info")
        logger.log("SUMMARY", "info")
        logger.log("=" * 70, "info")
        
        summary = f"""
Multi-File Processing with DaskProcessor:
    - Total files processed: {len(file_list)}
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
        logger.log("SAM definition files not found.", "error")
        raise
    except Exception as e:
        logger.log(f"Error during processing: {e}", "error")
        raise

logger.log("\nScript completed!", "success")
