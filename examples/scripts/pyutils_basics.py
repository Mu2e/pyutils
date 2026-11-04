"""
Getting started: pyutils basics
===============================

Introduction

`pyutils` is a suite of tools designed to streamline Mu2e Python-based analysis tasks,
enabling you to:

- Import and process data
- Handle selection cuts
- Inspect and visualise data
- Perform vector operations
- Create publication-quality plots

Whether you're new to analysis at Mu2e or looking to improve your workflow, this
tutorial will walk you through the basic features of `pyutils` through a demonstration
of a simple analysis using `EventNtuple` data.

Table of Contents
1. Setting up your environment
1. Importing data
1. Applying selection cuts
1. Inspecting your data
1. Performing vector operations
1. Plotting
1. Complete example

Generated from examples/notebooks/pyutils_basics.ipynb by scripts/nb_to_script.py.
Edit the notebook and regenerate, rather than editing this script.
"""

# ======================================================================================
# 1. Setting up your environment
# ======================================================================================
#
# First, we need to import the necessary external packages. Use the `mu2e_env`
# interactive kernel to make sure you have the packages you need, if not using a custom
# environment (see [tutorial](https://github.com/Mu2e/Tutorial/blob/main/EAF/Docs/06-The
# Mu2eEnvironment.md)).

# Import external packages
import awkward as ak

# ======================================================================================
# 2. Processing data
# ======================================================================================
#
# The `pyprocess` module handles this, using `pyimport` and `pyread` as dependencies.
# The `Processor` class is your main tool for processing data.
#
# Note that help information for all `pyutils` classes and functions can be accessed by
# running `help()`, e.g. `help(Processor)`.

# Import the Procssor class
from pyutils.pyprocess import Processor 

# Initialise the Importer with increased verbosity 
# verbosity=0 will only show errors
processor = Processor(verbosity=2)

# Define the path to our example file
# This is one file from the MDS4b ensemble, on persistent dCache
file_name = "/pnfs/mu2e/persistent/datasets/phy-nts/nts/mu2e/ensembleNtuple/MDC2025-001_MDS4b/root/e0/23/nts.mu2e.ensembleNtuple.MDC2025-001_MDS4b.001430_00002163.root"

# Define the branches we want to access
# For a complete list of available branches, see:
# https://github.com/Mu2e/EventNtuple/blob/main/doc/branches.md
# Also refer to ntuplehelper, available after mu2e setup EventNtuple
branches = ["trk.pdg", "trksegs"]

# Import the branches from the file
# This loads the data into memory and returns an awkward array
data = processor.process_data(
    file_name=file_name,
    branches=branches
)

# When running with high verbosity, you'll see detailed output about the structure of
# the imported data.
#

# ======================================================================================
# 2.1 Choosing a fit hypothesis
# ======================================================================================
#
# Each track is fit with several particle hypotheses (e-, e+, mu-, mu+), and every fit
# is stored as its own entry in the track branches. Most events with a track therefore
# have four entries, so choose a hypothesis before doing anything else. Here we keep the
# e- fits, using `trk.pdg`, and then keep only `trksegs` for the rest of the notebook.

print(f"Tracks per event, all hypotheses: {ak.mean(ak.num(data['trk.pdg'])):.2f}")

# Track-level mask with shape (events, tracks)
is_e_fit = data["trk.pdg"] == 11

# Apply the mask, then keep only the trksegs field
data = data[is_e_fit][["trksegs"]]

print(f"Tracks per event, e- fits:      {ak.mean(ak.num(data['trksegs'])):.2f}")

# ======================================================================================
# 3. Applying selection cuts
# ======================================================================================
# Once data is loaded, you can filter you data with selection cuts. The `Select` class
# from `pyselect` provides tools for this purpose.

# Import the Select class
from pyutils.pyselect import Select 

# Initialise the selector with verbose output
selector = Select(verbosity=2)

# Create a mask to select track segments at the tracker entrance
# sid = 0 corresponds to the tracker entrance plane
at_trkent = selector.select_surface(
    data=data, 
    surface_name="TT_Front"  # Select the tracker entrance
)

# Optional: add the mask to the data array (useful for verification)
data["at_trkent"] = at_trkent

# Apply the mask to create a new array with only track segments at the tracker entrance
trkent = data[at_trkent]

# In this example, we're selecting track segments at the entrance to the tracker (sid =
# 0).
#
# Note: `Select` does not provide utilities for applying the mask to the data. This is a
# deliberate choice to enable the user to combine cuts as they like and to use different
# masking methods.
#
# Note: Surface names follow the official Offline
# [`SurfaceId`](https://github.com/Mu2e/Offline/blob/main/DataProducts/inc/SurfaceId.hh)
# names, such as `"TT_Front"`, `"ST_Foils"` or `"OPA"`. `selector.surface_id_map` lists
# all of them with their IDs.

# ======================================================================================
# 4. Inspecting Your Data
# ======================================================================================
#
# After selecting your data, you can verify that the selection cuts are behaving as
# expected. The `Print` class from `pyprint` allows you to examine individual events
# before and after applying cuts.

# Import the Print class
from pyutils.pyprint import Print

# Initialise the printer
# verbose=False is default and prevents overwhelming output with large arrays
# verbose=True prevents array truncation, for detailed debugging
printer = Print(verbose=False)

# Compare data before and after cuts
print("Before selection cuts:")
printer.print_n_events(data, n_events=1)

print("\nAfter selection cuts:")
printer.print_n_events(trkent, n_events=1)

# This will show you:
# - The data structure before applying the selection
# - The data structure after applying the selection
# - Confirmation that all selected track segments have sid = 0
# - Confirmation that the at_trkent flag is True for all track segments

# ======================================================================================
# 5. Performing Vector Operations
# ======================================================================================
#
# The `Vector` class from `pyvector` provides tools for common element wise vector
# operations on your array.

# Import the Vector class
from pyutils.pyvector import Vector

# Initialise the vector processor
vector = Vector(verbosity=2)

# Calculate the momentum magnitude from the momentum vector components
# This creates a new array with the magnitudes
mom_mag = vector.get_mag(
    branch=trkent["trksegs"],
    vector_name="mom" # either "mom" or "pos" in EventNtuple
)

# The `get_mag()` function calculates the magnitude of a 3D vector, using the `vector`
# package.

# ======================================================================================
# 6. Plotting
# ======================================================================================
#
# The `Plot` class from `pyplot` provides methods for creating publication-quality plots
# from flattened arrays.
#
# Note: `pyplot` will be extended to allow plotting histogram objects directly.
#

# ======================================================================================
# 6.1 Creating a 1D Histogram
# ======================================================================================
#
# First, let's create a 1D histogram of the time distribution for track segments at the
# tracker entrance (the $t_{0}$):

# Import the Plot class
from pyutils.pyplot import Plot 

# Initialise the plotter
plotter = Plot()

# Flatten the array 
# ak.flatten with axis=None removes all nesting, so that the result is always a 1D array 
time_flat = ak.flatten(trkent["trksegs"]["time"], axis=None)

# Create a 1D histogram of track times
plotter.plot_1D(
    time_flat,               # Data to plot
    nbins=100,               # Number of bins
    xmin=450,                # Minimum x-axis value
    xmax=1695,               # Maximum x-axis value
    title="Time at Tracker Entrance",
    xlabel="Fit time at Trk Ent [ns]",
    ylabel="Events per bin",
    out_path='h1_time.png',  # Output file path
    stat_box=True,           # Show statistics box
    error_bars=True          # Show error bars
)

# Again, to explore the full functionality of `pyutils` functions you can always run
# `help()`, as shown below:

help(plotter.plot_1D)

# ======================================================================================
# 6.2 Creating a 2D Histogram
# ======================================================================================
#
# Now, let's create a 2D histogram to explore the relationship between momentum and
# time:

# Flatten the momentum magnitudes for plotting
mom_mag_flat = ak.flatten(mom_mag, axis=None)

# Create a 2D histogram of momentum vs. time
plotter.plot_2D(
    x=mom_mag_flat,           # x-axis data
    y=time_flat,              # y-axis data
    nbins_x=100,              # Number of x bins
    xmin=85,                  # Minimum x value
    xmax=115,                 # Maximum x value
    nbins_y=100,              # Number of y bins
    ymin=450,                 # Minimum y value
    ymax=1650,                # Maximum y value
    title="Momentum vs. Time at Tracker Entrance",
    xlabel="Fit mom at Trk Ent [MeV/c]",
    ylabel="Fit time at Trk Ent [ns]",
    out_path='h2_timevmom.png'
)

# This 2D plot shows the relationship between track momentum and time at the entrance to
# the tracker.

# ======================================================================================
# 7. Complete example
# ======================================================================================
#
# To put everything together, here is a complete example that demonstrates importing
# data, applying selections, performing vector operations, and creating visualisations.

import awkward as ak

# Import data
from pyutils.pyprocess import Processor
processor = Processor(verbosity=1)
data = processor.process_data(
    file_name="/pnfs/mu2e/persistent/datasets/phy-nts/nts/mu2e/ensembleNtuple/MDC2025-001_MDS4b/root/e0/23/nts.mu2e.ensembleNtuple.MDC2025-001_MDS4b.001430_00002163.root",
    branches=["trk.pdg", "trksegs"]
)
data = data[data["trk.pdg"] == 11][["trksegs"]]

# Apply selection
from pyutils.pyselect import Select
selector = Select(verbosity=1)
at_trkent = selector.select_surface(data=data, surface_name="TT_Front")
trkent = data[at_trkent]

# Calculate momentum magnitude
from pyutils.pyvector import Vector
vector = Vector(verbosity=1)
mom_mag = vector.get_mag(branch=trkent["trksegs"], vector_name="mom")

# Create visualizations
from pyutils.pyplot import Plot
plotter = Plot()

# Flatten arrays for plotting
time_flat = ak.flatten(trkent["trksegs"]["time"], axis=None)
mom_mag_flat = ak.flatten(mom_mag, axis=None)

# 1D Histogram of time
plotter.plot_1D(
    time_flat,
    nbins=100,
    xmin=450,
    xmax=1695,
    title="Time Distribution at Tracker Entrance",
    xlabel="Time [ns]",
    ylabel="Events",
    out_path='time_distribution.png'
)

# 2D Histogram of momentum vs time
plotter.plot_2D(
    x=mom_mag_flat,
    y=time_flat,
    nbins_x=100,
    xmin=85,
    xmax=115,
    nbins_y=100,
    ymin=450,
    ymax=1650,
    title="Momentum vs. Time at Tracker Entrance",
    xlabel="Momentum [MeV/c]",
    ylabel="Time [ns]",
    out_path='momentum_vs_time.png'
)
