"""
pycut demo
==========

This notebook demonstrates `pyutils.pycut` module functionality for analysis cuts,
including:

- Cut definition and management
- Cut flow generation and display
- Cut configuration comparison
- State management

Note: Combining cut flows from many files processed in parallel is shown in
[pyutils_ensemble_analysis.ipynb](pyutils_ensemble_analysis.ipynb), together with a
more complete selection.

Generated from examples/notebooks/pycut_demo.ipynb by scripts/nb_to_script.py.
Edit the notebook and regenerate, rather than editing this script.
"""

# ======================================================================================
# 1. Setup and Imports
# ======================================================================================

import awkward as ak
import numpy as np
import pandas as pd
from IPython.display import display
from pyutils.pyprocess import Processor 
from pyutils.pylogger import Logger
from pyutils.pyvector import Vector
from pyutils.pyselect import Select
from pyutils.pycut import CutManager
from pyutils.pyplot import Plot

# Print wide tables in full when run as a script
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

# Initialise utilities
verbosity = 2

logger = Logger(
    verbosity=verbosity,
    print_prefix="[pycut_demo]"
)

selector = Select(verbosity=1)
vector = Vector(verbosity=1)

# ======================================================================================
# 2. Data processing
# ======================================================================================
#
# Load data using the pyprocess module with relevant branches for analysis. We use one
# file from the MDS4b ensemble, which mixes cosmics, DIO, RMC and RPC.

processor = Processor(verbosity=1)

# One file from the MDS4b ensemble
file_name = "/pnfs/mu2e/persistent/datasets/phy-nts/nts/mu2e/ensembleNtuple/MDC2025-001_MDS4b/root/e0/23/nts.mu2e.ensembleNtuple.MDC2025-001_MDS4b.001430_00002163.root"

# Grouped branches
branches = {
    "evt" : ["event"],
    "trk" : ["trk.nactive", "trk.pdg", "trkqual.result", "trkpid.result"],
    "trkfit" : ["trksegs"]
}

# Process the file
data = processor.process_data(
    file_name=file_name,
    branches=branches
)

# ======================================================================================
# 3. Cut manager initialisation
# ======================================================================================

cut_manager = CutManager(verbosity=verbosity)

# ======================================================================================
# 4. Cut definitions
# ======================================================================================
#
# Define realistic Mu2e analysis cuts organised into logical groups:
# - Preselect: Basic track reconstruction requirements
# - Reco: Track quality and hit requirements
# - Meas: Physics measurement cuts (timing and momentum)
#
# Note: Within `EventNtuple` all cuts must be defined on the same level of nesting, such
# as track-level.

# ======================================================================================
# 4.1 Preselection cuts
# ======================================================================================

logger.log("Defining preselection cuts...", "info")

# Tracker entrance segments
at_trk_front = selector.select_surface(data["trkfit"], surface_name="TT_Front") 
has_trk_front = ak.any(at_trk_front, axis=-1)

cut_manager.add_cut(
    name="has_trk_front",
    description="Segments at tracker entrance",
    mask=has_trk_front,
    active=True,
    group="Preselect"
)

# Electron track fits
# Each track is stored once per fit hypothesis (e-, e+, mu-, mu+), so this keeps the e- fits
is_reco_electron = selector.is_electron(data["trk"])
cut_manager.add_cut(
    name="is_reco_electron",
    description="Electron track fits",
    mask=is_reco_electron,
    active=True,
    group="Preselect"
)

# Downstream track requirement
is_downstream = selector.is_downstream(data["trkfit"]) 
is_downstream_track = ak.all(~at_trk_front | is_downstream, axis=-1) 
cut_manager.add_cut(
    name="is_downstream",
    description=r"Downstream tracks ($p_{z} > 0$)",
    mask=is_downstream_track,
    active=True,
    group="Preselect"
)

# ======================================================================================
# 4.2 Reconstruction cuts
# ======================================================================================

logger.log("Defining reconstruction cuts...", "info")

# Track quality requirement
good_trkqual = selector.select_trkqual(data["trk"], 0.8)
cut_manager.add_cut(
    name="good_trkqual",
    description="Track quality > 0.8",
    mask=good_trkqual,
    active=True,
    group="Reco"
)

# Track PID requirement (-999 means no score, and fails)
good_trkpid = selector.select_trkpid(data["trk"], 0.95)
cut_manager.add_cut(
    name="good_trkpid",
    description="Track PID > 0.95",
    mask=good_trkpid,
    active=True,
    group="Reco"
)

# Sufficient hits requirement
has_hits = selector.has_n_hits(data["trk"], 20)
cut_manager.add_cut(
    name="has_hits",
    description="Active hits >= 20", 
    mask=has_hits,
    active=True,
    group="Reco"
)

# ======================================================================================
# 4.3 Measurement cuts
# ======================================================================================

logger.log("Defining measurement cuts...", "info")

# Timing window (t0)
within_t0 = ((640 < data["trkfit"]["trksegs"]["time"]) & 
            (data["trkfit"]["trksegs"]["time"] < 1650))
within_t0_track = ak.all(~at_trk_front | within_t0, axis=-1) 
cut_manager.add_cut(
    name="within_t0",
    description="t0 window (640 < t0 < 1650 ns)",
    mask=within_t0_track, 
    active=True,
    group="Meas"
)

# Momentum signal window
mom = vector.get_mag(data["trkfit"]["trksegs"], "mom") 
within_sig_win = ((mom > 103.6) & (mom < 104.9))
within_sig_win_track = ak.all(~at_trk_front | within_sig_win, axis=-1)
cut_manager.add_cut(
    name="within_sig_win",
    description="Signal window (103.6 < p < 104.9 MeV/c)", 
    mask=within_sig_win_track,
    active=True,
    group="Meas"
)

# ======================================================================================
# 5. Nominal cut flow
# ======================================================================================
#
# Generate and display the nominal (full analysis) cut flow.

# Generate and display cut flow
nominal_cut_flow = cut_manager.create_cut_flow(data)
nominal_df = cut_manager.format_cut_flow(nominal_cut_flow)

logger.log("Nominal cut flow:", "info")
display(nominal_df)

# Save nominal state
cut_manager.save_state("nominal_cuts")

# ======================================================================================
# 6. Loose cut flow
# ======================================================================================
#
# Create a loose cuts configuration by removing the measurement requirements while
# keeping reconstruction quality cuts.

logger.log("Creating loose cuts configuration...", "info")

# Remove timing and momentum windows for loose selection
cut_manager.toggle_cut({
    "within_t0": False,      # Remove t0 window
    "within_sig_win": False  # Remove momentum window
})

# Generate loose cut flow
loose_cut_flow = cut_manager.create_cut_flow(data)
loose_df = cut_manager.format_cut_flow(loose_cut_flow)

logger.log("Loose cuts flow:", "info")
display(loose_df)

# Save loose cuts state
cut_manager.save_state("loose_cuts")

# ======================================================================================
# 7. Preselection-only cut flow
# ======================================================================================
#
# Create a very loose configuration with only basic preselection cuts active.

logger.log("Creating preselection-only configuration...", "info")

# Restore to nominal cuts first
cut_manager.restore_state("nominal_cuts")

# Deactivate reconstruction and measurement cuts
cut_manager.toggle_group({
    "Reco": False,     # Deactivate reconstruction cuts
    "Meas": False      # Deactivate measurement cuts
})

# Generate preselection-only cut flow
presel_cut_flow = cut_manager.create_cut_flow(data)
presel_df = cut_manager.format_cut_flow(presel_cut_flow)

logger.log("Preselection-only cut flow:", "info")
display(presel_df)

# Save preselection cuts state
cut_manager.save_state("preselection_cuts")

# ======================================================================================
# 8. Cut flow comparison
# ======================================================================================
#
# Apply different cut configurations and create momentum distributions for comparison.

# Reinitialise selector and vector with low verbosity for cleaner output
vector = Vector(verbosity=0)
selector = Select(verbosity=0)

logger.log("Applying cuts and extracting momentum distributions...", "info")

# Define cut configurations to compare
cut_configs = {
   "Preselection": "preselection_cuts",
   "Loose cuts": "loose_cuts",
   "Nominal cuts": "nominal_cuts"
}

mom_cut = {}

for config_name, state_name in cut_configs.items():
    # Restore cut state
    cut_manager.restore_state(state_name)
    
    # Get the combined cut mask
    trk_mask = cut_manager.combine_cuts(active_only=True)
    
    # Apply mask to tracks and track segments 
    filtered_data = ak.copy(data)
    filtered_data["trk"] = filtered_data["trk"][trk_mask]
    filtered_data["trkfit"] = filtered_data["trkfit"][trk_mask]
    
    # Clean up events with no tracks after cuts 
    evt_mask = ak.any(trk_mask, axis=-1)
    filtered_data = filtered_data[evt_mask]
    
    # Get momentum for selected events
    mom_mag = vector.get_mag(filtered_data["trkfit"]["trksegs"], "mom")
    at_trk_front = selector.select_surface(filtered_data["trkfit"], surface_name="TT_Front")
    mom_at_trk_front = mom_mag[at_trk_front]
    flat_mom_at_trk_front = ak.flatten(mom_at_trk_front)

    # Store momentum distribution
    mom_cut[config_name] = flat_mom_at_trk_front
       
    logger.log(f"{config_name}: {len(flat_mom_at_trk_front)} track segments", "info")

# ======================================================================================
# 9. Visualisation
# ======================================================================================

plotter = Plot(verbosity=1)

plotter.plot_1D_overlay(
    mom_cut,
    nbins=100, 
    xmin=0,
    xmax=500,
    log_y=True,
    xlabel="Momentum [MeV/c]",
    ylabel="Events",
    out_path="momentum_cut_comparison.png"
)

# ======================================================================================
# 10. Write cut flow to file
# ======================================================================================
#
# Cut flows are formatted as `pandas` DataFrames, making them easy to save for
# documentation or further analysis.

logger.log("Saving cut flow to file...", "info")

out_path = "nominal_cut_flow.csv"
nominal_df.to_csv(out_path, index=False)
logger.log(f"Wrote {out_path}", "success")

# ======================================================================================
# 11. State management summary
# ======================================================================================
#
# Demonstrate the state management capabilities of the cut manager.

logger.log("State management summary:", "info")

# List all saved states
saved_states = cut_manager.list_saved_states()
logger.log(f"Available saved states: {saved_states}", "info")

# Demonstrate restoration to original state
cut_manager.restore_original_state()
original_cut_flow = cut_manager.create_cut_flow(data)
original_df = cut_manager.format_cut_flow(original_cut_flow)

logger.log("Restored to original configuration:", "info")
display(original_df)

# ======================================================================================
# Summary
# ======================================================================================
#
# This notebook demonstrated the key features of the `pycut` module:
#
# 1. Cut definition: Organised cuts into logical groups (Preselect, Reco, Meas)
# 2. Cut flow analysis: Generated detailed cut flow tables showing event retention
# 3. Configuration management: Compared nominal, loose, and preselection-only
#    configurations
# 4. State management: Saved and restored different cut configurations
# 5. Event selection: Applied cuts to data for physics analysis
# 6. Visualisation: Created momentum distribution comparisons
# 7. Documentation: Exported cut flows to CSV for records
#
# The `pycut` module provides a flexible framework for managing complex analysis cut
# flows.
