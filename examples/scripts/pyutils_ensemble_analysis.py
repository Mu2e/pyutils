"""
Ensemble analysis with pyutils: MDS4b
=====================================

Introduction

This notebook runs a complete conversion electron (CE) style analysis on the MDS4b
mock data ensemble, using the `pyutils` modules together:

- `pyprocess` to read one file, and then many files in parallel with the `Skeleton`
  class
- `pyselect` and `pyvector` to build track-level selections
- `pycut` to manage the cuts and make cut flows, including combining cut flows from
  many files
- `pyplot` to plot the results

Because the ensemble is simulation, every track can be traced back to the process that
made it, so we can see how each background responds to each cut.

The MDS4b ensemble ntuple

| | |
|---|---|
| SAM definition | `nts.mu2e.ensembleNtuple.MDC2025-001_MDS4b.root` |
| Files | 48, about 6.4 GB and 180k events each |
| Location | persistent dCache (`location="disk"`), so no prestaging is needed |
| Parent | `mcs.mu2e.ensembleOnSpillReco.MDC2025ax_MDS4b_best_v1_1.art` |

The ensemble mixes the main backgrounds in proportion: cosmic rays (CRY), muon decay
in orbit (DIO), internal and external radiative muon capture (RMC), and internal and
external radiative pion capture (RPC).

Note: Each reconstructed track is stored once for every fit hypothesis (e-, e+, mu-,
mu+), so most events with a track have four entries in `trk`. Always choose a
hypothesis with `trk.pdg` before counting tracks.

Table of contents
1. Setup
1. Reading one file
1. The true origin of each track
1. Defining the selection
1. Cut flows
1. Momentum spectrum by origin
1. Momentum resolution
1. Scaling up to many files
1. Summary

Generated from examples/notebooks/pyutils_ensemble_analysis.ipynb by scripts/nb_to_script.py.
Edit the notebook and regenerate, rather than editing this script.
"""

# ======================================================================================
# 1. Setup
# ======================================================================================
#
# Use the `mu2e_env` kernel, or `pyenv ana` on the command line.

import awkward as ak
import numpy as np
import pandas as pd
import hist
import matplotlib.pyplot as plt
from IPython.display import display

from pyutils.pyprocess import Processor, Skeleton
from pyutils.pyselect import Select
from pyutils.pyvector import Vector
from pyutils.pycut import CutManager
from pyutils.pyplot import Plot
from pyutils.pylogger import Logger

# Print wide tables in full when run as a script
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

logger = Logger(print_prefix="[ensemble]", verbosity=1)

# The MDS4b ensemble ntuple
defname = "nts.mu2e.ensembleNtuple.MDC2025-001_MDS4b.root"

# One file from the dataset, on persistent dCache
file_name = "/pnfs/mu2e/persistent/datasets/phy-nts/nts/mu2e/ensembleNtuple/MDC2025-001_MDS4b/root/e0/23/nts.mu2e.ensembleNtuple.MDC2025-001_MDS4b.001430_00002163.root"

# ======================================================================================
# 2. Reading one file
# ======================================================================================
#
# We group the branches so that each group can be handled on its own, for example
# `data["trkfit"]` holds the fit at each surface. A complete list of branches is given
# in the [EventNtuple
# documentation](https://github.com/Mu2e/EventNtuple/blob/main/doc/branches.md).
#
# | Group | Branches | Use |
# |---|---|---|
# | `evt` | `event`, `run`, `subrun` | event ID |
# | `trk` | `trk.pdg`, `trk.nactive`, `trkqual.result`, `trkpid.result` | track summary, quality and PID |
# | `trkfit` | `trksegs`, `trksegpars_lh` | momentum, time and helix parameters at each surface |
# | `trkmc` | `trkmcsim`, `trksegsmc` | truth: MC particles of each track, true momentum at each surface |
# | `crv` | `crvcoincs.time`, `crvcoincs.PEs` | CRV coincidences, for the cosmic veto |
#
# `Processor` reads whole files, which is about 17 seconds and a few GB of memory for
# these branches.

branches = {
    "evt": ["event", "run", "subrun"],
    "trk": ["trk.pdg", "trk.nactive", "trkqual.result", "trkpid.result"],
    "trkfit": ["trksegs", "trksegpars_lh"],
    "trkmc": ["trkmcsim", "trksegsmc"],
    "crv": ["crvcoincs.time", "crvcoincs.PEs"],
}

processor = Processor(verbosity=1)
data = processor.process_data(file_name=file_name, branches=branches)

print(f"Events: {len(data)}")
print(f"Tracks per event (all fit hypotheses): {ak.mean(ak.num(data['trk']['trk.pdg'])):.2f}")

# ======================================================================================
# 3. The true origin of each track
# ======================================================================================
#
# `trkmcsim` lists, for each track, the MC particle that made most of its hits (`rank ==
# 0`) followed by its ancestors. The particle that made the hits is often a secondary.
# For example, an RMC track is usually an electron from a photon conversion, so its own
# `startCode` is a conversion and not RMC. To find where a track really came from, we
# look along the chain for the generator: the entry whose `startCode` is a Mu2e stopped
# particle process, or whose `gen` is a cosmic generator.
#
# The codes come from [`ProcessCode.hh`](https://github.com/Mu2e/Offline/blob/main/MCDat
# aProducts/inc/ProcessCode.hh) and
# [`GenId.hh`](https://github.com/Mu2e/Offline/blob/main/MCDataProducts/inc/GenId.hh).

# Stopped particle processes (ProcessCode)
STOPPED_ORIGINS = {
    166: "DIO",
    167: "CE- (endpoint)",
    168: "CE- (leading log)",
    171: "Internal RMC",
    172: "External RMC",
    178: "External RPC",
    179: "Internal RPC",
}
COSMIC = 0           # Label used here for cosmics
COSMIC_GEN = [38, 44]  # cosmicCRY, cosmicCORSIKA (GenId)
UNKNOWN = -1

ORIGIN_NAMES = {**STOPPED_ORIGINS, COSMIC: "Cosmic", UNKNOWN: "Other"}

def get_origin(trkmcsim):
    # Label each track by the generator of its MC particle chain
    is_stopped = ak.zeros_like(trkmcsim.startCode, dtype=bool)
    for code in STOPPED_ORIGINS:
        is_stopped = is_stopped | (trkmcsim.startCode == code)
    is_cosmic = ak.zeros_like(trkmcsim.gen, dtype=bool)
    for gen in COSMIC_GEN:
        is_cosmic = is_cosmic | (trkmcsim.gen == gen)

    origin = ak.firsts(trkmcsim.startCode[is_stopped], axis=-1)
    origin = ak.where(ak.any(is_cosmic, axis=-1), COSMIC, origin)
    return ak.fill_none(origin, UNKNOWN)

origin = get_origin(data["trkmc"]["trkmcsim"])

# Count tracks with the e- fit hypothesis by origin
is_e_fit = data["trk"]["trk.pdg"] == 11
counts = pd.Series(ak.to_list(ak.flatten(origin[is_e_fit]))).map(ORIGIN_NAMES).value_counts()
display(counts.rename("e- tracks").to_frame())

# For comparison, here is what you get from the `startCode` of the `rank == 0` particle
# alone. Many tracks start from ordinary Geant4 processes, such as Compton scattering
# (12), photon conversion (13) or, for cosmics, muon ionisation (31), and their origin
# would be missed:

rank0_code = ak.firsts(data["trkmc"]["trkmcsim"]["startCode"], axis=-1)
display(pd.Series(ak.to_list(ak.flatten(rank0_code[is_e_fit]))).value_counts().head(8).rename("e- tracks").to_frame())

# ======================================================================================
# 4. Defining the selection
# ======================================================================================
#
# The cuts below are track-level masks, with shape (events, tracks). We compute each fit
# quantity at the tracker entrance (`TT_Front`) once, and then build every cut from
# these. The values follow the usual CE selection; they are a starting point and should
# be tuned for your own analysis.
#
# | Group | Cut | Value |
# |---|---|---|
# | Preselect | Fit hypothesis | e- (`trk.pdg == 11`) |
# | Preselect | Tracker entrance | has a fit at `TT_Front` |
# | Preselect | Direction | downstream ($p_z > 0$) at `TT_Front` |
# | Reco | Track quality | `trkqual.result > 0.8` |
# | Reco | Track PID | `trkpid.result > 0.95`, where -999 means no score |
# | Reco | Active hits | `trk.nactive >= 20` |
# | Geom | Impact parameter | $-80 < d_0 < 105$ mm |
# | Geom | Maximum radius | $450 < r_{max} < 680$ mm |
# | Time | $t_0$ | $640 < t_0 < 1650$ ns |
# | Veto | CRV | no coincidence with $-50 < t_0 - t_{CRV} < 150$ ns |
# | Signal | Momentum window | $103.6 < p < 104.9$ MeV/c |
#
# The function returns the `CutManager`, and the momentum and origin of each track, so
# that we can reuse it for every file in section 8.

def get_selection(data, verbosity=0):
    # Return a CutManager with the CE selection, and the momentum and origin of each track
    selector = Select(verbosity=verbosity)
    vector = Vector(verbosity=verbosity)
    cut_manager = CutManager(verbosity=verbosity)

    trk, trkfit = data["trk"], data["trkfit"]

    # Fit quantities at the tracker entrance, one value per track (None if no entrance segment)
    at_front = selector.select_surface(trkfit, surface_name="TT_Front")
    segs = trkfit["trksegs"][at_front]
    pars = trkfit["trksegpars_lh"][at_front]
    mom = ak.firsts(vector.get_mag(segs, "mom"), axis=-1)
    t0 = ak.firsts(segs["time"], axis=-1)
    pz = ak.firsts(segs["mom"]["fCoordinates"]["fZ"], axis=-1)
    d0 = ak.firsts(pars["d0"], axis=-1)
    maxr = ak.firsts(pars["maxr"], axis=-1)

    # Time difference to every CRV coincidence, shape (events, tracks, coincidences)
    dt_crv = t0[:, :, None] - data["crv"]["crvcoincs.time"][:, None, :]
    crv_coinc = ak.any((dt_crv > -50) & (dt_crv < 150), axis=-1)

    def add(name, description, mask, group):
        # Tracks with no entrance segment have None values, and fail the cut
        cut_manager.add_cut(name, description, ak.fill_none(mask, False), group=group)

    add("is_e_fit", "e- fit hypothesis", trk["trk.pdg"] == 11, "Preselect")
    add("has_trk_front", "Fit at tracker entrance", ak.any(at_front, axis=-1), "Preselect")
    add("is_downstream", r"Downstream ($p_z > 0$)", pz > 0, "Preselect")
    add("good_trkqual", "trkqual > 0.8", selector.select_trkqual(trk, 0.8), "Reco")
    add("good_trkpid", "trkpid > 0.95", selector.select_trkpid(trk, 0.95), "Reco")
    add("has_hits", "Active hits >= 20", selector.has_n_hits(trk, 20), "Reco")
    add("within_d0", "-80 < d0 < 105 mm", (d0 > -80) & (d0 < 105), "Geom")
    add("within_maxr", "450 < maxr < 680 mm", (maxr > 450) & (maxr < 680), "Geom")
    add("within_t0", "640 < t0 < 1650 ns", (t0 > 640) & (t0 < 1650), "Time")
    add("no_crv", "No CRV coincidence (-50 < dt < 150 ns)", ~crv_coinc, "Veto")
    add("within_sig_win", "103.6 < p < 104.9 MeV/c", (mom > 103.6) & (mom < 104.9), "Signal")

    origin = get_origin(data["trkmc"]["trkmcsim"])
    return cut_manager, ak.fill_none(mom, np.nan), origin

cut_manager, mom, origin = get_selection(data)

# ======================================================================================
# 5. Cut flows
# ======================================================================================
#

# ======================================================================================
# 5.1 Nominal cut flow
# ======================================================================================
#
# `create_cut_flow` applies the active cuts in order, and counts the events with at
# least one track passing all of them so far.

cut_flow = cut_manager.create_cut_flow(data)
display(cut_manager.format_cut_flow(cut_flow))

# ======================================================================================
# 5.2 Cut flow by origin
# ======================================================================================
#
# Masks are ordinary awkward arrays, so we can combine them with the origin to see which
# cuts remove which backgrounds. Here we count tracks rather than events. The time cut
# removes the RPC, which arrives promptly after the proton pulse, while the CRV veto and
# the PID cut act on cosmics.

def cut_flow_by_origin(cut_manager, origin):
    # Count the tracks of each origin passing each successive cut
    rows = {}
    cumulative = None
    for name, cut in cut_manager.get_active_cuts().items():
        mask = cut["mask"]
        cumulative = mask if cumulative is None else cumulative & mask
        rows[name] = pd.Series(ak.to_list(ak.flatten(origin[cumulative]))).map(ORIGIN_NAMES).value_counts()
    return pd.DataFrame(rows).T.fillna(0).astype(int)

display(cut_flow_by_origin(cut_manager, origin))

# ======================================================================================
# 6. Momentum spectrum by origin
# ======================================================================================
#
# To look at the momentum spectrum we turn off the momentum window and take the
# remaining selection. `save_state` lets us come back to the nominal configuration
# afterwards.

cut_manager.save_state("nominal")
cut_manager.toggle_cut({"within_sig_win": False})

selected = cut_manager.combine_cuts(active_only=True)
mom_sel = ak.flatten(mom[selected])
origin_sel = ak.flatten(origin[selected])

mom_by_origin = {
    ORIGIN_NAMES[code]: ak.to_numpy(mom_sel[origin_sel == code])
    for code in ORIGIN_NAMES
    if ak.sum(origin_sel == code) > 0
}
for name, values in mom_by_origin.items():
    logger.log(f"{name}: {len(values)} tracks", "info")

plotter = Plot(verbosity=0)

fig, ax = plt.subplots()
plotter.plot_1D_overlay(
    mom_by_origin,
    nbins=60,
    xmin=90,
    xmax=110,
    log_y=True,
    xlabel="Momentum at tracker entrance [MeV/c]",
    ylabel="Tracks per 0.33 MeV/c",
    ax=ax,
    show=False,
)
ax.axvspan(103.6, 104.9, color="gray", alpha=0.2, label="Signal window")
ax.legend()
plt.savefig("mom_by_origin.png")
plt.show()

# And the yields in the signal window, with the nominal configuration restored:

cut_manager.restore_state("nominal")
in_window = cut_manager.combine_cuts(active_only=True)

window_yields = pd.Series(ak.to_list(ak.flatten(origin[in_window]))).map(ORIGIN_NAMES).value_counts()
display(window_yields.rename("Tracks in signal window").to_frame())

# Note: These are raw counts from one file. To turn them into expected yields you need
# the livetime or POT of the ensemble, which is recorded in the ensemble configuration
# (`cnf.mu2e.MDS4b-info.MDC2025ax.txt`) rather than in the ntuple.

# ======================================================================================
# 7. Momentum resolution
# ======================================================================================
#
# `trksegsmc` holds the true momentum at each surface, with the same `sid` labels as
# `trksegs`, so the resolution is the difference between the fit and the truth at the
# tracker entrance (`sid == 0`).

vector = Vector(verbosity=0)
selector = Select(verbosity=0)

trksegsmc = data["trkmc"]["trksegsmc"]
at_front_mc = trksegsmc["sid"] == selector.get_surface_id("TT_Front")
mom_true = ak.firsts(vector.get_mag(trksegsmc[at_front_mc], "mom"), axis=-1)

cut_manager.toggle_cut({"within_sig_win": False})
selected = cut_manager.combine_cuts(active_only=True)
cut_manager.restore_state("nominal")

dp = ak.to_numpy(ak.flatten(ak.drop_none((mom - mom_true)[selected])))

plotter.plot_1D(
    dp,
    nbins=100,
    xmin=-2,
    xmax=1,
    log_y=True,
    xlabel=r"$p_{reco} - p_{true}$ at tracker entrance [MeV/c]",
    ylabel="Tracks",
    stat_box=True,
    out_path="mom_resolution.png",
)

# ======================================================================================
# 8. Scaling up to many files
# ======================================================================================
#
# The whole dataset is about 300 GB, which is too much to hold in memory. Instead we use
# a `Skeleton` subclass: `process_file` runs on each file in parallel and returns only
# small summaries (a histogram for each origin and a cut flow), and `postprocess`
# combines them.
#
# We get the file names from the SAM definition and write the first few to a file list.
# With `use_remote=True`, `pyutils` turns each name into an xroot URL with `mdh`, so
# this also works on EAF. Increase `n_files` to run on more of the dataset.
#
# Note: Run `getToken` first. Be mindful of the number of workers on a shared machine:
# each worker holds one file's branches in memory.
#
# A file that cannot be opened (for example, if a dCache pool is unavailable) is logged
# as failed and skipped, so `postprocess` compares the number of results with the file
# list and warns if any are missing. Check this before using the combined results.

n_files = 4

file_processor = Processor(use_remote=True, location="disk", verbosity=0)
all_files = file_processor.get_file_list(defname=defname)
file_list = all_files[:n_files]

file_list_path = "MDS4b_file_list.txt"
with open(file_list_path, "w") as f:
    f.write("\n".join(file_list))

logger.log(f"Wrote {len(file_list)} of {len(all_files)} files to {file_list_path}", "info")

class EnsembleProcessor(Skeleton):
    # Fill momentum histograms by origin and a cut flow for each file

    def __init__(self, file_list_path, verbosity=1):
        super().__init__(verbosity=verbosity)
        self.file_list_path = file_list_path
        self.branches = branches
        self.use_remote = True
        self.location = "disk"
        self.max_workers = 2  # Each worker holds one file in memory
        self.logger.print_prefix = "[EnsembleProcessor]"

    def process_file(self, file_name):
        processor = Processor(use_remote=self.use_remote, location=self.location, verbosity=0)
        data = processor.process_data(file_name=file_name, branches=self.branches)
        if data is None:
            return None

        cut_manager, mom, origin = get_selection(data)
        cut_flow = cut_manager.create_cut_flow(data)

        # Momentum spectrum without the momentum window, by origin
        cut_manager.toggle_cut({"within_sig_win": False})
        selected = cut_manager.combine_cuts(active_only=True)
        h = hist.Hist(
            hist.axis.IntCategory(list(ORIGIN_NAMES), name="origin"),
            hist.axis.Regular(60, 90, 110, name="mom", label="Momentum at tracker entrance [MeV/c]"),
        )
        h.fill(
            origin=ak.to_numpy(ak.flatten(origin[selected])),
            mom=ak.to_numpy(ak.flatten(mom[selected])),
        )

        return {"file_name": file_name, "cut_flow": cut_flow, "hist": h}

    def postprocess(self, results):
        # Files that failed to open are not in results, so compare with the file list
        with open(self.file_list_path) as f:
            n_requested = len([line for line in f if line.strip()])
        if len(results) < n_requested:
            self.logger.log(f"{n_requested - len(results)} of {n_requested} files failed, see the errors above", "warning")

        self.logger.log(f"Combining {len(results)} files", "info")
        cut_manager = CutManager(verbosity=0)
        return {
            "n_files": len(results),
            "cut_flow": cut_manager.combine_cut_flows([r["cut_flow"] for r in results]),
            "hist": sum(r["hist"] for r in results),
        }

ensemble_processor = EnsembleProcessor(file_list_path)
results = ensemble_processor.execute()

# The combined cut flow over all the processed files:

display(results["cut_flow"])

# And the combined momentum spectrum. `hist` objects support slicing by category, so
# each origin is one line:

h = results["hist"]

fig, ax = plt.subplots()
for code, name in ORIGIN_NAMES.items():
    h_origin = h[hist.loc(code), :]
    if h_origin.sum() > 0:
        h_origin.plot1d(ax=ax, label=name, yerr=False)
ax.axvspan(103.6, 104.9, color="gray", alpha=0.2, label="Signal window")
ax.set_yscale("log")
ax.set_ylabel("Tracks per 0.33 MeV/c")
ax.set_title(f"MDS4b, {results['n_files']} files")
ax.legend()
plt.savefig("mom_by_origin_combined.png")
plt.show()

# ======================================================================================
# 9. Summary
# ======================================================================================
#
# This notebook showed how to:
#
# 1. Read grouped branches from an MDS4b ensemble file with `Processor`
# 2. Find the true origin of each track by following the `trkmcsim` chain to its
#    generator
# 3. Build a CE-style selection as track-level masks with `Select`, `Vector` and
#    `CutManager`
# 4. Make cut flows, overall and by origin
# 5. Plot the momentum spectrum by origin and count the yields in the signal window
# 6. Measure the momentum resolution with `trksegsmc`
# 7. Run the same selection over many files with a `Skeleton` subclass, and combine
#    histograms and cut flows
#
# For more on each module, see the other notebooks in this directory, and `help()` on
# any class or function.
