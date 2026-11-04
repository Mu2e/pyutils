"""
MC Truth Interface
==================

`pyutils.pymcutil` contains accessors related to the MC truth information. It can be
used in the same way as any other `pyutils` class.

This notebook uses one file from the MDS4b ensemble, which mixes cosmics, DIO, RMC and
RPC, and covers:

1. Reading the MC truth branches
1. A quick overview with `count_particle_types`
1. Selecting tracks by MC truth with the `MC` masks
1. Following the MC particle chain of a track to its generator
1. Event-level truth with the `primary` branch
1. True momentum at the tracker entrance

The MC truth branches are described in the [EventNtuple
documentation](https://github.com/Mu2e/EventNtuple/blob/main/doc/branches.md).

| Branch | Shape | Content |
|---|---|---|
| `trkmc` | (events, tracks) | summary of the MC particle that made each track |
| `trkmcsim` | (events, tracks, particles) | the MC particle that made most hits (`rank == 0`), then its ancestors |
| `trksegsmc` | (events, tracks, surfaces) | true momentum, position and time at each surface |
| `primary` | (events, primaries) | the generated primary particles of each event, whether reconstructed or not |

Generated from examples/notebooks/pyutils_mctruth.ipynb by scripts/nb_to_script.py.
Edit the notebook and regenerate, rather than editing this script.
"""

# ======================================================================================
# 1. Reading the MC truth branches
# ======================================================================================
#
# First import the relevant python packages, followed by other `pyutils` classes that
# will aid the analysis:

# Import external packages
import awkward as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# Import the pyutils classes
from pyutils.pyprocess import Processor
from pyutils.pymcutil import MC
from pyutils.pyselect import Select
from pyutils.pyvector import Vector
from pyutils.pyplot import Plot

# Print wide tables in full when run as a script
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", None)

# Initialise the Processor with increased verbosity
# verbosity=0 will only show errors
processor = Processor(verbosity=1)

# One file from the MDS4b ensemble, on persistent dCache
file_name = "/pnfs/mu2e/persistent/datasets/phy-nts/nts/mu2e/ensembleNtuple/MDC2025-001_MDS4b/root/e0/23/nts.mu2e.ensembleNtuple.MDC2025-001_MDS4b.001430_00002163.root"

# Define the branches we want to access
# Also refer to ntuplehelper, available after mu2e setup EventNtuple
branches = ["trk.pdg", "trkmc", "trkmcsim", "trksegsmc", "primary"]

# Import the branches from the file
# This loads the data into memory and returns an awkward array
data = processor.process_data(
    file_name=file_name,
    branches=branches
)

print(f"Events: {len(data)}")
print("trkmcsim fields:", data["trkmcsim"].fields)

# ======================================================================================
# 2. Overview with `count_particle_types`
# ======================================================================================
#
# `count_particle_types` gives a quick summary, labelling each event by the `startCode`
# of the `rank == 0` MC particle of its first track.
#
# Note: Pass only events with at least one track. Events with no tracks currently get a
# `None` label, and are counted in every category.

mc_example = MC(verbosity=1)
has_track = ak.num(data["trk.pdg"]) > 0
particle_types = mc_example.count_particle_types(data[has_track])

# Note: The `rank == 0` particle is the one that made most of the track's hits, which is
# often a secondary. Most RMC and RPC tracks, for example, are electrons from a photon
# conversion, so they appear as "others" here. Section 4 shows how to find the generator
# of every track.

# ======================================================================================
# 3. Selecting by MC truth
# ======================================================================================
#
# `MC` also provides masks with the same shape as `trkmcsim`, (events, tracks,
# particles). Combine them with `is_track_particle`, which selects the MC particle that
# contributed most to each track, to select tracks by their true origin.
#
# Each track is stored once for every fit hypothesis, so we also keep only the e- fits.
# `trk.pdg` has shape (events, tracks), so we add an axis with `[:, :, None]` to
# broadcast it over the particles.

is_primary = mc_example.is_track_particle(data)
is_e_fit = (data["trk.pdg"] == 11)[:, :, None]

def count_tracks(mask):
    # Number of e- fit tracks whose rank 0 MC particle passes the mask
    return int(ak.sum(ak.any(mask & is_primary & is_e_fit, axis=-1)))

selections = {
    "CE- (endpoint)": mc_example.is_CeMinusEndpoint(data),
    "CE- (leading log)": mc_example.is_CeMinusLeadingLog(data),
    "CE+ (leading log)": mc_example.is_CePlusLeadingLog(data),
    "DIO in target": mc_example.is_target_DIO(data),
    "Cosmic": mc_example.is_cosmic(data),
    "True electron": mc_example.is_electron(data),
    "True mu-": mc_example.is_muon(data),
}

print(f"Total e- fit tracks: {ak.sum(is_e_fit)}")
for name, mask in selections.items():
    print(f"  {name}: {count_tracks(mask)}")

# There is no CE signal in this ensemble file, so the CE masks are empty. Most cosmic
# tracks fit with the e- hypothesis are really muons.
#

# ======================================================================================
# 4. Following the MC particle chain
# ======================================================================================
#
# Here is the chain of one track whose `rank == 0` particle was made by an ordinary
# Geant4 process, such as Compton scattering (`startCode` 12) or a photon conversion
# (13). The entries after `rank == 0` are its ancestors, back to the proton on the
# production target, and the one with a Mu2e stopped particle process code shows that
# the track came from RMC:

sim = data["trkmcsim"]

# Find an e- fit track whose rank 0 particle is not from a generator, but whose ancestor is RMC
from_rmc = ak.any((sim.startCode == 171) | (sim.startCode == 172), axis=-1)
rank0_code = ak.firsts(sim.startCode, axis=-1)
candidates = ak.flatten(ak.fill_none(from_rmc & (rank0_code < 100) & (data["trk.pdg"] == 11), False))
example = np.flatnonzero(ak.to_numpy(candidates))[0]

chain = ak.flatten(sim, axis=1)[example]
display(pd.DataFrame({
    "rank": chain.rank.to_list(),
    "pdg": chain.pdg.to_list(),
    "startCode": chain.startCode.to_list(),
    "gen": chain.gen.to_list(),
    "p [MeV/c]": np.round(Vector(verbosity=0).get_mag(chain, "mom").to_numpy(), 2),
}))

# We can label every track at once by looking along the chain for the generator: a
# stopped particle process code from [`ProcessCode.hh`](https://github.com/Mu2e/Offline/
# blob/main/MCDataProducts/inc/ProcessCode.hh), or a cosmic generator from
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
COSMIC = 0             # Label used here for cosmics
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

origin = get_origin(data["trkmcsim"])

e_fit = data["trk.pdg"] == 11
by_chain = pd.Series(ak.to_list(ak.flatten(origin[e_fit]))).map(ORIGIN_NAMES).value_counts()
# For the rank 0 particle, a primary cosmic has startCode 56 (mu2ePrimary)
by_rank0 = pd.Series(ak.to_list(ak.flatten(rank0_code[e_fit]))).map({**ORIGIN_NAMES, 56: "Cosmic"}).fillna("Other").value_counts()
display(pd.DataFrame({"rank 0 particle": by_rank0, "generator in chain": by_chain}).fillna(0).astype(int))

# ======================================================================================
# 5. Event-level truth with `primary`
# ======================================================================================
#
# The track branches only know about particles that were reconstructed. The `primary`
# branch lists the generated primaries of every event, so it tells us what was
# simulated, whether or not a track was found. The fields are named `primary.<field>`.

primary = data["primary"]
print("primary fields:", primary.fields[:12], "...")

# Event composition by primary process
primary_code = ak.firsts(primary["primary.startCode"], axis=-1)
primary_gen = ak.firsts(primary["primary.gen"], axis=-1)
primary_origin = ak.where(np.isin(ak.to_numpy(ak.fill_none(primary_gen, -1)), COSMIC_GEN), COSMIC, primary_code)
primary_origin = ak.fill_none(primary_origin, UNKNOWN)

# Fraction of events of each type with at least one e- fit track
has_e_track = ak.any(e_fit, axis=-1)
summary = pd.DataFrame({
    "origin": pd.Series(ak.to_list(primary_origin)).map(ORIGIN_NAMES).fillna("Other"),
    "has_e_track": ak.to_list(has_e_track),
})
display(summary.groupby("origin")["has_e_track"].agg(events="size", with_e_track="sum").assign(
    fraction=lambda df: (df.with_e_track / df.events).round(3)
))

# ======================================================================================
# 6. True momentum at the tracker entrance
# ======================================================================================
#
# `trksegsmc` has the true momentum at each surface, with the same surface IDs as
# `trksegs`. Here is the true momentum of e- fit tracks at the tracker entrance, for
# each origin:

selector = Select(verbosity=0)
vector = Vector(verbosity=0)
plotter = Plot(verbosity=0)

trksegsmc = data["trksegsmc"]
at_front = trksegsmc["sid"] == selector.get_surface_id("TT_Front")
mom_true = ak.firsts(vector.get_mag(trksegsmc[at_front], "mom"), axis=-1)

mom_by_origin = {}
for code, name in ORIGIN_NAMES.items():
    values = ak.to_numpy(ak.flatten(ak.drop_none(mom_true[e_fit & (origin == code)])))
    if len(values) > 0:
        mom_by_origin[name] = values

plotter.plot_1D_overlay(
    mom_by_origin,
    nbins=100,
    xmin=0,
    xmax=200,
    log_y=True,
    xlabel="True momentum at tracker entrance [MeV/c]",
    ylabel="e- fit tracks",
    out_path="mom_true_by_origin.png",
)

# ======================================================================================
# Summary
# ======================================================================================
#
# This notebook showed how to:
#
# 1. Read the MC truth branches `trkmc`, `trkmcsim`, `trksegsmc` and `primary`
# 2. Get a quick overview with `count_particle_types`
# 3. Select tracks by their rank 0 MC particle with the `MC` masks
# 4. Follow the `trkmcsim` chain to find the generator of every track
# 5. Use `primary` for event-level truth, including events with no track
# 6. Plot the true momentum at the tracker entrance by origin
#
# For a full analysis using these labels, see
# [pyutils_ensemble_analysis.ipynb](pyutils_ensemble_analysis.ipynb).
