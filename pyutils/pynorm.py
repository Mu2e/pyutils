#! /usr/bin/env
import DbService
import argparse
import ROOT
import math
import random
import os
import numpy as np


#-------------------------------------------------------------------------------------#  
# Constants related to muon and pion interactions.
# References for these values can be found in the referenced documentation.

# --- Muon Interactions ---
# Fraction of stopped muons that undergo nuclear capture
CAPTURES_PER_STOPPED_MUON = 0.609
# Rate of radiative muon capture (RMC) events resulting in a gamma ray > 57 MeV, per capture event
RMC_GT_57_PER_CAPTURE  = 1.43e-5 # Source: Phys. Rev. C 59, 2853 (1999)
# Fraction of stopped muons that undergo standard DIO (Decay In Orbit)
DIO_PER_STOPPED_MUON = 0.391 # Calculated as: 1 - CAPTURES_PER_STOPPED_MUON
# Rate of Incoming Particle Decay After Stopping (IPA)
IPA_DECAYS_PER_STOPPED_MUON  = 0.92990

# --- Pion Interactions ---
# Fraction of stopped pions that result in a Radiative Pion Capture (RPC)
RPC_PER_STOPPED_PION = 0.0215 # Source: Reference uploaded on DocDB-469
# --- Internal Conversion Ratios ---
# Ratio of internal conversion events per RMC event (assuming RPC value is applicable)
INTERNAL_PER_RMC = 0.00690
# Ratio of internal conversion events per RPC event
INTERNAL_RPC_PER_RPC = 0.00690 # Source: Reference uploaded on DocDB-717


# --- Configuration Placeholders (Mutable Variables) ---
# These values get overwritten later in the script or during runtime.

# Expected rates per Proton on Target (POT)
target_stopped_muons_per_pot = 1.0
target_stopped_pions_per_pot = 1.0
ipa_stopped_mu_per_POT = 1.0
ipa_stopping_rate = 1.0

# Event counters (initialized to zero)
num_pion_stops = 0.0
num_pion_resamples = 0.0
num_pion_filters = 0.0
selected_sum_of_weights = 0.0

# Rate and operational parameters
rate = 1.0
dutyfactor = 1.0
total_pot = 0.

#-------------------------------------------------------------------------------------#

# --- Database Interaction ---

# Establish a connection and retrieve simulation efficiencies from the database.

# Initialize the database tool
db_tool = DbService.DbTool()
db_tool.init()

# Define arguments for the database query
query_arguments = [
    "print-run",
    "--purpose", "Sim_best",
    "--version", "v1_1",
    "--run", "1430",
    "--table", "SimEfficiencies2",
    "--content"
]

# Execute the database query
db_tool.setArgs(query_arguments)
db_tool.run()

# Store the raw result for further processing
rr = db_tool.getResult()

# Fill varaibles associated with muon stops in target
lines= rr.split("\n")
for line in lines:
    words = line.split(",")
    if words[0] == "MuminusStopsCat" or words[0] == "MuBeamCat" :
        #print(f"Including {words[0]} with rate {words[3]}")
        rate = rate * float(words[3])
        target_stopped_muons_per_pot = rate * 1000 


# Fill variables associated with pion stops in target
lines= rr.split("\n")
for line in lines:
    words = line.split(",")
    if words[0] == "PiBeamCat" or words[0] == "PiTargetStops":
        target_stopped_pions_per_pot *= float(words[3]) # 0.001880093 * 0.5165587875
    if words[0] == "PiTargetStops":
        num_pion_stops = words[1]    #41324703
    if words[0] == "PiMinusFilter" :
        num_pion_filters = words[1] # 6634478
    if words[0] == "PhysicalPionStops" :
        num_pion_resamples= words[2]  # 10000000000
    if words[0] == "PiSelectedLifeimeWeight_sampler" :
        selected_sum_of_weights = words[3] #2393.604874

# Fill variables associated with IPA stopped muons
lines= rr.split("\n")
for line in lines:
    words = line.split(",")
    if words[0] == "IPAStopsCat" or words[0] == "MuBeamCat" :
        ipa_stopping_rate = ipa_stopping_rate * float(words[3])
        ipa_stopped_mu_per_POT = ipa_stopping_rate
print("IPAStopMuonRate=", ipa_stopped_mu_per_POT)
    
#-------------------------------------------------------------------------------------#    
def get_duty_factor(run_mode='1BB'):
    """
    Returns the estimated duty factor based on the operational run mode (beam structure).

    Args:
        run_mode (str): The operational mode, either '1BB' (1 beam batch)
                        or '2BB' (2 beam batches). Defaults to '1BB'.

    Returns:
        float: The corresponding duty factor for the specified mode.
    """
    if run_mode == '1BB':
        # Duty factor for a single beam batch operation
        duty_factor = 0.323
    elif run_mode == '2BB':
        # Duty factor for two beam batch operation
        duty_factor = 0.246
    else:
        # Handle unrecognized modes or provide a default fallback if necessary
        # print(f"Warning: Unknown run mode '{run_mode}'. Using default duty factor.")
        duty_factor = 0.323

    return duty_factor

def get_pot(on_spill_time, run_mode='1BB', printout=False, frac=1):
    """
    Calculates the total number of Protons on Target (POT) for a given live time.

    Args:
        on_spill_time (float): The actual time the beam was on spill (in seconds).
        run_mode (str): The operational mode ('1BB', '2BB', or 'custom').
                        Defaults to '1BB'.
        printout (bool): If True, prints calculation details. Defaults to False.
        frac (float): Used only in 'custom' mode as a scaling fraction.

    Returns:
        float: The calculated total number of Protons on Target (total_pot).
    """
    # Numbers based on SU2020 analysis.
    # See https://github.com/Mu2e/su2020/blob/master/analysis/pot_normalization.org

    # Initialize variables that will be dynamically assigned
    mean_pbi = 0.0
    t_cycle = 0.0
    pot_per_cycle = 0.0

    if run_mode == 'custom':
        # Assume some fraction of 1BB
        mean_pbi = 1.6e7 * frac
        t_cycle = 1.33 # seconds
        pot_per_cycle = 4e12 * (1 - frac)

    elif run_mode == '1BB':
        # Single beam batch operation
        mean_pbi = 1.6e7
        t_cycle = 1.33 # seconds
        pot_per_cycle = 4e12

    elif run_mode == '2BB':
        # Two beam batch operation
        mean_pbi = 3.9e7
        t_cycle = 1.4 # seconds
        pot_per_cycle = 8e12

    else:
        raise ValueError(f"Unknown run_mode specified: {run_mode}")

    # --- Common Calculation Steps ---
    num_cycles = on_spill_time / t_cycle
    total_pot = num_cycles * pot_per_cycle

    if printout:
        current_duty_factor = get_duty_factor(run_mode) if run_mode != 'custom' else 'N/A'
        
        print(f"Tcycle= {t_cycle}")
        print(f"POT_per_cycle= {pot_per_cycle:.2e}")
        # 'Livetime' here seems to mean 'Total experiment duration accounting for gaps'
        print(f"Total_Duration= {on_spill_time / current_duty_factor}")
        print(f"NPOT= {total_pot:.2e}")

    return total_pot

#-------------------------------------------------------------------------------------# 

# FORWARD-DIRECTION NORMALIZATION FUNCTIONS REMOVED
# Users should now use the inverted POT calculation functions:
# - pot_for_rmc_events()
# - pot_for_dio_events()  
# - pot_for_rpc_events()
#
# To calculate event expectations, use get_pot() and manual scaling if needed.

# Helper function: work from signal to rmue  
def get_ce_rmue(onspilltime, nsig, run_mode = '1BB'):
    POT = get_pot(onspilltime, run_mode)
    rmue = nsig/(POT * target_stopped_muons_per_pot * CAPTURES_PER_STOPPED_MUON)
    return  rmue


"""
The following section derives the cosmic yield for on spill/off spill for two specific generators (CRY/CORSIKA)
The cosmics are normalized according to the livetime fraction which overlaps with beam (Depends on duty factor and BB mode)
"""
# note this returns CosmicLivetime not # of generated events
def cry_onspill_normalization(livetime, run_mode = '1BB'):
    return livetime
  
# note this returns CosmicLivetime not # of generated events
def corsika_onspill_normalization(livetime, run_mode = '1BB'):
    return livetime


#-------------------------------------------------------------------------------------#
# INVERTED NORMALIZATION FUNCTIONS: Given reconstructed event counts, calculate POT
#-------------------------------------------------------------------------------------#

def pot_for_rmc_events(n_reconstructed, internal, e_min, k_max=90.1, run_mode='1BB'):
    """
    Inverts the RMC normalization: given a number of reconstructed RMC events,
    calculate the required Protons on Target (POT).

    Args:
        n_reconstructed (float): Number of reconstructed RMC events.
        internal (int/bool): Flag (1 or 0) indicating if internal conversion was used.
        e_min (float): Minimum energy threshold for spectrum cut (MeV).
        k_max (float): Maximum possible RMC energy (MeV). Defaults to 90.1 MeV.
        run_mode (str): The operational mode ('1BB' or '2BB'). Defaults to '1BB'.

    Returns:
        float: The required Protons on Target (POT).
    """
    # Generate the RMC energy spectrum internally using the closure approximation
    energies = []
    values = []
    
    start_energy = 57.05
    bin_width = 0.1
    num_bins = int((float(k_max) - start_energy) / bin_width)
    
    for i in range(num_bins):
        temp_e = start_energy + i * bin_width
        x_fit = temp_e / float(k_max)
        spectrum_value = (1 - 2*x_fit + 2*x_fit*x_fit) * x_fit * (1 - x_fit) * (1 - x_fit)
        energies.append(temp_e)
        values.append(spectrum_value)
  
    # Calculate normalization (fraction of spectrum above e_min threshold)
    total_norm = sum(values)
    cut_norm = 0
    
    for i in range(len(values)):
        bin_center = energies[i]
        if (bin_center - bin_width / 2.0) >= float(e_min):
            cut_norm += values[i]

    if total_norm == 0:
        fraction_sampled = 0.0
    else:
        fraction_sampled = cut_norm / total_norm
    
    # Build the scaling factor
    scaling_factor = (
        target_stopped_muons_per_pot *
        CAPTURES_PER_STOPPED_MUON *
        RMC_GT_57_PER_CAPTURE *
        fraction_sampled
    )
    
    # Apply internal conversion scaling if requested
    is_internal_conversion = bool(int(internal))
    if is_internal_conversion:
        scaling_factor *= INTERNAL_PER_RMC
    
    # Avoid division by zero
    if scaling_factor == 0:
        raise ValueError("Scaling factor is zero. Cannot calculate POT with these parameters.")
    
    # Invert: POT = n_events / scaling_factor
    required_pot = n_reconstructed / scaling_factor
    
    print(f"RMC: n_events={n_reconstructed}, e_min={e_min}, k_max={k_max}, internal={internal}")
    print(f"RMC: fraction_sampled={fraction_sampled}, scaling_factor={scaling_factor:.6e}")
    print(f"RMC: required_POT={required_pot:.6e}")
    
    return required_pot


def pot_for_dio_events(n_reconstructed, e_min, run_mode='1BB'):
    """
    Inverts the DIO normalization: given a number of reconstructed DIO events,
    calculate the required Protons on Target (POT).

    Args:
        n_reconstructed (float): Number of reconstructed DIO events.
        e_min (float): Minimum energy threshold for the DIO spectrum cut (MeV).
        run_mode (str): The operational mode ('1BB' or '2BB'). Defaults to '1BB'.

    Returns:
        float: The required Protons on Target (POT).
    """
    # Load the DIO energy spectrum data
    spectrum_file_path = os.path.join(
        os.environ["MUSE_WORK_DIR"],
        "Production/JobConfig/ensemble/tables/heeck_finer_binning_2016_szafron.tbl"
    )
    
    energies = []
    values = []

    try:
        with open(spectrum_file_path, 'r') as spec_file:
            for line in spec_file:
                if not line.strip() or line.strip().startswith('#'): continue
                try:
                    energy, value = map(float, line.split())
                    energies.append(energy)
                    values.append(value)
                except ValueError:
                    print(f"Warning: Could not parse line in spectrum file: {line.strip()}")

    except FileNotFoundError:
        raise FileNotFoundError(f"DIO spectrum file not found at: {spectrum_file_path}")

    # Calculate normalization (fraction of spectrum above e_min)
    total_norm = sum(values)
    cut_norm = 0

    for i in range(len(values)):
        if energies[i] >= e_min:
            cut_norm += values[i]

    if total_norm == 0:
        fraction_sampled = 0.0
    else:
        fraction_sampled = cut_norm / total_norm

    # Build the scaling factor
    scaling_factor = (
        target_stopped_muons_per_pot *
        DIO_PER_STOPPED_MUON *
        fraction_sampled
    )
    
    # Avoid division by zero
    if scaling_factor == 0:
        raise ValueError("Scaling factor is zero. Cannot calculate POT with these parameters.")
    
    # Invert: POT = n_events / scaling_factor
    required_pot = n_reconstructed / scaling_factor
    
    print(f"DIO: n_events={n_reconstructed}, e_min={e_min}")
    print(f"DIO: fraction_sampled={fraction_sampled}, scaling_factor={scaling_factor:.6e}")
    print(f"DIO: required_POT={required_pot:.6e}")
    
    return required_pot


def pot_for_rpc_events(n_reconstructed, t_min, internal, e_min, run_mode='1BB'):
    """
    Inverts the RPC normalization: given a number of reconstructed RPC events,
    calculate the required Protons on Target (POT).

    Handles both standard RPC and internal conversion events based on the 'internal' flag.

    Args:
        n_reconstructed (float): Number of reconstructed RPC events.
        t_min (float): Minimum time threshold (seconds) (Note: not used in current logic).
        internal (int/bool): Flag (1 or 0) to include internal conversion scaling.
        e_min (float): Minimum energy threshold for spectrum cut (MeV).
        run_mode (str): The operational mode ('1BB' or '2BB'). Defaults to '1BB'.

    Returns:
        float: The required Protons on Target (POT).
    """
    # Load the RPC energy spectrum data
    spectrum_file_path = os.path.join(
        os.environ["MUSE_WORK_DIR"],
        "Production/JobConfig/ensemble/tables/rpcspectrum.tbl"
    )
    
    energies = []
    values = []

    try:
        with open(spectrum_file_path, 'r') as spec_file:
            for line in spec_file:
                if not line.strip() or line.strip().startswith('#'): continue
                try:
                    energy, value = map(float, line.split())
                    energies.append(energy)
                    values.append(value)
                except ValueError:
                    print(f"Warning: Could not parse line in spectrum file: {line.strip()}")

    except FileNotFoundError:
        raise FileNotFoundError(f"RPC spectrum file not found at: {spectrum_file_path}")

    # Calculate normalization (fraction of spectrum above e_min)
    total_norm = sum(values)
    cut_norm = 0
    for i in range(len(values)):
        if energies[i] >= float(e_min):
            cut_norm += values[i]

    if total_norm == 0:
        rpc_e_sample_frac = 0.0
    else:
        rpc_e_sample_frac = cut_norm / total_norm

    # Calculate efficiency terms based on simulation globals
    filter_efficiency = float(num_pion_filters) / float(num_pion_stops)
    survival_probability_weight = float(selected_sum_of_weights) / float(num_pion_resamples)

    # Build the scaling factor
    scaling_factor = (
        target_stopped_pions_per_pot *
        filter_efficiency *
        survival_probability_weight *
        RPC_PER_STOPPED_PION *
        rpc_e_sample_frac
    )

    # Apply internal conversion scaling if requested
    is_internal_conversion = bool(int(internal))
    if is_internal_conversion:
        scaling_factor *= INTERNAL_RPC_PER_RPC
    
    # Avoid division by zero
    if scaling_factor == 0:
        raise ValueError("Scaling factor is zero. Cannot calculate POT with these parameters.")
    
    # Invert: POT = n_events / scaling_factor
    required_pot = n_reconstructed / scaling_factor
    
    print(f"RPC: n_events={n_reconstructed}, e_min={e_min}, t_min={t_min}, internal={internal}")
    print(f"RPC: fraction_sampled={rpc_e_sample_frac}, scaling_factor={scaling_factor:.6e}")
    print(f"RPC: required_POT={required_pot:.6e}")
    
    return required_pot


#-------------------------------------------------------------------------------------#
# RECONSTRUCTION EFFICIENCY CALCULATION
#-------------------------------------------------------------------------------------#

def get_reco_eff(signal, filelist_path=None, verbose=False):
    """
    Calculates the reconstruction efficiency for a given signal process
    by counting reconstructed events vs. generated events across ROOT files.

    Args:
        signal (str): Name of the signal process (e.g., 'RMC', 'DIO', 'RPC').
        filelist_path (str): Path to the file containing list of ROOT file paths.
                            If None, assumes a file named 'filenames_{signal}' exists.
        verbose (bool): If True, prints progress information. Defaults to False.

    Returns:
        float: Reconstruction efficiency (reco_events / gen_events).
               Returns 0.0 if gen_events is 0 or files cannot be processed.
    """
    # Determine the filelist path
    if filelist_path is None:
        filelist_path = f"filenames_{signal}"
    
    # Initialize event counters
    reco_events = 0
    gen_events = 0
    
    try:
        # Open and read the filelist
        with open(filelist_path, 'r') as flist:
            files = [line.strip() for line in flist if line.strip()]
    except FileNotFoundError:
        raise FileNotFoundError(f"Filelist not found: {filelist_path}")
    
    if verbose:
        print(f"Processing signal: {signal}")
        print(f"Number of files to process: {len(files)}")
    
    # Loop over files in the list
    for fn in files:
        if verbose:
            print(f"  Processing file: {fn}")
        
        try:
            # Open the ROOT file
            fin = ROOT.TFile(fn)
            
            if fin.IsZombie():
                print(f"  Warning: Could not open {fn}")
                continue
            
            # Get the Events tree and count reconstructed events
            te = fin.Get("Events")
            if te:
                n_entries = te.GetEntries()
                reco_events += n_entries
                if verbose:
                    print(f"    Reco events in file: {n_entries}, cumulative: {reco_events}")
            
            # Get the SubRuns tree for generated event count
            t_subruns = fin.Get("SubRuns")
            if t_subruns:
                # Find the GenEventCount branch
                bl = t_subruns.GetListOfBranches()
                branch_name = ""
                
                for i in range(bl.GetEntries()):
                    if bl[i].GetName().startswith("mu2e::GenEventCount"):
                        branch_name = bl[i].GetName()
                        break
                
                # Sum up generated events
                if branch_name:
                    for i in range(t_subruns.GetEntries()):
                        t_subruns.GetEntry(i)
                        gen_count_obj = getattr(t_subruns, branch_name)
                        gen_events += gen_count_obj.product().count()
                else:
                    if verbose:
                        print(f"    Warning: GenEventCount branch not found in SubRuns tree")
            
            fin.Close()
            
        except Exception as e:
            print(f"  Error processing {fn}: {str(e)}")
            continue
    
    # Calculate reconstruction efficiency
    if gen_events == 0:
        if verbose:
            print("Warning: No generated events found. Returning 0.0")
        return 0.0
    
    reco_eff = reco_events / gen_events
    
    if verbose:
        print(f"\nSummary for {signal}:")
        print(f"  Total reco events: {reco_events}")
        print(f"  Total gen events: {gen_events}")
        print(f"  Reconstruction efficiency: {reco_eff:.6f}")
    
    return reco_eff


if __name__ == '__main__':
  tst_1BB = get_pot(9.52e6)
  tst_2BB = get_pot(1.58e6)
  print("SU2020 POT:", tst_1BB, tst_2BB)