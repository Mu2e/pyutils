import hist
import gc
import sys
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import uproot
import awkward as ak
import argparse
from sklearn.model_selection import train_test_split
import pandas as pd
import xgboost as xgb

# refana
from compare import Compare
from count import Count
from analyze import Analyze
from cut_manager import CutManager
from MLSelect import MLSelect
from fits import Fits
import stats
from pyutils.pyprocess import Processor, Skeleton
from pyutils.pyplot import Plot
from pyutils.pyprint import Print
from pyutils.pyselect import Select
from pyutils.pyvector import Vector

class AnaProcessor(Skeleton):
    """custom file processor 
    
    This class inherits from the Skeleton defined in pyutils/pyprocess base class, which provides the 
    basic structure and methods withing the Processor framework 
    """
    def __init__(self, file_list_path, jobs=1, sign="minus", cuts=[], location='disk'):
        """Initialise your processor with specific configuration
        
        This method sets up all the parameters needed for this specific analysis.
        """
        # Call the parent class's __init__ method first
        # This ensures we have all the base functionality properly set up
        super().__init__()

        # Now override parameters from the Skeleton with the ones we need
        self.file_list_path = file_list_path

        self.branches = { 
            "evt" : [
                "run",
                "subrun",
                "event",
                "trig_tprHelixDe_ipa",
                "trig_TriggerableLHPath"
            ],
            "crv" : [
                "crvcoincs.time",
            ],
            "trk" : [
                "trk.nactive", 
                "trk.pdg", 
                "trk.status",
                "trkqual.valid",
                "trkqual.result",
                "trkpid.valid",
                "trkpid.result"
            ],
            "trkfit" : [
                "trksegs",
                "trksegsmc",
                "trksegpars_lh"
            ],
            "trkmc" : [
                "trkmcsim",
                "trkmc.valid"
            ]
            
            
        }
        self.tree_path = "ntuple"
        #self.filelist = "filelist.txt"          # text file containing list of files
        self.use_remote = True     # Use remote file via mdh
        if str(location)  == "local":
          self.use_remote = False
        self.location = str(location)     # File location
        self.max_workers = jobs      # Limit the number of workers
        self.verbosity = 2         # Set verbosity 
        self.use_processes = True  # Use processes rather than threads
        
        # Now add your own analysis-specific parameters 

        # Init analysis methods
        # Would be good to load an analysis config here 
        self.analyse = Analyze(verbosity=0, sign=sign, cut_switch=cuts)
            
        # Custom prefix for log messages from this processor
        self.print_prefix = "[AnaProcessor] "
        print(f"{self.print_prefix}Initialised")
    
    # ==========================================
    # Define the core processing logic
    # ==========================================
    # This method overrides the parent class's process_file method
    # It will be called automatically for each file by the execute method
    def process_file(self, file_name): 
        """Process a single ROOT file
        
        This method will be called for each file in our list.
        It extracts data, processes it, and returns a result.
        
        Args:
            file_name: Path to the ROOT file to process
            
        Returns:
            A tuple containing the histogram (counts and bin edges)
        """
        try:
            # Create a local pyprocess Processor to extract data from this file
            # This uses the configuration parameters from our class
            processor = Processor(
                use_remote=self.use_remote,     # Use remote file via mdh
                location=self.location,         # File location
                verbosity=0 # self.verbosity        # Reduce output in worker threads
            )
            
            # Process the files using multithreading
            data = processor.process_data(
                file_name = file_name,
                branches = self.branches
            )
            
            # ---- Analysis ----            
            results = self.analyse.execute(data, file_name)

            # Clean up
            gc.collect()

            return results 
        
        except Exception as e:
            # Handle any errors that occur during processing
            print(f"{self.print_prefix}Error processing {file_name}: {e}")
            return None
            
def combine_cut_flows( cut_flow_list):
    """Combine a list of cut flows after multiprocessing 
    
    Args:
        cut_flows: List of cut statistics lists from different files

    Returns:
        list: Combined cut statistics
    """        
    # Return empty list if no input
    if not cut_flow_list:
        self.logger.log(f"No cut flows to combine", "error")
        return []

    try:
        # Use the first (now filtered) list as template
        template = cut_flow_list[0]
        
        # Use the template to initialise combined stats
        combined_cut_flow = []
        for cut in template:
            # Create a copy (needed?)
            cut_copy = {k: v for k, v in cut.items()}
            # Reset the event count
            cut_copy["events_passing"] = 0
            combined_cut_flow.append(cut_copy)
        
        # Create a mapping of cut names to indices in combined_stats 
        cut_name_to_index = {cut["name"]: i for i, cut in enumerate(combined_cut_flow)}
        
        # Sum up events_passing for each cut across all files
        for cut_flow in cut_flow_list:
            for cut in cut_flow:
                cut_name = cut["name"]
                # Only process cuts that are in our combined_stats
                if cut_name in cut_name_to_index:
                    idx = cut_name_to_index[cut_name]
                    combined_cut_flow[idx]["events_passing"] += cut["events_passing"]
        
        # Recalculate percentages
        if combined_cut_flow and combined_cut_flow[0]["events_passing"] > 0:
            total_events = combined_cut_flow[0]["events_passing"]
            
            for i, cut in enumerate(combined_cut_flow):
                events = cut["events_passing"]
                
                # Absolute percentage
                cut["absolute_frac"] = (events / total_events) * 100.0
                
                # Relative percentage
                if i == 0:  # "No cuts"
                    cut["relative_frac"] = 100.0
                else:
                    prev_events = combined_cut_flow[i-1]["events_passing"]
                    cut["relative_frac"] = (events / prev_events) * 100.0 if prev_events > 0 else 0.0

        cut_manager = CutManager(verbosity=0)
        print("================== Total Cut Flow =======================")
        cut_manager.print_cut_stats(stats=combined_cut_flow, active_only=True, csv_name="cut_stats.csv")
        return combined_cut_flow
    
    except Exception as e:
        print(f"Exception when combining cut flows: {e}", "error")
        raise
                    
def combine_arrays(results):
    """Combine filtered arrays from multiple files
    """
    arrays_to_combine = []
    # Check if we have results
    if not results:
        return None
    # Loop through all files
    for i, result in enumerate(results): #
        if len(result) == 0:
            continue
        # Concatenate arrays
        arrays_to_combine.append(result["filtered_data"])
    return ak.concatenate(arrays_to_combine)


def count_particle_types(data):
    """
    Counts the occurrences of different particle types based on
    simulation data, leveraging the properties of Awkward Arrays.

    Args:
        data (ak.Array): An Awkward Array containing simulation data,
                         including 'trkmc' with 'trkmcsim' nested field.

    Returns:
        list: A list containing particle type identifiers for each event.
    """

    # Check for empty data
    if ak.num(data['trkmc'], axis=0) == 0:
        print("No events found in the data.")
        return []

    # Vectorized approach for efficiency using Awkward Array operations
    #  This is generally faster than looping through events individually for large datasets.

    # Get startCode for the first track in each event, handling empty lists
    # Use ak.firsts to safely get the first element or None if the list is empty
    proc_codes = ak.firsts(data['trkmc']['trkmcsim', 'startCode'], axis=1) 
    gen_codes = ak.firsts(data['trkmc']['trkmcsim', 'gen'], axis=1)
    vector = Vector()

    #rhos = vector.get_rho(data['trkmc','trkmcsim'],'pos')
    vec = vector.get_vector(branch=data['trkmc','trkmcsim'],vector_name='pos')
    rhos = vec.rho
    position = ak.firsts(rhos, axis=1) 

    #position = ak.firsts(sim_pos_vec.rho, axis = 1)
    # Use vectorized comparisons and selection for counting
    dio_mask = (proc_codes == 166) & (position <= 75) # Create boolean mask for DIO events
    ipa_mask = (proc_codes == 166) & (position > 75) # Create boolean mask for IPA DIO events
    cem_mask = ((proc_codes == 168)  | (proc_codes == 167)  ) # Create boolean mask for CE events
    cep_mask = ((proc_codes == 176) | (proc_codes == 169) )  # Create boolean mask for CE events
    erpc_mask = (proc_codes == 178)  # Create boolean mask for external RPC events
    irpc_mask = (proc_codes == 179)  # Create boolean mask for internal RPC events
    ermc_mask = (proc_codes == 172)  # Create boolean mask for external RMC events
    irmc_mask = (proc_codes == 171)  # Create boolean mask for internal RMC events
    flate_mask = (proc_codes == 173)  # Create boolean mask for internal flate events
    flateplus_mask = (proc_codes == 174)  # Create boolean mask for internal flate events
    cosmic_mask = ((gen_codes == 44) | (gen_codes == 38))  # Create boolean mask for cosmic events
    #combined_rpc_mask = (proc_codes == 178) |  (proc_codes == 179) # Create boolean mask for all RPC events

    # Combine masks to identify 'other' events
    other_mask = ~(dio_mask | cem_mask | erpc_mask | irpc_mask | cosmic_mask | ipa_mask | irmc_mask | ermc_mask | cep_mask)

    # Initialize particle_count with -2 for 'others'
    particle_count = ak.zeros_like(proc_codes, dtype=int) - 2
    
    # Assign particle types based on masks
    particle_count = ak.where(dio_mask, 166, particle_count)
    particle_count = ak.where(ipa_mask, 0, particle_count)
    particle_count = ak.where(cosmic_mask, -1, particle_count)
    particle_count = ak.where(other_mask, -2, particle_count)
    particle_count = ak.where(irpc_mask, 179, particle_count)
    particle_count = ak.where(erpc_mask, 178, particle_count)
    particle_count = ak.where(irmc_mask, 171, particle_count)
    particle_count = ak.where(ermc_mask, 172, particle_count)
    particle_count = ak.where(cem_mask, 168, particle_count)
    particle_count = ak.where(cep_mask, 176, particle_count)
    particle_count = ak.where(flate_mask, 173, particle_count)
    particle_count = ak.where(flateplus_mask, 174, particle_count)
    #particle_count = ak.where(combined_rpc_mask, 999, particle_count)
    particle_count_return = particle_count
    #particle_count = ak.any(dio_mask, axis=1)
    # Count the occurrences of each particle type
    counts = {
        166: (len(particle_count[ak.any(dio_mask, axis=1)==True])),
        0: (len(particle_count[ak.any(ipa_mask, axis=1)==True])),
        168:  (len(particle_count[ak.any(cem_mask, axis=1)==True])),
        176:  (len(particle_count[ak.any(cep_mask, axis=1)==True])),
        178:  (len(particle_count[ak.any(erpc_mask, axis=1)==True])),
        179:  (len(particle_count[ak.any(irpc_mask, axis=1)==True])),
        171:  (len(particle_count[ak.any(irmc_mask, axis=1)==True])),
        172:  (len(particle_count[ak.any(ermc_mask, axis=1)==True])), 
        173:  (len(particle_count[ak.any(flate_mask, axis=1)==True])),
        174:  (len(particle_count[ak.any(flateplus_mask, axis=1)==True])), 
        -1:  (len(particle_count[ak.any(cosmic_mask, axis=1)==True])),
        -2:  (len(particle_count[ak.any(other_mask, axis=1)==True])),
        #999: (len(particle_count[ak.any(combined_rpc_mask, axis=1)==True])),
    }
      
    # Print the yields to terminal for cross-check
    print("===== MC truth yields for full momentum and time range=====")
    print("N_DIO: ", counts[166])
    print("N_IPA: ", counts[0])
    print("N_CEM: ", counts[168])
    print("N_CEP: ", counts[176])
    print("N_eRPC: ", counts[178])
    print("N_iRPC: ", counts[179])
    #print("N_combined:",counts[999])
    print("N_eRMC: ", counts[171])
    print("N_iRMC: ", counts[172])
    print("N_flateminus: ", counts[173])
    print("N_flateplus: ", counts[174])
    print("N_cosmic: ", counts[-1])
    print("N_others: ", counts[-2])
    
    # Now return a 1D list with one element per event corresponding to the primary trk
    #particle_count_return = ak.flatten(particle_count_return, axis=None)
    #    The mask will be True for values that are not -2.
    primary_mask = particle_count_return != -2

    # Apply the mask to the flattened array to select desired elements
    particle_count_return = particle_count_return[primary_mask]
    particle_count_return = [[sublist[0]] for sublist in particle_count_return]
    particle_count_return = ak.flatten(particle_count_return, axis=None)
    print("returned particle count length",len(particle_count_return))
    
    return particle_count_return


def selectML( files, cuts, locations, columns):
    """ creates features lists in useful form for XGBoost cuts """

    t0err = []
    active = []
    recomom = []
    recotime = []
    mc_count = []
    closest_crv_diff = []
    nST = []
    nOPA = []
    
    comparison = Compare()
    for i, fil in enumerate(files):
      ana_processor = AnaProcessor(fil, args.jobs, args.sign, cuts[i], locations[i])
      results = ana_processor.execute()
     
      # Create an instance of our custom processor
      combine_result = combine_arrays(results)
      cutlist = []
      for i, result in enumerate(results):
        cutlist.append(result["cut_stats"])
      combine_cutflows = combine_cut_flows(cutlist)
    
      # run cat
      mc_count.append(count_particle_types(combine_result))

      selector = Select()
      
      # select only track front to fit to
      trk_front = selector.select_surface(combine_result['trkfit'], surface_name="TT_Front")

      # did the track intersect the ST?
      has_st  = selector.has_ST(combine_result['trkfit'])

      # combined mask
      trkfit_ent = combine_result['trkfit']["trksegs"].mask[(trk_front) ] #& (no_opa) & (has_st)
       
      # make vector mag branch
      vector = Vector()
      mom_mag = vector.get_mag(trkfit_ent ,'mom')

      time = ak.nan_to_none(trkfit_ent['time'])
      time = ak.drop_none(time)
      
      
      # plot cut distributions
      test_mask = (trk_front) & (has_st) #& (no_opa)& (has_st)
      
     
      # Get track and coincidence times
      trk_times = combine_result['trkfit']["trksegs"]["time"][test_mask]  # events × tracks × segments
      coinc_times = combine_result["crv"]["crvcoincs.time"]                  # events × coincidences

      coinc_broadcast = coinc_times[:, None, None, :]  # Add dimensions for tracks and segments
      trk_broadcast = trk_times[:, :, :, None]         # Add dimension for coincidences

      # Calculate time differences
      dt = abs(trk_broadcast - coinc_broadcast)
      
      
      nST.append(ak.sum(selector.select_surface(combine_result['trkfit'], surface_name="ST_Foils"), axis=-1))
      nOPA.append(ak.sum(selector.select_surface(combine_result['trkfit'], surface_name="OPA"), axis=-1))
      t0err.append(combine_result['trkfit']["trksegpars_lh"].mask[test_mask]['t0err']) 
      closest_crv_diff.append(dt)
      active.append(combine_result['trk'].mask[test_mask]["trk.nactive"])
      recotime.append(trkfit_ent['time'])
      recomom.append(mom_mag)
    ML = MLSelect()
    ML.runML(recomom, recotime, t0err, active, nST, nOPA,  closest_crv_diff, mc_count)
 


def compare_datasets( files, cuts, locations, columns, signs):
    """
    Allows for different types of comparisons:
    
    1) could compare different files same cuts
    2) could compare same file different cut sets
    
    Args:
      files : list of file lists (.txt files)
      cuts : list of cut switches (True/False of each cut)
      locations : list of locations e.g. tape or disk
      columns : labels for the two things you are comparing eg. [dataset 1, dataset 2]
    """

    rmax = []
    d0 = []
    tanDip = []
    t0err = []
    active = []
    trkqual = []
    recomom = []
    truemom = []
    mc_count = []
    resolutions = []
    nST = []
    nOPA = []
    originmom = []
    losses = []
    times = []
    crv = []
    trkpid = []
    
    comparison = Compare()
    fit = Fits()
    for i, fil in enumerate(files):
      ana_processor = AnaProcessor(fil, args.jobs, signs[i], cuts[i], locations[i])
      results = ana_processor.execute()
     
      # Create an instance of our custom processor
      combine_result = combine_arrays(results)
      cutlist = []
      for i, result in enumerate(results):
        cutlist.append(result["cut_stats"])
      combine_cutflows = combine_cut_flows(cutlist)
    
      # run cat
      mc_count.append(count_particle_types(combine_result))

      selector = Select()
      
      # select only track front to fit to
      trk_front = selector.select_surface(combine_result['trkfit'], surface_name="TT_Front")

      # did the track intersect the ST?
      has_st  = selector.has_ST(combine_result['trkfit'])

      # did the track intersect the OPA?
      no_opa  = selector.has_OPA(combine_result['trkfit'])

      # combined mask
      trkfit_ent = combine_result['trkfit']["trksegs"].mask[(trk_front) ] #& (no_opa) & (has_st)
          
      trk_front_mc = selector.select_surface(combine_result['trkfit'], surface_name="TT_Front",branch_name="trksegsmc")
      trkfit_ent_mc = combine_result['trkfit']["trksegsmc"].mask[(trk_front_mc) ]

      # make vector mag branch
      vector = Vector()
      mom_mag = vector.get_mag(trkfit_ent ,'mom')

      #mom_mag = ak.nan_to_none(mom_mag)
      #mom_mag = ak.drop_none(mom_mag)

      time = ak.nan_to_none(trkfit_ent['time'])
      time = ak.drop_none(time)
      
      vector = Vector()
      mom_mag_mc = vector.get_mag(trkfit_ent_mc ,'mom')
      
      # get resolution:
      resolution = comparison.compare_resolution(mom_mag,mom_mag_mc)
      
      # for loss studies:
      origin = combine_result['trkmc']["trkmcsim"].mask[(combine_result['trkmc']["trkmcsim"]["rank"] == 0) & (combine_result['trkmc']["trkmcsim"]["nhits"] > 0)]
      originmom.append((vector.get_mag(origin,'mom')))

      # get resolution:
      resolution = comparison.compare_resolution(mom_mag_mc, mom_mag)
      loss  = comparison.compare_resolution( (vector.get_mag(origin,'mom')), mom_mag_mc)
      
      # plot cut distributions
      test_mask = (trk_front) & (has_st) #& (no_opa)& (has_st)
      
      # for CRV:
      # Get track and coincidence times
      trk_times = combine_result['trkfit']["trksegs"]["time"][trk_front]  # events × tracks × segments
      coinc_times = combine_result["crv"]["crvcoincs.time"]                  # events × coincidences

      coinc_broadcast = coinc_times[:, None, None, :]  # Add dimensions for tracks and segments
      trk_broadcast = trk_times[:, :, :, None]         # Add dimension for coincidences

      # Calculate time differences
      dt = abs(trk_broadcast - coinc_broadcast)
      
      nST.append(ak.sum(selector.select_surface(combine_result['trkfit'], surface_name="ST_Foils"), axis=-1))
      nOPA.append(ak.sum(selector.select_surface(combine_result['trkfit'], surface_name="OPA"), axis=-1))
      rmax.append(combine_result['trkfit']["trksegpars_lh"].mask[test_mask]['maxr']) 
      d0.append(combine_result['trkfit']["trksegpars_lh"].mask[test_mask]['d0']) 
      tanDip.append(combine_result['trkfit']["trksegpars_lh"].mask[test_mask]['tanDip']) 
      t0err.append(combine_result['trkfit']["trksegpars_lh"].mask[test_mask]['t0err']) 
      trkqual.append(combine_result['trk'].mask[test_mask]["trkqual.result"])
      trkpid.append(combine_result['trk'].mask[test_mask]["trkpid.result"])
      active.append(combine_result['trk'].mask[test_mask]["trk.nactive"])
      recomom.append(mom_mag)
      times.append(time)
      losses.append(loss)
      truemom.append(mom_mag_mc)
      resolutions.append(resolution)
      crv.append(dt)
      #print(">>>>> Has Trk Front ", len(mom_mag))
      #print(">>>>>>> Is in Fit region ",len(mom_mag.mask[mom_mag < 115 & mom_mag > 95]))
    fit.fit_momentum(originmom, 85, 105, opt="poly", label = "Origin Momentum [MeV/c]", compare_labels=columns)
    fit.fit_momentum(losses, 0.1,2,opt="landau",label = "Origin - True at TrkEnt Momentum [MeV/c]", compare_labels=columns)
    fit.fit_momentum(resolutions, -1,1,opt="dscb", label = "Reco - True Momentum at TrkEnt [MeV/c]",compare_labels=columns)
    comparison.plot_particle_counts(mc_count, columns)
    comparison.plot_variable(crv, "|DT| [ns]","DT",0,300, [150,150],[150,150], mc_count,columns)
    comparison.plot_variable(nST, "nST","nST",0,15, [1,1],[1,1], mc_count,columns, 15)
    comparison.plot_variable(nOPA, "nOPA","nOPA",0,4, [0,0],[0,0], mc_count,columns,4)
    comparison.plot_variable(rmax, "rmax", "rmax",300,750, [450,450],[680,680], mc_count,columns)
    comparison.plot_variable(d0, "d0", "d0",0, 250, [100,100], [100,100], mc_count,columns)
    comparison.plot_variable(tanDip, "tanDip","tanDip",-1,2.5, [0.557,0.557], [1.0,1.0],mc_count,columns)
    comparison.plot_variable(trkqual, "trkqual", "trkqual", 0,1,[0.2,0.2], [0.2, 0.2], mc_count,columns)
    comparison.plot_variable(trkpid, "trkpid", "trkpid", 0,1,[0.2,0.2], [0.2, 0.2], mc_count,columns)
    comparison.plot_variable(t0err, "t0err","t0err", 0,1, [0.9,0.9],[0.9,0.9], mc_count,columns)
    comparison.plot_variable(active, "nactive","nactive", 0,50, [20,20],[0.9,0.9], mc_count,columns)
    comparison.plot_variable(recomom, "Reconstructed Momentum at TrkEnt [MeV/c]","recomom", 95,115, [103.9,103.9],[105.1,105.1], mc_count,columns)
    comparison.plot_variable(times, "Time at TrkEnt [ns]","time", 0, 1700, [640,640],[1650,1650], mc_count,columns)
    comparison.plot_variable(truemom, "True Momentum at TrkEnt [MeV/c]","truemom", 95,115, [103.9,103.9],[105.1,105.1], mc_count,columns)
    #comparison.plot_resolution(resolutions, "True - Reco at TrkEnt [MeV/c]","recomom", -10,3, columns, density=False)
    
    

def trkqual_compare( files, cuts, locations, columns):
    trkquals = []
    mc_count = []
    comparison = Compare()
    for k, fil in enumerate(files):
      ana_processor = AnaProcessor(fil, args.jobs, args.sign, cuts[k], locations[k])
      results = ana_processor.execute()
   
      # Create an instance of our custom processor
      combine_result = combine_arrays(results)
      cutlist = []
      for i, result in enumerate(results):
        cutlist.append(result["cut_stats"])
      combine_cutflows = combine_cut_flows(cutlist)

      # run cat
      mc_count.append(count_particle_types(combine_result))

      selector = Select()
      
      # select only track front to fit to
      trk_front = selector.select_surface(combine_result['trkfit'], surface_name="TT_Front")

      # did the track intersect the ST?
      has_st  = selector.has_ST(combine_result['trkfit'])

      # did the track intersect the OPA?
      no_opa  = selector.has_OPA(combine_result['trkfit'])

      # combined mask
      trkfit_ent = combine_result['trkfit']["trksegs"].mask[(trk_front) ] #& (no_opa) & (has_st)

      # plot cut distributions
      test_mask = (trk_front) #& (has_st) #& (no_opa)& (has_st)

      if(k==0): 
        trkquals.append((combine_result['trk'].mask[test_mask]["trkqual.result"]))
      if(k==1):
        trkquals.append((combine_result['trk'].mask[test_mask]["trkqual3.result"])) 
    comparison.plot_variable(trkquals, "trkqual","trkqual", 0, 1, [0.2,0.2],[0.2,0.2], mc_count,columns)

   
def run_count(args, cuts):
  """ runs count analysis """
  ana_processor = AnaProcessor(args.file, args.jobs, args.sign, cuts, args.loc)
  results = ana_processor.execute()

  # Create an instance of our custom processor
  combine_result = combine_arrays(results)
  cutlist = []
  for i, result in enumerate(results):
    cutlist.append(result["cut_stats"])
  combine_cutflows = combine_cut_flows(cutlist)

  # run cat
  # make vector mag branch
  mc_count = count_particle_types(combine_result)

  selector = Select()
  
  # select only track front to fit to
  trk_front = selector.select_surface(combine_result['trkfit'], surface_name="TT_Front")

  # did the track intersect the ST?
  has_st  = selector.has_ST(combine_result['trkfit'])

  # did the track intersect the OPA?
  no_opa  = selector.has_OPA(combine_result['trkfit'])

  # combined mask
  trkfit_ent = combine_result['trkfit']["trksegs"].mask[(trk_front)]#& (no_opa)& (has_st)] 
  
  # make vector mag branch
  vector = Vector()
  mom_mag = vector.get_mag(trkfit_ent ,'mom')

  mom_mag = ak.nan_to_none(mom_mag)
  mom_mag = ak.drop_none(mom_mag)

  time = ak.nan_to_none(trkfit_ent['time'])
  time = ak.drop_none(time)
  
  # run count
  signal_mom = []
  if (str(args.sign) == "minus"):
    signal_mom = [103.9, 105.1]
  if (str(args.sign) == "plus"):
    signal_mom = [90.0,92.0]
  signal_time = [640., 1650.]
  
  counter = Count(mom_mag, time, mc_count,signal_mom,signal_time, args.sign)
  signal, background = counter.ExtractReco()
  
  signal_SR, background_SR = counter.CheckMCTruth()

  #stats.anlayze_result_with_err(signal_SR, background_SR )
  print("signal", signal, "background",background) 

def plot_eff(val_overlay, mc_count, code, name, lo=95, hi=120):
  """
  used to plot efficiency as a 
  """
  fig, ax = plt.subplots(figsize=(10, 6))
  bin_centers = []
  bin_contents = []
  for i, val in enumerate(val_overlay):
    val = ak.drop_none(val)
    val = val.mask[mc_count[i] == int(code)]
    val = np.array(ak.flatten(val, axis=None))
    n, bins, patch = ax.hist(val,range=(lo,hi), bins=20)
    bin_centers.append(0.5 * (bins[:-1] + bins[1:]))
    bin_contents.append(n)
  comparison = Compare()
  comparison.plot_cut_eff(bin_contents[0], bin_contents[1],bin_centers[0],"Efficiency",name)






# Create an instance of our custom processor
def  main(args):
  """ main driver function to run analysis
  """

  """
  list which cuts to switch on/off:
  0) Is electron  1) Downstream 2) Track fit quality  3) trkpid 4) Minimum hits 5) t0 6) t0err 7) rMax 8) d0 9) tanDip 10) CRV 11) newST 12) new OPA 13) early time
  """
  old = [True, True, True, False, True, False,True, True, True, True,True, False, False, False]
  old_not0 = [True, True, True, False, True, False,True, True, True, True,True, False, False]
  
  new = [True, True, True,  True, True, False, True, False, False,False,True, True,True, False]
  nocuts = [True, True, False, False, False, False, False, False, False,False,False,False,False, False]

  # general plots
  files = [args.file]
  signs = [args.sign]
  locations = [args.loc]
  columns = ["Mock Data Example"]
  compare_datasets(files, [new], locations, columns, signs)
  run_count(args, new)
  """
  # run ML test
  files = ["test_CE.txt","test_MDS.txt"]
  locations = ["disk", "local"]
  columns = ["new","new"]
  selectML( files, [new,new], locations, columns)

  
  # + and -
  signs = [args.sign, args.sign]
  files = [args.file, args.file]
  locations = [args.loc,args.loc]
  columns = ["CutSetC","new"]
  compare_datasets(files, [old,new], locations, columns, signs)
  #files = [args.file,args.file]
  #trkqual_compare(files, [new,new], [args.loc,args.loc], ["v1","v2"])
  
  # run the counting analysis on the chosen current file
  #run_count(args, new)
  """
  
def PrintArgs(args):
  """
  prints users input parameters
  """
  print("========= [RefAna/pyCount/process.py]✅  Analyzing with user opts: ===========")
  print("file:", args.file)
  print("number of processes (njobs - optimal is 1 per file):", args.jobs)
  print("verbose: ", args.verbose)

if __name__ == "__main__":
    # list of input arguments, defaults should be overridden
    parser = argparse.ArgumentParser(description='command arguments', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--file", type=str, required=True, help="filename or file list name (text file list,fullpaths)")
    parser.add_argument("--loc", type=str, required=False, default='disk', help="location of files")
    parser.add_argument("--sign", type=str, required="minus", help="sign of the signal being sought in words")
    parser.add_argument("--jobs", type=int, required=False, default=1,help="use if more than one file, should be nfiles")
    parser.add_argument("--verbose", default=1, help="verbose")
    args = parser.parse_args()
    (args) = parser.parse_args()

    # if verbose print the user input
    if(args.verbose > 0):
      PrintArgs(args)
    
    # run main function
    main(args)




