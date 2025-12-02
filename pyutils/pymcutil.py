import numpy as np
import awkward as ak
import sys
import matplotlib.pyplot as plt
from pyutils.pyselect import Select
from pyutils.pyvector import Vector
from pyutils.pyplot import Plot 
class MC:
    """Utility class for importing mc information for use in analysis
    """
    
    def __init__(self):
        """Initialise the importer
        
        Args:
            data : array in format produced from processing
        """
        self.particle_count_return = None
        self.print_prefix = "[pymcutil] "
        
        print(f"{self.print_prefix}Initialised")
      
      
    
  def is_muon(data): # true if the MCParticle is a muon
    try:
        # Construct & return mask
        mask = (data['trkmcsim']['pdg'] == 13)
        self.logger.log(f"Returning mask for is_muon", "success")
        return mask
    except Exception as e:
        self.logger.log(f"Exception in is_muon: {e}", "error")
        return None

    
  def is_electron(data): # true if the MCParticle is a electron
    try:
        # Construct & return mask
        mask = (data['trkmcsim']['pdg'] == 11)
        self.logger.log(f"Returning mask for is_electron", "success")
        return mask
    except Exception as e:
        self.logger.log(f"Exception in is_electron: {e}", "error")
        return None

    
  def is_particle(data, code): # true if the MCParticle is the chosen particle 
    try:
        # Construct & return mask
        mask = (data['trkmcsim']['pdg'] == int(code))
        self.logger.log(f"Returning mask for is_particle", "success")
        return mask
    except Exception as e:
        self.logger.log(f"Exception in is_particle: {e}", "error")
        return None


  def start_process(data, process):
    try:
        # Construct & return mask
        mask = (data['trkmcsim', 'startCode'] == int(process))
        self.logger.log(f"Returning mask for start_process", "success")
        return mask
    except Exception as e:
        self.logger.log(f"Exception in start_process: {e}", "error")
        return None

  
  def is_CeMinusEndpoint(data):
    try:
          # Construct & return mask
          mask = (data['trkmcsim', 'startCode'] == 168)
          self.logger.log(f"Returning mask for is_CeMinusEndpoint", "success")
          return mask
      except Exception as e:
          self.logger.log(f"Exception in is_CeMinusEndpoint: {e}", "error")
          return None

  def is_CeMinusLeadingLog(data):
    try:
          # Construct & return mask
          mask = (data['trkmcsim', 'startCode'] == 170)
          self.logger.log(f"Returning mask for is_CeMinusEndpoint", "success")
          return mask
      except Exception as e:
          self.logger.log(f"Exception in is_CeMinusEndpoint: {e}", "error")
          return None

  def is_CePlusEndpoint(data):
    try:
          # Construct & return mask
          mask = (data['trkmcsim', 'startCode'] == 168)
          self.logger.log(f"Returning mask for is_CeMinusEndpoint", "success")
          return mask
      except Exception as e:
          self.logger.log(f"Exception in is_CeMinusEndpoint: {e}", "error")
          return None

  def is_CePlusLeadingLog(data):
    try:
          # Construct & return mask
          mask = (data['trkmcsim', 'startCode'] == 168)
          self.logger.log(f"Returning mask for is_CeMinusEndpoint", "success")
          return mask
      except Exception as e:
          self.logger.log(f"Exception in is_CeMinusEndpoint: {e}", "error")
          return None

  def is_target_DIO(data):
    try:
          # Construct & return mask
          vector = Vector()

          rhos = vector.get_rho(data['trkmcsim'],'pos')
          position = ak.firsts(rhos, axis=1) 

          # Use vectorized comparisons and selection for counting
          mask = (proc_codes == 166) & (position <= 75) 
          self.logger.log(f"Returning mask for is_target_DIO", "success")
          return mask
      except Exception as e:
          self.logger.log(f"Exception in is_target_DIO: {e}", "error")
          return None


  def end_processes(data,process):
    try:
        # Construct & return mask
        mask = (data['trkmcsim', 'stopCode'] == int(process))
        self.logger.log(f"Returning mask for end_process", "success")
        return mask
    except Exception as e:
        self.logger.log(f"Exception in end_process: {e}", "error")
        return None


//+ MCParticle Cuts - Generator ID
bool from_gen_id(MCParticle& particle, mu2e::GenId::enum_type genid) { // MCParticle has this generator ID
  if (particle.mcsim->gen == genid) { return true; }
  else { return false; }
}

bool is_cosmic(MCParticle& particle) { // MCParticle is a cosmic. This uses the function GenId::isCosmic() from Offline
  return mu2e::GenId(mu2e::GenId::enum_type(particle.mcsim->gen)).isCosmic();
}


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
    # Check for empty data
    if ak.num(data, axis=0) == 0:
        print("No events found in the data.")
        return []

    # Vectorized approach for efficiency using Awkward Array operations
    #  This is generally faster than looping through events individually for large datasets.

    # Get startCode for the first track in each event, handling empty lists
    # Use ak.firsts to safely get the first element or None if the list is empty
    proc_codes = ak.firsts(data['trkmcsim', 'startCode'], axis=1) 
    gen_codes = ak.firsts(data['trkmcsim', 'gen'], axis=1) 
    vector = Vector()

    rhos = vector.get_rho(data['trkmcsim'],'pos')
    #vec = vector.get_vector(branch=data['trkmc','trkmcsim'],vector_name='pos')
    #rhos = vec.rho
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
    cosmic_mask = ((gen_codes == 44) | (gen_codes == 38))  # Create boolean mask for cosmic events

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
        -1:  (len(particle_count[ak.any(cosmic_mask, axis=1)==True])),
        -2:  (len(particle_count[ak.any(other_mask, axis=1)==True])),
    }
      
    # Print the yields to terminal for cross-check
    print("===== MC truth yields for full momentum and time range=====")
    print("N_DIO: ", counts[166])
    print("N_IPA: ", counts[0])
    print("N_CEM: ", counts[168])
    print("N_CEP: ", counts[176])
    print("N_eRPC: ", counts[178])
    print("N_iRPC: ", counts[179])
    print("N_eRMC: ", counts[171])
    print("N_iRMC: ", counts[172])
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
    
    
    def plot_variable(val, val_label, lo, hi,  mc_count):
    """
    Plots distributions of the given parameter (val), splitting by process code

    Args:
        val : list of values e.g. rmax
        val_label : text formated value name e.g. "rmax"
        lo : plot range lower bound
        hi : plot range upper bound
        mc_counts : list of process codes

    Returns:
        plots saved as pdfs
    """

    cols = ['red','blue','green','black','cyan','magenta','grey','orange']
    labs = ['cosmic','irpc','erpc','irmc','ermc','ipa dio','dio', 'signal']
    
    val = ak.drop_none(val)
    val_signal = val.mask[mc_count[i] == 168]
    val_signal = np.array(ak.flatten(val_signal, axis=None))
    val_cosmics = val.mask[mc_count[i] == -1]
    val_cosmics = np.array(ak.flatten(val_cosmics, axis=None))
    val_dio = val.mask[mc_count[i] == 166]
    val_dio = np.array(ak.flatten(val_dio, axis=None))
    val_erpc = val.mask[mc_count[i] == 178]
    val_erpc = np.array(ak.flatten(val_erpc,axis=None))
    val_irpc = val.mask[mc_count[i] == 179]
    val_irpc = np.array(ak.flatten(val_irpc,axis=None))
    val_ermc = val.mask[mc_count[i] == 171]
    val_ermc = np.array(ak.flatten(val_ermc,axis=None))
    val_irmc = val.mask[mc_count[i] == 172]
    val_irmc = np.array(ak.flatten(val_irmc,axis=None))
    val_ipa = val.mask[mc_count[i] == 0]
    val_ipa = np.array(ak.flatten(val_ipa,axis=None))
    if log:
      plt.yscale('log')
    sets = [val_cosmics,val_irpc,val_erpc,val_irmc,val_ermc,val_ipa, val_dio, val_signal])
    n,bins,patch = plt.hist(sets,range=(lo,hi), color=cols, label=labs, bins=50, histtype="bar", alpha=0.5, stacked=True)
    plt.xlabel(str(val_label))
    plt.legend()
    plt.show()


    def is_track_particle(self, data):
      """
      TODO: function looks at tracks and finds particle with most contributions
      """
      nEvents = 0
      selector = Select()
      is_reco_electron = selector.is_electron(data["trk"])
      is_downstream = selector.is_downstream(data["trkfit"])

      data = data.mask[(is_downstream) & (is_reco_electron)]
      
      trk_front = selector.select_surface(data['trkfit'], surface_name="TT_Front")
      trkfit_ent = data['trkfit']["trksegs"].mask[(trk_front)]
      # make vector mag branch
      vector = Vector()
      mom_mag = vector.get_mag(trkfit_ent ,'mom')

      mom_mag = ak.nan_to_none(mom_mag)
      mom_mag = ak.drop_none(mom_mag)
      print("length mag",len(mom_mag))
      mom_flat = np.array(ak.flatten(mom_mag, axis=None))
      print("length mom",len(mom_flat))
      """
      plotter = Plot()
      plotter.plot_1D(
          mom_flat,               # Data to plot
          nbins=100,               # Number of bins
          xmin=70,                # Minimum x-axis value
          xmax=110,               # Maximum x-axis value
          xlabel="Reconstructed Mom [MeV/c]",
          ylabel="# occurances",
          out_path='event.png',  # Output file path
          stat_box=True,           # Show statistics box
          error_bars=True          # Show error bars
      )
      """
      n,bins,patch = plt.hist(mom_flat,range=(70,110))
      plt.xlabel("Reconstructed Mom [MeV/c]")
 
      plt.show()
      print("nevts",len(data))
      """
      trks = data['trk'].mask[(is_downstream) & (is_reco_electron)]
      trkmcs = data['trkmc'].mask[(is_downstream) & (is_reco_electron)]
      trkfits = data['trkfit'].mask[(is_downstream) & (is_reco_electron)]
      evts = data["evt"].mask[(is_downstream) & (is_reco_electron)]
      for i, event in enumerate(data):
        nEvents += 1
        if(nEvents < 20):
          print("======== event ========")
          #print("is trk part", mask)
          print("event : ",data["evt"]["event"][i],data["evt"]["subrun"][i],data["evt"]["run"][i])
          
          print("ntacks",len(trks[i]))
          for k, trk in enumerate(trks[i]):
            
            print("-------- track --------")
            for l, segs in enumerate(trkfits['trksegs'][i][k]):
              if(( is_downstream[i][k][l] == True and is_reco_electron[i][k] == True) ):
                  print("evnt",i,"trk",k,"seg",l)
                  for m, seg in enumerate(segs):
                    print("downstream", is_downstream[i][k][l])
                    print("is electron",is_reco_electron[i][k])
                   
                    print( "proc", trkmcs['trkmcsim', 'startCode'][i][k][l][m])
                    #print( "gen", trkmcs['trkmcsim', 'gen'][i][k][l][l])
                  
      """ 
      """
      try:
            # Construct & return mask
            mask = (data['trkmcsim', 'rank']== 0)
            self.logger.log(f"Returning mask for downstream track segments (p_z > 0)", "success")
            return mask
        except Exception as e:
            self.logger.log(f"Exception in is_track_particle()", "error")
            return None
      """
      
      
    def has_hits_on_track():
      """
      TODO asks if a given particle contributes some hits to track
      """
      #is rank == 0
      
    def is_track_parent():
      """
      TODO will tell if the parents daughter made the track, we dont care that its a secondary, we just care who its parent was
      """


