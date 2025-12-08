import numpy as np
import awkward as ak
import sys
import matplotlib.pyplot as plt
from pyutils.pyselect import Select
from pyutils.pyvector import Vector
from pyutils.pyplot import Plot
from .pylogger import Logger

class MC:
  """Utility class for importing mc information for use in analysis
  """
  
  def __init__(self, verbosity=1):
      """Initialise the importer
      
      Args:
          data : array in format produced from processing
      """
      self.particle_count_return = None
      self.print_prefix = "[pymcutil] "
      self.logger = Logger( 
            print_prefix = "[pymcutil]", 
            verbosity = verbosity
        )
      print(f"{self.print_prefix}Initialised")
      
      
  def is_track_particle(self, data):
    """
    function looks at tracks and finds particle with most contributions
    """
    try:
        rank = data['trkmcsim', 'rank']
        mask = (rank == 0)
        self.logger.log(f"Returning mask for is_track_particle", "success")
        return mask
    except Exception as e:
        self.logger.log(f"Exception in is_track_particle: {e}", "error")
        return None
    
  def has_hits_on_track(self, data):
    """
    TODO: requires MCRelationship
    """
    
    
  def is_track_parent(self, data):
    """
    TODO: requires MCRelationship
    """


  def is_muon(self, data):
    """ return mask for all events with true muon """
    try:
        rank = data['trkmcsim', 'rank']
        pdg = (data['trkmcsim', 'pdg'])
        mask = (pdg == 13) & (rank == 0)
        self.logger.log(f"Returning mask for is_muon", "success")
        return mask
    except Exception as e:
        self.logger.log(f"Exception in is_muon: {e}", "error")
        return None

    
  def is_electron(self, data):
    """ return mask for all events with true electron """
    try:
        rank = data['trkmcsim', 'rank']
        pdg = (data['trkmcsim', 'pdg'])
        mask = (pdg == 11) & (rank == 0)
        self.logger.log(f"Returning mask for is_electron", "success")
        return mask
    except Exception as e:
        self.logger.log(f"Exception in is_electron: {e}", "error")
        return None

  def is_positron(self, data):
    """ return mask for all events with true e+n """
    try:
        rank = data['trkmcsim', 'rank']
        pdg = (data['trkmcsim', 'pdg'])
        mask = (pdg == -11) & (rank == 0)
        self.logger.log(f"Returning mask for is_positron", "success")
        return mask
    except Exception as e:
        self.logger.log(f"Exception in is_positron: {e}", "error")
        return None
        
  def is_particle(self, data, code):
    """ returns true if the trkmcsim has pdg code for chosen particle """
    try:
        mask = (data['trkmcsim']['pdg'] == int(code))
        self.logger.log(f"Returning mask for is_particle", "success")
        return mask
    except Exception as e:
        self.logger.log(f"Exception in is_particle: {e}", "error")
        return None


  def start_process(self, data, process):
    """ returns true if the trkmcsim has process code for chosen start code """
    try:
        mask = (data['trkmcsim', 'startCode'] == int(process))
        self.logger.log(f"Returning mask for start_process", "success")
        return mask
    except Exception as e:
        self.logger.log(f"Exception in start_process: {e}", "error")
        return None

  def stop_process(self, data, process):
    """ returns true if the trkmcsim has process code for chosen stop code"""
    try:
        mask = (data['trkmcsim', 'stopCode'] == int(process))
        self.logger.log(f"Returning mask for stop_process", "success")
        return mask
    except Exception as e:
        self.logger.log(f"Exception in stop_process: {e}", "error")
        return None
        
  def is_CeMinusEndpoint(self, data):
    """ returns true if the trkmcsim has process code for ce- endpoint generator """
    try:
        mask = (data['trkmcsim', 'startCode'] == 167)
        self.logger.log(f"Returning mask for is_CeMinusEndpoint", "success")
        return mask
    except Exception as e:
        self.logger.log(f"Exception in is_CeMinusEndpoint: {e}", "error")
        return None

  def is_CeMinusLeadingLog(self, data):
    """ returns true if the trkmcsim has process code for ce- leading log generator """
    try:
        mask = (data['trkmcsim', 'startCode'] == 168)
        self.logger.log(f"Returning mask for is_CeMinusLeadingLog", "success")
        return mask
    except Exception as e:
        self.logger.log(f"Exception in is_CeMinusLeadingLog: {e}", "error")
        return None

  def is_CePlusEndpoint(self, data):
    """ returns true if the trkmcsim has process code for ce+ endpoint generator """
    try:
        mask = (data['trkmcsim', 'startCode'] == 169)
        self.logger.log(f"Returning mask for is_CePlusEndpoint", "success")
        return mask
    except Exception as e:
        self.logger.log(f"Exception in is_CePlusEndpoint: {e}", "error")
        return None

  def is_CePlusLeadingLog(self, data):
    """ returns true if the trkmcsim has process code for ce+ leading log generator """
    try:
        mask = (data['trkmcsim', 'startCode'] == 170)
        self.logger.log(f"Returning mask for is_CePlusLeadingLog", "success")
        return mask
    except Exception as e:
        self.logger.log(f"Exception in is_CePlusLeadingLog: {e}", "error")
        return None

  def is_target_DIO(self, data):
    """ returns true if the trkmcsim has a DIO process code and originates at radius consistant with target
     """
    try:
        # Need to separate from IPA DIO
        vector = Vector()
        rhos = vector.get_rho(data['trkmcsim'],'pos')
        position = ak.firsts(rhos, axis=1) 

        mask = (data['trkmcsim', 'startCode'] == 166) & (position <= 75) 
        self.logger.log(f"Returning mask for is_target_DIO", "success")
        return mask
    except Exception as e:
        self.logger.log(f"Exception in is_target_DIO: {e}", "error")
        return None


  def is_cosmic(self, data):
    """ returns true if the trkmcsim is a cosmic generated particle """
    try:
        mask = (data['trkmcsim', 'gen'] == 44) | (data['trkmcsim', 'gen'] == 38)
        self.logger.log(f"Returning mask for is_cosmic", "success")
        return mask
    except Exception as e:
        self.logger.log(f"Exception in is_cosmic: {e}", "error")
        return None


  def count_particle_types(self, data):
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

    if ak.num(data, axis=0) == 0:
        print("No events found in the data.")
        return []


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
    
    primary_mask = particle_count_return != -2

    # Apply the mask to the flattened array to select desired elements
    particle_count_return = particle_count_return[primary_mask]
    particle_count_return = [[sublist[0]] for sublist in particle_count_return]
    particle_count_return = ak.flatten(particle_count_return, axis=None)
    print("returned particle count length",len(particle_count_return))
    
    return particle_count_return
