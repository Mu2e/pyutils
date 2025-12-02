#! /usr/bin/env python
import uproot
import awkward as ak
import numpy as np
from .pylogger import Logger

class Select:
    """
    Class for standard selection cuts with EventNtuple data in Awkward format
    
    """
    def __init__(self, verbosity=1):
        """Initialise the selector
        
            Args: 
                verbosity (int, optional): Print detail level (0: minimal, 1: medium, 2: maximum). Defaults to 1. 
        """
        # Start logger
        self.logger = Logger( 
            print_prefix = "[pyselect]", 
            verbosity = verbosity
        )
        # PDG ID dict
        self.particles = { 
            "e-" : 11,
            "e+" : -11,
            "mu-" : 13,
            "mu+" : -13
        }
        
        # SIDs see: https://github.com/Mu2e/Offline/blob/main/DataProducts/inc/SurfaceId.hh for definitions
        self.surface_id_map = { # Storing the mapping in a dictionary
            "TT_Front" :0,
            "TT_Mid" : 1,
            "TT_Back" : 2,
            "TT_Inner" : 3,
            "TT_Outer" : 4,
            "DS_Inner" : 82,
            "DS_Outer" : 83,
            "ST_Front": 100,
            "ST_Back" : 101,
            "ST_Inner" : 102,
            "ST_Outer" : 103,
            "ST_Foils" : 104,
            "ST_Wires" : 105,
            "IPA" : 90,
            "IPA Front" : 91,
            "IPA Back" : 92,
            "OPA" : 95,
            "TSDA" : 96,
            "TCRV" : 200        
        }

    def get_surface_name(self, sid):
        """Convert an integer surface ID (sid) to a meaningful name.
        """       
        return self.surface_id_map.get(sid)
        
    def is_electron(self, data):
        """ Return boolean array for electron tracks which can be used as a mask 

            Args:
                data (awkward.Array): Input array containing the "trk" branch
        """
        try:
            # Construct & return mask
            mask = (data["trk.pdg"] == self.particles["e-"])
            self.logger.log(f"Returning mask for e- tracks", "success")
            return mask
        except Exception as e:
            self.logger.log(f"Exception in is_electron(): {e}", "error")
            return None
            
    def is_positron(self, data):
        """ Return boolean array for positron tracks which can be used as a mask 

            Args:
                data (awkward.Array): Input array containing the "trk" branch
        """
        try:
            # Construct & return mask
            mask = (data["trk.pdg"] == self.particles["e+"])
            self.logger.log(f"Returning mask for e+ tracks", "success")
            return mask
        except Exception as e:
            self.logger.log(f"Exception in is_positron(): {e}", "error")
            return None
            
    def is_mu_minus(self, data):
        """ Return boolean array for negative muon tracks which can be used as a mask 

            Args:
                data (awkward.Array): Input array containing the "trk" branch
        """
        try:
            # Construct & return mask
            mask = (data["trk.pdg"] == self.particles["mu-"]) 
            self.logger.log(f"Returning mask for mu- tracks", "success")
            return mask
        except Exception as e:
            self.logger.log(f"Exception in is_mu_minus(): {e}", "error")
            return None

    def is_mu_plus(self, data):
        """ Return boolean array for positive muon tracks which can be used as a mask 

            Args:
                data (awkward.Array): Input array containing the "trk" branch
        """
        try:
            # Construct & return mask
            mask = (data["trk.pdg"] == self.particles["mu+"])
            self.logger.log(f"Returning mask for mu+ tracks", "success")
            return mask 
        except Exception as e:
            self.logger.log(f"Exception in is_mu_plus(): {e}", "error")
            return None

    # More general function for particle selection
    def is_particle(self, data, particle):
        """ Return boolean array for tracks of a specific particle type which can be used as a mask 

            Args:
                data (awkward.Array): Input array containing the "trk" branch
                particle (string): particle type, 'e-', 'e+', 'mu-', or 'mu+'
        """
        # "trk" is not the only branch that uses pdgIDs, crvcoincsmc is another, how to handle that? 
        try:
            # Construct & return mask
            mask = (data["trk.pdg"] == self.particles[particle])
            self.logger.log(f"Returning mask for {particle} tracks", "success")
            return mask
        except Exception as e:
            self.logger.log(f"Exception in is_particle(): {e}", "error")
            return None
            
    def is_downstream(self, data, branch_name="trksegs"):
        """ Return boolean array for upstream track segments

            Args:
                data (awkward.Array): Input array containing the segments branch
                branch_name (str, optional): Name of the segments branch for backwards compatibility. Defaults to 'trksegs'
        """
        try:
            # Construct & return mask
            mask = (data[branch_name]["mom"]["fCoordinates"]["fZ"] > 0)
            self.logger.log(f"Returning mask for downstream track segments (p_z > 0)", "success")
            return mask
        except Exception as e:
            self.logger.log(f"Exception in is_downstream(): {e}", "error")
            return None

    def is_upstream(self, data, branch_name="trksegs"):
        """ Return boolean array for downstream track segments 

            Args:
                data (awkward.Array): Input array containing the segments branch
                branch_name (str, optional): Name of the segments branch for backwards compatibility. Defaults to 'trksegs'
        """
        try:
            # Construct & return mask
             mask = (data["trksegs"]["mom"]["fCoordinates"]["fZ"] < 0)
             self.logger.log(f"Returning mask for upstream track segments (p_z < 0)", "success")
             return mask
        except Exception as e:
            self.logger.log(f"Exception in is_upstream(): {e}", "error")
            return None

    def select_surface(self, data, surface_name="TT_Front", sindex=0, branch_name="trksegs"):
        """ Return boolean array for track segments intersecting a specific surface 
        
        Args:
            data (awkward.Array): Input array containing segments branch
            surface_name (str) : official name of the intersected surface 
            sindex (int, optional): Index to the intersected surface (for multi-surface elements). Defaults to 0. 
            branch_name (str, optional): Name of the segments branch for backwards compatibility. Defaults to 'trksegs'
        """
        # convert the string to the int underneath
        sid = self.get_surface_name(surface_name)
        
        try:
            # Construct & return mask
            mask = (data[branch_name]['sid']==sid)# & (data[branch_name]['sindex']==sindex)
            self.logger.log(f"Returning mask for {branch_name} with sid = {sid}", "success") #and sindex = {sindex}"
            return mask
        except Exception as e:
            self.logger.log(f"Exception in select_surface(): {e}", "error")
            return None

    def has_ST(self, data):
      """returns mask True if the event has at least 1 ST viable extrapolation
      """
      try:
        trk_st  = self.select_surface(data, surface_name="ST_Foils")
        nst_array = ak.sum(trk_st, axis=-1)
        mask = (nst_array > 0)
        return mask
      except Exception as e:
        self.logger.log(f"Exception in has_ST(): {e}", "error")
        return None
            
    def has_OPA(self, data):
      """returns mask True if the event has at no OPA viable extrapolation
      """
      try:
        trk_opa  = self.select_surface(data, surface_name="OPA")
        nopa_array = ak.sum(trk_opa, axis=-1)
        mask = (nopa_array == 0)
        return mask
      except Exception as e:
        self.logger.log(f"Exception in has_OPA(): {e}", "error")
        return None
  
    def is_reflected(self, data, branch_name="trksegs"):
        """ Return boolean array for reflected tracks  
        
        Reflected tracks have both upstream and downstream segments at the tracker entrance 
        
        Args:
            data (awkward.Array): Input array containing segments branch
            branch_name (str, optional): Name of the segments branch for backwards compatibility. Defaults to 'trksegs'
        """
        try:
            # Construct track segment conditions
            trkent = self.select_surface(data, surface_name="TT_Front", branch_name=branch_name)
            upstream = self.is_upstream(data, branch_name=branch_name)
            downstream = self.is_downstream(data, branch_name=branch_name)
            # Construct condition for reflected tracks 
            # Does the track have any segments with p_z > 0 at the tracker entrance 
            # AND at any segments with p_z < 0 at the tracker entrance?
            reflected = (ak.any((upstream & trkent), axis=-1) & ak.any((downstream & trkent), axis=-1))
            self.logger.log(f"Returning mask for reflected tracks", "success")
            return reflected
        except Exception as e:
            self.logger.log(f"Exception in is_reflected(): {e}", "error")
            return None
            
    def select_trkqual(self, data, quality):
        """ Return boolean array for tracks above a specified quality   

        Args: 
            data (awkward.Array): Input array containing the trkqual.result branch
            quality (float): The numerical output of the MVA

        """
        #FIXME - need to ensure this is using the latest
        try:
            # Construct & return mask
            mask = (data["trkqual.result"] > quality)
            self.logger.log(f"Returning mask for trkqual > {quality}", "success")
            return mask
        except Exception as e:
            self.logger.log(f"Exception in select_trkqual(): {e}", "error")
            return None
     
    def select_trkpid(self, data, value):
      """ TODO: placeholder """
      return None
             
    def has_n_hits(self, data, n_hits):
        """ Return boolean array for tracks with hits above a specified value 

        Hits in this context is nactive planes

        Args: 
            data (awkward.Array): Input array containing the trk.nactive branch
            n_hits (int): The minimum number of track hits (nactive)

        """
        try:
            # Construct & return mask
            # Why nactive and not nhits?
            mask = (data["trk.nactive"] >= n_hits)
            self.logger.log(f"Returning mask for nactive > {n_hits}", "success")
            return mask
        except Exception as e:
            self.logger.log(f"Exception in has_n_hits(): {e}", "error")
            return None
    
    def hasTrkCrvCoincs(self, data, dt_threshold=150):
        """ simple version of the crv coincidence checker """

        at_trk_front = self.select_surface(data['trkfit'], surface_name="TT_Front")
        
        # Get track and coincidence times
        trk_times = data["trksegs"]["time"][at_trk_front]  # events × tracks × segments
        coinc_times = data["crvcoincs.time"]                  # events × coincidences
        
        # Broadcast CRV times to match track structure, so that we can compare element-wise
        coinc_broadcast = coinc_times[:, None, None, :]  # Add dimensions for tracks and segments
        trk_broadcast = trk_times[:, :, :, None]         # Add dimension for coincidences

        # Calculate time differences
        dt = abs(trk_broadcast - coinc_broadcast)
        
        # Check if within threshold
        within_threshold = dt < dt_threshold

        # Check if any coincidence
        any_coinc = ak.any(within_threshold, axis=3)
        
        # Then reduce over trks (axis=2) 
        veto = ak.any(any_coinc, axis=2)

        return veto
