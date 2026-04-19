#! /usr/bin/env python
"""
Calorimeter Analysis and Visualization Utilities

Classes:
    CaloAnalysis: Data matching and pipeline tracing utilities
    CaloVisualization: Crystal map drawing and event display utilities
"""

import numpy as np
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os

from .pylogger import Logger
from .pycrystal import PipeGeometry


class CaloAnalysis:
    """
    Utility class for calorimeter data matching and pipeline tracing.
    
    Provides methods to create mappings between different stages of the reconstruction:
    - calodigis (raw calorimeter digitizations)
    - calorecdigis (reconstructed digis with energy)
    - calohits (crystal hits from recdigis)
    - caloclusters (clusters of hits)
    - calodigismc (MC truth for digis)
    """
    
    def __init__(self, verbosity=1):
        """
        Initialize CaloAnalysis.
        
        Args:
            verbosity (int): Output detail level (0: errors only, 1: info & warnings, 2: max)
        """
        self.verbosity = verbosity
        self.logger = Logger(
            print_prefix="[pycalo_ana]",
            verbosity=verbosity
        )
        self.logger.log("Initialised CaloAnalysis", "info")
    
    def map_digi_to_reco(self, data):
        """
        Create mapping from calorimeter digis to reconstructed digis.
        
        For each event, returns caloDigiIdx values that link each reco digi back
        to the original calo digi(s) it was reconstructed from.
        
        Args:
            data (ak.Array): Data array with 'calorecodigis.caloDigiIdx_' branch
        
        Returns:
            list: For each event, list of caloDigiIdx values (one per reco digi)
                 Each entry can be a single index or list of indices (multi-source reco)
        
        Example:
            digi_to_reco = calo_ana.map_digi_to_reco(data)
            # For event 0, reco digi i was made from digi(s) at: digi_to_reco[0][i]
            for evt_idx in range(len(digi_to_reco)):
                for reco_idx, calo_idx in enumerate(digi_to_reco[evt_idx]):
                    print(f"Reco {reco_idx} made from calo digi {calo_idx}")
        """
        try:
            reco_caloDigiIdx = data["calorecodigis.caloDigiIdx_"]
            
            mapping = []
            for evt_idx in range(len(reco_caloDigiIdx)):
                idx_list = ak.to_list(reco_caloDigiIdx[evt_idx])
                
                # Handle nested lists (in case of multi-source recdigis)
                if isinstance(idx_list, list) and len(idx_list) > 0 and isinstance(idx_list[0], list):
                    idx_list = [val for sublist in idx_list for val in (sublist if isinstance(sublist, list) else [sublist])]
                
                mapping.append(idx_list)
            
            self.logger.log(f"Created digi->reco mapping for {len(mapping)} events", "debug")
            return mapping
        except Exception as e:
            self.logger.log(f"Error mapping digi to reco: {e}", "error")
            return None
    
    def map_reco_to_digi(self, data):
        """
        Create reverse mapping from reco digis back to calo digis with energy.
        
        Returns both the mapping AND the energy for each reco digi.
        
        Args:
            data (ak.Array): Data array with reco digi branches
        
        Returns:
            tuple: (mapping, energies) where:
                - mapping: list of caloDigiIdx per event (maps reco -> calo)
                - energies: list of eDep values per event (energy of each reco digi)
        
        Example:
            reco_to_digi, reco_energy = calo_ana.map_reco_to_digi(data)
            for evt_idx in range(len(reco_to_digi)):
                for reco_idx, calo_idx in enumerate(reco_to_digi[evt_idx]):
                    print(f"Reco digi {calo_idx} has energy {reco_energy[evt_idx][reco_idx]} MeV")
        """
        try:
            reco_caloDigiIdx = data["calorecodigis.caloDigiIdx_"]
            reco_eDep = data["calorecodigis.eDep_"]
            
            mapping = []
            energies = []
            
            for evt_idx in range(len(reco_caloDigiIdx)):
                idx_list = ak.to_list(reco_caloDigiIdx[evt_idx])
                edep_list = ak.to_list(reco_eDep[evt_idx])
                
                # Handle nested lists
                if isinstance(idx_list, list) and len(idx_list) > 0 and isinstance(idx_list[0], list):
                    idx_list = [val for sublist in idx_list for val in (sublist if isinstance(sublist, list) else [sublist])]
                if isinstance(edep_list, list) and len(edep_list) > 0 and isinstance(edep_list[0], list):
                    edep_list = [val for sublist in edep_list for val in (sublist if isinstance(sublist, list) else [sublist])]
                
                mapping.append(idx_list)
                energies.append(edep_list)
            
            self.logger.log(f"Created reco->digi+energy mapping for {len(mapping)} events", "debug")
            return mapping, energies
        except Exception as e:
            self.logger.log(f"Error mapping reco to digi: {e}", "error")
            return None, None
    
    def get_matching_reco_for_digi(self, data, digi_idx, evt_idx):
        """
        Get reconstructed digi(s) that were made from a specific calo digi.
        
        Convenience function: finds all reco digis with caloDigiIdx pointing to given digi.
        
        Args:
            data (ak.Array): Data array
            digi_idx (int): Index of calo digi to search for
            evt_idx (int): Event index
        
        Returns:
            list: Indices of reco digis that contain this calo digi in their caloDigiIdx
        
        Example:
            reco_indices = calo_ana.get_matching_reco_for_digi(data, digi_idx=2, evt_idx=0)
            # Extract energy of matching reco digis
            reco_energy = data["calorecodigis.eDep_"]
            for reco_idx in reco_indices:
                print(f"Reco digi {reco_idx} energy: {reco_energy[evt_idx][reco_idx]}")
        """
        try:
            reco_caloDigiIdx = data["calorecodigis.caloDigiIdx_"]
            
            idx_list = ak.to_list(reco_caloDigiIdx[evt_idx])
            if isinstance(idx_list, list) and len(idx_list) > 0 and isinstance(idx_list[0], list):
                idx_list = [val for sublist in idx_list for val in (sublist if isinstance(sublist, list) else [sublist])]
            
            matching_reco = []
            for reco_idx, calo_idx_data in enumerate(idx_list):
                # Handle both single values and lists
                if isinstance(calo_idx_data, (list, tuple)):
                    if digi_idx in calo_idx_data:
                        matching_reco.append(reco_idx)
                else:
                    if int(calo_idx_data) == digi_idx:
                        matching_reco.append(reco_idx)
            
            return matching_reco
        except Exception as e:
            self.logger.log(f"Error finding matching reco: {e}", "error")
            return []
    
    def compare_reco_vs_mc_energy(self, reco, mc, reco_caloDigiIdx, calo_sipmid):
        """
        Compare reconstructed vs MC energy on a per-crystal basis within each event.
        
        For each event, within each crystal (SiPMID), aggregates total reco and MC energy.
        Results are organized by event - no global flattening across events/crystals.
        
        Args:
            reco (ak.Array or list): Reconstructed digi energies (calorecodigis.eDep_)
            mc (ak.Array or list): MC truth energies (calodigismc.eDep)
            reco_caloDigiIdx (ak.Array or list): Indices mapping reco -> calo digis (calorecodigis.caloDigiIdx_)
            calo_sipmid (ak.Array or list): SiPM IDs identifying crystals (calodigis.SiPMID_)
        
        Returns:
            dict: Per-event, per-crystal aggregated energies:
                - by_event: dict {evt_idx: {sipmid: {'reco': total_reco, 'mc': total_mc, 'residual': residual}}}
                - n_crystals_per_evt: list of hit crystal counts per event
                - n_events: total number of events
        
        Example:
            # Extract required arrays
            reco_edeps = calo_data["calorecodigis.eDep_"]
            mc_edeps = calo_data["calodigismc.eDep"]
            reco_idx = calo_data["calorecodigis.caloDigiIdx_"]
            sipmids = calo_data["calodigis.SiPMID_"]
            
            # Compare per crystal within each event
            comparison = calo_ana.compare_reco_vs_mc_energy(reco_edeps, mc_edeps, reco_idx, sipmids)
            
            # Access results per event
            for evt_idx, crystals_dict in comparison['by_event'].items():
                for sipmid, energies in crystals_dict.items():
                    print(f"Event {evt_idx}, Crystal {sipmid}: " +
                          f"reco={energies['reco']:.2f}, mc={energies['mc']:.2f}, " +
                          f"residual={energies['residual']:.2f}")
        """
        try:
            if reco is None or mc is None or reco_caloDigiIdx is None or calo_sipmid is None:
                self.logger.log("Missing required energy or mapping arrays", "warn")
                return None
            
            by_event = {}
            n_crystals_per_evt = []
            
            n_events = len(reco)
            
            for evt_idx in range(n_events):
                try:
                    # Extract event data
                    reco_evt = ak.to_list(reco[evt_idx])
                    mc_evt = ak.to_list(mc[evt_idx])
                    reco_idx_evt = ak.to_list(reco_caloDigiIdx[evt_idx])
                    sipmid_evt = ak.to_list(calo_sipmid[evt_idx])
                    
                    # Handle nested lists
                    if reco_evt and isinstance(reco_evt[0], list):
                        reco_evt = [e for sublist in reco_evt for e in (sublist if isinstance(sublist, list) else [sublist])]
                    if mc_evt and isinstance(mc_evt[0], list):
                        mc_evt = [e for sublist in mc_evt for e in (sublist if isinstance(sublist, list) else [sublist])]
                    if reco_idx_evt and isinstance(reco_idx_evt[0], list):
                        reco_idx_evt = [e for sublist in reco_idx_evt for e in (sublist if isinstance(sublist, list) else [sublist])]
                    if sipmid_evt and isinstance(sipmid_evt[0], list):
                        sipmid_evt = [e for sublist in sipmid_evt for e in (sublist if isinstance(sublist, list) else [sublist])]
                    
                    # Group energies by crystal within this event
                    crystal_energies = {}  # sipmid -> {'reco': sum, 'mc': sum}
                    
                    # First, map calodigis to their SiPM IDs
                    calo_sipmid_map = {}  # calo_idx -> sipmid
                    for calo_idx, sipmid in enumerate(sipmid_evt):
                        calo_sipmid_map[calo_idx] = int(sipmid)
                    
                    # Aggregate energies by crystal
                    for reco_idx, calo_idx_ref in enumerate(reco_idx_evt):
                        try:
                            calo_idx = int(calo_idx_ref)
                            if calo_idx in calo_sipmid_map:
                                sipmid = calo_sipmid_map[calo_idx]
                                reco_energy = float(reco_evt[reco_idx]) if reco_idx < len(reco_evt) else 0
                                mc_energy = float(mc_evt[calo_idx]) if calo_idx < len(mc_evt) else 0
                                
                                if sipmid not in crystal_energies:
                                    crystal_energies[sipmid] = {'reco': 0, 'mc': 0}
                                
                                crystal_energies[sipmid]['reco'] += reco_energy
                                crystal_energies[sipmid]['mc'] += mc_energy
                        except (ValueError, TypeError, IndexError):
                            pass
                    
                    # Compute residuals for this event's crystals
                    evt_crystals = {}
                    for sipmid, energies in crystal_energies.items():
                        residual = energies['mc'] - energies['reco']  # MC - reco
                        evt_crystals[sipmid] = {
                            'reco': energies['reco'],
                            'mc': energies['mc'],
                            'residual': residual
                        }
                    
                    by_event[evt_idx] = evt_crystals
                    n_crystals_per_evt.append(len(evt_crystals))
                    
                except Exception as e:
                    self.logger.log(f"Error processing event {evt_idx}: {e}", "warn")
                    by_event[evt_idx] = {}
                    n_crystals_per_evt.append(0)
            
            self.logger.log(f"Compared reco vs MC energy for {n_events} events", "debug")
            
            return {
                'by_event': by_event,
                'n_crystals_per_evt': n_crystals_per_evt,
                'n_events': n_events
            }
        
        except Exception as e:
            self.logger.log(f"Error in compare_reco_vs_mc_energy: {e}", "error")
            import traceback
            self.logger.log(traceback.format_exc(), "debug")
            return None


class CaloVisualization:
    """
    Utility class for calorimeter visualization.
    Handles crystal map drawing, 2D projections, and particle overlays.
    """
    
    def __init__(self, verbosity=1):
        """
        Initialize CaloVisualization.
        
        Args:
            verbosity (int): Output detail level (0: errors only, 1: info & warnings, 2: max)
        """
        self.verbosity = verbosity
        self.logger = Logger(
            print_prefix="[pycalo_vis]",
            verbosity=verbosity
        )
        self.logger.log("Initialised CaloVisualization", "info")
    
    def load_crystal_map(self):
        """
        Load crystal map from caloDMAP_nominal.dat file.
        
        Returns:
            CrystalMap object or None if loading fails
        
        Example:
            crysmap = calo_vis.load_crystal_map()
            if crysmap:
                print(f"Loaded crystal map with disks")
        """
        try:
            import sys
            import importlib.util
            
            # Get path to pycrystal.py in this module directory
            module_dir = os.path.dirname(__file__)
            pycrystal_path = os.path.join(module_dir, 'pycrystal.py')
            
            if not os.path.exists(pycrystal_path):
                self.logger.log(f"pycrystal.py not found at {pycrystal_path}", "error")
                return None
            
            # Load pycrystal module with proper package context
            try:
                spec = importlib.util.spec_from_file_location(
                    "pyutils.pycrystal", 
                    pycrystal_path,
                    submodule_search_locations=[module_dir]
                )
                pycrystal_module = importlib.util.module_from_spec(spec)
                
                # Add to sys.modules so relative imports work
                sys.modules["pyutils.pycrystal"] = pycrystal_module
                sys.modules["pyutils"] = type(sys)("module")  # Create minimal pyutils module
                
                # Now execute the module (this will resolve relative imports)
                spec.loader.exec_module(pycrystal_module)
                
                CrystalMap = pycrystal_module.CrystalMap
                self.logger.log(f"Successfully imported CrystalMap using importlib", "debug")
            
            except Exception as e:
                self.logger.log(f"Failed to import CrystalMap: {e}", "error")
                import traceback
                self.logger.log(f"Traceback: {traceback.format_exc()}", "debug")
                return None
            
            # Use local caloDMAP_nominal.dat file
            map_file = os.path.join(module_dir, 'caloDMAP_nominal.dat')
            
            if not os.path.exists(map_file):
                self.logger.log(f"caloDMAP_nominal.dat not found at {map_file}", "error")
                self.logger.log(f"Please ensure caloDMAP_nominal.dat exists in {module_dir}", "error")
                return None
            
            self.logger.log(f"Loading CrystalMap from {map_file}", "info")
            
            try:
                crysmap = CrystalMap(map_file, verbosity=0)
                self.logger.log(f"Loaded CrystalMap successfully", "success")
                return crysmap
            except Exception as e:
                self.logger.log(f"CrystalMap initialization failed: {e}", "error")
                import traceback
                self.logger.log(f"Traceback: {traceback.format_exc()}", "debug")
                return None
        
        except Exception as e:
            self.logger.log(f"Could not load CrystalMap: {e}", "error")
            import traceback
            self.logger.log(f"Traceback: {traceback.format_exc()}", "debug")
            return None
    
    def plot_crystal_grid(self, ax, crysmap, disk=0, show_labels=False, 
                          alpha=0.7, crystal_color='lightblue', edge_color='black'):
        """
        Draw crystal grid on a matplotlib axes.
        
        Draws all crystals for a specified disk as square patches.
        
        Args:
            ax (plt.Axes): Matplotlib axes to draw on
            crysmap (CrystalMap): Crystal map object
            disk (int): Disk number to draw (0 or 1)
            show_labels (bool): Whether to annotate crystal IDs (default: False)
            alpha (float): Transparency for crystal patches (default: 0.7)
            crystal_color (str): Fill color for crystals (default: 'lightblue')
            edge_color (str): Edge color for crystal squares (default: 'black')
        
        Returns:
            int: Number of crystals drawn
        
        Example:
            fig, ax = plt.subplots()
            calo_vis = CaloVisualization()
            crysmap = calo_vis.load_crystal_map()
            n_crys = calo_vis.plot_crystal_grid(ax, crysmap, disk=0, show_labels=True)
            print(f"Drew {n_crys} crystals")
        """
        try:
            if crysmap is None:
                self.logger.log("CrystalMap is None, cannot draw grid", "warning")
                return 0
            
            if disk not in crysmap.crystals_by_disk:
                self.logger.log(f"Disk {disk} not found in crystal map", "warning")
                return 0
            
            disk_data = crysmap.crystals_by_disk[disk]
            n_drawn = 0
            
            for _, row in disk_data.iterrows():
                xcry = row['xcry']
                ycry = row['ycry']
                cryid = int(row['cryID'])
                
                # Draw crystal as 34x34 mm square centered at (xcry, ycry)
                square = patches.Rectangle(
                    (xcry - 17, ycry - 17),
                    34, 34,
                    linewidth=1,
                    edgecolor=edge_color,
                    facecolor=crystal_color,
                    alpha=alpha,
                    zorder=1
                )
                ax.add_patch(square)
                
                # Optionally add crystal ID label
                if show_labels:
                    ax.text(xcry, ycry, str(cryid), fontsize=5, ha='center', va='center',
                           color='black', alpha=1.0, zorder=2)
                
                n_drawn += 1
            
            self.logger.log(f"Drew {n_drawn} crystals for disk {disk}", "info")
            return n_drawn
        
        except Exception as e:
            self.logger.log(f"Error plotting crystal grid: {e}", "error")
            return 0
    
    def plot_pipes(self, ax, crysmap, disk=0, pipe_color='red', pipe_alpha=0.3, label=True):
        """
        Draw calibration pipes on a matplotlib axes.
        
        Args:
            ax (plt.Axes): Matplotlib axes to draw on
            crysmap (CrystalMap): Crystal map object
            disk (int): Disk number for pipes (default: 0)
            pipe_color (str): Color for pipe lines (default: 'red')
            pipe_alpha (float): Transparency for pipes (default: 0.3)
            label (bool): Whether to label pipe IDs (default: True)
        
        Returns:
            int: Number of pipes drawn
        
        Example:
            n_pipes = calo_vis.plot_pipes(ax, crysmap, disk=0, label=True)
            print(f"Drew {n_pipes} pipes")
        """
        try:
            if crysmap is None:
                self.logger.log("CrystalMap is None, cannot draw pipes", "warning")
                return 0
            
            # Use internal CrystalMap method if available
            if hasattr(crysmap, '_draw_pipes'):
                crysmap._draw_pipes(ax, pipe_color=pipe_color, pipe_alpha=pipe_alpha, 
                                   label_pipes=label)
                self.logger.log("Drew pipes using CrystalMap._draw_pipes", "info")
                return 1  # Return 1 to indicate success
            else:
                self.logger.log("CrystalMap does not have _draw_pipes method", "warning")
                return 0
        
        except Exception as e:
            self.logger.log(f"Error plotting pipes: {e}", "error")
            return 0
    
    def plot_2d_energy_heatmap(self, ax, crysmap, digi_positions, digi_energies, 
                               disk=0, title=None, cmap='YlOrRd', show_labels=False):
        """
        Plot 2D energy heatmap on crystal grid.
        
        Colors crystals based on energy deposited, overlaying on crystal map.
        
        Args:
            ax (plt.Axes): Matplotlib axes to draw on
            crysmap (CrystalMap): Crystal map object
            digi_positions (list): List of (x, y) positions for digis
            digi_energies (np.ndarray): Energy values for each digi
            disk (int): Disk number to display (default: 0)
            title (str, optional): Plot title
            cmap (str): Colormap name (default: 'YlOrRd')
            show_labels (bool): Show crystal IDs (default: False)
        
        Returns:
            tuple: (n_crystals_drawn, colorbar_mappable)
        
        Example:
            sm = calo_vis.plot_2d_energy_heatmap(ax, crysmap, pos_list, energy_array)
        """
        try:
            if crysmap is None:
                self.logger.log("CrystalMap is None, cannot draw heatmap", "warning")
                return 0, None
            
            # Draw background grid first
            self.plot_crystal_grid(ax, crysmap, disk=disk, show_labels=show_labels, 
                                  alpha=0.5, crystal_color='lightgray')
            
            # Set up color map
            digi_energies = np.array(digi_energies, dtype=float)
            if len(digi_energies) > 0 and np.max(digi_energies) > 0:
                norm = Normalize(vmin=0, vmax=np.max(digi_energies))
            else:
                norm = Normalize(vmin=0, vmax=1)
            
            cmap_obj = plt.cm.get_cmap(cmap)
            sm = ScalarMappable(norm=norm, cmap=cmap_obj)
            
            # Draw energy-colored crystals
            n_drawn = 0
            for i, (x, y) in enumerate(digi_positions):
                if i < len(digi_energies):
                    energy = digi_energies[i]
                    color = cmap_obj(norm(energy))
                    
                    # Draw highlighted crystal
                    highlight = patches.Rectangle(
                        (x - 17, y - 17),
                        34, 34,
                        linewidth=2,
                        edgecolor='black',
                        facecolor=color,
                        alpha=0.9,
                        zorder=3
                    )
                    ax.add_patch(highlight)
                    n_drawn += 1
            
            # Add colorbar
            cbar = plt.colorbar(sm, ax=ax, label='Energy (MeV)', shrink=0.8)
            
            if title:
                ax.set_title(title, fontsize=12)
            
            self.logger.log(f"Drew {n_drawn} energy-colored crystals", "info")
            return n_drawn, sm
        
        except Exception as e:
            self.logger.log(f"Error plotting energy heatmap: {e}", "error")
            return 0, None
    
    def highlight_hit_crystals(self, ax, crysmap, crystal_ids, disk=0, 
                               color='yellow', edge_color='orange', linewidth=2,
                               show_labels=True, fontsize=8):
        """
        Highlight specific crystals that were hit and optionally show their IDs.
        
        Draws bright patches over specified crystals with optional crystal ID labels.
        
        Args:
            ax (plt.Axes): Matplotlib axes to draw on
            crysmap (CrystalMap): Crystal map object
            crystal_ids (list): List of crystal IDs to highlight
            disk (int): Disk number (default: 0)
            color (str): Highlight color (default: 'yellow')
            edge_color (str): Edge color (default: 'orange')
            linewidth (int): Edge line width (default: 2)
            show_labels (bool): Show crystal ID numbers on highlighted crystals (default: True)
            fontsize (int): Font size for crystal ID labels (default: 8)
        
        Returns:
            int: Number of crystals highlighted
        
        Example:
            n_high = calo_vis.highlight_hit_crystals(ax, crysmap, [100, 101, 102], show_labels=True)
        """
        try:
            if crysmap is None:
                self.logger.log("CrystalMap is None", "warning")
                return 0
            
            if disk not in crysmap.crystals_by_disk:
                self.logger.log(f"Disk {disk} not found", "warning")
                return 0
            
            disk_data = crysmap.crystals_by_disk[disk]
            crystal_ids = set([int(cid) for cid in crystal_ids])
            
            self.logger.log(f"highlight_hit_crystals: Looking for {len(crystal_ids)} crystal IDs in disk {disk} (map has {len(disk_data)} total crystals)", "debug")
            
            n_highlighted = 0
            
            for _, row in disk_data.iterrows():
                cryid = int(row['cryID'])
                
                if cryid in crystal_ids:
                    xcry = row['xcry']
                    ycry = row['ycry']
                    
                    # Draw highlight patch
                    highlight = patches.Rectangle(
                        (xcry - 17, ycry - 17),
                        34, 34,
                        linewidth=linewidth,
                        edgecolor=edge_color,
                        facecolor=color,
                        alpha=0.8,
                        zorder=50
                    )
                    ax.add_patch(highlight)
                    
                    # Add crystal ID label if requested
                    if show_labels:
                        ax.text(xcry, ycry, str(cryid),
                               fontsize=fontsize,
                               ha='center', va='center',
                               fontweight='bold',
                               color='black',
                               zorder=51)
                    
                    n_highlighted += 1
            
            self.logger.log(f"highlight_hit_crystals: Successfully drew {n_highlighted}/{len(crystal_ids)} requested crystals", "info")
            return n_highlighted
        
        except Exception as e:
            self.logger.log(f"Error highlighting crystals: {e}", "error")
            return 0
    
    def annotate_crystal_info(self, ax, crysmap, crystal_ids, info_dict, disk=0, fontsize=7):
        """
        Add text annotations to crystals with information (energy, ID, etc).
        
        Args:
            ax (plt.Axes): Matplotlib axes to draw on
            crysmap (CrystalMap): Crystal map object
            crystal_ids (list): List of crystal IDs to annotate
            info_dict (dict): Map of crystal_id -> info_string
            disk (int): Disk number (default: 0)
            fontsize (int): Font size for annotations (default: 7)
        
        Returns:
            int: Number of annotations added
        
        Example:
            info = {100: "5.2MeV", 101: "3.1MeV"}
            n_ann = calo_vis.annotate_crystal_info(ax, crysmap, [100, 101], info)
        """
        try:
            if crysmap is None:
                self.logger.log("CrystalMap is None", "warning")
                return 0
            
            if disk not in crysmap.crystals_by_disk:
                self.logger.log(f"Disk {disk} not found", "warning")
                return 0
            
            disk_data = crysmap.crystals_by_disk[disk]
            n_annotated = 0
            
            for _, row in disk_data.iterrows():
                cryid = int(row['cryID'])
                
                if cryid in crystal_ids and cryid in info_dict:
                    xcry = row['xcry']
                    ycry = row['ycry']
                    info_text = str(info_dict[cryid])
                    
                    # Add text annotation
                    ax.text(xcry, ycry, info_text, fontsize=fontsize, ha='center', va='center',
                           color='black', weight='bold', zorder=75,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
                    n_annotated += 1
            
            self.logger.log(f"Added annotations to {n_annotated} crystals", "info")
            return n_annotated
        
        except Exception as e:
            self.logger.log(f"Error annotating crystals: {e}", "error")
            return 0
    
    def overlay_particles_on_crystals(self, ax, particle_x, particle_y, particle_labels=None,
                                     marker='*', size=200, color='lime', edge_color='green'):
        """
        Overlay simulated particle positions on crystal map.
        
        Args:
            ax (plt.Axes): Matplotlib axes to draw on
            particle_x (np.ndarray): X coordinates of particles
            particle_y (np.ndarray): Y coordinates of particles
            particle_labels (np.ndarray, optional): Labels for particles (PDG, ID, etc)
            marker (str): Marker style (default: '*')
            size (int): Marker size (default: 200)
            color (str): Marker fill color (default: 'lime')
            edge_color (str): Marker edge color (default: 'green')
        
        Returns:
            int: Number of particles plotted
        
        Example:
            n_particles = calo_vis.overlay_particles_on_crystals(ax, x_arr, y_arr, labels)
        """
        try:
            particle_x = np.array(particle_x, dtype=float)
            particle_y = np.array(particle_y, dtype=float)
            
            if len(particle_x) == 0 or len(particle_y) == 0:
                self.logger.log("No particles to overlay", "warning")
                return 0
            
            # Plot particles
            ax.scatter(particle_x, particle_y, alpha=0.9, s=size, marker=marker,
                      edgecolors=edge_color, c=color, zorder=100)
            
            # Add labels if provided
            if particle_labels is not None:
                particle_labels = np.array(particle_labels)
                for i, (x, y) in enumerate(zip(particle_x, particle_y)):
                    label = str(particle_labels[i]) if i < len(particle_labels) else ""
                    ax.text(x, y, label, fontsize=8, ha='center', va='bottom', color='black')
            
            self.logger.log(f"Overlaid {len(particle_x)} particles", "info")
            return len(particle_x)
        
        except Exception as e:
            self.logger.log(f"Error overlaying particles: {e}", "error")
            return 0
    
    def set_2d_axis_limits(self, ax, crysmap, disk=0, margin=200):
        """
        Set 2D axis limits to show crystal map with margin.
        
        Args:
            ax (plt.Axes): Matplotlib axes to configure
            crysmap (CrystalMap): Crystal map object
            disk (int): Disk number (default: 0)
            margin (float): Margin around crystals in mm (default: 200)
        
        Returns:
            None
        
        Example:
            calo_vis.set_2d_axis_limits(ax, crysmap, disk=0, margin=150)
            ax.set_aspect('equal')
        """
        try:
            if crysmap is None:
                return
            
            if disk not in crysmap.crystals_by_disk:
                return
            
            disk_data = crysmap.crystals_by_disk[disk]
            
            if len(disk_data) == 0:
                return
            
            all_x = disk_data['xcry'].values
            all_y = disk_data['ycry'].values
            
            if len(all_x) > 0 and len(all_y) > 0:
                ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
                ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
                ax.set_aspect('equal')
                self.logger.log(f"Set axis limits for disk {disk}", "info")
        
        except Exception as e:
            self.logger.log(f"Error setting axis limits: {e}", "error")
    
    def create_2d_event_display(self, crysmap, disk=0, digi_positions=None, digi_energies=None,
                               particle_x=None, particle_y=None, particle_labels=None,
                               show_pipes=False, figsize=(12, 10)):
        """
        Create a complete 2D event display with all overlays.
        
        Convenience function that combines crystal grid, digis, and particles in one plot.
        
        Args:
            crysmap (CrystalMap): Crystal map object
            disk (int): Disk number to display (default: 0)
            digi_positions (list): List of (x, y) digi positions (optional)
            digi_energies (np.ndarray): Energy values for digis (optional)
            particle_x (np.ndarray): X coordinates of particles (optional)
            particle_y (np.ndarray): Y coordinates of particles (optional)
            particle_labels (np.ndarray): Labels for particles (optional)
            show_pipes (bool): Whether to draw pipes (default: False)
            figsize (tuple): Figure size (default: (12, 10))
        
        Returns:
            tuple: (fig, ax)
        
        Example:
            fig, ax = calo_vis.create_2d_event_display(crysmap, disk=0,
                  digi_positions=pos, digi_energies=energy, particle_x=px, particle_y=py)
            plt.savefig("event_display.png")
        """
        try:
            fig, ax = plt.subplots(figsize=figsize)
            
            # Draw crystal grid
            self.plot_crystal_grid(ax, crysmap, disk=disk, show_labels=False, alpha=0.7)
            
            # Draw pipes if requested
            if show_pipes:
                self.plot_pipes(ax, crysmap, disk=disk, pipe_color='red', pipe_alpha=0.3)
            
            # Draw energy heatmap if digi info provided
            if digi_positions is not None and digi_energies is not None:
                self.plot_2d_energy_heatmap(ax, crysmap, digi_positions, digi_energies,
                                           disk=disk, cmap='YlOrRd')
            
            # Overlay particles if provided
            if particle_x is not None and particle_y is not None:
                self.overlay_particles_on_crystals(ax, particle_x, particle_y, 
                                                  particle_labels=particle_labels)
            
            # Configure axis
            self.set_2d_axis_limits(ax, crysmap, disk=disk, margin=200)
            ax.set_xlabel('X (mm)', fontsize=11)
            ax.set_ylabel('Y (mm)', fontsize=11)
            ax.set_title(f'Disk {disk}: 2D Event Display', fontsize=13)
            ax.grid(True, alpha=0.3)
            
            self.logger.log("Created 2D event display", "success")
            return fig, ax
        
        except Exception as e:
            self.logger.log(f"Error creating 2D event display: {e}", "error")
            return None, None
    
    def find_crystals_by_position(self, crysmap, positions_x, positions_y, disk=0, tolerance=17):
        """
        Find crystal IDs for given (x, y) positions on a disk.
        
        Each crystal is a 34x34 mm square. A position belongs to a crystal if it falls
        within the crystal's bounds (xcry ± tolerance, ycry ± tolerance).
        
        Args:
            crysmap (CrystalMap): Crystal map object
            positions_x (np.ndarray or list): X coordinates of positions
            positions_y (np.ndarray or list): Y coordinates of positions
            disk (int): Disk number (default: 0)
            tolerance (float): Half-width of crystal in mm (default: 17, for 34x34 mm crystals)
        
        Returns:
            list: Crystal IDs that contain each position (or empty if no match)
        
        Example:
            crystal_ids = calo_vis.find_crystals_by_position(crysmap, [100, 150], [-50, 75], disk=0)
            # Returns list of crystal IDs at those positions
        """
        try:
            if crysmap is None:
                self.logger.log("CrystalMap is None", "warning")
                return []
            
            if disk not in crysmap.crystals_by_disk:
                self.logger.log(f"Disk {disk} not found", "warning")
                return []
            
            disk_data = crysmap.crystals_by_disk[disk]
            positions_x = np.array(positions_x, dtype=float)
            positions_y = np.array(positions_y, dtype=float)
            
            if len(positions_x) == 0 or len(positions_y) == 0:
                return []
            
            crystal_ids = []
            
            for px, py in zip(positions_x, positions_y):
                # Find crystals containing this position
                matching = disk_data[
                    (abs(disk_data['xcry'] - px) <= tolerance) &
                    (abs(disk_data['ycry'] - py) <= tolerance)
                ]
                
                if len(matching) > 0:
                    # Get the crystal ID (typically only one crystal per position)
                    cry_id = int(matching.iloc[0]['cryID'])
                    crystal_ids.append(cry_id)
                    self.logger.log(f"Position ({px:.1f}, {py:.1f}) -> Crystal {cry_id}", "debug")
                else:
                    self.logger.log(f"Position ({px:.1f}, {py:.1f}) -> No crystal found", "debug")
            
            self.logger.log(f"Found {len(crystal_ids)} crystals for {len(positions_x)} positions", "info")
            return crystal_ids
        
        except Exception as e:
            self.logger.log(f"Error finding crystals by position: {e}", "error")
            return []
    
    
    def highlight_by_sipms(self, ax, crysmap, sipm_ids, disk=0, 
                          color='yellow', edge_color='orange', linewidth=2):
        """
        Highlight crystals corresponding to a list of SiPM IDs from event data.
        
        Looks up SiPM IDs directly in the crystal map to find which crystals to highlight.
        
        Args:
            ax (plt.Axes): Matplotlib axes to draw on
            crysmap (CrystalMap): Crystal map object
            sipm_ids (list or np.ndarray): List of SiPM IDs from event data
            disk (int): Disk number (default: 0)
            color (str): Highlight color (default: 'yellow')
            edge_color (str): Edge color (default: 'orange')
            linewidth (int): Edge line width (default: 2)
        
        Returns:
            int: Number of crystals highlighted
        
        Example:
            # Highlight digis from event's SiPM IDs
            n_high = calo_vis.highlight_by_sipms(ax, crysmap, event_digi_sipms)
        """
        try:
            if crysmap is None:
                self.logger.log("CrystalMap is None", "warning")
                return 0
            
            if disk not in crysmap.crystals_by_disk:
                self.logger.log(f"Disk {disk} not found in crystal map", "warning")
                return 0
            
            # Convert to set of integers for fast lookup
            sipm_ids = set([int(sid) for sid in sipm_ids])
            self.logger.log(f"Looking up {len(sipm_ids)} SiPM IDs in crystal map", "debug")
            
            disk_data = crysmap.crystals_by_disk[disk]
            n_highlighted = 0
            found_sipms = set()
            
            # Look for crystals with matching SiPM IDs
            for _, row in disk_data.iterrows():
                sipm_id = int(row.get('SiPMID', -1)) if 'SiPMID' in row else int(row.get('sipmid', -1)) if 'sipmid' in row else -1
                
                if sipm_id in sipm_ids:
                    found_sipms.add(sipm_id)
                    xcry = row['xcry']
                    ycry = row['ycry']
                    
                    # Draw highlight patch
                    highlight = patches.Rectangle(
                        (xcry - 17, ycry - 17),
                        34, 34,
                        linewidth=linewidth,
                        edgecolor=edge_color,
                        facecolor=color,
                        alpha=0.8,
                        zorder=50
                    )
                    ax.add_patch(highlight)
                    n_highlighted += 1
            
            self.logger.log(f"Found {len(found_sipms)}/{len(sipm_ids)} SiPM IDs in crystal map, drew {n_highlighted} patches", "info")
            if len(found_sipms) < len(sipm_ids):
                missing = sipm_ids - found_sipms
                self.logger.log(f"Missing SiPM IDs: {sorted(list(missing))[:10]}", "debug")
            
            return n_highlighted
        
        except Exception as e:
            self.logger.log(f"Error highlighting by SiPMs: {e}", "error")
            return 0
    
    def highlight_digis(self, ax, crysmap, data, evt_idx, digi_indices, disk=0,
                       color='yellow', edge_color='orange', linewidth=2):
        """
        Highlight crystals corresponding to specific calorimeter digis.
        
        Extracts SiPM IDs from specified digi indices and highlights their crystals.
        
        Args:
            ax (plt.Axes): Matplotlib axes to draw on
            crysmap (CrystalMap): Crystal map object
            data (ak.Array): Data array with calo digi branches
            evt_idx (int): Event index
            digi_indices (list or np.ndarray): Indices of digis to highlight (relative to event)
            disk (int): Disk number (default: 0)
            color (str): Highlight color (default: 'yellow')
            edge_color (str): Edge color (default: 'orange')
            linewidth (int): Edge line width (default: 2)
        
        Returns:
            int: Number of crystals highlighted
        
        Example:
            # Highlight digis 0, 2, 5 in event 0
            n_high = calo_vis.highlight_digis(ax, crysmap, data, evt_idx=0,
                                              digi_indices=[0, 2, 5])
        """
        try:
            import awkward as ak
            
            # Extract SiPM IDs for this event
            sipmid = data["calo"]["calodigis.SiPMID_"]
            sipm_list = ak.to_list(sipmid[evt_idx])
            
            # Handle nested lists
            if isinstance(sipm_list, list) and len(sipm_list) > 0 and isinstance(sipm_list[0], list):
                sipm_list = [val for sublist in sipm_list for val in (sublist if isinstance(sublist, list) else [sublist])]
            
            # Extract SiPMs for specified digi indices
            digi_indices = np.array(digi_indices, dtype=int)
            selected_sipms = [int(sipm_list[i]) for i in digi_indices if i < len(sipm_list)]
            
            # Highlight by SiPM
            n_highlighted = self.highlight_by_sipms(
                ax, crysmap, selected_sipms, disk=disk,
                color=color, edge_color=edge_color, linewidth=linewidth
            )
            
            self.logger.log(f"Highlighted {len(selected_sipms)} digis ({n_highlighted} crystals) for event {evt_idx}", "info")
            return n_highlighted
        except Exception as e:
            self.logger.log(f"Error highlighting digis: {e}", "error")
            return 0
    
    def highlight_reco_digis(self, ax, crysmap, data, evt_idx, reco_indices, disk=0,
                            color='cyan', edge_color='blue', linewidth=2):
        """
        Highlight crystals corresponding to specific reconstructed digis.
        
        Maps reco digis back to calo digis via caloDigiIdx, then highlights their crystals.
        
        Args:
            ax (plt.Axes): Matplotlib axes to draw on
            crysmap (CrystalMap): Crystal map object
            data (ak.Array): Data array with reco and calo digi branches
            evt_idx (int): Event index
            reco_indices (list or np.ndarray): Indices of reco digis to highlight
            disk (int): Disk number (default: 0)
            color (str): Highlight color (default: 'cyan')
            edge_color (str): Edge color (default: 'blue')
            linewidth (int): Edge line width (default: 2)
        
        Returns:
            int: Number of crystals highlighted
        
        Example:
            # Highlight reco digis 0, 1 in event 0
            n_high = calo_vis.highlight_reco_digis(ax, crysmap, data, evt_idx=0,
                                                   reco_indices=[0, 1])
        """
        try:
            import awkward as ak
            
            # Extract mapping from reco to calo digis
            reco_caloDigiIdx = data["calo"]["calorecodigis.caloDigiIdx_"]
            reco_indices = np.array(reco_indices, dtype=int)
            
            idx_list = ak.to_list(reco_caloDigiIdx[evt_idx])
            if isinstance(idx_list, list) and len(idx_list) > 0 and isinstance(idx_list[0], list):
                idx_list = [val for sublist in idx_list for val in (sublist if isinstance(sublist, list) else [sublist])]
            
            # Get calo digi indices for specified reco indices
            calo_digi_indices = []
            for reco_idx in reco_indices:
                if reco_idx < len(idx_list):
                    calo_idx_data = idx_list[reco_idx]
                    # Handle both single values and lists
                    if isinstance(calo_idx_data, (list, tuple)):
                        calo_digi_indices.extend(calo_idx_data)
                    else:
                        calo_digi_indices.append(int(calo_idx_data))
            
            # Now highlight those calorimeter digis
            n_highlighted = self.highlight_digis(
                ax, crysmap, data, evt_idx, calo_digi_indices, disk=disk,
                color=color, edge_color=edge_color, linewidth=linewidth
            )
            
            self.logger.log(f"Highlighted {len(reco_indices)} reco digis ({n_highlighted} crystals) for event {evt_idx}", "info")
            return n_highlighted
        except Exception as e:
            self.logger.log(f"Error highlighting reco digis: {e}", "error")
            return 0
    
    def plot_event_display(self, crysmap, data, evt_idx, disk=0, output_file=None,
                              show_digis=True, show_reco_digis=True, show_hits=False, 
                              show_clusters=False, show_pipes=False, figsize=(14, 12)):
        """
        Create flexible event display with selectable products highlighted on crystal map.
        
        Users can choose which products to display (digis, reco digis, hits, clusters, pipes).
        Each product type highlights its associated crystals with a distinct color.
        
        Args:
            crysmap (CrystalMap): Crystal map object
            data (ak.Array): Flat data array with calodigis.* and calorecodigis.* branches
                            (should be data["calo"] extracted from Processor output)
            evt_idx (int): Event index to display
            disk (int): Disk number to show (default: 0)
            output_file (str): Optional path to save figure
            show_digis (bool): Display digis - yellow crystals with orange edges
            show_reco_digis (bool): Display reconstructed digis - cyan crystals with blue edges
            show_hits (bool): Display hits - green crystals with dark green edges
            show_clusters (bool): Display clusters - magenta crystals with red edges
            show_pipes (bool): Display calibration pipe geometry - red pipe outlines (default: False)
            figsize (tuple): Figure size in inches (default: (14, 12))
        
        Returns:
            tuple: (fig, ax) matplotlib figure and axes
        
        Example:
            # Extract calo data from nested Processor output
            data_full = processor.process_data(...)  # Returns {"evt": ..., "calo": ...}
            calo_data = data_full["calo"]
            
            # Show only digis and reco digis
            fig, ax = calo_vis.plot_event_display(crysmap, calo_data, evt_idx=0,
                                                  show_digis=True, show_reco_digis=True,
                                                  show_hits=False, output_file='event.png')
            
            # Show all available products
            fig, ax = calo_vis.plot_event_display(crysmap, calo_data, evt_idx=0,
                                                  show_digis=True, show_reco_digis=True,
                                                  show_hits=True, show_clusters=True)
        """
        try:
            import awkward as ak
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Draw crystal grid with labels
            self.plot_crystal_grid(ax, crysmap, disk=disk, show_labels=True,
                                  alpha=0.5, crystal_color='lightgray', edge_color='darkgray')
            
            # Extract digi data - data parameter should be the flat 'calo' portion
            try:
                sipmid = data["calodigis.SiPMID_"]
                diskID = data["calodigis.diskID_"]
                posX = data["calodigis.posX_"]
                posY = data["calodigis.posY_"]
            except (KeyError, TypeError) as e:
                self.logger.log(f"Could not extract digi data branches: {e}", "error")
                return None, None
            
            sipm_list = ak.to_list(sipmid[evt_idx])
            disk_id_list = ak.to_list(diskID[evt_idx])
            posX_list = ak.to_list(posX[evt_idx])
            posY_list = ak.to_list(posY[evt_idx])
            
            self.logger.log(f"Event {evt_idx}: Extracted {len(sipm_list)} digis", "debug")
            
            # Flatten if nested
            if isinstance(sipm_list, list) and len(sipm_list) > 0 and isinstance(sipm_list[0], list):
                sipm_list = [val for sublist in sipm_list for val in (sublist if isinstance(sublist, list) else [sublist])]
            if isinstance(disk_id_list, list) and len(disk_id_list) > 0 and isinstance(disk_id_list[0], list):
                disk_id_list = [val for sublist in disk_id_list for val in (sublist if isinstance(sublist, list) else [sublist])]
            if isinstance(posX_list, list) and len(posX_list) > 0 and isinstance(posX_list[0], list):
                posX_list = [val for sublist in posX_list for val in (sublist if isinstance(sublist, list) else [sublist])]
            if isinstance(posY_list, list) and len(posY_list) > 0 and isinstance(posY_list[0], list):
                posY_list = [val for sublist in posY_list for val in (sublist if isinstance(sublist, list) else [sublist])]
            
            # Filter to only requested disk
            disk_indices = [i for i, d in enumerate(disk_id_list) if int(d) == disk]
            disk_posX = [float(posX_list[i]) for i in disk_indices if i < len(posX_list)]
            disk_posY = [float(posY_list[i]) for i in disk_indices if i < len(posY_list)]
            disk_sipms = [int(sipm_list[i]) for i in disk_indices if i < len(sipm_list)]
            
            self.logger.log(f"Event {evt_idx}: Found {len(disk_sipms)} digis on disk {disk}", "info")
            
            # Find crystal IDs for these positions
            digi_crystal_ids = []
            if len(disk_posX) > 0 and len(disk_posY) > 0:
                digi_crystal_ids = self.find_crystals_by_position(crysmap, disk_posX, disk_posY, disk=disk)
                self.logger.log(f"Event {evt_idx}: Found {len(digi_crystal_ids)} crystals with digis", "info")
            
            n_total_highlighted = 0
            legend_patches = []
            
            # Show digis
            if show_digis and len(digi_crystal_ids) > 0:
                n_high = self.highlight_hit_crystals(ax, crysmap, digi_crystal_ids, disk=disk,
                                                     color='yellow', edge_color='orange', linewidth=2,
                                                     show_labels=True, fontsize=8)
                n_total_highlighted += n_high
                # Create legend patch with actual color
                legend_patches.append(patches.Patch(
                    facecolor='yellow', edgecolor='orange', linewidth=2,
                    label=f'Digis (n={len(disk_sipms)}, yellow highlight)'
                ))
                self.logger.log(f"Highlighted {n_high} crystals for {len(disk_sipms)} digis on disk {disk}", "info")
            
            # Show reco digis
            if show_reco_digis:
                try:
                    reco_caloDigiIdx = data["calorecodigis.caloDigiIdx_"]
                    reco_caloDigiIdx_list = ak.to_list(reco_caloDigiIdx[evt_idx])
                    
                    if isinstance(reco_caloDigiIdx_list, list) and len(reco_caloDigiIdx_list) > 0 and isinstance(reco_caloDigiIdx_list[0], list):
                        reco_caloDigiIdx_list = [val for sublist in reco_caloDigiIdx_list for val in (sublist if isinstance(sublist, list) else [sublist])]
                    
                    # Get positions for reco digis (via their calo digi indices)
                    reco_posX = []
                    reco_posY = []
                    for reco_idx, calo_idx_ref in enumerate(reco_caloDigiIdx_list):
                        try:
                            calo_idx = int(calo_idx_ref)
                            if calo_idx < len(posX_list) and calo_idx < len(posY_list):
                                reco_posX.append(float(posX_list[calo_idx]))
                                reco_posY.append(float(posY_list[calo_idx]))
                        except (ValueError, TypeError, IndexError):
                            pass
                    
                    # Find crystals for reco digi positions
                    if len(reco_posX) > 0 and len(reco_posY) > 0:
                        reco_crystal_ids = self.find_crystals_by_position(crysmap, reco_posX, reco_posY, disk=disk)
                        
                        if len(reco_crystal_ids) > 0:
                            n_high = self.highlight_hit_crystals(ax, crysmap, reco_crystal_ids, disk=disk,
                                                                 color='cyan', edge_color='blue', linewidth=2,
                                                                 show_labels=True, fontsize=8)
                            n_total_highlighted += n_high
                            legend_patches.append(patches.Patch(
                                facecolor='cyan', edgecolor='blue', linewidth=2,
                                label=f'Reco Digis (n={len(reco_posX)}, cyan highlight)'
                            ))
                            self.logger.log(f"Highlighted {n_high} crystals for {len(reco_posX)} reco digis", "info")
                
                except Exception as e:
                    self.logger.log(f"Could not display reco digis: {e}", "warning")
            
            # Show hits (if available)
            if show_hits:
                try:
                    # Try to load calohits data
                    # For now, skip if not available
                    self.logger.log("Hits not yet implemented in data loading", "debug")
                except Exception as e:
                    self.logger.log(f"Could not display hits: {e}", "debug")
            
            # Show clusters (if available)
            if show_clusters:
                try:
                    # Try to load caloclusters data
                    # For now, skip if not available
                    self.logger.log("Clusters not yet implemented in data loading", "debug")
                except Exception as e:
                    self.logger.log(f"Could not display clusters: {e}", "debug")
            
            # Draw pipes if requested
            if show_pipes:
                try:
                    pipe_geom = PipeGeometry()
                    # Create minimal logger for pipe drawing
                    from .pylogger import Logger as PipeLogger
                    # Draw pipes using lightweight version for clarity
                    self._draw_pipes_lightweight(ax, pipe_geom)
                    legend_patches.append(patches.Patch(
                        facecolor='none', edgecolor='red', linewidth=2, linestyle='-',
                        label='Calibration Pipes (red)'
                    ))
                    self.logger.log("Drew calibration pipe geometry", "info")
                except Exception as e:
                    self.logger.log(f"Could not draw pipes: {e}", "warning")
            
            # Set axis limits and labels
            self.set_2d_axis_limits(ax, crysmap, disk=disk, margin=300)
            ax.set_xlabel('X (mm)', fontsize=12)
            ax.set_ylabel('Y (mm)', fontsize=12)
            
            # Build title with detailed product information
            product_info = []
            if show_digis:
                product_info.append(f'Digis={len(disk_sipms)}')
            if show_reco_digis:
                try:
                    reco_caloDigiIdx = data["calorecodigis.caloDigiIdx_"]
                    reco_count = len(ak.to_list(reco_caloDigiIdx[evt_idx]))
                    product_info.append(f'Reco={reco_count}')
                except Exception:
                    pass
            if show_pipes:
                product_info.append('Pipes')
            
            title = f'Event {evt_idx}, Disk {disk}'
            if product_info:
                title += ': [' + ', '.join(product_info) + ']'
            ax.set_title(title, fontsize=14, fontweight='bold')
            
            ax.grid(True, alpha=0.2)
            ax.set_aspect('equal')
            
            # Add legend with color-coded patches if items highlighted
            if legend_patches:
                # Always add background crystal info
                legend_patches.insert(0, patches.Patch(
                    facecolor='lightgray', edgecolor='darkgray', linewidth=1,
                    label='Crystal Grid (no hits)'
                ))
                ax.legend(handles=legend_patches, loc='upper right', fontsize=10, 
                         title='Data Products (click entry to toggle)', title_fontsize=11,
                         framealpha=0.95)
            
            if output_file:
                plt.savefig(output_file, dpi=100, bbox_inches='tight')
                self.logger.log(f"Saved event display to {output_file}", "success")
            
            return fig, ax
        
        except Exception as e:
            self.logger.log(f"Error plotting event display: {e}", "error")
            import traceback
            self.logger.log(traceback.format_exc(), "debug")
            return None, None
    
    def _draw_pipes_lightweight(self, ax, pipe_geometry, pipe_color='red', pipe_alpha=0.5, label_pipes=False):
        """
        Draw lightweight pipe centerlines without fills.
        
        Efficient version that only draws pipe centerlines. Based on CaloCalibGun geometry.
        
        Args:
            ax (plt.Axes): Matplotlib axes to draw on
            pipe_geometry (PipeGeometry): Pipe geometry object with parameters
            pipe_color (str): Color for pipe outlines (default: 'red')
            pipe_alpha (float): Transparency of pipe elements (default: 0.5)
            label_pipes (bool): Whether to label pipes (default: False)
        """
        try:
            geom = pipe_geometry
            n_pipes = len(geom.pipe_tor_radii)
            sign_values = [-1.0, 1.0]
            
            quad_idx = 0
            for sign_x in sign_values:
                for sign_y in sign_values:
                    for idx in range(n_pipes):
                        rad_lg_tor = geom.pipe_tor_radii[idx]
                        phi_lbd = geom.large_tor_phi[idx]
                        phi_sbd = geom.small_tor_phi[idx]
                        phi_end = geom.phi_end[idx]
                        y_pos = geom.yposition[idx] if idx < len(geom.yposition) else geom.ysmall[idx]
                        
                        # Large torus arc centerline
                        phi_lg_max = np.radians(phi_lbd / 2)
                        theta_lg = np.linspace(0, phi_lg_max, 40)
                        x_lg = sign_x * rad_lg_tor * np.cos(theta_lg)
                        y_lg = sign_y * rad_lg_tor * np.sin(theta_lg)
                        ax.plot(x_lg, y_lg, color=pipe_color, linewidth=2, alpha=pipe_alpha)
                        
                        # Small torus arc centerline
                        x_sm_center = geom.xsmall + geom.xdistance * idx
                        phi_sm_start = 180.0 + phi_lbd / 2.0 - phi_sbd
                        phi_sm_end = 180.0 + phi_lbd / 2.0
                        
                        theta_sm = np.linspace(np.radians(phi_sm_start), np.radians(phi_sm_end), 40)
                        x_sm = sign_x * (x_sm_center + geom.rad_smtor * np.cos(theta_sm))
                        y_sm = sign_y * (y_pos + geom.rad_smtor * np.sin(theta_sm))
                        ax.plot(x_sm, y_sm, color=pipe_color, linewidth=2, alpha=pipe_alpha)
                        
                        # Straight section
                        y_manifold = sign_y * geom.rinner_manifold * np.sin(np.radians(90.0 - phi_end))
                        x_start = sign_x * (geom.xsmall + geom.xdistance * idx - geom.rad_smtor * np.cos(np.radians(phi_end)))
                        y_start = sign_y * (y_pos + geom.rad_smtor * np.sin(np.radians(phi_end)))
                        x_end = sign_x * (geom.xsmall + geom.xdistance * idx)
                        y_end = y_manifold
                        
                        ax.plot([x_start, x_end], [y_start, y_end], color=pipe_color, linewidth=2, alpha=pipe_alpha)
                        
                        # Label pipes (only first quadrant to avoid clutter)
                        if label_pipes and quad_idx == 0:
                            mid_idx = len(x_lg) // 2
                            if mid_idx < len(x_lg):
                                ax.text(x_lg[mid_idx], y_lg[mid_idx] - 30, f'Pipe {idx}',
                                       fontsize=8, ha='center', va='top', fontweight='bold',
                                       color=pipe_color)
                    
                    quad_idx += 1
        
        except Exception as e:
            self.logger.log(f"Error drawing pipes: {e}", "warning")
