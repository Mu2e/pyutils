#! /usr/bin/env python
"""
Calorimeter Analysis and Visualization Utilities

Classes:
    CaloAnalysis: Data matching and pipeline tracing utilities
    CaloVisualization: Crystal map drawing and event display utilities
"""

import numpy as np
import pandas as pd
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import os
import traceback
from typing import List, Tuple, Optional, Dict

from .pylogger import Logger


class PipeGeometry:
    """
    Represents the calibration pipe geometry from CaloCalibGun.
    
    Encapsulates pipe geometry parameters and calculations from the mu2e
    calorimeter calibration gun system, directly matching CaloCalibGun_module.cc.
    
    Configuration from calorimeter_CsI_v2.txt:
    - nPipes = 5
    - pipeTorRadius = {397, 457, 517, 577, 637} mm
    - pipeRadius = 4.75 mm (cross-sectional radius, not used for centerline visualization)
    - radSmTor = 41.0 mm (small torus radius)
    - xsmall = 71.0 mm
    - xdistance = 60.0 mm
    - rInnerManifold = 681.6 mm
    """
    
    def __init__(
        self,
        pipe_tor_radii: List[float] = None,
        large_tor_phi: List[float] = None,
        small_tor_phi: List[float] = None,
        phi_end: List[float] = None,
        ysmall: List[float] = None,
        rad_smtor: float = 41.0,
        xsmall: float = 71.0,
        xdistance: float = 60.0,
        rinner_manifold: float = 681.6,
        pipe_radius: float = 4.75,
        pipe_thickness: float = 0.5,
        pipe_init_separation: float = 25.4,
    ):
        """
        Initialize pipe geometry parameters from calorimeter config.
        
        All parameters match calorimeter_CsI_v2.txt exactly.
        """
        self.pipe_tor_radii = pipe_tor_radii or [397.0, 457.0, 517.0, 577.0, 637.0]
        self.pipe_radius = pipe_radius
        self.pipe_thickness = pipe_thickness
        self.pipe_init_separation = pipe_init_separation
        self.large_tor_phi = large_tor_phi or [161.34, 149.50, 139.50, 132.07, 125.39]
        self.small_tor_phi = small_tor_phi or [84.63, 85.28, 85.79, 86.20, 86.53]
        self.phi_end = phi_end or [3.96, 10.53, 15.80, 20.16, 23.84]
        self.ysmall = ysmall or [432.2, 480.5, 524.3, 564.7, 602.5]
        self.rad_smtor = rad_smtor
        self.xsmall = xsmall
        self.xdistance = xdistance
        self.rinner_manifold = rinner_manifold
        self.yposition = self.ysmall
    
    def calculate_pipe_centers(self) -> List[Tuple[float, float]]:
        """
        Calculate the center positions (x, y) of each pipe.
        
        Based on CaloCalibGun geometry, returns centers in mm.
        """
        centers = []
        n_pipes = len(self.pipe_tor_radii)
        
        for idx in range(n_pipes):
            # Approximate center position of each pipe
            # Using large torus radius as primary center location
            x_center = self.pipe_tor_radii[idx] * np.cos(np.radians(self.large_tor_phi[idx] / 2.0))
            y_center = self.ysmall[idx]
            
            centers.append((x_center, y_center))
        
        return centers


class CrystalMap:
    """
    Visualization class for mu2e calorimeter crystals from map files.
    
    This class reads a detector map file and visualizes crystals as squares
    with their crystal IDs, positioned according to their x,y coordinates.
    
    Optionally includes overlay of calibration pipes from the CaloCalibGun system.
    
    Parameters:
        map_file (str): Path to the detector map file
        verbosity (int): Logger verbosity level (0: errors only, 1: info, 2: debug, 3: deep debug)
        style_path (str, optional): Path to matplotlib style file
        pipe_geometry (PipeGeometry, optional): Pipe geometry parameters. Uses defaults if None.
    """
    
    def __init__(
        self,
        map_file: str,
        verbosity: int = 1,
        style_path: Optional[str] = None,
        pipe_geometry: Optional[PipeGeometry] = None,
    ):
        """Initialize CrystalMap instance."""
        self.map_file = map_file
        self.verbosity = verbosity
        self.style_path = style_path
        self.data = None
        self.crystals_by_disk = {}
        
        # Initialize pipe geometry
        self.pipe_geometry = pipe_geometry or PipeGeometry()
        
        if self.style_path is None:
            self.style_path = os.path.join(os.path.dirname(__file__), "mu2e.mplstyle")
        
        if os.path.exists(self.style_path):
            plt.style.use(self.style_path)
        
        self.logger = Logger(
            print_prefix="[pycrystal]",
            verbosity=self.verbosity
        )
        
        # Load the map file
        self._load_map()
        self.logger.log(f"Initialized CrystalMap from {map_file} with verbosity={self.verbosity}", "info")
        self.logger.log(f"Pipe geometry initialized with {len(self.pipe_geometry.pipe_tor_radii)} pipes", "info")
    
    def _load_map(self) -> None:
        """Load and parse the detector map file."""
        try:
            # Read the map file with whitespace delimiter
            self.data = pd.read_csv(
                self.map_file,
                sep='\s+',
                dtype={
                    'y': int,
                    'x': int,
                    'disk': int,
                    'xcry': float,
                    'ycry': float,
                    'cryID': int,
                    'Type': str
                }
            )
            
            # Filter to only CAL (calorimeter) crystals and valid disks
            self.data = self.data[(self.data['Type'] == 'CAL') & (self.data['disk'] >= 0)].copy()
            
            # Organize crystals by disk
            for disk in sorted(self.data['disk'].unique()):
                disk_data = self.data[self.data['disk'] == disk].copy()
                self.crystals_by_disk[int(disk)] = disk_data
            
            self.logger.log(
                f"Loaded map with {len(self.data)} CAL crystals from {len(self.crystals_by_disk)} disks",
                "info"
            )
            
            for disk_id, disk_crystals in self.crystals_by_disk.items():
                self.logger.log(f"  Disk {disk_id}: {len(disk_crystals)} crystals", "info")
        
        except Exception as e:
            self.logger.log(f"Error loading map file: {e}", "error")
            raise
    
    def _draw_pipes(
        self,
        ax: plt.Axes,
        pipe_color: str = 'red',
        pipe_alpha: float = 0.5,
        label_pipes: bool = True,
    ) -> None:
        """
        Draw calibration pipe geometry on the given axes.
        
        Directly implements CaloCalibGun pipe geometry from calorimeter_CsI_v2.txt:
        - Each pipe consists of 3 sections: large torus arc, small torus arc, straight segment
        - Pipes are drawn in all 4 quadrants using sign combinations (xsn, ysn)
        - Total of 20 pipe sets (5 pipes × 4 quadrants)
        
        Parameters from calorimeter_CsI_v2.txt:
        - pipeTorRadius = {397, 457, 517, 577, 637} mm (large torus radii)
        - radSmTor = 41.0 mm (small torus radius)
        - xsmall = 71.0 mm, xdistance = 60.0 mm (small torus positioning)
        - largeTorPhi, smallTorPhi, straightEndPhi (angular extents)
        - yposition (straight section y-positions)
        - rInnerManifold = 681.6 mm
        
        Args:
            ax (plt.Axes): Matplotlib axes to draw on
            pipe_color (str): Color for pipe outlines
            pipe_alpha (float): Transparency of pipe elements
            label_pipes (bool): Whether to label each pipe with its index
        """
        geom = self.pipe_geometry
        n_pipes = len(geom.pipe_tor_radii)
        sign_values = [-1.0, 1.0]  # Two possible sign values (xsn and ysn)
        
        # Draw pipes in all 4 quadrants (combinations of sign_x and sign_y)
        quad_idx = 0
        for sign_x in sign_values:
            for sign_y in sign_values:
                for idx in range(n_pipes):
                    # Get geometry for this pipe
                    rad_lg_tor = geom.pipe_tor_radii[idx]  # Large torus radius
                    phi_lbd = geom.large_tor_phi[idx]      # Large torus full angle (degrees)
                    phi_sbd = geom.small_tor_phi[idx]      # Small torus full angle (degrees)
                    phi_end = geom.phi_end[idx]            # Straight section ending angle (degrees)
                    y_pos = geom.yposition[idx] if idx < len(geom.yposition) else geom.ysmall[idx]
                    
                    # ============================================================
                    # SECTION 1: Large Torus Arc
                    # Center at origin, radius = rad_lg_tor
                    # Angle range: 0 to phi_lbd/2 degrees
                    # Apply sign to both x and y
                    # ============================================================
                    phi_lg_max = np.radians(phi_lbd / 2)
                    theta_lg = np.linspace(0, phi_lg_max, 60)
                    x_lg = sign_x * rad_lg_tor * np.cos(theta_lg)
                    y_lg = sign_y * rad_lg_tor * np.sin(theta_lg)
                    
                    # Draw pipe with thickness using outer and inner edges
                    x_lg_outer = sign_x * (rad_lg_tor + geom.pipe_radius) * np.cos(theta_lg)
                    y_lg_outer = sign_y * (rad_lg_tor + geom.pipe_radius) * np.sin(theta_lg)
                    x_lg_inner = sign_x * (rad_lg_tor - geom.pipe_radius) * np.cos(theta_lg)
                    y_lg_inner = sign_y * (rad_lg_tor - geom.pipe_radius) * np.sin(theta_lg)
                    
                    # Fill between outer and inner edges
                    x_lg_fill = np.concatenate([x_lg_outer, x_lg_inner[::-1]])
                    y_lg_fill = np.concatenate([y_lg_outer, y_lg_inner[::-1]])
                    ax.fill(x_lg_fill, y_lg_fill, color=pipe_color, alpha=pipe_alpha * 0.6,
                           label='Large Torus Arc' if quad_idx == 0 and idx == 0 else '')
                    ax.plot(x_lg_outer, y_lg_outer, color=pipe_color, linewidth=1, alpha=pipe_alpha)
                    ax.plot(x_lg_inner, y_lg_inner, color=pipe_color, linewidth=1, alpha=pipe_alpha)
                    
                    # ============================================================
                    # SECTION 2: Small Torus Arc
                    # Center at (xsmall + xdistance*idx, yposition[idx])
                    # Radius = radSmTor
                    # Angle range: 180 + phi_lbd/2 - phi_sbd to 180 + phi_lbd/2 (in degrees)
                    # ============================================================
                    x_sm_center = geom.xsmall + geom.xdistance * idx
                    phi_sm_start = 180.0 + phi_lbd / 2.0 - phi_sbd
                    phi_sm_end = 180.0 + phi_lbd / 2.0
                    
                    theta_sm = np.linspace(np.radians(phi_sm_start), np.radians(phi_sm_end), 60)
                    x_sm = sign_x * (x_sm_center + geom.rad_smtor * np.cos(theta_sm))
                    y_sm = sign_y * (y_pos + geom.rad_smtor * np.sin(theta_sm))
                    
                    # Draw pipe with thickness using outer and inner edges
                    x_sm_outer = sign_x * (x_sm_center + (geom.rad_smtor + geom.pipe_radius) * np.cos(theta_sm))
                    y_sm_outer = sign_y * (y_pos + (geom.rad_smtor + geom.pipe_radius) * np.sin(theta_sm))
                    x_sm_inner = sign_x * (x_sm_center + (geom.rad_smtor - geom.pipe_radius) * np.cos(theta_sm))
                    y_sm_inner = sign_y * (y_pos + (geom.rad_smtor - geom.pipe_radius) * np.sin(theta_sm))
                    
                    # Fill between outer and inner edges
                    x_sm_fill = np.concatenate([x_sm_outer, x_sm_inner[::-1]])
                    y_sm_fill = np.concatenate([y_sm_outer, y_sm_inner[::-1]])
                    ax.fill(x_sm_fill, y_sm_fill, color=pipe_color, alpha=pipe_alpha * 0.6,
                           label='Small Torus Arc' if quad_idx == 0 and idx == 0 else '')
                    ax.plot(x_sm_outer, y_sm_outer, color=pipe_color, linewidth=1, alpha=pipe_alpha)
                    ax.plot(x_sm_inner, y_sm_inner, color=pipe_color, linewidth=1, alpha=pipe_alpha)
                    
                    # ============================================================
                    # SECTION 3: Straight Section
                    # From end of small torus to inner manifold
                    # Angle of manifold edge: 90 - phi_end (degrees)
                    # ============================================================
                    y_manifold = sign_y * geom.rinner_manifold * np.sin(np.radians(90.0 - phi_end))
                    x_start = sign_x * (geom.xsmall + geom.xdistance * idx - geom.rad_smtor * np.cos(np.radians(phi_end)))
                    y_start = sign_y * (y_pos + geom.rad_smtor * np.sin(np.radians(phi_end)))
                    
                    # Straight line from (x_start, y_start) to manifold intersection
                    x_end = sign_x * (geom.xsmall + geom.xdistance * idx)
                    y_end = y_manifold
                    
                    # Draw straight section with thickness
                    # Calculate perpendicular offset for pipe width
                    dx = x_end - x_start
                    dy = y_end - y_start
                    length = np.sqrt(dx**2 + dy**2)
                    
                    if length > 0:
                        # Unit perpendicular vector
                        perp_x = -dy / length
                        perp_y = dx / length
                        
                        # Offset points for thickness (pipe_radius on each side)
                        x_str_1 = x_start + perp_x * geom.pipe_radius
                        y_str_1 = y_start + perp_y * geom.pipe_radius
                        x_str_2 = x_end + perp_x * geom.pipe_radius
                        y_str_2 = y_end + perp_y * geom.pipe_radius
                        x_str_3 = x_end - perp_x * geom.pipe_radius
                        y_str_3 = y_end - perp_y * geom.pipe_radius
                        x_str_4 = x_start - perp_x * geom.pipe_radius
                        y_str_4 = y_start - perp_y * geom.pipe_radius
                        
                        # Fill rectangle
                        x_str_fill = [x_str_1, x_str_2, x_str_3, x_str_4]
                        y_str_fill = [y_str_1, y_str_2, y_str_3, y_str_4]
                        ax.fill(x_str_fill, y_str_fill, color=pipe_color, alpha=pipe_alpha * 0.6,
                               label='Straight Section' if quad_idx == 0 and idx == 0 else '')
                        
                        # Draw edges
                        ax.plot([x_str_1, x_str_2], [y_str_1, y_str_2], color=pipe_color, linewidth=1, alpha=pipe_alpha)
                        ax.plot([x_str_4, x_str_3], [y_str_4, y_str_3], color=pipe_color, linewidth=1, alpha=pipe_alpha)
                    
                    # ============================================================
                    # Draw Pipe Label (only for first quadrant to avoid clutter)
                    # ============================================================
                    if label_pipes and quad_idx == 0:
                        # Label at midpoint of large torus arc
                        mid_idx = len(x_lg) // 2
                        if mid_idx < len(x_lg):
                            ax.text(x_lg[mid_idx], y_lg[mid_idx] - 30, f'Pipe {idx}',
                                   fontsize=9, ha='center', va='top', fontweight='bold',
                                   color=pipe_color,
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                           alpha=0.8, edgecolor=pipe_color, linewidth=1))
                
                quad_idx += 1

    def _draw_pipes_lightweight(
        self,
        ax: plt.Axes,
        pipe_color: str = 'red',
        pipe_alpha: float = 0.5,
        label_pipes: bool = True,
    ) -> None:
        """
        Draw lightweight pipe centerlines (no fills).
        
        Memory-efficient version that only draws pipe centerlines without filled regions.
        Useful for overlays with many other elements.
        
        Args:
            ax (plt.Axes): Matplotlib axes to draw on
            pipe_color (str): Color for pipe outlines
            pipe_alpha (float): Transparency of pipe elements
            label_pipes (bool): Whether to label each pipe with its index
        """
        geom = self.pipe_geometry
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
                    
                    # Straight section centerline
                    y_manifold = sign_y * geom.rinner_manifold * np.sin(np.radians(90.0 - phi_end))
                    x_start = sign_x * (geom.xsmall + geom.xdistance * idx - geom.rad_smtor * np.cos(np.radians(phi_end)))
                    y_start = sign_y * (y_pos + geom.rad_smtor * np.sin(np.radians(phi_end)))
                    x_end = sign_x * (geom.xsmall + geom.xdistance * idx)
                    y_end = y_manifold
                    
                    ax.plot([x_start, x_end], [y_start, y_end], color=pipe_color, linewidth=2, alpha=pipe_alpha)
                
                quad_idx += 1

    
    def visualize_all_disks(
        self,
        title_prefix: str = "Mu2e Calorimeter Map",
        output_file: Optional[str] = None,
        crystal_size: float = 34.0,
        figsize: Optional[Tuple] = None,
        show_grid: bool = True,
        show_pipes: bool = False,
    ) -> Optional[plt.Figure]:
        """
        Visualize all CAL crystals from all disks.
        
        Creates one panel per disk showing all crystals as squares with their cryID labels.
        Optionally overlays calibration pipes from the CaloCalibGun system.
        
        Args:
            title_prefix (str): Prefix for the plot title
            output_file (str, optional): Path to save PNG file
            crystal_size (float): Size of crystal squares (in mm)
            figsize (tuple, optional): Figure size. Auto-calculated if None.
            show_grid (bool): Whether to display grid lines
            show_pipes (bool): Whether to overlay calibration pipes
        
        Returns:
            plt.Figure or None: The figure object if successful
        """
        try:
            n_disks = len(self.crystals_by_disk)
            if n_disks == 0:
                self.logger.log("No disks found in map data", "error")
                return None
            
            if figsize is None:
                figsize = (8 * n_disks, 8)
            
            fig, axes = plt.subplots(1, n_disks, figsize=figsize)
            if n_disks == 1:
                axes = [axes]
            
            for ax_idx, (disk_id, disk_data) in enumerate(sorted(self.crystals_by_disk.items())):
                ax = axes[ax_idx]
                
                # Plot crystals as squares
                for _, row in disk_data.iterrows():
                    xcry = row['xcry']
                    ycry = row['ycry']
                    cryID = row['cryID']
                    
                    # Create square patch centered at (xcry, ycry)
                    # crystal_size is the width/height of the square in mm
                    square = patches.Rectangle(
                        (xcry - crystal_size/2, ycry - crystal_size/2),
                        crystal_size,
                        crystal_size,
                        linewidth=1,
                        edgecolor='black',
                        facecolor='lightblue',
                        alpha=0.7
                    )
                    ax.add_patch(square)
                    
                    # Add cryID label
                    ax.text(
                        xcry, ycry,
                        str(int(cryID)),
                        fontsize=6,
                        ha='center',
                        va='center',
                        fontweight='bold'
                    )
                
                # Optionally overlay pipes
                if show_pipes:
                    self._draw_pipes(ax, pipe_color='red', pipe_alpha=0.3, label_pipes=True)
                
                # Formatting
                ax.set_xlabel("X Position (mm)", fontsize=11)
                ax.set_ylabel("Y Position (mm)", fontsize=11)
                ax.set_title(f"Disk {disk_id} ({len(disk_data)} crystals)", fontsize=12, fontweight='bold')
                ax.set_aspect('equal')
                
                if show_grid:
                    ax.grid(True, alpha=0.3)
                
                # Auto-scale axes based on data
                all_x = disk_data['xcry'].values
                all_y = disk_data['ycry'].values
                if len(all_x) > 0:
                    margin = 50
                    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
                    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
            
            fig.suptitle(f"{title_prefix}: All CAL Crystals", fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            if output_file:
                fig.savefig(output_file, dpi=300, bbox_inches='tight')
                self.logger.log(f"Saved visualization to {output_file}", "success")
            
            return fig
        
        except Exception as e:
            self.logger.log(f"Error in visualize_all_disks: {e}", "error")
            return None
    
    def visualize_disk(
        self,
        disk_id: int,
        title: Optional[str] = None,
        output_file: Optional[str] = None,
        crystal_size: float = 34.0,
        figsize: Tuple = (12, 12),
        show_grid: bool = True,
        show_pipes: bool = False,
    ) -> Optional[plt.Figure]:
        """
        Visualize all CAL crystals for a single disk.
        
        Optionally overlays calibration pipes from the CaloCalibGun system.
        
        Args:
            disk_id (int): Disk ID to visualize
            title (str, optional): Plot title. Auto-generated if None.
            output_file (str, optional): Path to save figure
            crystal_size (float): Size of crystal squares (in mm)
            figsize (tuple): Figure size
            show_grid (bool): Whether to show grid
            show_pipes (bool): Whether to overlay calibration pipes
        
        Returns:
            plt.Figure or None: The figure object if successful
        """
        try:
            if disk_id not in self.crystals_by_disk:
                self.logger.log(f"Disk {disk_id} not found in map data", "error")
                return None
            
            disk_data = self.crystals_by_disk[disk_id]
            
            fig, ax = plt.subplots(figsize=figsize)
            
            # Plot crystals as squares
            for idx, (_, row) in enumerate(disk_data.iterrows()):
                xcry = row['xcry']
                ycry = row['ycry']
                cryID = int(row['cryID'])
                
                # Create square patch centered at (xcry, ycry)
                # crystal_size is the width/height of the square in mm
                square = patches.Rectangle(
                    (xcry - crystal_size/2, ycry - crystal_size/2),
                    crystal_size,
                    crystal_size,
                    linewidth=1,
                    edgecolor='black',
                    facecolor='lightblue',
                    alpha=0.7
                )
                ax.add_patch(square)
                
                # Add cryID label
                ax.text(
                    xcry, ycry,
                    str(cryID),
                    fontsize=7,
                    ha='center',
                    va='center',
                    fontweight='bold'
                )
            
            # Optionally overlay pipes
            if show_pipes:
                self._draw_pipes(ax, pipe_color='red', pipe_alpha=0.3, label_pipes=True)
            
            # Formatting
            ax.set_xlabel("X Position (mm)", fontsize=12)
            ax.set_ylabel("Y Position (mm)", fontsize=12)
            
            if title is None:
                title = f"Disk {disk_id} ({len(disk_data)} CAL crystals)"
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_aspect('equal')
            
            if show_grid:
                ax.grid(True, alpha=0.3)
            
            # Auto-scale axes
            all_x = disk_data['xcry'].values
            all_y = disk_data['ycry'].values
            if len(all_x) > 0:
                margin = 50
                ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
                ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
            
            plt.tight_layout()
            
            if output_file:
                fig.savefig(output_file, dpi=300, bbox_inches='tight')
                self.logger.log(f"Saved visualization to {output_file}", "success")
            
            return fig
        
        except Exception as e:
            self.logger.log(f"Error in visualize_disk: {e}", "error")
            return None
    
    def get_map_info(self) -> Dict:
        """Get summary information about the loaded map."""
        if self.data is None:
            return {}
        
        info = {
            "total_crystals": len(self.data),
            "n_disks": len(self.crystals_by_disk),
            "disks": {}
        }
        
        for disk_id, disk_data in self.crystals_by_disk.items():
            info["disks"][disk_id] = len(disk_data)
        
        return info
    
    def get_pipe_geometry(self) -> PipeGeometry:
        """Get the current pipe geometry configuration."""
        return self.pipe_geometry
    
    def set_pipe_geometry(self, pipe_geometry: PipeGeometry) -> None:
        """
        Update the pipe geometry configuration.
        
        Args:
            pipe_geometry (PipeGeometry): New pipe geometry configuration
        """
        self.pipe_geometry = pipe_geometry
        self.logger.log(f"Updated pipe geometry with {len(self.pipe_geometry.pipe_tor_radii)} pipes", "info")
    
    def print_map_info(self) -> None:
        """Print summary information about the loaded map."""
        info = self.get_map_info()
        self.logger.log(f"Map Information", "info")
        self.logger.log(f"  Total crystals: {info.get('total_crystals', 0)}", "info")
        self.logger.log(f"  Number of disks: {info.get('n_disks', 0)}", "info")
        for disk_id, n_crys in info.get("disks", {}).items():
            self.logger.log(f"  Disk {disk_id}: {n_crys} crystals", "info")
    
    def _calculate_circle_square_intersection(
        self,
        circle_x: float,
        circle_y: float,
        circle_r: float,
        square_x: float,
        square_y: float,
        square_hw: float
    ) -> Tuple[float, float, float]:
        """
        Calculate intersection between a circle and square at 2D plane using pure numpy.
        
        Returns: (overlap_area, cog_x, cog_y)
        - overlap_area: Area of intersection (0 if no overlap)
        - cog_x, cog_y: Center of gravity of intersection region
        
        Uses geometric calculation with monte carlo fallback for robustness.
        """
        # Quick distance check
        dx = circle_x - square_x
        dy = circle_y - square_y
        dist_sq = dx**2 + dy**2
        
        # No intersection if circle center is too far
        max_dist_sq = (circle_r + square_hw * np.sqrt(2))**2
        if dist_sq > max_dist_sq:
            return 0.0, circle_x, circle_y
        
        # Try geometric approach first
        vertices = []
        eps = 1e-9
        
        # 1. Add square corners that are inside the circle
        corners = [
            (square_x - square_hw, square_y - square_hw),
            (square_x + square_hw, square_y - square_hw),
            (square_x + square_hw, square_y + square_hw),
            (square_x - square_hw, square_y + square_hw),
        ]
        for cx_cor, cy_cor in corners:
            dist_sq_cor = (cx_cor - circle_x)**2 + (cy_cor - circle_y)**2
            if dist_sq_cor <= circle_r**2 + eps:
                vertices.append((cx_cor, cy_cor))
        
        # 2. Find circle-edge intersections for all 4 square edges
        edges = [
            ((square_x - square_hw, square_y - square_hw), (square_x + square_hw, square_y - square_hw)),
            ((square_x + square_hw, square_y - square_hw), (square_x + square_hw, square_y + square_hw)),
            ((square_x + square_hw, square_y + square_hw), (square_x - square_hw, square_y + square_hw)),
            ((square_x - square_hw, square_y + square_hw), (square_x - square_hw, square_y - square_hw)),
        ]
        
        for (x1, y1), (x2, y2) in edges:
            dx_seg = x2 - x1
            dy_seg = y2 - y1
            fx = x1 - circle_x
            fy = y1 - circle_y
            
            a = dx_seg**2 + dy_seg**2
            if abs(a) < eps:
                continue
            
            b = 2.0 * (fx * dx_seg + fy * dy_seg)
            c = fx**2 + fy**2 - circle_r**2
            
            discriminant = b**2 - 4.0*a*c
            if discriminant < 0:
                continue
            
            sqrt_disc = np.sqrt(max(0, discriminant))
            t1 = (-b - sqrt_disc) / (2.0*a)
            t2 = (-b + sqrt_disc) / (2.0*a)
            
            for t in [t1, t2]:
                if -eps <= t <= 1.0 + eps:
                    int_x = x1 + t * dx_seg
                    int_y = y1 + t * dy_seg
                    vertices.append((int_x, int_y))
        
        # Remove duplicates
        unique_vertices = []
        for v in vertices:
            is_dup = False
            for u in unique_vertices:
                if abs(v[0] - u[0]) < eps and abs(v[1] - u[1]) < eps:
                    is_dup = True
                    break
            if not is_dup:
                unique_vertices.append(v)
        
        # Try geometric approach if we have enough vertices
        if len(unique_vertices) >= 3:
            vertices = unique_vertices
            
            # Sort vertices counter-clockwise by angle from their centroid
            centroid_x = np.mean([v[0] for v in vertices])
            centroid_y = np.mean([v[1] for v in vertices])
            
            def angle_key(v):
                return np.arctan2(v[1] - centroid_y, v[0] - centroid_x)
            
            vertices.sort(key=angle_key)
            
            # Compute area and centroid using shoelace formula
            area = 0.0
            cog_x = 0.0
            cog_y = 0.0
            
            n = len(vertices)
            for i in range(n):
                x1, y1 = vertices[i]
                x2, y2 = vertices[(i + 1) % n]
                cross = x1 * y2 - x2 * y1
                area += cross
                cog_x += (x1 + x2) * cross
                cog_y += (y1 + y2) * cross
            
            area = abs(area) / 2.0
            
            if area > eps:
                cog_x = cog_x / (6.0 * area)
                cog_y = cog_y / (6.0 * area)
                return area, cog_x, cog_y
        
        # Fallback: Monte Carlo sampling for robustness
        # Sample points uniformly in the square and check if they're in the circle
        crystal_area = (2.0 * square_hw) ** 2
        n_samples = 10000
        
        inside_count = 0
        cog_x_sum = 0.0
        cog_y_sum = 0.0
        
        np.random.seed(42)  # For reproducibility
        for _ in range(n_samples):
            # Random point in square
            px = square_x + (np.random.random() - 0.5) * 2.0 * square_hw
            py = square_y + (np.random.random() - 0.5) * 2.0 * square_hw
            
            # Check if inside circle
            dist_to_center = np.sqrt((px - circle_x)**2 + (py - circle_y)**2)
            if dist_to_center <= circle_r:
                inside_count += 1
                cog_x_sum += px
                cog_y_sum += py
        
        if inside_count > 0:
            fraction = inside_count / n_samples
            area = fraction * crystal_area
            cog_x = cog_x_sum / inside_count
            cog_y = cog_y_sum / inside_count
            return area, cog_x, cog_y
        
        return 0.0, circle_x, circle_y
    
    def print_pipe_info(self) -> None:
        """Print summary information about the pipe geometry."""
        self.logger.log(f"Pipe Geometry Information", "info")
        self.logger.log(f"  Pipe radius: {self.pipe_geometry.pipe_radius:.2f} mm", "info")
        self.logger.log(f"  Number of pipes: {len(self.pipe_geometry.pipe_tor_radii)}", "info")
        for pipe_idx, (rad, phi_lg) in enumerate(zip(self.pipe_geometry.pipe_tor_radii, self.pipe_geometry.large_tor_phi)):
            self.logger.log(f"  Pipe {pipe_idx}: large torus radius={rad:.2f} mm, phi={phi_lg:.2f}°", "info")
    
    def _find_closest_point_on_pipe_to_crystal(
        self,
        geom: PipeGeometry,
        sign_x: float,
        sign_y: float,
        pipe_idx: int,
        cx: float,
        cy: float
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Find the closest point on a pipe 3D curve to a crystal center.
        Returns: (x_closest, y_closest, distance) or (None, None, None) if too far.
        """
        min_dist = float('inf')
        best_x = None
        best_y = None
        
        rad_lg = geom.pipe_tor_radii[pipe_idx]
        phi_lbd = geom.large_tor_phi[pipe_idx]
        phi_sbd = geom.small_tor_phi[pipe_idx]
        phi_end = geom.phi_end[pipe_idx]
        y_pos = geom.yposition[pipe_idx] if pipe_idx < len(geom.yposition) else geom.ysmall[pipe_idx]
        
        # SECTION 1: Sample large torus arc at multiple Y points (increased from 20)
        theta_range = np.radians(phi_lbd / 2)
        for theta in np.linspace(0, theta_range, 50):
            y_arc = sign_y * rad_lg * np.sin(theta)
            x_arc = sign_x * rad_lg * np.cos(theta)
            dist = np.sqrt((x_arc - cx)**2 + (y_arc - cy)**2)
            if dist < min_dist:
                min_dist = dist
                best_x = x_arc
                best_y = y_arc
        
        # SECTION 2: Sample small torus arc at multiple Y points (increased from 20)
        x_sm_center = geom.xsmall + geom.xdistance * pipe_idx
        phi_sm_start = 180.0 + phi_lbd / 2.0 - phi_sbd
        phi_sm_end = 180.0 + phi_lbd / 2.0
        for theta in np.linspace(np.radians(phi_sm_start), np.radians(phi_sm_end), 50):
            y_arc = sign_y * (y_pos + geom.rad_smtor * np.sin(theta))
            x_arc = sign_x * (x_sm_center + geom.rad_smtor * np.cos(theta))
            dist = np.sqrt((x_arc - cx)**2 + (y_arc - cy)**2)
            if dist < min_dist:
                min_dist = dist
                best_x = x_arc
                best_y = y_arc
        
        # SECTION 3: Sample straight section at multiple Y points (increased from 20)
        y_manifold = sign_y * geom.rinner_manifold * np.sin(np.radians(90.0 - phi_end))
        x_start = sign_x * (geom.xsmall + geom.xdistance * pipe_idx - geom.rad_smtor * np.cos(np.radians(phi_end)))
        y_start = sign_y * (y_pos + geom.rad_smtor * np.sin(np.radians(phi_end)))
        x_end = sign_x * (geom.xsmall + geom.xdistance * pipe_idx)
        y_end = y_manifold
        
        for t in np.linspace(0, 1, 50):
            x_arc = x_start + t * (x_end - x_start)
            y_arc = y_start + t * (y_end - y_start)
            dist = np.sqrt((x_arc - cx)**2 + (y_arc - cy)**2)
            if dist < min_dist:
                min_dist = dist
                best_x = x_arc
                best_y = y_arc
        
        # Only return if reasonably close (within 150mm for broad search)
        if min_dist < 150.0:
            return best_x, best_y, min_dist
        return None, None, None

    def export_crystal_pipe_analysis(
        self,
        output_file: str = "crystal_pipe_analysis.csv",
        disk_id: int = 0,
        crystal_size: float = 34.0,
    ) -> None:
        """
        Export CSV with crystal positions and pipe proximity/overlap analysis.
        
        For each crystal on the specified disk, computes:
        - Crystal number (cryID)
        - Center of gravity (COG) X and Y positions
        - Closest point on each pipe to crystal center in 3D
        - Estimated pipe-crystal overlap area
        
        Args:
            output_file (str): Output CSV filename (default: "crystal_pipe_analysis.csv")
            disk_id (int): Disk ID to analyze (default: 0)
            crystal_size (float): Crystal size in mm (default: 34.0)
        """
        import csv
        
        if disk_id not in self.crystals_by_disk:
            self.logger.log(f"Disk {disk_id} not found", "error")
            return
        
        disk_crystals = self.crystals_by_disk[disk_id]
        geom = self.pipe_geometry
        all_rows = []
        
        # Process each crystal
        for _, crystal_row in disk_crystals.iterrows():
            cryid = int(crystal_row['cryID'])
            cx = float(crystal_row['xcry'])
            cy = float(crystal_row['ycry'])
            
            hw = crystal_size / 2.0  # Half-width of crystal square
            pipe_r = geom.pipe_radius
            
            # Track all pipes by distance to crystal
            pipe_candidates = []  # List of (distance, x_pipe, y_pipe, overlap_area, cog_x, cog_y, is_overlapping)
            
            # Check all pipe instances (all quadrants and pipe indices)
            for sign_x in [-1.0, 1.0]:
                for sign_y in [-1.0, 1.0]:
                    for pipe_idx in range(len(geom.pipe_tor_radii)):
                        # Find closest point on this pipe to the crystal
                        x_pipe, y_pipe, dist = self._find_closest_point_on_pipe_to_crystal(
                            geom, sign_x, sign_y, pipe_idx, cx, cy
                        )
                        
                        if x_pipe is None or dist is None:
                            continue
                        
                        # Calculate intersection
                        overlap_area, cog_x, cog_y = self._calculate_circle_square_intersection(
                            x_pipe, y_pipe, pipe_r, cx, cy, hw
                        )
                        
                        is_overlapping = overlap_area > 0
                        pipe_candidates.append((dist, x_pipe, y_pipe, overlap_area, cog_x, cog_y, is_overlapping))
            
            # Sort by: overlapping first (False < True in sort), then by distance
            pipe_candidates.sort(key=lambda x: (not x[6], x[0]))
            
            # Use closest pipe
            min_distance = -1.0
            best_x_pipe = 0.0
            best_y_pipe = 0.0
            max_overlap_area = 0.0
            cog_x_overlap = 0.0
            cog_y_overlap = 0.0
            
            if pipe_candidates:
                dist, x_pipe, y_pipe, overlap_area, cog_x, cog_y, is_overlapping = pipe_candidates[0]
                min_distance = dist
                best_x_pipe = x_pipe
                best_y_pipe = y_pipe
                max_overlap_area = overlap_area
                cog_x_overlap = cog_x
                cog_y_overlap = cog_y
            
            all_rows.append([
                cryid,
                f'{cx:.2f}', f'{cy:.2f}',
                f'{best_x_pipe:.2f}', f'{best_y_pipe:.2f}',
                f'{cog_x_overlap:.2f}', f'{cog_y_overlap:.2f}',
                f'{min_distance:.2f}',
                f'{max_overlap_area:.4f}'
            ])
        
        # Write CSV
        try:
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'cryID', 
                    'COG_X_mm', 'COG_Y_mm',
                    'Pipe_X_mm', 'Pipe_Y_mm',
                    'Overlap_COG_X_mm', 'Overlap_COG_Y_mm',
                    'Min_Pipe_Distance_mm',
                    'Overlap_Area_mm2'
                ])
                writer.writerows(all_rows)
            
            self.logger.log(f"Exported crystal-pipe analysis to {output_file}", "success")
            self.logger.log(f"  Analyzed {len(disk_crystals)} crystals on disk {disk_id}", "info")
        except Exception as e:
            self.logger.log(f"Error exporting crystal-pipe analysis: {e}", "error")


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
        # Store crystal map data when loaded
        self.crystals_by_disk = {}
        self.crystal_map_data = None
        self.pipe_geometry = None
        self.logger.log("Initialised CaloVisualization", "info")
    
    def load_crystal_map(self):
        """
        Load crystal map from caloDMAP_nominal.dat file directly into crystals_by_disk.
        
        Returns:
            bool: True if loading succeeds, False otherwise
        
        Example:
            calo_vis = CaloVisualization()
            if calo_vis.load_crystal_map():
                print(f"Loaded {len(calo_vis.crystals_by_disk)} disks")
        """
        try:
            # Use local caloDMAP_nominal.dat file
            module_dir = os.path.dirname(__file__)
            map_file = os.path.join(module_dir, 'configs/caloDMAP_nominal.dat')
            
            if not os.path.exists(map_file):
                self.logger.log(f"caloDMAP_nominal.dat not found at {map_file}", "error")
                return False
            
            self.logger.log(f"Loading crystal map from {map_file}", "info")
            
            # Read the map file with whitespace delimiter
            self.crystal_map_data = pd.read_csv(
                map_file,
                sep='\s+',
                dtype={
                    'y': int,
                    'x': int,
                    'disk': int,
                    'xcry': float,
                    'ycry': float,
                    'cryID': int,
                    'Type': str
                }
            )
            
            # Filter to only CAL (calorimeter) crystals and valid disks
            self.crystal_map_data = self.crystal_map_data[(self.crystal_map_data['Type'] == 'CAL') & (self.crystal_map_data['disk'] >= 0)].copy()
            
            # Organize crystals by disk
            for disk in sorted(self.crystal_map_data['disk'].unique()):
                disk_data = self.crystal_map_data[self.crystal_map_data['disk'] == disk].copy()
                self.crystals_by_disk[int(disk)] = disk_data
            
            self.logger.log(
                f"Loaded {len(self.crystal_map_data)} CAL crystals from {len(self.crystals_by_disk)} disks",
                "info"
            )
            
            for disk_id, disk_crystals in self.crystals_by_disk.items():
                self.logger.log(f"  Disk {disk_id}: {len(disk_crystals)} crystals", "info")
            
            # Initialize PipeGeometry if not already done
            if self.pipe_geometry is None:
                self.pipe_geometry = PipeGeometry()
                self.logger.log(f"Initialized pipe geometry with {len(self.pipe_geometry.pipe_tor_radii)} pipes", "info")
            
            return True
        
        except Exception as e:
            self.logger.log(f"Error loading crystal map: {e}", "error")
            import traceback
            self.logger.log(traceback.format_exc(), "debug")
            return False
    
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
    


