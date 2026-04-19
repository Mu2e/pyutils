"""
Crystal Visualization Module

This module provides the CrystalMap class for visualizing calorimeter crystals
from the mu2e detector map file. Optionally includes overlay of calibration pipes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import Optional, Tuple, Dict, List
import os

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
        # From calorimeter_CsI_v2.txt line 42
        self.pipe_tor_radii = pipe_tor_radii or [397.0, 457.0, 517.0, 577.0, 637.0]
        
        # From calorimeter_CsI_v2.txt line 43
        self.pipe_radius = pipe_radius
        
        # From calorimeter_CsI_v2.txt line 44
        self.pipe_thickness = pipe_thickness
        
        # From calorimeter_CsI_v2.txt line 45
        self.pipe_init_separation = pipe_init_separation
        
        # From calorimeter_CsI_v2.txt line 47
        self.large_tor_phi = large_tor_phi or [161.34, 149.50, 139.50, 132.07, 125.39]
        
        # From calorimeter_CsI_v2.txt line 48
        self.small_tor_phi = small_tor_phi or [84.63, 85.28, 85.79, 86.20, 86.53]
        
        # From calorimeter_CsI_v2.txt line 49
        self.phi_end = phi_end or [3.96, 10.53, 15.80, 20.16, 23.84]
        
        # From calorimeter_CsI_v2.txt line 50
        self.ysmall = ysmall or [432.2, 480.5, 524.3, 564.7, 602.5]
        
        # From calorimeter_CsI_v2.txt line 51
        self.rad_smtor = rad_smtor
        
        # From calorimeter_CsI_v2.txt line 52-54
        self.xsmall = xsmall
        self.xdistance = xdistance
        self.rinner_manifold = rinner_manifold
        
        # Alias for compatibility (same as ysmall in CaloCalibGun)
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
    
    def print_pipe_info(self) -> None:
        """Print summary information about the pipe geometry."""
        self.logger.log(f"Pipe Geometry Information", "info")
        self.logger.log(f"  Pipe radius: {self.pipe_geometry.pipe_radius:.2f} mm", "info")
        self.logger.log(f"  Number of pipes: {len(self.pipe_geometry.pipe_tor_radii)}", "info")
        for pipe_idx, (rad, phi_lg) in enumerate(zip(self.pipe_geometry.pipe_tor_radii, self.pipe_geometry.large_tor_phi)):
            self.logger.log(f"  Pipe {pipe_idx}: large torus radius={rad:.2f} mm, phi={phi_lg:.2f}°", "info")
