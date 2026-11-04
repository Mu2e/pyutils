#! /usr/bin/env python
"""
Calorimeter Analysis and Visualization Utilities

Classes:
    PipeGeometry: Calibration pipe geometry (matches CaloCalibGun)
    CrystalMap: Crystal map loading and visualization, with pipe overlays and analysis
    CaloAnalysis: Data matching and pipeline tracing utilities
    CaloVisualization: Crystal map drawing and event display utilities
"""

import numpy as np
import pandas as pd
import awkward as ak
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
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

    def pipe_sections(self, idx, sign_x=1.0, sign_y=1.0, n_points=60):
        """
        Centerline of one pipe in one quadrant, as its three sections.

        Matches the geometry in CaloCalibGun_module.cc:
        - Large torus arc: centred on the origin, angle 0 to largeTorPhi/2
        - Small torus arc: centred on (xsmall + xdistance*idx, yposition), angle
          180 + largeTorPhi/2 - smallTorPhi to 180 + largeTorPhi/2
        - Straight pipe: from the small torus, tilted by straightEndPhi from the y axis,
          up to the inner manifold (y = rInnerManifold * cos(straightEndPhi))

        Args:
            idx (int): Pipe index
            sign_x, sign_y (float): Quadrant signs (+1 or -1)
            n_points (int): Number of points per section

        Returns:
            dict: {"large_torus": (x, y), "small_torus": (x, y), "straight": (x, y)} arrays in mm
        """
        phi_lbd = self.large_tor_phi[idx]
        phi_sbd = self.small_tor_phi[idx]
        phi_end = np.radians(self.phi_end[idx])
        rad_lg_tor = self.pipe_tor_radii[idx]
        x_sm_center = self.xsmall + self.xdistance * idx
        y_sm_center = self.yposition[idx]

        # Large torus arc
        theta = np.linspace(0, np.radians(phi_lbd / 2), n_points)
        large = (rad_lg_tor * np.cos(theta), rad_lg_tor * np.sin(theta))

        # Small torus arc
        theta = np.linspace(np.radians(180.0 + phi_lbd / 2 - phi_sbd), np.radians(180.0 + phi_lbd / 2), n_points)
        small = (x_sm_center + self.rad_smtor * np.cos(theta), y_sm_center + self.rad_smtor * np.sin(theta))

        # Straight pipe
        x_start = x_sm_center - self.rad_smtor * np.cos(phi_end)
        y_start = y_sm_center + self.rad_smtor * np.sin(phi_end)
        y_end = self.rinner_manifold * np.cos(phi_end)
        x_end = x_start + (y_end - y_start) * np.tan(phi_end)
        t = np.linspace(0, 1, n_points)
        straight = (x_start + t * (x_end - x_start), y_start + t * (y_end - y_start))

        return {
            name: (sign_x * x, sign_y * y)
            for name, (x, y) in [("large_torus", large), ("small_torus", small), ("straight", straight)]
        }


# Default crystal map packaged with pyutils
DEFAULT_MAP_FILE = os.path.join(os.path.dirname(__file__), "configs", "caloDMAP_nominal.dat")
# CsI crystal face in mm (crystal pitch in the map is 34.3 mm including wrapping)
CRYSTAL_SIZE = 34.0
# Each crystal is read out by two SiPMs, crystal ID = SiPMID // 2 (see Offline CaloSiPMId)
N_SIPM_PER_CRYSTAL = 2
# Pipe quadrants as (sign_x, sign_y), in the same order as the CaloCalibGun loops
PIPE_QUADRANTS = [(-1.0, -1.0), (-1.0, 1.0), (1.0, -1.0), (1.0, 1.0)]
PIPE_SECTION_LABELS = {"large_torus": "Large Torus Arc", "small_torus": "Small Torus Arc", "straight": "Straight Section"}


def _load_crystal_map_file(map_file):
    """
    Read a calorimeter map file.

    The map has one row per SiPM (two per crystal), so rows are reduced to one per crystal.

    Returns:
        tuple: (DataFrame of CAL crystals, dict of disk ID -> DataFrame)
    """
    data = pd.read_csv(
        map_file,
        sep=r'\s+',
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

    # Filter to only CAL (calorimeter) crystals and valid disks, one row per crystal
    data = data[(data['Type'] == 'CAL') & (data['disk'] >= 0)]
    data = data.drop_duplicates(subset=['disk', 'cryID']).copy()

    crystals_by_disk = {int(disk): disk_data.copy() for disk, disk_data in data.groupby('disk')}
    return data, crystals_by_disk


def _draw_crystals(ax, disk_data, crystal_size=CRYSTAL_SIZE, facecolor='lightblue', edgecolor='black',
                   alpha=0.7, linewidth=1, zorder=1, show_labels=True, fontsize=6, fontweight='bold'):
    """
    Draw crystals as squares centred on (xcry, ycry), optionally labelled with cryID.

    Returns:
        int: Number of crystals drawn
    """
    half = crystal_size / 2
    squares = [
        patches.Rectangle((x - half, y - half), crystal_size, crystal_size)
        for x, y in zip(disk_data['xcry'], disk_data['ycry'])
    ]
    ax.add_collection(PatchCollection(squares, facecolor=facecolor, edgecolor=edgecolor,
                                      alpha=alpha, linewidth=linewidth, zorder=zorder))
    if show_labels:
        for x, y, cryid in zip(disk_data['xcry'], disk_data['ycry'], disk_data['cryID']):
            ax.text(x, y, str(int(cryid)), fontsize=fontsize, ha='center', va='center',
                    fontweight=fontweight, color='black', zorder=zorder + 1)
    ax.autoscale_view()
    return len(squares)


def _offset_curve(x, y, offset):
    """Shift a polyline sideways by `offset` (perpendicular to its direction)"""
    dx, dy = np.gradient(x), np.gradient(y)
    norm = np.hypot(dx, dy)
    return x - offset * dy / norm, y + offset * dx / norm


def draw_pipes(ax, pipe_geometry, filled=True, pipe_color='red', pipe_alpha=0.5, label_pipes=False):
    """
    Draw the calibration pipes in all four quadrants.

    Args:
        ax (plt.Axes): Matplotlib axes to draw on
        pipe_geometry (PipeGeometry): Pipe geometry
        filled (bool): Draw the pipe walls (pipe_radius either side of the centerline) if True,
                       otherwise only the centerline
        pipe_color (str): Color for pipes
        pipe_alpha (float): Transparency of pipe elements
        label_pipes (bool): Label each pipe with its index (first quadrant only)
    """
    geom = pipe_geometry
    for quad_idx, (sign_x, sign_y) in enumerate(PIPE_QUADRANTS):
        for idx in range(len(geom.pipe_tor_radii)):
            sections = geom.pipe_sections(idx, sign_x, sign_y)

            for name, (x, y) in sections.items():
                if filled:
                    x_outer, y_outer = _offset_curve(x, y, geom.pipe_radius)
                    x_inner, y_inner = _offset_curve(x, y, -geom.pipe_radius)
                    ax.fill(np.concatenate([x_outer, x_inner[::-1]]), np.concatenate([y_outer, y_inner[::-1]]),
                            color=pipe_color, alpha=pipe_alpha * 0.6,
                            label=PIPE_SECTION_LABELS[name] if quad_idx == 0 and idx == 0 else None)
                    ax.plot(x_outer, y_outer, color=pipe_color, linewidth=1, alpha=pipe_alpha)
                    ax.plot(x_inner, y_inner, color=pipe_color, linewidth=1, alpha=pipe_alpha)
                else:
                    ax.plot(x, y, color=pipe_color, linewidth=2, alpha=pipe_alpha)

            # Label at midpoint of large torus arc (only first quadrant to avoid clutter)
            if label_pipes and quad_idx == 0:
                x_lg, y_lg = sections["large_torus"]
                mid_idx = len(x_lg) // 2
                bbox = dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8,
                            edgecolor=pipe_color, linewidth=1) if filled else None
                ax.text(x_lg[mid_idx], y_lg[mid_idx] - 30, f'Pipe {idx}', fontsize=9, ha='center', va='top',
                        fontweight='bold', color=pipe_color, bbox=bbox)


class CrystalMap:
    """
    Visualization class for mu2e calorimeter crystals from map files.
    
    This class reads a detector map file and visualizes crystals as squares
    with their crystal IDs, positioned according to their x,y coordinates.
    
    Optionally includes overlay of calibration pipes from the CaloCalibGun system.
    
    Parameters:
        map_file (str, optional): Path to the detector map file. Defaults to the packaged caloDMAP_nominal.dat
        verbosity (int): Logger verbosity level (0: errors only, 1: info, 2: debug, 3: deep debug)
        style_path (str, optional): Path to matplotlib style file
        pipe_geometry (PipeGeometry, optional): Pipe geometry parameters. Uses defaults if None.
    """

    def __init__(
        self,
        map_file: Optional[str] = None,
        verbosity: int = 1,
        style_path: Optional[str] = None,
        pipe_geometry: Optional[PipeGeometry] = None,
    ):
        """Initialize CrystalMap instance."""
        self.map_file = map_file or DEFAULT_MAP_FILE
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
        self.logger.log(f"Initialized CrystalMap from {self.map_file} with verbosity={self.verbosity}", "info")
        self.logger.log(f"Pipe geometry initialized with {len(self.pipe_geometry.pipe_tor_radii)} pipes", "info")
    
    def _load_map(self) -> None:
        """Load and parse the detector map file."""
        try:
            self.data, self.crystals_by_disk = _load_crystal_map_file(self.map_file)

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
        """Draw calibration pipes with their wall thickness (see draw_pipes)."""
        draw_pipes(ax, self.pipe_geometry, filled=True, pipe_color=pipe_color,
                   pipe_alpha=pipe_alpha, label_pipes=label_pipes)

    def _draw_pipes_lightweight(
        self,
        ax: plt.Axes,
        pipe_color: str = 'red',
        pipe_alpha: float = 0.5,
        label_pipes: bool = True,
    ) -> None:
        """Draw pipe centerlines only, no fills (see draw_pipes)."""
        draw_pipes(ax, self.pipe_geometry, filled=False, pipe_color=pipe_color,
                   pipe_alpha=pipe_alpha, label_pipes=label_pipes)

    
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
                
                # Plot crystals as squares with their cryID labels
                _draw_crystals(ax, disk_data, crystal_size=crystal_size, fontsize=6)

                # Optionally overlay pipes
                if show_pipes:
                    draw_pipes(ax, self.pipe_geometry, pipe_alpha=0.3, label_pipes=True)
                
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
            fig.tight_layout()
            
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
            
            # Plot crystals as squares with their cryID labels
            _draw_crystals(ax, disk_data, crystal_size=crystal_size, fontsize=7)

            # Optionally overlay pipes
            if show_pipes:
                draw_pipes(ax, self.pipe_geometry, pipe_alpha=0.3, label_pipes=True)
            
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
            
            fig.tight_layout()
            
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
        Find the closest point on a pipe centerline (in the disk x-y plane) to a crystal center.
        Returns: (x_closest, y_closest, distance) or (None, None, None) if too far.
        """
        # Sample all three pipe sections
        sections = geom.pipe_sections(pipe_idx, sign_x, sign_y, n_points=50)
        x = np.concatenate([x_sec for x_sec, _ in sections.values()])
        y = np.concatenate([y_sec for _, y_sec in sections.values()])

        dist = np.hypot(x - cx, y - cy)
        i_min = np.argmin(dist)

        # Only return if reasonably close (within 150mm for broad search)
        if dist[i_min] < 150.0:
            return float(x[i_min]), float(y[i_min]), float(dist[i_min])
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
            
            self.logger.log(f"Created digi->reco mapping for {len(mapping)} events", "max")
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
            
            self.logger.log(f"Created reco->digi+energy mapping for {len(mapping)} events", "max")
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

    The crystal map is held by the object (see load_crystal_map). Methods taking a
    ``crysmap`` argument use it by default; passing a CrystalMap or a crystals_by_disk
    dict is still supported for backwards compatibility.
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

    def load_crystal_map(self, map_file=None):
        """
        Load crystal map into crystals_by_disk.

        Args:
            map_file (str, optional): Path to map file. Defaults to the packaged caloDMAP_nominal.dat

        Returns:
            bool: True if loading succeeds, False otherwise

        Example:
            calo_vis = CaloVisualization()
            if calo_vis.load_crystal_map():
                print(f"Loaded {len(calo_vis.crystals_by_disk)} disks")
        """
        map_file = map_file or DEFAULT_MAP_FILE
        try:
            self.logger.log(f"Loading crystal map from {map_file}", "info")
            self.crystal_map_data, self.crystals_by_disk = _load_crystal_map_file(map_file)

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
            self.logger.log(traceback.format_exc(), "max")
            return False

    def _get_disk_data(self, crysmap, disk):
        """
        Return the crystal DataFrame for one disk, or None if unavailable.

        Uses the loaded map (loading it on first use) unless ``crysmap`` is given,
        which may be a CrystalMap or a crystals_by_disk dict.
        """
        if crysmap is None:
            if not self.crystals_by_disk and not self.load_crystal_map():
                return None
            crystals_by_disk = self.crystals_by_disk
        elif isinstance(crysmap, dict):
            crystals_by_disk = crysmap
        else:
            crystals_by_disk = crysmap.crystals_by_disk

        if disk not in crystals_by_disk:
            self.logger.log(f"Disk {disk} not found in crystal map", "warning")
            return None
        return crystals_by_disk[disk]

    def plot_crystal_grid(self, ax, crysmap=None, disk=0, show_labels=False,
                          alpha=0.7, crystal_color='lightblue', edge_color='black'):
        """
        Draw crystal grid on a matplotlib axes.

        Draws all crystals for a specified disk as square patches.

        Args:
            ax (plt.Axes): Matplotlib axes to draw on
            crysmap (optional): Crystal map to use instead of the loaded one
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
            calo_vis.load_crystal_map()
            n_crys = calo_vis.plot_crystal_grid(ax, disk=0, show_labels=True)
            print(f"Drew {n_crys} crystals")
        """
        try:
            disk_data = self._get_disk_data(crysmap, disk)
            if disk_data is None:
                return 0

            n_drawn = _draw_crystals(ax, disk_data, facecolor=crystal_color, edgecolor=edge_color,
                                     alpha=alpha, zorder=1, show_labels=show_labels, fontsize=5, fontweight='normal')

            self.logger.log(f"Drew {n_drawn} crystals for disk {disk}", "info")
            return n_drawn

        except Exception as e:
            self.logger.log(f"Error plotting crystal grid: {e}", "error")
            return 0

    def highlight_hit_crystals(self, ax, crysmap=None, crystal_ids=None, disk=0,
                               color='yellow', edge_color='orange', linewidth=2,
                               show_labels=True, fontsize=8):
        """
        Highlight specific crystals that were hit and optionally show their IDs.

        Draws bright patches over specified crystals with optional crystal ID labels.

        Args:
            ax (plt.Axes): Matplotlib axes to draw on
            crysmap (optional): Crystal map to use instead of the loaded one
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
            n_high = calo_vis.highlight_hit_crystals(ax, crystal_ids=[100, 101, 102], show_labels=True)
        """
        try:
            if crystal_ids is None:
                self.logger.log("No crystal_ids given", "warning")
                return 0

            disk_data = self._get_disk_data(crysmap, disk)
            if disk_data is None:
                return 0

            crystal_ids = set(int(cid) for cid in crystal_ids)
            self.logger.log(f"highlight_hit_crystals: Looking for {len(crystal_ids)} crystal IDs in disk {disk} (map has {len(disk_data)} total crystals)", "max")

            hit_data = disk_data[disk_data['cryID'].isin(crystal_ids)]
            n_highlighted = _draw_crystals(ax, hit_data, facecolor=color, edgecolor=edge_color, alpha=0.8,
                                           linewidth=linewidth, zorder=50, show_labels=show_labels, fontsize=fontsize)

            self.logger.log(f"highlight_hit_crystals: Successfully drew {n_highlighted}/{len(crystal_ids)} requested crystals", "info")
            return n_highlighted

        except Exception as e:
            self.logger.log(f"Error highlighting crystals: {e}", "error")
            return 0

    def set_2d_axis_limits(self, ax, crysmap=None, disk=0, margin=200):
        """
        Set 2D axis limits to show crystal map with margin.

        Args:
            ax (plt.Axes): Matplotlib axes to configure
            crysmap (optional): Crystal map to use instead of the loaded one
            disk (int): Disk number (default: 0)
            margin (float): Margin around crystals in mm (default: 200)

        Returns:
            None

        Example:
            calo_vis.set_2d_axis_limits(ax, disk=0, margin=150)
        """
        try:
            disk_data = self._get_disk_data(crysmap, disk)
            if disk_data is None or len(disk_data) == 0:
                return

            all_x = disk_data['xcry'].values
            all_y = disk_data['ycry'].values
            ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
            ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
            ax.set_aspect('equal')
            self.logger.log(f"Set axis limits for disk {disk}", "info")

        except Exception as e:
            self.logger.log(f"Error setting axis limits: {e}", "error")

    def find_crystals_by_position(self, crysmap=None, positions_x=None, positions_y=None, disk=0, tolerance=17):
        """
        Find crystal IDs for given (x, y) positions on a disk.

        Each crystal is a 34x34 mm square. A position belongs to a crystal if it falls
        within the crystal's bounds (xcry ± tolerance, ycry ± tolerance).

        Args:
            crysmap (optional): Crystal map to use instead of the loaded one
            positions_x (np.ndarray or list): X coordinates of positions
            positions_y (np.ndarray or list): Y coordinates of positions
            disk (int): Disk number (default: 0)
            tolerance (float): Half-width of crystal in mm (default: 17, for 34x34 mm crystals)

        Returns:
            list: Crystal IDs that contain each position (positions with no match are skipped)

        Example:
            crystal_ids = calo_vis.find_crystals_by_position(positions_x=[100, 150], positions_y=[-50, 75], disk=0)
        """
        try:
            disk_data = self._get_disk_data(crysmap, disk)
            if disk_data is None or positions_x is None or positions_y is None:
                return []

            positions_x = np.array(positions_x, dtype=float)
            positions_y = np.array(positions_y, dtype=float)

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
                    self.logger.log(f"Position ({px:.1f}, {py:.1f}) -> Crystal {cry_id}", "max")
                else:
                    self.logger.log(f"Position ({px:.1f}, {py:.1f}) -> No crystal found", "max")

            self.logger.log(f"Found {len(crystal_ids)} crystals for {len(positions_x)} positions", "info")
            return crystal_ids

        except Exception as e:
            self.logger.log(f"Error finding crystals by position: {e}", "error")
            return []

    def plot_event_display(self, crysmap=None, data=None, evt_idx=0, disk=0, output_file=None,
                              show_digis=True, show_reco_digis=True, show_hits=False,
                              show_clusters=False, show_pipes=False, figsize=(14, 12)):
        """
        Create flexible event display with selectable products highlighted on crystal map.

        Users can choose which products to display (digis, reco digis, pipes).
        Each product type highlights its associated crystals with a distinct color.
        Crystals are identified from the digi SiPM ID (crystal ID = SiPMID // 2).

        Args:
            crysmap (optional): Crystal map to use instead of the loaded one
            data (ak.Array): Flat data array with calodigis.* and calorecodigis.* branches
                            (should be data["calo"] extracted from Processor output)
            evt_idx (int): Event index to display
            disk (int): Disk number to show (default: 0)
            output_file (str): Optional path to save figure
            show_digis (bool): Display digis - yellow crystals with orange edges
            show_reco_digis (bool): Display reconstructed digis - cyan crystals with blue edges
            show_hits (bool): Not implemented yet
            show_clusters (bool): Not implemented yet
            show_pipes (bool): Display calibration pipe geometry - red pipe centerlines (default: False)
            figsize (tuple): Figure size in inches (default: (14, 12))

        Returns:
            tuple: (fig, ax) matplotlib figure and axes

        Example:
            # Extract calo data from nested Processor output
            data_full = processor.process_data(...)  # Returns {"evt": ..., "calo": ...}
            calo_data = data_full["calo"]

            calo_vis = CaloVisualization()
            calo_vis.load_crystal_map()
            fig, ax = calo_vis.plot_event_display(data=calo_data, evt_idx=0,
                                                  show_digis=True, show_reco_digis=True,
                                                  output_file='event.png')
        """
        try:
            if data is None:
                self.logger.log("No data given", "error")
                return None, None

            # Extract digi data - data parameter should be the flat 'calo' portion
            try:
                sipm_ids = ak.to_numpy(ak.ravel(data["calodigis.SiPMID_"][evt_idx])).astype(int)
                digi_disks = ak.to_numpy(ak.ravel(data["calodigis.diskID_"][evt_idx])).astype(int)
            except (KeyError, TypeError) as e:
                self.logger.log(f"Could not extract digi data branches: {e}", "error")
                return None, None

            # Each crystal is read out by two SiPMs (CaloSiPMId::crystal())
            digi_crystal_ids = sipm_ids // N_SIPM_PER_CRYSTAL
            digi_on_disk = (digi_disks == disk)
            n_digis = int(np.sum(digi_on_disk))
            self.logger.log(f"Event {evt_idx}: Found {n_digis}/{len(sipm_ids)} digis on disk {disk}", "info")

            fig, ax = plt.subplots(figsize=figsize)

            # Draw crystal grid with labels
            self.plot_crystal_grid(ax, crysmap, disk=disk, show_labels=True,
                                  alpha=0.5, crystal_color='lightgray', edge_color='darkgray')

            legend_patches = []
            product_info = []

            # Show digis
            if show_digis:
                product_info.append(f'Digis={n_digis}')
                if n_digis > 0:
                    n_high = self.highlight_hit_crystals(ax, crysmap, np.unique(digi_crystal_ids[digi_on_disk]), disk=disk,
                                                         color='yellow', edge_color='orange', linewidth=2,
                                                         show_labels=True, fontsize=8)
                    legend_patches.append(patches.Patch(
                        facecolor='yellow', edgecolor='orange', linewidth=2,
                        label=f'Digis (n={n_digis})'
                    ))
                    self.logger.log(f"Highlighted {n_high} crystals for {n_digis} digis on disk {disk}", "info")

            # Show reco digis, located via the calo digi they were reconstructed from
            if show_reco_digis:
                try:
                    reco_digi_idx = ak.to_numpy(ak.ravel(data["calorecodigis.caloDigiIdx_"][evt_idx])).astype(int)
                    valid = (reco_digi_idx >= 0) & (reco_digi_idx < len(sipm_ids))
                    if not np.all(valid):
                        self.logger.log(f"Event {evt_idx}: {np.sum(~valid)} reco digis have out-of-range caloDigiIdx", "warning")
                    reco_digi_idx = reco_digi_idx[valid]
                    reco_on_disk = reco_digi_idx[digi_disks[reco_digi_idx] == disk]
                    n_reco = len(reco_on_disk)
                    product_info.append(f'Reco={n_reco}')

                    if n_reco > 0:
                        n_high = self.highlight_hit_crystals(ax, crysmap, np.unique(digi_crystal_ids[reco_on_disk]), disk=disk,
                                                             color='cyan', edge_color='blue', linewidth=2,
                                                             show_labels=True, fontsize=8)
                        legend_patches.append(patches.Patch(
                            facecolor='cyan', edgecolor='blue', linewidth=2,
                            label=f'Reco Digis (n={n_reco})'
                        ))
                        self.logger.log(f"Highlighted {n_high} crystals for {n_reco} reco digis on disk {disk}", "info")

                except Exception as e:
                    self.logger.log(f"Could not display reco digis: {e}", "warning")

            if show_hits:
                self.logger.log("show_hits is not implemented yet, ignoring", "warning")
            if show_clusters:
                self.logger.log("show_clusters is not implemented yet, ignoring", "warning")

            # Draw pipes if requested
            if show_pipes:
                draw_pipes(ax, self.pipe_geometry or PipeGeometry(), filled=False)
                legend_patches.append(patches.Patch(
                    facecolor='none', edgecolor='red', linewidth=2, linestyle='-',
                    label='Calibration Pipes'
                ))
                product_info.append('Pipes')
                self.logger.log("Drew calibration pipe geometry", "info")

            # Set axis limits and labels
            self.set_2d_axis_limits(ax, crysmap, disk=disk, margin=300)
            ax.set_xlabel('X (mm)', fontsize=12)
            ax.set_ylabel('Y (mm)', fontsize=12)

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
                         title='Data Products', title_fontsize=11,
                         framealpha=0.95)

            if output_file:
                fig.savefig(output_file, dpi=100, bbox_inches='tight')
                self.logger.log(f"Saved event display to {output_file}", "success")

            return fig, ax

        except Exception as e:
            self.logger.log(f"Error plotting event display: {e}", "error")
            self.logger.log(traceback.format_exc(), "max")
            return None, None

    def _draw_pipes_lightweight(self, ax, pipe_geometry=None, pipe_color='red', pipe_alpha=0.5, label_pipes=False):
        """Draw pipe centerlines without fills (see draw_pipes)."""
        draw_pipes(ax, pipe_geometry or self.pipe_geometry or PipeGeometry(), filled=False,
                   pipe_color=pipe_color, pipe_alpha=pipe_alpha, label_pipes=label_pipes)
