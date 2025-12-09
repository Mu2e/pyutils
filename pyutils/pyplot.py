#! /usr/bin/env python
import os
import awkward as ak
import matplotlib.pyplot as plt
import math
import numpy as np
from scipy import stats
from matplotlib.ticker import ScalarFormatter
import matplotlib.colors as colors

from .pylogger import Logger

class Plot:
    """ 
    Methods for creating various types of plots. It also includes methods 
    for statistical analysis, automatic formatting, and scientific notation handling.
      
    """

    def __init__(self, style_path=None, verbosity=1):
        """
        Initialise the Plot class.
        
        Args:
            style_path (str, opt): Path to matplotlib style file. (Default: Mu2e style)
            verbosity (int, opt): Level of output detail (0: errors only, 1: info & warnings, 2: max)
        """
        self.style_path = style_path
        self.verbosity = verbosity 
        
        if self.style_path is None:
            self.style_path = os.path.join(os.path.dirname(__file__), "mu2e.mplstyle")                   
        plt.style.use(self.style_path)

        self.logger = Logger( # Start logger
            print_prefix = "[pyplot]", 
            verbosity = verbosity
        )

        self.logger.log(f"Initialised Plot with {self.style_path.rsplit("/", 1)[-1]} and verbosity = {self.verbosity}", "info")

    def round_to_sig_fig(self, val, sf): 
        """
        Round a value to a specified number of significant figures.
        
        Args:
            val (float): Value to round
            sf (int): Number of significant figures
            
        Returns:
            float: Rounded value
            
        Note:
            Returns original value for 0 or NaN inputs
        """
        if val == 0 or math.isnan(val): # Edge cases
            return val
    
        # Determine the order of magnitude
        mag = math.floor(math.log10(abs(val))) 
        # Calculate the scale factor
        scale = 10 ** (sf - mag - 1)
        # Round to the nearest number of significant figures
        return round(val * scale) / scale

    def get_stats(self, array, xmin, xmax): 
        """
        Calculate "stat box" statistics from a 1D array.
        
        Args:
          array (np.ndarray): Input array
          xmin (float): Minimum x-axis value
          xmax (float): Maximum x-axis value
            
        Returns:
          tuple: (n_entries, mean, mean_err, std_dev, std_dev_err, underflows, overflows)
        """
        array = ak.to_numpy(array) # Ensure numpy array
        array = array[~np.isnan(array) & ~np.isinf(array)] # Filter out NaN values
        n_entries = len(array) # Number of entries
        if n_entries == 0:
            return 0, np.nan, np.nan, np.nan, np.nan, 0, 0
        mean = np.mean(array) # Mean
        mean_err = stats.sem(array) # Mean error (standard error on the mean from scipy)
        std_dev = np.std(array) # Standard deviation
        std_dev_err = np.sqrt(std_dev**2 / (2*n_entries)) # Standard deviation error assuming normal distribution
        underflows = len(array[array < xmin]) # Number of underflows
        overflows = len(array[array > xmax]) # Number of overflows
        return n_entries, mean, mean_err, std_dev, std_dev_err, underflows, overflows

    def _scientific_notation(self, ax, lower_limit=1e-3, upper_limit=1e4, cbar=None): 
        """
        Set scientific notation on axes when appropriate.
        
        Args:
          ax (plt.Axes): Matplotlib axes object
          lower_limit (float): Lower threshold for scientific notation
          upper_limit (float): Upper threshold for scientific notation
          cbar (plt.colorbar.Colorbar, optional): Colorbar object
            
        Note:
          Scientific notation is applied when values are outside the specified limits
        """
        # Convert limits to scilimits format (powers of 10)
        lower_exp = int(np.log10(lower_limit))
        upper_exp = int(np.log10(upper_limit))
        scilimits = (lower_exp, upper_exp)
        
        # Configure x-axis
        if ax.get_xscale() != "log": 
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci", axis="x", scilimits=scilimits)
        
        # Configure y-axis
        if ax.get_yscale() != "log": 
            ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci", axis="y", scilimits=scilimits)
        
        # Configure colourbar
        if cbar is not None: 
            # Get the colorbar range
            cmin, cmax = cbar.norm.vmin, cbar.norm.vmax
            
            # Check if any values are outside the limits
            if (abs(cmax) >= upper_limit or abs(cmax) <= lower_limit or 
                abs(cmin) >= upper_limit or abs(cmin) <= lower_limit):
                
                cbar.ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
                cbar.ax.ticklabel_format(style="sci", axis="y", scilimits=scilimits)

    def plot_1D(        
        self,
        array,
        nbins=100,
        xmin=-1.0,
        xmax=1.0,
        weights=None,
        title=None,
        xlabel=None,
        ylabel=None,
        col="black",
        leg_pos="best",
        out_path=None,
        dpi=300,
        log_x=False,
        log_y=False,
        norm_by_area=False,
        under_over=False,
        stat_box=True,
        stat_box_errors=False,
        error_bars=False,
        styles=None,  # NEW: Style configuration for the histogram
        ax=None,
        show=True
        ):
        """
        Create a 1D histogram from an array of values.
        
        Args:
          array (np.ndarray): Input data array
          nbins (int, opt): Number of bins. Defaults to 100
          xmin (float, opt): Minimum x-axis value. Defaults to -1.0
          xmax (float, opt): Maximum x-axis value. Defaults to 1.0
          weights (np.ndarray, opt): Weights for each value. Defaults to None
          title (str, opt): Plot title. Defaults to None
          xlabel (str, opt): X-axis label. Defaults to None
          ylabel (str, opt): Y-axis label. Defaults to None
          col (str, opt): Histogram color. Defaults to "black"
          leg_pos (str, opt): Legend position. Defaults to "best"
          out_path (str, opt): Path to save the plot. Defaults to None
          dpi (int, opt): DPI for saved plot. Defaults to 300
          log_x (bool, opt): Use log scale for x-axis. Defaults to False
          log_y (bool, opt): Use log scale for y-axis. Defaults to False
          norm_by_area (bool, opt): Normalise the curve so that the integral is one. Defaults to False
          under_over (bool, opt): Show overflow/underflow stats. Defaults to False
          stat_box (bool, opt): Show statistics box. Defaults to True
          stat_box_errors (bool, opt): Show errors in stats box. Defaults to False
          error_bars (bool, opt): Show error bars on bins. Defaults to False
          styles (Dict, opt): Style configuration. Defaults to None. Can contain:
              - "histtype": "step", "bar", "stepfilled" (default: "step")
              - "color": matplotlib color (overrides col parameter)
              - "alpha": transparency 0-1 (default: 1.0 for step, 0.3 for filled)
              - "linewidth": line width for step histograms (default: 1.5)
              - "linestyle": line style (default: "-")
              - "fill": True/False for step histograms (default: False)
          ax (plt.Axes, opt): External custom axes. Defaults to None
          show (bool, opt): Display the plot. Defaults to True
            
        Raises:
          ValueError: If array is empty or None
        """
        # Input validation
        if array is None or len(array) == 0:
            self.logger.log(f"Empty or None array passed to plot_1D", "error")
            return None
        
        # Create or use provided axes
        new_fig = False
        if ax is None:
            # Create figure and axes
            fig, ax = plt.subplots()
            new_fig = True
    
        # Process style configuration
        style = styles if styles else {}
        histtype = style.get("histtype", "step")
        color = style.get("color", col)  # Use styles color if provided, otherwise use col parameter
        linewidth = style.get("linewidth", 1.5 if histtype == "step" else 1)
        linestyle = style.get("linestyle", "-")
        
        # Set alpha based on histogram type if not specified
        if "alpha" in style:
            alpha = style["alpha"]
        else:
            alpha = 1.0 if histtype == "step" else 0.3
            
        # Set fill parameter
        fill = style.get("fill", histtype != "step")

        # Create the histogram with style parameters
        hist_kwargs = {
            "bins": int(nbins),
            "range": (xmin, xmax),
            "histtype": histtype,
            "density": norm_by_area,
            "weights": weights,
            "alpha": alpha,
            "linewidth": linewidth,
            "linestyle": linestyle
        }
        
        # Add color and fill parameters based on histogram type
        if histtype == "step":
            hist_kwargs["edgecolor"] = color
            hist_kwargs["fill"] = fill
        else:
            hist_kwargs["color"] = color
            
        counts, bin_edges, _ = ax.hist(array, **hist_kwargs)
        
        bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = bin_edges[1] - bin_edges[0]
    
        # Calculate errors
        if weights is None:
            # For unweighted data
            raw_counts, _ = np.histogram(array, bins=int(nbins), range=(xmin, xmax))
            bin_errors = np.sqrt(raw_counts)  # Poisson errors
            if norm_by_area:
                # Scale errors by the same normalization factor as the counts
                total_entries = np.sum(raw_counts)
                if total_entries > 0:
                    bin_errors = bin_errors / (total_entries * bin_width)
        else:
            # For weighted data
            raw_counts, _ = np.histogram(array, bins=int(nbins), range=(xmin, xmax), weights=weights)
            weights_squared, _ = np.histogram(array, bins=int(nbins), range=(xmin, xmax), weights=np.square(weights))
            bin_errors = np.sqrt(weights_squared)
            if norm_by_area:
                # Scale errors by the same normalization factor
                total_weight = np.sum(raw_counts)
                if total_weight > 0:
                    bin_errors = bin_errors / (total_weight * bin_width)
        
        # Add error bars if requested
        if error_bars:
            ax.errorbar(bin_centres, counts, yerr=bin_errors, ecolor=color, fmt=".", 
                       color=color, capsize=2, elinewidth=1)
    
        # Set x-axis limits
        ax.set_xlim(xmin, xmax)
        
        # Log scale 
        if log_x: 
            ax.set_xscale("log")
        if log_y: 
            ax.set_yscale("log")
      
        # Statistics (use original unnormalized data for stats)
        N, mean, mean_err, std_dev, std_dev_err, underflows, overflows = self.get_stats(array, xmin, xmax)
    
        # Create legend text (imitating the ROOT statbox)
        leg_txt = f"Entries: {N}\nMean: {self.round_to_sig_fig(mean, 3)}\nStd Dev: {self.round_to_sig_fig(std_dev, 3)}"
    
        # Stats box
        if stat_box_errors: 
            leg_txt = f"Entries: {N}\nMean: {self.round_to_sig_fig(mean, 3)}" + rf"$\pm$" + f"{self.round_to_sig_fig(mean_err, 1)}\nStd Dev: {self.round_to_sig_fig(std_dev, 3)}" rf"$\pm$" + f"{self.round_to_sig_fig(std_dev_err, 1)}"
        if under_over: 
            leg_txt += f"\nUnderflows: {underflows}\nOverflows: {overflows}"
    
        # Add legend to the plot
        if stat_box: 
            ax.legend([leg_txt], loc=leg_pos)
    
        # Formatting 
        ax.set_title(title)
        ax.set_xlabel(xlabel) 
        ax.set_ylabel(ylabel) 
    
        # Scientific notation 
        self._scientific_notation(ax)
    
        # If no axis is provided, draw the figure
        if new_fig:
            # Draw
            plt.tight_layout()
            
        # Save
        if out_path:
            plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
            self.logger.log(f"Wrote:\n\t{out_path}", "success")
            
        # Show 
        if show: 
            plt.show()
        
    def plot_1D_overlay(
        self,
        hists_dict, 
        weights=None, 
        nbins=100,
        xmin=-1.0,
        xmax=1.0,
        title=None,
        xlabel=None,
        ylabel=None,
        out_path=None,
        dpi=300,
        leg=True,
        leg_pos="best",
        log_x=False,
        log_y=False,
        norm_by_area=False,
        styles=None,  # Dictionary of styles for each histogram
        ax=None,
        show=True
        ):
        """
        Overlay multiple 1D histograms from a dictionary of arrays.
        
        Args:
            hists_dict (Dict[str, np.ndarray]): Dictionary mapping labels to arrays
            weights (List[np.ndarray], opt): List of weight arrays for each histogram. Defaults to None
            nbins (int, opt): Number of bins. Defaults to 100
            xmin (float, opt): Minimum x-axis value. Defaults to -1.0
            xmax (float, opt): Maximum x-axis value. Defaults to 1.0
            title (str, opt): Plot title. Defaults to None
            xlabel (str, opt): X-axis label. Defaults to None
            ylabel (str, opt): Y-axis label. Defaults to None
            out_path (str, opt): Path to save the plot. Defaults to None
            dpi (int, opt): DPI for saved plot. Defaults to 300
            leg (bool, opt): Whether to include legend. Defaults to True
            leg_pos (str, opt): Legend position. Defaults to "best"
            log_x (bool, opt): Use log scale for x-axis. Defaults to False
            log_y (bool, opt): Use log scale for y-axis. Defaults to False
            norm_by_area (bool, opt): Normalize histograms by area. Defaults to False
            styles (Dict[str, Dict], opt): Style configuration for each histogram. Defaults to None
                Keys should match hists_dict keys. Style dict can contain:
                - "histtype": "step", "bar", "stepfilled" (default: "step")
                - "color": matplotlib color (default: auto-assigned)
                - "alpha": transparency 0-1 (default: 1.0 for step, 0.3 for filled)
                - "linewidth": line width for step histograms (default: 1.5)
                - "linestyle": line style (default: "-")
                - "fill": True/False for step histograms (default: False)
            ax (plt.Axes, opt): External custom axes. Defaults to None
            show (bool, opt): Display the plot. Defaults to True
            
        Raises:
            ValueError: If hists_dict is empty or None
            ValueError: If weights length doesn't match number of histograms
            
        Example:
            # Physics-style overlay
            styles = {
                "Reco": {"histtype": "step", "color": "blue", "linewidth": 2},
                "MC truth": {"histtype": "bar", "color": "red", "alpha": 0.3}
            }
            plotter.plot_1D_overlay(data_dict, styles=styles)
        """
        # Input validation
        if not hists_dict:
            self.logger.log("Empty or None histogram dictionary provided", "error")
            return None
        if weights is not None and len(weights) != len(hists_dict):
            self.logger.log("Number of weight arrays does not match the number of histograms", "error")
            return None
            
        # Create or use provided axes
        new_fig = False
        if ax is None:
            fig, ax = plt.subplots()
            new_fig = True
            
        # Iterate over the histograms and plot each one
        for i, (label, hist) in enumerate(hists_dict.items()):
            weight = None if weights is None else weights[i]
            
            # Get style configuration for this histogram
            style = styles.get(label, {}) if styles else {}
            
            # Set default style parameters
            histtype = style.get("histtype", "step")
            color = style.get("color", None)  # Let matplotlib use style file defaults
            linewidth = style.get("linewidth", 1.5 if histtype == "step" else 1)
            linestyle = style.get("linestyle", "-")
            
            # Set alpha based on histogram type if not specified
            if "alpha" in style:
                alpha = style["alpha"]
            else:
                alpha = 1.0 if histtype == "step" else 0.3
                
            # Set fill parameter
            fill = style.get("fill", histtype != "step")
            
            # Plot the histogram
            hist_kwargs = {
                "bins": nbins,
                "range": (xmin, xmax),
                "histtype": histtype,
                "fill": fill,
                "density": norm_by_area,
                "label": label,
                "weights": weight,
                "alpha": alpha,
                "linewidth": linewidth,
                "linestyle": linestyle
            }
            
            # Only add color if explicitly specified
            if color is not None:
                hist_kwargs["color"] = color
                
            ax.hist(hist, **hist_kwargs)
        
        # Configure axes scales
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
      
        # Set plot limits
        ax.set_xlim(xmin, xmax)
        
        # Format titles
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Set legend
        if leg:
            ax.legend(loc=leg_pos)
        
        # Configure scientific notation
        self._scientific_notation(ax)
    
        # Handle figure if not using external axes
        if new_fig:
            plt.tight_layout()
            
        # Save if output path provided
        if out_path:
            plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
            self.logger.log(f"Wrote:\n\t{out_path}", "success")
            
        # Show 
        if show: 
            plt.show()
        
    def plot_2D(
        self,
        x,
        y,
        weights=None,
        nbins_x=100,
        xmin=-1.0,
        xmax=1.0,
        nbins_y=100,
        ymin=-1.0,
        ymax=1.0,
        title=None,
        xlabel=None,
        ylabel=None,
        zlabel=None,
        out_path=None,
        cmap="inferno",
        dpi=300,
        log_x=False,
        log_y=False,
        log_z=False,
        colorbar=True,
        ax=None,
        show=True
        ):
    
        """
        Plot a 2D histogram from two arrays of the same length.
        
        Args:
            x (np.ndarray): Array of x-values
            y (np.ndarray): Array of y-values 
            weights (np.ndarray, opt): Optional weights for each point. Defaults to None
            nbins_x (int, opt): Number of bins in x. Defaults to 100
            xmin (float, opt): Minimum x value. Defaults to -1.0
            xmax (float, opt): Maximum x value. Defaults to 1.0
            nbins_y (int, opt): Number of bins in y. Defaults to 100
            ymin (float, opt): Minimum y value. Defaults to -1.0
            ymax (float, opt): Maximum y value. Defaults to 1.0
            title (str, opt): Plot title. Defaults to None
            xlabel (str, opt): X-axis label. Defaults to None
            ylabel (str, opt): Y-axis label. Defaults to None
            zlabel (str, opt): Colorbar label. Defaults to None
            out_path (str, opt): Path to save the plot. Defaults to None
            cmap (str, opt): Matplotlib colormap name. Defaults to "inferno"
            dpi (int, opt): DPI for saved plot. Defaults to 300
            log_x (bool, opt): Use log scale for x-axis. Defaults to False
            log_y (bool, opt): Use log scale for y-axis. Defaults to False
            log_z (bool, opt): Use log scale for color values. Defaults to False
            colorbar (bool, opt): Whether to show colorbar. Defaults to True
            ax (plt.Axes, opt): External custom axes. Defaults to None
            show (bool, opt): Display the plot. Defaults to True
            
        Raises:
            ValueError: If input arrays are empty or different lengths
        """
        # Ensure numpy arrays
        x = ak.to_numpy(x)
        y = ak.to_numpy(y)

        # Filter out empty entries
        valid_indices = [i for i in range(len(x)) if np.any(x[i]) and np.any(y[i])]
        x = [x[i] for i in valid_indices]
        y = [y[i] for i in valid_indices]
    
        if weights is not None:
            weights = [weights[i] for i in valid_indices]

        # Validate inputs
        if len(x) == 0 or len(y) == 0:
            self.logger.log("Input arrays are empty", "error")
            return None
        if len(x) != len(y):
            self.logger.log("Input arrays have different lengths", "error")
            return None
        
        # Create or use provided axes
        new_fig = False
        if ax is None:
            fig, ax = plt.subplots()
            new_fig = True
        
        # Create 2D histogram
        hist, _, _ = np.histogram2d(
            x, y,
            bins=[nbins_x, nbins_y],
            range=[[xmin, xmax], [ymin, ymax]],
            weights=weights
        )
    
        # Set up normalisation
        if log_z:
            norm = colors.LogNorm(vmin=1, vmax=np.max(hist))
        else:
            norm = colors.Normalize(vmin=np.min(hist), vmax=np.max(hist))
        
        # Plot the 2D histogram
        im = ax.imshow(
            hist.T,
            cmap=cmap,
            extent=[xmin, xmax, ymin, ymax],
            aspect="auto",
            origin="lower",
            norm=norm
        )
    
        # Configure axes scales
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")

        # Add colorbar
        cbar = None
        if colorbar:
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label(zlabel)
    
        # Set labels
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    
        # Scientific notation
        self._scientific_notation(ax, cbar=cbar)
    
        if new_fig:
            # Draw
            plt.tight_layout()

        # Save if path provided
        if out_path:
            plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
            self.logger.log(f"Wrote:\n\t{out_path}", "success")

        # Show if requested
        if show:
            plt.show()

    def plot_2D_overlay(
        self,
        x1, y1, x2, y2,
        weights1=None, weights2=None,
        nbins_x=100,
        xmin=-1.0,
        xmax=1.0,
        nbins_y=100,
        ymin=-1.0,
        ymax=1.0,
        title=None,
        xlabel=None,
        ylabel=None,
        zlabel=None,
        out_path=None,
        cmap1="Blues",
        cmap2="Reds", 
        alpha=0.7,
        dpi=300,
        log_x=False,
        log_y=False,
        log_z=False,
        show_cbar=False,
        leg_pos="best",
        ax=None,
        show=True,
        labels=None
        ):
    
        """
        Plot two overlaid 2D histograms from two pairs of arrays with different colourmaps.
        
        Args:
            x1, y1 (np.ndarray): Arrays for first dataset
            x2, y2 (np.ndarray): Arrays for second dataset  
            weights1, weights2 (np.ndarray, opt): Optional weights for each dataset. Defaults to None
            nbins_x (int, opt): Number of bins in x. Defaults to 100
            xmin (float, opt): Minimum x value. Defaults to -1.0
            xmax (float, opt): Maximum x value. Defaults to 1.0
            nbins_y (int, opt): Number of bins in y. Defaults to 100
            ymin (float, opt): Minimum y value. Defaults to -1.0
            ymax (float, opt): Maximum y value. Defaults to 1.0
            title (str, opt): Plot title. Defaults to None
            xlabel (str, opt): X-axis label. Defaults to None
            ylabel (str, opt): Y-axis label. Defaults to None
            zlabel (str, opt): Colourbar label. Defaults to None
            out_path (str, opt): Path to save the plot. Defaults to None
            cmap1 (str, opt): Matplotlib colourmap for first dataset. Defaults to "Blues"
            cmap2 (str, opt): Matplotlib colourmap for second dataset. Defaults to "Reds"
            alpha (float, opt): Transparency level. Defaults to 0.7
            dpi (int, opt): DPI for saved plot. Defaults to 300
            log_x (bool, opt): Use log scale for x-axis. Defaults to False
            log_y (bool, opt): Use log scale for y-axis. Defaults to False
            log_z (bool, opt): Use log scale for colour values. Defaults to False
            show_cbar (bool, opt): Whether to show colourbar. Defaults to False (not recommended for overlays)
            leg_pos (str, opt): Legend position. Defaults to "best"
            ax (plt.Axes, opt): External custom axes. Defaults to None
            show (bool, opt): Display the plot. Defaults to True
            labels (list, opt): Labels for legend ['Dataset1', 'Dataset2']. Defaults to None
            
        Raises:
            ValueError: If input arrays are empty or different lengths
        """
        
        # Process first dataset
        x1 = ak.to_numpy(x1)
        y1 = ak.to_numpy(y1)
        
        # Filter out empty entries for dataset 1
        valid_indices1 = [i for i in range(len(x1)) if np.any(x1[i]) and np.any(y1[i])]
        x1 = [x1[i] for i in valid_indices1]
        y1 = [y1[i] for i in valid_indices1]
    
        if weights1 is not None:
            weights1 = [weights1[i] for i in valid_indices1]
    
        # Process second dataset
        x2 = ak.to_numpy(x2)
        y2 = ak.to_numpy(y2)
        
        # Filter out empty entries for dataset 2
        valid_indices2 = [i for i in range(len(x2)) if np.any(x2[i]) and np.any(y2[i])]
        x2 = [x2[i] for i in valid_indices2]
        y2 = [y2[i] for i in valid_indices2]
    
        if weights2 is not None:
            weights2 = [weights2[i] for i in valid_indices2]
    
        # Validate inputs
        if len(x1) == 0 or len(y1) == 0:
            self.logger.log("First dataset arrays are empty", "error")
            return None
        if len(x1) != len(y1):
            self.logger.log("First dataset arrays have different lengths", "error")
            return None
        if len(x2) == 0 or len(y2) == 0:
            self.logger.log("Second dataset arrays are empty", "error")
            return None
        if len(x2) != len(y2):
            self.logger.log("Second dataset arrays have different lengths", "error")
            return None
        
        # Create or use provided axes
        new_fig = False
        if ax is None:
            fig, ax = plt.subplots()
            new_fig = True
        
        # Create 2D histograms
        hist1, _, _ = np.histogram2d(
            x1, y1,
            bins=[nbins_x, nbins_y],
            range=[[xmin, xmax], [ymin, ymax]],
            weights=weights1
        )
        
        hist2, _, _ = np.histogram2d(
            x2, y2,
            bins=[nbins_x, nbins_y],
            range=[[xmin, xmax], [ymin, ymax]],
            weights=weights2
        )
    
        # Set up normalisation
        if log_z:
            norm1 = colors.LogNorm(vmin=max(1, np.min(hist1[hist1>0])), vmax=np.max(hist1))
            norm2 = colors.LogNorm(vmin=max(1, np.min(hist2[hist2>0])), vmax=np.max(hist2))
        else:
            norm1 = colors.Normalize(vmin=0, vmax=np.max(hist1))
            norm2 = colors.Normalize(vmin=0, vmax=np.max(hist2))
        
        # Plot the first 2D histogram
        im1 = ax.imshow(
            hist1.T,
            cmap=cmap1,
            extent=[xmin, xmax, ymin, ymax],
            aspect="auto",
            origin="lower",
            norm=norm1,
            alpha=1.0 # otherwise it will be washed out
        )
        
        # Overlay the second 2D histogram
        im2 = ax.imshow(
            hist2.T,
            cmap=cmap2,
            extent=[xmin, xmax, ymin, ymax],
            aspect="auto",
            origin="lower", 
            norm=norm2,
            alpha=alpha
        )
    
        # Configure axes scales
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
    
        # Add colourbar (optional, usually not recommended for overlays)
        cbar = None
        if show_cbar:
            # Show colourbar for first dataset only
            cbar = plt.colorbar(im1, ax=ax)
            cbar.set_label(zlabel)
    
        # Set labels
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Add legend if labels provided
        if labels and len(labels) >= 2:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=plt.cm.get_cmap(cmap1)(0.7), alpha=alpha, label=labels[0]),
                Patch(facecolor=plt.cm.get_cmap(cmap2)(0.7), alpha=alpha, label=labels[1])
            ]
            ax.legend(handles=legend_elements, loc=leg_pos)
    
        # Scientific notation
        self._scientific_notation(ax, cbar=cbar)
    
        if new_fig:
            # Draw
            plt.tight_layout()
    
        # Save if path provided
        if out_path:
            plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
            self.logger.log(f"Wrote:\n\t{out_path}", "success")
    
        # Show if requested
        if show:
            plt.show()
            
    def plot_graph(
        self,
        x,
        y,
        xerr=None,
        yerr=None,
        title=None,
        xlabel=None,
        ylabel=None,
        xmin=None,
        xmax=None,
        ymin=None,
        ymax=None,
        col="black",
        linestyle="None",
        out_path=None,
        dpi=300,
        log_x=False,
        log_y=False,
        ax=None,
        show=True
        ):
        """
        Plot a scatter graph with optional error bars.
        
        Args:
          x (np.ndarray): Array of x-values
          y (np.ndarray): Array of y-values
          xerr (np.ndarray, opt): X error bars. Defaults to None
          yerr (np.ndarray, opt): Y error bars. Defaults to None
          title (str, opt): Plot title. Defaults to None
          xlabel (str, opt): X-axis label. Defaults to None
          ylabel (str, opt): Y-axis label. Defaults to None
          xmin (float, opt): Minimum x value. Defaults to None
          xmax (float, opt): Maximum x value. Defaults to None
          ymin (float, opt): Minimum y value. Defaults to None
          ymax (float, opt): Maximum y value. Defaults to None
          col (str, opt): Marker and error bar color. Defaults to "black"
          linestyle (str, opt): Style for connecting lines. Defaults to "None"
          out_path (str, opt): Path to save the plot. Defaults to None
          dpi (int, opt): DPI for saved plot. Defaults to 300
          log_x (bool, opt): Use log scale for x-axis. Defaults to False
          log_y (bool, opt): Use log scale for y-axis. Defaults to False
          ax (plt.Axes, opt): Optional matplotlib axes to plot on. Defaults to None
          show (bool, opt): Whether to display plot. Defaults to True
        
        Raises:
          ValueError: If input arrays have different lengths
        """
        # Input validation
        if len(x) != len(y):
            self.logger.log("Input arrays have different lengths", "error")
            return None

        # Create or use provided axes
        new_fig = False
        if ax is None:
            fig, ax = plt.subplots()
            new_fig = True
        
        # Create graph with error bars
        ax.errorbar(
            x, y,
            xerr=xerr,
            yerr=yerr,
            fmt="o",
            color=col,
            markersize=4,
            ecolor=col,
            capsize=2,
            elinewidth=1,
            linestyle=linestyle,
            linewidth=1
        )

        # Set axis limits if provided
        if xmin is not None or xmax is not None:
            ax.set_xlim(left=xmin, right=xmax)
        if ymin is not None or ymax is not None:
            ax.set_ylim(bottom=ymin, top=ymax)

        # Configure log scales
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
      
        # Set labels
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Scientific notation
        self._scientific_notation(ax)
    
        if new_fig:
            # Draw
            plt.tight_layout()

        # Save if path provided
        if out_path:
            plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
            self.logger.log(f"Wrote:\n\t{out_path}", "success")

        # Show if requested
        if show:
            plt.show()
  
    def plot_graph_overlay(
        self,
        graphs,
        title=None,
        xlabel=None,
        ylabel=None,
        xmin=None,
        xmax=None,
        ymin=None,
        ymax=None,
        leg_pos="best",
        linestyle="None",
        out_path=None,
        log_x=False,
        log_y=False,
        dpi=300,
        ax=None,
        show=True
        ):
        """
        Overlay multiple scatter graphs with optional error bars.
        
        Args:
          graphs (dict): Dictionary of graphs to plot, where each graph is a dictionary:
            {
              "label1": {
                "x": x_array,
                "y": y_array,
                "xerr": xerr_array,  # optional
                "yerr": yerr_array   # optional
              },
              "label2": {...}
            }
          title (str, opt): Plot title. Defaults to None
          xlabel (str, opt): X-axis label. Defaults to None
          ylabel (str, opt): Y-axis label. Defaults to None
          xmin (float, opt): Minimum x value. Defaults to None
          xmax (float, opt): Maximum x value. Defaults to None
          ymin (float, opt): Minimum y value. Defaults to None
          ymax (float, opt): Maximum y value. Defaults to None
          leg_pos (str, opt): Position of legend. Defaults to "best"
          linestyle (str, opt): Style for connecting lines. Defaults to "None"
          out_path (str, opt): Path to save plot. Defaults to None
          log_x (bool, opt): Use log scale for x-axis. Defaults to False
          log_y (bool, opt): Use log scale for y-axis. Defaults to False
          dpi (int, opt): DPI for saved plot. Defaults to 300
          ax (plt.Axes, opt): Optional matplotlib axes to plot on. Defaults to None
          show (bool, opt): Whether to display plot. Defaults to True
            
        Raises:
            ValueError: If any graph data is malformed or arrays have different lengths
        """
        # Create or use provided axes
        new_fig = False
        if ax is None:
            fig, ax = plt.subplots()
            new_fig = True

        # Loop through graphs and plot
        for label, graph_data in graphs.items():
            # Validate graph data
            if not isinstance(graph_data, dict):
                self.logger.log(f"Graph data for {label} must be a dictionary", "error")
                return None
            if "x" not in graph_data or "y" not in graph_data:
                self.logger.log(f"Graph data for {label} must contain 'x' and 'y' arrays", "error")
                return None 
            if len(graph_data["x"]) != len(graph_data["y"]):
                self.logger.log(f"X and Y arrays for {label} must have same length", "error")
                return None 
                
            # Get data
            x = graph_data["x"]
            y = graph_data["y"]
            xerr = graph_data.get("xerr", None)  # Use .get() to handle missing error bars
            yerr = graph_data.get("yerr", None)
            
            # Create this graph
            ax.errorbar(
                x, y,
                yerr=yerr,
                xerr=xerr,
                fmt="o",
                label=label,
                markersize=4,
                capsize=2,
                elinewidth=1,
                linestyle=linestyle,
                linewidth=1
            )

        # Set axis limits if provided
        if xmin is not None or xmax is not None:
            ax.set_xlim(left=xmin, right=xmax)
        if ymin is not None or ymax is not None:
            ax.set_ylim(bottom=ymin, top=ymax)

        # Configure log scales
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
      
        # Set labels
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Scientific notation
        self._scientific_notation(ax)
        
        # Add legend
        ax.legend(loc=leg_pos)
    
        if new_fig:
            # Draw
            plt.tight_layout()

        # Save if path provided
        if out_path:
            plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
            self.logger.log(f"Wrote:\n\t{out_path}", "success")

        # Show 
        if show:
            plt.show()