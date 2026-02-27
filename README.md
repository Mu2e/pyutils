# `pyutils`: Mu2e Python utilities

`pyutils` is a suite of tools intended for Python-based analyses of Mu2e data, particularly `EventNtuple`. We use packages available in the standard Mu2e Python environment and provide functionality that will be common to many Mu2e analysis groups. The goal is to minimise the amount of overhead required when setting up an analysis.  

>**Note**: This is a standlone version of `pyutils` which is installed in the Mu2e Python environment v2.0.0. 

## 1. Setting up

### on Mu2e machines using Mu2e supported environment

`pyutils` is designed to work with packages installed in the Mu2e Python environment, which is currently maintained by the Analysis Tools Group Leaders Andy Edmonds and Sophie Middleton along with L4 for Analysis Interfaces, Sam Grant.

To activate the environment, first run `mu2einit` and then **one** `pyenv` command, like

```
mu2einit # or "source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh"
pyenv ana # Setup the current environment
pyenv rootana # Setup the current environment, plus ROOT for pyROOT users 
pyenv ana 1.2.0 # Setup a specific version
pyenv -h # Get help (--help and pyenv with no flag will also return help)
```

>**Note**: `mu2einit` should be aliased to `source /cvmfs/mu2e.opensciencegrid.org/setupmu2e-art.sh` in your `~/.my_bashrc`.

See the [tutorial](https://github.com/Mu2e/Tutorial/blob/main/EAF/Docs/06-TheMu2eEnvironment.md) on GitHub and the [wiki](https://mu2ewiki.fnal.gov/wiki/Elastic_Analysis_Facility_(EAF)#The_Mu2e_environment) page for more information.


### on your own computer/own environment

To install pyutils in your custom environment you can do so using pip

```
pip install git+https://github.com/Mu2e/pyutils.git
```

pyutils should then operate in the same as it does within the Mu2e environment.

## 2. Using `pyutils` 

The suite consists of the following modules.
```python
pyread      # Data reading 
pyprocess   # Listing and parallelisation 
pydask      # Dask-based distributed processing (inherits from pyprocess.py)
pyimport    # TTree (EventNtuple) importing interface 
pyplot      # Plotting and visualisation 
pyprint     # Array visualisation 
pyselect    # Data selection 
pycut       # Cut management 
pyvector    # Element wise wector operations
pymcutil    # Monte Carlo utilities (coming soon)
pylogger    # Helper module for managing printouts
pydisplay   # allows user to call Mu2e/EventDisplay directly through python
```

### 2.1 Demos and tutorials

To learn by example, follow the `pyutils` tutorial series.

1. [pyutils_basics.ipynb](examples/notebooks/pyutils_basics.ipynb) - Introduction to core functionality
1. [pyutils_on_EAF.ipynb](examples/notebooks/pyutils_on_EAF.ipynb) - Reading data with `pyutils` from the Elastic Analysis Facility (EAF) 
1. [pyutils_multifile.ipynb](examples/notebooks/pyutils_multifile.ipynb) - Basic parallelisation with file lists and SAM definitions, as well as complex parallelised analysis tasks using the `pyprocess` `Skeleton` template class.
1. [pyplot_demo.ipynb](examples/notebooks/pyplot_demo.ipynb) - A comprehensive demonstration of the `pyplot.Plot` class.
1. [pycut_demo.ipynb](examples/notebooks/pycut_demo.ipynb) - A comprehensive demonstration of the `pycut.CutManager` class.
1. [pyutils_mctruth.ipynb](examples/notebooks/pyutils_mctruth.ipynb) - Examples of how to use the mc truth interface for simulation based studies.

To open a jupyter notebook on the mu2e machines:

```
jupyter-lab --no-browser #--port 1234 
```

replacing `1234` with a chosen port number. The default is  `8888`. On your local machine create a ssh tunnel:

```
ssh -KXY -L 08888:localhost:08888 <user>@mu2egpvm0<machine>.fnal.gov
```

replacing the port (8888) with your chosen number and the user with your username and mu2e machine ID with that on which you launched jupyter-lab.

To connect to the notebook copy the URL with the unique token printed in the terminal of the mu2e machine into your local browser.

### 2.1a Parallel Processing: `pyprocess` vs `pydask`

`pyutils` currently provides two parallel processing options:

**`pyprocess`** (ThreadPoolExecutor/ProcessPoolExecutor):
- Uses Python's built-in `concurrent.futures` for parallelisation
- Best for: Small to medium datasets, local processing on single machine
- Simpler to use, less overhead
- Limited to resources available on your machine

**`pydask`** (Dask-based):
- Uses Dask distributed computing framework
- Best for: Large datasets, leveraging clusters, EAF integration
- Supports both local and remote Dask schedulers
- Better for scaling to many workers across multiple machines

#### Using `pydask` Locally

```python
from pyutils.pydask import DaskProcessor

# Create processor
dp = DaskProcessor(
    tree_path="EventNtuple/ntuple",
    use_remote=True,
    location="disk"
)

# Process data - automatically creates local Dask cluster
data = dp.process_data(
    file_list_path="files.txt",
    branches=["trksegs"],
    n_workers=4,
    threads_per_worker=1,
    processes=True  # Use processes instead of threads
)
```

This creates a temporary local Dask cluster with 4 workers, processes the files, then shuts down.

#### Using `pydask` on EAF with Remote Scheduler

We are in conversations with EAF team. We are currently not able to use a remote, centralized scheduler. Contact Sophie or Sam for more information on the progress of these conversations.

### 2.2 Module documentation 

Help information be accessed with `help(name)`, where `name` can be the module name, a class name, or a function name. 

---

#### `pyread` 

Contains the `Reader` class: a utility for reading files with uproot (supports local and remote files). Called by `pyprocess`, which is the main interface for processing data.

<details>
<summary>Click for details</summary>
    
```
NAME
    pyread

CLASSES
    builtins.object
        Reader

    class Reader(builtins.object)
     |  Reader(use_remote=False, location='tape', schema='root', verbosity=1)
     |
     |  Unified interface for accessing files, either locally or remotely
     |
     |  Methods defined here:
     |
     |  __init__(self, use_remote=False, location='tape', schema='root', verbosity=1)
     |      Initialise the reader
     |
     |      Args:
     |          use_remote (bool, opt): Whether to use remote access methods
     |          location (str, opt): File location for remote files: tape (default), disk, scratch, nersc
     |          schema (str, opt): Schema for remote file path: root (default), http, path , dcap, samFile
     |          verbosity (int, opt): Level of output detail (0: errors only, 1: info & warnings, 2: max)
     |
     |  read_file(self, file_path)
     |      Read a file using the appropriate method
     |
     |      Args:
     |          file_path: Path to the file
     |
     |      Returns:
     |          Uproot file object
     |
     |  ----------------------------------------------------------------------

```
</details>

---

#### `pyimport`

Contains the `Importer` class: a utility for importing ROOT TTree branches into Awkward arrays. Called by `pyprocess`, which is the main interface for processing data.

<details>
<summary>Click for details</summary>

```
NAME
    pyimport

CLASSES
    builtins.object
        Importer

    class Importer(builtins.object)
     |  Importer(file_name, branches, dir_name='EventNtuple', tree_name='ntuple', use_remote=False, location='tape', schema='root', verbosity=1)
     |
     |  Utility class for importing branches from ROOT TTree files
     |
     |  Intended to used via by the pyprocess Processor class
     |
     |  Methods defined here:
     |
     |  __init__(self, file_name, branches, dir_name='EventNtuple', tree_name='ntuple', use_remote=False, location='tape', schema='root', verbosity=1)
     |      Initialise the importer
     |
     |      Args:
     |          file_name: Name of the file
     |          branches: Flat list or grouped dict of branches to import
     |          dir_name: Ntuple directory in file
     |          tree_name: Ntuple name in file directory
     |          use_remote: Flag for reading remote files
     |          location: Remote files only. File location: tape (default), disk, scratch, nersc
     |          schema: Remote files only. Schema used when writing the URL: root (default), http, path, dcap, samFile
     |          verbosity: Print detail level (0: minimal, 1: medium, 2: maximum)
     |
     |  import_branches(self)
     |      Internal function to open ROOT file and import specified branches
     |
     |      Returns:
     |          Awkward array with imported data
     |
     |  ----------------------------------------------------------------------
```

</details>

---

#### `pyprocess`

**This is the primary interface for processing data**: supports processing of single files, file lists, and SAM definitions. Contains the `Processor` class, which provides methods for producing file lists and parallel processing, where the `process_data` method provides a single entry point for these utilties. It also contains the `Skeleton` class, which provides a template class for performing complex analyses.

<details>
<summary>Click for details</summary>

```
NAME
    pyprocess

CLASSES
    builtins.object
        Processor
        Skeleton

    class Processor(builtins.object)
     |  Processor(dir_name='EventNtuple', tree_name='ntuple', use_remote=False, location='tape', schema='root', verbosity=1)
     |
     |  Interface for processing files or datasets
     |
     |  Methods defined here:
     |
     |  __init__(self, dir_name='EventNtuple', tree_name='ntuple', use_remote=False, location='tape', schema='root', verbosity=1)
     |      Initialise the processor
     |
     |      Args:
     |          dir_name (str, opt): Ntuple directory in file
     |          tree_name (str, opt): Ntuple name in file directory
     |          use_remote (bool, opt): Flag for reading remote files
     |          location (str, opt): Remote files only. File location: tape (default), disk, scratch, nersc
     |          schema (str, opt): Remote files only. Schema used when writing the URL: root (default), http, path, dcap, samFile
     |          verbosity (int, opt): Level of output detail (0: errors only, 1: info, warnings, 2: max)
     |
     |  get_file_list(self, defname=None, file_list_path=None)
     |      Utility to get a list of files from a SAM definition OR a text file
     |
     |      Args:
     |          defname: SAM definition name
     |          file_list_path: Path to a plain text file containing file paths
     |
     |      Returns:
     |          List of file paths
     |
     |  process_data(self, file_name=None, file_list_path=None, defname=None, branches=None, max_workers=None, custom_process_func=None, use_processes=False)
     |      Process the data
     |
     |      Args:
     |          file_name: File name
     |          defname: SAM definition name
     |          file_list_path: Path to file list
     |          branches: Flat list or grouped dict of branches to import
     |          max_workers: Maximum number of parallel workers
     |          custom_process_func: Optional custom processing function for each file
     |          use_processes: Whether to use processes rather than threads
     |
     |      Returns:
     |          - If custom_process_func is None: a concatenated awkward array with imported data from all files
     |          - If custom_process_func is not None: a list of outputs from the custom process
     |
     |  ----------------------------------------------------------------------

    class Skeleton(builtins.object)
     |  Skeleton(verbosity=1)
     |
     |  Template class for creating a custom analysis processor
     |
     |  This template demonstrates how to create a class to run
     |  custom analysis jobs with the Processor framework
     |
     |  To use this skeleton:
     |  1. Either initilaise the entire class or pass it as an argument to your Processor class
     |  2. Customize the __init__ method with your configuration
     |  3. Implement your processing logic in the process method
     |  4. Add any additional helper methods you need
     |  5. Override methods as needed
     |
     |  Methods defined here:
     |
     |  __init__(self, verbosity=1)
     |      Initialise your file processor with configuration parameters
     |
     |      Customise this method to include parameters specific to your analysis.
     |
     |      Args:
     |          verbosity (int, opt): Level of output detail (0: errors only, 1: info, 2: debug, 3: max)
     |
     |  execute(self)
     |      Run the processor on the configured files
     |
     |      Returns:
     |          Combined results from all processed files
     |
     |  process_file(self, file_name)
     |      Process a single file
     |
     |      This is the core method that will be called for each file.
     |      Implement your file processing logic here.
     |
     |      Args:
     |          file_name: Name of the file to process
     |
     |      Returns:
     |          Any data structure representing the processed result
     |
     |  process_results(self)
     |      Run post processing on the results list
     |
     |  ----------------------------------------------------------------------q
```

</details>

---

#### `pyplot`

Tools for creating publication-quality histograms and graphs from flattened arrays. It uses the `mu2e.mplstyle` style file by default, although users can choose to import a custom style file.

>**Note**: Does not support plotting for histogram objects. Working on resolving this.

<details>
<summary>Click for details</summary>

```
     |  Methods defined here:
     |
     |  __init__(self, style_path=None, verbosity=1)
     |      Initialise the Plot class.
     |
     |      Args:
     |          style_path (str, opt): Path to matplotlib style file. (Default: Mu2e style)
     |          verbosity (int, opt): Level of output detail (0: errors only, 1: info & warnings, 2: max)
     |
     |  get_stats(self, array, xmin, xmax)
     |      Calculate 'stat box' statistics from a 1D array.
     |
     |      Args:
     |        array (np.ndarray): Input array
     |        xmin (float): Minimum x-axis value
     |        xmax (float): Maximum x-axis value
     |
     |      Returns:
     |        tuple: (n_entries, mean, mean_err, std_dev, std_dev_err, underflows, overflows)
     |
     |  plot_1D(self, array, nbins=100, xmin=-1.0, xmax=1.0, weights=None, title=None, xlabel=None, ylabel=None, col='black', leg_pos='best', out_path=None, dpi=300, log_x=False, log_y=False, norm_by_area=False, under_over=False, stat_box=True, stat_box_errors=False, error_bars=False, ax=None, show=True)
     |      Create a 1D histogram from an array of values.
     |
     |      Args:
     |        array (np.ndarray): Input data array
     |        weights (np.ndarray, optional): Weights for each value
     |        nbins (int, optional): Number of bins. Defaults to 100
     |        xmin (float, optional): Minimum x-axis value. Defaults to -1.0
     |        xmax (float, optional): Maximum x-axis value. Defaults to 1.0
     |        title (str, optional): Plot title
     |        xlabel (str, optional): X-axis label
     |        ylabel (str, optional): Y-axis label
     |        col (str, optional): Histogram color. Defaults to 'black'
     |        leg_pos (str, optional): Legend position. Defaults to 'best'
     |        out_path (str, optional): Path to save the plot
     |        dpi (int, optional): DPI for saved plot. Defaults to 300
     |        log_x (bool, optional): Use log scale for x-axis. Defaults to False
     |        log_y (bool, optional): Use log scale for y-axis. Defaults to False
     |        under_over (bool, optional): Show overflow/underflow stats. Defaults to False
     |        stat_box (bool, optional): Show statistics box. Defaults to True
     |        stat_box_errors (bool, optional): Show errors in stats box. Defaults to False
     |        error_bars (bool, optional): Show error bars on bins. Defaults to False
     |        ax (plt.Axes, optional): External custom axes
     |        show (bool, optional): Display the plot, defaults to True
     |
     |      Raises:
     |        ValueError: If array is empty or None
     |
     |  plot_1D_overlay(self, hists_dict, weights=None, nbins=100, xmin=-1.0, xmax=1.0, title=None, xlabel=None, ylabel=None, out_path=None, dpi=300, leg_pos='best', log_x=False, log_y=False, norm_by_area=False, ax=None, show=True)
     |      Overlay multiple 1D histograms from a dictionary of arrays.
     |
     |      Args:
     |          hists_dict (Dict[str, np.ndarray]): Dictionary mapping labels to arrays
     |          weights (List[np.ndarray], optional): List of weight arrays for each histogram
     |          nbins (int, optional): Number of bins. Defaults to 100
     |          xmin (float, optional): Minimum x-axis value. Defaults to -1.0
     |          xmax (float, optional): Maximum x-axis value. Defaults to 1.0
     |          title (str, optional): Plot title
     |          xlabel (str, optional): X-axis label
     |          ylabel (str, optional): Y-axis label
     |          out_path (str, optional): Path to save the plot
     |          dpi (int, optional): DPI for saved plot. Defaults to 300
     |          leg_pos (str, optional): Legend position. Defaults to 'best'
     |          log_x (bool, optional): Use log scale for x-axis. Defaults to False
     |          log_y (bool, optional): Use log scale for y-axis. Defaults to False
     |          ax (plt.Axes, optional): External custom axes.
     |          show (bool, optional): Display the plot. Defaults to True
     |
     |      Raises:
     |          ValueError: If hists_dict is empty or None
     |          ValueError: If weights length doesn't match number of histograms
     |
     |  plot_2D(self, x, y, weights=None, nbins_x=100, xmin=-1.0, xmax=1.0, nbins_y=100, ymin=-1.0, ymax=1.0, title=None, xlabel=None, ylabel=None, zlabel=None, out_path=None, cmap='inferno', dpi=300, log_x=False, log_y=False, log_z=False, colorbar=True, ax=None, show=True)
     |      Plot a 2D histogram from two arrays of the same length.
     |
     |      Args:
     |          x (np.ndarray): Array of x-values
     |          y (np.ndarray): Array of y-values
     |          weights (np.ndarray, optional): Optional weights for each point
     |          nbins_x (int): Number of bins in x. Defaults to 100
     |          xmin (float): Minimum x value. Defaults to -1.0
     |          xmax (float): Maximum x value. Defaults to 1.0
     |          nbins_y (int): Number of bins in y. Defaults to 100
     |          ymin (float): Minimum y value. Defaults to -1.0
     |          ymax (float): Maximum y value. Defaults to 1.0
     |          title (str, optional): Plot title
     |          xlabel (str, optional): X-axis label
     |          ylabel (str, optional): Y-axis label
     |          zlabel (str, optional): Colorbar label
     |          out_path (str, optional): Path to save the plot
     |          cmap (str): Matplotlib colormap name. Defaults to 'inferno'
     |          dpi (int): DPI for saved plot. Defaults to 300
     |          log_x (bool): Use log scale for x-axis
     |          log_y (bool): Use log scale for y-axis
     |          log_z (bool): Use log scale for color values
     |          cbar (bool): Whether to show colorbar. Defaults to True
     |          ax (plt.Axes, optional): External custom axes.
     |          show (bool): show (bool, optional): Display the plot. Defaults to True
     |
     |      Raises:
     |          ValueError: If input arrays are empty or different lengths
     |
     |  plot_graph(self, x, y, xerr=None, yerr=None, title=None, xlabel=None, ylabel=None, xmin=None, xmax=None, ymin=None, ymax=None, col='black', linestyle='None', out_path=None, dpi=300, log_x=False, log_y=False, ax=None, show=True)
     |      Plot a scatter graph with optional error bars.
     |
     |      Args:
     |        x (np.ndarray): Array of x-values
     |        y (np.ndarray): Array of y-values
     |        xerr (np.ndarray, optional): X error bars
     |        yerr (np.ndarray, optional): Y error bars
     |        title (str, optional): Plot title
     |        xlabel (str, optional): X-axis label
     |        ylabel (str, optional): Y-axis label
     |        xmin (float, optional): Minimum x value
     |        xmax (float, optional): Maximum x value
     |        ymin (float, optional): Minimum y value
     |        ymax (float, optional): Maximum y value
     |        color (str): Marker and error bar color, defaults to 'black'
     |        linestyle (str): Style for connecting lines, defaults to 'None'
     |        out_path (str, optional): Path to save the plot
     |        dpi (int): DPI for saved plot. Defaults to 300
     |        log_x (bool): Use log scale for x-axis, defaults to False
     |        log_y (bool): Use log scale for y-axis, defaults to False
     |        ax (plt.Axes, optional): Optional matplotlib axes to plot on
     |        show (bool): Whether to display plot, defaults to True
     |
     |      Raises:
     |        ValueError: If input arrays have different lengths
     |
     |  plot_graph_overlay(self, graphs, title=None, xlabel=None, ylabel=None, xmin=None, xmax=None, ymin=None, ymax=None, legend_position='best', linestyle='None', out_path=None, log_x=False, log_y=False, dpi=300, ax=None, show=True)
     |      Overlay multiple scatter graphs with optional error bars.
     |
     |      Args:
     |        graphs (dict): Dictionary of graphs to plot, where each graph is a dictionary:
     |          {
     |            'label1': {
     |              'x': x_array,
     |              'y': y_array,
     |              'xerr': xerr_array,  # optional
     |              'yerr': yerr_array   # optional
     |            },
     |            'label2': {...}
     |          }
     |        title (str, optional): Plot title
     |        xlabel (str, optional): X-axis label
     |        ylabel (str, optional): Y-axis label
     |        xmin (float, optional): Minimum x value
     |        xmax (float, optional): Maximum x value
     |        ymin (float, optional): Minimum y value
     |        ymax (float, optional): Maximum y value
     |        leg_pos (str): Position of legend. Defaults to 'best'
     |        linestyle (str): Style for connecting lines. Defaults to 'None'
     |        out_path (str, optional): Path to save plot
     |        log_x (bool): Use log scale for x-axis, defaults to False
     |        log_y (bool): Use log scale for y-axis, defaults to False
     |        dpi (int): DPI for saved plot, defaults to 300
     |        ax (plt.Axes, optional): Optional matplotlib axes to plot on
     |        show (bool): Whether to display plot. Defaults to True
     |
     |      Raises:
     |          ValueError: If any graph data is malformed or arrays have different lengths
     |
     |  round_to_sig_fig(self, val, sf)
     |      Round a value to a specified number of significant figures.
     |
     |      Args:
     |          val (float): Value to round
     |          sf (int): Number of significant figures
     |
     |      Returns:
     |          float: Rounded value
     |
     |      Note:
     |          Returns original value for 0 or NaN inputs
     |
     |  ----------------------------------------------------------------------
```

</details>

---

#### `pyprint`

For array visualisation, allowing the user to print out the structure of their array per event in a human-readable format

<details>
<summary>Click for details</summary>
    
```
NAME
    pyprint

CLASSES
    builtins.object
        Print

    class Print(builtins.object)
     |  Print(verbose=False, precision=1)
     |
     |  Utility class for printing structured event data in a human-readable format.
     |
     |  This class provides methods to print individual events or multiple events from
     |  an Awkward array, handling nested fields and subfields recursively.
     |
     |  Methods defined here:
     |
     |  __init__(self, verbose=False, precision=1)
     |      Initialise Print
     |
     |      Args:
     |          verbose (bool, optional): Print full arrays without truncation. Defaults to False.
     |          precision (int, optional): Specifiy the number of decimal points when using verbose option. Defaults to 1.
     |
     |  print_event(self, event, prefix='')
     |      Print a single event in human-readable format, including all fields and subfields.
     |
     |      Args:
     |        event (awkward.Array): Event to print, containing fields and possibly subfields
     |        prefix (str, optional): Prefix to prepend to field names. Used for nested fields. Defaults to empty string.
     |
     |      Note:
     |        Recursively handles nested fields, e.g. field.subfield.value
     |
     |  print_n_events(self, array, n_events=1)
     |      Print the first n events from an array in human-readable format.
     |
     |      Args:
     |        array_ (awkward.Array): Array of events to print
     |        n (int, optional): Number of events to print. Defaults to 1.
     |
     |      Note:
     |        Prints a separator line between events for better readability.
     |        Events are numbered starting from 1.
     |
     |      Example:
     |        >>> printer = Print()
     |        >>> printer.PrintNEvents(events, n_events=2)
     |
     |        ---> Printing 2 event(s)...
     |
     |        -------------------------------------------------------------------------------------
     |        field1: value
     |        field2.subfield1: value
     |        -------------------------------------------------------------------------------------
     |
     |        -------------------------------------------------------------------------------------
     |        field1: value
     |        field2.subfield1: value
     |        -------------------------------------------------------------------------------------
     |
     |  ----------------------------------------------------------------------
```

</details>

---

#### `pyselect`

Tools for creating and managing selection cut masks. 


<details>
<summary>Click for details</summary>

```
NAME
    pyselect

CLASSES
    builtins.object
        Select

    class Select(builtins.object)
     |  Select(verbosity=1)
     |
     |  Class for standard selection cuts with EventNtuple data in Awkward format
     |
     |  Methods defined here:
     |
     |  __init__(self, verbosity=1)
     |      Initialise the selector
     |
     |      Args:
     |          verbosity (int, optional): Print detail level (0: minimal, 1: medium, 2: maximum). Defaults to 1.
     |
     |  hasTrkCrvCoincs(self, trks, ntuple, tmax)
     |      simple function to remove anything close to a crv coinc
     |
     |  has_n_hits(self, data, n_hits)
     |      Return boolean array for tracks with hits above a specified value
     |
     |      Hits in this context is nactive planes
     |
     |      Args:
     |          data (awkward.Array): Input array containing the trk.nactive branch
     |          n_hits (int): The minimum number of track hits (nactive)
     |
     |  is_downstream(self, data, branch_name='trksegs')
     |      Return boolean array for upstream track segments
     |
     |      Args:
     |          data (awkward.Array): Input array containing the segments branch
     |          branch_name (str, optional): Name of the segments branch for backwards compatibility. Defaults to 'trksegs'
     |
     |  is_electron(self, data)
     |      Return boolean array for electron tracks which can be used as a mask
     |
     |      Args:
     |          data (awkward.Array): Input array containing the "trk" branch
     |
     |  is_mu_minus(self, data)
     |      Return boolean array for negative muon tracks which can be used as a mask
     |
     |      Args:
     |          data (awkward.Array): Input array containing the "trk" branch
     |
     |  is_mu_plus(self, data)
     |      Return boolean array for positive muon tracks which can be used as a mask
     |
     |      Args:
     |          data (awkward.Array): Input array containing the "trk" branch
     |
     |  is_particle(self, data, particle)
     |      Return boolean array for tracks of a specific particle type which can be used as a mask
     |
     |      Args:
     |          data (awkward.Array): Input array containing the "trk" branch
     |          particle (string): particle type, 'e-', 'e+', 'mu-', or 'mu+'
     |
     |  is_positron(self, data)
     |      Return boolean array for positron tracks which can be used as a mask
     |
     |      Args:
     |          data (awkward.Array): Input array containing the "trk" branch
     |
     |  is_reflected(self, data, branch_name='trksegs')
     |      Return boolean array for reflected tracks
     |
     |      Reflected tracks have both upstream and downstream segments at the tracker entrance
     |
     |      Args:
     |          data (awkward.Array): Input array containing segments branch
     |          branch_name (str, optional): Name of the segments branch for backwards compatibility. Defaults to 'trksegs'
     |
     |  is_upstream(self, data, branch_name='trksegs')
     |      Return boolean array for downstream track segments
     |
     |      Args:
     |          data (awkward.Array): Input array containing the segments branch
     |          branch_name (str, optional): Name of the segments branch for backwards compatibility. Defaults to 'trksegs'
     |
     |  select_surface(self, data, surface_name="TT_Front", sindex=0, branch_name="trksegs"):
     |     Return boolean array for track segments intersecting a specific surface 
     |   
     |   Args:
     |       data (awkward.Array): Input array containing segments branch
     |       surface_name (str) : official name of the intersected surface 
     |       sindex (int, optional): Index to the intersected surface (for multi-surface elements). Defaults to 0. 
     |       branch_name (str, optional): Name of the segments branch for backwards compatibility. Defaults to 'trksegs'
     |   
     |
     |  select_trkqual(self, data, quality)
     |      Return boolean array for tracks above a specified quality
     |
     |      Args:
     |          data (awkward.Array): Input array containing the trkqual.result branch
     |          quality (float): The numerical output of the MVA
     |
     |   has_ST(self, data):
     |      Returns mask True if the event has at least 1 ST viable extrapolation
     | 
     |   has_OPA(self, data):
     |      Returns mask True if the event has at no OPA viable extrapolation
     | 
     |
     |   select_trkpid(self, data, value):
     |     Return boolean array for tracks above a specified PID score (range 0 - 1, -1=No score)   
     |
     |     Args: 
     |       data (awkward.Array): Input array containing the trkqual.result branch
     |       value (float): The numerical output of the MVA
     |
     |    get_trigger(self, data, name):
     |       Return boolean array for the chosen trigger name 
     |       Args: 
     |           data (awkward.Array): Input array containing the data and including the trigger branches which you want to select
     |           name : name of the trigger
     |           usage: get_trigger(data, "trig_tprHelixDe_ipa")
     |           
     |     get_triggers(self, data, names):
     |    
     |      Return a single boolean array (mask) for events where ALL specified triggers are true (== 1).
     |
     |       Assumes all 'names' exist as branches in 'data' and contain only 0s or 1s.
     |
     |       Args: 
     |           data (awkward.Array): Input array containing the data and the trigger branches.
     |           names (list[str]): A list of trigger names.
     |           
     |       Usage: 
     |           combined_mask = get_triggers(data, ["trig_tprHelixDe_ipa", "another_trigger_name"])
     |
     |  ----------------------------------------------------------------------
```
    
</details>

---
#### `pycut`

A comprehensive framework for managing analysis cuts.

<details>
<summary>Click for details</summary>

## Features 

### Cut definition and management
- **`add_cut(name, description, mask, active=True, group=None)`** - Define analysis cuts with boolean masks
- **`toggle_cut(cut_dict)`** - Enable/disable individual cuts using dictionary mapping
- **`toggle_group(group_dict)`** - Enable/disable entire groups of cuts 

### Cut flow generation
- **`create_cut_flow(data)`** - Generate detailed cut flow showing progressive event retention
- **`format_cut_flow(cut_flow)`** - Format cut flow as `pandas` DataFrame
- **`combine_cut_flows(cut_flow_list)`** - Combine multiple cut flows (useful for multiprocessing)

### Selection application
- **`combine_cuts(cut_names=None, active_only=True)`** - Generate combined boolean mask from selected cuts
- **`get_active_cuts()`** - Retrieve currently active cuts for inspection

### State management
- **`save_state(state_name)`** - Save current cut configuration for later restoration
- **`restore_state(state_name)`** - Restore previously saved cut configuration
- **`restore_original_state()`** - Reset all cuts to their initial active states
- **`list_saved_states()`** - Display all available saved configurations

### Organisation and inspection
- **`get_groups()`** - Retrieve cuts organised by group membership
- **`list_groups()`** - Display summary of all groups and their cut contents

## Typical workflow

1. Define cuts using `add_cut()` with appropriate groups
2. Generate baseline cut flow with `create_cut_flow()`
3. Save nominal configuration with `save_state()`
4. Create alternative configurations using `toggle_cut()` or `toggle_group()`
5. Compare efficiencies between configurations
6. Apply selected cuts using `combine_cuts()` for analysis
7. Restore configurations as needed with `restore_state()`

The module integrates well with the broader pyutils ecosystem, working with data processed by `pyprocess` and selections created by `pyselect`.

```
Help on module pyutils.pycut in pyutils:

NAME
    pyutils.pycut

CLASSES
    builtins.object
        CutManager

    class CutManager(builtins.object)
     |  CutManager(verbosity=1)
     |
     |  Class to manage analysis cuts
     |
     |  Methods defined here:
     |
     |  __init__(self, verbosity=1)
     |      Initialise
     |
     |      Args:
     |          verbosity (int, optional): Printout level (0: minimal, 1: normal, 2: detailed)
     |
     |  add_cut(self, name, description, mask, active=True, group=None)
     |      Add a cut to the collection.
     |
     |      Args:
     |          name (str): Name of the cut
     |          description (str): Description of what the cut does
     |          mask (awkward.Array): Boolean mask array for the cut
     |          active (bool, optional): Whether the cut is active by default
     |          group (str, optional): Group name for organizing cuts
     |
     |  combine_cut_flows(self, cut_flow_list, format_as_df=True)
     |      Combine a list of cut flows after multiprocessing
     |
     |      Args:
     |          cut_flows: List of cut statistics lists from different files
     |          format_as_df (bool, optional): Format output as a pd.DataFrame. Defaults to True.
     |
     |      Returns:
     |          list: Combined cut statistics
     |
     |  combine_cuts(self, cut_names=None, active_only=True)
     |      Return a Boolean combined mask from specified cuts. Applies an AND operation across all cuts.
     |      Args:
     |
     |      cut_names (list, optional): List of cut names to include (if None, use all cuts)
     |      active_only (bool, optional): Whether to only include active cuts
     |
     |  create_cut_flow(self, data)
     |      Utility to calculate cut flow from array and cuts object
     |
     |      Args:
     |          data (awkward.Array): Input data
     |
     |  format_cut_flow(self, cut_flow, include_group=True)
     |      Format cut flow as a DataFrame with more readable column names
     |
     |      Args:
     |          cut_flow (dict): The cut flow to format
     |          include_group (bool, optional): Whether to include group column
     |      Returns:
     |          df_cut_flow (pd.DataFrame)
     |
     |  get_active_cuts(self)
     |      Utility to get all active cutss
     |
     |  get_groups(self)
     |      Get all unique group names and their cuts
     |
     |      Returns:
     |          dict: Dictionary mapping group names to lists of cut names
     |
     |  list_groups(self)
     |      Print all groups and their cuts
     |
     |  list_saved_states(self)
     |      List all saved states
     |
     |  restore_original_state(self)
     |      Restore all cuts to their original active states (as defined when added)
     |
     |  restore_state(self, state_name='default')
     |      Restore previously saved cut states
     |
     |      Args:
     |          state_name (str): Name of the saved state to restore
     |
     |  save_state(self, state_name='default')
     |      Save current active states of all cuts
     |
     |      Args:
     |          state_name (str): Name for this saved state
     |
     |  toggle_cut(self, cut_dict)
     |      Utility to set cut(s) as inactive or active based on input dictionary
     |
     |      Args:
     |          cut_dict (dict): Dictionary mapping cut names to their desired active state
     |                          e.g., {"cut_name_1": False, "cut_name_2": True}
     |
     |  toggle_group(self, group_dict)
     |      Utility to set entire group(s) of cuts as inactive or active
     |
     |      Args:
     |          group_dict (dict): Dictionary mapping group names to their desired active state
     |                            e.g., {"quality_cuts": False, "momentum_cuts": True}
     |
     |  ----------------------------------------------------------------------
```
    
</details>

---
#### `pycut`

A comprehensive framework for managing analysis cuts.

<details>
<summary>Click for details</summary>

## Features 

### Cut definition and management
- **`add_cut(name, description, mask, active=True, group=None)`** - Define analysis cuts with boolean masks
- **`toggle_cut(cut_dict)`** - Enable/disable individual cuts using dictionary mapping
- **`toggle_group(group_dict)`** - Enable/disable entire groups of cuts 

### Cut flow generation
- **`create_cut_flow(data)`** - Generate detailed cut flow showing progressive event retention
- **`format_cut_flow(cut_flow)`** - Format cut flow as `pandas` DataFrame
- **`combine_cut_flows(cut_flow_list)`** - Combine multiple cut flows (useful for multiprocessing)

### Selection application
- **`combine_cuts(cut_names=None, active_only=True)`** - Generate combined boolean mask from selected cuts
- **`get_active_cuts()`** - Retrieve currently active cuts for inspection

### State management
- **`save_state(state_name)`** - Save current cut configuration for later restoration
- **`restore_state(state_name)`** - Restore previously saved cut configuration
- **`restore_original_state()`** - Reset all cuts to their initial active states
- **`list_saved_states()`** - Display all available saved configurations

### Organisation and inspection
- **`get_groups()`** - Retrieve cuts organised by group membership
- **`list_groups()`** - Display summary of all groups and their cut contents

## Typical workflow

1. Define cuts using `add_cut()` with appropriate groups
2. Generate baseline cut flow with `create_cut_flow()`
3. Save nominal configuration with `save_state()`
4. Create alternative configurations using `toggle_cut()` or `toggle_group()`
5. Compare efficiencies between configurations
6. Apply selected cuts using `combine_cuts()` for analysis
7. Restore configurations as needed with `restore_state()`

The module integrates well with the broader pyutils ecosystem, working with data processed by `pyprocess` and selections created by `pyselect`.

```
Help on module pyutils.pycut in pyutils:

NAME
    pyutils.pycut

CLASSES
    builtins.object
        CutManager

    class CutManager(builtins.object)
     |  CutManager(verbosity=1)
     |
     |  Class to manage analysis cuts
     |
     |  Methods defined here:
     |
     |  __init__(self, verbosity=1)
     |      Initialise
     |
     |      Args:
     |          verbosity (int, optional): Printout level (0: minimal, 1: normal, 2: detailed)
     |
     |  add_cut(self, name, description, mask, active=True, group=None)
     |      Add a cut to the collection.
     |
     |      Args:
     |          name (str): Name of the cut
     |          description (str): Description of what the cut does
     |          mask (awkward.Array): Boolean mask array for the cut
     |          active (bool, optional): Whether the cut is active by default
     |          group (str, optional): Group name for organizing cuts
     |
     |  combine_cut_flows(self, cut_flow_list, format_as_df=True)
     |      Combine a list of cut flows after multiprocessing
     |
     |      Args:
     |          cut_flows: List of cut statistics lists from different files
     |          format_as_df (bool, optional): Format output as a pd.DataFrame. Defaults to True.
     |
     |      Returns:
     |          list: Combined cut statistics
     |
     |  combine_cuts(self, cut_names=None, active_only=True)
     |      Return a Boolean combined mask from specified cuts. Applies an AND operation across all cuts.
     |      Args:
     |
     |      cut_names (list, optional): List of cut names to include (if None, use all cuts)
     |      active_only (bool, optional): Whether to only include active cuts
     |
     |  create_cut_flow(self, data)
     |      Utility to calculate cut flow from array and cuts object
     |
     |      Args:
     |          data (awkward.Array): Input data
     |
     |  format_cut_flow(self, cut_flow, include_group=True)
     |      Format cut flow as a DataFrame with more readable column names
     |
     |      Args:
     |          cut_flow (dict): The cut flow to format
     |          include_group (bool, optional): Whether to include group column
     |      Returns:
     |          df_cut_flow (pd.DataFrame)
     |
     |  get_active_cuts(self)
     |      Utility to get all active cutss
     |
     |  get_groups(self)
     |      Get all unique group names and their cuts
     |
     |      Returns:
     |          dict: Dictionary mapping group names to lists of cut names
     |
     |  list_groups(self)
     |      Print all groups and their cuts
     |
     |  list_saved_states(self)
     |      List all saved states
     |
     |  restore_original_state(self)
     |      Restore all cuts to their original active states (as defined when added)
     |
     |  restore_state(self, state_name='default')
     |      Restore previously saved cut states
     |
     |      Args:
     |          state_name (str): Name of the saved state to restore
     |
     |  save_state(self, state_name='default')
     |      Save current active states of all cuts
     |
     |      Args:
     |          state_name (str): Name for this saved state
     |
     |  toggle_cut(self, cut_dict)
     |      Utility to set cut(s) as inactive or active based on input dictionary
     |
     |      Args:
     |          cut_dict (dict): Dictionary mapping cut names to their desired active state
     |                          e.g., {"cut_name_1": False, "cut_name_2": True}
     |
     |  toggle_group(self, group_dict)
     |      Utility to set entire group(s) of cuts as inactive or active
     |
     |      Args:
     |          group_dict (dict): Dictionary mapping group names to their desired active state
     |                            e.g., {"quality_cuts": False, "momentum_cuts": True}
     |
     |  ----------------------------------------------------------------------
```
    
</details>

---

#### `pyvector`

Tools for 3D element-wise vector operations in a pure Python environment. 

<details>
<summary>Click for details</summary>

```
NAME
    pyvector

CLASSES
    builtins.object
        Vector

    class Vector(builtins.object)
     |  Vector(verbosity=1)
     |
     |  Methods for handling vector operations with Awkward arrays
     |
     |  Methods defined here:
     |
     |  __init__(self, verbosity=1)
     |      Initialise Vector
     |
     |      Args:
     |          Print detail level (0: minimal, 1: medium, 2: maximum)
     |
     |  get_mag(self, branch, vector_name)
     |      Return an array of vector magnitudes for specified branch
     |
     |      Args:
     |          branch (awkward.Array): The branch, such as trgsegs or crvcoincs
     |          vector_name: The parameter associated with the vector, such as 'mom' or 'pos'
     |
     |  get_vector(self, branch, vector_name)
     |      Return an array of XYZ vectors for specified branch
     |
     |      Args:
     |          branch (awkward.Array): The branch, such as trgsegs or crvcoincs
     |          vector_name: The parameter associated with the vector, such as 'mom' or 'pos'
     |
     |   get_rho(self, branch, vector_name):
     |       Return an array of vector rho (transverse magnitude) for specified branch
     |
     |    Args:
     |        branch (awkward.Array): The branch, such as trksegs or crvcoincs
     |        vector_name: The parameter associated with the vector, such as 'mom' or 'pos'
     |  ----------------------------------------------------------------------
```

</details>

---

#### `pylogger`

Helper module for managing printouts across the `pyutils` suite.


<details>
<summary>Click for details</summary>

    
```
NAME
    pylogger

CLASSES
    builtins.object
        Logger

    class Logger(builtins.object)
     |  Logger(verbosity=1, print_prefix='[pylogger]')
     |
     |  Helper class for consistent logging with emoji indicators
     |
     |  Methods defined here:
     |
     |  __init__(self, verbosity=1, print_prefix='[pylogger]')
     |      Initialize the Logger
     |
     |      Args:
     |          verbosity (int, opt): Level of output detail (0: errors only, 1: info, 2: max)
     |          print_prefix (str, opt): Prefix for printouts, e.g. "[pyprocess]"
     |
     |  log(self, message, level_name=None)
     |      Print a message based on verbosity level
     |
     |      Args:
     |          message (str): The message to print
     |          level (str, optional): Level name (error, info, success, warning, debug, max)
     |
     |  ----------------------------------------------------------------------
```

</details>

---

#### `pymcutil`

Utility for helping users to understand the MC origins of given tracks.


<details>
<summary>Click for details</summary>

```
class MC(builtins.object)
     |  pymcutil(verbosity=1, print_prefix='[pymcutil]')
     |
     |  To help identify true origin of an event
     |
     |  Methods defined here:
     |
     |  __init__(self, verbosity=1, print_prefix='[pylogger]')
     |      Initialize the mcutil
     |
     |      Args:
     |          particle_count_return : array of primary particle start codes (from trkmcsim)
     |          print_prefix (str, opt): Prefix for printouts, e.g. "[pyprocess]"
     |
     |  count_particle_types(self, data)
     |      looks at the 'trkmcsim' field of the data list and finds start code.
     |      Assumes 1 primary per event, gives -2 code to anything else
     |
     |      Returns:
     |          particle_count_return : list of primary codes associated with input data 1 per event
     |
     |   is_track_particle(self, data):
     |       function looks at tracks and finds particle with most contributions
     |
     |
     |   is_muon(self, data):
     |    return mask for all events with true muon 
     |    
     |   is_electron(self, data):
     |     return mask for all events with true electron 
     |     
     |   is_positron(self, data):
     |     return mask for all events with true e+n 
     |     
     |         
     |   is_particle(self, data, code):
     |     returns true if the trkmcsim has pdg code for chosen particle 
     |
     |
     |   start_process(self, data, process):
     |     returns true if the trkmcsim has process code for chosen start code 
     |
     |   stop_process(self, data, process):
     |    returns true if the trkmcsim has process code for chosen stop code
     |        
     |   is_CeMinusEndpoint(self, data):
     |     returns true if the trkmcsim has process code for ce- endpoint generator 
     |
     |
     |   is_CeMinusLeadingLog(self, data):
     |     returns true if the trkmcsim has process code for ce- leading log generator 
     |
     |   is_CePlusEndpoint(self, data):
     |     returns true if the trkmcsim has process code for ce+ endpoint generator 
     |
     |   is_CePlusLeadingLog(self, data):
     |     returns true if the trkmcsim has process code for ce+ leading log generator 
     |
     |   is_target_DIO(self, data):
     |     returns true if the trkmcsim has a DIO process code and originates at radius consistant with target
     |      
     |     
     |   is_cosmic(self, data):
     |     returns true if the trkmcsim is a cosmic generated particle 
     |
     |  ----------------------------------------------------------------------
  
```

</details>


---

#### `pydisplay`

Utility for calling the EventDisplay [https://github.com/Mu2e/EventDisplay].

In order to use this utility the user needs to run:

```
mu2einit
muse setup
```

the user must have `EventDisplay` in their working area (either a clone or via the Analysis Musing). An example of how to use this script can be found in `examples/scripts/rundisplay.py` that script can in fact be used as a means to launch the display in isolation to `pyutils`.


<details>
<summary>Click for details</summary>


```
class Display:
    """
    Class for executing the EventDisplay
    
    Note:
      For this to work:
      * mu2einit
      * muse setup
      * assumes local copy of EventDisplay via clone or musing
    """
    __init__(self, verbosity=1):
      # Start logger
      self.logger = Logger( 
          print_prefix = "[pydisplay]", 
          verbosity = verbosity
      )
    
    pick_event(self, dataset, run, subrun, event):
      """ use pickEvent tool to extract event, run, subrun from given data set """
      result = subprocess.run(['pickEvent', '-e','-v',str(dataset),' ',str(run)+'/'+str(subrun)+'/'+str(event)], capture_output=True, text=True)
      print(result.stdout)
    
    launch_display(self, dataset, run, subrun, event):
      """ launches the mu2e event display 
      """
      launch_display = subprocess.run(['mu2e','-c','EventDisplay/examples/nominal_example.fcl', str(dataset)+'_'+str(run)+'_'+str(subrun)+'_'+str(event)+'.art'], capture_output=True, text=True)
      print(launch_display.stdout)
```
</details>




## 3. Instructions for developers  

You can develop and test `pyutils` code by creating an editable install, as follows:

```bash
mu2einit
pyenv ana
git clone https://github.com/Mu2e/pyutils.git
cd pyutils
pip install -e . --user 
```

To verify that Python can import and use your local install:

```bash
python -c "import pyutils;print(pyutils.__file__)"
```

which should return

```bash
/path/to/dev/area/pyutils/pyutils/__init__.py 
```

Your changes will be automatically be applied to the `pyutils` installed in your environment, with no need to rerun the `pip` command, and you can import modules and classes using the same syntax as normal.

## Contact

Reach out via Slack (#analysis-tools or #analysis-tools-devel) if you need help or would like to contribute.

