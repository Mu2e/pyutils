"""
Accessing /pnfs on EAF with pyutils
===================================

Introduction

This turorial is primarily intended to show users how to access files hosted on
`/pnfs` (and defined in SAM) while working on EAF. For general information on EAF, see
[tutorial](https://github.com/Mu2e/Tutorial/tree/main/EAF).

`/pnfs` is not mounted on EAF, so the `pyread` module provides functionality to read
these files with `xroot` via `mdh`.

Table of contents
1. Using `pyread` directly
1. Using `pyimport`

Generated from examples/notebooks/pyutils_on_EAF.ipynb by scripts/nb_to_script.py.
Edit the notebook and regenerate, rather than editing this script.
"""

# ======================================================================================
# 1. Using `pyread` directly
# ======================================================================================
#
# We can use the `Reader` class with the `use_remote` flag set to `True` to generate an
# xroot URL for a file and read it with uproot.
#
# `pyread` relies on `mdh` to generate the URL, and so `Reader` inherits the `location`
# and `schema` arguments from `mdh`, which are explained below:
#
# $ mdh print-url --help
# usage: mdh print-url [-h] [-l LOCATION] [-s SCHEMA] names [names ...]
#
# ...
#
#   -l LOCATION, --location LOCATION
#                         standard location to use:
#                           tape (default), disk, scratch, nersc
#   -s SCHEMA, --schema SCHEMA
#                         Schema to use in writing the url:
#                           path (default),http,root,dcap,sam
#
# Note: When using remote files `pyutils`, will set the necessary environment variables
# (allowing us to leverage Mu2e file handling tools) once per process.

# Import the Reader class
from pyutils.pyread import Reader

# Define our file name (from the MDS4b ensemble SAM definition)
file_name = "nts.mu2e.ensembleNtuple.MDC2025-001_MDS4b.001430_00002163.root"

# Initialise the Reader with the appropriate arguments
reader = Reader(
    use_remote = True,
    location = "disk", # This file is on persistent dCache (use "tape" for files archived on tape, after prestaging)
    verbosity = 2
)

# Retrieve an uproot object for this file
file = reader.read_file(file_name)

# The URL has been created and the file has been opened, we can now inspect the file...

print(file.keys())

# ======================================================================================
# 2. Using `pyprocess`
# ======================================================================================
#
# `pyprocess` employs `pyread` as a dependancy, so if we are working with TTrees
# (`EventNtuple`) we can skip `pyread` and use `pyprocess` to retrieve our data as a
# structured array.

# Import the Importer class
from pyutils.pyprocess import Processor

# Define our file name (from the MDS4b ensemble SAM definition)
file_name = "nts.mu2e.ensembleNtuple.MDC2025-001_MDS4b.001430_00002163.root"

#  Initialise the Importer with the appropriate arguments
processor = Processor( 
    use_remote = True,
    location = "disk",
    verbosity = 2
)

# Define branches
# Branches can also be defined as a list of a grouped dict of lists, depending on requirements
branches = [
    "event",
    "trk.nactive",
    "trkqual.result",
    "crvcoincs.PEs",
    "trksegs"
]

# Import branches 
data = processor.process_data(
    file_name = file_name,
    # branches = "*" # read all branches
    branches = branches
)

# We now have an awkward array called `data` which can be used for analysis.
