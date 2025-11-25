import uproot
import awkward as ak
from .pylogger import Logger

class Importer:
    """Utility class for importing branches from ROOT TTree files

    Intended to used via by the pyprocess Processor class
    """
    
    def __init__(self, file_name, branches, tree_path="EventNtuple/ntuple", verbosity=1):
        """Initialise the importer
        
        Args:
            file_name: Name of the file
            branches: Flat list or grouped dict of branches to import
            tree_path (str, opt): Path to the Ntuple in file directory. Default is "EventNtuple/ntuple".
            verbosity: Print detail level (0: minimal, 1: medium, 2: maximum) 
            
        """
        self.file_name = file_name
        self.branches = branches
        self.tree_path = tree_path 
        self.verbosity = verbosity

        self.logger = Logger( # Start logger
            print_prefix = "[pyimport]", 
            verbosity = verbosity
        )

    def _read_file(self, file_path):
        """Open file with uproot"""
        try: 
            file = uproot.open(file_path)
            self.logger.log(f"Opened {file_path}", "success")
            return file
        except Exception as e:
            self.logger.log(f"Exception while opening {file_path}: {e}", "warning")
            raise # propagate exception up
            
    def import_branches(self):
        """Internal function to open ROOT file and import specified branches
            
        Returns:
            Awkward array with imported data
        """
        try:
            # Open file 
            file = self._read_file(self.file_name) 
            # Access the tree
            components = self.tree_path.split('/')
            current = file
            # Navigate through file directory
            for component in components:
                if component in current:
                    current = current[component]
                else:
                    # Handle cases where path component doesn't exist
                    self.logger.log(f"'{component}' not found in {self.file_name}", "error")
                    return None
            # Set tree
            tree = current 
                
            # Result container
            result = {}

            if self.branches is None: 
                self.logger.log("Please provide a list of branches, or self.branches='*' to import all", "error")
                return None
    
            # Flat list
            elif isinstance(self.branches, list):
                result = tree.arrays(self.branches, library="ak")
    
            # Grouped dictionary
            elif isinstance(self.branches, dict):
                data = {}
                # Get arrays per field/group
                for group, sub_branches in self.branches.items():
                    data[group] = tree.arrays(sub_branches, library="ak")
                # Zip them together 
                result = ak.zip(data) 
    
            # If using "*" get all branches
            elif self.branches == "*":
                self.branches = [branch for branch in tree.keys()] 
                self.logger.log("Importing all branches", "info")
                # Return array 
                result = tree.arrays(filter_name=self.branches, library="ak")
                
            else: 
                self.logger.log(f"Branches type {self.branches.type} not recognised", "error")
                return None
            
            if result is not None:
                self.logger.log(f"Imported branches", "success")
                self.logger.log(f"Array structure:", "max")
                if self.verbosity > 1:
                    result.type.show()
            else:
                self.logger.log(f"Failed to import branches", "error")           

            # Return
            return result
    
        except Exception as e:
            self.logger.log(f"Exception getting branches in file {self.file_name}: {e}", "error")
            raise # Propagate exception

        finally:
            # Ensure the file is closed
            if hasattr(file, "close"):
                file.close()