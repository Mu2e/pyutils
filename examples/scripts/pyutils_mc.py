# this is an example for looking up the true primary creating a given reconstructed event

# Import external packages
import awkward as ak


# Import the Procssor class
from pyutils.pyprocess import Processor 
from pyutils.pymcutil import MC

def main():
  # Initialise the Importer with increased verbosity 
  # verbosity=0 will only show errors
  processor = Processor(verbosity=2)

  # Define the path to our example file
  file_name = "/exp/mu2e/data/users/sophie/ensembles/MDS1/MDS1av0.root"

  # Define the branches we want to access
  # For a complete list of available branches, see:
  # https://github.com/Mu2e/EventNtuple/blob/main/doc/branches.md
  # Also refer to ntuplehelper, available after mu2e setup EventNtuple
  branches = ["trkmcsim"]

  # Import the branches from the file
  # This loads the data into memory and returns an awkward array
  data = processor.process_data(
      file_name=file_name,
      branches=branches
  )
  
  mc_example = MC()
  mc_example.count_particle_types(data)
  
if __name__ == "__main__":
    main()
