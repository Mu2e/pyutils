import numpy as np
import matplotlib.pyplot as plt
import awkward as ak

from pyutils.pyselect import Select
from pyutils.pyvector import Vector



class Compare():
    """Class to conduct comparisons between cut or data sets
    """
    def __init__(self ):
      """
      """
      
      # Custom prefix for log messages from this processor
      self.print_prefix = "[Compare] "
      print(f"{self.print_prefix}Initialised")

    def plot_variable(self, val_overlay, val_label, filenames, lo, hi, cut_lo, cut_hi, mc_count, columns=[], nbins = 50, density=False, residuals = False):
      """
      Plots distributions of the given parameter (val), splitting by process code

      Args:
          val : list of values e.g. rmax
          val_label : text formated value name e.g. "rmax"
          lo : plot range lower bound
          hi : plot range upper bound
          cut_lo : lower cut choice
          cut_hi : upper cut choice
          mc_counts : list of process codes

      Returns:
          plots saved as pdfs
      """
      sets = []

      if residuals:
        fig, (ax1, ax2) = plt.subplots(2,1, height_ratios=[3,1])
      else:
        fig, (ax1) = plt.subplots(1,1)

      cols = ['red','blue','green','black','cyan','magenta','grey','orange']
      labs = ['Cosmic','int. RPC','ext. RPC','int. RMC','ext. RMC','IPA Decays','DIO', 'Signal']
      styles = ['bar','step','step']
      lines=["","-","--"]
      alphas = [1,1,1]
      
      for i, val in enumerate(val_overlay):
        val = ak.drop_none(val)
        val_signal = val.mask[mc_count[i] == 168]
        val_signal = np.array(ak.flatten(val_signal, axis=None))
        val_cosmics = val.mask[mc_count[i] == -1]
        val_cosmics = np.array(ak.flatten(val_cosmics, axis=None))
        val_dio = val.mask[mc_count[i] == 166]
        val_dio = np.array(ak.flatten(val_dio, axis=None))
        val_erpc = val.mask[mc_count[i] == 178]
        val_erpc = np.array(ak.flatten(val_erpc,axis=None))
        val_irpc = val.mask[mc_count[i] == 179]
        val_irpc = np.array(ak.flatten(val_irpc,axis=None))
        val_ermc = val.mask[mc_count[i] == 171]
        val_ermc = np.array(ak.flatten(val_ermc,axis=None))
        val_irmc = val.mask[mc_count[i] == 172]
        val_irmc = np.array(ak.flatten(val_irmc,axis=None))
        val_ipa = val.mask[mc_count[i] == 0]
        val_ipa = np.array(ak.flatten(val_ipa,axis=None))
        sets.append([val_cosmics,val_irpc,val_erpc,val_irmc,val_ermc, val_ipa, val_dio, val_signal])
      bin_centers = []
      bin_contents = []
      bin_errors = []
      for i in range(0,len(sets)):
        ax1.set_yscale('log')
        dummy_handle = ax1.plot([], marker="",color='white', label=columns[i])
        n, bins, patch = ax1.hist(sets[i],range=(lo,hi), color=cols, label=labs, bins=nbins, histtype=styles[i], alpha=alphas[i], stacked=True, density=density)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        bin_contents.append((n[7]))

      ax1.set_xlabel(str(val_label))
      ax1.set_xlim(lo,hi)
      # draw cuts
      
      if(len(cut_lo) !=0):
        ax1.plot(cut_lo, [0,1000], 'magenta')
      if(len(cut_hi) !=0):
        ax1.plot(cut_hi, [0,1000], 'magenta')
      
      ax1.legend(ncol=len(columns),loc='upper right')

      if residuals:
        # residuals for signal only
        residuals = []
        residuals_err = []
        residuals = (bin_contents[0] - bin_contents[1])/np.sqrt(bin_contents[1])
        residuals_err= np.sqrt(np.sqrt(bin_contents[1])*np.sqrt(bin_contents[1])  + np.sqrt(bin_contents[0])*np.sqrt(bin_contents[0]))/np.sqrt(bin_contents[1])
        
        ax2.errorbar(bin_centers, residuals, yerr=residuals_err, fmt='.', color='red', capsize=3, label='Error Bars')
        ax2.set_xlabel(str(val_label))
        ax2.set_xlim(lo,hi)
        ax2.set_ylabel("(Old - New)/Sigma")
      plt.savefig(str(filenames)+"_selection.pdf")
      plt.show()
      
    

    def compare_resolution(self, recomom, truemom):
      """
      stores difference between recon and true momentum for resolution comparison
      """
      truemom = truemom.mask[truemom > 85] # removes anything that we dont care about on the reconstruction
      recomom = ak.nan_to_none(recomom)
      recomom = ak.drop_none(recomom)
      truemom = ak.nan_to_none(truemom)
      truemom = ak.drop_none(truemom)

      differences = [
        reco[0] - truemom[i][j][0]
        for i, reco_list in enumerate(recomom)
        for j, reco in enumerate(reco_list)
        if len(reco) != 0 and len(truemom[i][j]) != 0
      ]
      
      return differences

    def plot_resolution(self, val_overlay, val_label, filenames, lo, hi, columns=[], density=True):
      """
      Plots distributions of the given parameter (val), splitting by process code

      Args:
          val : list of values e.g. rmax
          val_label : text formated value name e.g. "rmax"
          lo : plot range lower bound
          hi : plot range upper bound

      Returns:
          plots saved as pdfs
      """
      fig, (ax1, ax2) = plt.subplots(2,1, height_ratios=[3,1])
      sets=[]
      cols = ['orange']
      labs = ['signal']
      styles = ['bar','step']
      lines=["","-"]
      alphas = [0.2,1]
      for i, val in enumerate(val_overlay):
        val = ak.drop_none(val)
        val = np.array(ak.flatten(val,axis=None))
        sets.append([val])
      bin_centers = []
      bin_contents = []
      bin_errors = []
      for i in range(0,len(sets)):
        ax1.set_yscale('log')
        dummy_handle = ax1.plot([], marker="",color='white', label=columns[i])
        n, bins, patch = ax1.hist(sets[i],range=(lo,hi), color=cols, label=labs, bins=50, histtype=styles[i], alpha=alphas[i], stacked=True, density=density)

        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        
        bin_contents.append((n))

      ax1.set_xlabel(str(val_label))
      ax1.set_xlim(lo,hi)
      ax1.legend(ncol=len(columns))

      # residuals for signal only
      residuals = []
      residuals_err = []

      residuals = (bin_contents[0] - bin_contents[1])/np.sqrt(bin_contents[1])
      residuals_err= np.sqrt(np.sqrt(bin_contents[1])*np.sqrt(bin_contents[1])  + np.sqrt(bin_contents[0])*np.sqrt(bin_contents[0]))/np.sqrt(bin_contents[1])

      ax2.errorbar(bin_centers, residuals, yerr=residuals_err, fmt='o', color='red', capsize=3, label='Error Bars')
      ax2.set_xlabel(str(val_label))
      ax2.set_ylabel("(Old - New)/Sigma")
      ax2.set_xlim(lo,hi)
      
      plt.savefig(str(filenames)+"_resolution.pdf")
      plt.show()
      
    def plot_particle_counts(self, mc_counts, columns):
      """
      Plot a grouped horizontal bar chart comparing particle type counts
      between different datasets and adds percentage change labels.
      
      Args:
          mc_counts : list of arrays/lists of particle codes (one per dataset)
          columns   : labels for datasets (e.g., ["old_cuts", "no_cuts"])
      """
      # Map PDG/startCodes to categories
      labels = ["DIO", "IPA", "CEMLL", "CEPLL", "eRPC", "iRPC", "eRMC", "iRMC", "Cosmic", "Other"]
      pdg_codes = [166, 114, 168, 176, 178, 179, 171, 172, -1, -2]
      num_categories = len(pdg_codes)
      num_datasets = len(mc_counts)

      # Use NumPy's vectorized operations for efficient counting
      datasets = np.zeros((num_datasets, num_categories), dtype=int)
      for i, mc in enumerate(mc_counts):
          if mc is not None and len(mc) > 0:
              mc_array = np.array(mc)
              for j, code in enumerate(pdg_codes):
                  datasets[i, j] = np.sum(mc_array == code)
      
      # Check that there are at least two datasets for a comparison
      if num_datasets < 2:
          print("Not enough datasets for percentage change calculation. Plotting without it.")
          # Re-run the original plotting logic if needed
          # ...
          return

      # Calculate percentage change based on the first dataset
      # Avoids division by zero by setting change to 0 if the original value is 0
      with np.errstate(divide='ignore', invalid='ignore'):
          old_counts = datasets[0]
          new_counts = datasets[1]
          percent_changes = ((new_counts - old_counts) / old_counts) * 100
          percent_changes[np.isinf(percent_changes) | np.isnan(percent_changes)] = 0

      # Plot grouped horizontal bars
      y = np.arange(num_categories)
      bar_height = 0.8 / num_datasets
      
      fig, ax = plt.subplots(figsize=(12, 6))

      bars = []
      for i, data in enumerate(datasets):
          bars.append(ax.barh(y + i * bar_height, data, height=bar_height, label=columns[i]))
      
      # Add percentage change labels to the second set of bars
      for i, bar in enumerate(bars[1]): # Iterate over the bars of the second dataset
          # Get the percentage change for the corresponding category
          change = percent_changes[i]
          
          # Format the label string
          label_text = f'{change:.1f}%'
          
          # Choose color based on whether change is positive or negative
          color = 'red' if change < 0 else 'green'
          
          # Position the label
          # Get the y-position and width (x-value) of the bar
          ax.text(
              bar.get_width(), 
              bar.get_y() + bar.get_height() / 2, 
              label_text, 
              ha='left', 
              va='center',
              color=color,
              fontsize=8
          )

      # Center the y-tick labels correctly
      ax.set_yticks(y + bar_height * (num_datasets - 1) / 2)
      ax.set_yticklabels(labels)
      ax.set_xlabel("Event counts")
      ax.set_title("Comparison of particle types with Percentage Change")
      #ax.set_xlim(0, 60000)
      ax.legend()
      
      plt.tight_layout()
      plt.savefig("particle_comparison_with_changes.pdf")
      plt.show()
      
    def plot_cut_eff(self, numerator_array, denominator_array, bin_centers, title="Ratio of Arrays", name="all", x_label="true momentums", y_label="Efficiency"):
      """
      Calculates the element-wise ratio of two arrays and plots the result.
      
      Args:
          numerator_array (np.ndarray): The array for the numerator.
          denominator_array (np.ndarray): The array for the denominator.
          title (str): The title of the plot.
          x_label (str): The label for the x-axis.
          y_label (str): The label for the y-axis.
      """
      # 1. Ensure arrays have the same shape
      if numerator_array.shape != denominator_array.shape:
          raise ValueError("Input arrays must have the same shape.")

      # 2. Handle potential division by zero
      # Use np.divide with 'where' to perform division only where denominator is not zero.
      # Specify the `out` array to hold the result and set values to 0 where denominator is zero.
      ratio = np.divide(numerator_array, denominator_array, out=np.zeros_like(numerator_array, dtype=float), where=denominator_array != 0)

      # 3. Create the plot
      fig, ax = plt.subplots(figsize=(10, 6))
      
      #ax.plot(ratio, marker='o', linestyle='-', color='b')
      #plt.scatter(, marker="-")
      plt.plot(bin_centers, ratio, marker='o', linestyle='-', label='Connected Points')
      ax.set_title(title)
      ax.set_xlabel(x_label)
      ax.set_ylabel(y_label)
      ax.grid(True)
      
      # Optional: Highlight where denominator was zero
      zero_indices = np.where(denominator_array == 0)[0]
      #if zero_indices.size > 0:
      #    ax.plot(zero_indices, ratio[zero_indices], 'rx', label='Denominator was zero')
      #    ax.legend()
      
      
      plt.savefig("eff_"+str(name)+".pdf")
      plt.show()
      
    def plot_2D(self, xs, ys):
      
      for i, x in enumerate(xs):
        y = ys[i]
        x_sync = []
        y_sync = []
        for j, element in enumerate(x):
          for k, subelement in enumerate (element):
            for l, subsubelement in enumerate (subelement):
              if(x[j][k][l] != None):
                for m, y_els in enumerate (y[j][k]):
                  if(y[j][k][m] != None):
                    
                    x_sync.append(x[j][k][l])
                    y_sync.append(y[j][k][m])
              

        # Plot the 2D histogram
        fig, ax = plt.subplots(figsize=(8, 6))
        h = ax.hist2d(y_sync, x_sync, bins=50, cmin = 1, cmap='viridis')

        # Add labels and a color bar
        ax.set_xlabel("True Momentum at TrkEnt [MeV/c]")
        ax.set_ylabel("rmax")
       
        fig.colorbar(h[3], ax=ax, label='Counts in bin') # Changed from h to h[3] to match hist2d output

        # Display the plot
        plt.show()




 


  

