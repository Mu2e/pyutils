import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import tensorflow as tf
import math

from pyutils.pyselect import Select
from pyutils.pyvector import Vector

import zfit
import zfit.z.numpy as znp

# 1. Define the custom PDF based on erf
class CustomErfPDF(zfit.pdf.ZPDF):
    """A custom PDF based on a scaled and shifted error function."""
    _PARAMS = ['mu', 'sigma']
    _N_OBS = 1

    def _unnormalized_pdf(self, x):
        """The unnormalized PDF is the derivative of the error function."""
        x = zfit.z.unstack_x(x)
        mu = self.params['mu']
        sigma = self.params['sigma']
        
        z = (x - mu) / (sigma * znp.sqrt(2.0))
        
        return znp.exp(-z**2)


class CustomLandau(zfit.pdf.ZPDF):
    # Specify the names of the parameters
    _PARAMS = ['mpv', 'width']
    _N_OBS = 1

    # Implement the unnormalized PDF calculation
    def _unnormalized_pdf(self, x):
        x = zfit.z.unstack_x(x)
        
        mpv = self.params['mpv']
        width = self.params['width']

        z = (x - mpv) / width
        return znp.exp(-0.5 * (z + znp.exp(-z))) / width
        
        
        
class Fits():
    """Class to conduct unbinned ML fits to understand differences in resolution, loss and efficiency
    """
    def __init__(self ):
      """
      """
      
      # Custom prefix for log messages from this processor
      self.print_prefix = "[Fits] "
      print(f"{self.print_prefix}Initialised")

    def plot_variable(self, val_overlay, val_label, filenames, lo, hi, cut_lo, cut_hi, mc_count, columns=[], nbins = 50, density=True):
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
      rpc = ["e+", "e-"]
      code = [173,174]
      fig, ax1 = plt.subplots(1,1)
      #cols = ['black']
      labs = ['flat']
      styles = ['step']
      lines=["","-","--"]
      alphas = [0.2,1,1]
      text_contents = []
      for i, val in enumerate(val_overlay):
        val = ak.drop_none(val)

        val_test = val.mask[mc_count[i] == int(code[i])]
        val_test = np.array(ak.flatten(val_test,axis=None))

        mean_val = np.mean(val_test)
        std_dev = np.std(val_test)
        text_contents.append(str(rpc[i])+ f"Mean: {mean_val:.2f}\nStd Dev: {std_dev:.2f}")
        sets.append([val_test])
      for i in range(0,len(sets)):
        ax1.set_yscale('log')
        dummy_handle = ax1.plot([], marker="",color='white', label=columns[i])
        n, bins, patch = ax1.hist(sets[i],range=(lo,hi), label=labs, bins=nbins, histtype='step', stacked=True, density=density)


      ax1.set_xlabel(str(val_label))
      ax1.set_xlim(lo,hi)
 
      ax1.legend(ncol=len(columns))#,loc='upper center')
      for i in range(0,len(text_contents)):
        plt.text(0.1, 0.95-i*0.15, text_contents[i], 
                 transform=plt.gca().transAxes,
                 fontsize=10,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))

      plt.savefig(str(filenames)+"_selection.pdf")
      plt.show()


    def fit_momentum(self, data_list, start, end, opt, label,compare_labels):
        """
        Fits a simple Polynomial shape to the reconstructed momentum data
        using an extended unbinned maximum likelihood fit.
        """
        # Create figure with two subplots: main plot and ratio plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        colors = [ "blue","green"]
        labels = compare_labels
        
        # Store text box y-positions to avoid overlap
        text_y_pos = [0.8, 0.5]
        text_x_pos = [0.8, 0.5]
        norm = 0.
        for i, data in enumerate(data_list):
            mom_mag_skim = ak.nan_to_none(data)
            mom_mag_skim = ak.drop_none(mom_mag_skim)

            # Define the observable space for the fit
            obs_mom = zfit.Space('x', limits=(start, end))
            mom_np = ak.to_numpy(ak.flatten(mom_mag_skim, axis=None))
            mom_zfit = zfit.Data.from_numpy(array=mom_np, obs=obs_mom)
            
            # Define parameters for the Polynomial shape and yield
            
            N_Flat = zfit.Parameter('N_Flat', 5000, 10, 150000)
            
            # Create parameters for the coefficients
            if opt == "poly":

              c1 = zfit.Parameter("c1", 0.1, -2, 2)
              c2 = zfit.Parameter("c2", 0.1, -2, 2)
              c3 = zfit.Parameter("c3", 0.1, -2, 2)
              c4 = zfit.Parameter("c4", 0.1, -2, 2)
              c5 = zfit.Parameter("c5", 0.1, -2, 2)
              coeffs = [c1, c2, c3, c4, c5] #--> 3 seems good
              text_y_pos = [0.98, 0.65]
              text_x_pos = [0.05, 0.05]
              # Create a Chebyshev polynomial PDF
              poly_model = zfit.pdf.Chebyshev(obs=obs_mom, coeffs=coeffs, extended=N_Flat)

            if opt == "erf":
              mu = zfit.Parameter("mu", 103, 100, 110)
              sigma = zfit.Parameter("sigma", 7, 1, 20)
              poly_model = CustomErfPDF(mu=mu, sigma=sigma, obs=obs_mom, extended=N_Flat)

            if opt == "landau":

              mpv = zfit.Parameter('mpv', 0, 0, 0.45)
              width = zfit.Parameter('width', 0.1, 0.05, 0.15)
              text_y_pos = [0.98, 0.8]
              text_x_pos = [0.05, 0.05]

              poly_model = CustomLandau(mpv=mpv, width=width, obs=obs_mom, extended=N_Flat)
              #poly_model = zfit.pdf.FFTConvPDFV1(func=landau_pdf, kernel=cb_pdf, obs=obs_mom, extended=N_Flat)
            if opt == "logN":
              mu = zfit.Parameter('mu', 0, -1, 1)
              sigma = zfit.Parameter('sigma', 0.1, 0.1,0.7)
              poly_model = zfit.pdf.LogNormal(mu=mu, sigma=sigma, obs=obs_mom, extended=N_Flat)


            if opt == "dscb":
              mu = zfit.Parameter("mu", 0, -1,1)
              sigma = zfit.Parameter("sigma", 0.5, 0.1, 5.0)
              alphal = zfit.Parameter("alphal", 0.5, 0.1, 2.0)
              nl = zfit.Parameter("nl", 2.0, 1.0, 20.0)
              alphar = zfit.Parameter("alphar", 0.5, 0.1, 2.0)
              nr = zfit.Parameter("nr", 2.0, 1.0, 20.0)
              text_y_pos = [0.98, 0.65]
              text_x_pos = [0.05, 0.05]
              ax1.set_yscale('log')
              poly_model = zfit.pdf.DoubleCB(mu=mu, sigma=sigma, alphal=alphal, nl=nl, alphar=alphar, nr=nr, obs=obs_mom, extended=N_Flat)

      
            # Create the extended unbinned negative log-likelihood loss
            nll = zfit.loss.ExtendedUnbinnedNLL(model=poly_model , data=mom_zfit)
            
            # Minimize the loss and get the result
            minimizer = zfit.minimize.Minuit()
            result = minimizer.minimize(loss=nll)
            hesse_errors = result.hesse()
            print(result)
            
            # --- Plotting the fit result ---
            
            fit_range = (obs_mom.lower[0, 0], obs_mom.upper[0, 0])
            n_bins = 100
            bin_width = (fit_range[1] - fit_range[0]) / n_bins
            
            # --- Main plot ---
            
            mom_plot = np.linspace(fit_range[0], fit_range[1], 200).reshape(-1, 1)

            poly_model_curve = zfit.run(poly_model.pdf(mom_plot) * result.params[N_Flat]['value'] * bin_width)
            ax1.plot(mom_plot.flatten(), poly_model_curve.flatten(), color=colors[i], linestyle="--", label=str(labels[i])+' Fit')
            ax1.grid(True)
            
            data_hist, data_bins, _ = ax1.hist(mom_np, color=colors[i], bins=n_bins, range=fit_range, label=labels[i], histtype='step')
            data_bin_center = (data_bins[:-1] + data_bins[1:]) / 2
            ax1.errorbar(data_bin_center, data_hist, yerr=np.sqrt(data_hist), fmt='.', color=colors[i], capsize=2)
            
            ax1.set_xlabel(str(label))
            ax1.set_ylabel('# of events per bin')
            #ax1.set_yscale('log')
            ax1.legend()
            #ax1.set_title('Polynomial Fit to Momentum Data (Extended Unbinned)')
            
            # --- Add text box with fit parameters ---
            if opt == "poly":
              param_text = (
              f"Fit parameters for {labels[i]}:\n"
              f"$N_{{Flat}} = {result.params[N_Flat]['value']:.0f} \\pm {hesse_errors[N_Flat]['error']:.2f}$\n"
              f"$c_{{1}} = {result.params[c1]['value']:.3f} \\pm {hesse_errors[c1]['error']:.4f}$\n"
              f"$c_{{2}} = {result.params[c2]['value']:.3f} \\pm {hesse_errors[c2]['error']:.4f}$\n"
              f"$c_{{3}} = {result.params[c3]['value']:.3f} \\pm {hesse_errors[c3]['error']:.4f}$\n"
              f"$c_{{4}} = {result.params[c4]['value']:.3f} \\pm {hesse_errors[c4]['error']:.4f}$\n"
              f"$c_{{5}} = {result.params[c5]['value']:.3f} \\pm {hesse_errors[c5]['error']:.4f}$\n"
              )
            if opt == "erf":
              param_text = (
              f"Fit parameters for {labels[i]}:\n"
              f"$N_{{Flat}} = {result.params[N_Flat]['value']:.0f} \\pm {hesse_errors[N_Flat]['error']:.2f}$\n"
              f"$\\mu = {result.params[mu]['value']:.3f} \\pm {hesse_errors[mu]['error']:.4f}$\n"
              f"$\\sigma = {result.params[sigma]['value']:.3f} \\pm {hesse_errors[sigma]['error']:.4f}$\n"
              )
            if opt == "logN":
              param_text = (
              f"Fit parameters for {labels[i]}:\n"
              f"$N_{{Flat}} = {result.params[N_Flat]['value']:.1f} \\pm {hesse_errors[N_Flat]['error']:.2f}$\n"
              f"$\\mu = {result.params[mu]['value']:.3f} \\pm {hesse_errors[mu]['error']:.4f}$\n"
              f"$\\sigma = {result.params[sigma]['value']:.3f} \\pm {hesse_errors[sigma]['error']:.4f}$\n"
              )
              
            if opt == "landau":
              param_text = (
              f"Fit parameters for {labels[i]}:\n"
              f"$N_{{Flat}} = {result.params[N_Flat]['value']:.1f} \\pm {hesse_errors[N_Flat]['error']:.2f}$\n"
              f"$mpv = {result.params[mpv]['value']:.3f} \\pm {hesse_errors[mpv]['error']:.4f}$\n"
              f"$width = {result.params[width]['value']:.3f} \\pm {hesse_errors[width]['error']:.4f}$\n"
              )
              
            if opt == "dscb":
              param_text = (
              f"Fit parameters for {labels[i]}:\n"
              f"$N_{{Flat}} = {result.params[N_Flat]['value']:.0f} \\pm {hesse_errors[N_Flat]['error']:.2f}$\n"
              f"$\\mu = {result.params[mu]['value']:.3f} \\pm {hesse_errors[mu]['error']:.4f}$\n"
              f"$\\sigma = {result.params[sigma]['value']:.3f} \\pm {hesse_errors[sigma]['error']:.4f}$\n"
              f"$\\alpha_{{l}} = {result.params[alphal]['value']:.3f} \\pm {hesse_errors[alphal]['error']:.4f}$\n"
              f"$n_{{l}} = {result.params[nl]['value']:.3f} \\pm {hesse_errors[nl]['error']:.4f}$\n"
              f"$\\alpha_{{r}} = {result.params[alphar]['value']:.3f} \\pm {hesse_errors[alphar]['error']:.4f}$\n"
              f"$n_{{r}} = {result.params[nr]['value']:.3f} \\pm {hesse_errors[nr]['error']:.4f}$\n"
              )

            props = dict(boxstyle='round', facecolor=colors[i], alpha=0.3)
            
            # Position the text box in the upper left corner of the subplot
            # with an offset for each iteration
            ax1.text(text_x_pos[i], text_y_pos[i], param_text, transform=ax1.transAxes,
                     fontsize=10, verticalalignment='top', bbox=props)
            
            # --- Ratio plot ---
            
            data_bin_center_2d = data_bin_center.reshape(-1, 1)
            fit_at_bin_center = zfit.run(poly_model.pdf(data_bin_center_2d) * result.params[N_Flat]['value'] * bin_width)
            ratio = data_hist / fit_at_bin_center
            
            ax2.errorbar(data_bin_center, ratio, yerr=np.sqrt(data_hist) / fit_at_bin_center, fmt='.', color=colors[i], capsize=2)
            ax2.axhline(1, color='gray', linestyle='--')
            ax2.set_ylabel('Ratio (Data/Fit)')
            ax2.set_ylim(0.5, 1.5)
            ax2.grid(True)

            norm = result.params[N_Flat]['value']
        plt.tight_layout()
        plt.savefig("Flatfit_"+str(opt)+".pdf")
        plt.show()
        return  norm
