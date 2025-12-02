import numpy as np
import matplotlib.pyplot as plt
import uproot
import awkward as ak
from pyutils.pyplot import Plot

class Count():
    """Class to conduct the cut and count analysis
    """
    def __init__(self, mom_list, time_list, mc_list, mom_range, time_range, sign="minus" ):
      """Class for counting events in chosen signal region
      
        Parameters
        ----------
        mom_list : array of floats
        time_list : array of floats
        mc_list : array of ints
        mom_range, time_range : [float,float], [float,float]
          optimized mom and time region for signal search
      """
      self.mom_range = mom_range
      self.time_range = time_range
      self.mom_list = mom_list
      self.time_list = time_list
      self.mc_list = mc_list
      self.reco_cut_lists = {'mom' : None, 'time': None}
      self.sign = sign
      self.plot_range_minus = [95., 110.]
      self.plot_range_plus = [85., 95.]
      # Custom prefix for log messages from this processor
      self.print_prefix = "[Count] "
      print(f"{self.print_prefix}Initialised")

    def ExtractReco(self):
      """
      Extracts number of events in defined signal and background region.

      Returns:
          tuple: (signal, background) - Number of events in signal and background regions.
      """
      # Flatten the momentum and time lists/arrays efficiently
      moms = np.array(ak.flatten(self.mom_list, axis=None))
      times = np.array(ak.flatten(self.time_list, axis=None))

      # Perform filtering using boolean indexing with NumPy
      # This is much more efficient than the manual loop
      signal_mask = (moms > self.mom_range[0]) & (moms < self.mom_range[1]) & \
                    (times > self.time_range[0]) & (times < self.time_range[1])
      
      # Calculate signal and background counts
      signal = np.sum(signal_mask)
      background = len(moms) - signal
      
      # Print results
      print(f"For momentum and time criteria: signal = {signal}, background = {background}")

      self.reco_cut_lists['mom'] = moms[(moms > self.mom_range[0]) & (moms < self.mom_range[1])]
      self.reco_cut_lists['time'] = times[(times > self.time_range[0]) & (times < self.time_range[1])]
      return signal, background

      
    def CheckMCTruth(self):
      """function checks the MC_list, counts true events in this region
      
      Returns:
        True events in signal and background regions
      """

      # apply cuts to all three lists, make sure elements are aligned
      
      # for all plottable information
      mom=[]
      time = []
      mc=[]
      
      # for final signal selection
      sig_mom=[]
      sig_time = []
      sig_mc=[]
      for i, j in enumerate(self.mom_list):
        
          if len(self.mom_list[i][0]) !=0 and len(self.time_list[i][0]) !=0:
            if len(self.mom_list[i][0]) > 1:
              #print("vetoing multi track")
              continue
            if self.sign == "minus":
              
              if((self.mom_list[i][0] > self.plot_range_minus[0]) and (self.mom_list[i][0] < self.plot_range_minus[1]) and (self.time_list[i][0] > 0.) and (self.time_list[i][0] < 1695.)):
                mom.append(self.mom_list[i][0])
                time.append(self.time_list[i][0])
                mc.append(self.mc_list[i])
                
                if((self.mom_list[i][0] > self.mom_range[0]) and (self.mom_list[i][0] < self.mom_range[1]) and (self.time_list[i][0] > self.time_range[0]) and (self.time_list[i][0] < self.time_range[1])):
                  sig_mom.append(self.mom_list[i][0])
                  sig_time.append(self.time_list[i][0])
                  sig_mc.append(self.mc_list[i])
            else:
              if((self.mom_list[i][0] > self.plot_range_plus[0]) and (self.mom_list[i][0] < self.plot_range_plus[1]) and (self.time_list[i][0] > 0.) and (self.time_list[i][0] < 1695.)):
                mom.append(self.mom_list[i][0])
                time.append(self.time_list[i][0])
                mc.append(self.mc_list[i])
            
                if((self.mom_list[i][0] > self.mom_range[0]) and (self.mom_list[i][0] < self.mom_range[1]) and (self.time_list[i][0] > self.time_range[0]) and (self.time_list[i][0] < self.time_range[1])):
                  sig_mom.append(self.mom_list[i][0])
                  sig_time.append(self.time_list[i][0])
                  sig_mc.append(self.mc_list[i])
              
      # look for true signal
      processes_signal=[[],[]]
      processes_background=[[],[]] #TODO add [] extra in here, need RMC, RPC, DIO, Cosmic
      
      for i, process in enumerate(mc):
        if ((self.sign == "minus" and (process ==168 or process == 167)) or (self.sign == "plus" and (process == 169 or process == 176)) ):
          processes_signal[0].append(mom[i][0])
          processes_signal[1].append(time[i][0])
        elif (self.sign == "plus" and (process != 166 and process != 169 and process != 176)):
          processes_background[0].append(mom[i][0])
          processes_background[1].append(time[i][0])
        else:
          processes_background[0].append(mom[i][0])
          processes_background[1].append(time[i][0])
      # print the true count in selected signal region:
      print("======= MC truth yields in Plot Region =====")
      print("CE :", len(processes_signal[0]))
      print("All backgrounds :", len(processes_background[0]))
      
      # make MC truth plots - mom
      mom_lo = 0
      mom_hi = 300
      if self.sign == "minus":
        mom_lo = self.plot_range_minus[0]
        mom_hi = self.plot_range_minus[1]
      if self.sign == "plus":
        mom_lo = self.plot_range_plus[0]
        mom_hi = self.plot_range_plus[1]
      
      
      filtered_list = ak.flatten(self.mom_list, axis=None)[(ak.flatten(self.mom_list, axis=None) >= mom_lo) & (ak.flatten(self.mom_list, axis=None) <= mom_hi)]
      print("entries", len(filtered_list))
      print("mean:", np.mean(filtered_list))
      print("std dev:", np.std(filtered_list))
      
      n,bins,patch = plt.hist([processes_signal[0],processes_background[0]],range=(mom_lo,mom_hi), color=['blue','red'], alpha=0.5, bins=50, histtype='bar', label = ['True Signal','True Background'], stacked=True)
      plt.yscale('log')
      
      # overlay reco mom selections - mom
      
      
      n,bins,patch = plt.hist(ak.flatten(self.mom_list, axis=None), color='black', bins=50,range=(mom_lo,mom_hi), histtype='step', label = 'All Reco')
      n,bins,patch = plt.hist(self.reco_cut_lists['mom'], color='red', bins=50,range=(mom_lo,mom_hi), histtype='step',label = 'Selected Reco')
      plt.yscale('log')
      plt.legend(fontsize='large')
      plt.xlabel('Reconstructed Momentum at Trk Ent [MeV/c]', fontsize=16)
      plt.savefig("1D_Mom_selected.pdf")
      plt.show()
      
      
      
      # make MC truth plots - time
      n,bins,patch = plt.hist([processes_signal[1],processes_background[1]],range=(0.,1695.), color=['blue','red'], alpha=0.5, bins=50, histtype='bar', stacked=True, label = [ 'True Signal','True Background'])
      plt.yscale('log')
      

      n,bins,patch = plt.hist(ak.flatten(self.time_list, axis=None),range=(0.,1695.), color='black', bins=50, histtype='step', label = 'All Reco')
      n,bins,patch = plt.hist(self.reco_cut_lists['time'],range=(0.,1695.), color='red', bins=50, histtype='step', label = 'Selected Reco')
      plt.yscale('log')
      plt.legend(fontsize='large')
      plt.xlabel('Arrival Time at Trk Ent [ns]', fontsize=16)
      plt.savefig("1D_Time_selected.pdf")
      plt.show()
      
      # print the true count in selected signal region:
      processes_signal_SR=[[],[]]
      processes_background_SR =[[],[]]
      for i, process in enumerate(sig_mc):
        if ((self.sign == "minus" and (process ==168 or process == 167)) or (self.sign == "plus" and (process == 169 or process == 176)) ):
          processes_signal_SR[0].append(sig_mom[i][0])
          processes_signal_SR[1].append(sig_time[i][0])
        elif (self.sign == "plus" and (process != 166 and process != 169 and process != 176)):
          print("process code of the backgrounds in signal region :", process)
          processes_background_SR[0].append(sig_mom[i][0])
          processes_background_SR[1].append(sig_time[i][0])
        else:
          print("process code of the backgrounds in signal region :", process)
          processes_background_SR[0].append(sig_mom[i][0])
          processes_background_SR[1].append(sig_time[i][0])
      
      # print the true count in selected signal region:
      print("======= MC truth yields in Signal Region =====")
      print("CE :", len(processes_signal_SR[0]))
      print("All backgrounds :", len(processes_background_SR[0]))
      
      # add final plots for 2D discussion
      plt.scatter(processes_signal[0], processes_signal[1], color='blue', alpha=0.2,label='All True Signal', marker = 'x')
      plt.scatter(processes_background[0], processes_background[1], color='red', alpha=0.2, label='All True Background', marker = 'x')
      
      plt.scatter(processes_signal_SR[0], processes_signal_SR[1], color='blue', label='Selected True Signal', marker = 'o')
      plt.scatter(processes_background_SR[0], processes_background_SR[1], color='red', label='Selected True Background', marker = 'o')
      
      plt.xlim(mom_lo,mom_hi)
      plt.ylim(400.,1695.)
      plt.plot([self.mom_range[0],self.mom_range[0]], [400,1695.], 'k-')
      plt.plot([self.mom_range[1],self.mom_range[1]], [400,1695.], 'k-')
      plt.plot([mom_lo,mom_hi],[self.time_range[0],self.time_range[0]], 'k-')
      plt.plot( [mom_lo,mom_hi],[self.time_range[1],self.time_range[1]], 'k-')
      plt.legend(fontsize='large')
      plt.xlabel("Reconstructed Momentum at Trk Ent [MeV/c]", fontsize=16)
      plt.ylabel("Arrival Time at Trk Ent [ns]", fontsize=16)
      plt.savefig("2D_selected.pdf")
      plt.show()
      #return processes_signal_SR, processes_background_SR
      return len(processes_signal_SR[0]), len(processes_background_SR[0])
