import uproot
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
import awkward as ak
  
class MLSelect():
    """Class to conduct ML based signal/background selection
    """
    def __init__(self ):
      """
      """
      
      # Custom prefix for log messages from this processor
      self.print_prefix = "[MLSelect] "
      print(f"{self.print_prefix}Initialised")



    def runML(self, reco_mom, reco_time, t0err, nactive, nst, nopa, closest_crv_diff, mc_count):
      """ Currently a skeleton for a more useful BDT for signal selection /cut and count analysis
      """
      combined_t0err = []
      combined_nactive = []
      combined_recomom = []
      combined_recotime = []
      combined_nst = []
      combined_nopa = []
      combined_crv = []
      combined_mc_count = []
      combined_crv = []
      combined_labels = []
      
      for i, val in enumerate(reco_mom):
            val = ak.drop_none(val)
            if i == 0:
              reco_mom_signal = val.mask[mc_count[i] == 168]
              reco_mom_signal = np.array(ak.flatten(reco_mom_signal, axis=None))
            if i == 1:
              reco_mom_background = val.mask[mc_count[i] != 168]
              reco_mom_background = np.array(ak.flatten(reco_mom_background, axis=None))
      
      lists_in = [reco_mom_signal,reco_mom_background]
      labels = [1,0]
      for n, list_in in enumerate(lists_in):
        for i, val in enumerate(list_in):
          combined_recomom.append(val)
          combined_labels.append(labels[n])
      #print(len(combined_labels),len(combined_recomom))
      
      for i, val in enumerate(reco_time):
            val = ak.drop_none(val)
            if i == 0:
              reco_time_signal = val.mask[mc_count[i] == 168]
              reco_time_signal = np.array(ak.flatten(reco_time_signal, axis=None))
            if i == 1:
              reco_time_background = val.mask[mc_count[i] != 168]
              reco_time_background = np.array(ak.flatten(reco_time_background, axis=None))
              
      lists_in = [reco_time_signal,reco_time_background]
      for n, list_in in enumerate(lists_in):
        for i, val in enumerate(list_in):
          combined_recotime.append(val)
      #print(len(combined_labels),len(combined_recotime))
      
      for i, val in enumerate(nst):
            val = ak.drop_none(val)
            if i == 0:
              nst_signal = val.mask[mc_count[i] == 168]
              nst_signal = np.array(ak.flatten(nst_signal, axis=None))
            if i == 1:
              nst_background = val.mask[mc_count[i] != 168]
              nst_background = np.array(ak.flatten(nst_background, axis=None))
      lists_in = [nst_signal,nst_background]
      for n, list_in in enumerate(lists_in):
        for i, val in enumerate(list_in):
          combined_nst.append(val)
      #print(len(combined_labels),len(combined_nst))
         
      for i, val in enumerate(nopa):
          val = ak.drop_none(val)
          if i == 0:
            nopa_signal = val.mask[mc_count[i] == 168]
            nopa_signal = np.array(ak.flatten(nopa_signal, axis=None))
          if i == 1:
            nopa_background = val.mask[mc_count[i] != 168]
            nopa_background = np.array(ak.flatten(nopa_background, axis=None)) 

      
      lists_in = [nopa_signal,nopa_background]
      for n, list_in in enumerate(lists_in):
        for i, val in enumerate(list_in):
          combined_nopa.append(val)
      #print(len(combined_labels),len(combined_nopa))


      for i, val in enumerate(closest_crv_diff):
            large_number = 99999

            mask = (mc_count[i] == 168)

            if i == 0:
              signal_data = val[mask]
              mins_per_event = ak.min(val, axis=0, keepdims=False)
              masked_events = val[mask] 
              flattened_twice = ak.flatten(ak.flatten(masked_events, axis=2), axis=1)
              signal_mins = ak.min(flattened_twice, axis=-1)
              closest_crv_diff_signal = ak.fill_none(signal_mins, large_number)
              closest_crv_diff_signal = np.array(closest_crv_diff_signal)
              

            if i == 1:
              mask = (mc_count[i] != 168)
              background_events = val[mask]
              flattened_twice = ak.flatten(ak.flatten(background_events, axis=2), axis=1)
              background_mins = ak.min(flattened_twice, axis=-1)
              closest_crv_diff_background = ak.fill_none(background_mins, large_number)
              
              closest_crv_diff_background = np.array(closest_crv_diff_background)
              
      
      lists_in = [closest_crv_diff_signal,closest_crv_diff_background]

      for n, list_in in enumerate(lists_in):
        for i, val in enumerate(list_in):
          combined_crv.append(val)
      print("array combined lengths", len(combined_labels),len(combined_crv))
      
      
      # Create a dictionary of features
      data = {
          'recomom': combined_recomom,
          'recotime': combined_recotime,
          #'t0err': combined_t0err,
          #'nactive': combined_nactive,
          'nst' : combined_nst,
          'nopa' : combined_nopa,
          'closest_crv_diff': combined_crv
      }
      
      # Convert the dictionary to a pandas DataFrame
      X = pd.DataFrame(data)
      y = pd.Series(combined_labels) 

      print("Feature DataFrame (X):")
      print(X.head())
      print("\nTarget Series (y):")
      print(y.head())


      X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
      model = xgb.XGBClassifier(objective="binary:logistic", use_label_encoder=False, eval_metric="logloss")

      model.fit(X_train, y_train)


      predictions = model.predict(X_test)
      print("\nPredictions:", predictions)
      


      # Evaluate the model
      print("\nAccuracy:", accuracy_score(y_test, predictions))
      print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
      print("\nClassification Report:\n", classification_report(y_test, predictions))
      from sklearn.metrics import roc_curve, auc
      predictions_proba = model.predict_proba(X_test)[:, 1]

      # Calculate the ROC curve points
      fpr, tpr, thresholds = roc_curve(y_test, predictions_proba)
      roc_auc = auc(fpr, tpr)

      # Plot the ROC curve
      plt.figure(figsize=(8, 6))
      plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
      plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.05])
      plt.xlabel('False Positive Rate')
      plt.ylabel('True Positive Rate')
      plt.title('Receiver Operating Characteristic (ROC) Curve')
      plt.legend(loc="lower right")
      plt.show()
      
      # Create the Feature Importance Plot
      fig, ax = plt.subplots(figsize=(10, 8))
      # The plot_importance function uses the feature names from the DataFrame X_train
      xgb.plot_importance(model, ax=ax, importance_type='gain') # 'gain' is a common metric
      plt.title('Feature Importance Plot')
      plt.show()
      
      # Separate the scores into signal and background groups based on the true labels
      predictions_proba = model.predict_proba(X_test)
      signal_scores = predictions_proba[:, 1]
      signal_scores_true = signal_scores[y_test == 1]
      background_scores_true = signal_scores[y_test == 0]

      plt.figure(figsize=(10, 6))
      plt.hist(background_scores_true, bins=50, alpha=0.2, label='Background (True 0)', color='blue', density=True)
      plt.hist(signal_scores_true, bins=50, alpha=0.2, label='Signal (True 1)', color='red', density=True)
      plt.xlabel('Model Score (Probability of Signal)')
      plt.ylabel('Normalized Frequency')
      plt.title('Signal vs. Background Score Distribution')
      ax.set_yscale('log')
      plt.legend()
      plt.show()
      
      plt.figure(figsize=(10, 6))
      plt.hist(background_scores_true, range=(0.95,1.0), bins=50, alpha=0.2, label='Background (True 0)', color='blue', density=True)
      plt.hist(signal_scores_true, range=(0.95,1.0),bins=50, alpha=0.2, label='Signal (True 1)', color='red', density=True)
      plt.xlabel('Model Score (Probability of Signal)')
      plt.ylabel('Normalized Frequency')
      plt.title('Signal vs. Background Score Distribution')
      ax.set_yscale('log')
      plt.legend()
      plt.show()
      

      from sklearn.metrics import precision_recall_curve, auc

      predictions_proba = model.predict_proba(X_test)[:, 1]

      num_thresholds = 100
      thresholds = np.linspace(0.0, 1.0, num_thresholds)

      signal_efficiencies = []
      background_efficiencies = []
      background_rejections = [] # Bkg Rejection = 1 - Bkg Efficiency

      for t in thresholds:
          y_pred_at_threshold = (predictions_proba >= t).astype(int)

          tn, fp, fn, tp = confusion_matrix(y_test, y_pred_at_threshold).ravel()
          
          # Calculate efficiencies
          sig_eff = tp / (tp + fn) if (tp + fn) > 0 else 0
          bkg_eff = fp / (fp + tn) if (fp + tn) > 0 else 0
          
          signal_efficiencies.append(sig_eff)
          background_efficiencies.append(bkg_eff)
          background_rejections.append(1 - bkg_eff)


      plt.figure(figsize=(10, 6))

      # Plot Signal Efficiency vs Threshold
      plt.plot(thresholds, signal_efficiencies, label='Signal Efficiency (TPR)', color='red', lw=2)

      # Plot Background Efficiency vs Threshold
      plt.plot(thresholds, background_rejections, label='Background Rejection (TNR)', color='blue', linestyle='--', lw=2)

      # plt.plot(thresholds, background_efficiencies, label='Background Efficiency (FPR)', color='blue', lw=2)


      plt.xlabel('Classifier Score Threshold')
      plt.ylabel('Efficiency / Rejection Value')
      plt.title('TMVA-Style: Signal Efficiency & Background Rejection vs. Threshold')
      plt.xlim([0.0, 1.0])
      plt.ylim([0.0, 1.05])
      plt.legend()
      plt.grid(True)
      plt.show()

