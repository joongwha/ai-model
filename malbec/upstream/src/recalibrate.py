import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score

from hparams import hparams_dict

class MergedIsotonicModel:
    def __init__(self, thresholds_x, thresholds_y):
        self.X_thresholds_ = thresholds_x
        self.y_thresholds_ = thresholds_y
        
    def predict(self, X):
        return np.interp(X, self.X_thresholds_, self.y_thresholds_)

    
def enforce_min_samples_per_bin(isotonic_model, X, min_samples=40):
    # Get isotonic thresholds and predictions
    X_thresholds = isotonic_model.X_thresholds_
    y_thresholds = isotonic_model.y_thresholds_
    
    # Calculate samples per bin
    bin_samples = np.histogram(X, bins=X_thresholds)[0]
    
    # Merge bins with insufficient samples
    merged_X = [X_thresholds[0]]
    merged_y = [y_thresholds[0]]
    current_samples = 0
    for i in range(1, len(X_thresholds)):
        current_samples += bin_samples[i-1]
        if current_samples >= min_samples:
            merged_X.append(X_thresholds[i])
            merged_y.append(y_thresholds[i])
            current_samples = 0
        else:
            # Merge with previous bin
            merged_X[-1] = X_thresholds[i]
            merged_y[-1] = y_thresholds[i]
    
    return MergedIsotonicModel(merged_X, merged_y)


def recalibrate_with_isotonic(X, y, min_samples=40, model_filename=""):
    """
    Recalibrate probabilities using isotonic regression.

    Parameters:
    X (array-like or pd.Series): Predictor variable (probabilities to be recalibrated).
    y (array-like or pd.Series): Binary outcome variable.
    model_filename (str): Filename to save the model.

    Returns:
    model: Trained isotonic regression model.
    recalibrated_probs (np.array): Recalibrated probabilities.
    auc_score (float): AUC score of the recalibrated model.
    """
    # Ensure X is a 1D array
    X = np.array(X).ravel()
    y = np.array(y)
    
    y_min = np.quantile(y,0.02)
    y_max = np.quantile(y,0.98)

    # Create and fit isotonic regression model
    isotonic_model = IsotonicRegression(out_of_bounds='clip',y_min=y_min, y_max=y_max )
    isotonic_model.fit(X, y)
    isotonic_model = enforce_min_samples_per_bin(isotonic_model, X, min_samples)

    # Predict recalibrated probabilities
    recalibrated_probs = isotonic_model.predict(X)

    # Compute AUC
    auc_score = roc_auc_score(y, recalibrated_probs)
    print(f"Recalibrated AUC Score: {auc_score:.2f}")

    # Save the model
    if model_filename:
        model_filename = f"../calibration_models/model_{model_filename}.pkl"
    else:
        model_filename = "../calibration_models/model_isotonic.pkl"
    joblib.dump(isotonic_model, model_filename)
    print(f"Model saved as {model_filename}")

    return isotonic_model, recalibrated_probs, auc_score
