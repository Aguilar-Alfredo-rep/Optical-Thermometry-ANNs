# TEMPERATURE PREDICTION (CNN) FOR ONE SPECTRUM
import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import glob, json
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
from CNN import SelfAttention

#-----------------------------------------------------------------------------------------------------------
# Compatibility patch for Keras 3 with models saved in h5 format
from tensorflow.keras.layers import Dense
_original_dense_from_config = Dense.from_config
@classmethod
def _patched_dense_from_config(cls, config):
    config.pop("quantization_config", None)
    return _original_dense_from_config(config)
Dense.from_config = _patched_dense_from_config
#-----------------------------------------------------------------------------------------------------------

# routes
##########################################################################
MODEL_PATH    = glob.glob("resultados/*.h5")[0]
SCALER_X_PATH = "resultados/scaler_x.pkl"
SCALER_Y_PATH = "resultados/scaler_y.pkl"
META_PATH     = "resultados/metadata.json"
WAVES_PATH    = "wavelengths.csv"
SPEC_PATH     = "3640.csv"                          # -------------------------------> NAME OF THE SPECTRUM!

##########################################################################
scaler_x = joblib.load(SCALER_X_PATH)
scaler_y = joblib.load(SCALER_Y_PATH)
meta = json.load(open(META_PATH, "r", encoding="utf-8"))

lam_min, lam_max = float(meta["λ_min"]), float(meta["λ_max"])
offset_T         = float(meta.get("offset_T"))

lmbdas = pd.read_csv(WAVES_PATH, header=None).iloc[:, 0].values.astype(float)
mask = (lmbdas >= lam_min) & (lmbdas <= lam_max)
lmbdas_rec = lmbdas[mask]

##########################################################################
model = load_model(MODEL_PATH, custom_objects={"SelfAttention": SelfAttention})

spec_raw = pd.read_csv(SPEC_PATH, header=None).iloc[:, 0].values.astype(float)
spec_rec = spec_raw[mask]

X_in = np.vstack([lmbdas_rec, spec_rec]).T.astype(np.float32)           # (Ni, 2)

X_scaled = scaler_x.transform(X_in.reshape(-1, 1)).reshape(X_in.shape)  # (Ni, 2)
X_scaled = X_scaled.reshape(1, X_scaled.shape[0], X_scaled.shape[1])    # (1, Ni, 2)

T_pred_C = scaler_y.inverse_transform(model.predict(X_scaled, verbose=0)).ravel()[0]
T_corr_C = T_pred_C - offset_T


##########################################################################
print(f"Temperature prediction      = {T_pred_C:.6f} °C")


print(f"offset_T    = {offset_T:.6f} °C")
print(f"Offset-corrected temperature prediction = {T_corr_C:.6f} °C")
##########################################################################
