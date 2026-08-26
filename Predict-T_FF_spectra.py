# TEMPERATURE PREDICTION (FF) FOR MULTIPLE SPECTRA (FOLDER)

import os, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import glob, json
import numpy as np
import pandas as pd
import joblib
from tqdm import tqdm
from tensorflow.keras.models import load_model

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
SPEC_DIR      = "Espectros_rename"   # carpeta con 1.csv, 2.csv, 4.csv, ...
##########################################################################

##########################################################################
scaler_x = joblib.load(SCALER_X_PATH)
scaler_y = joblib.load(SCALER_Y_PATH)
meta = json.load(open(META_PATH, "r", encoding="utf-8"))

lam_min, lam_max = float(meta["λ_min"]), float(meta["λ_max"])
offset_T         = float(meta.get("offset_T", 0.0))

lmbdas = pd.read_csv(WAVES_PATH, header=None).iloc[:, 0].values.astype(float)
mask = (lmbdas >= lam_min) & (lmbdas <= lam_max)
lmbdas_rec = lmbdas[mask]
##########################################################################

##########################################################################
model = load_model(MODEL_PATH)

# list of spectra (numerical order if the name is "N.csv")
spec_files = glob.glob(os.path.join(SPEC_DIR, "*.csv"))
def _numkey(p):
    b = os.path.splitext(os.path.basename(p))[0]
    try: return int(b)
    except: return b

spec_files = sorted(spec_files, key=_numkey)
##########################################################################

##########################################################################
# prediction batch (iterative)
rows = []
for fp in tqdm(spec_files, desc="Predicting  ...", ncols=100, leave=True, dynamic_ncols=True):
    spec_raw = pd.read_csv(fp, header=None).iloc[:, 0].values.astype(float)
    spec_rec = spec_raw[mask]

    X_in = np.vstack([lmbdas_rec, spec_rec]).T.reshape(1, -1).astype(np.float32)

    T_pred_C = scaler_y.inverse_transform(
        model.predict(scaler_x.transform(X_in), verbose=0)
    ).ravel()[0]
    T_corr_C = T_pred_C - offset_T

    rows.append((os.path.basename(fp), float(T_pred_C), float(T_corr_C)))
##########################################################################

##########################################################################
# output
out = pd.DataFrame(rows, columns=["espectro", "T_pred_C", "T_corr_C"])
out.to_csv("predictions_temperatures.csv", index=False)
##########################################################################

##########################################################################
print(out.head())
print(f"\nTotal spectra processed: {len(out)}")
print("Saving: predictions_temperatures.csv")
##########################################################################
