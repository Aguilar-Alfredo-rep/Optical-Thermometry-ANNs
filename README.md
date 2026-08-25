# Optical-Thermometry-ANNs

This repository provides an implementation of physics-informed machine learning pipelines designed to predict temperatures via optical thermometry using various upconversion nanoparticles synthesized for this purpose, following the methodology described in the paper "Calibration versus training: improving upconversion thermometry through Bayesian optimization". Built entirely in Python with TensorFlow, the framework processes raw spectral signals through two distinct architectures: a convolutional model featuring a custom-designed physics-informed attention mechanism and an unconventional feedforward architecture, achieving sub-0.3°C RMSE on independent spectra. The objective is to compare performance against classical calibration methods from the literature, such as Boltzmann analysis and multilinear regression. For reference, all files corresponding to the F-highSNR experimental dataset detailed in the paper are uploaded for immediate use.

================================================================================

TECHNICAL INSTRUCTION: NEURAL NETWORK PIPELINES (CNN AND FF)
================================================================================

--------------------------------------------------------------------------------

1. GENERAL DESCRIPTION AND ARCHITECTURE

This work consists of two notebooks ready to be uploaded and used directly
on the Colab root without creating folders:

- CNN_Pipeline.ipynb: Notebook dedicated to the model based on Convolutional
  Neural Networks, which imports functions and classes from the complementary
  CNN.py module.
  
- FF_Pipeline.ipynb: Notebook dedicated to the Fully Connected (Feedforward)
  Neural Network model, which imports functions and classes from the
  complementary FF.py module.

All corresponding libraries are imported at the beginning. Both notebooks
are explicitly configured in their first cells for dependency loading,
in addition to data decompression and reference file reading.


--------------------------------------------------------------------------------

2. FILE STRUCTURE AND LOADING IN THE GOOGLE COLAB ROOT

For the correct operation of either pipeline, the following files must
be loaded directly into the Google Colab environment root (/content/)
(the cells guide you through this as they execute; nothing needs to be uploaded
beforehand, just execute in order following the instructions in the comments):

- Corresponding network module:
  * "CNN.py" (exclusive to CNN_Pipeline)
  * "FF.py" (exclusive to FF_Pipeline)
    
- Espectros.zip: Compressed file containing the "Espectros_rename/" directory.
  When unzipped in the environment, it generates a folder named "Espectros_rename"
  that stores individual spectrum files in CSV format
  (1.csv, 2.csv, 3.csv, ...).
  
- temperature_values.csv: Plain text file with reference temperature values
  corresponding to each acquisition.
  
- time_values.csv: Plain text file with measured times between
  successive experimental acquisitions.
  
- wavelengths.csv: File with wavelengths.

--------------------------------------------------------------------------------

3. ALIGNMENT, SEQUENTIALITY, AND DATA MAPPING

All experimental acquisition was performed in a strictly sequential and
ordered manner, so there is a one-to-one and rigorous correspondence between
the rows of the tabular files and the individual spectrum files:

- Spectra and Temperatures: The individual file "1.csv" from the
  Espectros_rename/ folder corresponds exactly to the first row (index 0)
  of the "temperature_values.csv" file. Successively, "2.csv" corresponds
  to row 1, "3.csv" to row 2, and so on until completing all records.
- Temporal Record: The "time_values.csv" file maintains the same
  
  sequential temporal and ordered relationship, being aligned row by row with the
  reference temperature records and the sequence in the numbers of the
  spectrum names.
  
- Wavelengths: Each individual spectrum CSV file contains an intensity vector
  of 1044 points. The intensities of each row correspond in an ordered manner,
  position by position, to the wavelength values listed in "wavelengths.csv"
  (which contains the 1044 values of the spectral axis).

--------------------------------------------------------------------------------

4. GOOGLE COLAB EXECUTION INSTRUCTIONS

A. Start a new Google Colab session and upload the respective notebook
   ("CNN_Pipeline.ipynb" or "FF_Pipeline.ipynb").
   
B. Load as the cells execute, guided by the comments:
   - The corresponding network module ("CNN.py" or "FF.py" depending on the
     notebook to use).
     
   - The compressed file "Espectros.zip" along with the files:
     "temperature_values.csv" and "wavelengths.csv" ("time_values.csv" is not
     strictly necessary).
     
   - NOTE: One of these cells handles automatically unzipping the ZIP file,
     verifying the space-temporal alignment described in this instruction
     (it is maintained from then on).
     
C. Spectra cropping: Limits are defined via λ_min and λ_max, filtering
   the combined matrix of wavelengths and intensities (data) for each
   i+1.csv file using a boolean mask (mask), storing only the selected
   range in the X array.
   
D. Data splitting: A sequential manual partition is performed where
   num_train_test splits the total set into X_train/X_test and y_train/y_test,
   and subsequently num_train_val subdivides the training block into
   X_train/X_val and y_train/y_val.
   
E. Continue executing the cells in sequential order. Guide yourself by the
   comments (both in the notebooks and in the modules).
   
F. Saving and downloading results: The trained model is stored in
   .h5 format inside the resultados/ folder, along with the scalers
   (scaler_x.pkl and scaler_y.pkl) using joblib and a metadata.json file with the
   key configuration parameters. Finally, the entire directory is compressed
   into a .zip file and Colab's download function is invoked to export
   the local results.
   
================================================================================

================================================================================

Once the model is trained and saved, inference can be performed either on a
single spectrum file or on an entire folder of spectra using specialized scripts
(e.g., Predict-T_CNN_spectrum.py and Predict-T_CNN_spectra.py). These scripts
automatically load the trained model (.h5), the data scalers (.pkl), and the
metadata configuration (.json) generated during training.

================================================================================

--------------------------------------------------------------------------------

5. PREDICTION FOR A SINGLE SPECTRUM 
   (Predict-T_CNN_spectrum.py or Predict-T_FF_spectrum.py)

- Purpose: Computes the temperature prediction for one isolated spectrum file
  (e.g., "3640.csv").
  
- Workflow:
  
  A. Loads configuration parameters (`λ_min`, `λ_max`, and `offset_T`) from
     `metadata.json`.
  
  B. Reads the reference wavelengths (`wavelengths.csv`) and applies the spectral
     mask to crop the signal to the exact trained window.
  
  C. Loads the trained model (including custom layers like `SelfAttention`).
  
  D. Reads the raw intensity vector, stacks it with the cropped wavelengths,
     scales the inputs using `scaler_x.pkl`, and reshapes it to match the network
     input dimensions.
  
  E. Performs inference, applies inverse scaling via `scaler_y.pkl`, and subtracts
     the thermal offset (`offset_T`) to print both the raw and corrected temperature
     predictions.

--------------------------------------------------------------------------------

6. PREDICTION FOR MULTIPLE SPECTRA IN A FOLDER 
   (Predict-T_CNN_spectra.py or Predict-T_FF_spectra.py)
   

- Purpose: Batch processes an entire directory containing multiple CSV spectra
  (stored in `Espectros_rename/`), generating an output table with predictions.
  
- Workflow:
  
  A. Automatically detects paths for the model, scalers, metadata, and the 
     spectrum directory.
  
  B. Sorts the spectrum files numerically (following the natural order of "N.csv").
  
  C. Iterates sequentially through each spectrum using a progress bar (`tqdm`),
     applying the wavelength mask, feature scaling, and model prediction.
  
  D. Computes both the predicted temperature (`T_pred_C`) and the offset-corrected
     temperature (`T_corr_C`) for every individual file.
  
  E. Compiles all results into a structured Pandas DataFrame and exports them
     to a CSV file named `predictions_temperatures.csv`.


Note: For these prediction scripts to function correctly (sections 5 and 6), the 
trained model .h5 and its associated files (scaler_x.pkl, scaler_y.pkl, and 
metadata.json) must be located inside the downloaded folder named resultados/, 
while the target files (Espectros_rename/ or individual .csv spectra), the 
wavelengths.csv reference file, and the architecture module (FF.py or CNN.py) 
must be located in the same root directory as the script.

================================================================================
