# main.py

# Importamos la función desde nuestro nuevo módulo
import emg_utils as EMGU
from scipy.io import loadmat
import os
import glob
import numpy as np

#####################################################################

# Load data
# mat_data_HC1 = loadmat("C:/Users/Usuario/Documents/Universidad_MDU/database_sEMG_plus_IMUs_kaggel/FORS-EMG Dataset/FORS-EMG Dataset/FORS-EMG/Subject1/rest/Hand_Close-1.mat")
# data_HC1 = mat_data_HC1["value"]
# print(data_HC1.shape)

# Relevant params
# fs = 985
# num_channels = 8
# lowcut = 20.0
# highcut = 450.0
# notch_freq = 50.0
# window_size_ms = 200
# overlap_ms = 50

# # Aplying preprocessing 
# prepr_data_HC1 = EMGU.preprocess_emg_signal(data_HC1, fs, lowcut, highcut, notch_freq)
# print(prepr_data_HC1.shape)

# # Visualize EMG channels
# # EMGU.plot_emg_channels(prepr_data_HC1, fs, 8)

# # Visualize EMG PSD
# # EMGU.plot_psd_comparison(data_HC1, prepr_data_HC1, fs)

# # Windowing
# # Labels [HC:1, HO:0]
# windows, labels = EMGU.create_windows(prepr_data_HC1, 0, fs, 200, 50)
# print(windows.shape)

# # visualize some windows
# # EMGU.plot_emg_windows(windows, [0, 10, 40])

# # Feature extraction

# feat_samp = EMGU.extract_time_domain_features(windows, data_HC1.shape[0])
# print(feat_samp.shape)

#####################################################################

#CHECK THE DIRECTORY YOU ARE AND FROM THERE, FOUND RELATIVE PATH

RELATIVE_DATA_PATH = "DV474/sEMG_project/data_rest"
# C:\Users\Usuario\Documents\Universidad_MDU\DV474\sEMG_project\data_rest
BASE_PATH = os.path.abspath(RELATIVE_DATA_PATH) 
print(BASE_PATH)
SUBJECTS_TO_PROCESS = ["S1", "S2"]  # << SELECT the subjects 
# GESTURES_TO_PROCESS = ["Hand_Close", "Hand_Open"]  # << SELECT the gestures you want
GESTURES_TO_PROCESS = ["Hand_Close"]  # << SELECT the gestures 
LABELS_MAP = {"Hand_Close": 1, "Hand_Open": 0} 

# --- Define pipeline parameters ---
fs = 985
num_channels = 8
lowcut = 20.0
highcut = 450.0
notch_freq = 50.0
window_size_ms = 200
overlap_ms = 50

all_features_X = []
all_labels_y = []

for subject_id in SUBJECTS_TO_PROCESS:

    subject_path = os.path.join(BASE_PATH, subject_id)
    print(subject_path)
#     print(f"[INFO] Processing Subject: {subject_id}")
    
# #     # --- Loop through each selected gesture ---
    for gesture_name in GESTURES_TO_PROCESS:
        
#         # --- Find all .mat files for this subject and gesture ---
        search_pattern = os.path.join(subject_path, f"{gesture_name}-*.mat")
        print(search_pattern)
        files_found = glob.glob(search_pattern)
        
        print(f"  [INFO] Found {len(files_found)} files for gesture: {gesture_name}")
        
#         # --- Process each file ---
        for file_path in files_found:
#             # --- APPLY THE FULL PIPELINE ---
            
#             # a. Load the signal
            raw_signal = loadmat(file_path)["value"]
            print(raw_signal.shape)
            
#             # b. Preprocess the signal
            preprocessed_signal = EMGU.preprocess_emg_signal(raw_signal, fs, lowcut, highcut, notch_freq)
            print(preprocessed_signal.shape)
            
#             # c. Create windows and labels
            label = LABELS_MAP[gesture_name]
            windows, labels = EMGU.create_windows(preprocessed_signal, label, fs, window_size_ms, overlap_ms)
            print(windows.shape)
            
#             # d. Extract time-domain features
            features = EMGU.extract_time_domain_features(windows, num_channels)
            print(features.shape)
            print()
            print("----------------------")
            print()
            
#             # e. Append the results to our master lists
            all_features_X.append(features)
            all_labels_y.append(labels)

if all_features_X:
    X_final = np.vstack(all_features_X)
    y_final = np.concatenate(all_labels_y)

    print("\n--- Dataset Creation Complete! ---")
    print(f"Final feature matrix shape (X): {X_final.shape}")
    print(f"Final labels vector shape (y): {y_final.shape}")

    # np.savez("emg_gestures_dataset.npz", X=X_final, y=y_final)
    # print("Dataset saved to 'emg_gestures_dataset.npz'")
else:
    print("\n[ERROR] No data was processed. Please check your BASE_PATH and SUBJECTS/GESTURES lists.")








