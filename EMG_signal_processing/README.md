## EMG Signal Processing

End-to-end pipeline for sEMG preprocessing, feature extraction, sequence dataset creation, and a simple LSTM classifier.

### Contents
- `emg_utils.py`: Preprocessing, windowing, feature extraction, and sequence utilities.
- `create_sequence_dataset.py`: Walks a dataset directory, builds features and overlapping sequences, and saves `emg_sequences_dataset.npz`.
- `simple_lstm.py`: Loads the NPZ dataset, standardizes features, trains a small Keras LSTM, evaluates, and prints a quick inference sample.

### Installation
Create a Python environment (3.9+ recommended) and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install numpy scipy matplotlib scikit-learn tensorflow
```

If you need to read MATLAB `.mat` files, ensure SciPy is installed (already in the list). No extra packages are needed beyond SciPy for `loadmat`.

### Data Layout
`create_sequence_dataset.py` expects a directory structure like:

```text
EMG_signal_processing/
  data_rest/
    S1/
      Hand_Open-1.mat
      Hand_Open-2.mat
      Hand_Close-1.mat
      ...
    S2/
      Hand_Open-1.mat
      Hand_Close-1.mat
      ...
    ...
```

Each `.mat` file must contain a variable `value` with shape `(channels, samples)`.

Default parameters inside `create_sequence_dataset.py` (adjust as needed):
- **Subjects**: `S1` … `S19`
- **Gestures**: `Hand_Open`, `Hand_Close`
- **Label map**: `{"Hand_Close": 1, "Hand_Open": 0}`
- **Sampling rate** `fs`: 985 Hz
- **Channels**: 8
- **Band-pass**: 20–450 Hz and **Notch**: 50 Hz
- **Window**: 200 ms, **Overlap**: 50 ms
- **Sequence length**: 10 windows per sequence

You can change the base path, subjects, gestures, and processing parameters directly in `create_sequence_dataset.py`.

### Pipeline Overview
1. Preprocess each recording
   - Remove DC offset per channel
   - Band-pass filter (Butterworth) and 50 Hz notch
2. Window the preprocessed signal into overlapping windows
3. Extract 6 time-domain features per channel per window
   - MAV, RMS, WL, ZC, SSC, VAR
4. Build overlapping sequences of length `seq_len` (default 10)
5. Save dataset to `emg_sequences_dataset.npz` containing:
   - `X` with shape `(num_sequences, seq_len, features_per_window)`
   - `y` with shape `(num_sequences,)`

### How to Run
1) Create the sequence dataset

```bash
cd EMG_signal_processing
python create_sequence_dataset.py
```

On success, you should see logs like final shapes and a file `emg_sequences_dataset.npz` in the same folder.

2) Train and evaluate the LSTM

```bash
python simple_lstm.py
```

This will:
- Load `emg_sequences_dataset.npz`
- Split (train/val/test), standardize features, and train for 1 epoch (adjustable)
- Report test accuracy and loss
- Perform sample inference on 5 random test sequences

### Key Functions (`emg_utils.py`)
- `preprocess_emg_signal(signal, fs, lowcut, highcut, notch_freq, order=4)`
  - DC removal, band-pass, and notch filtering (applied channel-wise)
- `create_windows(signal, label, fs, window_size_ms, overlap_ms)`
  - Overlapping flattened windows and corresponding labels
- `extract_time_domain_features(window_matrix, num_channels)`
  - Produces a feature matrix with 6 features per channel per window
- `create_sequences(features, labels, seq_length=10)`
  - Generates overlapping sequences and sequence-level labels

### Adjusting Parameters
- Edit constants in `create_sequence_dataset.py` for dataset paths, gestures, labels, and signal parameters.
- Edit epochs, batch size, and LSTM units in `simple_lstm.py` for model/training tweaks.

### Notes and Tips
- Ensure the sampling rate `fs`, channels, and filters match your hardware/data.
- If your powerline frequency differs, change the notch (e.g., 60 Hz).
- `simple_lstm.py` uses a minimal model for demonstration; expand as needed (more layers, regularization, callbacks, etc.).
- Standardization is fit on the training set and applied to val/test consistently.

### License
Project-internal academic use. Adapt as needed for your institution or project.


