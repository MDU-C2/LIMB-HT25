## EMG LSTM Pipeline (PyTorch)

Minimal, modular PyTorch pipeline for classifying hand state (open/close) from EMG sequences.

### Data Format

**Input**: CSV files in `data/raw_data/{subject_id}/` (e.g., `data/raw_data/S2/`)
- Each CSV file contains 80 windows (rows)
- Each row = 1 window with 400 EMG samples (v0 to v399)
- Data collection protocol:
  - Windows 0-19: Rest (label 0 - Hand_Open)
  - Windows 20-59: Gesture (label 1 - Hand_Close)
  - Windows 60-79: Rest (label 0 - Hand_Open)

**Output**: NPZ file with sequences ready for LSTM training
- `X`: shape `(N, seq_len, features)` - sequences of features
- `y`: shape `(N,)` - labels (0 = Hand_Open, 1 = Hand_Close)

### Modules

- **Data Processing**:
  - `generate_csv_dataset.py`: CSV → NPZ conversion script
  - `emg_processing_utils.py`: Feature extraction and sequence creation utilities

- **Training Pipeline**:
  - `main.py`: Main CLI entry point with `train` command
  - `datasets.py`: `EMGSequenceDataset`, `load_standardize_splits(npz_path)`
  - `models.py`: `SimpleLSTM`, `get_simple_lstm(input_dim, hidden_dim=32, ...)`
  - `training.py`: Training utility functions (`train_epoch`, `eval_model`)
  
- **Hardware Verification**:
  - `capture_verification.py`: Capture 80 windows (20 rest → 40 grip → 20 rest) and display plot

- **Testing**:
  - `test_lstm.py`: Unit tests for models, datasets, and training

### Installation

```bash
# Core dependencies
pip install torch numpy scikit-learn pandas scipy

# For live inference (BLE connection)
pip install bleak

# For capture mode (spacebar detection and plotting)
pip install keyboard matplotlib
```

**Note**: On macOS, you may need to grant Terminal/Python accessibility permissions for keyboard detection to work. On Linux, you may need to run with `sudo` or configure permissions.

### Step-by-Step Guide

Follow these steps to process your EMG data and train an LSTM model:

#### Step 1: Prepare Your Data

Ensure your CSV files are organized in the correct directory structure:

```
src/emg/
  data/
    raw_data/
      S0/
        *.csv
      S1/
        *.csv
      S2/
        *.csv
      ...
    processed_data/
      emg_sequences_*.npz
```

Each CSV file should contain:
- 80 rows (windows)
- 400 EMG columns (v0 to v399)
- Columns: `Raw_Label`, `Timestamp`, `v0`, `v1`, ..., `v399`

#### Step 2: Generate Dataset from CSV Files

Navigate to your project root directory and run the dataset generation script:

```bash
# Basic usage - process all subjects with default settings
python -m src.emg.generate_csv_dataset --data-path data --num-channels 8

# Process specific subjects only
python -m src.emg.generate_csv_dataset --data-path data --subjects S2 S3 --num-channels 8

# Extract single channel (e.g., channel 1)
python -m src.emg.generate_csv_dataset --data-path data --channel 1 --num-channels 8

# Custom output filename
python -m src.emg.generate_csv_dataset --data-path data --output my_emg_dataset.npz --num-channels 8
```

**What happens:**
1. Script reads all CSV files from specified subject folders
2. Segments each file into rest/gesture/rest windows (20/40/20)
3. Extracts 6 time-domain features per channel from each window
4. Creates overlapping sequences of length 10 (default)
5. Saves everything to an NPZ file (e.g., `emg_sequences_ch1.npz`)

**Expected output:**
```
Looking for data in: /path/to/src/emg/data/raw_data
Processing S2: 60 CSV files found in /path/to/src/emg/data/raw_data/S2
...
======================================================================
Dataset Creation Complete!
======================================================================
Output: /path/to/src/emg/data/processed_data/emg_sequences_ch1.npz
Sequences shape (X): (8500, 10, 6)
Labels shape (y): (8500,)
Label distribution:
  Hand_Open (label 0): 4250 sequences
  Hand_Close (label 1): 4250 sequences
```

#### Step 3: Verify Dataset

Check that the NPZ file was created successfully:

```bash
# From Python
import numpy as np
data = np.load('src/emg/emg_sequences_ch1.npz')
print(f"X shape: {data['X'].shape}")  # Should be (N, seq_len, features)
print(f"y shape: {data['y'].shape}")  # Should be (N,)
print(f"Labels: {np.unique(data['y'], return_counts=True)}")
```

#### Step 4: Train the LSTM Model

Train your model using the generated dataset:

```bash
# Basic training with default parameters
python -m src.emg.main train

# Training with custom parameters
python -m src.emg.main train \
    --dataset data/processed_data/emg_sequences_all.npz \
    --epochs 20 \
    --batch-size 64 \
    --hidden-dim 64 \
    --lr 0.001
```

**After training**, the model, scaler, and config will be automatically saved to `checkpoints/`:
- `model_YYYYMMDD_HHMMSS.pt` - Trained model weights
- `scaler_YYYYMMDD_HHMMSS.pkl` - StandardScaler for feature normalization
- `config_YYYYMMDD_HHMMSS.json` - Model configuration

**What happens:**
1. Loads NPZ dataset
2. Splits into train/val/test (70%/15%/15%)
3. Standardizes features (fits scaler on train, applies to val/test)
4. Creates PyTorch DataLoaders
5. Initializes LSTM model
6. Trains for specified epochs with Adam optimizer
7. Prints training progress after each epoch
8. Evaluates on test set
9. Shows inference demo on 5 random samples

**Expected output:**
```
Using device: cpu
Training configuration:
  Dataset: data/processed_data/emg_sequences_all.npz
  Epochs: 20
  Batch size: 64
  Hidden dim: 64
  Learning rate: 0.001
  Features per window: 6
  Number of classes: 2

Epoch 1: Train acc=0.650 loss=0.6234 | Val acc=0.680 loss=0.5892
Epoch 2: Train acc=0.720 loss=0.5123 | Val acc=0.750 loss=0.4789
...
Epoch 20: Train acc=0.920 loss=0.2101 | Val acc=0.890 loss=0.2456

Test Accuracy: 88.50% | Test Loss: 0.2512

Inference Demo:
  Sample 1: True=Hand_Open, Pred=Hand_Open
  Sample 2: True=Hand_Close, Pred=Hand_Close
  ...

Model saved to: checkpoints/model_20251219_120000.pt
Scaler saved to: checkpoints/scaler_20251219_120000.pkl
Config saved to: checkpoints/config_20251219_120000.json

To run live inference, use:
  python -m src.emg.main predict --model checkpoints/model_20251219_120000.pt --scaler checkpoints/scaler_20251219_120000.pkl --config checkpoints/config_20251219_120000.json
```

#### Step 5: Interpret Results

- **Training accuracy**: How well the model fits the training data
- **Validation accuracy**: How well the model generalizes (most important)
- **Test accuracy**: Final performance on unseen data
- **Loss values**: Should decrease over epochs (lower is better)

**Good signs:**
- Validation accuracy increases steadily
- Training and validation accuracies are close (no overfitting)
- Test accuracy matches validation accuracy

**Warning signs:**
- Large gap between train and val accuracy → overfitting
- Accuracy not improving → may need more data or different hyperparameters
- Loss not decreasing → learning rate may be too high/low

#### Step 6: Hardware Verification Capture

Verify that your hardware is working correctly by capturing a structured sequence:

```bash
python -m src.emg.capture_verification
```

**What happens:**
1. Connects to your BLE device ("LIMBServer")
2. Waits for data stream to stabilize
3. Captures exactly 80 windows in sequence:
   - Windows 0-19: Rest phase (2 seconds) - keep hand at rest
   - Windows 20-59: Grip phase (4 seconds) - grip a mug/object
   - Windows 60-79: Rest phase (2 seconds) - release and rest
4. **Automatically displays a plot** of the captured EMG data with phase zones highlighted
   - Green zones: Rest periods (should show lower EMG activity)
   - Orange zone: Grip period (should show higher EMG activity)
   - Verify the signal looks reasonable and hardware is working

**Output example:**
```
======================================================================
EMG Hardware Verification Capture
======================================================================
Target: 80 windows (20 rest → 40 grip → 20 rest)
======================================================================

Scanning for device 'LIMBServer'...
Connecting to BF7B8309-C97C-6463-2908-8935DBCC47DC...
Waiting for data stream to stabilize...
✓ Receiving data: 100 packets/sec, 10 windows/sec
Clearing initial data...

--- CAPTURING 80 WINDOWS ---
Instructions:
  1. Keep your hand at REST for 2 seconds (windows 0-19)
  2. GRIP a mug/object for 4 seconds (windows 20-59)
  3. Release and keep hand at REST for 2 seconds (windows 60-79)

Starting capture in 2 seconds...
CAPTURING NOW!

Progress: 80/80 windows captured

Capture complete! Captured 80 windows.

--- Creating verification plot ---
Plot displayed. Verify that:
  - Green zones show rest (low EMG activity)
  - Orange zone shows grip (higher EMG activity)
  - Signal looks reasonable (no obvious errors)

Close the plot window to exit.
[Plot window opens showing EMG signal with phase zones highlighted]
```

**Requirements:**
- Install matplotlib library: `pip install matplotlib`
- BLE device "LIMBServer" must be connected and sending data

**Notes:**
- The script automatically waits for data stream to stabilize before capturing
- Follow the instructions: rest → grip → rest
- The system will capture exactly 80 windows
- After capture, a plot automatically displays showing the EMG signal with phase zones
- **Verify the plot** shows reasonable EMG activity (rest zones should be lower, grip zone higher)
- Close the plot window to exit

#### Step 7: Experiment and Iterate

Try different configurations to improve performance:

```bash
# Try different sequence lengths
python -m src.emg.generate_csv_dataset --seq-len 15 --num-channels 8

# Try different model architectures
python -m src.emg.main train --dataset data/processed_data/emg_sequences_all.npz --hidden-dim 128

# Try different learning rates
python -m src.emg.main train --dataset data/processed_data/emg_sequences_all.npz --lr 0.0001
```

### Quick Start Example

For a complete end-to-end example:

```bash
# 1. Generate dataset
cd /path/to/LIMB-HT25
python -m src.emg.generate_csv_dataset --num-channels 8 --channel 1

# 2. Train model
python -m src.emg.main train --dataset data/processed_data/emg_sequences_ch1.npz --epochs 10

# 3. Verify hardware (optional)
python -m src.emg.capture_verification

# 4. Check results
# Model checkpoints saved in: src/emg/checkpoints/
# Training logs show accuracy and loss metrics
```

### Common Workflows

**Workflow 1: Single Subject Analysis**
```bash
# Process one subject
python -m src.emg.generate_csv_dataset --subjects S2 --num-channels 8

# Train on single subject data
python -m src.emg.main train --dataset data/processed_data/emg_sequences_all.npz --epochs 15
```

**Workflow 2: Multi-Channel Comparison**
```bash
# Generate datasets for different channels
python -m src.emg.generate_csv_dataset --channel 1 --num-channels 8
python -m src.emg.generate_csv_dataset --channel 2 --num-channels 8

# Compare performance
python -m src.emg.main train --dataset data/processed_data/emg_sequences_ch1.npz
python -m src.emg.main train --dataset data/processed_data/emg_sequences_ch2.npz
```

**Workflow 3: Hyperparameter Tuning**
```bash
# Generate dataset once
python -m src.emg.generate_csv_dataset --num-channels 8

# Try different hyperparameters
python -m src.emg.main train --dataset data/processed_data/emg_sequences_all.npz --hidden-dim 32 --lr 0.001
python -m src.emg.main train --dataset data/processed_data/emg_sequences_all.npz --hidden-dim 64 --lr 0.001
python -m src.emg.main train --dataset data/processed_data/emg_sequences_all.npz --hidden-dim 32 --lr 0.0005
```

### Usage

#### 1. Generate Dataset from CSV Files

Process CSV files and create NPZ dataset:

```bash
# Process all subjects (S0-S7), all channels
python -m src.emg.generate_csv_dataset --data-path data --num-channels 8

# Process specific subjects, single channel
python -m src.emg.generate_csv_dataset --data-path data --channel 1 --subjects S2 S3 --num-channels 8

# Custom output filename and sequence length
python -m src.emg.generate_csv_dataset --data-path data --output my_dataset.npz --num-channels 8 --seq-len 10
```

**Parameters**:
- `--data-path`: Path to folder containing subject folders (default: `data/raw_data`)
- `--num-channels`: Number of EMG channels for reshaping 400 samples (default: 8)
- `--channel`: Extract single channel (1-based) or None for all channels
- `--subjects`: List of subjects to process (default: S0-S7)
- `--seq-len`: Sequence length for LSTM (default: 10)
- `--output`: Output NPZ filename (default: auto-generated)

#### 2. Train LSTM Model

Train/evaluate the LSTM:

```bash
# Basic training with default parameters
python -m src.emg.main train

# Training with custom parameters
python -m src.emg.main train --dataset data/processed_data/emg_sequences_all.npz --epochs 10 --batch-size 64
```

**Training Parameters**:
- `--dataset`: Path to NPZ dataset file (default: `data/processed_data/emg_sequences_all.npz`)
- `--epochs`: Number of training epochs (default: 5)
- `--batch-size`: Batch size for training (default: 64)
- `--hidden-dim`: Hidden dimension for LSTM (default: 32)
- `--lr`: Learning rate (default: 1e-3)

#### 3. Hardware Verification Capture

Verify hardware is working correctly:

```bash
python -m src.emg.capture_verification
```

**What it does:**
- Connects to BLE device "LIMBServer"
- Captures 80 windows (20 rest → 40 grip → 20 rest)
- Displays a plot with phase zones highlighted
- Verifies EMG signal looks reasonable

**Requirements:**
- BLE device "LIMBServer" must be connected
- Matplotlib installed: `pip install matplotlib`

**Training will:**
- Load and standardize data (fit scaler on train, apply to val/test)
- Split data into train/val/test sets (70%/15%/15%)
- Train for specified epochs (Adam optimizer, CrossEntropy loss)
- Print validation metrics after each epoch
- Evaluate on test set and show final metrics
- Show inference demo on 5 random test samples
- **Automatically save** model, scaler, and config to `checkpoints/` directory

### Pipeline Flow

```
CSV Files (data/raw_data/S*/)
    ↓
generate_csv_dataset.py
    ↓ (feature extraction, sequence creation)
NPZ File (data/processed_data/emg_sequences_*.npz)
    ↓
main.py train
    ↓ (load, standardize, split)
LSTM Training
    ↓
Trained Model + Scaler + Config (checkpoints/)
    ↓
capture_verification.py
    ↓ (connect to BLE, capture 80 windows)
Hardware Verification Plot
```

### Customization

**Dataset Generation** (`generate_csv_dataset.py`):
- `--num-channels`: Number of EMG channels (for reshaping 400 samples)
  - 8 channels: reshape to (80, 8, 50) per window
  - 2 channels: reshape to (80, 2, 200) per window
  - 1 channel: reshape to (80, 1, 400) per window
- `--seq-len`: Sequence length for LSTM (default: 10)
- `--channel`: Extract single channel (1-based) or all channels

**Model** (`models.py`):
- Hidden size, number of layers, dropout rate
- Modify `SimpleLSTM` class or `get_simple_lstm()` function

**Training** (`main.py train`):
- Epochs, batch size, learning rate
- Pass via command line arguments or modify defaults in `main.py`

### Data Structure

Each CSV file represents one capture session:
- **80 windows total** per file
- **400 EMG samples** per window (v0 to v399)
- **Segmentation**:
  - First 20 windows: Rest (Hand_Open)
  - Middle 40 windows: Gesture (Hand_Close)
  - Last 20 windows: Rest (Hand_Open)

Feature extraction:
- 6 time-domain features per channel: MAV, RMS, WL, ZC, SSC, VAR
- Features are extracted from each window
- Sequences are created with overlapping windows for LSTM input

### Notes

- Data is pre-windowed at collection time (no need for windowing step)
- Feature extraction happens per window (6 features × num_channels)
- Sequences are created with overlapping windows for LSTM training
- Default sequence length is 10 windows per sequence
- Train/val/test split: 70%/15%/15% with stratification

### Troubleshooting

**Import errors**: Make sure you're running from the project root:
```bash
# From project root
python -m src.emg.generate_csv_dataset
```

**File not found**: Check that CSV files are in `data/raw_data/{subject_id}/` folders:
```
data/
  raw_data/
    S2/
      *.csv
    S3/
      *.csv
```

**BLE connection issues**: 
- Make sure your BLE device is powered on and advertising
- Check that the device name matches "LIMBServer"
- Verify Bluetooth is enabled on your system
- Try running with elevated permissions if on Linux

**Shape errors**: Verify `--num-channels` matches your data structure (400 samples must be divisible by num_channels)

**Model loading errors**: Ensure all three files are from the same training run:
- Model file (`.pt`)
- Scaler file (`.pkl`) 
- Config file (`.json`)

They should have matching timestamps in their filenames.

**Hardware verification capture not working**: 
- Verify the BLE device is connected and sending data (check debug output)
- Ensure the device name matches "LIMBServer"
- Check that packets are being received (should show packets/sec in output)
- If timeout occurs, verify device is actively streaming data
- Make sure matplotlib is installed for plotting
