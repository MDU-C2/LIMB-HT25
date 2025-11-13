## EMG LSTM (PyTorch)

Modular PyTorch pipeline for classifying hand state (open/close) from EMG sequences with full training infrastructure.

### Modules
- `datasets.py`: `EMGSequenceDataset`, `load_standardize_splits(npz_path)` - returns data splits and scaler
- `models.py`: `SimpleLSTM`, `get_simple_lstm(input_dim, hidden_dim=32, ...)`
- `training.py`: `train_epoch`, `eval_model` - training and evaluation functions
- `train_lstm.py`: Full training pipeline with checkpointing, early stopping, and logging
- `create_sequence_dataset.py`: Create sequence dataset from raw .mat files
- `utils.py`: Model saving/loading, checkpoint management
- `main.py`: CLI entry point (`--train`, `--create-dataset`)

### Expected Dataset
NPZ file created upstream (see `EMG_signal_processing`):
- `X`: shape `(N, seq_len, features)`
- `y`: shape `(N,)` with labels (0 = Hand_Open, 1 = Hand_Close)
Default path: `EMG_signal_processing/emg_sequences_dataset.npz` (relative to `src/emg/`)

### Install
Use your existing environment or install minimal deps:
```bash
pip install torch numpy scikit-learn
```

### Run

#### Create Sequence Dataset from Raw .mat Files
```bash
python -m src.emg.main --create-dataset
```

With custom options:
```bash
python -m src.emg.main --create-dataset \
    --data-dir ./EMG_signal_processing/data_rest \
    --output ./EMG_signal_processing/emg_sequences_dataset.npz \
    --subjects S1 S2 S3 \
    --gestures Hand_Close Hand_Open \
    --seq-len 10
```

#### Basic Training
```bash
python -m src.emg.main --train
```

#### Advanced Training with Options
```bash
python -m src.emg.main --train \
    --epochs 50 \
    --batch-size 64 \
    --hidden-dim 64 \
    --dropout 0.5 \
    --lr 0.001 \
    --early-stopping \
    --patience 10 \
    --output-dir ./my_experiment
```

#### View All Options
```bash
python -m src.emg.main --train --help
python -m src.emg.main --create-dataset --help
```

### Key Features

#### 1. **Model Saving & Loading**
- Automatically saves best model, scaler, and config
- Checkpoint system for resuming training
- Saved to timestamped directory: `checkpoints/YYYYMMDD_HHMMSS/`

#### 2. **Reproducibility**
- Random seed control (default: 42)
- Deterministic training with `--seed` argument
- All random operations are seeded

#### 3. **Checkpointing & Early Stopping**
- Saves best model based on validation accuracy
- Early stopping with configurable patience
- Resume training from checkpoints with `--resume`

#### 4. **Configuration Management**
- All hyperparameters configurable via CLI
- Config saved with each model for reproducibility
- Training history saved as JSON

#### 5. **Comprehensive Logging**
- Console and file logging (saved to `training.log`)
- Detailed training progress tracking
- Training history saved for analysis

### Command-Line Arguments

**Dataset:**
- `--dataset`: Path to NPZ file (default: auto-detect)
- `--test-size`: Test set ratio (default: 0.3)
- `--val-ratio`: Validation ratio of temp split (default: 0.5)
- `--seed`: Random seed (default: 42)

**Model:**
- `--hidden-dim`: LSTM hidden dimension (default: 32)
- `--dropout`: Dropout rate (default: 0.5)

**Training:**
- `--epochs`: Number of epochs (default: 50)
- `--batch-size`: Training batch size (default: 64)
- `--val-batch-size`: Validation batch size (default: 128)
- `--lr`: Learning rate (default: 1e-3)

**Checkpointing:**
- `--output-dir`: Output directory (default: `checkpoints/TIMESTAMP`)
- `--resume`: Path to checkpoint to resume from
- `--save-best-only`: Only save best model (not every epoch)

**Early Stopping:**
- `--early-stopping`: Enable early stopping
- `--patience`: Patience epochs (default: 10)
- `--min-delta`: Minimum improvement threshold (default: 0.0)

### Output Structure

After training, you'll find in the output directory:
```
checkpoints/YYYYMMDD_HHMMSS/
├── best_model.pt          # Best model state dict
├── scaler.pkl             # Fitted StandardScaler
├── config.json            # Training configuration
├── checkpoint_best.pt     # Best checkpoint
├── checkpoint_latest.pt   # Latest checkpoint
├── training_history.json  # Training metrics per epoch
└── training.log           # Detailed training log
```

### Dataset Creation

The `create_sequence_dataset.py` script processes raw EMG `.mat` files and creates sequences for training:

**Default behavior:**
- Reads from `EMG_signal_processing/data_rest/`
- Processes subjects S1-S19
- Processes gestures: Hand_Close, Hand_Open
- Saves to `EMG_signal_processing/emg_sequences_dataset.npz`

**Key arguments:**
- `--data-dir`: Path to data directory with subject folders
- `--output`: Output NPZ file path
- `--subjects`: List of subjects to process (e.g., `S1 S2 S3`)
- `--gestures`: List of gestures (e.g., `Hand_Close Hand_Open`)
- `--seq-len`: Sequence length (default: 10)
- `--fs`: Sampling frequency (default: 985 Hz)
- `--window-size-ms`: Window size in ms (default: 200)
- `--overlap-ms`: Window overlap in ms (default: 50)

### Customize
- Model architecture: Edit `models.py`
- Training loop: Edit `train_lstm.py`
- Data processing: Edit `datasets.py`
- Dataset creation: Edit `create_sequence_dataset.py` or use CLI arguments
- All hyperparameters: Use CLI arguments


