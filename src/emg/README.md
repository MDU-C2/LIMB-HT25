## EMG LSTM (PyTorch)

Minimal, modular PyTorch pipeline for classifying hand state (open/close) from EMG sequences.

### Modules
- `datasets.py`: `EMGSequenceDataset`, `load_standardize_splits(npz_path)`
- `models.py`: `SimpleLSTM`, `get_simple_lstm(input_dim, hidden_dim=32, ...)`
- `training.py`: `train_epoch`, `eval_model`
- `train_lstm.py`: Training/evaluation entry wiring data, model, and loops
- `main.py`: CLI entry point (`--train`)

### Expected Dataset
NPZ file created upstream (see `EMG_signal_processing`):
- `X`: shape `(N, seq_len, features)`
- `y`: shape `(N,)` with labels (0 = Hand_Open, 1 = Hand_Close)
Default path in code: `../../EMG_signal_processing/emg_sequences_dataset.npz`

### Install
Use your existing environment or install minimal deps:
```bash
pip install torch numpy scikit-learn
```

### Run
Train/evaluate the LSTM:
```bash
python -m src.emg.main --train
```

This will:
- Load and standardize data (fit on train, apply to val/test)
- Train for 5 epochs (Adam, CrossEntropy)
- Print validation after each epoch and final test metrics
- Show a small inference demo on 5 samples

### Customize
- Change dataset path in `train_lstm.py`
- Adjust model in `models.py` (hidden size, layers, dropout)
- Modify training hyperparams in `train_lstm.py` (epochs, batch size, LR)


