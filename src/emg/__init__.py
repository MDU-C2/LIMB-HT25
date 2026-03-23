"""
EMG LSTM Package

This package provides tools for processing EMG signals, creating sequence datasets,
and training LSTM models for hand gesture classification.

Main components:
- datasets: Dataset loading and preprocessing
- models: LSTM model definitions
- training: Training and evaluation functions
- utils: Model saving/loading utilities
- create_sequence_dataset: Script to create datasets from raw .mat files
- train_lstm: Main training script with checkpointing and early stopping
"""
from .datasets import EMGSequenceDataset, load_standardize_splits
from .models import get_simple_lstm, SimpleLSTM
from .training import train_epoch, eval_model

__all__ = [
    'EMGSequenceDataset',
    'load_standardize_splits',
    'get_simple_lstm',
    'SimpleLSTM',
    'train_epoch',
    'eval_model',
]


