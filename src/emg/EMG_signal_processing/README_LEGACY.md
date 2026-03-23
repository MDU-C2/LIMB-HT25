# Legacy Scripts

This directory contains legacy scripts that are not part of the main PyTorch training pipeline.

## simple_lstm.py

**Status:** Legacy / Not Integrated

This is an old TensorFlow/Keras implementation of the LSTM model. It is kept for reference but is **not used** by the current PyTorch training pipeline.

**Note:** The current training pipeline uses PyTorch and is located in:
- `../train_lstm.py` - Main PyTorch training script
- `../models.py` - PyTorch model definitions

If you need to use this TensorFlow version, you'll need to:
1. Install TensorFlow: `pip install tensorflow`
2. Update the dataset path in the script
3. Run it independently

**Recommendation:** Use the PyTorch implementation (`../train_lstm.py`) instead, as it has better features:
- Checkpointing and early stopping
- Comprehensive logging
- CLI interface
- Model saving/loading

