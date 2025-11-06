# EMG package exports
from .datasets import EMGSequenceDataset, load_standardize_splits
from .models import get_simple_lstm, SimpleLSTM
from .training import train_epoch, eval_model


