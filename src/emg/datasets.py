import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class EMGSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_standardize_splits(npz_path, test_size=0.3, val_ratio_of_temp=0.5, random_state=42):
    """
    Load and standardize the EMG sequence dataset.

    Args:
        npz_path: Path to the NPZ file containing the dataset.
        test_size: Size of the test set.
        val_ratio_of_temp: Ratio of the validation set to the temporary set.
        random_state: Random state for reproducibility.

    Returns:
        Tuple of (X_train, y_train), (X_val, y_val), (X_test, y_test)
    """
    data = np.load(npz_path)
    X, y = data['X'], data['y']

    N, L, F = X.shape
    scaler = StandardScaler()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=val_ratio_of_temp, stratify=y_temp, random_state=random_state
    )

    X_train_flat = X_train.reshape(-1, F)
    scaler.fit(X_train_flat)

    X_train_std = scaler.transform(X_train_flat).reshape(X_train.shape)
    X_val_std = scaler.transform(X_val.reshape(-1, F)).reshape(X_val.shape)
    X_test_std = scaler.transform(X_test.reshape(-1, F)).reshape(X_test.shape)

    return (X_train_std, y_train), (X_val_std, y_val), (X_test_std, y_test)


