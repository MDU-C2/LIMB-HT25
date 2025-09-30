"""EMG data processing and dataset management.

This module provides utilities for loading, preprocessing, and managing
EMG datasets for training and evaluation.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple, Union
import os
import json
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle


class EMGDataset(Dataset):
    """
    PyTorch Dataset for EMG data.
    
    Handles loading and preprocessing of EMG sequences for LSTM training.
    """
    
    def __init__(
        self,
        sequences: np.ndarray,
        labels: np.ndarray,
        transform: Optional[callable] = None
    ):
        """
        Initialize the EMG dataset.
        
        Args:
            sequences: EMG sequences of shape (num_samples, seq_len, feature_dim)
            labels: Class labels of shape (num_samples,)
            transform: Optional transform to apply to sequences
        """
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
        self.transform = transform
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset."""
        sequence = self.sequences[idx]
        label = self.labels[idx]
        
        if self.transform:
            sequence = self.transform(sequence)
        
        return sequence, label


class EMGDataProcessor:
    """
    Data processor for EMG datasets.
    
    Handles loading, preprocessing, feature extraction, and dataset creation
    for EMG intent classification.
    """
    
    def __init__(
        self,
        sampling_rate: int = 1000,
        window_size: int = 200,
        overlap: float = 0.5,
        seq_len: int = 10
    ):
        """
        Initialize the data processor.
        
        Args:
            sampling_rate: Sampling rate of EMG signals in Hz
            window_size: Size of feature extraction window in samples
            overlap: Overlap ratio between windows
            seq_len: Number of windows in each sequence for LSTM
        """
        self.sampling_rate = sampling_rate
        self.window_size = window_size
        self.overlap = overlap
        self.seq_len = seq_len
        
        # Initialize feature extractor
        from .feature_extractor import EMGFeatureExtractor
        self.feature_extractor = EMGFeatureExtractor(sampling_rate)
        
        # Preprocessing components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Data storage
        self.raw_data = {}
        self.features = {}
        self.sequences = {}
        self.labels = {}
        
    def load_emg_data(
        self,
        data_path: str,
        file_format: str = "npy"
    ) -> Dict[str, np.ndarray]:
        """
        Load EMG data from files.
        
        Args:
            data_path: Path to data directory or file
            file_format: Format of data files ("npy", "csv", "json")
            
        Returns:
            Dictionary mapping gesture names to EMG data
        """
        if os.path.isfile(data_path):
            # Single file
            if file_format == "npy":
                data = np.load(data_path, allow_pickle=True).item()
            elif file_format == "json":
                with open(data_path, 'r') as f:
                    data = json.load(f)
                # Convert to numpy arrays
                for key, value in data.items():
                    data[key] = np.array(value)
            else:
                raise ValueError(f"Unsupported file format: {file_format}")
            
            self.raw_data = data
            
        elif os.path.isdir(data_path):
            # Directory of files
            data = {}
            for filename in os.listdir(data_path):
                if filename.endswith(f".{file_format}"):
                    gesture_name = os.path.splitext(filename)[0]
                    file_path = os.path.join(data_path, filename)
                    
                    if file_format == "npy":
                        data[gesture_name] = np.load(file_path)
                    elif file_format == "csv":
                        data[gesture_name] = np.loadtxt(file_path, delimiter=',')
                    elif file_format == "json":
                        with open(file_path, 'r') as f:
                            gesture_data = json.load(f)
                        data[gesture_name] = np.array(gesture_data)
            
            self.raw_data = data
        
        else:
            raise ValueError(f"Data path does not exist: {data_path}")
        
        print(f"Loaded EMG data for {len(self.raw_data)} gestures")
        for gesture, data in self.raw_data.items():
            print(f"  {gesture}: {data.shape}")
        
        return self.raw_data
    
    def extract_features(self, include_time_freq: bool = True) -> Dict[str, np.ndarray]:
        """
        Extract features from raw EMG data.
        
        Args:
            include_time_freq: Whether to include time-frequency features
            
        Returns:
            Dictionary mapping gesture names to feature matrices
        """
        if not self.raw_data:
            raise ValueError("No raw data loaded. Call load_emg_data() first.")
        
        self.features = {}
        
        for gesture_name, emg_data in self.raw_data.items():
            print(f"Extracting features for {gesture_name}...")
            
            # Extract features from sliding windows
            feature_matrix, feature_names = self.feature_extractor.extract_features_from_windows(
                emg_data=emg_data,
                window_size=self.window_size,
                overlap=self.overlap,
                include_time_freq=include_time_freq
            )
            
            self.features[gesture_name] = feature_matrix
            print(f"  Extracted {feature_matrix.shape[0]} windows with {feature_matrix.shape[1]} features")
        
        # Store feature names for later use
        self.feature_names = feature_names
        self.feature_dim = len(feature_names)
        
        return self.features
    
    def create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training.
        
        Returns:
            Tuple of (sequences, labels)
            sequences: Shape (num_sequences, seq_len, feature_dim)
            labels: Shape (num_sequences,)
        """
        if not self.features:
            raise ValueError("No features extracted. Call extract_features() first.")
        
        all_sequences = []
        all_labels = []
        
        # Encode gesture labels
        gesture_names = list(self.features.keys())
        self.label_encoder.fit(gesture_names)
        
        for gesture_name, feature_matrix in self.features.items():
            # Create sequences from feature matrix
            num_windows = feature_matrix.shape[0]
            num_sequences = max(1, num_windows - self.seq_len + 1)
            
            for i in range(num_sequences):
                sequence = feature_matrix[i:i + self.seq_len]
                
                # Pad sequence if necessary
                if sequence.shape[0] < self.seq_len:
                    padding = np.zeros((self.seq_len - sequence.shape[0], sequence.shape[1]))
                    sequence = np.vstack([sequence, padding])
                
                all_sequences.append(sequence)
                all_labels.append(gesture_name)
        
        sequences = np.array(all_sequences)
        labels = self.label_encoder.transform(all_labels)
        
        self.sequences = sequences
        self.labels = labels
        
        print(f"Created {len(sequences)} sequences of length {self.seq_len}")
        print(f"Feature dimension: {self.feature_dim}")
        print(f"Number of classes: {len(gesture_names)}")
        
        return sequences, labels
    
    def normalize_features(self, fit_scaler: bool = True) -> np.ndarray:
        """
        Normalize features using StandardScaler.
        
        Args:
            fit_scaler: Whether to fit the scaler on current data
            
        Returns:
            Normalized sequences
        """
        if self.sequences is None:
            raise ValueError("No sequences created. Call create_sequences() first.")
        
        # Reshape for scaling: (num_sequences * seq_len, feature_dim)
        original_shape = self.sequences.shape
        sequences_flat = self.sequences.reshape(-1, self.feature_dim)
        
        if fit_scaler:
            sequences_normalized = self.scaler.fit_transform(sequences_flat)
        else:
            sequences_normalized = self.scaler.transform(sequences_flat)
        
        # Reshape back to original shape
        sequences_normalized = sequences_normalized.reshape(original_shape)
        
        self.sequences = sequences_normalized
        
        return sequences_normalized
    
    def split_data(
        self,
        test_size: float = 0.2,
        val_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            test_size: Fraction of data for testing
            val_size: Fraction of remaining data for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if self.sequences is None or self.labels is None:
            raise ValueError("No sequences and labels available. Call create_sequences() first.")
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.sequences, self.labels,
            test_size=test_size,
            random_state=random_state,
            stratify=self.labels
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=random_state,
            stratify=y_temp
        )
        
        # Create datasets
        train_dataset = EMGDataset(X_train, y_train)
        val_dataset = EMGDataset(X_val, y_val)
        test_dataset = EMGDataset(X_test, y_test)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        print(f"Data split:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Validation: {len(val_dataset)} samples")
        print(f"  Test: {len(test_dataset)} samples")
        
        return train_loader, val_loader, test_loader
    
    def save_processed_data(self, filepath: str):
        """Save processed data and preprocessing components."""
        data_to_save = {
            'sequences': self.sequences,
            'labels': self.labels,
            'feature_names': self.feature_names,
            'feature_dim': self.feature_dim,
            'seq_len': self.seq_len,
            'window_size': self.window_size,
            'overlap': self.overlap,
            'sampling_rate': self.sampling_rate,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data_to_save, f)
        
        print(f"Processed data saved to {filepath}")
    
    def load_processed_data(self, filepath: str):
        """Load processed data and preprocessing components."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.sequences = data['sequences']
        self.labels = data['labels']
        self.feature_names = data['feature_names']
        self.feature_dim = data['feature_dim']
        self.seq_len = data['seq_len']
        self.window_size = data['window_size']
        self.overlap = data['overlap']
        self.sampling_rate = data['sampling_rate']
        self.scaler = data['scaler']
        self.label_encoder = data['label_encoder']
        
        print(f"Processed data loaded from {filepath}")
    
    def get_class_names(self) -> List[str]:
        """Get the names of gesture classes."""
        return self.label_encoder.classes_.tolist()
    
    def get_data_info(self) -> Dict:
        """Get information about the processed data."""
        if self.sequences is None:
            return {"error": "No data processed yet"}
        
        return {
            "num_sequences": len(self.sequences),
            "sequence_length": self.seq_len,
            "feature_dimension": self.feature_dim,
            "num_classes": len(self.get_class_names()),
            "class_names": self.get_class_names(),
            "window_size": self.window_size,
            "overlap": self.overlap,
            "sampling_rate": self.sampling_rate
        }
