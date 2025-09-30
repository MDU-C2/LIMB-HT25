"""LSTM-based temporal model for EMG intent classification.

This module implements an LSTM neural network for classifying EMG signals
into different gestures/intents over time sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class EMGLSTMModel(nn.Module):
    """
    LSTM-based model for EMG intent classification.
    
    Architecture:
    - Input: (batch_size, seq_len, feature_dim)
    - LSTM layers for temporal modeling
    - Dropout for regularization
    - Dense layers for classification
    - Output: (batch_size, num_classes)
    
    Args:
        feature_dim: Number of features per time window per channel
        seq_len: Number of time windows in the sequence
        num_classes: Number of gesture/intent classes
        hidden_dim: LSTM hidden dimension
        num_layers: Number of LSTM layers
        dropout_rate: Dropout rate for regularization
        bidirectional: Whether to use bidirectional LSTM
    """
    
    def __init__(
        self,
        feature_dim: int,
        seq_len: int,
        num_classes: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout_rate: float = 0.3,
        bidirectional: bool = True
    ):
        super(EMGLSTMModel, self).__init__()
        
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate LSTM output dimension
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Attention mechanism (optional)
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_dim,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if 'lstm' in name:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.kaiming_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, feature_dim)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling over time dimension
        pooled = torch.mean(attn_out, dim=1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for interpretability.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, feature_dim)
            
        Returns:
            Attention weights of shape (batch_size, seq_len, seq_len)
        """
        batch_size = x.size(0)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Get attention weights
        _, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        return attn_weights


class EMGClassifier:
    """
    High-level interface for EMG intent classification.
    
    This class provides a convenient interface for training, evaluation,
    and inference with the EMG LSTM model.
    """
    
    def __init__(
        self,
        feature_dim: int,
        seq_len: int,
        num_classes: int,
        model_config: Optional[Dict] = None,
        device: str = "auto"
    ):
        """
        Initialize the EMG classifier.
        
        Args:
            feature_dim: Number of features per time window
            seq_len: Number of time windows in sequence
            num_classes: Number of gesture classes
            model_config: Model configuration dictionary
            device: Device to run on ("cpu", "cuda", or "auto")
        """
        self.feature_dim = feature_dim
        self.seq_len = seq_len
        self.num_classes = num_classes
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Default model configuration
        default_config = {
            "hidden_dim": 128,
            "num_layers": 2,
            "dropout_rate": 0.3,
            "bidirectional": True
        }
        
        if model_config:
            default_config.update(model_config)
        
        # Initialize model
        self.model = EMGLSTMModel(
            feature_dim=feature_dim,
            seq_len=seq_len,
            num_classes=num_classes,
            **default_config
        ).to(self.device)
        
        # Initialize optimizer and loss function
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=0.001,
            weight_decay=1e-5
        )
        self.criterion = nn.CrossEntropyLoss()
        
        # Training history
        self.train_history = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }
    
    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None
    ) -> Dict[str, float]:
        """
        Train the model for one epoch.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update weights
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        
        metrics = {"loss": avg_loss, "accuracy": accuracy}
        
        # Validation
        if val_loader:
            val_metrics = self.evaluate(val_loader)
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        
        # Update history
        self.train_history["loss"].append(metrics["loss"])
        self.train_history["accuracy"].append(metrics["accuracy"])
        if "val_loss" in metrics:
            self.train_history["val_loss"].append(metrics["val_loss"])
            self.train_history["val_accuracy"].append(metrics["val_accuracy"])
        
        return metrics
    
    def evaluate(self, data_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader: Data loader for evaluation
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in data_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(data_loader)
        accuracy = 100.0 * correct / total
        
        return {"loss": avg_loss, "accuracy": accuracy}
    
    def predict(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            x: Input data of shape (batch_size, seq_len, feature_dim)
            
        Returns:
            Predicted class probabilities
        """
        self.model.eval()
        
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x)
        
        x = x.to(self.device)
        
        with torch.no_grad():
            output = self.model(x)
            probabilities = F.softmax(output, dim=1)
        
        return probabilities.cpu().numpy()
    
    def predict_class(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            x: Input data of shape (batch_size, seq_len, feature_dim)
            
        Returns:
            Predicted class labels
        """
        probabilities = self.predict(x)
        return np.argmax(probabilities, axis=1)
    
    def save_model(self, filepath: str):
        """Save the trained model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'feature_dim': self.feature_dim,
            'seq_len': self.seq_len,
            'num_classes': self.num_classes,
            'train_history': self.train_history
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load a trained model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_history = checkpoint['train_history']
    
    def get_model_summary(self) -> str:
        """Get a summary of the model architecture."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        summary = f"""
EMG LSTM Model Summary:
======================
Input shape: (batch_size, {self.seq_len}, {self.feature_dim})
Output shape: (batch_size, {self.num_classes})
Total parameters: {total_params:,}
Trainable parameters: {trainable_params:,}
Device: {self.device}

Architecture:
- LSTM layers: {self.model.num_layers}
- Hidden dimension: {self.model.hidden_dim}
- Bidirectional: {self.model.bidirectional}
- Dropout rate: {self.model.dropout_rate}
        """
        
        return summary.strip()
