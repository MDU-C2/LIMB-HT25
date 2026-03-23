import unittest
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import tempfile
import os

from .datasets import EMGSequenceDataset, load_standardize_splits
from .models import SimpleLSTM, get_simple_lstm
from .training import train_epoch, eval_model


class TestModels(unittest.TestCase):
    """Test the LSTM model components."""

    def test_simple_lstm_creation(self):
        """Test that SimpleLSTM can be created with different parameters."""
        model = SimpleLSTM(input_dim=48, hidden_dim=32, num_classes=2, dropout=0.5)
        self.assertIsNotNone(model)
        self.assertEqual(model.fc.out_features, 2)

    def test_simple_lstm_forward(self):
        """Test that SimpleLSTM forward pass works correctly."""
        batch_size = 4
        seq_len = 10
        input_dim = 48
        num_classes = 2
        
        model = SimpleLSTM(input_dim=input_dim, hidden_dim=32, num_classes=num_classes)
        x = torch.randn(batch_size, seq_len, input_dim)
        
        output = model(x)
        self.assertEqual(output.shape, (batch_size, num_classes))
        self.assertTrue(torch.all(torch.isfinite(output)))

    def test_get_simple_lstm(self):
        """Test the factory function for creating LSTM models."""
        model = get_simple_lstm(input_dim=48, hidden_dim=64, num_classes=2, dropout=0.3)
        self.assertIsInstance(model, SimpleLSTM)
        self.assertEqual(model.fc.out_features, 2)


class TestDatasets(unittest.TestCase):
    """Test the dataset loading and processing."""

    def test_emg_sequence_dataset(self):
        """Test EMGSequenceDataset creation and indexing."""
        n_samples = 20
        seq_len = 10
        n_features = 48
        
        X = np.random.randn(n_samples, seq_len, n_features)
        y = np.random.randint(0, 2, size=n_samples)
        
        dataset = EMGSequenceDataset(X, y)
        self.assertEqual(len(dataset), n_samples)
        
        # Test indexing
        x_sample, y_sample = dataset[0]
        self.assertEqual(x_sample.shape, (seq_len, n_features))
        self.assertIsInstance(y_sample, torch.Tensor)
        self.assertEqual(y_sample.dtype, torch.long)

    def test_load_standardize_splits(self):
        """Test loading and standardizing data splits."""
        # Create synthetic dataset
        n_samples = 100
        seq_len = 10
        n_features = 48
        
        X = np.random.randn(n_samples, seq_len, n_features)
        y = np.random.randint(0, 2, size=n_samples)
        
        # Save to temporary NPZ file
        with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp_file:
            np.savez(tmp_file.name, X=X, y=y)
            tmp_path = tmp_file.name
        
        try:
            (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = load_standardize_splits(
                tmp_path, test_size=0.3, val_ratio_of_temp=0.5, random_state=42
            )
            
            # Check shapes
            self.assertGreater(len(X_train), 0)
            self.assertGreater(len(X_val), 0)
            self.assertGreater(len(X_test), 0)
            
            # Check that scaler is returned
            self.assertIsNotNone(scaler)
            
            # Check that data is standardized (mean ~0, std ~1 for train)
            X_train_flat = X_train.reshape(-1, n_features)
            means = np.mean(X_train_flat, axis=0)
            stds = np.std(X_train_flat, axis=0)
            self.assertTrue(np.allclose(means, 0, atol=1e-6))
            self.assertTrue(np.allclose(stds, 1, atol=1e-6))
            
        finally:
            os.unlink(tmp_path)


class TestTraining(unittest.TestCase):
    """Test training and evaluation functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')
        self.batch_size = 4
        self.seq_len = 10
        self.input_dim = 48
        self.num_classes = 2
        
        # Create synthetic data
        n_samples = 20
        X = np.random.randn(n_samples, self.seq_len, self.input_dim)
        y = np.random.randint(0, self.num_classes, size=n_samples)
        
        self.dataset = EMGSequenceDataset(X, y)
        self.loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        
        # Create model
        self.model = get_simple_lstm(
            input_dim=self.input_dim,
            hidden_dim=16,
            num_classes=self.num_classes
        ).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def test_train_epoch(self):
        """Test training for one epoch."""
        initial_loss, initial_acc = eval_model(
            self.model, self.loader, self.criterion, self.device
        )
        
        train_loss, train_acc = train_epoch(
            self.model, self.loader, self.criterion, self.optimizer, self.device
        )
        
        # Check that training returns valid metrics
        self.assertIsInstance(train_loss, float)
        self.assertIsInstance(train_acc, float)
        self.assertGreaterEqual(train_acc, 0.0)
        self.assertLessEqual(train_acc, 1.0)
        self.assertGreaterEqual(train_loss, 0.0)

    def test_eval_model(self):
        """Test model evaluation."""
        eval_loss, eval_acc = eval_model(
            self.model, self.loader, self.criterion, self.device
        )
        
        # Check that evaluation returns valid metrics
        self.assertIsInstance(eval_loss, float)
        self.assertIsInstance(eval_acc, float)
        self.assertGreaterEqual(eval_acc, 0.0)
        self.assertLessEqual(eval_acc, 1.0)
        self.assertGreaterEqual(eval_loss, 0.0)

    def test_training_improves_model(self):
        """Test that training actually updates the model."""
        # Get initial predictions
        self.model.eval()
        with torch.no_grad():
            x_sample = self.dataset[0][0].unsqueeze(0).to(self.device)
            initial_output = self.model(x_sample)
        
        # Train for a few epochs
        for _ in range(3):
            train_epoch(self.model, self.loader, self.criterion, self.optimizer, self.device)
        
        # Get predictions after training
        self.model.eval()
        with torch.no_grad():
            final_output = self.model(x_sample)
        
        # Outputs should be different (model weights changed)
        self.assertFalse(torch.allclose(initial_output, final_output, atol=1e-6))


class TestIntegration(unittest.TestCase):
    """Integration tests for the full pipeline."""

    def test_full_training_loop(self):
        """Test a complete training loop with synthetic data."""
        device = torch.device('cpu')
        
        # Create synthetic dataset
        n_samples = 50
        seq_len = 10
        input_dim = 48
        num_classes = 2
        
        X = np.random.randn(n_samples, seq_len, input_dim)
        y = np.random.randint(0, num_classes, size=n_samples)
        
        # Split manually
        split_idx = int(0.7 * n_samples)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create datasets and loaders
        train_dataset = EMGSequenceDataset(X_train, y_train)
        val_dataset = EMGSequenceDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
        
        # Create and train model
        model = get_simple_lstm(
            input_dim=input_dim,
            hidden_dim=16,
            num_classes=num_classes
        ).to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Train for a few epochs
        for epoch in range(3):
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = eval_model(
                model, val_loader, criterion, device
            )
            
            # Check that metrics are valid
            self.assertGreaterEqual(train_acc, 0.0)
            self.assertLessEqual(train_acc, 1.0)
            self.assertGreaterEqual(val_acc, 0.0)
            self.assertLessEqual(val_acc, 1.0)


def run_tests():
    """Run all tests."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestModels))
    suite.addTests(loader.loadTestsFromTestCase(TestDatasets))
    suite.addTests(loader.loadTestsFromTestCase(TestTraining))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)

