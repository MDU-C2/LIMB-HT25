"""
EMG LSTM Pipeline - Main CLI Entry Point

Usage:
    python -m src.emg.main train --dataset <path> [options]
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .datasets import EMGSequenceDataset, load_standardize_splits
from .models import get_simple_lstm
from .training import train_epoch, eval_model


def train_command(args):
    """Train LSTM model for EMG gesture classification."""
    # Resolve dataset path relative to script location if needed
    import os
    if not os.path.isabs(args.dataset) and not os.path.exists(args.dataset):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        dataset_path = os.path.join(script_dir, args.dataset)
        if os.path.exists(dataset_path):
            args.dataset = dataset_path
    
    npz_path = args.dataset
    
    # Device to train on
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load the dataset and create data loaders
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_standardize_splits(npz_path)
    train_ds = EMGSequenceDataset(X_train, y_train)
    val_ds = EMGSequenceDataset(X_val, y_val)
    test_ds = EMGSequenceDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)
    test_loader = DataLoader(test_ds, batch_size=128)

    # Get the number of features and classes
    n_features = X_train.shape[2]
    n_classes = int(np.max(y_train)) + 1
    model = get_simple_lstm(
        input_dim=n_features,
        hidden_dim=args.hidden_dim,
        num_classes=n_classes,
        dropout=0.5
    ).to(device)

    # Create the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print(f"\nTraining configuration:")
    print(f"  Dataset: {npz_path}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Features per window: {n_features}")
    print(f"  Number of classes: {n_classes}")
    print()

    # Train the model
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch}: Train acc={tr_acc:.3f} loss={tr_loss:.4f} | Val acc={val_acc:.3f} loss={val_loss:.4f}")

    # Evaluate the model on the test set
    test_loss, test_acc = eval_model(model, test_loader, criterion, device)
    print(f"\nTest Accuracy: {test_acc*100:.2f}% | Test Loss: {test_loss:.4f}")

    # Quick inference demo
    model.eval()
    idxs = np.random.choice(len(test_ds), size=5, replace=False)
    X_demo = torch.tensor(X_test[idxs], dtype=torch.float32).to(device)
    y_demo = y_test[idxs]

    # Make predictions
    with torch.no_grad():
        pred_logits = model(X_demo)
        preds = pred_logits.argmax(1).cpu().numpy()

    # Print the predictions
    label_names = {0: 'Hand_Open', 1: 'Hand_Close'}
    print("\nInference Demo:")
    for i in range(len(idxs)):
        t = label_names.get(y_demo[i], str(y_demo[i]))
        p = label_names.get(preds[i], str(preds[i]))
        print(f"  Sample {i+1}: True={t}, Pred={p}")
    
    # Save model, scaler, and config
    import json
    import pickle
    from datetime import datetime
    from sklearn.preprocessing import StandardScaler
    
    # Create checkpoints directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoints_dir = os.path.join(script_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Generate timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model
    model_path = os.path.join(checkpoints_dir, f"model_{timestamp}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Load full dataset to fit scaler (needed for inference)
    data = np.load(npz_path)
    X_full = data['X']
    scaler = StandardScaler()
    scaler.fit(X_full.reshape(-1, X_full.shape[2]))
    
    # Save scaler
    scaler_path = os.path.join(checkpoints_dir, f"scaler_{timestamp}.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_path}")
    
    # Save model config
    config = {
        'input_dim': n_features,
        'hidden_dim': args.hidden_dim,
        'num_classes': n_classes,
        'dropout': 0.5,
        'seq_length': X_train.shape[1]  # sequence length
    }
    config_path = os.path.join(checkpoints_dir, f"config_{timestamp}.json")
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to: {config_path}")
    
    print(f"\nTo verify hardware, use:")
    print(f"  python -m src.emg.capture_verification")


def main():
    parser = argparse.ArgumentParser(
        description="EMG LSTM Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings
  python -m src.emg.main train

  # Train with custom parameters
  python -m src.emg.main train --dataset data/processed_data/emg_sequences_all.npz --epochs 20 --batch-size 64

  # Verify hardware
  python -m src.emg.capture_verification
                """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train LSTM model')
    train_parser.add_argument(
        '--dataset',
        type=str,
        default='data/processed_data/emg_sequences_all.npz',
        help='Path to the NPZ dataset file (default: data/processed_data/emg_sequences_all.npz)'
    )
    train_parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='Number of training epochs (default: 5)'
    )
    train_parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for training (default: 64)'
    )
    train_parser.add_argument(
        '--hidden-dim',
        type=int,
        default=32,
        help='Hidden dimension for LSTM (default: 32)'
    )
    train_parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate (default: 1e-3)'
    )
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Run live EMG inference')
    predict_parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model (.pt file)'
    )
    predict_parser.add_argument(
        '--scaler',
        type=str,
        required=True,
        help='Path to StandardScaler (.pkl file)'
    )
    predict_parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to model config (.json file)'
    )
    predict_parser.add_argument(
        '--num-channels',
        type=int,
        default=1,
        help='Number of EMG channels (default: 1)'
    )
    predict_parser.add_argument(
        '--seq-length',
        type=int,
        default=10,
        help='Sequence length for LSTM (default: 10)'
    )
    predict_parser.add_argument(
        '--capture-mode',
        action='store_true',
        help='Run in capture mode: press SPACEBAR to capture 80 windows (20 rest, 40 grip, 20 rest)'
    )
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_command(args)
    elif args.command == 'predict':
        print("Error: Live inference functionality is currently unavailable.")
        print("Use 'capture_verification.py' for hardware verification capture:")
        print("  python -m src.emg.capture_verification")
        print("\nFor live predictions, the live_inference module needs to be restored.")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        parser.print_help()
        print(f"\nDetected device: {device}")
        print("\nUse 'train' command to start training.")
        print("Use 'python -m src.emg.capture_verification' for hardware verification.")


if __name__ == "__main__":
    main()
