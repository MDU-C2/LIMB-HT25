import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
import logging
from datetime import datetime

from .datasets import EMGSequenceDataset, load_standardize_splits
from .models import get_simple_lstm
from .training import train_epoch, eval_model
from .utils import save_model, save_checkpoint, load_checkpoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make deterministic (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train EMG LSTM model")
    
    # Dataset arguments
    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help='Path to NPZ dataset file (default: auto-detect in EMG_signal_processing/)'
    )
    
    # Model arguments
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=32,
        help='LSTM hidden dimension (default: 32)'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.5,
        help='Dropout rate (default: 0.5)'
    )
    
    # Training arguments
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Training batch size (default: 64)'
    )
    parser.add_argument(
        '--val-batch-size',
        type=int,
        default=128,
        help='Validation/test batch size (default: 128)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate (default: 1e-3)'
    )
    
    # Data split arguments
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.3,
        help='Test set size ratio (default: 0.3)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.5,
        help='Validation ratio of temp split (default: 0.5)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    # Checkpointing and saving
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for models and logs (default: ./checkpoints/TIMESTAMP)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--save-best-only',
        action='store_true',
        help='Only save the best model (not every epoch)'
    )
    
    # Early stopping
    parser.add_argument(
        '--early-stopping',
        action='store_true',
        help='Enable early stopping'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=10,
        help='Early stopping patience (default: 10)'
    )
    parser.add_argument(
        '--min-delta',
        type=float,
        default=0.0,
        help='Minimum change to qualify as improvement (default: 0.0)'
    )
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Set random seed for reproducibility
    set_seed(args.seed)
    
    # Setup output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', timestamp)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging to file
    log_file = os.path.join(args.output_dir, 'training.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info("="*60)
    logger.info("Starting EMG LSTM Training")
    logger.info("="*60)
    logger.info(f"Arguments: {vars(args)}")
    
    # Determine dataset path
    if args.dataset is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        npz_path = os.path.join(current_dir, 'EMG_signal_processing', 'emg_sequences_dataset.npz')
    else:
        npz_path = args.dataset
    
    if not os.path.exists(npz_path):
        logger.error(f"Dataset not found at {npz_path}")
        raise FileNotFoundError(f"Dataset not found at {npz_path}")
    
    logger.info(f"Loading dataset from {npz_path}")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Load and split dataset
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = load_standardize_splits(
        npz_path, 
        test_size=args.test_size, 
        val_ratio_of_temp=args.val_ratio, 
        random_state=args.seed
    )
    
    logger.info(f"Dataset shapes - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    logger.info(f"Label distribution - Train: {np.bincount(y_train)}, Val: {np.bincount(y_val)}, Test: {np.bincount(y_test)}")
    
    # Create data loaders
    train_ds = EMGSequenceDataset(X_train, y_train)
    val_ds = EMGSequenceDataset(X_val, y_val)
    test_ds = EMGSequenceDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.val_batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.val_batch_size, shuffle=False)
    
    # Get model dimensions
    n_features = X_train.shape[2]
    n_classes = int(np.max(y_train)) + 1
    
    # Create model
    model = get_simple_lstm(
        input_dim=n_features, 
        hidden_dim=args.hidden_dim, 
        num_classes=n_classes, 
        dropout=args.dropout
    ).to(device)
    
    logger.info(f"Model created: {n_features} features, {n_classes} classes, hidden_dim={args.hidden_dim}")
    
    # Create loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training configuration
    config = {
        'input_dim': n_features,
        'hidden_dim': args.hidden_dim,
        'num_classes': n_classes,
        'dropout': args.dropout,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'seed': args.seed,
        'test_size': args.test_size,
        'val_ratio': args.val_ratio,
    }
    
    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_acc = 0.0
    training_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint_info = load_checkpoint(args.resume, model, optimizer, device)
        start_epoch = checkpoint_info['epoch'] + 1
        best_val_acc = checkpoint_info['best_val_acc']
        logger.info(f"Resuming from epoch {start_epoch}, best val acc so far: {best_val_acc:.4f}")
    
    # Early stopping setup
    early_stopping_counter = 0
    best_epoch = 0
    
    logger.info("Starting training...")
    logger.info("-"*60)
    
    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        # Train
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)
        
        # Track history
        training_history['train_loss'].append(tr_loss)
        training_history['train_acc'].append(tr_acc)
        training_history['val_loss'].append(val_loss)
        training_history['val_acc'].append(val_acc)
        
        # Log progress
        logger.info(
            f"Epoch {epoch}/{args.epochs}: "
            f"Train acc={tr_acc:.4f} loss={tr_loss:.4f} | "
            f"Val acc={val_acc:.4f} loss={val_loss:.4f}"
        )
        
        # Check if this is the best model
        is_best = val_acc > (best_val_acc + args.min_delta)
        
        if is_best:
            best_val_acc = val_acc
            best_epoch = epoch
            early_stopping_counter = 0
            
            # Save best model
            save_model(model, scaler, config, args.output_dir, 'best_model.pt')
        else:
            early_stopping_counter += 1
        
        # Save checkpoint (if not save_best_only or if it's the best)
        if not args.save_best_only or is_best:
            save_checkpoint(
                model, optimizer, epoch, val_acc, best_val_acc,
                scaler, config, args.output_dir, is_best=is_best
            )
        
        # Early stopping check
        if args.early_stopping and early_stopping_counter >= args.patience:
            logger.info(
                f"Early stopping triggered! No improvement for {args.patience} epochs. "
                f"Best val acc: {best_val_acc:.4f} at epoch {best_epoch}"
            )
            break
    
    logger.info("-"*60)
    logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    
    # Load best model for final evaluation
    best_model_path = os.path.join(args.output_dir, 'best_model.pt')
    if os.path.exists(best_model_path):
        logger.info("Loading best model for final evaluation...")
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_loss, test_acc = eval_model(model, test_loader, criterion, device)
    logger.info(f"Test Accuracy: {test_acc*100:.2f}% | Test Loss: {test_loss:.4f}")
    
    # Save training history
    import json
    history_path = os.path.join(args.output_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")
    
    # Quick inference demo
    logger.info("Running inference demo on 5 random test samples...")
    model.eval()
    idxs = np.random.choice(len(test_ds), size=min(5, len(test_ds)), replace=False)
    X_demo = torch.tensor(X_test[idxs], dtype=torch.float32).to(device)
    y_demo = y_test[idxs]
    
    with torch.no_grad():
        pred_logits = model(X_demo)
        preds = pred_logits.argmax(1).cpu().numpy()
    
    label_names = {0: 'Hand_Open', 1: 'Hand_Close'}
    for i in range(len(idxs)):
        t = label_names.get(y_demo[i], str(y_demo[i]))
        p = label_names.get(preds[i], str(preds[i]))
        logger.info(f"Sample {i+1}: True={t}, Pred={p}")
    
    logger.info("="*60)
    logger.info(f"All outputs saved to: {args.output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
