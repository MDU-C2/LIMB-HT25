"""
Utility functions for model saving, loading, and checkpointing.
"""
import os
import pickle
import torch
import logging

logger = logging.getLogger(__name__)


def save_model(model, scaler, config, save_dir, filename='best_model.pt'):
    """
    Save model, scaler, and configuration to disk.
    
    Args:
        model: The trained PyTorch model
        scaler: The fitted StandardScaler
        config: Dictionary containing training configuration
        save_dir: Directory to save the model
        filename: Name of the model file
    """
    os.makedirs(save_dir, exist_ok=True)
    
    model_path = os.path.join(save_dir, filename)
    scaler_path = os.path.join(save_dir, 'scaler.pkl')
    config_path = os.path.join(save_dir, 'config.json')
    
    # Save model state dict
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': config.get('input_dim'),
            'hidden_dim': config.get('hidden_dim'),
            'num_classes': config.get('num_classes'),
            'dropout': config.get('dropout'),
        }
    }, model_path)
    
    # Save scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    # Save configuration as JSON
    import json
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Model saved to {model_path}")
    logger.info(f"Scaler saved to {scaler_path}")
    logger.info(f"Config saved to {config_path}")


def load_model(model_path, device='cpu'):
    """
    Load a saved model from disk.
    
    Args:
        model_path: Path to the saved model file
        device: Device to load the model on
        
    Returns:
        Dictionary containing model_state_dict and model_config
    """
    checkpoint = torch.load(model_path, map_location=device)
    logger.info(f"Model loaded from {model_path}")
    return checkpoint


def load_scaler(scaler_path):
    """
    Load a saved scaler from disk.
    
    Args:
        scaler_path: Path to the saved scaler file
        
    Returns:
        The loaded StandardScaler
    """
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    logger.info(f"Scaler loaded from {scaler_path}")
    return scaler


def save_checkpoint(model, optimizer, epoch, val_acc, best_val_acc, 
                   scaler, config, save_dir, is_best=False):
    """
    Save a training checkpoint.
    
    Args:
        model: The PyTorch model
        optimizer: The optimizer
        epoch: Current epoch number
        val_acc: Current validation accuracy
        best_val_acc: Best validation accuracy so far
        scaler: The fitted StandardScaler
        config: Training configuration
        save_dir: Directory to save checkpoints
        is_best: Whether this is the best model so far
    """
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'best_val_acc': best_val_acc,
        'model_config': {
            'input_dim': config.get('input_dim'),
            'hidden_dim': config.get('hidden_dim'),
            'num_classes': config.get('num_classes'),
            'dropout': config.get('dropout'),
        }
    }
    
    # Save latest checkpoint
    latest_path = os.path.join(save_dir, 'checkpoint_latest.pt')
    torch.save(checkpoint, latest_path)
    
    # Save best model if this is the best
    if is_best:
        best_path = os.path.join(save_dir, 'checkpoint_best.pt')
        torch.save(checkpoint, best_path)
        logger.info(f"New best model saved! Val acc: {val_acc:.4f}")
        
        # Also save the scaler with the best model
        scaler_path = os.path.join(save_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)


def load_checkpoint(checkpoint_path, model, optimizer=None, device='cpu'):
    """
    Load a training checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model: The model to load weights into
        optimizer: Optional optimizer to load state into
        device: Device to load on
        
    Returns:
        Dictionary with epoch, val_acc, best_val_acc
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Checkpoint loaded from {checkpoint_path}")
    logger.info(f"Resuming from epoch {checkpoint['epoch']}, val_acc: {checkpoint['val_acc']:.4f}")
    
    return {
        'epoch': checkpoint['epoch'],
        'val_acc': checkpoint['val_acc'],
        'best_val_acc': checkpoint.get('best_val_acc', checkpoint['val_acc']),
        'model_config': checkpoint.get('model_config', {})
    }

