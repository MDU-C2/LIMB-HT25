import torch


def train_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one epoch.

    Args:
        model: The model to train.
        loader: The data loader.
        criterion: The loss function.
        optimizer: The optimizer.
        device: The device to train on.

    Returns:
        Tuple of (total_loss, correct)
    """
    model.train()
    total_loss, correct = 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(xb)
        correct += (logits.argmax(1) == yb).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


def eval_model(model, loader, criterion, device):
    """
    Evaluate the model for one epoch.

    Args:
        model: The model to evaluate.
        loader: The data loader.
        criterion: The loss function.
        device: The device to evaluate on.

    Returns:
        Tuple of (total_loss, correct)
    """
    model.eval()
    total_loss, correct = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total_loss += criterion(logits, yb).item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()
    return total_loss / len(loader.dataset), correct / len(loader.dataset)


