import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .datasets import EMGSequenceDataset, load_standardize_splits
from .models import get_simple_lstm
from .training import train_epoch, eval_model


def main():

    # Path to the NPZ file containing the dataset.
    npz_path = '../../EMG_signal_processing/emg_sequences_dataset.npz'

    # Device to train on.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')


    # Load the dataset and create data loaders.
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_standardize_splits(npz_path)
    train_ds = EMGSequenceDataset(X_train, y_train)
    val_ds = EMGSequenceDataset(X_val, y_val)
    test_ds = EMGSequenceDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=128)
    test_loader = DataLoader(test_ds, batch_size=128)

    # Get the number of features and classes.
    n_features = X_train.shape[2]
    n_classes = int(np.max(y_train)) + 1
    model = get_simple_lstm(input_dim=n_features, hidden_dim=32, num_classes=n_classes, dropout=0.5).to(device)

    # Create the loss function and optimizer.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Train the model.
    for epoch in range(1, 6):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch}: Train acc={tr_acc:.3f} loss={tr_loss:.4f} | Val acc={val_acc:.3f} loss={val_loss:.4f}")

    # Evaluate the model on the test set.
    test_loss, test_acc = eval_model(model, test_loader, criterion, device)
    print(f"Test Accuracy: {test_acc*100:.2f}% | Test Loss: {test_loss:.4f}")

    # Quick inference demo.
    model.eval()
    idxs = np.random.choice(len(test_ds), size=5, replace=False)
    X_demo = torch.tensor(X_test[idxs], dtype=torch.float32).to(device)
    y_demo = y_test[idxs]

    # Make predictions.
    with torch.no_grad():
        pred_logits = model(X_demo)
        preds = pred_logits.argmax(1).cpu().numpy()

    # Print the predictions.
    label_names = {0: 'Hand_Open', 1: 'Hand_Close'}
    for i in range(len(idxs)):
        t = label_names.get(y_demo[i], str(y_demo[i]))
        p = label_names.get(preds[i], str(preds[i]))
        print(f"Sample {i+1}: True={t}, Pred={p}")

    # Maybe add: save the model.
if __name__ == "__main__":
    main()


