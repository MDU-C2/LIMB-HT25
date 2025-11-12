import torch.nn as nn


class SimpleLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h = self.dropout(h_n[-1])
        return self.fc(h)


def get_simple_lstm(input_dim, hidden_dim=32, num_classes=2, dropout=0.5):
    return SimpleLSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout)


