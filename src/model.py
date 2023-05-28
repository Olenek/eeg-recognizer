import torch
import torch.nn as nn


class CNN_LSTM(nn.Module):
    def __init__(self, num_classes):
        super(CNN_LSTM, self).__init__()
        self.norm = nn.LayerNorm([28, 28])
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.5)
        self.lstm = nn.LSTM(input_size=1152, hidden_size=128, num_layers=3, dropout=0.3, batch_first=True)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        batch_size, seq_length, channels, height, width = x.size()
        c_in = x.view(batch_size * seq_length, channels, height, width)

        c_out = self.pool(self.dropout(nn.functional.relu(self.conv1(c_in))))
        c_out = self.pool(self.dropout(nn.functional.relu(self.conv2(c_out))))
        c_out = self.pool(self.dropout(nn.functional.relu(self.conv3(c_out))))

        r_in = c_out.view(batch_size, seq_length, -1)
        r_out, _ = self.lstm(r_in)
        r_out = r_out[:, -1, :]  # Take the output of the last time step

        fc1_out = self.dropout(nn.functional.relu(self.fc1(r_out)))
        fc2_out = self.fc2(fc1_out)

        return fc2_out


class EnsembleModel(nn.Module):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = models
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

        self.fc1 = nn.Linear(2 * len(models), 8)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        x = torch.concatenate([model(x) for model in self.models], dim=1)
        x = self.dropout(nn.functional.relu((self.fc1(x))))
        x = self.fc2(x)
        return torch.clamp(x, min=-1, max=1)