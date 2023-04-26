import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(AttentionModel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        batch_size, seq_len, _ = x.shape

        # apply the first fully connected layer
        x = x.view(-1, self.input_size)  # reshape to [batch_size * seq_len, input_size]
        x = torch.relu(self.fc1(x))
        x = x.view(batch_size, seq_len, -1)  # reshape back to [batch_size, seq_len, hidden_size]

        # apply attention mechanism
        energy = self.attention(x)  # energy shape: [batch_size, seq_len, 1]
        weights = torch.softmax(energy, dim=1)  # weights shape: [batch_size, seq_len, 1]
        weighted_x = torch.sum(weights * x, dim=1)  # weighted_x shape: [batch_size, hidden_size]

        # apply the second fully connected layer
        output = self.fc2(weighted_x)  # output shape: [batch_size, num_classes]
        return output


class CNNModel(nn.Module):
    def __init__(self, num_channels=14, num_freq_bands=5):
        self.num_channel = num_channels
        self.num_freq_bangs = num_freq_bands
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(num_channels, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(32 * num_freq_bands // 2, 64)
        self.fc2 = nn.Linear(64, 2)  # 2 for the 2 labels

    def forward(self, x):
        # Reshape input to (batch_size, num_channels, num_freq_bands)
        x = x.view(x.size(0), self.num_channel, self.num_freq_bangs)

        # Pass through convolutional layers
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = self.maxpool(x)

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)

        return x


class Attention1D(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Attention1D, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Define the attention layer
        self.attention = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )

        # Define the output layer
        self.output = nn.Linear(input_size, output_size)

    def forward(self, x):
        # Reshape the input tensor to (batch_size, num_channels, freq_bands)
        x = x.view(-1, self.input_size)

        # Compute the attention weights
        attn_weights = self.attention(x)

        # Apply attention to the input
        attn_output = torch.mul(x, attn_weights)

        # Compute the output
        output = self.output(attn_output)

        return output


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1)

        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=1)

        # Output layers
        self.fc1 = nn.Linear(384, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        # Input shape: (batch_size, channels, height, width)
        # Output shape: (batch_size, num_filters, height, width)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)

        # Flatten feature maps
        # Input shape: (batch_size, num_filters, height, width)
        # Output shape: (batch_size, num_filters * height * width)
        x = x.view(x.size(0), -1)

        # Output layers
        x = self.fc1(nn.Dropout(0.3)(x))
        x = F.relu(x)
        x = self.fc2(x)

        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64 * 17, 256)
        self.fc2 = nn.Linear(256, 4)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 17)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x