import torch
import torch.nn as nn


class AudioLocationNet1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2)

        # Dummy forward pass to calculate the size after convolutions
        self._calculate_flatten_size()

        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Output 2D location

    def _calculate_flatten_size(self):
        # Create a dummy input with the same shape as your actual input
        dummy_input = torch.zeros(1, 4, 7680)  # (batch_size, channels, length)
        x = torch.relu(self.conv1(dummy_input))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        self.flatten_size = x.numel()  # Calculate flattened size
        # print(f"Flattened size after conv layers: {self.flatten_size}")

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))



class AudioLocationNet2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=5, stride=2)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=2)
        self.conv4 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=2)
        self.conv5 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=2)

        # Dummy forward pass to calculate the size after convolutions
        self._calculate_flatten_size()

        # Fully connected layers
        self.fc1 = nn.Linear(self.flatten_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 2)  # Output 2D location

    def _calculate_flatten_size(self):
        # Create a dummy input with the same shape as your actual input
        dummy_input = torch.zeros(1, 4, 7680)  # (batch_size, channels, length)
        x = torch.relu(self.conv1(dummy_input))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        self.flatten_size = x.numel()  # Calculate flattened size
        # print(f"Flattened size after conv layers: {self.flatten_size}")

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = torch.relu(self.conv5(x))
        x = torch.flatten(x, start_dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
