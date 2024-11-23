import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        # First conv layer: 1 input channel, 16 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        # Second conv layer: 16 input channels, 25 output channels, 3x3 kernel
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=25, kernel_size=3)
        # Max pooling layer with 2x2 kernel
        self.pool = nn.MaxPool2d(kernel_size=2)
        # First linear layer: 25 * 5 * 5 input features, 32 output features
        self.fc1 = nn.Linear(in_features=25 * 5 * 5, out_features=32)
        # Second linear layer: 32 input features, 10 output features (classes)
        self.fc2 = nn.Linear(in_features=32, out_features=10)

    def forward(self, x):
        # First conv block
        x = self.pool(F.relu(self.conv1(x)))
        # Second conv block
        x = self.pool(F.relu(self.conv2(x)))
        # Flatten the tensor
        x = x.view(-1, 25 * 5 * 5)
        # First linear layer with ReLU
        x = F.relu(self.fc1(x))
        # Second linear layer with log_softmax
        x = F.log_softmax(self.fc2(x), dim=1)
        return x 