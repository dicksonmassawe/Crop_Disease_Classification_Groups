import torch
import torch.nn as nn


class Model4(nn.Module):
    def __init__(self, num_classes=10):
        super(Model4, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)  # Input channels: 3 (RGB), Output channels: 6
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)  # Input channels: 6, Output channels: 16

        # Activation and pooling
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Placeholder for fully connected layers; will initialize dynamically
        self.f1 = None
        self.f2 = nn.Linear(120, 84)
        self.f3 = nn.Linear(84, num_classes)  # 10 classes for classification

    def forward(self, x):
        # Convolutional layers
        out = self.pool(self.relu(self.conv1(x)))  # After conv1 and pool
        out = self.pool(self.relu(self.conv2(out)))  # After conv2 and pool

        # Dynamically initialize the fully connected layer based on input size
        if self.f1 is None:
            flattened_size = out.numel() // out.size(0)  # Compute the flattened size dynamically
            self.f1 = nn.Linear(flattened_size, 120).to(x.device)

        # Flatten the output
        out = out.view(out.size(0), -1)

        # Fully connected layers
        out = self.relu(self.f1(out))
        out = self.relu(self.f2(out))
        pred = self.f3(out)  # Final output (no ReLU here for classification)

        return pred