import torch
from torch import nn


class Inception(nn.Module):
    def __init__(self, in_planes, n1, n3_reduce, n3, n5_reduce, n5, pool_proj):
        super(Inception, self).__init__()

        # 1x1 Convolution
        self.block1 = nn.Sequential(
            nn.Conv2d(in_planes, n1, kernel_size=1),
            nn.BatchNorm2d(n1),
            nn.ReLU(inplace=True),
        )

        # 1x1 Convolution followed by 3x3 Convolution
        self.block2 = nn.Sequential(
            nn.Conv2d(in_planes, n3_reduce, kernel_size=1),
            nn.BatchNorm2d(n3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3_reduce, n3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3),
            nn.ReLU(inplace=True),
        )

        # 1x1 Convolution followed by two 3x3 Convolutions
        self.block3 = nn.Sequential(
            nn.Conv2d(in_planes, n5_reduce, kernel_size=1),
            nn.BatchNorm2d(n5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5_reduce, n5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5, n5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5),
            nn.ReLU(inplace=True),
        )

        # 3x3 MaxPooling followed by 1x1 Convolution
        self.block4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_planes, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        y1 = self.block1(x)
        y2 = self.block2(x)
        y3 = self.block3(x)
        y4 = self.block4(x)

        return torch.cat([y1, y2, y3, y4], dim=1)


class Model1(nn.Module):
    def __init__(self, num_classes=10):
        super(Model1, self).__init__()

        # Initial convolutional layer
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )

        # Inception blocks
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)
        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        # Adaptive average pooling and linear layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(1024, num_classes)

    def forward(self, x):
        # Forward pass through layers
        out = self.pre_layers(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)  # Flatten
        out = self.linear(out)

        return out