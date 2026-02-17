import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self,
                 in_channels=3,
                 num_classes=8,
                 c1=16, c2=32, c3=64, c4=128,
                 fc1=128, fc2=32,
                 dropout=0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(c1),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(c2),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(c3),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(c3, c4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(c4),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),

            nn.LazyLinear(fc1),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(fc1, fc2),
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(fc2, num_classes)
        )

    def forward(self, x):
        return self.net(x)
