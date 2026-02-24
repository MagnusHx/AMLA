import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self,
                 in_channels=1,
                 num_classes=24,
                 c1=8, c2=16, c3=32,
                 fc1=64,
                 dropout=0.2):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(c2, c3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout2d(p=dropout),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),

            nn.Linear(2048, fc1),   # avoids hardcoding flatten size
            nn.ReLU(),
            nn.Dropout(p=dropout),

            nn.Linear(fc1, num_classes)
        )

    def forward(self, x):
        return self.net(x)
