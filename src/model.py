import torch
import torch.nn as nn

class CNNWithGAP(nn.Module):  # <--- Class definition

    # --- Start of first method (e.g., 4 spaces indentation) ---
    def __init__(self, n_classes=2):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, n_classes)
    # --- End of first method ---

    # --- Start of second method (MUST have same indentation as __init__) ---
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
    # --- End of second method ---