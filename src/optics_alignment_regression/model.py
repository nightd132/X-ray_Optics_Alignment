import torch.nn as nn
import torch

class OpticsAlignmentRegressionModel(nn.Module):
    def __init__(self, output_dim=2, input_dim=(1, 86, 81)):
        super().__init__()

        self.CNN_model = nn.Sequential(
            nn.Conv2d(input_dim[0], 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
        )

        with torch.no_grad():
            n_flat = self.CNN_model(torch.zeros(1, *input_dim)).numel()

        self.regressor = nn.Sequential(
            nn.Linear(n_flat, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = self.CNN_model(x)
        x = x.flatten(start_dim=1)
        x = self.regressor(x)
        return x
