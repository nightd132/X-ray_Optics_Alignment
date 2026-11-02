import torch.nn as nn

class OpticsAlignmentRegressionModel(nn.Module):
    def __init__(self, output_dim=2):
        super().__init__()

        self.CNN_model = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.regressor = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)

        )

    def forward(self, x):
        x = self.CNN_model(x)
        x = x.view(x.size(0), -1)
        x = self.regressor(x)
        return x
