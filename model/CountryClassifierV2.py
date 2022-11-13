import torch.nn as nn


class CountryClassifierV2(nn.Module):
    def __init__(self):
        super(CountryClassifierV2, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=(6, 6)),  # [3x320x200]
            nn.BatchNorm2d(num_features=8),  # [16x310x190]
            nn.ReLU(),
            nn.MaxPool2d(3, 3),  # [16x155x95]
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(6, 6)),
            nn.BatchNorm2d(num_features=8),  # [32x145x85]
            nn.ReLU(),
            nn.MaxPool2d(3, 3),  # [32x73x43]
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(6, 6)),
            nn.BatchNorm2d(num_features=8),  # [48x63x33]
            nn.ReLU(),
            nn.MaxPool2d(3, 3),  # [48x32x17]
            nn.Flatten(),
            nn.Linear(360, 91),
            nn.Softmax()
        )

    def forward(self, x):
        return self.model(x)
