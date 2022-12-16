import torch.nn as nn


class CountryClassifierV4(nn.Module):
    def __init__(self):
        super(CountryClassifierV4, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3), # [64x308x308]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3), # [64x306x306]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # [64x153x153]
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3), # [128x151x151]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3), # [128x149x149]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # [128x74x74]
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3), # [256x72x72]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3), # [256x70x70]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # [256x35x35]
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3), # [512x33x33]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3), # [512x31x31]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3), # [512x29x29]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2), # [512x14x14]
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3), # [512x12x12]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3), # [512x10x10]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3), # [512x8x8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3), # [256x6x6]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # [512x3x3]
        )

        self.flatten_layer = nn.Flatten()

        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=4608, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=101),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        conv_result = self.conv_layers(x)
        conv_result_flattened = self.flatten_layer(conv_result)
        return self.fc_layers(conv_result_flattened)
