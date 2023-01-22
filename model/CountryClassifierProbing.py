import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from transformers import ViTForImageClassification, ViTFeatureExtractor

import matplotlib.pyplot as plt


class CountryClassifierProbing(nn.Module):
    def __init__(self):
        super(CountryClassifierProbing, self).__init__()

        self.patch_size = 16

        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-384', num_labels=101, ignore_mismatched_sizes=True)
        self.classifier = nn.Sequential(
            nn.Linear(4 * 768, 768),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(768, 101)
        )

    def forward(self, x):
        output = self.model(x, output_hidden_states=True)
        output = self.classifier(torch.cat((
            output.hidden_states[-1][:,0],
            output.hidden_states[-2][:,0],
            output.hidden_states[-3][:,0],
            output.hidden_states[-4][:,0]
        ), dim=1).detach())
        return output
