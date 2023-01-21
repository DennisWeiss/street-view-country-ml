import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from transformers import ViTModel, ViTFeatureExtractor

import matplotlib.pyplot as plt



class CountryClassifierTransformer(nn.Module):
    def __init__(self):
        super(CountryClassifierTransformer, self).__init__()

        self.model = ViTModel.from_pretrained('google/vit-base-patch16-384')
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 101),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.model(x, output_hidden_states=True).hidden_states[-1][:, 0]
        x = self.classifier(x)
        return x
