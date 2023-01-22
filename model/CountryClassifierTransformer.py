import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from transformers import ViTForImageClassification, ViTFeatureExtractor


class CountryClassifierTransformer(nn.Module):
    def __init__(self):
        super(CountryClassifierTransformer, self).__init__()

        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-384', num_labels=101, ignore_mismatched_sizes=True)

    def forward(self, x):
        return self.model(x).logits
