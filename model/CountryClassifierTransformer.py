import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from transformers import ViTForImageClassification, ViTFeatureExtractor

import matplotlib.pyplot as plt


class CountryClassifierTransformer(nn.Module):
    def __init__(self):
        super(CountryClassifierTransformer, self).__init__()

        self.patch_size = 16

        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-384', num_labels=101, ignore_mismatched_sizes=True)

    def forward(self, x):
        output = self.model(x, output_attentions=True)
        # if not self.training:
        #     for i in range(x.size(dim=0)):
        #         attention_map = output.attentions[-1][i].amax(dim=0)[1:,0].reshape(24, 24).detach().cpu().numpy()
        #         plt.matshow(attention_map)
        #         plt.show()

        return output.logits
