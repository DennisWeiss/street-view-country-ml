import torch
import torch.nn as nn
import torchvision.transforms.functional as F
from transformers import ViTModel, ViTFeatureExtractor, ViTForImageClassification



class CIFAR10Classifier(nn.Module):
    def __init__(self):
        super(CIFAR10Classifier, self).__init__()

        self.model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=10, ignore_mismatched_sizes=True)
        # self.classifier = torch.nn.Sequential(
        #     torch.nn.Linear(768, 10),
        #     torch.nn.Softmax(dim=1)
        # )

    def forward(self, x):
        # x = self.model(x, output_hidden_states=True).hidden_states[-1][:, 0]
        # x = self.classifier(x)
        # return x
        output = self.model(x)
        return torch.nn.functional.softmax(output.logits, dim=1)