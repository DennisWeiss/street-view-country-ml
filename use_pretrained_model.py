import torch.nn as nn
import pretrainedmodels


def load_imagenet_vgg13():
    pretrained_model = pretrainedmodels.__dict__['vgg13'](num_classes=1000, pretrained='imagenet')

    model = nn.Sequential(*list(pretrained_model.modules())[:-1])

    model.add_module('last_linear', nn.Linear(in_features=4096, out_features=101, bias=True))
    print(model)

    return pretrained_model




