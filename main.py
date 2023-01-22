import torch.optim
import torch.utils.data
import torch.nn as nn
import torchvision.transforms
import torchvision.transforms.functional
import torch.nn.functional as F
from transformers import ViTFeatureExtractor, ViTModel
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataloader.SV101CountryPanorama import SV101CountryPanorama
from dataloader.SV101CountryPanoramaSeparate import SV101CountryPanoramaSeparate
from dataloader.SV6Country import SV6Country
from dataloader.SV91Country import SV91Country
from dataloader.SV101Country import SV101Country
from model.CountryClassifierPanoramaV4 import CountryClassifierPanoramaV4
from model.CountryClassifier import CountryClassifier
from model.CountryClassifierTransformer import CountryClassifierTransformer
from model.CountryClassifierV2 import CountryClassifierV2
from model.CountryClassifierV3 import CountryClassifierV3
from model.CountryClassifierV3_1 import CountryClassifierV3_1
from model.CountryClassifierV4 import CountryClassifierV4
from model.CountryClassifierPanoramaV5 import CountryClassifierPanoramaV5
from model.CIFAR10Classifier import CIFAR10Classifier


NUM_EPOCHS = 50
BATCH_SIZE = 2
USE_CUDA_IF_AVAILABLE = True



if torch.cuda.is_available():
    print('GPU is available with the following device: {}'.format(torch.cuda.get_device_name()))
else:
    print('GPU is not available')

device = torch.device('cuda' if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else 'cpu')
print('The model will run with {}'.format(device))


def get_description(epoch, train_total_loss, train_correct, num_samples):
    return f"Epoch {epoch + 1}/{NUM_EPOCHS} " \
           f"- loss: {(train_total_loss / num_samples if num_samples > 0 else float('inf')):.5f} " \
           f"- acc: {(100 * train_correct / num_samples if num_samples > 0 else 0):.3f}%"


image_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-384')
# image_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')


def image_feature_extract(x):
    result = image_feature_extractor(x, return_tensors="pt").pixel_values
    return result[0]

transform = torchvision.transforms.Compose([
    torchvision.transforms.Lambda(lambda x: torchvision.transforms.functional.crop(x, 0, 13, 614, 614)),
    torchvision.transforms.Lambda(image_feature_extract),
])
# transform = torchvision.transforms.Compose([
#     torchvision.transforms.Resize((224, 224)),
#     torchvision.transforms.Lambda(image_feature_extract),
# ])

train_data = SV101CountryPanoramaSeparate(transform=transform, train=True)
# train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)

test_data = SV101CountryPanoramaSeparate(transform=transform, train=False)
# test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1)

loss = nn.CrossEntropyLoss()

model = CountryClassifierTransformer().to(device)
# model.load_state_dict(torch.load('snapshots/model_street_view_epoch3'))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0)
count = 0
for param in model.parameters():
    count += torch.numel(param)

print(count)

for epoch in range(NUM_EPOCHS):
    train_ce_loss = 0
    train_total_loss = 0
    train_correct = 0

    num_samples = 0

    model.train()

    train_loop = tqdm(train_dataloader, desc=get_description(epoch, train_total_loss, train_correct, num_samples), unit='batch', colour='blue')
    for (X0, X1, X2, X3), target in train_loop:
        X0 = X0.to(device)
        X1 = X1.to(device)
        X2 = X2.to(device)
        X3 = X3.to(device)

        optimizer.zero_grad()

        Y = torch.nn.functional.softmax(model(X0) + model(X1) + model(X2) + model(X3), dim=1)

        target = target.to(device)

        # ce_loss = torch.square(Y - F.one_hot(target, num_classes=101)).mean()
        ce_loss = loss(Y, target)
        total_loss = ce_loss
        train_total_loss += total_loss.item()
        train_ce_loss += ce_loss.item()
        train_correct += (torch.argmax(Y, dim=1) == target).sum().item()

        num_samples += X0.size(dim=0)

        train_loop.set_description(get_description(epoch, train_total_loss, train_correct, num_samples))
        total_loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), f'snapshots/model_street_view_panorama_separate_epoch{epoch+1}')
    # torch.cuda.empty_cache()

    # model = model.to('cpu')
    # model.eval()

    # test_loss = 0
    # test_acc = 0
    # for step, (X, target) in enumerate(test_dataloader):
    #     X = X
    #     target = target
    #     Y = model(X)
    #     test_loss += torch.square(Y - F.one_hot(target, num_classes=101)).sum() / len(test_data)
    #     test_acc += (torch.argmax(Y, dim=1) == target).sum() / len(test_data)


    print(f'Train total loss: {(train_total_loss / len(train_data)):.5f}')
    print(f'Train loss: {(train_ce_loss / len(train_data)):.5f}')
    print(f'Train accuracy: {(100 * train_correct / len(train_data)):.3f}')
    # print(f'Test loss: {test_loss}')
    # print(f'Test accuracy: {test_acc}')




