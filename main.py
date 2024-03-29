import torch.optim
import torch.utils.data
import torch.nn as nn
import torchvision.transforms
import torchvision.transforms.functional
import torch.nn.functional as F
from numpy import Infinity
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
from model.CountryClassifierProbing import CountryClassifierProbing
from model.CountryClassifierTransformer import CountryClassifierTransformer
from model.CountryClassifierV2 import CountryClassifierV2
from model.CountryClassifierV3 import CountryClassifierV3
from model.CountryClassifierV3_1 import CountryClassifierV3_1
from model.CountryClassifierV4 import CountryClassifierV4
from model.CountryClassifierPanoramaV5 import CountryClassifierPanoramaV5
from model.CIFAR10Classifier import CIFAR10Classifier


NUM_EPOCHS = 50
BATCH_SIZE = 6
USE_CUDA_IF_AVAILABLE = True

PROBING = False


if torch.cuda.is_available():
    print('GPU is available with the following device: {}'.format(torch.cuda.get_device_name()))
else:
    print('GPU is not available')

device = torch.device('cuda' if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else 'cpu')
print('The model will run with {}'.format(device))


def get_description(epoch, train_loss, train_acc):
    return f"Epoch {epoch + 1}/{NUM_EPOCHS} - loss: {train_loss:.5f} - acc: {(100 * train_acc):.3f}%"


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

train_data = SV101Country(transform=transform, train=True)
# train_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)

test_data = SV101Country(transform=transform, train=False)
# test_data = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1)

loss_fn = nn.CrossEntropyLoss()

model = CountryClassifierTransformer().to(device)
if not PROBING:
    model.load_state_dict(torch.load('snapshots/model_street_view__epoch11'))

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=0)
if not PROBING:
    optimizer.load_state_dict(torch.load('snapshots/model_street_view__optimizer_epoch11'))

count = 0
for param in model.parameters():
    count += torch.numel(param)

print(count)

for epoch in range(11, NUM_EPOCHS):
    train_loss = 0
    train_correct = 0

    num_samples = 0

    model.train()

    train_loop = tqdm(train_dataloader, desc=get_description(epoch, len(train_data) * train_loss / num_samples if num_samples > 0 else Infinity, train_correct / num_samples if num_samples > 0 else 0), unit='batch', colour='blue')
    for X, target in train_loop:
        X = X.to(device)

        optimizer.zero_grad()

        Y = model(X, probing=PROBING)

        target = target.to(device)

        # loss = torch.square(torch.nn.functional.softmax(Y, dim=1) - F.one_hot(target, num_classes=101)).sum(dim=1).mean()
        loss = loss_fn(Y, target)
        train_loss += X.size(dim=0) * loss.item() / len(train_data)
        train_correct += (torch.argmax(Y, dim=1) == target).sum().item()

        num_samples += X.size(dim=0)

        train_loop.set_description(get_description(epoch, len(train_data) * train_loss / num_samples if num_samples > 0 else Infinity, train_correct / num_samples if num_samples > 0 else 0))
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), f"snapshots/model_street_view_{'probing' if PROBING else ''}_epoch{epoch+1}")
    torch.save(optimizer.state_dict(), f"snapshots/model_street_view_{'probing' if PROBING else ''}_optimizer_epoch{epoch+1}")

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


    print(f'Train loss: {(train_loss):.5f}')
    print(f'Train accuracy: {(100 * train_correct / len(train_data)):.3f}%')
    # print(f'Test loss: {test_loss}')
    # print(f'Test accuracy: {test_acc}')




