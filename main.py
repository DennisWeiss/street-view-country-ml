import torch.optim
import torch.utils.data
import torch.nn as nn
import torchvision.transforms
import torchvision.transforms.functional
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataloader.SV101CountryPanorama import SV101CountryPanorama
from dataloader.SV6Country import SV6Country
from dataloader.SV91Country import SV91Country
from dataloader.SV101Country import SV101Country
from model.CountryClassifierPanoramaV4 import CountryClassifierPanoramaV4
from model.CountryClassifier import CountryClassifier
from model.CountryClassifierV2 import CountryClassifierV2
from model.CountryClassifierV3 import CountryClassifierV3
from model.CountryClassifierV3_1 import CountryClassifierV3_1
from model.CountryClassifierV4 import CountryClassifierV4
from model.CountryClassifierPanoramaV5 import CountryClassifierPanoramaV5


NUM_EPOCHS = 50
BATCH_SIZE = 8
USE_CUDA_IF_AVAILABLE = True


MODEL = CountryClassifierPanoramaV5

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


transform = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop((600, 600)),
    # torchvision.transforms.RandomApply(torch.nn.ModuleList([torchvision.transforms.RandomCrop((300, 300))]), p=0.5),
    torchvision.transforms.Resize((300, 300))
])
train_data = SV101CountryPanorama(train=True)

train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)
test_data = SV101CountryPanorama(train=False)

test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)
model = MODEL().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-5, weight_decay=1e-3)


loss = nn.CrossEntropyLoss()


for epoch in range(NUM_EPOCHS):
    train_ce_loss = 0
    train_total_loss = 0
    train_correct = 0

    num_samples = 0

    train_loop = tqdm(train_dataloader, desc=get_description(epoch, train_total_loss, train_correct, num_samples), unit='batch', colour='blue')
    for X, target in train_loop:
        X = X.to(device)
        # plt.imshow(torchvision.transforms.functional.to_pil_image(X[0]))
        # plt.show()
        target = target.to(device)
        optimizer.zero_grad()
        Y = model(X)
        ce_loss = torch.square(Y - F.one_hot(target, num_classes=101)).sum()
        # ce_loss = loss(Y, target)
        total_loss = ce_loss
        train_total_loss += total_loss.item()
        train_ce_loss += ce_loss.item()
        train_correct += (torch.argmax(Y, dim=1) == target).sum().item()

        num_samples += X.size(dim=0)

        train_loop.set_description(get_description(epoch, train_total_loss, train_correct, num_samples))
        total_loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), f'snapshots/model_101country_mse_panorama_v5_lr3e-5_wd1e-3_epoch{epoch+1}')

    # test_loss = 0
    # test_acc = 0
    # for step, (X, target) in enumerate(test_dataloader):
    #     X = X.to(device)
    #     target = target.to(device)
    #     Y = model(X)
    #     test_loss += loss(Y, F.one_hot(target, num_classes=91)) / len(test_data)
    #     test_acc += (torch.argmax(Y, dim=1) == target).sum() / len(test_data)


    print(f'Train total loss: {(train_total_loss / len(train_data)):.5f}')
    print(f'Train loss: {(train_ce_loss / len(train_data)):.5f}')
    print(f'Train accuracy: {(100 * train_correct / len(train_data)):.3f}')
    # print(f'Test loss: {test_loss}')
    # print(f'Test accuracy: {test_acc}')




