import torch.optim
import torch.utils.data
import torch.nn as nn
import torchvision.transforms
import torch.nn.functional as F
from tqdm import tqdm

from dataloader.SV6Country import SV6Country
from dataloader.SV91Country import SV91Country
from model.CountryClassifier import CountryClassifier
from model.CountryClassifierV2 import CountryClassifierV2


NUM_EPOCHS = 50
USE_CUDA_IF_AVAILABLE = True
MODEL = CountryClassifierV2

if torch.cuda.is_available():
    print('GPU is available with the following device: {}'.format(torch.cuda.get_device_name()))
else:
    print('GPU is not available')

device = torch.device('cuda' if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else 'cpu')
print('The model will run with {}'.format(device))


transform = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop((400, 640)),
    torchvision.transforms.Resize((200, 320))
])

train_data = SV91Country(train=True, transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=8)

test_data = SV91Country(train=False, transform=transform)
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=8)

model = MODEL().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

loss = nn.CrossEntropyLoss()

for epoch in range(NUM_EPOCHS):
    train_ce_loss = 0
    train_total_loss = 0
    train_acc = 0
    for X, target in tqdm(train_dataloader, desc=f'Epoch {epoch + 1}/{NUM_EPOCHS}', unit='batch'):
        X = X.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        Y = model(X)
        l2_reg_loss = 0 * sum(param.pow(2).sum() for param in model.parameters())
        # ce_loss = torch.divide(torch.square(Y - F.one_hot(target, num_classes=91)).sum(), len(train_data))
        ce_loss = loss(Y, target)
        total_loss = ce_loss
        train_total_loss += total_loss.item() / len(train_data)
        train_ce_loss += ce_loss.item() / len(train_data)
        train_acc += (torch.argmax(Y, dim=1) == target).sum() / len(train_data)
        total_loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), f'snapshots/model_91country_ce_lr3e-4_epoch{epoch+1}')

    test_loss = 0
    test_acc = 0
    # for step, (X, target) in enumerate(test_dataloader):
    #     X = X.to(device)
    #     target = target.to(device)
    #     Y = model(X)
    #     test_loss += loss(Y, F.one_hot(target, num_classes=91)) / len(test_data)
    #     test_acc += (torch.argmax(Y, dim=1) == target).sum() / len(test_data)


    print(f'Train total loss: {train_total_loss}')
    print(f'Train loss: {train_ce_loss}')
    print(f'Train accuracy: {train_acc}')
    print(f'Test loss: {test_loss}')
    print(f'Test accuracy: {test_acc}')




