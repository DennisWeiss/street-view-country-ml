import torch
import torchvision
import torch.utils.data
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from model.CountryClassifier import CountryClassifier
from dataloader.SV6Country import SV6Country
from dataloader.SV91Country import SV91Country
from model.CountryClassifierV2 import CountryClassifierV2

USE_CUDA_IF_AVAILABLE = True

if torch.cuda.is_available():
    print('GPU is available with the following device: {}'.format(torch.cuda.get_device_name()))
else:
    print('GPU is not available')

device = torch.device('cuda' if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else 'cpu')
print('The model will run with {}'.format(device))

torch.set_printoptions(precision=1, sci_mode=False)


countries = ['ALB', 'ASM', 'AND', 'ARG', 'AUS', 'BGD', 'BEL', 'BMU', 'BTN', 'BOL', 'BWA', 'BRA', 'IOT', 'BGR',
  'KHM', 'CAN', 'CHL', 'COL', 'HRV', 'CUW', 'CZE', 'DNK', 'DOM', 'ECU', 'EST', 'SWZ', 'FRO', 'FIN', 'FRA', 'DEU', 'GHA',
  'GIB', 'GRC', 'GRL', 'HKG', 'ITA', 'JPN', 'JEY', 'JOR', 'KEN', 'KOR', 'KGZ', 'LVA', 'LSO', 'LTU', 'LUX', 'MAC', 'MYS',
  'MLT', 'MEX', 'MCO', 'MNG', 'NLD', 'NZL', 'NGA', 'MKD', 'MNP', 'NOR', 'PSE', 'PER', 'PHL', 'PCN', 'POL', 'PRT', 'PRI',
  'ROU', 'RUS', 'SPM', 'SMR', 'SEN', 'SRB', 'SGP', 'SVK', 'SVN', 'ZAF', 'SGS', 'ESP', 'LKA', 'SWE', 'CHE', 'TWN', 'THA',
  'TUN', 'TUR', 'UGA', 'UKR', 'ARE', 'GBR', 'USA', 'VIR', 'URY']

transform = torchvision.transforms.Compose([
    torchvision.transforms.CenterCrop((400, 640)),
    torchvision.transforms.Resize((200, 320))
])

model = CountryClassifierV2().to(device)
model.load_state_dict(torch.load('snapshots/model_91country_ce_lr1e-3_epoch15'))

# data = torch.utils.data.Subset(SV91Country(train=False, transform=transform), list(range(470, 478)))
data = SV91Country(train=False, transform=transform)
data_loader = torch.utils.data.DataLoader(data, batch_size=16)


def show_best_estimates(y):
    estimates = []
    for i in range(len(countries)):
        estimates.append((countries[i], y[i]))
    return sorted(estimates, key=lambda x: x[1], reverse=True)


test_acc = 0

for step, (X, target) in enumerate(data_loader):
    # for i in range(X.size(dim=0)):
    #     plt.imshow(F.to_pil_image(X[i]))
    #     plt.show()
    X = X.to(device)
    target = target.to(device)
    Y = model(X)
    test_acc += (torch.argmax(Y, dim=1) == target).sum() / len(data)
    # for i in range(X.size(dim=0)):
    #     print(countries[torch.argmax(Y[i])] + " - " + countries[target[i]])
    #     for country, certainty in show_best_estimates(Y[i])[0:10]:
    #         print(f'{country}: {100 * certainty:.1f}%')

print(f'Test acc: {test_acc}')