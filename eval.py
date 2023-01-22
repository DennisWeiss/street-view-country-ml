import torch
import torchvision
import torch.utils.data
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
from transformers import ViTFeatureExtractor

from dataloader.SV101Country import SV101Country
from dataloader.SV101CountryPanorama import SV101CountryPanorama
from model.CountryClassifier import CountryClassifier
from dataloader.SV6Country import SV6Country
from dataloader.SV91Country import SV91Country
from model.CountryClassifierPanoramaV4 import CountryClassifierPanoramaV4
from model.CountryClassifierPanoramaV5 import CountryClassifierPanoramaV5
from model.CountryClassifierProbing import CountryClassifierProbing
from model.CountryClassifierTransformer import CountryClassifierTransformer
from model.CountryClassifierV2 import CountryClassifierV2
from model.CountryClassifierV3 import CountryClassifierV3
from model.CountryClassifierV3_1 import CountryClassifierV3_1
from model.CountryClassifierV4 import CountryClassifierV4

USE_CUDA_IF_AVAILABLE = True

if torch.cuda.is_available():
    print('GPU is available with the following device: {}'.format(torch.cuda.get_device_name()))
else:
    print('GPU is not available')

device = torch.device('cuda' if USE_CUDA_IF_AVAILABLE and torch.cuda.is_available() else 'cpu')
print('The model will run with {}'.format(device))

torch.set_printoptions(precision=1, sci_mode=False)


countries = [
    'ALB',
    'ASM',
    'AND',
    'ARG',
    'AUS',
    'BGD',
    'BEL',
    'BMU',
    'BTN',
    'BOL',
    'BWA',
    'BRA',
    'IOT',
    'BGR',
    'KHM',
    'CAN',
    'CHL',
    'COL',
    'HRV',
    'CUW',
    'CZE',
    'DNK',
    'DOM',
    'ECU',
    'EST',
    'SWZ',
    'FRO',
    'FIN',
    'FRA',
    'DEU',
    'GHA',
    'GIB',
    'GRC',
    'GRL',
    'GTM',
    'HKG',
    'HUN',
    'ISL',
    'IND',
    'IDN',
    'IRL',
    'IMN',
    'ISR',
    'ITA',
    'JPN',
    'JEY',
    'JOR',
    'KEN',
    'KOR',
    'KGZ',
    'LVA',
    'LSO',
    'LTU',
    'LUX',
    'MAC',
    'MYS',
    'MLT',
    'MEX',
    'MCO',
    'MNG',
    'MNE',
    'NLD',
    'NZL',
    'NGA',
    'MKD',
    'MNP',
    'NOR',
    'PSE',
    'PER',
    'PHL',
    'PCN',
    'POL',
    'PRT',
    'PRI',
    'QAT',
    'ROU',
    'RUS',
    'SPM',
    'SMR',
    'SEN',
    'SRB',
    'SGP',
    'SVK',
    'SVN',
    'ZAF',
    'SGS',
    'ESP',
    'LKA',
    'SWE',
    'CHE',
    'TWN',
    'THA',
    'TUN',
    'TUR',
    'UGA',
    'UKR',
    'ARE',
    'GBR',
    'USA',
    'VIR',
    'URY'
]


image_feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-384')


def image_feature_extract(x):
    result = image_feature_extractor(x, return_tensors="pt").pixel_values
    return result[0]


transform = torchvision.transforms.Compose([
    torchvision.transforms.Lambda(lambda x: torchvision.transforms.functional.crop(x, 0, 13, 614, 614)),
    torchvision.transforms.Lambda(image_feature_extract),
])

transform_vanilla = torchvision.transforms.Compose([
    torchvision.transforms.Lambda(lambda x: torchvision.transforms.functional.crop(x, 0, 13, 614, 614))
])

model = CountryClassifierTransformer().to(device)
model.load_state_dict(torch.load('snapshots/model_street_view_epoch5'))
model.eval()

start_index = 1376
n_samples = 4

indices = range(start_index, start_index + n_samples)

data = torch.utils.data.Subset(SV101Country(transform=transform, train=False), indices)
data_loader = torch.utils.data.DataLoader(data, batch_size=4)

data_vanilla = torch.utils.data.Subset(SV101Country(transform=transform_vanilla, train=False), indices)
data_vanilla_loader = torch.utils.data.DataLoader(data_vanilla, batch_size=4)


def show_best_estimates(y):
    estimates = []
    for i in range(len(countries)):
        estimates.append((countries[i], y[i]))
    return sorted(estimates, key=lambda x: x[1], reverse=True)


test_acc = 0

for step, (X, target) in enumerate(data_vanilla_loader):
    for i in range(X.size(dim=0)):
        plt.imshow(F.to_pil_image(X[i]))
        plt.show()

for step, (X, target) in enumerate(data_loader):
    X = X.to(device)
    target = target.to(device)
    p0 = torch.nn.functional.softmax(model(X[0].reshape((1, X.size(dim=1), X.size(dim=2), X.size(dim=3)))), dim=1)
    p1 = torch.nn.functional.softmax(model(X[1].reshape((1, X.size(dim=1), X.size(dim=2), X.size(dim=3)))), dim=1)
    p2 = torch.nn.functional.softmax(model(X[2].reshape((1, X.size(dim=1), X.size(dim=2), X.size(dim=3)))), dim=1)
    p3 = torch.nn.functional.softmax(model(X[3].reshape((1, X.size(dim=1), X.size(dim=2), X.size(dim=3)))), dim=1)
    Y = [p0, p1, p2, p3]
    p = p0 * p1 * p2 * p3
    p = torch.nn.functional.normalize(p, p=1, dim=1)
    print(countries[torch.argmax(p, dim=1)[0]])
    for country, certainty in show_best_estimates(p[0])[0:10]:
        print(f'{country}: {100 * certainty:.1f}%', end=', ')
    print()
    print('=================')
    test_acc += 4 * int(torch.argmax(p, dim=1)[0] == target[0]) / len(data)
    for i in range(X.size(dim=0)):
        print(countries[torch.argmax(Y[i])] + " - " + countries[target[i]])
        for country, certainty in show_best_estimates(Y[i][0])[0:10]:
            print(f'{country}: {100 * certainty:.1f}%', end=', ')
        print()

print(f'Test acc: {(100 * test_acc):.3f}%')