import torch
import torchvision
from torch.utils.data import Dataset
import torch.nn.functional as F


class SV6Country(Dataset):
    def __init__(self, transform=None, train=True):
        super(Dataset, self).__init__()

        COUNTRY_TO_IDX = {
            'UKR': 0,
            'USA': 1,
            'CAN': 2,
            'BRA': 3,
            'AUS': 4,
            'ZAF': 5
        }

        self.path = 'datasets/6country/'
        self.transform = transform
        self.train = train
        self.total_samples = 9167
        self.train_split_idx = int(0.9 * self.total_samples)
        self.countries = torch.asarray(list(map(lambda countryCode: COUNTRY_TO_IDX[countryCode], open(self.path + 'countries.txt').read().split(','))))

    def __len__(self):
        return self.train_split_idx if self.train else self.total_samples - self.train_split_idx

    def __getitem__(self, idx):
        if not self.train:
            idx += self.train_split_idx
        image = torchvision.io.read_image(f'{self.path}images/{idx+1}.jpg') / 255
        return self.transform(image), self.countries[idx]
