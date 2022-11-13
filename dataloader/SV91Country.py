import torch
import torchvision
from torch.utils.data import Dataset
import torch.nn.functional as F


class SV91Country(Dataset):
    def __init__(self, transform=None, train=True):
        super(Dataset, self).__init__()

        COUNTRY_TO_IDX = {
            'ALB': 0,
            'ASM': 1,
            'AND': 2,
            'ARG': 3,
            'AUS': 4,
            'BGD': 5,
            'BEL': 6,
            'BMU': 7,
            'BTN': 8,
            'BOL': 9,
            'BWA': 10,
            'BRA': 11,
            'IOT': 12,
            'BGR': 13,
            'KHM': 14,
            'CAN': 15,
            'CHL': 16,
            'COL': 17,
            'HRV': 18,
            'CUW': 19,
            'CZE': 20,
            'DNK': 21,
            'DOM': 22,
            'ECU': 23,
            'EST': 24,
            'SWZ': 25,
            'FRO': 26,
            'FIN': 27,
            'FRA': 28,
            'DEU': 29,
            'GHA': 30,
            'GIB': 31,
            'GRC': 32,
            'GRL': 33,
            'HKG': 34,
            'ITA': 35,
            'JPN': 36,
            'JEY': 37,
            'JOR': 38,
            'KEN': 39,
            'KOR': 40,
            'KGZ': 41,
            'LVA': 42,
            'LSO': 43,
            'LTU': 44,
            'LUX': 45,
            'MAC': 46,
            'MYS': 47,
            'MLT': 48,
            'MEX': 49,
            'MCO': 50,
            'MNG': 51,
            'NLD': 52,
            'NZL': 53,
            'NGA': 54,
            'MKD': 55,
            'MNP': 56,
            'NOR': 57,
            'PSE': 58,
            'PER': 59,
            'PHL': 60,
            'PCN': 61,
            'POL': 62,
            'PRT': 63,
            'PRI': 64,
            'ROU': 65,
            'RUS': 66,
            'SPM': 67,
            'SMR': 68,
            'SEN': 69,
            'SRB': 70,
            'SGP': 71,
            'SVK': 72,
            'SVN': 73,
            'ZAF': 74,
            'SGS': 75,
            'ESP': 76,
            'LKA': 77,
            'SWE': 78,
            'CHE': 79,
            'TWN': 80,
            'THA': 81,
            'TUN': 82,
            'TUR': 83,
            'UGA': 84,
            'UKR': 85,
            'ARE': 86,
            'GBR': 87,
            'USA': 88,
            'VIR': 89,
            'URY': 90
        }

        self.path = 'datasets/91country/'
        self.transform = transform
        self.train = train
        self.total_samples = 16566 - 1865
        self.train_split_idx = int(0.9 * self.total_samples)
        self.countries = torch.tensor(list(map(lambda countryCode: COUNTRY_TO_IDX[countryCode], open(self.path + 'countries.txt').read().split(','))))

    def __len__(self):
        return self.train_split_idx if self.train else self.total_samples - self.train_split_idx

    def __getitem__(self, idx):
        if not self.train:
            idx += self.train_split_idx
        image = torchvision.io.read_image(f'{self.path}images/{idx+1+(1865 if idx >= 7302 else 0)}.jpg') / 255
        return self.transform(image), self.countries[idx]
