import torch
import torchvision
from torch.utils.data import Dataset
import torch.nn.functional as F


class SV101Country(Dataset):
    def __init__(self, transform=None, train=True):
        super(Dataset, self).__init__()

        COUNTRY_TO_IDX = {"ALB": 0, "ASM": 1, "AND": 2, "ARG": 3, "AUS": 4, "BGD": 5, "BEL": 6, "BMU": 7, "BTN": 8,
                          "BOL": 9, "BWA": 10, "BRA": 11, "IOT": 12, "BGR": 13, "KHM": 14, "CAN": 15, "CHL": 16,
                          "COL": 17, "HRV": 18, "CUW": 19, "CZE": 20, "DNK": 21, "DOM": 22, "ECU": 23, "EST": 24,
                          "SWZ": 25, "FRO": 26, "FIN": 27, "FRA": 28, "DEU": 29, "GHA": 30, "GIB": 31, "GRC": 32,
                          "GRL": 33, "GTM": 34, "HKG": 35, "HUN": 36, "ISL": 37, "IND": 38, "IDN": 39, "IRL": 40,
                          "IMN": 41, "ISR": 42, "ITA": 43, "JPN": 44, "JEY": 45, "JOR": 46, "KEN": 47, "KOR": 48,
                          "KGZ": 49, "LVA": 50, "LSO": 51, "LTU": 52, "LUX": 53, "MAC": 54, "MYS": 55, "MLT": 56,
                          "MEX": 57, "MCO": 58, "MNG": 59, "MNE": 60, "NLD": 61, "NZL": 62, "NGA": 63, "MKD": 64,
                          "MNP": 65, "NOR": 66, "PSE": 67, "PER": 68, "PHL": 69, "PCN": 70, "POL": 71, "PRT": 72,
                          "PRI": 73, "QAT": 74, "ROU": 75, "RUS": 76, "SPM": 77, "SMR": 78, "SEN": 79, "SRB": 80,
                          "SGP": 81, "SVK": 82, "SVN": 83, "ZAF": 84, "SGS": 85, "ESP": 86, "LKA": 87, "SWE": 88,
                          "CHE": 89, "TWN": 90, "THA": 91, "TUN": 92, "TUR": 93, "UGA": 94, "UKR": 95, "ARE": 96,
                          "GBR": 97, "USA": 98, "VIR": 99, "URY": 100}

        self.path = 'datasets/101country/'
        self.transform = transform
        self.train = train
        self.total_samples = 41236
        self.train_split_idx = int(0.9 * self.total_samples)
        locations = open(self.path + 'locations.csv').read().split('\n')
        self.locations = torch.tensor(list(map(lambda location: COUNTRY_TO_IDX[location.split(',')[0]], locations)))

    def __len__(self):
        return self.train_split_idx if self.train else self.total_samples - self.train_split_idx

    def __getitem__(self, idx):
        if not self.train:
            idx += self.train_split_idx
        image = torchvision.io.read_image(f'{self.path}images/{idx}.jpg') / 255
        return self.transform(image), self.locations[idx]
