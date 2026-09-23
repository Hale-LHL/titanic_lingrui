"""dataset：把特征和标签打包成 Dataset"""

import torch
from torch.utils.data import Dataset


class TitanicDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X)
        self.y = torch.as_tensor(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]
