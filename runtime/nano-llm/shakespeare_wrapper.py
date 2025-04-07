import torch
from torch.utils.data import Dataset
import numpy as np
import os

class ShakespeareDataset(Dataset):
    def __init__(self, data_path, block_size=1024, split='train'):
        assert split in ['train', 'val']
        bin_path = os.path.join(data_path, f'{split}.bin')
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.from_numpy((self.data[idx:idx+self.block_size]).astype(np.int64))
        y = torch.from_numpy((self.data[idx+1:idx+1+self.block_size]).astype(np.int64))
        return x, y
