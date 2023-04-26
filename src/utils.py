import os
import time
import pickle
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


# Function to load data from each participant file
def read_eeg_signal_from_file(filepath):
    return pickle.load(open(filepath, "rb"), encoding='latin1')


class NumpyArrayDataset(Dataset):
    def __init__(self, data, labels=None):
        self.data = torch.tensor(data, dtype=torch.float32)
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.float32)
        else:
            self.labels = None

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        x = self.data[index]
        if self.labels is not None:
            y = self.labels[index]
            return x, y
        else:
            return x