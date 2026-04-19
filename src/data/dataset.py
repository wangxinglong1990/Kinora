#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import Dataset


class MultiTaskKineticsDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).view(-1, 2)

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return {
            "x": self.features[idx],
            "y": self.targets[idx],
        }

