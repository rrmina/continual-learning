import torch
import numpy as np

import torchvision

class PermutedMNIST(torchvision.datasets.MNIST):
    def __init__(self, root="data", transform=None, train=True, permute_idx=None):
        super(PermutedMNIST, self).__init__(root, train=train, download=True, transform=transform)
        assert len(permute_idx) == 28 * 28
        if self.train:
            m = self.train_data.shape[0]
            self.train_data_new = self.train_data.view(m, -1).T[permute_idx].T.view(-1, 28, 28) / 255
        else:
            m = self.test_data.shape[0]
            self.test_data_new = self.test_data.view(m, -1).T[permute_idx].T.view(-1, 28, 28) / 255
    
    def __getitem__(self, index):
        if self.train:
            return self.train_data_new[index], self.train_labels[index]
        else:
            return self.test_data_new[index], self.test_labels[index]
            