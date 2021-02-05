import torch
import numpy as np
import random
import torchvision

import os

from copy import deepcopy

def get_loaders(args, indices=None, transform=None):
    train_loaders, test_loaders = {}, {}
    if (indices == None):
        indices = {}
        permute_idx = [i for i in range(28 * 28)]
        for i in range(args.num_tasks):
            indices[i] = deepcopy(permute_idx)
            random.shuffle(permute_idx)

        torch.save(indices, os.path.join("results", "curr_permute_1024" + ".permute_indices"))

    for i in range(args.num_tasks):
        permute_idx = indices[i]
        train_dataset = PermutedMNIST(transform=transform, train=True, permute_idx=permute_idx)
        train_loaders[i] = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_dataset = PermutedMNIST(transform=transform, train=False, permute_idx=permute_idx)
        test_loaders[i] = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)

    return train_loaders, test_loaders

class PermutedMNIST(torchvision.datasets.MNIST):
    def __init__(self, root="data", transform=None, train=True, permute_idx=None):
        super(PermutedMNIST, self).__init__(root, train=train, download=True, transform=transform)
        assert len(permute_idx) == 28 * 28

        m = self.data.shape[0]
        self.data = self.data.view(m, -1).T[permute_idx].T.view(-1, 28, 28) / 255
    
    def __getitem__(self, index):
        return self.data[index], self.targets[index]
