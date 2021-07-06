"""
This code is based on the Torchvision repository, which was licensed under the BSD 3-Clause.
"""
import os
import pickle
import sys
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils.mypath import MyPath
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision import datasets, models, transforms

data_dir = "./dataset/caltech101/Caltech101/Caltech101/"


def caltech101(root=data_dir, train=True, transform=None):
    if train:
        return datasets.ImageFolder(root + '/train', transform)
    else:
        return datasets.ImageFolder(root + '/test', transform)


class CALTECH101(Dataset):
    """`Caltech _ Dataset.
    """
    data_dir = "./dataset/caltech101/Caltech101/Caltech101/"

    def __init__(self, root=data_dir, train=True, transform=None):
        super(CALTECH101, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train  # training set or test set
        self.classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        if self.train:
            self.dataset = datasets.ImageFolder(self.root + '/train', transform)
        else:
            self.dataset = datasets.ImageFolder(self.root + '/test', transform)

    def __getitem__(self, index):
        sample = self.dataset.__getitem__(index)
        img = sample[0]
        target = sample[1]
        out = {'image': img, 'target': target}
        return out

    def __len__(self):
        return self.dataset.__len__()
