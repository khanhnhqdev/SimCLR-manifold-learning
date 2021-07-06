import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

get_ipython().system('pip install opendatasets --upgrade')
import opendatasets as od
dataset_url = 'https://www.kaggle.com/huangruichu/caltech101/version/2'
od.download(dataset_url)
import os

DATA_DIR = './caltech101'
print(os.listdir(DATA_DIR))