# epoch = 1 forwad and backward pass of all training samples
# batch_size = number of training samples in one forward and backward pass
# number of iterations = number of passes, each pass using [batch_size] number of samples
# e.g. 100 samples, batch_size=20 -> 100/20 = 5 iterations for 1 epoch

import torch
import torchvision 
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import math  

class WineDataset(Dataset):
    def __init__(self, transform=None):
        # data Loading 
        xy = np.loadtxt('./wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = xy[:, 1:] # n_samples, n_features
        self.y = xy[:, [0]] # n_samples, 1 
        self.n_samples = xy.shape[0] # number of samples， the row number of xy
        self.transform = transform


    def __getitem__(self, index):
        sample = self.x[index], self.y[index] # dataset[0]

        if self.transform:
            sample = self.transform(sample)
        
        return sample 

    def __len__(self):
        return self.n_samples 

class ToTensor():
    def __call__(self, sample):
        inputs, targets = sample 
        return torch.from_numpy(inputs), torch.from_numpy(targets)

class MulTransform():
    def __init__(self,factor):
        self.factor = factor 

    def __call__(self, sample):
        inputs, targets = sample 
        inputs *= self.factor 
        return inputs, targets 


print('Without Transform')
dataset = WineDataset()
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)

print('\nWith Transform')
dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data 
print(type(features), type(labels))
print(features, labels)


print('\nWith Tensor and MulTransform')
composed = torchvision.transforms.Compose([ToTensor(), MulTransform(3)])
dataset = WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(type(features), type(labels))
print(features, labels)