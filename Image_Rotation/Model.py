"""

"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from numpy import random
from scipy import ndimage
from torchvision import transforms, utils, datasets
import torchvision
import os


class FinalNet(nn.Module):
    def __init__(self, num_classes=4):
        super(FinalNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        #     print(x.shape)
        x = self.features(x)
        #     print(x.shape)
        x = x.view(x.size(0), 256 * 7 * 7)
        x = self.classifier(x)
        return x


class ImageDataset(Dataset):
    """
    Handles the loading in of pictures and resizes them for the models to use
    """
    def __init__(self, root='./pics', size=256,
                 Train=True):
        super(ImageDataset, self).__init__()
        #     transforms.CenterCrop((size,size)), DONT USE RIGHT NOW
        self.dataset_folder = torchvision.datasets.ImageFolder(os.path.join(root), transform=transforms.Compose(
            [transforms.Resize((size, size)), transforms.ToTensor()]))
        self.angles = [0, 90, 180, 270]

    def __getitem__(self, index):
        """
        Returns a rotated image and its original unrotated counterpart
        """
        img = np.absolute(self.dataset_folder[index][0])
        return img

    def __len__(self):
        #     return 1000
        return len(self.dataset_folder)