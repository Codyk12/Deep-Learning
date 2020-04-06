import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
import torchvision

import numpy as np
import os
import random
import matplotlib.pyplot as plt

from torchvision.transforms import ToPILImage
from scipy import ndimage
to_img = ToPILImage()

epochs = 60
epoch_loss = []

class ImageDataset(Dataset):
    """
    Handles the loading in of pictures and resizes them for the models to use
    """

    def __init__(self, root='/pics/', size=256,
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
        angle = random.choice(self.angles)
        rotated = torch.tensor(np.absolute(ndimage.interpolation.rotate(img, angle, (1, 2))))
        return rotated, self.angles.index(angle)

    def __len__(self):
        return len(self.dataset_folder)


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
        x = self.features(x)
        x = x.view(x.size(0), 256 * 7 * 7)
        x = self.classifier(x)
        return x

def train():
    train_dataset = ImageDataset(Train=True)
    train_loader = DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=True)

    net = FinalNet().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0005, momentum=0.9, weight_decay=0.0005)

    for epoch in range(epochs):  # loop over the dataset multiple times

        batch_loss = []
        for i, data in enumerate(train_loader):
            # get the inputs

            rotated_imgs, angles = data
            rotated_imgs = rotated_imgs.cuda()
            angles = angles.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            outputs = net(rotated_imgs)

            loss = criterion(outputs, angles)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

            if i % 10 == 0:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, batch_loss[-1]))
                print(angles)
                print(outputs)

        torch.save(net.cpu(), '/content/gdrive/My Drive/final_model.pt')
        epoch_loss.append(np.mean(batch_loss))

    print('Finished Training', np.mean(epoch_loss))

def plot():
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

if __name__ == "__name__":

    train()
    plot()
