"""

"""
import os

from imageio import imread, imwrite
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
from Model import FinalNet
from Model import ImageDataset
from torch.utils.data import Dataset, DataLoader


from torchvision import transforms, utils, datasets
import torch

from PIL import Image

class Rotate:
    """
    Loads the trained rotator model and rotates the images accordingly
    """
    def __init__(self):
        self.model_path = "Model/final_model1.pt"
        self.path = None
        self.pics = None
        self.num_pics = None
        self.angles = [0, 270, 180, 90]
        self.model = torch.load(self.model_path)
        os.system("mkdir fixes/")


    def set_path(self, path):
        self.path = os.path.expanduser(path)
        print(path)
        print(self.path)
        if(not os.path.exists(self.path)):
            print("False")
            return False
        else:
            print("True")
            self.pics = os.listdir(self.path)
            # print(self.pics)
            self.num_pics = len(self.pics)
            if(self.num_pics == 0):
                self.num_pics = 1
            return True

    def run(self, countChanged):

        dataset = ImageDataset(self.path)

        loader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=True)

        for i, img in enumerate(loader):

            angle = self.model(img).unsqueeze(0)
            off_angle = self.angles[np.argmax(angle.detach().numpy())]
            print(off_angle)

            img2 = img.squeeze(0)
            img2 = np.swapaxes(img2, 0, 1)
            img2 = np.swapaxes(img2, 1, 2)

            print(img2.shape)
            rotated = np.absolute(ndimage.interpolation.rotate(img2, off_angle, (0, 1)))
            print(rotated.shape)
            imwrite("./fixes/" + str(i) + ".jpg", rotated)
