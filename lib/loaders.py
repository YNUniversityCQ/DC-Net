from __future__ import print_function, division
import re
import os
import math
import torch
import random
import warnings
import numpy as np
import pandas as pd
from PIL import Image
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models

warnings.filterwarnings("ignore")

def extract_and_combine_numbers(name):

    numbers = re.findall(r'\d+', name)

    combined_numbers = ''.join(numbers)

    return combined_numbers

# known transmitter locations

class DC_Net(Dataset):

    def __init__(self,maps=np.zeros(1), phase='train',
                 num1=0,num2=0,
                 data="data/",
                 numTx=80,                  
                 thresh=0.2,
                 transform=transforms.ToTensor()):

        if maps.size == 1:
            self.maps = np.arange(0, 700, 1, dtype=np.int16)
            np.random.seed(42)
            np.random.shuffle(self.maps)
        else:
            self.maps = maps

        self.data = data
        self.numTx = numTx
        self.thresh = thresh
        self.transform = transform
        self.height = 256
        self.width = 256
        self.num1 = num1
        self.num2 = num2

        if phase == 'train':
            self.num1 = 0
            self.num2 = 500
        elif phase == 'val':
            self.num1 = 501
            self.num2 = 600
        elif phase == 'test':
            self.num1 = 601
            self.num2 = 700

        self.simulation = self.data+"image/"
        self.build = self.data + "build/"
        self.antenna = self.data + "antenna/"

        
    def __len__(self):
        return (self.num2-self.num1)*self.numTx

    def __getitem__(self, idx):

        idxr = np.floor(idx/self.numTx).astype(int)
        idxc = idx-idxr*self.numTx
        dataset_map = self.maps[idxr+self.num1]

        name1 = str(dataset_map) + ".png"
        name2 = str(dataset_map) + "_" + str(idxc) + ".png"

        builds = os.path.join(self.build, name1)
        arr_build = np.asarray(io.imread(builds))

        antennas = os.path.join(self.antenna, name2)
        arr_antenna = np.asarray(io.imread(antennas))

        target = os.path.join(self.simulation, name2)
        arr_target = np.asarray(io.imread(target))

        # threshold
        if self.thresh > 0:
            arr_target = arr_target / 255
            mask = arr_target < self.thresh
            arr_target[mask] = self.thresh
            arr_target = arr_target - self.thresh*np.ones(np.shape(arr_target))
            arr_target = arr_target / (1-self.thresh)

        # To tensor
        arr_builds = self.transform(arr_build).type(torch.float32)
        arr_antennas = self.transform(arr_antenna).type(torch.float32)
        arr_targets = self.transform(arr_target).type(torch.float32)

        return arr_builds, arr_antennas, arr_targets, name2

# Unknown transmitter locations
class DC_Net2(Dataset):

    def __init__(self, maps=np.zeros(1), phase='train',
                 num1=0, num2=0,
                 data="../data/",
                 numTx=80,
                 sample_size=1,
                 add_noise=False,
                 mean=0, sigma=10,  # Noise mean and standard deviation initialization
                 sample_num=100,
                 transform=transforms.ToTensor()):

        if maps.size == 1:
            self.maps = np.arange(0, 700, 1, dtype=np.int16)
            np.random.seed(42)
            np.random.shuffle(self.maps)
        else:
            self.maps = maps

        self.data = data
        self.numTx = numTx
        self.transform = transform
        self.height = 256
        self.width = 256
        self.mean = mean
        self.sigma = sigma
        self.add_noise = add_noise
        self.sample_size = sample_size
        self.sample_num = sample_num
        self.num1 = num1
        self.num2 = num2

        if phase == 'train':
            self.num1 = 0
            self.num2 = 500
        elif phase == 'val':
            self.num1 = 500
            self.num2 = 600
        elif phase == 'test':
            self.num1 = 601
            self.num2 = 700

        self.simulation = self.data + "image/"
        self.build = self.data + "build/"
        self.antenna = self.data + "antenna/"

    def __len__(self):
        return (self.num2 - self.num1) * self.numTx

    def __getitem__(self, idx):

        idxr = np.floor(idx / self.numTx).astype(int)
        idxc = idx - idxr * self.numTx
        dataset_map = self.maps[idxr + self.num1]

        name1 = str(dataset_map) + ".png"
        name2 = str(dataset_map) + "_" + str(idxc) + ".png"

        # loading target
        target_path = os.path.join(self.simulation, name2)
        target_image = Image.open(target_path)
        target_arr = np.asarray(target_image)

        # sampling
        numbers_combine = extract_and_combine_numbers(name2)

        x_seed = '1' + numbers_combine
        y_seed = '2' + numbers_combine

        # Build whiteboard diagram (to store sampling points)
        sample_image = Image.new("L", target_image.size, "black")

        num = 0
        # sample
        for i in range((self.width - self.sample_size) * (self.height - self.sample_size)):

            # Generate random points along the upper left corner
            random.seed(int(x_seed + str(i)))
            x = random.randint(0, self.width - self.sample_size)
            random.seed(int(y_seed + str(i)))
            y = random.randint(0, self.height - self.sample_size)

            # length * width
            block = target_image.crop((x, y, x + self.sample_size, y + self.sample_size))

            # Select the sample block area that meets the conditions
            if not np.any(np.any(np.array(block) == 0, axis=0)):
                if self.add_noise == True:
                    arr_block = np.asarray(block)
                    # noise
                    gaussian_noise = np.random.normal(self.mean, self.sigma, (4, 4))
                    # fuse
                    add_noise_block = arr_block + gaussian_noise
                    # transfer image
                    block = Image.fromarray(add_noise_block.astype(np.uint8))
                sample_image.paste(block, (x, y))
                num = num + 1
            # sample num
            if num == self.sample_num:
                break

        # Does not contain a sample of the building
        sample_arr = np.asarray(sample_image)

        build_arr = np.where(target_arr == 0, 255, 0)

        image_arr = sample_arr + build_arr

        # generate masks
        mask_arr = np.where(image_arr == 0, 255, 0)

        # transfer tensor
        arr_image = self.transform(image_arr / 255).type(torch.float32)
        arr_target = self.transform(target_arr).type(torch.float32)
        arr_mask = self.transform(mask_arr / 255).type(torch.float32)

        return arr_image, arr_mask, arr_target, name2

# def test():
#     dataset = DC_Net2(phase='train')
#     loader = DataLoader(dataset, batch_size=5)
#
#     for x, y, z, w in loader:
#         print(x.shape, y.shape, z.shape, w)
#
# if __name__ == "__main__":
#     test()