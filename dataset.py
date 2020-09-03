import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from albumentations import (
    Compose,
    HorizontalFlip,
    Cutout,
    Resize,
    RandomCrop,
    CenterCrop,
    RandomRotate90,
    CoarseDropout,
    VerticalFlip,
)
from sklearn.model_selection import train_test_split
from utils import JPEGdecompressYCbCr


def onehot_encoding(target, size=4):
    onehot = torch.zeros(size, dtype=torch.float32)
    onehot[target] = 1.
    return onehot


def load_dataset(data, test_size=0.1):
    dataset = pd.read_csv(data)

    train_dataset, test_dataset = train_test_split(
        dataset,
        test_size=test_size,
        shuffle=True,
        stratify=dataset["Label"],
    )

    return train_dataset, test_dataset


def transforms(size, p=0.5):
    area = size ** 2

    transform = [
        CoarseDropout(
            max_holes=256, max_height=8, max_width=8, 
            min_holes=16, min_height=1, min_width=1, p=p
        ),
        HorizontalFlip(p=p),
        VerticalFlip(p=p),
        RandomRotate90(p=p),
        Resize(size, size),
    ]
    return Compose(transform)


def inference_transforms(size):
    transform = [
        # CenterCrop(448, 448, p=1.),
        Resize(size, size),
    ]
    return Compose(transform)



class AlaskaDataset(data.Dataset):
    def __init__(self, data, folder="input", train=True, ycbcr=False, size=512):
        self.data = data
        self.folder = folder
        self.train = train
        self.ycbcr = ycbcr
        self.size = size
        self.transforms = transforms(size)
        self.inference_transforms = inference_transforms(size)

    def __getitem__(self, index):
        item = self.data.iloc[index]

        algorithm, name = item["Image"].split('/')
        label = item["Label"]

        image = self.load_image(algorithm, name)

        if self.train:
            t = self.transforms(image=image)
        else:
            t = self.inference_transforms(image=image)

        image = t["image"]

        return self.to_tensor(image), onehot_encoding(label)#torch.tensor(label).long()

    def __len__(self):
        return len(self.data)

    def load_image(self, algorithm, name):
        path = os.path.join(self.folder, algorithm, name)
        # image = cv2.imread(path, cv2.IMREAD_COLOR)

        if self.ycbcr:
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
            image = JPEGdecompressYCbCr(path)
        else:
            image = cv2.imread(path, cv2.IMREAD_COLOR)

        return image

    def to_tensor(self, x):
        x = x.transpose(2, 0, 1)
        if x.dtype == np.uint8:
            x = x / 255
        else:
            x = x / 128
        return torch.from_numpy(x).float()
