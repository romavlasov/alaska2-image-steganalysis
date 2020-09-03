import argparse
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from tqdm import tqdm
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

import models
from utils import JPEGdecompressYCbCr

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def inference_transforms(size):
    transform = [
        # CenterCrop(size, size, p=1.),
        # RandomCrop(size, size, p=1.)
        # HorizontalFlip(p=p),
        # VerticalFlip(p=p),
        # RandomRotate90(p=p),
        Resize(size, size),
    ]
    return Compose(transform)


class AlaskaTestDataset(data.Dataset):
    def __init__(self, folder="input", ycbcr=False, size=512):
        self.data = pd.read_csv(os.path.join(folder, "sample_submission.csv"))
        self.folder = os.path.join(folder, "Test")
        self.ycbcr = ycbcr
        self.size = size
        self.inference_transforms = inference_transforms(size)

    def __getitem__(self, index):
        item = self.data.iloc[index]

        image = self.load_image(os.path.join(self.folder, item["Id"]))
        image = self.inference_transforms(image=image)["image"]

        return self.to_tensor(image), item["Id"]

    def __len__(self):
        return len(self.data)

    def load_image(self, path):
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


def main(args):
    model = models.build_model(encoder=args.encoder, pretrained=False).to(device)
    model.load_state_dict(
        torch.load(args.weights, map_location=lambda storage, loc: storage)
    )
    model.eval()

    test_loader = data.DataLoader(
        AlaskaTestDataset(ycbcr=args.ycbcr, size=args.size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )

    names, probabilities = [], []

    with torch.no_grad():
        for source, name in tqdm(test_loader):
            source = source.to(device)

            probability = 0

            for k in range(1):
                image = source.rot90(k=k, dims=(2, 3))

                logit_n = model(image)
                logit_h = model(image.flip(2))
                logit_v = model(image.flip(3))
                logit_b = model(image.flip(3).flip(2))

                probability_n = torch.softmax(logit_n, dim=1)
                probability_h = torch.softmax(logit_h, dim=1)
                probability_v = torch.softmax(logit_v, dim=1)
                probability_b = torch.softmax(logit_b, dim=1)

                probability += (
                    probability_n + probability_h + probability_v + probability_b
                ) / 4

            probability /= 1

            names.extend(name)
            probabilities.extend(probability.cpu().numpy())

    if args.probability:
        probability = pd.DataFrame.from_dict({"Id": names, "probabilities": probabilities})
        probability = probability.groupby("Id").probabilities.apply(lambda x: np.mean(x)).reset_index()
        probability.to_csv(args.probability, index=False)


    probabilities = np.array(probabilities)
    labels = probabilities.argmax(1)

    aggregated = np.zeros((len(probabilities),))

    aggregated[labels != 0] = probabilities[labels != 0, 1:].sum(1)
    aggregated[labels == 0] = 1 - probabilities[labels == 0, 0]

    predictions = pd.DataFrame.from_dict({"Id": names, "probability": aggregated})

    predictions = predictions.groupby("Id").probability.mean().reset_index()

    predictions["Label"] = predictions.probability.tolist()
    predictions[["Id", "Label"]].to_csv(args.submission, index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--ycbcr", action="store_true")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--probability", type=str, default=None)
    parser.add_argument("--submission", type=str, required=True)
    args = parser.parse_args()

    main(args)
