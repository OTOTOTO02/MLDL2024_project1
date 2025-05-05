import torch
from torch.utils.data import Dataset
import numpy as np

import os
import cv2
from PIL import Image

from tqdm import tqdm
import random

import torchvision.transforms as TF
from utils import GTA5Labels_TaskCV2017

# TODO: decide other augmentations
class GTA5(Dataset):
    def __init__(self, rootdir, file_names, imgdir="images", targetdir="labels", augment=False, transform=None, target_transform=None):
        super(GTA5, self).__init__()

        self.rootdir = rootdir
        
        self.targetdir = os.path.join(self.rootdir, targetdir) # ./labels
        self.imgdir = os.path.join(self.rootdir, imgdir) # ./images

        self.augment = augment

        self.transform = transform
        self.target_transform = target_transform

        self.imgs_path = []
        self.targets_color_path = []
        self.targets_labelIds_path = []

        for image_file in file_names: # 00001.png
            self.imgs_path.append(os.path.join(self.imgdir, image_file)) #./images/00001.png

            target_color_path = image_file # 00001.png
            target_labelsId_path = image_file.split(".")[0]+"_labelIds.png" # 00001_labelIds.png

            self.targets_color_path.append(os.path.join(self.targetdir, target_color_path)) #./labels/00001.png
            self.targets_labelIds_path.append(os.path.join(self.targetdir, target_labelsId_path)) #./labels/00001_labelIDs.png

    def create_target_img(self):
        list_ = GTA5Labels_TaskCV2017().list_

        for i, img_path in tqdm(enumerate(self.targets_color_path)):
            image_numpy = np.asarray(Image.open(img_path).convert('RGB'))

            H, W, _ = image_numpy.shape
            label_image = 255*np.ones((H, W), dtype=np.uint8)

            for label in list_:
                label_image[(image_numpy == label.color).all(axis=-1)] = label.ID

            new_img = Image.fromarray(label_image)
            new_img.save(self.targets_labelIds_path[i])

    def __getitem__(self, idx):
        image = Image.open(self.imgs_path[idx]).convert('RGB')

        target_color = Image.open(self.targets_color_path[idx]).convert('RGB')
        target_labelIds = cv2.imread(self.targets_labelIds_path[idx], cv2.IMREAD_UNCHANGED).astype(np.int64)

        if self.transform is not None:
            image = self.transform(image)
            target_color = self.transform(target_color)
        if self.target_transform is not None:
            target_labelIds = self.target_transform(target_labelIds)

        if self.augment:
            image, target_color, target_labelIds = self.augment_data(image, target_color, target_labelIds)
        return image, target_color, target_labelIds

    def __len__(self):
        return len(self.imgs_path)

    def augment_data(self, image, target_color, target_labelIds):
        # original = image.detach().clone()
        val = random.random()
    # Geometric Transformations

        # Random horizontal flipping
        if val < 0.5:
            image = TF.functional.hflip(image)
            target_color = TF.functional.hflip(target_color)
            target_labelIds = TF.functional.hflip(target_labelIds)

    # Photometric Transformations

        # Gaussian Blur
        if val < 0.5:
            image = TF.functional.gaussian_blur(image, 5)

        # Multiply
        if val < 0.5:
            factor = random.uniform(0.6, 1.2)
            image *= factor
            image = torch.clamp(image, 0, 255)

        return image, target_color, target_labelIds

def GTA5_dataset_splitter(rootdir, train_split_percent, split_seed = None, imgdir="images", targetdir="labels", augment=False, transform=None, target_transform=None):
    assert 0.0 < train_split_percent < 1.0, "train_split_percent should be a float between 0 and 1"

    target_path = os.path.join(rootdir, targetdir) # ./labels
    img_path = os.path.join(rootdir, imgdir) # ./images

    file_names = [
        f for f in os.listdir(img_path)
        if f.endswith(".png") and os.path.exists(os.path.join(target_path, f.split(".")[0]+"_labelIds.png"))
    ]

    if split_seed is not None:
        random.seed(split_seed)
    random.shuffle(file_names)
    random.seed()

    split_idx = int(len(file_names) * train_split_percent)

    train_files = file_names[:split_idx]
    val_files = file_names[split_idx:]

    return GTA5(rootdir, train_files, imgdir, targetdir, augment, transform, target_transform), \
           GTA5(rootdir, val_files, imgdir, targetdir, False, transform, target_transform)
