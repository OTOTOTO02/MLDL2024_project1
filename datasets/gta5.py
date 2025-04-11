from torch.utils.data import Dataset

from abc import ABCMeta
from dataclasses import dataclass
from typing import Tuple

import os
import cv2
from PIL import Image
from tqdm import tqdm

import numpy as np

class BaseGTALabels(metaclass=ABCMeta):
    pass


@dataclass
class GTA5Label:
    ID: int
    color: Tuple[int, int, int]


class GTA5Labels_TaskCV2017(BaseGTALabels):
    road = GTA5Label(ID=0, color=(128, 64, 128))
    sidewalk = GTA5Label(ID=1, color=(244, 35, 232))
    building = GTA5Label(ID=2, color=(70, 70, 70))
    wall = GTA5Label(ID=3, color=(102, 102, 156))
    fence = GTA5Label(ID=4, color=(190, 153, 153))
    pole = GTA5Label(ID=5, color=(153, 153, 153))
    light = GTA5Label(ID=6, color=(250, 170, 30))
    sign = GTA5Label(ID=7, color=(220, 220, 0))
    vegetation = GTA5Label(ID=8, color=(107, 142, 35))
    terrain = GTA5Label(ID=9, color=(152, 251, 152))
    sky = GTA5Label(ID=10, color=(70, 130, 180))
    person = GTA5Label(ID=11, color=(220, 20, 60))
    rider = GTA5Label(ID=12, color=(255, 0, 0))
    car = GTA5Label(ID=13, color=(0, 0, 142))
    truck = GTA5Label(ID=14, color=(0, 0, 70))
    bus = GTA5Label(ID=15, color=(0, 60, 100))
    train = GTA5Label(ID=16, color=(0, 80, 100))
    motocycle = GTA5Label(ID=17, color=(0, 0, 230))
    bicycle = GTA5Label(ID=18, color=(119, 11, 32))
    void = GTA5Label(ID=255, color=(0,0,0))

    list_ = [
        road,
        sidewalk,
        building,
        wall,
        fence,
        pole,
        light,
        sign,
        vegetation,
        terrain,
        sky,
        person,
        rider,
        car,
        truck,
        bus,
        train,
        motocycle,
        bicycle,
        void
    ]

    @property
    def support_id_list(self):
        ret = [label.ID for label in self.list_]
        return ret

class GTA5(Dataset):
    def __init__(self, rootdir, split, train_split_percent, targetdir="labels", imgdir="images", transform=None, target_transform=None):
        super(GTA5, self).__init__()

        assert split in ["train", "val"]
        assert train_split_percent >= 0 and train_split_percent <= 1

        self.rootdir = rootdir
        self.split = split
        self.section = int(self.split=="train")
        self.train_split_percent = train_split_percent
        self.targetdir = os.path.join(self.rootdir, targetdir) # ./labels
        self.imgdir = os.path.join(self.rootdir, imgdir) # ./images
        self.transform = transform
        self.target_transform = target_transform

        all_names = os.listdir(self.imgdir)

        self.imgs_path = [[],[]]
        self.targets_color_path = [[],[]]
        self.targets_labelIds_path = [[],[]]

        self.train_val_mask = np.ones(len(all_names), dtype=bool)
        self.train_val_mask[:int(len(all_names)*(1-self.train_split_percent))] = False

        # np.random.seed(42) # reproducibility
        np.random.shuffle(self.train_val_mask)

        for i, image_file in enumerate(all_names): # 00001.png
            if image_file.endswith(".png"):
                section = int(self.train_val_mask[i])
                self.imgs_path[section].append(os.path.join(self.imgdir, image_file)) #./images/00001.png
                 
                target_color_path = image_file # 00001.png
                target_labelsId_path = image_file.split(".")[0]+"_labelIds.png" # labelIds_00001.png

                self.targets_color_path[section].append(os.path.join(self.targetdir, target_color_path)) #./labels/00001.png
                self.targets_labelIds_path[section].append(os.path.join(self.targetdir, target_labelsId_path)) #./labels/labelIDs_00001.png

    def create_target_img(self):
        for i, img_path in tqdm(enumerate(self.targets_color_path)):            
            image_numpy = np.asarray(Image.open(img_path).convert('RGB'))

            H, W, _ = image_numpy.shape
            label_image = 255*np.ones((H, W), dtype=np.uint8)

            for label in GTA5Labels_TaskCV2017().list_:
                label_image[(image_numpy == label.color).all(axis=-1)] = label.ID

            new_img = Image.fromarray(label_image)
            new_img.save(self.targets_labelIds_path[i])

    def __getitem__(self, idx):
        image = Image.open(self.imgs_path[self.section][idx]).convert('RGB')

        target_color = Image.open(self.targets_color_path[self.section][idx]).convert('RGB')
        target_labelIds = cv2.imread(self.targets_labelIds_path[self.section][idx], cv2.IMREAD_UNCHANGED)

        if self.transform is not None:
            image = self.transform(image)
            target_color = self.transform(target_color)
        if self.target_transform is not None:
            target_labelIds = self.target_transform(target_labelIds)

        return image, target_color, target_labelIds

    def __len__(self):
        return len(self.imgs_path[self.section])
