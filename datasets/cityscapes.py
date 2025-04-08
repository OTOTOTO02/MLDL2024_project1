from torch.utils.data import Dataset

import os
from torchvision.transforms import ToTensor
import cv2
from PIL import Image

class CityScapes(Dataset):
    def __init__(self, rootdir=".", targetdir="gtFine", imgdir="leftImg8bit", split="train", transform=ToTensor(), target_transform=None, instanceSegmentation=False, visualization=False):
        super(CityScapes, self).__init__()
        
        self.rootdir = rootdir
        self.split = split
        self.targetdir = os.path.join(self.rootdir, targetdir, self.split) # ./gtFine/train/
        self.imgdir = os.path.join(self.rootdir, imgdir, self.split) # ./leftImg8bit/train/
        self.transform = transform
        self.target_transform = target_transform

        self.imgs_path = []
        self.targets_path = []

        self.istanceSegmentation=instanceSegmentation
        self.visualization=visualization

        if self.istanceSegmentation:
            self.target_name = "instanceIds"
        elif self.visualization:
            self.target_name = "color"
        else:
            self.target_name = "labelIds"

        for city in os.listdir(self.imgdir): # frankfurt
            img_city_dir = os.path.join(self.imgdir, city) # ./gtFine/train/frankfurt/
            target_city_dir = os.path.join(self.targetdir, city) # ./leftImg8bit/train/frankfurt/

            for img_path in os.listdir(img_city_dir): # frankfurt_000000_000294_leftImg8bit.png
                if img_path.endswith(".png"):
                  self.imgs_path.append(os.path.join(img_city_dir, img_path)) # ./leftImg8bit/train/frankfurt/frankfurt_000000_000294_leftImg8bit.png
                  target_path = img_path.replace("leftImg8bit", "gtFine_"+ self.target_name) # frankfurt_000000_000294_gtFine_color.png
                  self.targets_path.append(os.path.join(target_city_dir, target_path)) # ./gtFine/train/frankfurt/frankfurt_000000_000294_gtFine_color.png

    def __getitem__(self, idx):
        image = Image.open(self.imgs_path[idx]).convert('RGB')
        if self.visualization:
            target = Image.open(self.targets_path[idx]).convert('RGB')
        else:
            target = cv2.imread(self.targets_path[idx], cv2.IMREAD_UNCHANGED)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target
        
    def __len__(self):
        return len(self.imgs_path)
