from torch.utils.data import Dataset

import os
from torchvision.transforms import ToTensor
import cv2
from PIL import Image

# TODO: chack support for test split is gt is not available
# TODO: check coherence with dataset on Google Drive (works with official dataset not the reduced one from profs)

class CityScapes(Dataset):
    def __init__(self, rootdir=".", targetdir="gtFine", imgdir="leftImg8bit", split="train", transform=ToTensor(), target_transform=None):
        super(CityScapes, self).__init__()
        
        self.rootdir = rootdir
        self.split = split
        self.targetdir = os.path.join(self.rootdir, targetdir, self.split) # ./gtFine/train/
        self.imgdir = os.path.join(self.rootdir, imgdir, self.split) # ./leftImg8bit/train/
        self.transform = transform
        self.target_transform = target_transform

        self.imgs_path = []
        self.targets_color_path = []
        self.targets_instanceIds_path = []
        self.targets_labelIds_path = []

        for city in os.listdir(self.imgdir): # frankfurt
            img_city_dir = os.path.join(self.imgdir, city) # ./leftImg8bit/train/frankfurt/ 
            target_city_dir = os.path.join(self.targetdir, city) # ./gtFine/train/frankfurt/

            for img_path in os.listdir(img_city_dir): # frankfurt_000000_000294_leftImg8bit.png
                if img_path.endswith(".png"):
                  self.imgs_path.append(os.path.join(img_city_dir, img_path)) # ./leftImg8bit/train/frankfurt/frankfurt_000000_000294_leftImg8bit.png
                  
                  target_color_path = img_path.replace("leftImg8bit", "gtFine_color") # frankfurt_000000_000294_gtFine_color.png
                  target_instanceIds_path = img_path.replace("leftImg8bit", "gtFine_instanceIds") # frankfurt_000000_000294_gtFine_instanceIds.png
                  target_labelIds_path = img_path.replace("leftImg8bit", "gtFine_labelIds") # frankfurt_000000_000294_gtFine_labelIds.png
                  
                  self.targets_color_path.append(os.path.join(target_city_dir, target_color_path)) # ./gtFine/train/frankfurt/frankfurt_000000_000294_gtFine_color.png
                  self.targets_instanceIds_path.append(os.path.join(target_city_dir, target_instanceIds_path)) # ./gtFine/train/frankfurt/frankfurt_000000_000294_gtFine_instanceIds.png
                  self.targets_labelIds_path.append(os.path.join(target_city_dir, target_labelIds_path)) # ./gtFine/train/frankfurt/frankfurt_000000_000294_gtFine_labelIds.png

    def __getitem__(self, idx):
        image = Image.open(self.imgs_path[idx]).convert('RGB')

        target_color = Image.open(self.targets_color_path[idx]).convert('RGB')
        target_instanceIds = cv2.imread(self.targets_instanceIds_path[idx], cv2.IMREAD_UNCHANGED)
        target_labelIds = cv2.imread(self.targets_labelIds_path[idx], cv2.IMREAD_UNCHANGED)

        if self.transform is not None:
            image = self.transform(image)
            target_color = self.transform(target_color)
        if self.target_transform is not None:
            target_instanceIds = self.target_transform(target_instanceIds)
            target_labelIds = self.target_transform(target_labelIds)

        return image, target_color, target_labelIds, target_instanceIds
        
    def __len__(self):
        return len(self.imgs_path)
