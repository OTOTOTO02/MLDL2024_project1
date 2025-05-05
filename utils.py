import numpy as np
import torch

from dataclasses import dataclass
from typing import Tuple

import wandb
import zipfile

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm

import time
from fvcore.nn import FlopCountAnalysis, flop_count_table

def pretty_extract(zip_path:str, extract_to:str) -> None:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        png_files = [f for f in zip_ref.namelist() if f.endswith('.png')]

        for file in tqdm(png_files, desc="Extracting PNGs"):
            zip_ref.extract(file, path=extract_to)

def poly_lr_scheduler(optimizer, init_lr:float, iter:int=0, lr_decay_iter:int=1, max_iter:int=50, power:float=0.9) -> float:
    """Polynomial decay of learning rate
            :param init_lr is base learning rate
            :param iter is a current iteration
            :param lr_decay_iter how frequently decay occurs, default is 1
            :param max_iter is number of maximum iterations
            :param power is a polymomial power

    """
    if ((iter % lr_decay_iter) != 0) or iter > max_iter:
        return optimizer.param_groups[0]['lr']

    lr = init_lr*(1 - iter/max_iter)**power
    optimizer.param_groups[0]['lr'] = lr
    return lr

def fast_hist(a:np.ndarray, b:np.ndarray, n:int) -> np.ndarray:
    '''
    a and b are label and prediction respectively
    n is the number of classes
    '''
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)

def per_class_iou(hist:np.ndarray) -> np.ndarray:
    epsilon = 1e-5
    return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)


@dataclass
class GTA5Label:
    name: str
    ID: int
    color: Tuple[int, int, int]

class GTA5Labels_TaskCV2017():
    road = GTA5Label(name = "road", ID=0, color=(128, 64, 128))
    sidewalk = GTA5Label(name = "sidewalk", ID=1, color=(244, 35, 232))
    building = GTA5Label(name = "building", ID=2, color=(70, 70, 70))
    wall = GTA5Label(name = "wall", ID=3, color=(102, 102, 156))
    fence = GTA5Label(name = "fence", ID=4, color=(190, 153, 153))
    pole = GTA5Label(name = "pole", ID=5, color=(153, 153, 153))
    light = GTA5Label(name = "light", ID=6, color=(250, 170, 30))
    sign = GTA5Label(name = "sign", ID=7, color=(220, 220, 0))
    vegetation = GTA5Label(name = "vegetation", ID=8, color=(107, 142, 35))
    terrain = GTA5Label(name = "terrain", ID=9, color=(152, 251, 152))
    sky = GTA5Label(name = "sky", ID=10, color=(70, 130, 180))
    person = GTA5Label(name = "person", ID=11, color=(220, 20, 60))
    rider = GTA5Label(name = "rider", ID=12, color=(255, 0, 0))
    car = GTA5Label(name = "car", ID=13, color=(0, 0, 142))
    truck = GTA5Label(name = "truck", ID=14, color=(0, 0, 70))
    bus = GTA5Label(name = "bus", ID=15, color=(0, 60, 100))
    train = GTA5Label(name = "train", ID=16, color=(0, 80, 100))
    motocycle = GTA5Label(name = "motocycle", ID=17, color=(0, 0, 230))
    bicycle = GTA5Label(name = "bicycle", ID=18, color=(119, 11, 32))
    void = GTA5Label(name = "void", ID=255, color=(0,0,0))

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

# Mapping labelId image to RGB image
def decode_segmap(mask:np.ndarray) -> np.ndarray:
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for label_id in GTA5Labels_TaskCV2017().list_:
        color_mask[mask == label_id.ID, :] = label_id.color

    return color_mask

def tensorToImageCompatible(t:torch.Tensor) -> np.ndarray:
    """
    convert from a tensor of shape [C, H, W] where a normalization has been applied
    to an unnormalized tensor of shape [H, W, C],
    so *plt.imshow(tensorToImageCompatible(tensor))* works as expected.\n
    Intended to be used to recover the original element
    when this transformation is used:
    - transform = TF.Compose([
        TF.ToTensor(),
        TF.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])])
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view([-1, 1, 1])
    std = torch.tensor([0.229, 0.224, 0.225]).view([-1, 1, 1])

    unnormalized = t * std + mean

    return (unnormalized.permute(1,2,0).clamp(0,1).numpy()*255).astype(np.uint8)


def log_confusion_matrix(title:str, hist:np.ndarray, tag:str, step_name:str, step_value:int):
    row_sums = hist.sum(axis=1, keepdims=True)
    safe_hist = np.where(row_sums == 0, 0, hist / row_sums)

    plt.figure(figsize=(10, 8))
    sns.heatmap(100.*safe_hist, fmt=".2f", annot=True, cmap="Blues", annot_kws={'size': 7})
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)

    wandb.log({tag: wandb.Image(plt), step_name:step_value})
    plt.close()

def num_flops(device, model:torch.nn.Module, H:int, W:int):
    model.eval()
    img = (torch.zeros((1,3,H,W), device=device),)

    flops = FlopCountAnalysis(model, img)
    return flop_count_table(flops)

def latency(device, model:torch.nn.Module, H:int, W:int):
    model.eval()

    img = torch.zeros((1,3,H,W)).to(device)
    iterations = 100
    latency_list = []
    FPS_list  = []

    with torch.no_grad():
        for _ in tqdm(range(iterations)):
            start_time = time.time()
            _ = model(img)
            end_time = time.time()

            latency = end_time - start_time

            latency_list.append(latency)
            FPS_list.append(1.0/latency)

    mean_latency = np.mean(latency_list)*1000
    std_latency = np.std(latency_list)*1000
    mean_FPS = np.mean(FPS_list)

    return mean_latency, std_latency, mean_FPS
