import random
import numpy as np
import torch

from dataclasses import dataclass
from typing import Tuple

import zipfile

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm

import time
from fvcore.nn import FlopCountAnalysis, flop_count_table

from collections import defaultdict
import json

from torchvision.transforms import v2


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

def fast_hist_cuda(a: torch.Tensor, b: torch.Tensor, n: int) -> torch.Tensor:
    """
    a and b are label and prediction respectively.
    n is the number of classes.
    This version works with CUDA tensors.
    """
    k = (a >= 0) & (a < n)
    a = a[k].to(torch.int64)
    b = b[k].to(torch.int64)

    indices = n * a + b

    hist = torch.zeros(n * n, dtype=torch.float32, device=a.device)

    hist.scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.float32))

    return hist.view(n, n)

def per_class_iou(hist:np.ndarray) -> np.ndarray:
    epsilon = 1e-5
    return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)

def per_class_iou_cuda(hist:torch.Tensor) -> torch.Tensor:
    epsilon = 1e-5
    diag = torch.diag(hist)

    sum_rows = hist.sum(dim=1)
    sum_cols = hist.sum(dim=0)

    iou = diag / (sum_rows + sum_cols - diag + epsilon)
    return iou

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
    - transform = v2.Compose([
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
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

def log_bar_chart_ioU(title:str, class_names:list, mIou:float, iou_class:np.ndarray, tag:str, step_name:str, epoch:int):
    iou_percent = [round(iou*100., 2) for iou in iou_class]
    miou_percent = round(mIou*100., 2)

    all_labels = ["mIoU"] + class_names
    all_values = [miou_percent] + iou_percent

    plt.figure(figsize=(14, 5))
    bars = plt.bar(range(len(all_values)), all_values, color='skyblue')
    plt.xticks(range(len(all_labels)), all_labels, rotation=45, ha="right")

    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 1, f'{height:.2f}',
                ha='center', va='bottom', fontsize=9)

    plt.ylabel("IoU (%)")
    plt.ylim(0, 105)
    plt.title(title)
    plt.tight_layout()


def num_flops(device, model:torch.nn.Module, H:int, W:int):
    model.eval()
    img = (torch.zeros((1,3,H,W), device=device),)

    flops = FlopCountAnalysis(model, img)
    # return flop_count_table(flops)
    return flops.total()/1e9

def num_param(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6

def latency(device, model:torch.nn.Module, H:int, W:int):
    model.eval()

    img = torch.zeros((1,3,H,W)).to(device)
    iterations = 1000
    latency_list = []
    FPS_list  = []

    with torch.no_grad():
        for _ in tqdm(range(iterations)):
            start_time = time.time()
            _ = model(img)
            end_time = time.time()

            latency = end_time - start_time

            if latency > 0.0:
                latency_list.append(latency)
                FPS_list.append(1.0/latency)

    mean_latency = np.mean(latency_list)*1000
    std_latency = np.std(latency_list)*1000
    mean_FPS = np.mean(FPS_list)

    return mean_latency, std_latency, mean_FPS

class RandomCrop:
    def __init__(self, size):
        self.size = tuple(size)

    def __call__(self, imgs, labels):
        B, C, H, W = imgs.size()
        th, tw = self.size

        imgs_cropped = []
        labels_cropped = []

        for i in range(B):
            x1 = random.randint(0, W - tw)
            y1 = random.randint(0, H - th)

            imgs_cropped.append(
                v2.functional.crop(imgs[i], y1, x1, th, tw)
            )

            if labels is not None:
                labels_cropped.append(
                    v2.functional.crop(labels[i], y1, x1, th, tw)
                )

        imgs_cropped = torch.stack(imgs_cropped)

        if labels is not None:
            labels_cropped = torch.stack(labels_cropped)
            return imgs_cropped, labels_cropped
        else:
            return imgs_cropped, None

def display_result_of_mixing(uleft, ucenter, uright, lleft, lcenter, lright):
    fig = plt.figure(figsize=(15,10))
    axs = fig.subplots(2, 3)

    axs[0, 0].imshow(tensorToImageCompatible(uleft))
    axs[0, 1].imshow(tensorToImageCompatible(ucenter))
    axs[0, 2].imshow(tensorToImageCompatible(uright))

    axs[1, 0].imshow(decode_segmap(lleft))
    axs[1, 1].imshow(decode_segmap(lcenter))
    axs[1, 2].imshow(decode_segmap(lright))

    fig.show()
    
def FDA(src_img: torch.Tensor, tgt_img: torch.Tensor, beta: float = 0.01) -> torch.Tensor:
    # FFT2 is applied to each of the 3 RGB channels
    # src_img: torch Tensor [3, H, W ]
    # tgt_img: torch Tensor [3, H, W ]

    fft_src = torch.fft.fft2(src_img, dim=(-2, -1))
    fft_tgt = torch.fft.fft2(tgt_img, dim=(-2, -1))

    fft_src_shift = torch.fft.fftshift(fft_src, dim=(-2, -1))
    fft_tgt_shift = torch.fft.fftshift(fft_tgt, dim=(-2, -1))
    
    _, H, W = src_img.shape
    c_h, c_w = H // 2, W // 2
    bH = int(beta * H/2)
    bW = int(beta * W/2)

    mask = torch.zeros_like(fft_src.real, dtype=torch.bool)
    mask[:, c_h - bH:c_h + bH, c_w - bW:c_w + bW] = True

    amplitude_src = torch.abs(fft_src_shift)
    phase_src = torch.angle(fft_src_shift)
    amplitude_tgt = torch.abs(fft_tgt_shift)

    a = torch.fft.ifftshift(
        torch.where(mask, amplitude_tgt, amplitude_src)* torch.exp(1j * phase_src), 
        dim=(-2, -1)
    )

    res = torch.fft.ifft2(a, dim=(-2, -1)).real.clamp(0, 1)

    return res 

class ClassMixer():
    def __init__(self, n_classes:int, selected_portion:float, device):
        self.n_classes = n_classes
        self.device = device
        self.selected_portion = selected_portion
    
    def __call__(self, source_imgs, target_imgs, source_labels, target_pseudo_labels):
        B = source_labels.size(0)
        
        mask = torch.zeros_like(source_labels, dtype=torch.bool, requires_grad=False, device=self.device)

        for i in range(B):
            present_classes = torch.unique(source_labels[i])
            num_select = max(1, int(len(present_classes)*self.selected_portion))
            selected_classes = present_classes[torch.randperm(len(present_classes))[:num_select]]

            mask[i] = torch.isin(source_labels[i], selected_classes)
        
        mask_img = mask.unsqueeze(1).expand_as(source_imgs)
        
        mixed_img = torch.where(mask_img, source_imgs, target_imgs)
        mixed_label = torch.where(mask, source_labels, target_pseudo_labels)

        return mixed_img, mixed_label

def samples_with_class():
    samples_with_class = defaultdict(list)

    with open('sample_class_stats.json', 'r') as f:
        sample_class_stats = json.load(f)

    for entry in sample_class_stats:
        filename = entry['file'].split('_')[0]+'.png'
        for class_id, pixel_count in entry.items():
            if class_id == 'file':
                continue
            if int(pixel_count) > 0:
                samples_with_class[int(class_id)].append(filename)
    return samples_with_class

def get_rcs_class_probs(temperature):
    with open('sample_class_stats.json', 'r') as of:
        sample_class_stats = json.load(of)
    overall_class_stats = {}
    for s in sample_class_stats:
        s.pop('file')
        for c, n in s.items():
            c = int(c)
            if c not in overall_class_stats:
                overall_class_stats[c] = n
            else:
                overall_class_stats[c] += n
    overall_class_stats = {
        k: v
        for k, v in sorted(
            overall_class_stats.items(), key=lambda item: item[1])
    }
    freq = torch.tensor(list(overall_class_stats.values()))
    freq = freq / torch.sum(freq)
    freq = 1 - freq
    freq = torch.softmax(freq / temperature, dim=-1)

    return list(overall_class_stats.keys()), freq.numpy()

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
