import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb

from utils import per_class_iou, fast_hist

# TODO: maybe last class (void) converted to n_classes-1 due to argmax
# TODO: do i only upload avg of last N batches or all, since different B produce different num_batch
def train(epoch:int, model:nn.Module, train_loader:DataLoader, criterion:nn.Module, optimizer:optim.Optimizer) -> tuple[float, float, np.ndarray]:
    global device
    global n_classes
    global ENABLE_PRINT
    global ENABLE_WANDB_LOG
    global train_step

    model.train()
    train_loss = 0.0
    train_hist = np.zeros((n_classes,n_classes))

    num_batch = len(train_loader)
    num_sample = len(train_loader.dataset)

    for batch_idx, (inputs, _, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.squeeze().to(device)

        outputs, cx1_sup, cx2_sup = model(inputs)

        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted = outputs.argmax(1)

        hist_batch = np.zeros((n_classes, n_classes))
        for i in range(len(inputs)):
            hist_batch += fast_hist(targets[i].cpu().numpy(), predicted[i].cpu().numpy(), n_classes)

        train_loss += loss.item()
        train_hist += hist_batch
        iou_batch = per_class_iou(hist_batch)

        if ENABLE_PRINT:
            if (batch_idx % (num_batch//10+1)) == 0:
                print(f'Train Epoch: {epoch} [{batch_idx * len(inputs)}/{num_sample} ({100. * batch_idx*len(inputs) / num_sample:.0f}%)]')
                print(f'\tLoss: {loss.item():.6f}')
                print(f"\tmIoU: {100.*iou_batch[iou_batch > 0].mean():.4f}")

        if ENABLE_WANDB_LOG:
            wandb.log({
                    "train/step": train_step,
                    "train/batch_loss": loss.item(),
                    "train/batch_mIou": 100.*iou_batch[iou_batch > 0].mean()
                },
                commit=True,
            )

        train_step += 1

    train_loss = train_loss / num_batch

    train_iou_class = per_class_iou(train_hist)
    train_mIou = train_iou_class[train_iou_class > 0].mean()

    return train_loss, train_mIou, train_hist


def validate(epoch:int, model:nn.Module, val_loader:DataLoader, criterion:nn.Module) -> tuple[float, float, np.ndarray]:
    global device
    global n_classes
    global ENABLE_PRINT
    global ENABLE_WANDB_LOG
    global val_step

    model.eval()
    val_loss = 0.0
    val_hist = np.zeros((n_classes,n_classes))

    num_batch = len(val_loader)
    num_sample = len(val_loader.dataset)

    with torch.no_grad():
        for batch_idx, (inputs, _, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.squeeze().to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            predicted = outputs.argmax(1)

            hist_batch = np.zeros((n_classes, n_classes))
            for i in range(len(inputs)):
                hist_batch += fast_hist(targets[i].cpu().numpy(), predicted[i].cpu().numpy(), n_classes)

            val_loss += loss.item()
            val_hist += hist_batch
            iou_batch = per_class_iou(hist_batch)

            if ENABLE_PRINT:
                if (batch_idx % (num_batch//10+1)) == 0:
                    print(f'Val Epoch: {epoch} [{batch_idx * len(inputs)}/{num_sample} ({100. * batch_idx*len(inputs) / num_sample:.0f}%)]')
                    print(f'\tLoss: {loss.item():.6f}')
                    print(f"\tmIoU: {100.*iou_batch[iou_batch > 0].mean():.4f}")

            if ENABLE_WANDB_LOG:
                wandb.log({
                        "validate/step": val_step,
                        "validate/batch_loss": loss.item(),
                        "validate/batch_mIou": 100.*iou_batch[iou_batch > 0].mean()
                    },
                    commit=True,
                )

            val_step += 1

    val_loss = val_loss / num_batch

    val_iou_class = per_class_iou(val_hist)
    val_mIou = val_iou_class[val_iou_class > 0].mean()

    return val_loss, val_mIou, val_hist