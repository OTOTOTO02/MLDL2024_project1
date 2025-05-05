import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb

from utils import per_class_iou, fast_hist

# TODO: maybe last class (void) converted to n_classes-1 due to argmax
# TODO: waiting confirmation for logging 1 batch every X
def train(model:nn.Module, train_loader:DataLoader, criterion:nn.Module, optimizer:optim.Optimizer) -> tuple[float, float, np.ndarray]:
    global device
    global n_classes
    global ENABLE_PRINT
    global ENABLE_WANDB_LOG
    global train_step
    global log_per_epoch

    model.train()

    num_batch = len(train_loader)
    chunk_batch = num_batch//log_per_epoch+1

    num_sample = len(train_loader.dataset)
    seen_sample = 0

    train_loss = 0.0
    train_hist = np.zeros((n_classes,n_classes))

    for batch_idx, (inputs, _, targets) in enumerate(train_loader):
        batch_size = inputs.size(0)
        seen_sample += batch_size

        inputs, targets = inputs.to(device), targets.squeeze().to(device)

        outputs, cx1_sup, cx2_sup = model(inputs)

        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted = outputs.argmax(1)

        hist_batch = np.zeros((n_classes, n_classes))
        for i in range(len(inputs)):
            hist_batch += fast_hist(targets[i].detach().cpu().numpy(), predicted[i].detach().cpu().numpy(), n_classes)

        train_loss += loss.item() * batch_size
        train_hist += hist_batch

        # TODO: maybe batch_idx+1? same for validate
        if ((batch_idx+1) % chunk_batch) == 0:
            iou_batch = per_class_iou(hist_batch)
            if ENABLE_PRINT:
                    print(f'Training [{seen_sample}/{num_sample} ({100. * seen_sample / num_sample:.0f}%)]')
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

    train_loss = train_loss / seen_sample

    train_iou_class = per_class_iou(train_hist)
    train_mIou = train_iou_class[train_iou_class > 0].mean()

    return train_loss, train_mIou, train_hist


def validate(model:nn.Module, val_loader:DataLoader, criterion:nn.Module) -> tuple[float, float, np.ndarray]:
    global device
    global n_classes
    global ENABLE_PRINT
    global ENABLE_WANDB_LOG
    global val_step
    global log_per_epoch

    model.eval()

    num_batch = len(val_loader)
    chunk_batch = num_batch//log_per_epoch+1

    num_sample = len(val_loader.dataset)
    seen_sample = 0
    chunk_sample = 0

    val_loss = 0.0
    val_hist = np.zeros((n_classes,n_classes))

    chunk_loss = 0.0
    chunk_hist = np.zeros((n_classes,n_classes))

    with torch.no_grad():
        for batch_idx, (inputs, _, targets) in enumerate(val_loader):
            batch_size = inputs.size(0)

            inputs, targets = inputs.to(device), targets.squeeze().to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            predicted = outputs.argmax(1)

            hist_batch = np.zeros((n_classes, n_classes))
            for i in range(len(inputs)):
                hist_batch += fast_hist(targets[i].detach().cpu().numpy(), predicted[i].detach().cpu().numpy(), n_classes)

            chunk_sample += batch_size
            chunk_loss += loss.item() * batch_size
            chunk_hist += hist_batch

            if ((batch_idx+1) % chunk_batch) == 0:
                seen_sample += chunk_sample
                val_loss += chunk_loss
                val_hist += chunk_hist

                if ENABLE_PRINT:
                    iou_batch = per_class_iou(hist_batch)
                    print(f'Validation [{seen_sample}/{num_sample} ({100. * seen_sample / num_sample:.0f}%)]')
                    print(f'\tLoss: {loss.item():.6f}')
                    print(f"\tmIoU: {100.*iou_batch[iou_batch > 0].mean():.4f}")

                if ENABLE_WANDB_LOG:
                    iou_batch = per_class_iou(chunk_hist)
                    wandb.log({
                            "validate/step": val_step,
                            "validate/batch_loss": chunk_loss/chunk_sample,
                            "validate/batch_mIou": 100.*iou_batch[iou_batch > 0].mean()
                        },
                        commit=True,
                    )

                    val_step += 1

                chunk_sample = 0
                chunk_loss = 0.0
                chunk_hist = np.zeros((n_classes, n_classes))

        if chunk_sample > 0:
            seen_sample += chunk_sample
            val_loss += chunk_loss
            val_hist += chunk_hist

            if ENABLE_PRINT:
                iou_batch = per_class_iou(hist_batch)
                print(f'Validation [{seen_sample}/{num_sample} ({100. * seen_sample / num_sample:.0f}%)]')
                print(f'\tLoss: {loss.item():.6f}')
                print(f"\tmIoU: {100.*iou_batch[iou_batch > 0].mean():.4f}")

            if ENABLE_WANDB_LOG:
                iou_batch = per_class_iou(chunk_hist)
                wandb.log({
                        "validate/step": val_step,
                        "validate/batch_loss": chunk_loss/chunk_sample,
                        "validate/batch_mIou": 100.*iou_batch[iou_batch > 0].mean()
                    },
                    commit=True,
                )

                val_step += 1

    val_loss = val_loss / seen_sample

    val_iou_class = per_class_iou(val_hist)
    val_mIou = val_iou_class[val_iou_class > 0].mean()

    return val_loss, val_mIou, val_hist
