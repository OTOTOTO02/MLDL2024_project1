import torch
import torch.nn as nn
import torch.nn.functional as F

import random

from bisenet.build_bisenet import BiSeNet

class SelfAttention(nn.Module):
    def __init__(self, in_channels, hidden_dim=64):
        super().__init__()
        self.query_conv = nn.Conv2d(in_channels, hidden_dim, 1)
        self.key_conv = nn.Conv2d(in_channels, hidden_dim, 1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.out_conv = nn.Conv2d(in_channels, 1, 1)

    def forward(self, x):
        """
        x: LR crop [B, C, H, W]
        Returns:
            spatial self attention feature map [B, C, H, W]
        """
        query = self.query_conv(x)          # [B, hidden, H, W]
        key = self.key_conv(x)              # [B, hidden, H, W]
        value = self.value_conv(x)          # [B, C, H, W]

        B, C, H, W = query.shape
        query = query.view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, hidden]
        key = key.view(B, -1, H * W)                       # [B, hidden, HW]
        value = value.view(B, -1, H * W).permute(0, 2, 1)  # [B, HW, C]

        attn = torch.bmm(query, key) / (key.shape[1] ** 0.5)  # [B, HW, HW]
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(attn, value)  # [B, HW, C]
        out = out.permute(0, 2, 1).view(B, -1, H, W)  # [B, C, H, W]

        return torch.sigmoid(self.out_conv(out)) # [B, 1, H, W]


class BiSeNetWithHRDA(BiSeNet):
    def __init__(self, num_classes, context_path, s, device):
        super().__init__(num_classes, context_path)
        self.device = device
        self.scale_factor = s
        self.attentionmodule = SelfAttention(in_channels=self.conv.out_channels, hidden_dim=32)

    def hrda_crop(self, img:torch.Tensor, label=None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[int,int,int,int]]:
        """
        Crop a high-res patch and return both full LR and cropped HR image + label if provided.
        """
        _, _, H, W = img.shape
        ch, cw = int(H*self.scale_factor), int(W*self.scale_factor) 

        lr_img = nn.functional.interpolate(img, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        lr_label = nn.functional.interpolate(label.unsqueeze(1).float(), scale_factor=self.scale_factor, mode='nearest').squeeze(1) if label is not None else None
          
        x1 = random.randint(0, W - cw)
        y1 = random.randint(0, H - ch)

        hr_img = img[:, :, y1:y1+ch, x1:x1+cw]
        hr_label = label[:, y1:y1+ch, x1:x1+cw] if label is not None else None

        return lr_img, lr_label, hr_img, hr_label, (x1, y1, cw, ch)

    def hrda_forward(self, lr_img, hr_img, crop_coords) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]: #, tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
        """
        Full HRDA forward: low-res + high-res + fusion
        """

        lr_feat, _, _ = self.encode(lr_img.detach())
        hr_feat, _, _ = self.encode(hr_img.detach())
        
        lr_out, hr_out = self.decode(torch.nn.functional.interpolate(lr_feat, scale_factor=2, mode='bilinear')), self.decode(hr_feat)

        attn = self.attentionmodule(lr_feat)

        fused = self.hrda_fuse(lr_out, hr_out, attn, crop_coords)
        
        return fused, (lr_out, hr_out) 
    
    def hrda_fuse(self, lr_out, hr_out, attn, crop_coords) -> torch.Tensor:
        """
        Overwrite or confidence-based fusion of HR patch into LR output.
        """
        x1, y1, cw, ch = crop_coords
        
        fused = lr_out.clone()

        masked_attn = torch.nn.functional.interpolate(attn, scale_factor=8./self.scale_factor, mode='bilinear')[:, :, y1:y1+ch, x1:x1+cw]
        fused[:, :, y1:y1+ch, x1:x1+cw] *= (1-masked_attn)
        fused[:, :, y1:y1+ch, x1:x1+cw] += hr_out*masked_attn

        return fused

    def hrda_eval(self, img):
        
        with torch.no_grad():
            lr_img, lr_label, hr_img, hr_label, crop_coords = self.hrda_crop(img.to(self.device), None)
            fused, (lr_out, hr_out) = self.hrda_forward(lr_img, hr_img, crop_coords)

        return fused, lr_out, hr_out, crop_coords
    
    def hrda_loss(self, criterion, fused_out, label, lr_out, lr_label, hr_out, hr_label):
        # Loss calculation:
        loss_hr_source = criterion(hr_out, hr_label)
        loss_lr_source = 0 
        loss_fused_src = criterion(fused_out, label)

        lambda_d = 0.1
        loss = (1-lambda_d)*loss_fused_src+lambda_d*(loss_lr_source+loss_hr_source)
    
        return loss