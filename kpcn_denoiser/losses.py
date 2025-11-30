import argparse
import random
import sys
from typing import Tuple

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.nn import functional as F
from torch.optim import Adam
from torchvision import transforms

from PIL import Image


def get_loss(name: str) -> _Loss:
    if name == "MSELoss":
        return nn.MSELoss()
    elif name == "L1Loss":
        return nn.L1Loss()
    elif name == "SobelL1":
        return SobelL1
    elif name == "CharbonnierLoss":
        return CharbonnierLoss()
    elif name == "EdgeCharbonnierLoss":
        return EdgeCharbonnierLoss()
    elif name == "AsymmetricCharbonnierLoss":
        return AsymmetricCharbonnierLoss()
    elif name == "KPCNDenoiserLoss":
        return KPCNDenoiserLoss()
    else:
        msg = f"Unknown loss name: {name}"
        raise ValueError(msg)


def sobel_grad(x: torch.Tensor) -> torch.Tensor:
    # simple finite-diff version
    w, h = x.size(-2), x.size(-1)
    l = F.pad(x, [1, 0, 0, 0])
    r = F.pad(x, [0, 1, 0, 0])
    u = F.pad(x, [0, 0, 1, 0])
    d = F.pad(x, [0, 0, 0, 1])
    return torch.abs((l - r)[..., 0:w, 0:h]) + torch.abs((u - d)[..., 0:w, 0:h])


def SobelL1(pred: torch.Tensor, target_batch: torch.Tensor):
    loss_l1 = F.l1_loss(pred, target_batch)  # your L1/L2
    loss_grad = F.l1_loss(sobel_grad(pred), sobel_grad(target_batch))
    return loss_l1 + 0.2 * loss_grad   # tune 0.1–0.5


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3, reduction: str = "mean") -> None:
        super().__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
        
def sobel_filter(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B, C, H, W]
    Returns gradients [B, C, H, W] (sqrt(gx^2 + gy^2)).
    """
    # Собель ядра 3x3
    kernel_x = torch.tensor(
        [[-1., 0., 1.],
         [-2., 0., 2.],
         [-1., 0., 1.]], device=x.device
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[-1., -2., -1.],
         [ 0.,  0.,  0.],
         [ 1.,  2.,  1.]], device=x.device
    ).view(1, 1, 3, 3)

    C = x.shape[1]
    kernel_x = kernel_x.repeat(C, 1, 1, 1)  # [C,1,3,3]
    kernel_y = kernel_y.repeat(C, 1, 1, 1)

    gx = F.conv2d(x, kernel_x, padding=1, groups=C)
    gy = F.conv2d(x, kernel_y, padding=1, groups=C)
    grad = torch.sqrt(gx * gx + gy * gy + 1e-6)
    return grad


class EdgeCharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3, reduction: str = "mean") -> None:
        super().__init__()
        self.charb = CharbonnierLoss(eps=eps, reduction=reduction)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_g = sobel_filter(pred)
        tgt_g = sobel_filter(target)
        return self.charb(pred_g, tgt_g)
    

class AsymmetricCharbonnierLoss(nn.Module):
    """
    Penalizes pred > target more than pred < target.
    """
    def __init__(
        self,
        eps: float = 1e-3,
        w_pos: float = 1.5,
        w_neg: float = 1.0,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        self.eps = eps
        self.w_pos = w_pos
        self.w_neg = w_neg
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target  # e
        pos = torch.clamp(diff, min=0.0)   # e_pos
        neg = torch.clamp(diff, max=0.0)   # e_neg

        loss_pos = torch.sqrt(pos * pos + self.eps * self.eps)
        loss_neg = torch.sqrt(neg * neg + self.eps * self.eps)

        loss = self.w_pos * loss_pos + self.w_neg * loss_neg
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
        

class KPCNDenoiserLoss(nn.Module):
    def __init__(
        self,
        eps: float = 1e-3,
        lambda_base: float = 1.0,
        lambda_edge: float = 0.3,
        lambda_asym: float = 0.3,
    ) -> None:
        super().__init__()
        self.lambda_base = lambda_base
        self.lambda_edge = lambda_edge
        self.lambda_asym = lambda_asym

        self.base_loss = CharbonnierLoss(eps=eps, reduction="mean")
        self.edge_loss = EdgeCharbonnierLoss(eps=eps, reduction="mean")
        self.asym_loss = AsymmetricCharbonnierLoss(
            eps=eps, w_pos=1.5, w_neg=1.0, reduction="mean"
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l_base = self.base_loss(pred, target)
        l_edge = self.edge_loss(pred, target)
        l_asym = self.asym_loss(pred, target)

        loss = (
            self.lambda_base * l_base
            + self.lambda_edge * l_edge
            + self.lambda_asym * l_asym
        )

        return loss