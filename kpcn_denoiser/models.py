import argparse
import random
import sys
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torchvision import transforms
from PIL import Image


def get_model(name: str, in_channel: int, out_channel: int, kernel_size: int) -> nn.Module:
    if name == "UNet6Residual":
        return  UNet6Residual(channels=in_channel, base=out_channel)
    elif name == "UNetResidual":
        return UNetResidual(channels=in_channel, base=out_channel)
    elif name == "UNetKPCMMedium":
        return UNetKPCMMedium(in_channels=in_channel, base=out_channel, kernel_size=kernel_size)
    elif name == "UNetKPCMLarge":
        return UNetKPCMLarge(in_channels=in_channel, base=out_channel, kernel_size=kernel_size)
    elif name == "HiqUnetKPCNMedium":
        return HiqUnetKPCNMedium(in_channels=in_channel)
    elif name == "HiqUnetKPCNLarge":
        return HiqUnetKPCNLarge(in_channels=in_channel)
    else:
        msg = f"Unknown model name: {name}"
        raise ValueError(msg)


def get_model_code_name(model: str, in_channels: int, out_channels) -> str:
    if model == "UNet6Residual":
        return f"unet6res_{in_channels}ch_{out_channels}base"
    elif model == "UNetResidual":
        return f"unetres_{in_channels}ch_{out_channels}base"
    elif model == "UNetKPCMMedium":
        return f"unetkpcmmed_{in_channels}ch_{out_channels}base"
    else:
        msg = f"Unknown model name: {model}"
        raise ValueError(msg)


class UNet6Residual(nn.Module):
    """Predict NOISE then subtract"""

    def __init__(self, channels=3, base=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.middle = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(base, base, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, channels, 3, padding=1),
        )

    def forward(self, noisy):
        x = self.encoder(noisy)
        x = self.middle(x)
        noise_pred = self.decoder(x)
        clean_pred = noisy - noise_pred
        return clean_pred


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetResidual(nn.Module):
    """Small U-Net that predicts noise and subtracts it."""

    def __init__(self, channels: int = 3, base: int = 64):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(channels, base)          # 64x64
        self.pool1 = nn.MaxPool2d(2)                   # 32x32

        self.enc2 = ConvBlock(base, base * 2)          # 32x32
        self.pool2 = nn.MaxPool2d(2)                   # 16x16

        # Bottleneck
        self.bottleneck = ConvBlock(base * 2, base * 4)

        # Decoder
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)  # 32x32
        self.dec2 = ConvBlock(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)      # 64x64
        self.dec1 = ConvBlock(base * 2, base)

        # Output: predict noise
        self.out_conv = nn.Conv2d(base, channels, 3, padding=1)

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.enc1(noisy)
        p1 = self.pool1(x1)

        x2 = self.enc2(p1)
        p2 = self.pool2(x2)

        # Bottleneck
        b = self.bottleneck(p2)

        # Decoder
        u2 = self.up2(b)
        u2 = torch.cat([u2, x2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = torch.cat([u1, x1], dim=1)
        d1 = self.dec1(u1)

        noise_pred = self.out_conv(d1)
        clean_pred = noisy - noise_pred
        return clean_pred

import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetKPCMMedium(nn.Module):
    def __init__(self, in_channels=3, base=64, kernel_size=21) -> None:
        super().__init__()
        self.k = kernel_size
        self.k2 = self.k * self.k  # k2 = kernel_size * kernel_size

        self.e1 = self._conv_block(in_channels, base)
        self.pool1 = nn.MaxPool2d(2)

        self.e2 = self._conv_block(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)

        self.bn = self._conv_block(base * 2, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(base * 2, base)

        self.out_kern = nn.Conv2d(base, out_channels=self.k2, kernel_size=1)

    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _match_size(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Подогнать spatial-размеры a и b к общему (минимальному) H×W."""
        _, _, ha, wa = a.shape
        _, _, hb, wb = b.shape
        h = min(ha, hb)
        w = min(wa, wb)
        if ha != h or wa != w:
            a = a[:, :, :h, :w]
        if hb != h or wb != w:
            b = b[:, :, :h, :w]
        return a, b

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        B, C, H_in, W_in = noisy.shape

        # Encoder
        x1 = self.e1(noisy)       # [B, base, H,   W]
        p1 = self.pool1(x1)       # [B, base, H/2, W/2]

        x2 = self.e2(p1)          # [B, 2*base, H/2, W/2]
        p2 = self.pool2(x2)       # [B, 2*base, H/4, W/4]

        # Bottleneck
        b = self.bn(p2)           # [B, 4*base, H/4, W/4]

        # Decoder: level 2
        u2 = self.up2(b)          # [B, 2*base, ~H/2, ~W/2]
        u2, x2 = self._match_size(u2, x2)
        u2 = torch.cat([u2, x2], dim=1)    # [B, 4*base, H/2, W/2]
        d2 = self.dec2(u2)                 # [B, 2*base, H/2, W/2]

        # Decoder: level 1
        u1 = self.up1(d2)          # [B, base, ~H, ~W]
        u1, x1 = self._match_size(u1, x1)
        u1 = torch.cat([u1, x1], dim=1)    # [B, 2*base, H_out, W_out]
        d1 = self.dec1(u1)                 # [B, base, H_out, W_out]

        # Per-pixel kernel weights (определяем истинные H_out, W_out здесь)
        raw_weights = self.out_kern(d1)               # [B, k2, H_out, W_out]
        B_w, k2, H, W = raw_weights.shape
        assert B_w == B
        assert k2 == self.k2

        # normalisation по сумме, без softmax
        # kernel_sum = raw_weights.sum(dim=1, keepdim=True)  # [B, 1, H, W]
        # kernel_sum = kernel_sum.clamp_min(1e-3)
        # weights = raw_weights / kernel_sum                 # [B, k2, H, W]
        #weights = torch.softmax(raw_weights, dim=1)
        weights = raw_weights / raw_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        # Flatten weights & add channel dim
        weights_flat = weights.view(B, self.k2, H * W)     # [B, k2, H*W]
        weights_exp  = weights_flat.unsqueeze(1)           # [B, 1, k2, H*W]

        # Extract patches from noisy
        patches = F.unfold(
            noisy, kernel_size=self.k, padding=self.k // 2
        )                                                  # [B, C*k2, H_in*W_in]
        patches_c = patches.view(B, C, self.k2, -1)        # [B, C, k2, H_in*W_in]

        # подрезаем патчи до того же количества позиций H*W,
        # что и у весов (верхний левый блок по сути)
        patches_c = patches_c[..., : H * W]                # [B, C, k2, H*W]

        # Apply kernel per channel
        out_flat = (weights_exp * patches_c).sum(dim=2)    # [B, C, H*W]
        out = out_flat.view(B, C, H, W)                    # [B, C, H, W]

        # residual в той же области
        noisy_cropped = noisy[:, :, :H, :W]                # [B, C, H, W]
        filtered = out
        clean_pred = noisy_cropped + (filtered - noisy_cropped)
        return clean_pred


import torch
import torch.nn as nn
import torch.nn.functional as F


class UNetKPCMLarge(nn.Module):
    def __init__(self, in_channels: int = 3, base: int = 128, kernel_size: int = 35) -> None:
        super().__init__()
        self.k = kernel_size
        self.k2 = self.k * self.k  # k^2 = kernel_size * kernel_size

        # ---------- ENCODER ----------
        self.e1 = self._conv_block(in_channels, base)          # -> base
        self.pool1 = nn.MaxPool2d(2)

        self.e2 = self._conv_block(base, base * 2)             # -> 2*base
        self.pool2 = nn.MaxPool2d(2)

        self.e3 = self._conv_block(base * 2, base * 4)         # -> 4*base
        self.pool3 = nn.MaxPool2d(2)

        self.e4 = self._conv_block(base * 4, base * 8)         # -> 8*base
        self.pool4 = nn.MaxPool2d(2)

        # ---------- BOTTLENECK ----------
        self.bn = self._conv_block(base * 8, base * 16)        # -> 16*base

        # ---------- DECODER ----------
        # level 4 -> 3
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(base * 8 + base * 8, base * 8)   # cat(8b, 8b) -> 8b

        # level 3 -> 2
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(base * 4 + base * 4, base * 4)   # cat(4b, 4b) -> 4b

        # level 2 -> 1
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(base * 2 + base * 2, base * 2)   # cat(2b, 2b) -> 2b

        # level 1 -> out
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(base + base, base)               # cat(b, b) -> b

        # ---------- KERNEL HEAD ----------
        self.out_kern = nn.Conv2d(base, out_channels=self.k2, kernel_size=1)

    # --- helpers ---

    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Module:
        """Two Conv+ReLU layers with 3x3 kernels and same padding."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _match_size(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Crop tensors `a` and `b` so that their spatial dimensions match.
        We crop both to the minimum common height/width (top-left aligned).
        """
        _, _, ha, wa = a.shape
        _, _, hb, wb = b.shape
        h = min(ha, hb)
        w = min(wa, wb)
        if ha != h or wa != w:
            a = a[:, :, :h, :w]
        if hb != h or wb != w:
            b = b[:, :, :h, :w]
        return a, b

    # --- forward ---

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Input:
            noisy: [B, C, H_in, W_in] noisy image

        Output:
            clean_pred: [B, C, H_out, W_out] denoised image
        """
        B, C, H_in, W_in = noisy.shape

        # ----- ENCODER -----
        x1 = self.e1(noisy)      # [B, base, H,   W]
        p1 = self.pool1(x1)      # [B, base, H/2, W/2]

        x2 = self.e2(p1)         # [B, 2b, H/2, W/2]
        p2 = self.pool2(x2)      # [B, 2b, H/4, W/4]

        x3 = self.e3(p2)         # [B, 4b, H/4, W/4]
        p3 = self.pool3(x3)      # [B, 4b, H/8, W/8]

        x4 = self.e4(p3)         # [B, 8b, H/8, W/8]
        p4 = self.pool4(x4)      # [B, 8b, H/16, W/16]

        # ----- BOTTLENECK -----
        b = self.bn(p4)          # [B, 16b, H/16, W/16]

        # ----- DECODER -----
        # level 4
        u4 = self.up4(b)         # [B, 8b, ~H/8, ~W/8]
        u4, x4 = self._match_size(u4, x4)
        d4_in = torch.cat([u4, x4], dim=1)   # [B, 16b, H/8, W/8]
        d4 = self.dec4(d4_in)                # [B, 8b,  H/8, W/8]

        # level 3
        u3 = self.up3(d4)        # [B, 4b, ~H/4, ~W/4]
        u3, x3 = self._match_size(u3, x3)
        d3_in = torch.cat([u3, x3], dim=1)   # [B, 8b, H/4, W/4]
        d3 = self.dec3(d3_in)                # [B, 4b, H/4, W/4]

        # level 2
        u2 = self.up2(d3)        # [B, 2b, ~H/2, ~W/2]
        u2, x2 = self._match_size(u2, x2)
        d2_in = torch.cat([u2, x2], dim=1)   # [B, 4b, H/2, W/2]
        d2 = self.dec2(d2_in)                # [B, 2b, H/2, W/2]

        # level 1
        u1 = self.up1(d2)        # [B, b, ~H, ~W]
        u1, x1 = self._match_size(u1, x1)
        d1_in = torch.cat([u1, x1], dim=1)   # [B, 2b, H_out, W_out]
        d1 = self.dec1(d1_in)                # [B, b, H_out, W_out]

        # true spatial size at UNet output (may be <= input size)
        _, _, H, W = d1.shape

        # ----- KERNEL PREDICTION -----
        raw_weights = self.out_kern(d1)                   # [B, k2, H, W]
        B_w, k2, H_w, W_w = raw_weights.shape
        assert B_w == B and k2 == self.k2 and H_w == H and W_w == W

        # normalize kernel weights per-pixel by sum
        # kernel_sum = raw_weights.sum(dim=1, keepdim=True)  # [B, 1, H, W]
        # kernel_sum = kernel_sum.clamp_min(1e-3)
        # weights = raw_weights / kernel_sum                 # [B, k2, H, W]
        weights = raw_weights / raw_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        # flatten weights and expand channel dimension
        weights_flat = weights.view(B, self.k2, H * W)     # [B, k2, H*W]
        weights_exp  = weights_flat.unsqueeze(1)           # [B, 1, k2, H*W]

        # crop noisy to match UNet output spatial size
        noisy_cropped = noisy[:, :, :H, :W]                # [B, C, H, W]

        # extract patches from noisy_cropped
        patches = F.unfold(
            noisy_cropped, kernel_size=self.k, padding=self.k // 2
        )                                                  # [B, C*k2, H*W]
        patches_c = patches.view(B, C, self.k2, H * W)     # [B, C, k2, H*W]

        # apply per-pixel kernels over patches
        out_flat = (weights_exp * patches_c).sum(dim=2)    # [B, C, H*W]
        out = out_flat.view(B, C, H, W)                    # [B, C, H, W]

        # residual prediction
        delta = out - noisy_cropped
        clean_pred = noisy_cropped + delta
        return clean_pred



class HiqUnetKPCNMedium(nn.Module):
    """
    3-уровневый U-Net kernel prediction:
    - рассчитан под kernel_size ~ 11–21
    - base по умолчанию 96 для более “богатых” фичей
    """

    def __init__(self, in_channels: int = 3, base: int = 96, kernel_size: int = 21) -> None:
        super().__init__()
        self.k = kernel_size
        self.k2 = self.k * self.k  # k^2

        # ---------- ENCODER ----------
        self.e1 = self._conv_block(in_channels, base)        # -> base
        self.pool1 = nn.MaxPool2d(2)

        self.e2 = self._conv_block(base, base * 2)           # -> 2b
        self.pool2 = nn.MaxPool2d(2)

        self.e3 = self._conv_block(base * 2, base * 4)       # -> 4b
        self.pool3 = nn.MaxPool2d(2)

        # ---------- BOTTLENECK ----------
        self.bn = self._conv_block(base * 4, base * 4)       # остаёмся на 4b

        # ---------- DECODER ----------
        # level 3 -> 2
        self.up3 = nn.ConvTranspose2d(base * 4, base * 4, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(base * 4 + base * 4, base * 4)   # cat(4b, 4b) -> 4b

        # level 2 -> 1
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(base * 2 + base * 2, base * 2)   # cat(2b, 2b) -> 2b

        # level 1 -> out
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(base + base, base)               # cat(b, b) -> b

        # ---------- KERNEL HEAD ----------
        self.out_kern = nn.Conv2d(base, out_channels=self.k2, kernel_size=1)

    # --- helpers ---

    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _match_size(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Подрезаем a и b до общих (минимальных) H×W.
        Это убирает off-by-one артефакты от пулов/апсемплов.
        """
        _, _, ha, wa = a.shape
        _, _, hb, wb = b.shape
        h = min(ha, hb)
        w = min(wa, wb)
        if ha != h or wa != w:
            a = a[:, :, :h, :w]
        if hb != h or wb != w:
            b = b[:, :, :h, :w]
        return a, b

    # --- forward ---

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        B, C, H_in, W_in = noisy.shape

        # ----- ENCODER -----
        x1 = self.e1(noisy)   # [B, b, H,   W]
        p1 = self.pool1(x1)   # [B, b, H/2, W/2]

        x2 = self.e2(p1)      # [B, 2b, H/2, W/2]
        p2 = self.pool2(x2)   # [B, 2b, H/4, W/4]

        x3 = self.e3(p2)      # [B, 4b, H/4, W/4]
        p3 = self.pool3(x3)   # [B, 4b, H/8, W/8]

        # ----- BOTTLENECK -----
        b = self.bn(p3)       # [B, 4b, H/8, W/8]

        # ----- DECODER -----
        # level 3
        u3 = self.up3(b)          # [B, 4b, ~H/4, ~W/4]
        u3, x3 = self._match_size(u3, x3)
        d3_in = torch.cat([u3, x3], dim=1)   # [B, 8b, H/4, W/4]
        d3 = self.dec3(d3_in)                # [B, 4b, H/4, W/4]

        # level 2
        u2 = self.up2(d3)        # [B, 2b, ~H/2, ~W/2]
        u2, x2 = self._match_size(u2, x2)
        d2_in = torch.cat([u2, x2], dim=1)   # [B, 4b, H/2, W/2]
        d2 = self.dec2(d2_in)                # [B, 2b, H/2, W/2]

        # level 1
        u1 = self.up1(d2)        # [B, b, ~H, ~W]
        u1, x1 = self._match_size(u1, x1)
        d1_in = torch.cat([u1, x1], dim=1)   # [B, 2b, H_out, W_out]
        d1 = self.dec1(d1_in)                # [B, b, H_out, W_out]

        _, _, H, W = d1.shape

        # ----- KERNEL PREDICTION -----
        raw_weights = self.out_kern(d1)           # [B, k2, H, W]

        # softplus + нормализация по сумме
        weights = F.softplus(raw_weights)         # [B, k2, H, W]
        kernel_sum = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        weights = weights / kernel_sum            # [B, k2, H, W]

        # разворачиваем ядра
        weights_flat = weights.view(B, self.k2, H * W)   # [B, k2, H*W]
        weights_exp = weights_flat.unsqueeze(1)          # [B, 1, k2, H*W]

        # подрезаем noisy под выход U-Net
        noisy_cropped = noisy[:, :, :H, :W]              # [B, C, H, W]

        # вытаскиваем патчи
        patches = F.unfold(
            noisy_cropped, kernel_size=self.k, padding=self.k // 2
        )                                                # [B, C*k2, H*W]
        patches_c = patches.view(B, C, self.k2, H * W)   # [B, C, k2, H*W]

        # применяем ядра
        out_flat = (weights_exp * patches_c).sum(dim=2)  # [B, C, H*W]
        out = out_flat.view(B, C, H, W)                  # [B, C, H, W]

        # residual формулировка (эквивалентно out, но явно)
        delta = out - noisy_cropped
        clean_pred = noisy_cropped + delta
        return clean_pred


class HiqUnetKPCNLarge(nn.Module):
    """
    4-уровневый U-Net kernel prediction:
    - рассчитан под kernel_size ~ 21–35
    - base по умолчанию 128 (production-style)
    """

    def __init__(self, in_channels: int = 3, base: int = 128, kernel_size: int = 35) -> None:
        super().__init__()
        self.k = kernel_size
        self.k2 = self.k * self.k

        # ---------- ENCODER ----------
        self.e1 = self._conv_block(in_channels, base)        # -> b
        self.pool1 = nn.MaxPool2d(2)

        self.e2 = self._conv_block(base, base * 2)           # -> 2b
        self.pool2 = nn.MaxPool2d(2)

        self.e3 = self._conv_block(base * 2, base * 4)       # -> 4b
        self.pool3 = nn.MaxPool2d(2)

        self.e4 = self._conv_block(base * 4, base * 8)       # -> 8b
        self.pool4 = nn.MaxPool2d(2)

        # ---------- BOTTLENECK ----------
        self.bn = self._conv_block(base * 8, base * 16)      # -> 16b

        # ---------- DECODER ----------
        # 4 -> 3
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(base * 8 + base * 8, base * 8)   # cat(8b, 8b) -> 8b

        # 3 -> 2
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(base * 4 + base * 4, base * 4)   # cat(4b, 4b) -> 4b

        # 2 -> 1
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(base * 2 + base * 2, base * 2)   # cat(2b, 2b) -> 2b

        # 1 -> out
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(base + base, base)               # cat(b, b) -> b

        # ---------- KERNEL HEAD ----------
        self.out_kern = nn.Conv2d(base, out_channels=self.k2, kernel_size=1)

    def _conv_block(self, in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _match_size(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, _, ha, wa = a.shape
        _, _, hb, wb = b.shape
        h = min(ha, hb)
        w = min(wa, wb)
        if ha != h or wa != w:
            a = a[:, :, :h, :w]
        if hb != h or wb != w:
            b = b[:, :, :h, :w]
        return a, b

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        B, C, H_in, W_in = noisy.shape

        # ----- ENCODER -----
        x1 = self.e1(noisy)     # [B, b, H,   W]
        p1 = self.pool1(x1)     # [B, b, H/2, W/2]

        x2 = self.e2(p1)        # [B, 2b, H/2, W/2]
        p2 = self.pool2(x2)     # [B, 2b, H/4, W/4]

        x3 = self.e3(p2)        # [B, 4b, H/4, W/4]
        p3 = self.pool3(x3)     # [B, 4b, H/8, W/8]

        x4 = self.e4(p3)        # [B, 8b, H/8, W/8]
        p4 = self.pool4(x4)     # [B, 8b, H/16, W/16]

        # ----- BOTTLENECK -----
        b = self.bn(p4)         # [B, 16b, H/16, W/16]

        # ----- DECODER -----
        # level 4
        u4 = self.up4(b)        # [B, 8b, ~H/8, ~W/8]
        u4, x4 = self._match_size(u4, x4)
        d4_in = torch.cat([u4, x4], dim=1)   # [B, 16b, H/8, W/8]
        d4 = self.dec4(d4_in)                # [B, 8b, H/8, W/8]

        # level 3
        u3 = self.up3(d4)       # [B, 4b, ~H/4, ~W/4]
        u3, x3 = self._match_size(u3, x3)
        d3_in = torch.cat([u3, x3], dim=1)   # [B, 8b, H/4, W/4]
        d3 = self.dec3(d3_in)                # [B, 4b, H/4, W/4]

        # level 2
        u2 = self.up2(d3)       # [B, 2b, ~H/2, ~W/2]
        u2, x2 = self._match_size(u2, x2)
        d2_in = torch.cat([u2, x2], dim=1)   # [B, 4b, H/2, W/2]
        d2 = self.dec2(d2_in)                # [B, 2b, H/2, W/2]

        # level 1
        u1 = self.up1(d2)       # [B, b, ~H, ~W]
        u1, x1 = self._match_size(u1, x1)
        d1_in = torch.cat([u1, x1], dim=1)   # [B, 2b, H_out, W_out]
        d1 = self.dec1(d1_in)                # [B, b, H_out, W_out]

        _, _, H, W = d1.shape

        # ----- KERNEL PREDICTION -----
        raw_weights = self.out_kern(d1)             # [B, k2, H, W]

        weights = F.softplus(raw_weights)
        kernel_sum = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        weights = weights / kernel_sum              # [B, k2, H, W]

        weights_flat = weights.view(B, self.k2, H * W)   # [B, k2, H*W]
        weights_exp = weights_flat.unsqueeze(1)          # [B, 1, k2, H*W]

        noisy_cropped = noisy[:, :, :H, :W]              # [B, C, H, W]

        patches = F.unfold(
            noisy_cropped, kernel_size=self.k, padding=self.k // 2
        )                                                # [B, C*k2, H*W]
        patches_c = patches.view(B, C, self.k2, H * W)   # [B, C, k2, H*W]

        out_flat = (weights_exp * patches_c).sum(dim=2)  # [B, C, H*W]
        out = out_flat.view(B, C, H, W)

        delta = out - noisy_cropped
        clean_pred = noisy_cropped + delta
        return clean_pred