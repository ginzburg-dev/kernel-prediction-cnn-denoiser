import argparse
import random
import sys
import re
import shutil
from typing import Tuple

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image

from kpcn_denoiser.models import UNet6Residual, UNetResidual
import json

def get_frame_number(path: Path) -> str | None:
    match = re.search(r'\.(\d+)', path.stem)
    if match and match.group(1).isdigit():
        return match.group(1)
    else:
        return None

def get_clean_basename(path: Path) -> str:
    return re.sub(r"\.\d+.*$", "", path.stem)

def get_model_name_and_channels(model: nn.Module):

    # Get base architecture name
    base_name = model.__class__.__name__

    # Find the first Conv2d layer in the model
    in_channels = None
    out_channels = None
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            in_channels = layer.in_channels
            base_channels = layer.out_channels
            break

    if in_channels is None or out_channels is None:
        raise ValueError("No Conv2d layer found in the model.")

    return base_name, in_channels, out_channels

def save_training_parameters(
        path: Path,
        model_name: str,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        epochs: int,
        batch_size: int,
        patches_per_image: int,
        patch_size: int,
        n_first_samples: int,
        n_first_frames: int,
        lr: float,
) -> None:
    """Save JSON training parameters."""
    # name, in_channels, out_channels = get_model_name_and_channels(model)
    training_parameters = {
        "model_name": model_name,
        "out_channels": out_channels,
        "in_channels": in_channels,
        "kernel_size": kernel_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "patches_per_image": patches_per_image,
        "patch_size": patch_size,
        "n_first_samples": n_first_samples,
        "n_first_frames": n_first_frames,
        "lr": lr,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(training_parameters, f, indent=4, ensure_ascii=False)


@torch.no_grad()
def apply_model_to_tensor_tiled(
    model: torch.nn.Module,
    noisy: torch.Tensor,
    device: torch.device,
    kernel_size: int = 35,
    core_step: int = 256,
    margin: int = 64,
) -> torch.Tensor:
    """
    Корректный tiled-inference без смещения:
    - картинку режем на "ядра" размером core_step x core_step;
    - вокруг каждого ядра берём расширенный тайл с полем margin;
    - сеть прогоняем на расширенном тайле;
    - в общий результат кладём только центральную область ядра.

    Это гарантирует:
    - никаких OOM (тайлы относительно маленькие);
    - никаких смещений (каждый пиксель пишет в свои координаты);
    - мягкие стыки (ядра не зависят от границ тайлов).
    """

    model.eval()

    # [1, C, H, W]
    if noisy.dim() == 3:
        noisy = noisy.unsqueeze(0)

    noisy = noisy.to(device, non_blocking=True)
    B, C, H, W = noisy.shape

    # Итоговый результат на CPU в float32
    out = torch.zeros((B, C, H, W), dtype=noisy.dtype)

    # Проходим по "ядрам" (core-область)
    for y0_core in range(0, H, core_step):
        y1_core = min(y0_core + core_step, H)

        for x0_core in range(0, W, core_step):
            x1_core = min(x0_core + core_step, W)

            # Расширенный тайл с полем margin вокруг ядра
            y0_ext = max(0, y0_core - margin)
            x0_ext = max(0, x0_core - margin)
            y1_ext = min(H, y1_core + margin)
            x1_ext = min(W, x1_core + margin)

            # Вырезаем расширенный тайл
            tile_in = noisy[:, :, y0_ext:y1_ext, x0_ext:x1_ext]

            # Прогоняем через модель
            tile_out = model(tile_in)    # [B, C, H_t, W_t]
            _, _, H_t, W_t = tile_out.shape

            # ВАЖНО: в твоём UNet мы внутри могли чуть обрезать низ/правый край
            # (match_size + cropping). Поэтому реальный "валидный" диапазон
            # по глобальным координатам ограничен:
            valid_y1_global = min(y1_core, y0_ext + H_t)
            valid_x1_global = min(x1_core, x0_ext + W_t)
            valid_y0_global = y0_core
            valid_x0_global = x0_core

            if valid_y1_global <= valid_y0_global or valid_x1_global <= valid_x0_global:
                continue

            # Координаты в tile_out
            ty0 = valid_y0_global - y0_ext
            tx0 = valid_x0_global - x0_ext
            ty1 = ty0 + (valid_y1_global - valid_y0_global)
            tx1 = tx0 + (valid_x1_global - valid_x0_global)

            # Нарезаем кусочек из tile_out
            inner = tile_out[:, :, ty0:ty1, tx0:tx1].detach().cpu()

            # И пишем его на те же глобальные координаты
            out[
                :,
                :,
                valid_y0_global:valid_y1_global,
                valid_x0_global:valid_x1_global,
            ] = inner

    return out


def get_lats_checkpoint(path: Path | str) -> Path:
    path = Path(path)
    checkpoints = sorted(path.glob("*checkpoint*.json"))
    return checkpoints[-1]

def clean_dir(path: Path):
    for item in path.iterdir():
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()