import sys
import os
import shutil
from typing import Tuple, Any
from abc import ABC, abstractmethod
from pathlib import Path

from kpcn_denoiser.dataset import (
    random_patch_pair,
)

from kpcn_denoiser.utils import (
    save_training_parameters,
    clean_dir
)

import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from torch.optim import Optimizer, Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    threshold: float = 1e-5,
) -> float:
    """One epoch train step."""
    model.train()
    num_steps = 0.0
    running_loss = 0.0

    total_batches = len(dataloader)
    print(f"[train_one_epoch] total batches: {total_batches}", flush=True)

    for batch_idx, (input_batch, target_batch) in enumerate(dataloader):
            input_batch = input_batch.to(device, non_blocking=True)
            target_batch = target_batch.to(device, non_blocking=True)

            if abs(input_batch - target_batch).mean() < threshold:
               continue

            pred = model(input_batch)
            loss = loss_fn(pred, target_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_steps += 1

            if (batch_idx + 1) % (max(total_batches/4, 1)) == 0 or (batch_idx + 1) == total_batches:
                print(f"[epoch batch] {batch_idx+1}/{total_batches}", flush=True)
    print(f"[train_one_epoc] trained batches: {int(num_steps)}/{total_batches}", flush=True)
    return running_loss/max(num_steps, 1)

@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate model."""
    model.eval()
    num_batches = 0.0
    running_loss = 0.0

    for input, target in dataloader:
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            pred = model(input)
            loss = loss_fn(pred, target)

            running_loss += loss.item()
            num_batches += 1

    avg = running_loss/max(num_batches, 1)
    print(f"[evaluate] Done. Avg loss: {avg:.6f}", flush=True)

    return avg


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    cv_loader: DataLoader | None,
    optimizer: Optimizer,
    loss_fn: nn.Module,
    device: torch.device,
    continue_training: bool,
    epochs: int,
    save_checkpoint_every: int,
    output_weights: str | Path,
    print_every_n_steps: int = 1,
    tensorboard_output: str | Path | None = None,
) -> Tuple[dict[str, Any], Path]: 
    """
    Training loop.

    Returns:
        tuple: A tuple containing:
            model_weights: The trained model weights.
            weights_out_file_name: Filename where the weights were saved.
            train_avg_loss: Average training loss.
            cross_validation_avg_loss: Average cross-validation loss.
    """
    output_weights = Path(output_weights)
    log_dir = output_weights.parent / "tensorboard_logs"
    if log_dir.exists():
        shutil.rmtree(log_dir)
        print(f"Tensorboard logs cleaned up.")

    if tensorboard_output:
        log_dir = Path(tensorboard_output)
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"Tensorboard log dir: {log_dir}")

    if output_weights.is_dir():
        weights_name = output_weights.parent.name
        checkpoint_output_dir = output_weights / "checkpoints"
        last_output_weights = output_weights / (weights_name + "_last.pt")
        output_best_weights = output_weights / (weights_name + "_best.pt")
    else:
        weights_name = output_weights.parent.parent.name
        checkpoint_output_dir = output_weights / "checkpoints"
        last_output_weights = output_weights.parent / (weights_name + "_last.pt")
        output_best_weights = output_weights.parent / (weights_name + "_best.pt")

    if checkpoint_output_dir.exists() and not continue_training:
        clean_dir(checkpoint_output_dir)
        print("Checkpoints cleaned up.")

    writer_train_loss = SummaryWriter(log_dir=str(log_dir / "train_loss"))
    writer_cv_loss = SummaryWriter(log_dir=str(log_dir / "cv_loss"))

    best_model_state = model.state_dict()
    best_cv_loss: float = float("inf")

    print("Starting training...", flush=True)
    print(f"Train batches per epoch: {len(train_loader)}", flush=True)
    print("PROGRESS: 0%", flush=True)

    start_epoch = model["epoch"] + 1 if continue_training else 1

    if start_epoch > epochs:
        print(f"Start epoch > total epoch. Skipping.")
        return (
        model.state_dict(),
        output_weights,
        )

    for epoch in range(start_epoch, epochs + 1):
        train_loss: float = train_one_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device
        )
        
        if cv_loader is not None:
            cv_loss = evaluate(model, cv_loader, loss_fn, device)
        else:
            cv_loss = float("nan")
        if cv_loader is not None and cv_loss < best_cv_loss:
            best_cv_loss = cv_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, output_best_weights)
        
        if tensorboard_output is not None:
            writer_train_loss.add_scalar("loss/train", train_loss, epoch)
            writer_cv_loss.add_scalar("loss/train",cv_loss, epoch)

        if epoch % save_checkpoint_every == 0:
            checkpoint_output_path = checkpoint_output_dir / f"{weights_name}_checkpoint.{epoch:04d}.json"
            checkpoint_data = {
                "train_loss": train_loss,
                "cv_loss": cv_loss,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict()
            }
            torch.save(checkpoint_data, checkpoint_output_path)
            print(f"Model checkpoint saved to {checkpoint_output_path}")

        torch.save(model.state_dict(), last_output_weights)
        print(f"Model last weights saved to {last_output_weights}")    

        if epoch % print_every_n_steps == 0:
            progress = int(epoch / epochs * 100)
            print(f"Epoch {epoch}/{epochs}, loss = {train_loss:.6f},",
                f"cv_loss = {cv_loss:.6f}," if cv_loader is not None else "", f"PROGRESS: {progress}%")
            sys.stdout.flush()

    print(f"\nModel was trained successfully!")
    print(f"Model best weights saved to {output_best_weights}")
    if output_weights.is_file():
        torch.save(model.state_dict(), output_weights)
        print(f"Model last weights also saved to {output_weights}")
    writer_train_loss.close()
    writer_cv_loss.close()
    
    return (
        model.state_dict(),
        output_weights,
    )

