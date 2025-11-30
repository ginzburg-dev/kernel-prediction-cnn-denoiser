import os
import sys
import af # pyright: ignore[reportMissingImports]
import argparse
import subprocess

from pathlib import Path

from kpcn_denoiser.config import KPCNConfig, ExperimentConfig
from kpcn_denoiser.utils import get_lats_checkpoint
from tools.submit_train_job import Command, CommandBlock, Job, submit_af_job
from tools.run_tensorboard import launch_tensorboard


config = ExperimentConfig(name=Path(__file__).stem)
global_config = KPCNConfig()


#MODEL = "UNetKPCMLarge"
MODEL = "UNetKPCMMedium"
SHOT = "TGB1004140"
IN_CHANNELS = 3
OUT_CHANNELS = 64
KERNEL_SIZE = 35
LOSS_SOBEL = "SobelL1"
LOSS_L1 = "L1Loss"
LOSS = LOSS_L1
CONTINUE_TRAINING = False
WEIGHT = f"output\experiments\{config.name}\weights\{config.name}_last.pt"
JOB_NAME = f"torch-{config.name}-job"


train_command_raw = [
    sys.executable,
    str(global_config.trainer_app_path),
    "--mode", "train",
    "--model", str(MODEL),
    "--model-in-channels", str(IN_CHANNELS),
    "--model-out-channels", str(OUT_CHANNELS),
    "--kernel-size", str(KERNEL_SIZE),
    "--loss", str(LOSS),
    "--input", str(config.training_dataset),
    "--target-shot", str(SHOT),
    "--checkpoints-dir", str(config.chekpoints_dir),
    "--output", str(config.output_dir),
    "--weights-out", str(config.weights_out_path),
    "--weights-in", str(config.weights_out_path),
    "--epochs", 50,
    "--continue-training" if CONTINUE_TRAINING else "",
    "--batch-size", 8,
    "--patches-per-image", 200,
    "--patch-size", 128,
    "--n-first-samples", 50,
    "--n-first-frames", 1,
    "--save-checkpoint-every", 10,
    "--print-every-n-step", 1,
    "--lr", 1e-4,
    "--tensorboard-output", str(config.tensorboard_log_dir),
]
train_command = str(config.kpcn_wrapper) + " " + " ".join(map(str, train_command_raw))


spatial_validation_command_raw = [
    sys.executable,
    str(global_config.trainer_app_path),
    "--mode", "apply",
    "--model", str(MODEL),
    "--model-in-channels", str(IN_CHANNELS),
    "--model-out-channels", str(OUT_CHANNELS),
    "--kernel-size", str(KERNEL_SIZE),
    "--input", str(config.validation_dataset),
    "--output", str(config.output_spatial_dir),
    "--weights-in", str(WEIGHT),
]
spatial_validation_command = str(config.kpcn_wrapper) + " " + " ".join(map(str, spatial_validation_command_raw))


seqence_over_epoch_command_raw = [
        sys.executable,
        str(global_config.trainer_app_path),
        "--mode", "apply_epoch_sequence",
        "--model", str(MODEL),
        "--model-in-channels", str(IN_CHANNELS),
        "--model-out-channels", str(OUT_CHANNELS),
        "--kernel-size", str(KERNEL_SIZE),
        "--input", str(config.validation_dataset),
        "--output", str(config.output_epoch_sequence_dir),
        "--weights-in", str(config.chekpoints_dir),
]
seqence_over_epoch_command = str(config.kpcn_wrapper) + " " + " ".join(map(str, seqence_over_epoch_command_raw))


temporal_esquence_dataset_path001 = config.training_dataset / "TGB1004140" / "chars" / "rgba" / "noisy"
temporal_esquence_out_path001 = config.output_temporal_sequence_dir / 'TGB1004140_char_human_closeup'
temporal_esquence_out_path001.mkdir(parents=True, exist_ok=True)
temporal_sequence_command_raw = [
        sys.executable,
        str(global_config.trainer_app_path),
        "--model", str(MODEL),
        "--mode", "apply",
        "--model-in-channels", str(IN_CHANNELS),
        "--model-out-channels", str(OUT_CHANNELS),
        "--kernel-size", str(KERNEL_SIZE),
        "--input", str(temporal_esquence_dataset_path001),
        "--output", str(temporal_esquence_out_path001),
        "--weights-in", str(WEIGHT),
]
temporal_sequence_command = str(config.kpcn_wrapper) + " " + " ".join(map(str, temporal_sequence_command_raw))


def submit_experiment():
    job = Job(
        name=JOB_NAME,
        command_blocks=[
            CommandBlock(
                title="Training Block",
                commands=[
                    Command(
                        title="Train Model",
                        command=train_command
                    )
                ]
            ),
            CommandBlock(
                title="Test Block",
                commands=[
                    Command(
                        title="Spatial Validation Set",
                        command=spatial_validation_command
                    ),
                    Command(
                        title="Sequence over Epochs",
                        command=seqence_over_epoch_command
                    ),
                    Command(
                        title="Temporal Sequence TGB1004140 char_human_closeup",
                        command=temporal_sequence_command
                    )
                ]
            )
        ]
    )
    submit_af_job(job)


def submit_validation():
    job = Job(
        name=JOB_NAME,
        command_blocks=[
            CommandBlock(
                title="Test Block",
                commands=[
                    Command(
                        title="Spatial Validation Set",
                        command=spatial_validation_command
                    ),
                    Command(
                        title="Sequence over Epochs",
                        command=seqence_over_epoch_command
                    ),
                    Command(
                        title="Temporal Sequence TGB1004140 char_human_closeup",
                        command=temporal_sequence_command
                    )
                ]
            )
        ]
    )
    submit_af_job(job)


def run_validation_subprocess():
    spatial_validation_subprocess = subprocess.Popen(
        spatial_validation_command_raw,
        stdout=subprocess.PIPE,
        text=True
    )
    # seqence_over_epoch_subprocess = subprocess.Popen(
    #     seqence_over_epoch_command_raw,
    #     stdin=spatial_validation_subprocess.stdout,
    #     stdout=subprocess.PIPE,
    #     text=True
    # )
    stdout, stderr = spatial_validation_subprocess.communicate()
    print(stdout)


if __name__ == "__main__":
    #submit_experiment()
    #submit_validation()
    run_validation_subprocess()
    #launch_tensorboard(log_dir=config.tensorboard_log_dir, port=6006)