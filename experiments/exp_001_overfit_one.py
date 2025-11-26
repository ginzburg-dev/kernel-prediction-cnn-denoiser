import os
import sys
import af # pyright: ignore[reportMissingImports]
import argparse

from pathlib import Path

from kpcn_denoiser.config import KPCNConfig, ExperimentConfig
from tools.submit_train_job import Command, CommandBlock, Job, submit_af_job
from tools.run_tensorboard import launch_tensorboard

MODEL = "UNetResidual"

def submit_experiment():
    config = ExperimentConfig(__file__)
    global_config = KPCNConfig()

    job_name = f"torch-{config.name}-job"
    train_command = [
        "python",
        global_config.trainer_app_path,
        "--mode train",
        f"--model {MODEL}",
        f"--model-in-channels 3",
        f"--model-out-channels 64",
        f"--loss L1Loss",
        f"--input {config.training_dataset}",
        f"--output {config.output_dir}",
        f"--weights-out {config.weights_out_path}",
        f"--weights-in {config.weights_out_path}",
        f"--epochs 240",
        f"--batch-size 8",
        f"--patches-per-image 200",
        f"--patch-size 128",
        f"--n-first-samples 200",
        f"--n-first-frames 3",
        f"--save-checkpoint-every 5",
        f"--print-every-n-step 1",
        f"--lr 1e-4",
        f"--tensorboard-output {config.tensorboard_log_dir}"
    ]
    train_command = " ".join(map(str, train_command))

    spatial_validation_command= [
        "python",
        global_config.trainer_app_path,
        "--mode apply",
        f"--input {config.validation_dataset}",
        f"--output {config.output_spatial_dir}",
        f"--weights-in {config.weights_out_path}",
    ]
    spatial_validation_command = " ".join(map(str, spatial_validation_command))

    seqence_over_epoch_command = [
        "python",
        global_config.trainer_app_path,
        "--mode apply_epoch_sequence",
        f"--input {config.validation_dataset}",
        f"--output {config.output_epoch_sequence_dir}",
        f"--weights-in {config.weights_out_path}",
    ]
    seqence_over_epoch_command = " ".join(map(str, seqence_over_epoch_command))


    char_human_closeup_temporal_sequence = config.training_dataset / "TGB1004140" / "chars" / "rgba" / "noisy"
    out = config.output_temporal_sequence_dir / 'TGB1004140_char_human_closeup'
    out.mkdir(parents=True, exist_ok=True)
    char_human_closeup_temporal_weights_in = config.weights_out_path / "exp_001_overfit_one_weights_checkpoint.0200.json"
    temporal_sequence_command = [
        "python",
        global_config.trainer_app_path,
        "--mode apply",
        f"--input {char_human_closeup_temporal_sequence}",
        f"--output {out}",
        f"--weights-in {char_human_closeup_temporal_weights_in}",
    ]
    temporal_sequence_command = " ".join(map(str, temporal_sequence_command))


    job = Job(
        name=job_name,
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
    launch_tensorboard(config.tensorboard_log_dir, port=6006)

if __name__ == "__main__":
    submit_experiment()
    

