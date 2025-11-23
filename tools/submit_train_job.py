import os
import sys
import af # pyright: ignore[reportMissingImports]
import argparse
from typing import List

from dataclasses import dataclass, field
from pathlib import Path

from config import PROJECT_ROOT, TRAINER_APP_PATH

@dataclass
class Command:
    title: str
    command: str

@dataclass
class CommandBlock:
    title: str
    commands: List[Command]
    service: str = 'torch'

@dataclass
class Job:
    name: str = "torch-train_job"
    command_blocks: list[CommandBlock] = field(default_factory=list)

def submit_af_job(job: Job, parser: str = "generic") -> None:
    """Submit task to AF Renderfarm.
    Args:
        job (Job): Configuration for the job to be submitted.
        parser (str): Parser type for the job.
        
    As CLI: python -m ml_denoiser.utils.submit_af_job <task name> <command wih args>
        example: python -m ml_denoiser.utils.submit_af_job denoise_dmx004x_model --mode train --input noisy.png --target clean.png --output denoised.png --epochs 100
    """
    af_job = af.Job(job.name)

    for cmd_block in job.command_blocks:
        block = af.Block(cmd_block.title, cmd_block.service)
        block.setWorkingDirectory(working_directory=str(PROJECT_ROOT))
        block.setParser(parser=parser)
        for cmd in cmd_block.commands:
            task = af.Task(cmd.title)
            task.setCommand(cmd.command)
            block.tasks.append(task)

        af_job.blocks.append(block)

    af_job.output()
    af_job.send()

if __name__ == "__main__":
    submit_af_job(Job(
        name=f"torch_{sys.argv[1]}_job",
        command_blocks=[
            CommandBlock(
                title="Training Block",
                commands=[
                    Command(
                        title="Training model",
                        command=f"python {TRAINER_APP_PATH} {' '.join(sys.argv[2:])}"
                    )
                ]
            )
        ]
    ))
