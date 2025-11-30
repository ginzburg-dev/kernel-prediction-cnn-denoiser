import os
from pathlib import Path
from dataclasses import dataclass

from typing import Any
from pathlib import Path


@dataclass
class KPCNConfig():
    project_root = Path(__file__).resolve().parent.parent
    dotenv_file = project_root / ".env"
    kpcn_denoiser = project_root / "kpcn_denoiser"
    trainer_app_path = project_root / "kpcn_denoiser" / "run_train.py"
    out_dir = project_root / "output"
    experiments_dir = out_dir / "experiments"


class ExperimentConfig():
    """Configuration for experiments."""
    def __init__(
            self,
            name: str,
            training_dataset_env: str = "KPCN_TRAINING_DATASET",
            validation_dataset_env: str = "KPCN_VALIDATION_DATASET",
            wrapper_env: str = "KPCN_WRAPPER"
    ) -> None:
        self.global_config = KPCNConfig()
        self.name: Path = Path(name)

        self.kpcn_wrapper: Path = Path(os.getenv(wrapper_env, ""))
        self.training_dataset: Path = Path(os.getenv(training_dataset_env, ""))
        self.validation_dataset: Path = Path(os.getenv(validation_dataset_env, ""))

        self.output_dir: Path = self.global_config.experiments_dir / name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.weights_out_path: Path = self.output_dir / "weights"
        self.weights_out_path.mkdir(parents=True, exist_ok=True)

        self.chekpoints_dir: Path = self.weights_out_path / "checkpoints"
        self.chekpoints_dir.mkdir(parents=True, exist_ok=True)

        self.weights_file_out_path = self.weights_out_path / f"{str(self.name)}_weights.pt"

        self.tensorboard_log_dir: Path = self.output_dir / "tensorboard_logs"
        self.tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

        self.output_images_dir: Path = self.output_dir / "images"
        self.output_images_dir.mkdir(parents=True, exist_ok=True)

        self.output_spatial_dir: Path = self.output_images_dir / "spatial_denoised_validation"
        self.output_spatial_dir.mkdir(parents=True, exist_ok=True)

        self.output_epoch_sequence_dir: Path = self.output_images_dir / "epoch_sequence"
        self.output_epoch_sequence_dir.mkdir(parents=True, exist_ok=True)

        self.output_temporal_sequence_dir: Path = self.output_images_dir / "temporal_sequence"
        self.output_temporal_sequence_dir.mkdir(parents=True, exist_ok=True)


def get_dotenv_value(
        dotenv_var: str,
        dotenv_file: str | Path | None = None
) -> str | None:
    """Get dotenv variable from .env file."""
    config = KPCNConfig()
    with open(dotenv_file if dotenv_file else config.dotenv_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(f"{dotenv_var}="):
                return line.strip().split("=", 1)[1]
    return None
