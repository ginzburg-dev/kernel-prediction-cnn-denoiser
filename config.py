from typing import Any
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DOTENV_FILE = PROJECT_ROOT / ".env"

# Package directory
KPCN_DENOISER_DIR = PROJECT_ROOT / "kpcn_denoiser"

# Path to trainer application script
TRAINER_APP_PATH = PROJECT_ROOT / "training/train.py"

OUTPUT_DIR = PROJECT_ROOT / "output"

EXPERIMENTS_DIR = OUTPUT_DIR / "experiments"


def get_dotenv_value(dotenv_var: str) -> str | None:
    """Get dotenv variable from .env file."""
    with open(DOTENV_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(f"{dotenv_var}="):
                return line.strip().split("=", 1)[1]
    return None


class ExperimentConfig:
    """Configuration for experiments."""
    def __init__(
            self,
            name: str,
    ) -> None:
        self.name: Path = Path(name)
        self.output_dir: Path = EXPERIMENTS_DIR / name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.weights_out_path: Path = self.output_dir / "weights"
        self.weights_out_path.parent.mkdir(parents=True, exist_ok=True)

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


