"""Configuration management for training and evaluation.

Loads YAML config files, merges with command-line overrides,
and provides typed access to all hyperparameters.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class ModelConfig:
    """Architecture configuration."""

    name: str = "SiameseUNet"
    encoder: str = "resnet34"
    pretrained: bool = True
    in_channels: int = 3
    num_classes: int = 1
    fusion: str = "both"
    deep_supervision: bool = True


@dataclass
class OptimizerConfig:
    """Optimizer settings."""

    name: str = "adamw"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    betas: list[float] = field(default_factory=lambda: [0.9, 0.999])


@dataclass
class SchedulerConfig:
    """Learning rate scheduler settings."""

    name: str = "cosine_warmup"
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    T_0: int = 20
    T_mult: int = 2


@dataclass
class LossConfig:
    """Loss function configuration."""

    name: str = "bce_dice"
    bce_weight: float = 0.5
    dice_weight: float = 0.5


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    epochs: int = 100
    batch_size: int = 16
    num_workers: int = 4
    mixed_precision: bool = True
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    early_stopping_patience: int = 15
    early_stopping_metric: str = "f1"


@dataclass
class EvalConfig:
    """Evaluation settings."""

    threshold: float = 0.5
    metrics: list[str] = field(
        default_factory=lambda: ["f1", "iou", "precision", "recall", "accuracy"]
    )


@dataclass
class PathConfig:
    """File system paths."""

    data_root: str = "data"
    checkpoint_dir: str = "models/checkpoints"
    results_dir: str = "results"
    log_dir: str = "runs"


@dataclass
class Config:
    """Top-level configuration container.

    Aggregates all sub-configs and provides loading/saving utilities.
    """

    experiment_name: str = "siamese_unet_levir"
    seed: int = 42
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    paths: PathConfig = field(default_factory=PathConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        """Load configuration from a YAML file.

        Args:
            path: Path to the YAML config file.

        Returns:
            Populated Config instance.
        """
        with open(path, "r") as f:
            raw = yaml.safe_load(f)

        config = cls()

        # Map YAML structure to dataclass fields
        if "experiment" in raw:
            config.experiment_name = raw["experiment"].get("name", config.experiment_name)
            config.seed = raw["experiment"].get("seed", config.seed)

        if "model" in raw:
            config.model = ModelConfig(**{
                k: v for k, v in raw["model"].items()
                if k in ModelConfig.__dataclass_fields__
            })

        if "training" in raw:
            t = raw["training"]
            config.training = TrainingConfig(
                epochs=t.get("epochs", 100),
                batch_size=t.get("batch_size", 16),
                num_workers=t.get("num_workers", 4),
                mixed_precision=t.get("mixed_precision", True),
                optimizer=OptimizerConfig(**t.get("optimizer", {})),
                scheduler=SchedulerConfig(**t.get("scheduler", {})),
                loss=LossConfig(**t.get("loss", {})),
                early_stopping_patience=t.get("early_stopping", {}).get("patience", 15),
                early_stopping_metric=t.get("early_stopping", {}).get("metric", "f1"),
            )

        if "evaluation" in raw:
            config.evaluation = EvalConfig(**{
                k: v for k, v in raw["evaluation"].items()
                if k in EvalConfig.__dataclass_fields__
            })

        if "paths" in raw:
            config.paths = PathConfig(**{
                k: v for k, v in raw["paths"].items()
                if k in PathConfig.__dataclass_fields__
            })

        return config

    def to_dict(self) -> dict[str, Any]:
        """Serialize config to a plain dictionary.

        Returns:
            Nested dictionary of all config values.
        """
        import dataclasses
        return dataclasses.asdict(self)

    def save_yaml(self, path: str | Path) -> None:
        """Write current config to a YAML file.

        Args:
            path: Destination file path.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for training/evaluation scripts.

    Returns:
        Parsed arguments with config path and optional overrides.
    """
    parser = argparse.ArgumentParser(description="Satellite Change Detection")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint for evaluation or resuming training",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override (e.g. 'cuda:0', 'cpu')",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    return parser.parse_args()


def load_config(args: Optional[argparse.Namespace] = None) -> Config:
    """Load config from YAML and apply any CLI overrides.

    Args:
        args: Parsed command-line arguments. If None, parses from sys.argv.

    Returns:
        Final Config with all overrides applied.
    """
    if args is None:
        args = parse_args()

    config = Config.from_yaml(args.config)

    # Apply CLI overrides
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.lr is not None:
        config.training.optimizer.lr = args.lr

    return config
