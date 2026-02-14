"""Tests for configuration loading and management."""

from pathlib import Path
import tempfile

import pytest
import yaml

from src.config import Config, ModelConfig


class TestConfig:
    """Test configuration loading and serialization."""

    def test_default_config(self) -> None:
        """Default config should have sensible values."""
        config = Config()
        assert config.seed == 42
        assert config.model.encoder == "resnet34"
        assert config.training.epochs == 100
        assert config.training.batch_size == 16

    def test_from_yaml(self, tmp_path: Path) -> None:
        """Config should load correctly from YAML files."""
        yaml_content = {
            "experiment": {"name": "test_run", "seed": 123},
            "model": {"encoder": "resnet34", "in_channels": 13},
            "training": {
                "epochs": 50,
                "batch_size": 8,
                "optimizer": {"lr": 0.001},
            },
        }
        yaml_path = tmp_path / "test_config.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_content, f)

        config = Config.from_yaml(yaml_path)
        assert config.experiment_name == "test_run"
        assert config.seed == 123
        assert config.model.in_channels == 13
        assert config.training.epochs == 50
        assert config.training.batch_size == 8

    def test_to_dict(self) -> None:
        """Config should serialize to a nested dictionary."""
        config = Config()
        d = config.to_dict()
        assert isinstance(d, dict)
        assert "model" in d
        assert d["model"]["encoder"] == "resnet34"

    def test_save_and_reload(self, tmp_path: Path) -> None:
        """Config should survive a save/load round trip."""
        config = Config()
        config.seed = 999
        config.training.epochs = 77

        save_path = tmp_path / "saved_config.yaml"
        config.save_yaml(save_path)

        loaded = Config.from_yaml(save_path)
        assert loaded.seed == 999
        assert loaded.training.epochs == 77
