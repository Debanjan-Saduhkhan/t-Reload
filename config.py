from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import json
from pathlib import Path

try:
    from .utils import load_config, save_config
except ImportError:
    from utils import load_config, save_config


@dataclass
class ModelConfig:
    """Configuration for the model architecture."""
    model_type: str = "t-reload"
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout: float = 0.1
    activation: str = "gelu"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dropout": self.dropout,
            "activation": self.activation
        }


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "gradient_clip": self.gradient_clip,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "logging_steps": self.logging_steps
        }


@dataclass
class DataConfig:
    """Configuration for data processing."""
    data_dir: str = "data"
    train_file: str = "train.json"
    val_file: str = "val.json"
    test_file: str = "test.json"
    max_length: int = 512
    preprocessing: str = "standard"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_dir": self.data_dir,
            "train_file": self.train_file,
            "val_file": self.val_file,
            "test_file": self.test_file,
            "max_length": self.max_length,
            "preprocessing": self.preprocessing
        }


@dataclass
class ExperimentConfig:
    """Main configuration class combining all configs."""
    experiment_name: str = "t-reload_experiment"
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "seed": self.seed,
            "device": self.device,
            "num_workers": self.num_workers,
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
            "data": self.data.to_dict()
        }
    
    def save(self, path: str) -> None:
        """Save configuration to file."""
        save_config(self.to_dict(), path)
    
    @classmethod
    def load(cls, path: str) -> "ExperimentConfig":
        """Load configuration from file."""
        config_dict = load_config(path)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ExperimentConfig":
        """Create config from dictionary."""
        model_config = ModelConfig(**config_dict.get("model", {}))
        training_config = TrainingConfig(**config_dict.get("training", {}))
        data_config = DataConfig(**config_dict.get("data", {}))
        
        return cls(
            experiment_name=config_dict.get("experiment_name", "t-reload_experiment"),
            seed=config_dict.get("seed", 42),
            device=config_dict.get("device", "cuda"),
            num_workers=config_dict.get("num_workers", 4),
            model=model_config,
            training=training_config,
            data=data_config
        )


def get_default_config() -> ExperimentConfig:
    """Get default configuration."""
    return ExperimentConfig()


def create_config_from_paper(paper_info: Dict[str, Any]) -> ExperimentConfig:
    """Create configuration based on paper information."""
    config = get_default_config()
    
    # Update based on paper information
    if paper_info.get("hyperparams"):
        # Extract hyperparameters from paper
        pass
    
    return config
