from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataConfig:
    image_size: int = 128
    seed: int = 5910
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    test_fraction: float = 0.15

    def validate(self) -> None:
        total = self.train_fraction + self.validation_fraction + self.test_fraction
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Split fractions must sum to 1.0, received {total:.6f}.")
        if self.image_size < 32:
            raise ValueError("image_size must be at least 32 pixels.")


@dataclass(frozen=True)
class TrainConfig:
    model_name: str = "multitask"
    epochs: int = 15
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    dropout: float = 0.30
    integrity_loss_weight: float = 0.50
    early_stopping_patience: int = 4
    num_workers: int = 0
    device: str = "cpu"
    seed: int = 5910
    initial_checkpoint: str | None = None

    def validate(self) -> None:
        if self.model_name not in {"baseline", "improved", "multitask"}:
            raise ValueError(f"Unsupported model_name: {self.model_name}")
        if self.epochs < 1 or self.batch_size < 1:
            raise ValueError("epochs and batch_size must be positive.")
        if not 0.0 <= self.integrity_loss_weight <= 2.0:
            raise ValueError("integrity_loss_weight must be between 0 and 2.")
        if self.initial_checkpoint and self.model_name != "multitask":
            raise ValueError("Checkpoint initialization is only supported for multitask training.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def processed_data(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def figures(self) -> Path:
        return self.root / "reports" / "figures"
