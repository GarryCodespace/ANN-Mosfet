from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


FEATURE_COLUMNS = [
    "vgs",
    "vds",
    "vbs",
    "width_um",
    "length_um",
    "temperature_c",
    "vth0",
    "kp",
    "lambda_",
    "gamma",
    "phi",
]
TARGET_COLUMN = "log_id"


@dataclass
class Metrics:
    mae_log10_a: float
    rmse_log10_a: float
    r2: float


@dataclass
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, values: np.ndarray) -> "Standardizer":
        mean = values.mean(axis=0)
        scale = values.std(axis=0)
        scale = np.where(scale == 0.0, 1.0, scale)
        return cls(mean=mean.astype(np.float32), scale=scale.astype(np.float32))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return ((values - self.mean) / self.scale).astype(np.float32)

    def to_state(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_state(cls, state: dict[str, list[float]]) -> "Standardizer":
        return cls(
            mean=np.asarray(state["mean"], dtype=np.float32),
            scale=np.asarray(state["scale"], dtype=np.float32),
        )


class MosfetMlp(nn.Module):
    def __init__(self, input_dim: int, hidden_layer_sizes: tuple[int, ...]):
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.Tanh())
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs).squeeze(-1)


class MosfetAnnModel:
    def __init__(
        self,
        network: MosfetMlp,
        feature_scaler: Standardizer,
        target_scaler: Standardizer,
        device: str = "cpu",
    ):
        self.network = network.to(device)
        self.feature_scaler = feature_scaler
        self.target_scaler = target_scaler
        self.device = device
        self.network.eval()

    def predict_log_current(self, frame: pd.DataFrame) -> np.ndarray:
        features = _features(frame)
        scaled = self.feature_scaler.transform(features)
        tensor = torch.as_tensor(scaled, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            predicted_scaled = self.network(tensor).cpu().numpy()
        return (predicted_scaled * self.target_scaler.scale[0] + self.target_scaler.mean[0]).astype(float)

    def predict_current(self, frame: pd.DataFrame) -> np.ndarray:
        return np.power(10.0, self.predict_log_current(frame))

    def evaluate(self, frame: pd.DataFrame) -> Metrics:
        actual = frame[TARGET_COLUMN].to_numpy(dtype=float)
        predicted = self.predict_log_current(frame)
        residual = actual - predicted
        mse = float(np.mean(residual**2))
        actual_variance = float(np.sum((actual - actual.mean()) ** 2))
        r2 = 1.0 - float(np.sum(residual**2)) / actual_variance if actual_variance > 0.0 else 0.0
        return Metrics(
            mae_log10_a=float(np.mean(np.abs(residual))),
            rmse_log10_a=float(np.sqrt(mse)),
            r2=float(r2),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "feature_columns": FEATURE_COLUMNS,
            "target_column": TARGET_COLUMN,
            "hidden_layer_sizes": _hidden_layer_sizes(self.network),
            "feature_scaler": self.feature_scaler.to_state(),
            "target_scaler": self.target_scaler.to_state(),
            "state_dict": self.network.cpu().state_dict(),
        }
        torch.save(checkpoint, path)
        self.network.to(self.device)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "MosfetAnnModel":
        checkpoint = torch.load(path, map_location=device)
        hidden_layer_sizes = tuple(checkpoint["hidden_layer_sizes"])
        network = MosfetMlp(input_dim=len(FEATURE_COLUMNS), hidden_layer_sizes=hidden_layer_sizes)
        network.load_state_dict(checkpoint["state_dict"])
        return cls(
            network=network,
            feature_scaler=Standardizer.from_state(checkpoint["feature_scaler"]),
            target_scaler=Standardizer.from_state(checkpoint["target_scaler"]),
            device=device,
        )

    
def train_model(
    frame: pd.DataFrame,
    hidden_layer_sizes: tuple[int, ...] = (96, 64, 32),
    random_state: int = 7,
    max_iter: int = 800,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cpu",
) -> tuple[MosfetAnnModel, Metrics, Metrics]:
    _seed_everything(random_state)
    train_frame, test_frame = _train_test_split(frame, test_fraction=0.2, seed=random_state)

    train_features = _features(train_frame)
    train_target = _target(train_frame)
    feature_scaler = Standardizer.fit(train_features)
    target_scaler = Standardizer.fit(train_target)

    train_dataset = TensorDataset(
        torch.as_tensor(feature_scaler.transform(train_features), dtype=torch.float32),
        torch.as_tensor(target_scaler.transform(train_target).reshape(-1), dtype=torch.float32),
    )
    generator = torch.Generator().manual_seed(random_state)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    network = MosfetMlp(input_dim=len(FEATURE_COLUMNS), hidden_layer_sizes=hidden_layer_sizes).to(device)
    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = nn.MSELoss()

    network.train()
    for _ in range(max_iter):
        for batch_features, batch_target in train_loader:
            batch_features = batch_features.to(device)
            batch_target = batch_target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(network(batch_features), batch_target)
            loss.backward()
            optimizer.step()

    model = MosfetAnnModel(network, feature_scaler, target_scaler, device=device)
    return model, model.evaluate(train_frame), model.evaluate(test_frame)


def _features(frame: pd.DataFrame) -> np.ndarray:
    return frame[FEATURE_COLUMNS].to_numpy(dtype=np.float32)


def _target(frame: pd.DataFrame) -> np.ndarray:
    return frame[[TARGET_COLUMN]].to_numpy(dtype=np.float32)


def _train_test_split(
    frame: pd.DataFrame,
    test_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(frame))
    test_size = max(1, int(len(frame) * test_fraction))
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    return frame.iloc[train_indices].reset_index(drop=True), frame.iloc[test_indices].reset_index(drop=True)


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _hidden_layer_sizes(network: MosfetMlp) -> tuple[int, ...]:
    return tuple(module.out_features for module in network.network if isinstance(module, nn.Linear))[:-1]
