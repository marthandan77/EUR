from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.research.features import BASE_FEATURES, N_CLASSES


MODEL_NAMES = ("gradient_boosted_trees", "lstm", "tcn")
DEFAULT_LOOKBACK = 64
DEFAULT_EPOCHS = 8


def set_reproducible_seed(seed: int = 1729) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


class LSTMClassifier(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 32, dropout: float = 0.2):
        super().__init__()
        self.recurrent = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, N_CLASSES)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.recurrent(inputs)
        return self.classifier(self.dropout(sequence[:, -1, :]))


class CausalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float):
        super().__init__()
        padding = 2 * dilation
        self.pad = padding
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, dilation=dilation)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)

    def _causal(self, layer: nn.Conv1d, values: torch.Tensor) -> torch.Tensor:
        return layer(nn.functional.pad(values, (self.pad, 0)))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.dropout(self.activation(self.norm1(self._causal(self.conv1, values))))
        values = self.dropout(self.activation(self.norm2(self._causal(self.conv2, values))))
        return self.activation(values + residual)


class TCNClassifier(nn.Module):
    def __init__(self, input_size: int, channels: int = 32, dropout: float = 0.2):
        super().__init__()
        self.input_projection = nn.Conv1d(input_size, channels, kernel_size=1)
        self.blocks = nn.Sequential(
            CausalBlock(channels, dilation=1, dropout=dropout),
            CausalBlock(channels, dilation=2, dropout=dropout),
            CausalBlock(channels, dilation=4, dropout=dropout),
        )
        self.classifier = nn.Linear(channels, N_CLASSES)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = inputs.transpose(1, 2)
        values = self.blocks(self.input_projection(values))
        return self.classifier(values[:, :, -1])


@dataclass
class FittedCandidate:
    name: str
    estimator: object
    imputer: SimpleImputer
    scaler: StandardScaler | None
    temperature: float
    lookback: int
    feature_names: tuple[str, ...]
    timeout_payoffs: dict[str, float] = field(default_factory=dict)


class TemperatureCalibrator:
    """Small multiclass temperature scaler fitted on chronological calibration data."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = float(temperature)

    def fit(self, probabilities: np.ndarray, labels: np.ndarray) -> "TemperatureCalibrator":
        from scipy.optimize import minimize_scalar

        probabilities = np.clip(probabilities, 1e-7, 1.0)
        log_probabilities = np.log(probabilities)

        def objective(temperature: float) -> float:
            scaled = _softmax(log_probabilities / temperature)
            return float(-np.log(np.clip(scaled[np.arange(len(labels)), labels], 1e-12, 1.0)).mean())

        result = minimize_scalar(objective, bounds=(0.5, 5.0), method="bounded")
        self.temperature = float(result.x) if result.success else 1.0
        return self

    def transform(self, probabilities: np.ndarray) -> np.ndarray:
        return _softmax(np.log(np.clip(probabilities, 1e-7, 1.0)) / self.temperature)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def _class_probability_matrix(estimator: object, inputs: np.ndarray) -> np.ndarray:
    raw = estimator.predict_proba(inputs)
    aligned = np.full((len(raw), N_CLASSES), 1e-8, dtype=float)
    for source_index, class_value in enumerate(estimator.classes_):
        aligned[:, int(class_value)] = raw[:, source_index]
    aligned /= aligned.sum(axis=1, keepdims=True)
    return aligned


def _make_sequences(
    samples: pd.DataFrame,
    scaled_values: np.ndarray,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    lookup = samples.reset_index(drop=True).copy()
    groups = [group.index.to_numpy() for _, group in lookup.groupby(["symbol", "side"], sort=False)]
    sequence_count = sum(max(0, len(indices) - lookback + 1) for indices in groups)
    if sequence_count == 0:
        raise ValueError(f"No sequences can be formed with lookback={lookback}.")
    sequences = np.empty(
        (sequence_count, lookback, scaled_values.shape[1]),
        dtype=np.float32,
    )
    target_index_parts: list[np.ndarray] = []
    output_start = 0
    for row_indices in groups:
        count = len(row_indices) - lookback + 1
        if count <= 0:
            continue
        group_values = scaled_values[row_indices]
        windows = np.lib.stride_tricks.sliding_window_view(
            group_values,
            window_shape=lookback,
            axis=0,
        ).transpose(0, 2, 1)
        sequences[output_start : output_start + count] = windows
        target_index_parts.append(row_indices[lookback - 1 :])
        output_start += count
    indices = np.concatenate(target_index_parts).astype(int, copy=False)
    return sequences, indices, lookup.iloc[indices].reset_index(drop=True)


def prepare_sequences(
    samples: pd.DataFrame,
    feature_names: tuple[str, ...] = tuple(BASE_FEATURES),
    lookback: int = DEFAULT_LOOKBACK,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, SimpleImputer, StandardScaler]:
    """Fit preprocessing on the given training sample only and form windows."""
    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(samples.loc[:, feature_names])
    scaler = StandardScaler()
    scaled = scaler.fit_transform(imputed).astype(np.float32)
    sequences, _, aligned_samples = _make_sequences(samples, scaled, lookback)
    return sequences, aligned_samples["label"].to_numpy(dtype=np.int64), aligned_samples, imputer, scaler


def transform_sequences(
    samples: pd.DataFrame,
    imputer: SimpleImputer,
    scaler: StandardScaler,
    feature_names: tuple[str, ...],
    lookback: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    imputed = imputer.transform(samples.loc[:, feature_names])
    scaled = scaler.transform(imputed).astype(np.float32)
    sequences, _, aligned_samples = _make_sequences(samples, scaled, lookback)
    return sequences, aligned_samples


def fit_candidate(
    name: str,
    train_samples: pd.DataFrame,
    calibration_samples: pd.DataFrame,
    feature_names: tuple[str, ...] = tuple(BASE_FEATURES),
    lookback: int = DEFAULT_LOOKBACK,
    epochs: int = DEFAULT_EPOCHS,
    seed: int = 1729,
) -> tuple[FittedCandidate, np.ndarray, pd.DataFrame]:
    """Fit one model and calibrate it on a later, disjoint chronological slice."""
    set_reproducible_seed(seed)
    train = train_samples.reset_index(drop=True)
    calibration = calibration_samples.reset_index(drop=True)
    timeout_payoffs = _estimate_timeout_payoffs(pd.concat([train, calibration], ignore_index=True))
    imputer = SimpleImputer(strategy="median")
    imputed_train = imputer.fit_transform(train.loc[:, feature_names])
    scaler = None
    temperature = 1.0

    if name == "gradient_boosted_trees":
        estimator = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=120,
            max_leaf_nodes=15,
            min_samples_leaf=40,
            l2_regularization=1.0,
            random_state=seed,
        )
        estimator.fit(imputed_train, train["label"].to_numpy(dtype=int))
        cal_x = imputer.transform(calibration.loc[:, feature_names])
        raw_probabilities = _class_probability_matrix(estimator, cal_x)
        calibrator = TemperatureCalibrator().fit(raw_probabilities, calibration["label"].to_numpy(dtype=int))
        temperature = calibrator.temperature
        calibrated = calibrator.transform(raw_probabilities)
        candidate = FittedCandidate(
            name, estimator, imputer, scaler, temperature, lookback, feature_names, timeout_payoffs
        )
        return candidate, calibrated, calibration.reset_index(drop=True)

    scaler = StandardScaler()
    scaled_train = scaler.fit_transform(imputed_train).astype(np.float32)
    train_x, _, train_aligned = _make_sequences(train, scaled_train, lookback)
    train_y = train_aligned["label"].to_numpy(dtype=np.int64)
    if len(np.unique(train_y)) < 2:
        raise ValueError(f"{name} training window contains fewer than two outcome classes.")

    context = pd.concat(
        [
            train.groupby(["symbol", "side"], sort=False).tail(lookback - 1),
            calibration,
        ],
        ignore_index=True,
    ).sort_values(["symbol", "side", "datetime"]).reset_index(drop=True)
    calibration_x, context_aligned = transform_sequences(
        context, imputer, scaler, feature_names, lookback
    )
    calibration_dates = set(calibration["datetime"])
    calibration_mask = context_aligned["datetime"].isin(calibration_dates).to_numpy()
    calibration_x = calibration_x[calibration_mask]
    calibration_aligned = context_aligned.loc[calibration_mask].reset_index(drop=True)
    if len(calibration_x) == 0:
        raise ValueError(f"{name} calibration window is too small for lookback={lookback}.")

    estimator = _make_neural_model(name, train_x.shape[2])
    estimator = _fit_neural(
        estimator,
        train_x,
        train_y,
        epochs=epochs,
        seed=seed,
    )
    raw_probabilities = _predict_neural(estimator, calibration_x)
    calibrator = TemperatureCalibrator().fit(
        raw_probabilities,
        calibration_aligned["label"].to_numpy(dtype=int),
    )
    temperature = calibrator.temperature
    calibrated = calibrator.transform(raw_probabilities)
    candidate = FittedCandidate(
        name, estimator, imputer, scaler, temperature, lookback, feature_names, timeout_payoffs
    )
    return candidate, calibrated, calibration_aligned


def _make_neural_model(name: str, input_size: int) -> nn.Module:
    if name == "lstm":
        return LSTMClassifier(input_size)
    if name == "tcn":
        return TCNClassifier(input_size)
    raise ValueError(f"Unknown neural model: {name}")


def _fit_neural(
    model: nn.Module,
    inputs: np.ndarray,
    labels: np.ndarray,
    epochs: int,
    seed: int,
) -> nn.Module:
    set_reproducible_seed(seed)
    model.cpu()
    inputs = np.ascontiguousarray(inputs, dtype=np.float32)
    labels = np.ascontiguousarray(labels, dtype=np.int64)
    if not inputs.flags.writeable:
        inputs = inputs.copy()
    if not labels.flags.writeable:
        labels = labels.copy()
    dataset = TensorDataset(torch.from_numpy(inputs), torch.from_numpy(labels))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, generator=generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    return model.eval()


def _predict_neural(model: nn.Module, inputs: np.ndarray) -> np.ndarray:
    model.eval()
    inputs = np.ascontiguousarray(inputs, dtype=np.float32)
    if not inputs.flags.writeable:
        inputs = inputs.copy()
    with torch.no_grad():
        batches = []
        for start in range(0, len(inputs), 512):
            batch = torch.from_numpy(inputs[start : start + 512])
            logits = model(batch)
            batches.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(batches, axis=0)


def predict_candidate(candidate: FittedCandidate, samples: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    if candidate.name == "gradient_boosted_trees":
        inputs = candidate.imputer.transform(samples.loc[:, candidate.feature_names])
        raw = _class_probability_matrix(candidate.estimator, inputs)
        return TemperatureCalibrator(candidate.temperature).transform(raw), samples.reset_index(drop=True)
    sequences, aligned = transform_sequences(
        samples.reset_index(drop=True),
        candidate.imputer,
        candidate.scaler,
        candidate.feature_names,
        candidate.lookback,
    )
    raw = _predict_neural(candidate.estimator, sequences)
    return TemperatureCalibrator(candidate.temperature).transform(raw), aligned


def _estimate_timeout_payoffs(samples: pd.DataFrame) -> dict[str, float]:
    """Estimate timeout R from matured pre-evaluation samples only."""
    timeouts = samples.loc[samples["label"].eq(1)]
    global_mean = float(timeouts["realized_r"].mean()) if not timeouts.empty else 0.0
    estimates = {"__global__": global_mean}
    if not timeouts.empty:
        grouped = timeouts.groupby(["symbol", "side"])["realized_r"].agg(["mean", "count"])
        for (symbol, side), row in grouped.iterrows():
            if int(row["count"]) >= 20:
                estimates[f"{symbol}|{int(side)}"] = float(row["mean"])
    return estimates
