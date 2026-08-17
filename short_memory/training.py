"""Training primitives for TOP40_SHORT_MEMORY V1.

This module preserves the original Phase 3B2 model architecture while freezing
sequence lengths to 8/5/3. Thresholds are derived only from OOS predictions.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import CONFIG, MODEL_ARCHITECTURE


M1_FEATURES = [
    "return_1m", "return_3m", "return_5m", "body_to_range", "upper_wick_to_range",
    "lower_wick_to_range", "close_location_in_bar", "range_atr", "body_atr",
    "ema_distance_atr", "ema9_slope_1", "ema9_slope_3", "ema21_slope_1",
    "ema21_slope_3", "rsi14", "rsi_delta_1", "rsi_delta_3", "atr_percentile",
    "atr_change_1", "atr_change_3", "bb_width_atr", "bb_width_change_1",
    "bb_width_change_3", "bb_position", "volume_ratio_5", "volume_ratio_20",
    "volume_zscore_20",
]
M5_FEATURES = [
    "return_5m", "m5_ema9", "m5_ema21", "m5_ema_distance_atr",
    "atr_percentile", "atr_change_3", "bb_width_atr", "bb_position", "volume_ratio_20",
]
M15_FEATURES = [
    "return_10m", "m15_ema50", "m15_ema50_slope", "price_distance_m15_ema50",
    "atr_percentile", "atr_regime_strength",
]
STATIC_NUMERIC = [
    "signal_score", "confirmation_count", "spread_points", "candidate_stop_distance",
    "signed_ema_distance_atr", "signed_m5_ema_distance_atr",
    "signed_price_distance_m15_ema50",
]
STATIC_CATEGORICAL = ["direction", "session", "regime", "m15_trend_direction"]


@dataclass(frozen=True)
class Fold:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp


def set_deterministic(seed: int = CONFIG.seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ModuleNotFoundError:
        pass


def fit_scalers_train_only(m1: np.ndarray, m5: np.ndarray, m15: np.ndarray, static_num: np.ndarray) -> dict[str, StandardScaler]:
    def fit(values: np.ndarray) -> StandardScaler:
        scaler = StandardScaler()
        scaler.fit(values.reshape(-1, values.shape[-1]))
        return scaler
    return {
        "m1": fit(m1),
        "m5": fit(m5),
        "m15": fit(m15),
        "static_numeric": fit(static_num),
    }


def transform_with_scalers(m1: np.ndarray, m5: np.ndarray, m15: np.ndarray, static_num: np.ndarray, scalers: dict[str, StandardScaler]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    def seq(values: np.ndarray, key: str) -> np.ndarray:
        shape = values.shape
        return scalers[key].transform(values.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)
    return (
        seq(m1, "m1"),
        seq(m5, "m5"),
        seq(m15, "m15"),
        scalers["static_numeric"].transform(static_num).astype(np.float32),
    )


def make_walkforward_folds(times: pd.Series, n_folds: int = 4) -> list[Fold]:
    t = pd.to_datetime(times, utc=True).sort_values().reset_index(drop=True)
    if len(t) < 100:
        raise ValueError("INSUFFICIENT_WALKFORWARD_ROWS")
    edges = np.linspace(0, len(t), n_folds + 2, dtype=int)
    folds: list[Fold] = []
    for i in range(n_folds):
        train_end = t.iloc[edges[i + 1] - 1]
        # 15m purge + 15m embargo = validation begins 30m after training end.
        valid_start = t.iloc[edges[i + 1]] + pd.Timedelta(minutes=CONFIG.purge_minutes + CONFIG.embargo_minutes)
        valid_end = t.iloc[edges[i + 2] - 1]
        folds.append(Fold(i + 1, t.iloc[0], train_end, valid_start, valid_end))
    return folds


def torch_model_class():
    import torch

    class ShortMemoryLSTM(torch.nn.Module):
        def __init__(self, m1_dim: int, m5_dim: int, m15_dim: int, static_dim: int, task: str):
            super().__init__()
            self.task = task
            self.m1 = torch.nn.LSTM(m1_dim, MODEL_ARCHITECTURE["m1_lstm_hidden"], batch_first=True)
            self.m5 = torch.nn.LSTM(m5_dim, MODEL_ARCHITECTURE["m5_lstm_hidden"], batch_first=True)
            self.m15 = torch.nn.LSTM(m15_dim, MODEL_ARCHITECTURE["m15_lstm_hidden"], batch_first=True)
            merged = MODEL_ARCHITECTURE["m1_lstm_hidden"] + MODEL_ARCHITECTURE["m5_lstm_hidden"] + MODEL_ARCHITECTURE["m15_lstm_hidden"] + static_dim
            self.head = torch.nn.Sequential(
                torch.nn.Linear(merged, MODEL_ARCHITECTURE["dense"]),
                torch.nn.ReLU(),
                torch.nn.Dropout(MODEL_ARCHITECTURE["dropout"]),
                torch.nn.Linear(MODEL_ARCHITECTURE["dense"], 1),
            )

        def forward(self, m1, m5, m15, static):
            _, (h1, _) = self.m1(m1)
            _, (h5, _) = self.m5(m5)
            _, (h15, _) = self.m15(m15)
            merged = torch.cat([h1[-1], h5[-1], h15[-1], static], dim=1)
            return self.head(merged).squeeze(1)

    return ShortMemoryLSTM


def freeze_oos_thresholds(oos_s2_scores: np.ndarray) -> dict[str, float]:
    scores = np.asarray(oos_s2_scores, dtype=float)
    scores = scores[np.isfinite(scores)]
    if len(scores) < 20:
        raise ValueError("INSUFFICIENT_OOS_SCORES")
    return {
        "TOP40_SHORT": float(np.quantile(scores, 0.60)),
        "TOP20_SHORT": float(np.quantile(scores, 0.80)),
    }


def write_threshold_freeze(path: Path, thresholds: dict[str, float], artifact_hashes: dict[str, str]) -> None:
    payload = {
        "sequence_lengths": {"m1": 8, "m5": 5, "m15": 3},
        "purge_minutes": CONFIG.purge_minutes,
        "embargo_minutes": CONFIG.embargo_minutes,
        "threshold_source": "OOS predictions only",
        "thresholds": thresholds,
        "economic_metrics_opened": False,
        "artifact_hashes": artifact_hashes,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
