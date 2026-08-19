from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CONFIG

M1_FEATURES = [
    "open", "high", "low", "close", "range", "body", "upper_wick", "lower_wick",
    "return_1", "ema9", "ema21", "atr14", "tick_volume_z20",
]
M5_FEATURES = ["m5_open", "m5_high", "m5_low", "m5_close", "ema9", "ema21", "atr14_m5"]
M15_FEATURES = ["m15_open", "m15_high", "m15_low", "m15_close", "ema50"]
STATIC_FEATURES = ["direction_sign", "m15_direction_sign", "direction_agreement", "trend_conflict"]


@dataclass
class TensorBundle611:
    frame: pd.DataFrame
    m1: np.ndarray
    m5: np.ndarray
    m15: np.ndarray
    static_num: np.ndarray
    y: np.ndarray


def _direction_sign(s: pd.Series) -> np.ndarray:
    return np.where(s.astype(str).eq("LONG"), 1.0, np.where(s.astype(str).eq("SHORT"), -1.0, 0.0)).astype(np.float32)


def build_tensor_bundle(root: Path) -> TensorBundle611:
    samples = pd.read_parquet(root / "samples_611_mar13aug.parquet").copy()
    m1f = pd.read_parquet(root / "m1_features_611.parquet").copy()
    samples["timestamp"] = pd.to_datetime(samples["timestamp"], utc=True)
    m1f["close_time"] = pd.to_datetime(m1f["close_time"], utc=True)
    samples = samples.sort_values("timestamp").reset_index(drop=True)
    m1f = m1f.sort_values("close_time").reset_index(drop=True)

    times = m1f["close_time"].to_numpy(dtype="datetime64[ns]")
    values = m1f[M1_FEATURES].to_numpy(np.float32)

    keep: list[int] = []
    seqs: list[np.ndarray] = []
    for i, ts in enumerate(samples["timestamp"]):
        end = int(np.searchsorted(times, ts.to_datetime64(), side="right") - 1)
        start = end - CONFIG.m1_length + 1
        if start < 0:
            continue
        seq_t = times[start:end + 1]
        if len(seq_t) != CONFIG.m1_length:
            continue
        d = np.diff(seq_t).astype("timedelta64[m]").astype(int)
        if not np.all(d == 1):
            continue
        seq = values[start:end + 1]
        if not np.isfinite(seq).all():
            continue
        keep.append(i)
        seqs.append(seq)

    frame = samples.iloc[keep].reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("NO_VALID_611_SEQUENCES")

    m5 = frame[M5_FEATURES].astype(float).to_numpy(np.float32)[:, None, :]
    m15 = frame[M15_FEATURES].astype(float).to_numpy(np.float32)[:, None, :]
    static = pd.DataFrame({
        "direction_sign": _direction_sign(frame["direction"]),
        "m15_direction_sign": _direction_sign(frame["direction_m15"]),
        "direction_agreement": frame["direction_agreement"].astype(float).to_numpy(np.float32),
        "trend_conflict": frame["trend_conflict"].astype(float).to_numpy(np.float32),
    })[STATIC_FEATURES].to_numpy(np.float32)
    y = pd.to_numeric(frame["label_quality_r"], errors="coerce").to_numpy(np.float32)
    valid = np.isfinite(y) & np.isfinite(m5).all(axis=(1,2)) & np.isfinite(m15).all(axis=(1,2)) & np.isfinite(static).all(axis=1)

    return TensorBundle611(
        frame=frame.loc[valid].reset_index(drop=True),
        m1=np.stack(seqs).astype(np.float32)[valid],
        m5=m5[valid],
        m15=m15[valid],
        static_num=static[valid],
        y=y[valid],
    )
