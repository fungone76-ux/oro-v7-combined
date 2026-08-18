"""Causal tensor builder for TOP40 SHORT MEMORY 5/3/2.

Isolated from short_memory (8/5/3). Feature/target definitions are deliberately
kept identical; only temporal window lengths come from short_memory_532.config.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .config import CONFIG
from short_memory.training import (
    M1_FEATURES, M5_FEATURES, M15_FEATURES,
    STATIC_NUMERIC, STATIC_CATEGORICAL,
)


@dataclass
class TensorBundle532:
    frame: pd.DataFrame
    m1: np.ndarray
    m5: np.ndarray
    m15: np.ndarray
    static_num: np.ndarray
    static_cat: pd.DataFrame
    y_s2: np.ndarray


def _unique_htf(market: pd.DataFrame, source: str, features: list[str]) -> pd.DataFrame:
    out = market[[source] + features].dropna(subset=[source]).copy()
    out[source] = pd.to_datetime(out[source], utc=True)
    return out.sort_values(source).drop_duplicates(source, keep="last").reset_index(drop=True)


def _continuous(times: np.ndarray, start: int, end: int, expected_minutes: int) -> bool:
    if start < 0 or end < start:
        return False
    d = np.diff(times[start:end + 1]).astype("timedelta64[m]").astype(int)
    return len(d) == (end - start) and bool(np.all(d == expected_minutes))


def _slice(values: np.ndarray, times: np.ndarray, ts: np.datetime64, length: int, step: int) -> np.ndarray | None:
    end = int(np.searchsorted(times, ts, side="right") - 1)
    start = end - length + 1
    if start < 0 or not _continuous(times, start, end, step):
        return None
    if times[end] > ts:
        raise AssertionError("LOOKAHEAD_DETECTED")
    return values[start:end + 1]


def target_s2(candidates: pd.DataFrame) -> np.ndarray:
    valid = candidates["label_validity_15m"].astype(str).eq("VALID")
    atr = candidates["atr14"].astype(float).replace(0, np.nan)
    q = (
        candidates["directional_MFE_price_15m"].astype(float) / atr
        - (candidates["directional_MAE_price_15m"].astype(float) / atr).abs()
    )
    return q.where(valid, np.nan).to_numpy(np.float32)


def build_tensor_bundle(market: pd.DataFrame, candidates: pd.DataFrame) -> TensorBundle532:
    market = market.sort_values("timestamp").reset_index(drop=True)
    candidates = candidates.sort_values("timestamp").reset_index(drop=True)
    m5 = _unique_htf(market, "m5_source_close_time", M5_FEATURES)
    m15 = _unique_htf(market, "m15_source_close_time", M15_FEATURES)

    t1 = market["timestamp"].to_numpy(dtype="datetime64[ns]")
    t5 = m5["m5_source_close_time"].to_numpy(dtype="datetime64[ns]")
    t15 = m15["m15_source_close_time"].to_numpy(dtype="datetime64[ns]")
    v1 = market[M1_FEATURES].to_numpy(np.float32)
    v5 = m5[M5_FEATURES].to_numpy(np.float32)
    v15 = m15[M15_FEATURES].to_numpy(np.float32)

    keep: list[int] = []
    a1: list[np.ndarray] = []
    a5: list[np.ndarray] = []
    a15: list[np.ndarray] = []
    for idx, row in candidates.iterrows():
        ts = np.datetime64(pd.Timestamp(row["timestamp"]).to_datetime64())
        w1 = _slice(v1, t1, ts, CONFIG.m1_length, 1)
        w5 = _slice(v5, t5, ts, CONFIG.m5_length, 5)
        w15 = _slice(v15, t15, ts, CONFIG.m15_length, 15)
        if w1 is None or w5 is None or w15 is None:
            continue
        if not (np.isfinite(w1).all() and np.isfinite(w5).all() and np.isfinite(w15).all()):
            continue
        keep.append(idx)
        a1.append(w1); a5.append(w5); a15.append(w15)

    frame = candidates.loc[keep].reset_index(drop=True)
    if frame.empty:
        raise ValueError("NO_VALID_532_SEQUENCES")
    static_num = frame[STATIC_NUMERIC].astype(float).to_numpy(np.float32)
    static_cat = frame[STATIC_CATEGORICAL].astype(str).copy()
    y2 = target_s2(frame)
    valid = np.isfinite(y2) & np.isfinite(static_num).all(axis=1)
    frame = frame.loc[valid].reset_index(drop=True)
    return TensorBundle532(
        frame=frame,
        m1=np.stack(a1)[valid],
        m5=np.stack(a5)[valid],
        m15=np.stack(a15)[valid],
        static_num=static_num[valid],
        static_cat=static_cat.loc[valid].reset_index(drop=True),
        y_s2=y2[valid],
    )
