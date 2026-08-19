"""Causal sequence extraction for SHORT_MEMORY_V1.

This module is intentionally independent from model training so its temporal
contract can be tested in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CONFIG


@dataclass(frozen=True)
class SequenceWindow:
    candidate_time: pd.Timestamp
    m1: pd.DataFrame
    m5: pd.DataFrame
    m15: pd.DataFrame


def _utc(ts: pd.Timestamp | str) -> pd.Timestamp:
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        return out.tz_localize("UTC")
    return out.tz_convert("UTC")


def _closed_window(
    frame: pd.DataFrame,
    candidate_time: pd.Timestamp,
    length: int,
    expected_minutes: int,
    time_col: str = "time",
) -> pd.DataFrame:
    """Return the last `length` closed rows at or before candidate_time.

    The timestamp column is interpreted as the close/availability timestamp of
    the row. A row with timestamp > candidate_time is never admissible.
    Continuity is strict; market-closure crossing invalidates the sequence.
    """
    if time_col not in frame.columns:
        raise ValueError(f"missing time column: {time_col}")
    f = frame.copy()
    f[time_col] = pd.to_datetime(f[time_col], utc=True)
    f = f[f[time_col] <= candidate_time].sort_values(time_col).drop_duplicates(time_col, keep="last")
    if len(f) < length:
        raise ValueError("INSUFFICIENT_HISTORY")
    out = f.iloc[-length:].copy().reset_index(drop=True)
    deltas = out[time_col].diff().dropna().dt.total_seconds().div(60).to_numpy()
    if len(deltas) and not np.all(deltas == expected_minutes):
        raise ValueError("CROSSES_MARKET_CLOSURE")
    if out[time_col].max() > candidate_time:
        raise AssertionError("LOOKAHEAD_DETECTED")
    return out


def build_short_memory_window(
    m1: pd.DataFrame,
    m5: pd.DataFrame,
    m15: pd.DataFrame,
    candidate_time: pd.Timestamp | str,
) -> SequenceWindow:
    ts = _utc(candidate_time)
    return SequenceWindow(
        candidate_time=ts,
        m1=_closed_window(m1, ts, CONFIG.m1_length, 1),
        m5=_closed_window(m5, ts, CONFIG.m5_length, 5),
        m15=_closed_window(m15, ts, CONFIG.m15_length, 15),
    )
