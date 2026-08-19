"""Adapter from original TOP40 Phase 3A parquets to SHORT_MEMORY_V1 tensors."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from .config import CONFIG
from .training import M1_FEATURES, M5_FEATURES, M15_FEATURES, STATIC_NUMERIC, STATIC_CATEGORICAL


@dataclass
class TensorBundle:
    frame: pd.DataFrame
    m1: np.ndarray
    m5: np.ndarray
    m15: np.ndarray
    static_num: np.ndarray
    static_cat: pd.DataFrame
    y_s1: np.ndarray
    y_s2: np.ndarray


def load_phase3a(market_path: Path, candidate_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    market = pd.read_parquet(market_path)
    candidates = pd.read_parquet(candidate_path)
    market['timestamp'] = pd.to_datetime(market['timestamp'], utc=True)
    candidates['timestamp'] = pd.to_datetime(candidates['timestamp'], utc=True)
    market = market[market['feature_valid'].astype(bool)].sort_values('timestamp').reset_index(drop=True)
    candidates = candidates[candidates['feature_valid'].astype(bool)].sort_values('timestamp').reset_index(drop=True)
    missing_market = [c for c in M1_FEATURES + M5_FEATURES + M15_FEATURES + ['m5_source_close_time','m15_source_close_time'] if c not in market.columns]
    missing_candidates = [c for c in STATIC_NUMERIC + STATIC_CATEGORICAL if c not in candidates.columns]
    if missing_market or missing_candidates:
        raise ValueError(f'PHASE3A_FEATURE_CONTRACT_MISMATCH market={missing_market} candidates={missing_candidates}')
    return market, candidates


def _unique_htf(market: pd.DataFrame, source: str, features: list[str]) -> pd.DataFrame:
    out = market[[source] + features].dropna(subset=[source]).copy()
    out[source] = pd.to_datetime(out[source], utc=True)
    return out.sort_values(source).drop_duplicates(source, keep='last').reset_index(drop=True)


def _continuous(times: np.ndarray, start: int, end: int, expected_minutes: int) -> bool:
    if start < 0 or end < start:
        return False
    d = np.diff(times[start:end+1]).astype('timedelta64[m]').astype(int)
    return len(d) == (end-start) and bool(np.all(d == expected_minutes))


def _slice(values: np.ndarray, times: np.ndarray, ts: np.datetime64, length: int, step: int) -> np.ndarray | None:
    end = int(np.searchsorted(times, ts, side='right') - 1)
    start = end - length + 1
    if start < 0 or not _continuous(times, start, end, step):
        return None
    if times[end] > ts:
        raise AssertionError('LOOKAHEAD_DETECTED')
    return values[start:end+1]


def target_s1(candidates: pd.DataFrame) -> np.ndarray:
    valid = candidates['label_validity_15m'].astype(str).eq('VALID')
    y = np.where(valid, (candidates['directional_future_return_15m'].astype(float) > 0).astype(float), np.nan)
    return y.astype(np.float32)


def target_s2(candidates: pd.DataFrame) -> np.ndarray:
    valid = candidates['label_validity_15m'].astype(str).eq('VALID')
    atr = candidates['atr14'].astype(float).replace(0, np.nan)
    q = candidates['directional_MFE_price_15m'].astype(float) / atr - (candidates['directional_MAE_price_15m'].astype(float) / atr).abs()
    return q.where(valid, np.nan).to_numpy(np.float32)


def build_tensor_bundle(market: pd.DataFrame, candidates: pd.DataFrame) -> TensorBundle:
    market = market.sort_values('timestamp').reset_index(drop=True)
    m5 = _unique_htf(market, 'm5_source_close_time', M5_FEATURES)
    m15 = _unique_htf(market, 'm15_source_close_time', M15_FEATURES)

    t1 = market['timestamp'].to_numpy(dtype='datetime64[ns]')
    t5 = m5['m5_source_close_time'].to_numpy(dtype='datetime64[ns]')
    t15 = m15['m15_source_close_time'].to_numpy(dtype='datetime64[ns]')
    v1 = market[M1_FEATURES].to_numpy(np.float32)
    v5 = m5[M5_FEATURES].to_numpy(np.float32)
    v15 = m15[M15_FEATURES].to_numpy(np.float32)

    keep = []
    a1 = []
    a5 = []
    a15 = []
    for idx, row in candidates.iterrows():
        ts = np.datetime64(pd.Timestamp(row['timestamp']).to_datetime64())
        w1 = _slice(v1, t1, ts, CONFIG.m1_length, 1)
        w5 = _slice(v5, t5, ts, CONFIG.m5_length, 5)
        w15 = _slice(v15, t15, ts, CONFIG.m15_length, 15)
        if w1 is None or w5 is None or w15 is None:
            continue
        if not (np.isfinite(w1).all() and np.isfinite(w5).all() and np.isfinite(w15).all()):
            continue
        keep.append(idx); a1.append(w1); a5.append(w5); a15.append(w15)

    frame = candidates.loc[keep].reset_index(drop=True)
    if frame.empty:
        raise ValueError('NO_VALID_SHORT_MEMORY_SEQUENCES')
    static_num = frame[STATIC_NUMERIC].astype(float).to_numpy(np.float32)
    static_cat = frame[STATIC_CATEGORICAL].astype(str).copy()
    y1 = target_s1(frame)
    y2 = target_s2(frame)
    valid_target = np.isfinite(y1) & np.isfinite(y2) & np.isfinite(static_num).all(axis=1)
    frame = frame.loc[valid_target].reset_index(drop=True)
    return TensorBundle(
        frame=frame,
        m1=np.stack(a1)[valid_target],
        m5=np.stack(a5)[valid_target],
        m15=np.stack(a15)[valid_target],
        static_num=static_num[valid_target],
        static_cat=static_cat.loc[valid_target].reset_index(drop=True),
        y_s1=y1[valid_target],
        y_s2=y2[valid_target],
    )
