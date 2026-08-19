"""Fold-local preprocessing parity with original Phase 3B2.

Numeric scalers and categorical vocabularies are fit from the training fold only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .dataset import TensorBundle
from .training import STATIC_CATEGORICAL


def _fit_seq_scaler(values: np.ndarray, train_idx: np.ndarray) -> tuple[StandardScaler, np.ndarray]:
    train_flat = values[train_idx].reshape(-1, values.shape[-1])
    med = np.nanmedian(train_flat, axis=0)
    train_filled = np.where(np.isnan(train_flat), med, train_flat)
    scaler = StandardScaler().fit(train_filled)
    all_flat = values.reshape(-1, values.shape[-1])
    all_filled = np.where(np.isnan(all_flat), med, all_flat)
    transformed = scaler.transform(all_filled).reshape(values.shape).astype(np.float32)
    return scaler, transformed


def _categorical_train_only(train: pd.DataFrame, all_values: pd.DataFrame) -> tuple[np.ndarray, dict[str, list[str]]]:
    pieces: list[np.ndarray] = []
    vocab: dict[str, list[str]] = {}
    for col in STATIC_CATEGORICAL:
        train_col = train[col].astype(str).fillna('<NA>')
        categories = sorted(train_col.unique().tolist())
        vocab[col] = categories
        all_col = all_values[col].astype(str).fillna('<NA>')
        for category in categories:
            pieces.append((all_col == category).astype(np.float32).to_numpy()[:, None])
    if not pieces:
        return np.empty((len(all_values), 0), dtype=np.float32), vocab
    return np.hstack(pieces).astype(np.float32), vocab


def preprocess_fold(bundle: TensorBundle, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    s1, m1 = _fit_seq_scaler(bundle.m1, train_idx)
    s5, m5 = _fit_seq_scaler(bundle.m5, train_idx)
    s15, m15 = _fit_seq_scaler(bundle.m15, train_idx)

    train_static = bundle.static_num[train_idx]
    med = np.nanmedian(train_static, axis=0)
    static_train = np.where(np.isnan(train_static), med, train_static)
    static_all = np.where(np.isnan(bundle.static_num), med, bundle.static_num)
    static_scaler = StandardScaler().fit(static_train)
    static_numeric = static_scaler.transform(static_all).astype(np.float32)

    static_cat, vocab = _categorical_train_only(bundle.static_cat.iloc[train_idx], bundle.static_cat)
    static = np.hstack([static_numeric, static_cat]).astype(np.float32)
    return m1, m5, m15, static, {
        'm1': s1, 'm5': s5, 'm15': s15,
        'static_numeric': static_scaler,
        'categories': vocab,
    }
