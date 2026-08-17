"""Economic/stability metrics for TOP40_SHORT_MEMORY research."""
from __future__ import annotations

import math
import numpy as np
import pandas as pd


def profit_factor(pnl) -> float:
    v = np.asarray(pnl, dtype=float)
    gp = float(v[v > 0].sum())
    gl = float(-v[v < 0].sum())
    if gl == 0:
        return math.inf if gp > 0 else 0.0
    return gp / gl


def max_drawdown_pct(pnl, initial_balance: float = 1000.0) -> float:
    v = np.asarray(pnl, dtype=float)
    eq = initial_balance + np.cumsum(v)
    if len(eq) == 0:
        return 0.0
    peak = np.maximum.accumulate(np.r_[initial_balance, eq])[:-1]
    dd = np.maximum(0.0, peak - eq)
    return float(np.max(np.where(peak > 0, dd / peak * 100.0, 0.0)))


def losing_streak(pnl) -> int:
    best = cur = 0
    for value in np.asarray(pnl, dtype=float):
        if value < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def trade_metrics(trades: pd.DataFrame, pnl_col: str = "pnl", initial_balance: float = 1000.0) -> dict[str, float | int]:
    if trades.empty:
        return {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0, "pf": 0.0, "net_pnl": 0.0, "expectancy": 0.0, "trades_per_day": 0.0, "max_dd_pct": 0.0, "max_consecutive_losses": 0}
    v = trades[pnl_col].astype(float).to_numpy()
    ts = pd.to_datetime(trades["timestamp"], utc=True)
    days = max(1, ts.dt.date.nunique())
    wins = int((v > 0).sum())
    losses = int((v < 0).sum())
    return {
        "trades": int(len(v)),
        "wins": wins,
        "losses": losses,
        "win_rate": float(wins / len(v) * 100.0),
        "pf": profit_factor(v),
        "net_pnl": float(v.sum()),
        "expectancy": float(v.mean()),
        "trades_per_day": float(len(v) / days),
        "max_dd_pct": max_drawdown_pct(v, initial_balance),
        "max_consecutive_losses": losing_streak(v),
    }


def stability_table(trades: pd.DataFrame, dimension: str, pnl_col: str = "pnl") -> pd.DataFrame:
    rows = []
    for value, g in trades.groupby(dimension, dropna=False):
        rows.append({dimension: value, **trade_metrics(g, pnl_col)})
    return pd.DataFrame(rows)


def compare_original_vs_short(original: dict, short: dict) -> pd.DataFrame:
    keys = ["trades", "trades_per_day", "win_rate", "pf", "net_pnl", "expectancy", "max_dd_pct"]
    rows = []
    for key in keys:
        a = float(original.get(key, np.nan))
        b = float(short.get(key, np.nan))
        pct = ((b - a) / abs(a) * 100.0) if np.isfinite(a) and a != 0 else np.nan
        rows.append({"metric": key, "original": a, "short_memory": b, "delta": b - a, "delta_pct": pct})
    return pd.DataFrame(rows)


def promotion_verdict(metrics: dict, annual: pd.DataFrame | None = None) -> str:
    if metrics.get("pf", 0.0) < 1.0 or metrics.get("expectancy", 0.0) <= 0:
        return "REJECT_SHORT_MEMORY"
    if annual is not None and len(annual) and ((annual["pf"] < 1.0).sum() > max(1, len(annual) // 3)):
        return "UNSTABLE"
    if metrics.get("pf", 0.0) >= 1.30 and metrics.get("max_dd_pct", 100.0) <= 15.0:
        return "SHORT_MEMORY_PROMISING"
    return "PROMISING_BUT_WEAK"
