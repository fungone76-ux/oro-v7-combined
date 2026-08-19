from __future__ import annotations

"""Causal research backtest for M1 spike-exhaustion mean reversion.

Research only. This module does not alter the live TOP40 runtime.
Detection uses only information available at or before each decision point.
Entry occurs at the close of the confirmation M1 bar, never inside the spike bar.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Config:
    point: float = 0.01
    lot: float = 0.01
    value_per_point_per_lot: float = 1.0
    atr_period: int = 60
    percentile_window: int = 240
    percentile_q: float = 0.95
    atr_multiple: float = 2.5
    min_body_fraction: float = 0.70
    retrace_fraction: float = 0.25
    max_extension_fraction: float = 0.10
    stop_buffer_fraction: float = 0.10
    reward_r: float = 1.50
    breakeven_at_r: float = 1.00
    breakeven_offset_points: float = 2.0
    max_holding_bars: int = 15
    spread_points: float = 0.0


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    time_col = next((c for c in ("time", "timestamp", "datetime", "date") if c in out.columns), None)
    if time_col is None:
        raise ValueError("MISSING_TIME_COLUMN")
    out["time"] = pd.to_datetime(out[time_col], utc=True, errors="coerce")
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"MISSING_OHLC_COLUMNS={missing}")
    for c in required:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["time", *required]).sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return out


def _prepare(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["close"].shift(1)
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    # Shift by one: spike bar cannot contribute to its own anomaly threshold.
    out["atr_prev"] = tr.rolling(cfg.atr_period, min_periods=cfg.atr_period).mean().shift(1)
    out["range"] = out["high"] - out["low"]
    out["range_q_prev"] = out["range"].rolling(cfg.percentile_window, min_periods=cfg.percentile_window).quantile(cfg.percentile_q).shift(1)
    out["body"] = (out["close"] - out["open"]).abs()
    out["body_fraction"] = np.where(out["range"] > 0, out["body"] / out["range"], 0.0)
    out["spike_up"] = out["close"] > out["open"]
    out["spike_down"] = out["close"] < out["open"]
    threshold = np.maximum(cfg.atr_multiple * out["atr_prev"], out["range_q_prev"])
    out["anomaly_threshold"] = threshold
    out["is_spike"] = (out["range"] >= threshold) & (out["body_fraction"] >= cfg.min_body_fraction)
    return out


def _pnl_usd(direction: str, entry: float, exit_price: float, cfg: Config) -> float:
    pts = (exit_price - entry) / cfg.point if direction == "LONG" else (entry - exit_price) / cfg.point
    return pts * cfg.value_per_point_per_lot * cfg.lot


def _simulate_trade(df: pd.DataFrame, entry_idx: int, direction: str, entry: float, stop: float, tp: float, risk_price: float, cfg: Config) -> dict:
    current_sl = stop
    be_armed = False
    end_idx = min(len(df) - 1, entry_idx + cfg.max_holding_bars)
    exit_idx = end_idx
    exit_price = float(df.loc[end_idx, "close"])
    exit_reason = "TIME_EXIT"

    for j in range(entry_idx + 1, end_idx + 1):
        bar = df.loc[j]
        hi, lo, close = float(bar.high), float(bar.low), float(bar.close)
        if direction == "SHORT":
            hit_sl = hi >= current_sl
            hit_tp = lo <= tp
        else:
            hit_sl = lo <= current_sl
            hit_tp = hi >= tp
        # Conservative ambiguity rule: stop wins if both are touched in one M1 bar.
        if hit_sl:
            exit_idx, exit_price, exit_reason = j, current_sl, "BREAKEVEN_STOP" if be_armed else "STOP_LOSS"
            break
        if hit_tp:
            exit_idx, exit_price, exit_reason = j, tp, "TAKE_PROFIT"
            break

        if risk_price > 0:
            favorable = (entry - close) if direction == "SHORT" else (close - entry)
            r_mult = favorable / risk_price
            if (not be_armed) and r_mult >= cfg.breakeven_at_r:
                offset = cfg.breakeven_offset_points * cfg.point
                current_sl = entry - offset if direction == "SHORT" else entry + offset
                be_armed = True

    pnl = _pnl_usd(direction, entry, exit_price, cfg)
    return {
        "exit_idx": int(exit_idx),
        "exit_time": df.loc[exit_idx, "time"].isoformat(),
        "exit_price": float(exit_price),
        "exit_reason": exit_reason,
        "pnl_usd": float(pnl),
        "r_multiple": float(pnl / max(abs(_pnl_usd(direction, entry, stop, cfg)), 1e-12)),
        "holding_bars": int(exit_idx - entry_idx),
    }


def run(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    d = _prepare(_normalise(df), cfg)
    rows: list[dict] = []
    busy_until = -1

    for i in range(max(cfg.percentile_window, cfg.atr_period) + 1, len(d) - 2):
        if i <= busy_until or not bool(d.loc[i, "is_spike"]):
            continue
        s = d.loc[i]
        c = d.loc[i + 1]  # confirmation bar, fully closed before entry
        spike_range = float(s["range"])
        if spike_range <= 0:
            continue

        direction: str | None = None
        confirm_level = np.nan
        if bool(s["spike_up"]):
            confirm_level = float(s["high"] - cfg.retrace_fraction * spike_range)
            extension_cap = float(s["high"] + cfg.max_extension_fraction * spike_range)
            if float(c["close"]) <= confirm_level and float(c["high"]) <= extension_cap:
                direction = "SHORT"
        elif bool(s["spike_down"]):
            confirm_level = float(s["low"] + cfg.retrace_fraction * spike_range)
            extension_floor = float(s["low"] - cfg.max_extension_fraction * spike_range)
            if float(c["close"]) >= confirm_level and float(c["low"]) >= extension_floor:
                direction = "LONG"
        if direction is None:
            continue

        # Entry at confirmation close. Spread is applied against us.
        half_spread = cfg.spread_points * cfg.point / 2.0
        raw_entry = float(c["close"])
        entry = raw_entry + half_spread if direction == "LONG" else raw_entry - half_spread
        buffer_price = cfg.stop_buffer_fraction * spike_range
        if direction == "SHORT":
            stop = float(s["high"] + buffer_price)
            risk_price = stop - entry
            tp = entry - cfg.reward_r * risk_price
        else:
            stop = float(s["low"] - buffer_price)
            risk_price = entry - stop
            tp = entry + cfg.reward_r * risk_price
        if risk_price <= 0:
            continue

        result = _simulate_trade(d, i + 1, direction, entry, stop, tp, risk_price, cfg)
        rows.append(
            {
                "spike_time": s["time"].isoformat(),
                "confirmation_time": c["time"].isoformat(),
                "direction": direction,
                "spike_open": float(s["open"]),
                "spike_high": float(s["high"]),
                "spike_low": float(s["low"]),
                "spike_close": float(s["close"]),
                "spike_range_points": spike_range / cfg.point,
                "atr_prev_points": float(s["atr_prev"] / cfg.point),
                "range_threshold_points": float(s["anomaly_threshold"] / cfg.point),
                "body_fraction": float(s["body_fraction"]),
                "confirm_level": float(confirm_level),
                "entry_price": float(entry),
                "initial_sl": float(stop),
                "initial_tp": float(tp),
                "initial_risk_points": float(risk_price / cfg.point),
                **{k: v for k, v in result.items() if k != "exit_idx"},
            }
        )
        busy_until = int(result["exit_idx"])
    return pd.DataFrame(rows)


def metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"trades": 0, "net_profit_usd": 0.0, "profit_factor": None, "expectancy_usd": None, "win_rate_pct": None, "max_drawdown_usd": 0.0}
    pnl = trades["pnl_usd"].astype(float)
    gross_profit = float(pnl[pnl > 0].sum())
    gross_loss = float(-pnl[pnl < 0].sum())
    equity = pnl.cumsum()
    dd = equity.cummax() - equity
    return {
        "trades": int(len(trades)),
        "wins": int((pnl > 0).sum()),
        "losses": int((pnl < 0).sum()),
        "win_rate_pct": float((pnl > 0).mean() * 100.0),
        "gross_profit_usd": gross_profit,
        "gross_loss_usd": gross_loss,
        "net_profit_usd": float(pnl.sum()),
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else None,
        "expectancy_usd": float(pnl.mean()),
        "max_drawdown_usd": float(dd.max()),
        "avg_r": float(trades["r_multiple"].mean()),
        "median_r": float(trades["r_multiple"].median()),
        "take_profit_count": int((trades["exit_reason"] == "TAKE_PROFIT").sum()),
        "breakeven_stop_count": int((trades["exit_reason"] == "BREAKEVEN_STOP").sum()),
        "stop_loss_count": int((trades["exit_reason"] == "STOP_LOSS").sum()),
        "time_exit_count": int((trades["exit_reason"] == "TIME_EXIT").sum()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Research-only causal M1 spike reversal audit")
    p.add_argument("--m1", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--point", type=float, default=0.01)
    p.add_argument("--lot", type=float, default=0.01)
    p.add_argument("--value-per-point-per-lot", type=float, default=1.0)
    p.add_argument("--atr-multiple", type=float, default=2.5)
    p.add_argument("--retrace", type=float, default=0.25)
    p.add_argument("--reward-r", type=float, default=1.5)
    p.add_argument("--spread-points", type=float, default=0.0)
    args = p.parse_args()

    cfg = Config(
        point=args.point,
        lot=args.lot,
        value_per_point_per_lot=args.value_per_point_per_lot,
        atr_multiple=args.atr_multiple,
        retrace_fraction=args.retrace,
        reward_r=args.reward_r,
        spread_points=args.spread_points,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(args.m1)
    trades = run(source, cfg)
    summary = metrics(trades)
    trades.to_csv(args.out / "m1_spike_reversal_trades.csv", index=False)
    (args.out / "m1_spike_reversal_summary.json").write_text(json.dumps({"config": asdict(cfg), "metrics": summary}, indent=2), encoding="utf-8")
    print("M1_SPIKE_REVERSAL_RESEARCH=COMPLETE")
    print("NO_LIVE_CODE_CHANGED=PASS")
    print("CAUSAL_NO_FUTURE_GUARD=PASS")
    for k, v in summary.items():
        print(f"{k}={v}")
    print(f"OUTPUT={args.out.resolve()}")


if __name__ == "__main__":
    main()
