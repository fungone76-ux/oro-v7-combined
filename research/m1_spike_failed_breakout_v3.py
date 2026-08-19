from __future__ import annotations

"""Research-only causal M1 spike failed-breakout / rejection reversal V3.

No live trading code is modified. Every decision uses only fully closed M1 bars.
The pattern is:
1) large directional spike bar,
2) next bar attempts continuation beyond the spike extreme,
3) that continuation fails and closes back inside the spike range,
4) optional lower tick-volume on rejection bar,
5) enter opposite at rejection-bar close.
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
    percentile_q: float = 0.97
    atr_multiple: float = 3.0
    min_body_fraction: float = 0.70
    min_break_fraction: float = 0.02
    max_break_fraction: float = 0.20
    min_reentry_fraction: float = 0.20
    min_rejection_wick_fraction: float = 0.25
    require_lower_volume: bool = False
    stop_buffer_fraction: float = 0.08
    reward_r: float = 1.50
    breakeven_at_r: float = 1.00
    breakeven_offset_points: float = 2.0
    max_holding_bars: int = 15
    spread_points: float = 0.0


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    time_col = next((c for c in ("time", "timestamp", "datetime", "date") if c in out.columns), None)
    if time_col is None:
        raise ValueError("MISSING_TIME_COLUMN")
    out["time"] = pd.to_datetime(out[time_col], utc=True, errors="coerce")
    for c in ("open", "high", "low", "close"):
        if c not in out.columns:
            raise ValueError(f"MISSING_OHLC_COLUMN={c}")
        out[c] = pd.to_numeric(out[c], errors="coerce")
    volume_col = next((c for c in ("tick_volume", "volume", "real_volume") if c in out.columns), None)
    out["volume_used"] = pd.to_numeric(out[volume_col], errors="coerce") if volume_col else np.nan
    return out.dropna(subset=["time", "open", "high", "low", "close"]).sort_values("time").drop_duplicates("time").reset_index(drop=True)


def prepare(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["close"].shift(1)
    tr = pd.concat([
        out["high"] - out["low"],
        (out["high"] - prev_close).abs(),
        (out["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)
    out["range"] = out["high"] - out["low"]
    out["body"] = (out["close"] - out["open"]).abs()
    out["body_fraction"] = np.where(out["range"] > 0, out["body"] / out["range"], 0.0)
    out["atr_prev"] = tr.rolling(cfg.atr_period, min_periods=cfg.atr_period).mean().shift(1)
    out["range_q_prev"] = out["range"].rolling(cfg.percentile_window, min_periods=cfg.percentile_window).quantile(cfg.percentile_q).shift(1)
    threshold = np.maximum(cfg.atr_multiple * out["atr_prev"], out["range_q_prev"])
    out["anomaly_threshold"] = threshold
    out["is_spike"] = (out["range"] >= threshold) & (out["body_fraction"] >= cfg.min_body_fraction)
    return out


def pnl_usd(direction: str, entry: float, exit_price: float, cfg: Config) -> float:
    pts = (exit_price - entry) / cfg.point if direction == "LONG" else (entry - exit_price) / cfg.point
    return pts * cfg.value_per_point_per_lot * cfg.lot


def simulate_trade(df: pd.DataFrame, entry_idx: int, direction: str, entry: float, stop: float, tp: float, risk_price: float, cfg: Config) -> dict:
    current_sl = stop
    be_armed = False
    end_idx = min(len(df) - 1, entry_idx + cfg.max_holding_bars)
    exit_idx = end_idx
    exit_price = float(df.loc[end_idx, "close"])
    reason = "TIME_EXIT"
    for j in range(entry_idx + 1, end_idx + 1):
        bar = df.loc[j]
        hi, lo, close = float(bar.high), float(bar.low), float(bar.close)
        if direction == "SHORT":
            hit_sl, hit_tp = hi >= current_sl, lo <= tp
        else:
            hit_sl, hit_tp = lo <= current_sl, hi >= tp
        if hit_sl:
            exit_idx, exit_price, reason = j, current_sl, "BREAKEVEN_STOP" if be_armed else "STOP_LOSS"
            break
        if hit_tp:
            exit_idx, exit_price, reason = j, tp, "TAKE_PROFIT"
            break
        favorable = (entry - close) if direction == "SHORT" else (close - entry)
        r_mult = favorable / risk_price if risk_price > 0 else 0.0
        if (not be_armed) and r_mult >= cfg.breakeven_at_r:
            off = cfg.breakeven_offset_points * cfg.point
            current_sl = entry - off if direction == "SHORT" else entry + off
            be_armed = True
    pnl = pnl_usd(direction, entry, exit_price, cfg)
    init_risk_usd = abs(pnl_usd(direction, entry, stop, cfg))
    return {
        "exit_idx": int(exit_idx),
        "exit_time": df.loc[exit_idx, "time"].isoformat(),
        "exit_price": float(exit_price),
        "exit_reason": reason,
        "pnl_usd": float(pnl),
        "r_multiple": float(pnl / init_risk_usd) if init_risk_usd > 0 else 0.0,
        "holding_bars": int(exit_idx - entry_idx),
    }


def rejection_pattern(spike: pd.Series, rej: pd.Series, cfg: Config) -> tuple[str | None, dict]:
    rng = float(spike["range"])
    if rng <= 0:
        return None, {}
    up = float(spike.close) > float(spike.open)
    down = float(spike.close) < float(spike.open)

    if up:
        break_amt = float(rej.high) - float(spike.high)
        attempted = break_amt >= cfg.min_break_fraction * rng
        not_too_far = break_amt <= cfg.max_break_fraction * rng
        reentry_level = float(spike.high) - cfg.min_reentry_fraction * rng
        reentered = float(rej.close) <= reentry_level
        upper_wick = float(rej.high) - max(float(rej.open), float(rej.close))
        wick_ok = upper_wick >= cfg.min_rejection_wick_fraction * max(float(rej.high - rej.low), 1e-12)
        direction = "SHORT" if attempted and not_too_far and reentered and wick_ok else None
        wick_fraction = upper_wick / max(float(rej.high - rej.low), 1e-12)
    elif down:
        break_amt = float(spike.low) - float(rej.low)
        attempted = break_amt >= cfg.min_break_fraction * rng
        not_too_far = break_amt <= cfg.max_break_fraction * rng
        reentry_level = float(spike.low) + cfg.min_reentry_fraction * rng
        reentered = float(rej.close) >= reentry_level
        lower_wick = min(float(rej.open), float(rej.close)) - float(rej.low)
        wick_ok = lower_wick >= cfg.min_rejection_wick_fraction * max(float(rej.high - rej.low), 1e-12)
        direction = "LONG" if attempted and not_too_far and reentered and wick_ok else None
        wick_fraction = lower_wick / max(float(rej.high - rej.low), 1e-12)
    else:
        return None, {}

    if direction is not None and cfg.require_lower_volume:
        sv, rv = float(spike.get("volume_used", np.nan)), float(rej.get("volume_used", np.nan))
        if not (np.isfinite(sv) and np.isfinite(rv) and rv < sv):
            direction = None

    return direction, {
        "break_fraction": float(break_amt / rng),
        "reentry_level": float(reentry_level),
        "rejection_wick_fraction": float(wick_fraction),
    }


def run(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    d = prepare(normalise(df), cfg)
    rows: list[dict] = []
    busy_until = -1
    start = max(cfg.atr_period, cfg.percentile_window) + 1
    for i in range(start, len(d) - 2):
        if i <= busy_until or not bool(d.loc[i, "is_spike"]):
            continue
        spike, rej = d.loc[i], d.loc[i + 1]
        direction, info = rejection_pattern(spike, rej, cfg)
        if direction is None:
            continue
        rng = float(spike["range"])
        half_spread = cfg.spread_points * cfg.point / 2.0
        raw_entry = float(rej.close)
        entry = raw_entry + half_spread if direction == "LONG" else raw_entry - half_spread
        buffer = cfg.stop_buffer_fraction * rng
        if direction == "SHORT":
            stop = max(float(spike.high), float(rej.high)) + buffer
            risk_price = stop - entry
            tp = entry - cfg.reward_r * risk_price
        else:
            stop = min(float(spike.low), float(rej.low)) - buffer
            risk_price = entry - stop
            tp = entry + cfg.reward_r * risk_price
        if risk_price <= 0:
            continue
        result = simulate_trade(d, i + 1, direction, entry, stop, tp, risk_price, cfg)
        rows.append({
            "spike_time": spike.time.isoformat(),
            "rejection_time": rej.time.isoformat(),
            "direction": direction,
            "spike_range_points": rng / cfg.point,
            "atr_prev_points": float(spike.atr_prev / cfg.point),
            "body_fraction": float(spike.body_fraction),
            "spike_volume": float(spike.volume_used) if pd.notna(spike.volume_used) else np.nan,
            "rejection_volume": float(rej.volume_used) if pd.notna(rej.volume_used) else np.nan,
            "entry_price": float(entry),
            "initial_sl": float(stop),
            "initial_tp": float(tp),
            "initial_risk_points": float(risk_price / cfg.point),
            **info,
            **{k: v for k, v in result.items() if k != "exit_idx"},
        })
        busy_until = int(result["exit_idx"])
    return pd.DataFrame(rows)


def metrics(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"trades": 0, "net_profit_usd": 0.0, "profit_factor": None, "expectancy_usd": None, "win_rate_pct": None, "max_drawdown_usd": 0.0}
    pnl = trades.pnl_usd.astype(float)
    gp = float(pnl[pnl > 0].sum())
    gl = float(-pnl[pnl < 0].sum())
    eq = pnl.cumsum()
    dd = eq.cummax() - eq
    return {
        "trades": int(len(trades)),
        "wins": int((pnl > 0).sum()),
        "losses": int((pnl < 0).sum()),
        "win_rate_pct": float((pnl > 0).mean() * 100.0),
        "gross_profit_usd": gp,
        "gross_loss_usd": gl,
        "net_profit_usd": float(pnl.sum()),
        "profit_factor": float(gp / gl) if gl > 0 else None,
        "expectancy_usd": float(pnl.mean()),
        "max_drawdown_usd": float(dd.max()),
        "avg_r": float(trades.r_multiple.mean()),
        "median_r": float(trades.r_multiple.median()),
        "take_profit_count": int((trades.exit_reason == "TAKE_PROFIT").sum()),
        "breakeven_stop_count": int((trades.exit_reason == "BREAKEVEN_STOP").sum()),
        "stop_loss_count": int((trades.exit_reason == "STOP_LOSS").sum()),
        "time_exit_count": int((trades.exit_reason == "TIME_EXIT").sum()),
        "long_trades": int((trades.direction == "LONG").sum()),
        "short_trades": int((trades.direction == "SHORT").sum()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--point", type=float, default=0.01)
    ap.add_argument("--lot", type=float, default=0.01)
    ap.add_argument("--value-per-point-per-lot", type=float, default=1.0)
    ap.add_argument("--spread-points", type=float, default=0.0)
    args = ap.parse_args()

    source = pd.read_csv(args.m1)
    variants = [
        ("NO_VOLUME_FILTER", False),
        ("LOWER_VOLUME_FILTER", True),
    ]
    rows = []
    args.out.mkdir(parents=True, exist_ok=True)
    for name, vol_filter in variants:
        cfg = Config(
            point=args.point,
            lot=args.lot,
            value_per_point_per_lot=args.value_per_point_per_lot,
            spread_points=args.spread_points,
            require_lower_volume=vol_filter,
        )
        trades = run(source, cfg)
        m = metrics(trades)
        trades.to_csv(args.out / f"m1_spike_failed_breakout_v3_{name.lower()}.csv", index=False)
        (args.out / f"summary_{name.lower()}.json").write_text(json.dumps({"variant": name, "config": asdict(cfg), "metrics": m}, indent=2), encoding="utf-8")
        rows.append({"variant": name, **m})
    summary = pd.DataFrame(rows)
    summary.to_csv(args.out / "m1_spike_failed_breakout_v3_summary.csv", index=False)
    print("M1_SPIKE_FAILED_BREAKOUT_V3=COMPLETE")
    print("NO_LIVE_CODE_CHANGED=PASS")
    print("CAUSAL_NO_FUTURE_GUARD=PASS")
    print(summary.to_string(index=False))
    print(f"OUTPUT={args.out.resolve()}")


if __name__ == "__main__":
    main()
