from __future__ import annotations

import argparse
from pathlib import Path
import sys

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

FIXED_LOT = 0.01
MAX_POSITIONS = 3
RATE_TARGETS = (10, 20, 30)
EVAL_START = pd.Timestamp("2026-04-01T00:00:00Z")
EVAL_END = pd.Timestamp("2026-08-14T00:00:00Z")


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)


def max_drawdown(values: pd.Series, initial_balance: float) -> tuple[float, float]:
    equity = initial_balance + values.cumsum()
    peak = equity.cummax()
    dd = peak - equity
    dd_usd = float(dd.max()) if len(dd) else 0.0
    denom = peak.where(peak > 0)
    dd_pct = float(((dd / denom) * 100.0).max()) if len(dd) else 0.0
    return dd_usd, dd_pct


def summarize(trades: pd.DataFrame, initial_balance: float, accepted: int, blocked: int,
              ambiguous: int, market_days: int, desired: int) -> dict[str, object]:
    if trades.empty:
        return {
            "desired_signals_per_day": desired, "ml_accepted": accepted, "trades": 0,
            "trades_per_market_day": 0.0, "wins": 0, "losses": 0, "win_rate_pct": 0.0,
            "gross_profit_usd": 0.0, "gross_loss_usd": 0.0, "net_profit_usd": 0.0,
            "profit_factor": 0.0, "expectancy_usd": 0.0, "max_drawdown_usd": 0.0,
            "max_drawdown_pct": 0.0, "blocked_max_positions": blocked,
            "ambiguous_intrabar": ambiguous, "market_days": market_days,
        }
    pnl = pd.to_numeric(trades["net_pnl"], errors="coerce").fillna(0.0)
    wins = int((pnl > 0).sum()); losses = int((pnl < 0).sum())
    gp = float(pnl[pnl > 0].sum()); gl = float(-pnl[pnl < 0].sum())
    dd_usd, dd_pct = max_drawdown(pnl, initial_balance)
    return {
        "desired_signals_per_day": desired, "ml_accepted": accepted, "trades": int(len(trades)),
        "trades_per_market_day": float(len(trades) / market_days) if market_days else 0.0,
        "wins": wins, "losses": losses, "win_rate_pct": float(wins / len(trades) * 100.0),
        "gross_profit_usd": gp, "gross_loss_usd": gl, "net_profit_usd": float(pnl.sum()),
        "profit_factor": float(gp / gl) if gl > 0 else float("inf"),
        "expectancy_usd": float(pnl.mean()), "max_drawdown_usd": dd_usd,
        "max_drawdown_pct": dd_pct, "blocked_max_positions": blocked,
        "ambiguous_intrabar": ambiguous, "market_days": market_days,
    }


def block_name(ts: pd.Series) -> pd.Series:
    out = ts.dt.strftime("%Y-%m")
    out = out.astype(object)
    out.loc[ts >= pd.Timestamp("2026-08-01T00:00:00Z")] = "2026-08-01_13"
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Causal economic replay of 6/1/1 V2 10/20/30 rate screens")
    ap.add_argument("--top40-root", type=Path, default=Path(r"D:\ORO_SHORT_MEMORY_ALLSESSIONS_FINAL"))
    ap.add_argument("--source-data", type=Path, default=Path(r"D:\ORO_532_MAR13AUG_2026"))
    ap.add_argument("--root611", type=Path, default=Path(r"D:\ORO_611_MAR13AUG_2026"))
    ap.add_argument("--out", type=Path, default=Path(r"D:\ORO_611_MAR13AUG_2026\v2_economic_replay"))
    ap.add_argument("--initial-balance", type=float, default=1000.0)
    ap.add_argument("--commission-per-lot", type=float, default=0.0)
    ap.add_argument("--slippage-points", type=float, default=0.0)
    a = ap.parse_args()

    top = a.top40_root.resolve(); sys.path.insert(0, str(top))
    from xau_bot.config.settings import BotConfig
    from xau_bot.core.models import Confirmation, Direction, RiskDecision, SymbolSpec, TechnicalSignal
    from xau_bot.research.backtest_models import BacktestSettings
    from xau_bot.research.simulated_executor import SimulatedExecutor

    pred_path = a.root611 / "v2_classifier_oos" / "oos_predictions_611_v2.parquet"
    rate_path = a.root611 / "v2_classifier_oos" / "causal_rate_screen_611_v2.csv"
    m1_path = a.source_data / "XAUUSD_M1_FRESH_MAR13AUG_2026.csv"
    for p in (pred_path, rate_path, m1_path):
        if not p.exists(): raise SystemExit(f"MISSING_INPUT: {p}")

    pred = pd.read_parquet(pred_path).copy()
    pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True)
    if "candidate_entry_time" in pred.columns:
        pred["candidate_entry_time"] = pd.to_datetime(pred["candidate_entry_time"], utc=True)
    else:
        pred["candidate_entry_time"] = pred["timestamp"]
    rates = pd.read_csv(rate_path)
    m1 = load_csv(m1_path)

    if (pred["timestamp"] >= EVAL_END).any():
        raise RuntimeError("AUG14_PRESENT_IN_V2_OOS_PREDICTIONS")

    if not mt5.initialize(): raise SystemExit(f"MT5_INIT_FAILED: {mt5.last_error()}")
    try:
        info = mt5.symbol_info("XAUUSD")
        if info is None: raise SystemExit("MT5_SYMBOL_INFO_FAILED")
        spec = SymbolSpec(
            name="XAUUSD", point=float(info.point), digits=int(info.digits),
            tick_size=float(info.trade_tick_size), tick_value=float(info.trade_tick_value),
            contract_size=float(info.trade_contract_size), volume_min=float(info.volume_min),
            volume_max=float(info.volume_max), volume_step=float(info.volume_step),
            trade_stops_level=int(info.trade_stops_level), filling_mode=int(info.filling_mode),
        )
    finally:
        mt5.shutdown()

    cfg = BotConfig()
    bars = m1[(m1["time"] >= EVAL_START) & (m1["time"] < EVAL_END)].copy()
    market_days = int(bars["time"].dt.date.nunique())
    a.out.mkdir(parents=True, exist_ok=True)

    # Causal ATR known at each M1 decision, used both for SL sizing and shared trailing core.
    atr_source = pred[["timestamp", "atr14_m5"]].dropna().sort_values("timestamp").drop_duplicates("timestamp", keep="last").copy()
    atr_source["atr_points"] = pd.to_numeric(atr_source["atr14_m5"], errors="coerce") / spec.point
    bars = pd.merge_asof(bars.sort_values("time"), atr_source[["timestamp", "atr_points"]],
                         left_on="time", right_on="timestamp", direction="backward")
    bars["atr_points"] = bars["atr_points"].ffill().fillna(0.0)

    overall_rows = []
    block_rows = []

    print("=" * 78)
    print("6/1/1 V2 CAUSAL ECONOMIC REPLAY - 10 / 20 / 30 RATE TARGETS")
    print(f"lot={FIXED_LOT:.2f} | max_positions={MAX_POSITIONS} | cooldown=OFF | sessions=ALL")
    print(f"SL=clamp(1.5 x M5 ATR, {cfg.risk.sl_min_points}-{cfg.risk.sl_max_points} pts)")
    print(f"TP={cfg.risk.tp_risk_reward:.2f}R | BE={cfg.risk.breakeven_at_r:.2f}R | trailing={cfg.risk.trailing_activation_r:.2f}R")
    print("ENTRY=accepted closed M1 decision -> next M1 open")
    print("RATE_THRESHOLDS=already calibrated on same-fold TRAIN predictions only")
    print("AUG14_USED=NO")
    print("=" * 78)

    for desired in RATE_TARGETS:
        rr = rates[pd.to_numeric(rates["desired_signals_per_day"], errors="coerce").eq(desired)].copy()
        if rr.empty: raise RuntimeError(f"NO_RATE_ROWS_{desired}")
        threshold_map = {str(r["block"]): float(r["train_calibrated_probability_threshold"]) for _, r in rr.iterrows()}

        candidates = []
        for block, thr in threshold_map.items():
            g = pred[pred["validation_block"].astype(str).eq(block)].copy()
            g = g[pd.to_numeric(g["good_probability"], errors="coerce") >= thr].copy()
            candidates.append(g)
        accepted = pd.concat(candidates, ignore_index=True) if candidates else pd.DataFrame()
        accepted = accepted[(accepted["timestamp"] >= EVAL_START) & (accepted["timestamp"] < EVAL_END)].copy()
        accepted = accepted.sort_values(["candidate_entry_time", "timestamp"]).reset_index(drop=True)

        by_time = {ts: g for ts, g in accepted.groupby("candidate_entry_time", sort=False)}
        settings = BacktestSettings(initial_balance=a.initial_balance, commission_per_lot=a.commission_per_lot,
                                    slippage_points=a.slippage_points, run_name=f"611_V2_RATE_{desired}")
        executor = SimulatedExecutor(spec, cfg.risk, settings)
        blocked = 0

        for _, bar in bars.iterrows():
            t = bar["time"]
            pending = by_time.get(t)
            if pending is not None:
                for _, c in pending.iterrows():
                    if len(executor.positions) >= MAX_POSITIONS:
                        blocked += 1; continue
                    direction = Direction(str(c["direction"]))
                    atr_pts = float(c.get("atr14_m5", np.nan)) / spec.point
                    if not np.isfinite(atr_pts) or atr_pts <= 0: continue
                    sl_points = float(np.clip(cfg.risk.sl_atr_multiplier * atr_pts,
                                              cfg.risk.sl_min_points, cfg.risk.sl_max_points))
                    value_per_point = spec.tick_value / spec.tick_size * spec.point * FIXED_LOT
                    risk_amount = sl_points * value_per_point
                    risk = RiskDecision(
                        approved=True, lot=FIXED_LOT, sl_points=sl_points,
                        reason_code=f"611_V2_RATE_{desired}_CAUSAL_ACCEPTED",
                        risk_amount=risk_amount,
                        risk_pct=(risk_amount / a.initial_balance * 100.0 if a.initial_balance else 0.0),
                    )
                    signal = TechnicalSignal(
                        direction=direction, setup="M1_TIMING_611_V2", score=0,
                        confirmations=[Confirmation("611_v2_classifier", True, 1), Confirmation("causal_rate_threshold", True, 1)],
                    )
                    executor.open_position(
                        risk=risk, signal=signal, direction=direction,
                        signal_time=pd.Timestamp(c["timestamp"]).to_pydatetime(), entry_bar=bar,
                        session=str(c.get("session", "")), regime="611_V2",
                    )
            executor.process_bar(bar, atr_points=float(bar.get("atr_points", 0.0)))

        if executor.positions: executor.close_all_end(bars.iloc[-1])
        trades = pd.DataFrame([x.to_dict() for x in executor.closed_trades])
        trades.to_csv(a.out / f"trades_rate_{desired}.csv", index=False)

        overall = summarize(trades, a.initial_balance, len(accepted), blocked,
                            executor.ambiguous_intrabar_count, market_days, desired)
        overall_rows.append(overall)
        print(f"RATE={desired:02d}/day accepted={len(accepted):,} trades={overall['trades']:,} "
              f"actual/day={overall['trades_per_market_day']:.2f} WR={overall['win_rate_pct']:.2f}% "
              f"PF={overall['profit_factor']:.4f} exp={overall['expectancy_usd']:.4f} "
              f"net={overall['net_profit_usd']:.2f} DD={overall['max_drawdown_pct']:.2f}% blocked={blocked}")

        if not trades.empty:
            trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
            trades["block"] = block_name(trades["entry_time"])
            accepted["block"] = accepted["validation_block"].astype(str)
            for block, g in trades.groupby("block", sort=True):
                if block == "2026-08-01_13": b0, b1 = pd.Timestamp("2026-08-01T00:00:00Z"), EVAL_END
                else:
                    b0 = pd.Timestamp(f"{block}-01T00:00:00Z"); b1 = b0 + pd.offsets.MonthBegin(1)
                days = int(bars[(bars["time"] >= b0) & (bars["time"] < b1)]["time"].dt.date.nunique())
                acc_n = int((accepted["block"] == block).sum())
                row = summarize(g, a.initial_balance, acc_n, 0, 0, days, desired)
                row["block"] = block
                block_rows.append(row)

    overall_df = pd.DataFrame(overall_rows)
    block_df = pd.DataFrame(block_rows)
    overall_df.to_csv(a.out / "FINAL_611_V2_ECONOMIC_REPLAY.csv", index=False)
    block_df.to_csv(a.out / "BY_BLOCK_611_V2_ECONOMIC_REPLAY.csv", index=False)

    print("\nFINAL 6/1/1 V2 CAUSAL ECONOMIC REPLAY")
    print(overall_df.to_string(index=False))
    if not block_df.empty:
        cols = ["desired_signals_per_day", "block", "ml_accepted", "trades", "trades_per_market_day",
                "win_rate_pct", "profit_factor", "expectancy_usd", "net_profit_usd", "max_drawdown_pct"]
        print("\nBY_BLOCK")
        print(block_df[cols].to_string(index=False))
    print("AUG14_USED_FOR_REPLAY=NO")
    print("DEPLOYMENT_THRESHOLD_SELECTED=NO")
    print(f"OUTPUT={a.out}")


if __name__ == "__main__":
    main()
