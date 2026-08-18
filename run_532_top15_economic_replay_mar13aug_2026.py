from __future__ import annotations

import argparse
from pathlib import Path
import sys

import MetaTrader5 as mt5
import pandas as pd

TOP_PCT = 15
FIXED_LOT = 0.01
MAX_POSITIONS = 3
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


def summarize(
    trades: pd.DataFrame,
    initial_balance: float,
    accepted: int,
    blocked: int,
    ambiguous: int,
    market_days: int,
) -> dict[str, object]:
    if trades.empty:
        return {
            "top_pct": TOP_PCT,
            "ml_accepted": accepted,
            "trades": 0,
            "trades_per_market_day": 0.0,
            "wins": 0,
            "losses": 0,
            "win_rate_pct": 0.0,
            "gross_profit_usd": 0.0,
            "gross_loss_usd": 0.0,
            "net_profit_usd": 0.0,
            "profit_factor": 0.0,
            "expectancy_usd": 0.0,
            "max_drawdown_usd": 0.0,
            "max_drawdown_pct": 0.0,
            "blocked_max_positions": blocked,
            "ambiguous_intrabar": ambiguous,
            "market_days": market_days,
        }

    pnl = pd.to_numeric(trades["net_pnl"], errors="coerce").fillna(0.0)
    wins = int((pnl > 0).sum())
    losses = int((pnl < 0).sum())
    gp = float(pnl[pnl > 0].sum())
    gl = float(-pnl[pnl < 0].sum())
    dd_usd, dd_pct = max_drawdown(pnl, initial_balance)
    return {
        "top_pct": TOP_PCT,
        "ml_accepted": accepted,
        "trades": int(len(trades)),
        "trades_per_market_day": float(len(trades) / market_days) if market_days else 0.0,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": float(wins / len(trades) * 100.0),
        "gross_profit_usd": gp,
        "gross_loss_usd": gl,
        "net_profit_usd": float(pnl.sum()),
        "profit_factor": float(gp / gl) if gl > 0 else float("inf"),
        "expectancy_usd": float(pnl.mean()),
        "max_drawdown_usd": dd_usd,
        "max_drawdown_pct": dd_pct,
        "blocked_max_positions": blocked,
        "ambiguous_intrabar": ambiguous,
        "market_days": market_days,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Economic replay of fold-local causal Top15 5/3/2 candidates, Apr-Aug13 2026")
    ap.add_argument("--top40-root", type=Path, default=Path(r"D:\ORO_SHORT_MEMORY_ALLSESSIONS_FINAL"))
    ap.add_argument("--data-dir", type=Path, default=Path(r"D:\ORO_532_MAR13AUG_2026"))
    ap.add_argument("--phase3a", type=Path, default=Path(r"D:\ORO_532_MAR13AUG_2026\phase3a_532"))
    ap.add_argument("--calibration-dir", type=Path, default=Path(r"D:\ORO_532_MAR13AUG_2026\percentile_calibration_532"))
    ap.add_argument("--out", type=Path, default=Path(r"D:\ORO_532_MAR13AUG_2026\economic_replay_top15"))
    ap.add_argument("--initial-balance", type=float, default=1000.0)
    ap.add_argument("--commission-per-lot", type=float, default=0.0)
    ap.add_argument("--slippage-points", type=float, default=0.0)
    a = ap.parse_args()

    top = a.top40_root.resolve()
    sys.path.insert(0, str(top))
    from xau_bot.config.settings import BotConfig
    from xau_bot.core.models import Confirmation, Direction, RiskDecision, SymbolSpec, TechnicalSignal
    from xau_bot.research.backtest_models import BacktestSettings
    from xau_bot.research.simulated_executor import SimulatedExecutor

    m1_path = a.data_dir / "XAUUSD_M1_FRESH_MAR13AUG_2026.csv"
    market_path = a.phase3a / "market_state_mar13aug_2026.parquet"
    accepted_path = a.calibration_dir / "accepted_candidates.parquet"
    for p in (m1_path, market_path, accepted_path):
        if not p.exists():
            raise SystemExit(f"MISSING_INPUT: {p}")

    if not mt5.initialize():
        raise SystemExit(f"MT5_INIT_FAILED: {mt5.last_error()}")
    try:
        info = mt5.symbol_info("XAUUSD")
        if info is None:
            raise SystemExit("MT5_SYMBOL_INFO_FAILED")
        spec = SymbolSpec(
            name="XAUUSD",
            point=float(info.point),
            digits=int(info.digits),
            tick_size=float(info.trade_tick_size),
            tick_value=float(info.trade_tick_value),
            contract_size=float(info.trade_contract_size),
            volume_min=float(info.volume_min),
            volume_max=float(info.volume_max),
            volume_step=float(info.volume_step),
            trade_stops_level=int(info.trade_stops_level),
            filling_mode=int(info.filling_mode),
        )
    finally:
        mt5.shutdown()

    cfg = BotConfig()
    m1 = load_csv(m1_path)
    market = pd.read_parquet(market_path)
    market["timestamp"] = pd.to_datetime(market["timestamp"], utc=True)

    accepted_all = pd.read_parquet(accepted_path)
    accepted_all["timestamp"] = pd.to_datetime(accepted_all["timestamp"], utc=True)
    accepted_all["candidate_entry_time"] = pd.to_datetime(accepted_all["candidate_entry_time"], utc=True)
    accepted = accepted_all[pd.to_numeric(accepted_all["top_pct"], errors="coerce").eq(TOP_PCT)].copy()
    accepted = accepted[(accepted["timestamp"] >= EVAL_START) & (accepted["timestamp"] < EVAL_END)].copy()
    accepted = accepted.sort_values(["candidate_entry_time", "timestamp"]).reset_index(drop=True)
    if accepted.empty:
        raise RuntimeError("NO_TOP15_ACCEPTED_CANDIDATES")
    if (accepted["timestamp"] >= EVAL_END).any():
        raise RuntimeError("AUG14_PRESENT_IN_ACCEPTED_CANDIDATES")

    # ATR points known at decision time, forward-filled to M1 bars for the shared trailing core.
    atr_map = market[["timestamp", "atr14"]].dropna().sort_values("timestamp").copy()
    atr_map["atr_points"] = pd.to_numeric(atr_map["atr14"], errors="coerce") / spec.point
    m1 = pd.merge_asof(
        m1.sort_values("time"),
        atr_map[["timestamp", "atr_points"]],
        left_on="time",
        right_on="timestamp",
        direction="backward",
    )
    m1["atr_points"] = m1["atr_points"].ffill().fillna(0.0)

    bars = m1[(m1["time"] >= EVAL_START) & (m1["time"] < EVAL_END)].copy()
    if bars.empty:
        raise RuntimeError("NO_M1_BARS_FOR_REPLAY")

    market_days = int(bars["time"].dt.date.nunique())
    by_time = {ts: g for ts, g in accepted.groupby("candidate_entry_time", sort=False)}

    settings = BacktestSettings(
        initial_balance=a.initial_balance,
        commission_per_lot=a.commission_per_lot,
        slippage_points=a.slippage_points,
        run_name="TOP40_532_FOLD_LOCAL_TOP15_APR_AUG13",
    )
    executor = SimulatedExecutor(spec, cfg.risk, settings)
    blocked = 0

    print("=" * 78)
    print("TOP40 5/3/2 FOLD-LOCAL TOP15 ECONOMIC REPLAY")
    print(f"accepted={len(accepted):,} | market_days={market_days}")
    print(f"lot={FIXED_LOT:.2f} | max_positions={MAX_POSITIONS} | cooldown=OFF | sessions=ALL")
    print(f"TP={cfg.risk.tp_risk_reward:.2f}R | BE={cfg.risk.breakeven_at_r:.2f}R | trailing={cfg.risk.trailing_activation_r:.2f}R")
    print("ENTRY=accepted M5 candidate -> candidate_entry_time / next M1 open")
    print("POSITION_CORE=xau_bot.research.simulated_executor / shared position_management_core")
    print("AUG14_USED=NO")
    print("=" * 78, flush=True)

    for _, bar in bars.iterrows():
        t = bar["time"]
        pending = by_time.get(t)
        if pending is not None:
            for _, c in pending.iterrows():
                if len(executor.positions) >= MAX_POSITIONS:
                    blocked += 1
                    continue

                direction = Direction(str(c["direction"]))
                sl_points = float(c["candidate_stop_points"])
                value_per_point = spec.tick_value / spec.tick_size * spec.point * FIXED_LOT
                risk_amount = sl_points * value_per_point
                risk = RiskDecision(
                    approved=True,
                    lot=FIXED_LOT,
                    sl_points=sl_points,
                    reason_code="FOLD_LOCAL_TOP15_ACCEPTED",
                    risk_amount=risk_amount,
                    risk_pct=(risk_amount / a.initial_balance * 100.0 if a.initial_balance else 0.0),
                )
                signal = TechnicalSignal(
                    direction=direction,
                    setup=str(c.get("setup", "PULLBACK")),
                    score=int(round(float(c.get("signal_score", 0)))),
                    confirmations=[
                        Confirmation("fold_local_top15", True, 1),
                        Confirmation("technical_candidate", True, 1),
                    ],
                )
                executor.open_position(
                    risk=risk,
                    signal=signal,
                    direction=direction,
                    signal_time=pd.Timestamp(c["timestamp"]).to_pydatetime(),
                    entry_bar=bar,
                    session=str(c.get("session", "")),
                    regime=str(c.get("regime", "")),
                )

        executor.process_bar(bar, atr_points=float(bar.get("atr_points", 0.0)))

    if executor.positions:
        executor.close_all_end(bars.iloc[-1])

    trades = pd.DataFrame([x.to_dict() for x in executor.closed_trades])
    a.out.mkdir(parents=True, exist_ok=True)
    trades.to_csv(a.out / "trades_top15_apr_aug13.csv", index=False)

    overall = summarize(
        trades=trades,
        initial_balance=a.initial_balance,
        accepted=len(accepted),
        blocked=blocked,
        ambiguous=executor.ambiguous_intrabar_count,
        market_days=market_days,
    )

    monthly_rows: list[dict[str, object]] = []
    if not trades.empty:
        trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
        trades["block"] = trades["entry_time"].dt.strftime("%Y-%m")
        trades.loc[trades["entry_time"] >= pd.Timestamp("2026-08-01T00:00:00Z"), "block"] = "2026-08-01_13"

        accepted["block"] = accepted["candidate_entry_time"].dt.strftime("%Y-%m")
        accepted.loc[accepted["candidate_entry_time"] >= pd.Timestamp("2026-08-01T00:00:00Z"), "block"] = "2026-08-01_13"

        for block, g in trades.groupby("block", sort=True):
            if block == "2026-08-01_13":
                b0 = pd.Timestamp("2026-08-01T00:00:00Z")
                b1 = EVAL_END
            else:
                b0 = pd.Timestamp(f"{block}-01T00:00:00Z")
                b1 = b0 + pd.offsets.MonthBegin(1)
            block_days = int(bars[(bars["time"] >= b0) & (bars["time"] < b1)]["time"].dt.date.nunique())
            block_acc = int((accepted["block"] == block).sum())
            r = summarize(g, a.initial_balance, block_acc, 0, 0, block_days)
            r["block"] = block
            monthly_rows.append(r)

    summary_df = pd.DataFrame([overall])
    block_df = pd.DataFrame(monthly_rows)
    summary_df.to_csv(a.out / "FINAL_532_TOP15_ECONOMIC_REPLAY.csv", index=False)
    block_df.to_csv(a.out / "BY_BLOCK_532_TOP15_ECONOMIC_REPLAY.csv", index=False)

    print("\nFINAL 5/3/2 FOLD-LOCAL TOP15 ECONOMIC REPLAY")
    print(summary_df.to_string(index=False))
    if not block_df.empty:
        cols = [
            "block", "ml_accepted", "trades", "trades_per_market_day", "win_rate_pct",
            "profit_factor", "expectancy_usd", "net_profit_usd", "max_drawdown_pct",
        ]
        print("\nBY_BLOCK")
        print(block_df[cols].to_string(index=False))
    print(f"BLOCKED_MAX_POSITIONS={blocked}")
    print(f"AMBIGUOUS_INTRABAR={executor.ambiguous_intrabar_count}")
    print("AUG14_USED_FOR_REPLAY=NO")
    print("PERCENTILE=TOP15_FROZEN_FOR_THIS_REPLAY")
    print(f"OUTPUT={a.out}")


if __name__ == "__main__":
    main()
