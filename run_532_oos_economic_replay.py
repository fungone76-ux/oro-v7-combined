from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from short_memory_532.dataset import build_tensor_bundle

THRESHOLDS = (0.40, 0.45, 0.50)
FIXED_LOT = 0.01
MAX_POSITIONS = 3


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


def summarize(trades: pd.DataFrame, initial_balance: float, threshold: float, accepted: int, blocked: int, ambiguous: int) -> dict[str, object]:
    if trades.empty:
        return {
            "threshold": threshold, "ml_accepted": accepted, "trades": 0,
            "wins": 0, "losses": 0, "win_rate_pct": 0.0,
            "gross_profit_usd": 0.0, "gross_loss_usd": 0.0,
            "net_profit_usd": 0.0, "profit_factor": 0.0,
            "expectancy_usd": 0.0, "max_drawdown_usd": 0.0,
            "max_drawdown_pct": 0.0, "blocked_max_positions": blocked,
            "ambiguous_intrabar": ambiguous,
        }
    pnl = pd.to_numeric(trades["net_pnl"], errors="coerce").fillna(0.0)
    wins = int((pnl > 0).sum()); losses = int((pnl < 0).sum())
    gp = float(pnl[pnl > 0].sum()); gl = float(-pnl[pnl < 0].sum())
    dd_usd, dd_pct = max_drawdown(pnl, initial_balance)
    return {
        "threshold": threshold,
        "ml_accepted": accepted,
        "trades": int(len(trades)),
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
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Economic OOS replay for 5/3/2 finalist thresholds")
    ap.add_argument("--top40-root", type=Path, default=Path(r"D:\ORO_SHORT_MEMORY_ALLSESSIONS_FINAL"))
    ap.add_argument("--data-dir", type=Path, default=Path(r"D:\ORO_532_2026_TUNING"))
    ap.add_argument("--phase3a", type=Path, default=Path(r"D:\ORO_532_2026_TUNING\phase3a_532"))
    ap.add_argument("--model-dir", type=Path, default=Path(r"D:\ORO_532_2026_TUNING\model_532"))
    ap.add_argument("--out", type=Path, default=Path(r"D:\ORO_532_2026_TUNING\economic_replay_532"))
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

    m1_path = a.data_dir / "XAUUSD_M1_FRESH_532_2026_TUNING.csv"
    market_path = a.phase3a / "market_state_2026.parquet"
    cand_path = a.phase3a / "technical_candidates_feb_jun_2026.parquet"
    pred_path = a.model_dir / "oos_s2_predictions_532.csv.gz"
    for p in (m1_path, market_path, cand_path, pred_path):
        if not p.exists():
            raise SystemExit(f"MISSING_INPUT: {p}")

    if not mt5.initialize():
        raise SystemExit(f"MT5_INIT_FAILED: {mt5.last_error()}")
    try:
        info = mt5.symbol_info("XAUUSD")
        if info is None:
            raise SystemExit("MT5_SYMBOL_INFO_FAILED")
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
    m1 = load_csv(m1_path)
    market = pd.read_parquet(market_path)
    cand = pd.read_parquet(cand_path)
    market["timestamp"] = pd.to_datetime(market["timestamp"], utc=True)
    cand["timestamp"] = pd.to_datetime(cand["timestamp"], utc=True)
    bundle = build_tensor_bundle(market, cand)

    pred = pd.read_csv(pred_path)
    pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True)
    pred["row_index"] = pd.to_numeric(pred["row_index"], errors="raise").astype(int)
    if pred["row_index"].min() < 0 or pred["row_index"].max() >= len(bundle.frame):
        raise RuntimeError("OOS_ROW_INDEX_OUT_OF_RANGE")

    frame = bundle.frame.reset_index(drop=True).copy()
    oos = frame.iloc[pred["row_index"].to_numpy()].copy().reset_index(drop=True)
    oos["s2_score"] = pd.to_numeric(pred["s2_score"], errors="raise").to_numpy()
    oos["oos_fold"] = pred["fold"].to_numpy()
    oos["candidate_entry_time"] = pd.to_datetime(oos["candidate_entry_time"], utc=True)
    oos["timestamp"] = pd.to_datetime(oos["timestamp"], utc=True)

    # Strong alignment guard: prediction timestamp must be the candidate timestamp.
    if not (oos["timestamp"].to_numpy(dtype="datetime64[ns]") == pred["timestamp"].to_numpy(dtype="datetime64[ns]")).all():
        raise RuntimeError("OOS_PREDICTION_ALIGNMENT_FAIL")

    # ATR points known at decision time, forward-filled to M1 bars for shared trailing core.
    atr_map = market[["timestamp", "atr14"]].copy().dropna().sort_values("timestamp")
    atr_map["atr_points"] = pd.to_numeric(atr_map["atr14"], errors="coerce") / spec.point
    m1 = pd.merge_asof(m1.sort_values("time"), atr_map[["timestamp", "atr_points"]], left_on="time", right_on="timestamp", direction="backward")
    m1["atr_points"] = m1["atr_points"].ffill().fillna(0.0)

    eval_start = oos["candidate_entry_time"].min()
    eval_end = oos["candidate_entry_time"].max() + pd.Timedelta(days=1)
    bars = m1[(m1["time"] >= eval_start) & (m1["time"] <= eval_end)].copy()
    if bars.empty:
        raise RuntimeError("NO_M1_BARS_FOR_REPLAY")

    a.out.mkdir(parents=True, exist_ok=True)
    all_summary = []
    all_monthly = []

    print("=" * 78)
    print("TOP40 5/3/2 OOS ECONOMIC REPLAY")
    print(f"OOS candidates={len(oos):,} | thresholds={THRESHOLDS}")
    print(f"lot={FIXED_LOT:.2f} | max_positions={MAX_POSITIONS} | cooldown=OFF | sessions=ALL")
    print(f"TP={cfg.risk.tp_risk_reward:.2f}R | BE={cfg.risk.breakeven_at_r:.2f}R | trailing={cfg.risk.trailing_activation_r:.2f}R")
    print("ENTRY=accepted M5 candidate -> candidate_entry_time / next M1 open")
    print("POSITION_CORE=xau_bot.research.simulated_executor / shared position_management_core")
    print("=" * 78, flush=True)

    for threshold in THRESHOLDS:
        accepted = oos[oos["s2_score"] >= threshold].copy().sort_values(["candidate_entry_time", "timestamp"])
        by_time = {ts: g for ts, g in accepted.groupby("candidate_entry_time", sort=False)}
        settings = BacktestSettings(
            initial_balance=a.initial_balance,
            commission_per_lot=a.commission_per_lot,
            slippage_points=a.slippage_points,
            run_name=f"TOP40_532_OOS_T{threshold:.2f}",
        )
        executor = SimulatedExecutor(spec, cfg.risk, settings)
        blocked = 0

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
                        approved=True, lot=FIXED_LOT, sl_points=sl_points,
                        reason_code="OOS_532_ACCEPTED", risk_amount=risk_amount,
                        risk_pct=(risk_amount / a.initial_balance * 100.0 if a.initial_balance else 0.0),
                    )
                    signal = TechnicalSignal(
                        direction=direction,
                        setup=str(c.get("setup", "PULLBACK")),
                        score=int(round(float(c.get("signal_score", 0)))),
                        confirmations=[Confirmation("oos_s2", True, 1), Confirmation("technical_candidate", True, 1)],
                    )
                    executor.open_position(
                        risk=risk, signal=signal, direction=direction,
                        signal_time=pd.Timestamp(c["timestamp"]).to_pydatetime(),
                        entry_bar=bar, session=str(c.get("session", "")), regime=str(c.get("regime", "")),
                    )
            executor.process_bar(bar, atr_points=float(bar.get("atr_points", 0.0)))

        if executor.positions:
            executor.close_all_end(bars.iloc[-1])

        trades = pd.DataFrame([x.to_dict() for x in executor.closed_trades])
        trades_path = a.out / f"trades_t{threshold:.2f}.csv"
        trades.to_csv(trades_path, index=False)
        s = summarize(trades, a.initial_balance, threshold, len(accepted), blocked, executor.ambiguous_intrabar_count)
        all_summary.append(s)

        if not trades.empty:
            trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
            trades["month"] = trades["entry_time"].dt.to_period("M").astype(str)
            for month, g in trades.groupby("month", sort=True):
                ms = summarize(g, a.initial_balance, threshold, len(g), 0, 0)
                ms["month"] = month
                all_monthly.append(ms)

        print(
            f"threshold={threshold:.2f} accepted={len(accepted):,} trades={s['trades']:,} "
            f"WR={s['win_rate_pct']:.2f}% PF={s['profit_factor']:.4f} "
            f"exp={s['expectancy_usd']:.4f} net={s['net_profit_usd']:.2f} "
            f"DD={s['max_drawdown_pct']:.2f}% blocked={blocked}", flush=True,
        )

    summary = pd.DataFrame(all_summary)
    monthly = pd.DataFrame(all_monthly)
    summary.to_csv(a.out / "FINAL_532_OOS_ECONOMIC_REPLAY.csv", index=False)
    monthly.to_csv(a.out / "MONTHLY_532_OOS_ECONOMIC_REPLAY.csv", index=False)

    print("\nFINAL 5/3/2 OOS ECONOMIC REPLAY")
    print(summary.to_string(index=False))
    if not monthly.empty:
        cols = ["threshold", "month", "trades", "win_rate_pct", "profit_factor", "expectancy_usd", "net_profit_usd", "max_drawdown_pct"]
        print("\nMONTHLY")
        print(monthly[cols].to_string(index=False))
    print("THRESHOLD_SELECTED=NO")
    print(f"OUTPUT={a.out}")


if __name__ == "__main__":
    main()
