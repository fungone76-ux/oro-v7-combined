from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

from short_memory_532.dataset import build_tensor_bundle
from run_532_train_s2_2026 import preprocess, model_class, loader, evaluate, set_seed

TOP_PCT = 10
FIXED_LOT = 0.01
MAX_POSITIONS = 3
JULY_START = pd.Timestamp("2026-07-01T00:00:00Z")
AUG_START = pd.Timestamp("2026-08-01T00:00:00Z")


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)


def max_drawdown(values: pd.Series, initial_balance: float) -> tuple[float, float]:
    equity = initial_balance + values.cumsum()
    peak = equity.cummax()
    dd = peak - equity
    dd_usd = float(dd.max()) if len(dd) else 0.0
    dd_pct = float(((dd / peak.where(peak > 0)) * 100.0).max()) if len(dd) else 0.0
    return dd_usd, dd_pct


def summarize_trades(trades: pd.DataFrame, initial_balance: float, accepted: int, blocked: int, market_days: int, ambiguous: int) -> dict[str, object]:
    if trades.empty:
        return {
            "accepted": accepted, "trades": 0, "trades_per_market_day": 0.0,
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
        "accepted": accepted,
        "trades": int(len(trades)),
        "trades_per_market_day": float(len(trades) / market_days) if market_days else float("nan"),
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
    ap = argparse.ArgumentParser(description="Frozen 5/3/2 Top10 final unseen test on July 2026")
    ap.add_argument("--top40-root", type=Path, default=Path(r"D:\ORO_SHORT_MEMORY_ALLSESSIONS_FINAL"))
    ap.add_argument("--tuning-dir", type=Path, default=Path(r"D:\ORO_532_2026_TUNING"))
    ap.add_argument("--phase3a", type=Path, default=Path(r"D:\ORO_532_2026_TUNING\phase3a_532"))
    ap.add_argument("--monthly-model-dir", type=Path, default=Path(r"D:\ORO_532_2026_TUNING\model_532_fullperiod_oos"))
    ap.add_argument("--july-dir", type=Path, default=Path(r"D:\ORO_532_2026_UNSEEN_JULY"))
    ap.add_argument("--out", type=Path, default=Path(r"D:\ORO_532_2026_UNSEEN_JULY\final_test_532_top10"))
    ap.add_argument("--initial-balance", type=float, default=1000.0)
    ap.add_argument("--commission-per-lot", type=float, default=0.0)
    ap.add_argument("--slippage-points", type=float, default=0.0)
    a = ap.parse_args()
    t0 = time.perf_counter()

    top = a.top40_root.resolve()
    sys.path.insert(0, str(top))
    from xau_bot.config.settings import BotConfig
    from xau_bot.core.models import Confirmation, Direction, RiskDecision, SymbolSpec, TechnicalSignal
    from xau_bot.research.backtest_models import BacktestSettings
    from xau_bot.research.simulated_executor import SimulatedExecutor
    from xau_bot.research.ml.feature_builder import build_market_state_features
    from xau_bot.research.ml.candidate_extractor import extract_strategy_candidates
    from xau_bot.research.ml.label_builder import build_candidate_labels

    tune_paths = {
        "M1": a.tuning_dir / "XAUUSD_M1_FRESH_532_2026_TUNING.csv",
        "M5": a.tuning_dir / "XAUUSD_M5_FRESH_532_2026_TUNING.csv",
        "M15": a.tuning_dir / "XAUUSD_M15_FRESH_532_2026_TUNING.csv",
    }
    july_paths = {
        "M1": a.july_dir / "XAUUSD_M1_FRESH_JULY_UNSEEN_2026.csv",
        "M5": a.july_dir / "XAUUSD_M5_FRESH_JULY_UNSEEN_2026.csv",
        "M15": a.july_dir / "XAUUSD_M15_FRESH_JULY_UNSEEN_2026.csv",
    }
    required = [*tune_paths.values(), *july_paths.values(), a.phase3a / "market_state_2026.parquet", a.phase3a / "technical_candidates_all_2026.parquet", a.monthly_model_dir / "oos_s2_fold_metrics_532_feb_jun_monthly.csv"]
    for p in required:
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

    # ---------------------------
    # Build July features/candidates using only June history + July/Aug tail.
    # ---------------------------
    raw = {}
    for tf in ("M1", "M5", "M15"):
        hist = load_csv(tune_paths[tf])
        hist = hist[(hist["time"] >= pd.Timestamp("2026-06-01T00:00:00Z")) & (hist["time"] < JULY_START)].copy()
        unseen = load_csv(july_paths[tf])
        raw[tf] = pd.concat([hist, unseen], ignore_index=True).drop_duplicates("time").sort_values("time").reset_index(drop=True)

    july_market, model_features, _, _ = build_market_state_features(raw["M1"], raw["M5"], raw["M15"], cfg)
    july_market = july_market[july_market["feature_valid"].astype(bool)].reset_index(drop=True)
    extracted = extract_strategy_candidates(raw["M1"], raw["M5"], raw["M15"], july_market, model_features, cfg, spec, trades_path=None, baseline_scope_end=None)
    july_candidates = extracted.candidates
    july_candidates = july_candidates[july_candidates["feature_valid"].astype(bool)].reset_index(drop=True)
    if not july_candidates.empty:
        labels = build_candidate_labels(july_candidates, raw["M1"], pd.DataFrame(), spec)
        july_candidates = pd.concat([july_candidates.reset_index(drop=True), labels.reset_index(drop=True)], axis=1)
    july_candidates["timestamp"] = pd.to_datetime(july_candidates["timestamp"], utc=True)
    july_candidates = july_candidates[(july_candidates["timestamp"] >= JULY_START) & (july_candidates["timestamp"] < AUG_START)].copy()
    if july_candidates.empty:
        raise RuntimeError("NO_JULY_CANDIDATES")

    # ---------------------------
    # Build one combined 5/3/2 bundle, but hard-freeze train rows to < July.
    # ---------------------------
    train_market = pd.read_parquet(a.phase3a / "market_state_2026.parquet")
    train_market["timestamp"] = pd.to_datetime(train_market["timestamp"], utc=True)
    train_market = train_market[train_market["timestamp"] < JULY_START].copy()
    combined_market = pd.concat([train_market, july_market], ignore_index=True)
    combined_market = combined_market.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)

    train_candidates = pd.read_parquet(a.phase3a / "technical_candidates_all_2026.parquet")
    train_candidates["timestamp"] = pd.to_datetime(train_candidates["timestamp"], utc=True)
    train_candidates = train_candidates[train_candidates["timestamp"] < JULY_START].copy()
    combined_candidates = pd.concat([train_candidates, july_candidates], ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    bundle = build_tensor_bundle(combined_market, combined_candidates)
    ts = pd.to_datetime(bundle.frame["timestamp"], utc=True)
    train_idx = np.flatnonzero(ts < JULY_START)
    july_idx = np.flatnonzero((ts >= JULY_START) & (ts < AUG_START))
    if len(train_idx) == 0 or len(july_idx) == 0:
        raise RuntimeError(f"BAD_TRAIN_JULY_SPLIT train={len(train_idx)} july={len(july_idx)}")

    # Epoch count comes ONLY from prior Feb-Jun monthly OOS folds.
    fold_metrics = pd.read_csv(a.monthly_model_dir / "oos_s2_fold_metrics_532_feb_jun_monthly.csv")
    final_epochs = int(round(pd.to_numeric(fold_metrics["best_epoch"], errors="raise").median()))

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(2026 + 532)
    m1, m5, m15, st, _ = preprocess(bundle, train_idx)
    arrays = (m1, m5, m15, st)
    Model = model_class()
    model = Model(m1.shape[2], m5.shape[2], m15.shape[2], st.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.SmoothL1Loss()
    train_loader = loader(arrays, bundle.y_s2, train_idx, True)
    for ep in range(1, max(1, final_epochs) + 1):
        model.train()
        for x1, x5, x15, xs, y in train_loader:
            x1, x5, x15, xs, y = x1.to(device), x5.to(device), x15.to(device), xs.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(x1, x5, x15, xs), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if ep == 1 or ep == final_epochs or ep % 5 == 0:
            print(f"FINAL_MODEL epoch={ep}/{final_epochs}", flush=True)

    # Frozen Top10 threshold from FINAL MODEL scores on Jan-Jun only.
    train_eval_loader = loader(arrays, bundle.y_s2, train_idx, False)
    _, train_scores = evaluate(model, train_eval_loader, device)
    frozen_threshold = float(pd.Series(train_scores).quantile(0.90))

    july_eval_loader = loader(arrays, bundle.y_s2, july_idx, False)
    july_y, july_scores = evaluate(model, july_eval_loader, device)
    july_frame = bundle.frame.iloc[july_idx].copy().reset_index(drop=True)
    july_frame["y_s2"] = july_y
    july_frame["s2_score"] = july_scores
    july_frame["accepted_top10"] = july_frame["s2_score"] >= frozen_threshold
    accepted = july_frame[july_frame["accepted_top10"]].copy().sort_values(["candidate_entry_time", "timestamp"])

    # ---------------------------
    # Economic replay on July only; Aug tail only closes Jul-31 positions.
    # ---------------------------
    replay_m1 = load_csv(july_paths["M1"])
    market_atr = july_market[["timestamp", "atr14"]].dropna().sort_values("timestamp").copy()
    market_atr["atr_points"] = pd.to_numeric(market_atr["atr14"], errors="coerce") / spec.point
    replay_m1 = pd.merge_asof(replay_m1.sort_values("time"), market_atr[["timestamp", "atr_points"]], left_on="time", right_on="timestamp", direction="backward")
    replay_m1["atr_points"] = replay_m1["atr_points"].ffill().fillna(0.0)
    replay_m1 = replay_m1[(replay_m1["time"] >= JULY_START) & (replay_m1["time"] <= pd.Timestamp("2026-08-03T06:00:00Z"))].copy()

    accepted["candidate_entry_time"] = pd.to_datetime(accepted["candidate_entry_time"], utc=True)
    by_time = {t: g for t, g in accepted.groupby("candidate_entry_time", sort=False)}
    settings = BacktestSettings(initial_balance=a.initial_balance, commission_per_lot=a.commission_per_lot, slippage_points=a.slippage_points, run_name="TOP40_532_JULY_UNSEEN_TOP10")
    executor = SimulatedExecutor(spec, cfg.risk, settings)
    blocked = 0

    for _, bar in replay_m1.iterrows():
        t = bar["time"]
        pending = by_time.get(t)
        if pending is not None:
            for _, c in pending.iterrows():
                if t >= AUG_START:
                    continue
                if len(executor.positions) >= MAX_POSITIONS:
                    blocked += 1
                    continue
                direction = Direction(str(c["direction"]))
                sl_points = float(c["candidate_stop_points"])
                value_per_point = spec.tick_value / spec.tick_size * spec.point * FIXED_LOT
                risk_amount = sl_points * value_per_point
                risk = RiskDecision(approved=True, lot=FIXED_LOT, sl_points=sl_points, reason_code="JULY_UNSEEN_TOP10", risk_amount=risk_amount, risk_pct=(risk_amount / a.initial_balance * 100.0 if a.initial_balance else 0.0))
                signal = TechnicalSignal(direction=direction, setup=str(c.get("setup", "PULLBACK")), score=int(round(float(c.get("signal_score", 0)))), confirmations=[Confirmation("frozen_top10", True, 1), Confirmation("technical_candidate", True, 1)])
                executor.open_position(risk=risk, signal=signal, direction=direction, signal_time=pd.Timestamp(c["timestamp"]).to_pydatetime(), entry_bar=bar, session=str(c.get("session", "")), regime=str(c.get("regime", "")))
        executor.process_bar(bar, atr_points=float(bar.get("atr_points", 0.0)))

    if executor.positions:
        executor.close_all_end(replay_m1.iloc[-1])

    trades = pd.DataFrame([x.to_dict() for x in executor.closed_trades])
    market_days = int(replay_m1[(replay_m1["time"] >= JULY_START) & (replay_m1["time"] < AUG_START)]["time"].dt.date.nunique())
    summary = summarize_trades(trades, a.initial_balance, len(accepted), blocked, market_days, executor.ambiguous_intrabar_count)

    # Barrier / label quality diagnostics on unseen July candidates only.
    dfr = pd.to_numeric(accepted.get("directional_future_return_15m"), errors="coerce")
    summary["positive_15m_pct"] = float((dfr > 0).mean() * 100.0) if len(accepted) else float("nan")
    summary["mean_target_quality_r"] = float(pd.to_numeric(accepted["y_s2"], errors="coerce").mean()) if len(accepted) else float("nan")
    summary["frozen_top10_threshold"] = frozen_threshold
    summary["train_rows_jan_jun"] = int(len(train_idx))
    summary["july_candidate_rows"] = int(len(july_idx))
    summary["july_accepted_pct"] = float(len(accepted) / len(july_idx) * 100.0) if len(july_idx) else float("nan")
    summary["final_epochs"] = final_epochs

    a.out.mkdir(parents=True, exist_ok=True)
    july_frame.to_parquet(a.out / "july_candidates_scored.parquet", index=False)
    accepted.to_parquet(a.out / "july_candidates_accepted_top10.parquet", index=False)
    trades.to_csv(a.out / "july_unseen_trades.csv", index=False)
    pd.DataFrame([summary]).to_csv(a.out / "JULY_UNSEEN_FINAL_SUMMARY.csv", index=False)

    print("\n" + "=" * 78)
    print("5/3/2 JULY 2026 FINAL UNSEEN TEST")
    print(f"DEVICE={device}")
    print(f"TRAIN_ROWS_JAN_JUN={len(train_idx):,}")
    print(f"FINAL_EPOCHS_FROM_PRIOR_OOS={final_epochs}")
    print(f"FROZEN_METHOD=TOP_{TOP_PCT}_PCT")
    print(f"FROZEN_THRESHOLD_FROM_JAN_JUN={frozen_threshold:.6f}")
    print(f"JULY_CANDIDATES={len(july_idx):,}")
    print(f"JULY_ACCEPTED={len(accepted):,} ({summary['july_accepted_pct']:.2f}%)")
    print(f"MARKET_DAYS={market_days}")
    print(f"TRADES={summary['trades']:,} | TRADES_PER_DAY={summary['trades_per_market_day']:.2f}")
    print(f"WIN_RATE={summary['win_rate_pct']:.2f}% | PF={summary['profit_factor']:.4f} | EXPECTANCY_USD={summary['expectancy_usd']:.4f}")
    print(f"NET_PROFIT_USD={summary['net_profit_usd']:.2f} | MAX_DD_PCT={summary['max_drawdown_pct']:.2f}%")
    print(f"POSITIVE_15M_PCT={summary['positive_15m_pct']:.2f}% | MEAN_TARGET_QUALITY_R={summary['mean_target_quality_r']:.4f}")
    print(f"BLOCKED_MAX_POSITIONS={blocked} | AMBIGUOUS_INTRABAR={executor.ambiguous_intrabar_count}")
    print("JULY_USED_FOR_TRAINING=NO")
    print("JULY_USED_FOR_EPOCH_SELECTION=NO")
    print("JULY_USED_FOR_THRESHOLD_SELECTION=NO")
    print("JULY_USED_ONLY_FOR_FINAL_EVALUATION=YES")
    print(f"OUTPUT={a.out}")
    print("=" * 78)


if __name__ == "__main__":
    main()
