from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import run_fresh_2022_frozen as fresh

THRESHOLDS = [0.10, 0.20, 0.25, 0.30, 0.32696733474731443, 0.35, 0.40]
FIXED_LOT = 0.01


def main() -> None:
    ap = argparse.ArgumentParser(description="Fresh 2022 S2 threshold sensitivity sweep; research only")
    ap.add_argument("--top40-root", type=Path, required=True)
    ap.add_argument("--short-out", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("research_output/fresh_2022_threshold_sweep"))
    a = ap.parse_args()

    t0 = time.perf_counter()
    repo = Path(__file__).resolve().parent
    top = a.top40_root.resolve()
    short = a.short_out.resolve()
    data_dir = a.data_dir.resolve()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(top))
    os.chdir(top)

    src = {
        "M1": data_dir / "XAUUSD_M1_FRESH_2022.csv",
        "M5": data_dir / "XAUUSD_M5_FRESH_2022.csv",
        "M15": data_dir / "XAUUSD_M15_FRESH_2022.csv",
    }
    for key, path in src.items():
        if not path.exists():
            raise RuntimeError(f"MISSING_FRESH_2022_{key}: {path}")
        actual = fresh.sha256(path)
        expected = fresh.EXPECTED_HASHES_2022[key]
        if actual != expected:
            raise RuntimeError(f"FRESH_2022_{key}_HASH_MISMATCH expected={expected} actual={actual}")

    base = fresh.load_patched_base_for_2022(repo)
    model_path = short / "selected_short_memory_s2.pt"
    scaler_path = short / "short_memory_scalers_s2.joblib"
    if not model_path.exists() or not scaler_path.exists():
        raise RuntimeError("FROZEN_ARTIFACT_MISSING")

    print("FRESH_2022_DATASET_HASHES=PASS", flush=True)
    print("REUSED_OLD_FILES=NO", flush=True)
    print("REUSED_OLD_PARQUETS=NO", flush=True)
    print("NO_RETRAINING=PASS", flush=True)
    print("MAX_POSITIONS=3", flush=True)
    print("SESSION_FILTER=DISABLED", flush=True)
    print("COOLDOWN=DISABLED", flush=True)
    print("FIXED_LOT=0.01", flush=True)
    print("THRESHOLD_SWEEP=RESEARCH_ONLY", flush=True)

    m1 = base.load_csv(src["M1"])
    m5 = base.load_csv(src["M5"])
    m15 = base.load_csv(src["M15"])
    if m1.time.min() > pd.Timestamp("2021-12-05T00:00:00Z") or m1.time.max() < pd.Timestamp("2023-01-02T00:00:00Z"):
        raise RuntimeError("INSUFFICIENT_2022_WARMUP_OR_TAIL_COVERAGE")

    from xau_bot.config.settings import BotConfig
    from xau_bot.research.ml.feature_builder import build_market_state_features
    from xau_bot.research.ml.candidate_extractor import extract_strategy_candidates
    from xau_bot.research.ml import phase3c_economic_replay as p3c

    cfg = BotConfig()
    if int(cfg.risk.max_open_positions) != 3:
        raise RuntimeError(f"MAX_POSITIONS_CONTRACT_BROKEN value={cfg.risk.max_open_positions}")
    spec = p3c.load_symbol_spec(p3c.METADATA_PATH)

    market, model_features, _diag, _ = build_market_state_features(m1, m5, m15, cfg)
    market = market[market["feature_valid"].astype(bool)].reset_index(drop=True)
    extracted = extract_strategy_candidates(
        m1, m5, m15, market, model_features, cfg, spec,
        trades_path=None, baseline_scope_end=None,
    )
    candidates = extracted.candidates
    candidates = candidates[candidates["feature_valid"].astype(bool)].reset_index(drop=True)
    print(f"REBUILT_TECHNICAL_CANDIDATES={len(candidates):,}", flush=True)

    frame, b1, b5, b15, sn, sc = base.build_inference_bundle(market, candidates)
    arrays = base.transform_bundle(b1, b5, b15, sn, sc, joblib.load(scaler_path))

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = base.load_model(
        model_path,
        (arrays[0].shape[2], arrays[1].shape[2], arrays[2].shape[2], arrays[3].shape[1]),
        device,
    )
    scores = base.predict(model, arrays, device)
    pred_base = pd.DataFrame({
        "sample_id": frame["sample_id"].astype(str),
        "timestamp": pd.to_datetime(frame["timestamp"], utc=True),
        "direction": frame["direction"].astype(str),
        "setup": frame["setup"].astype(str),
        "s1_score": np.nan,
        "s2_score": scores.astype(float),
    })
    pred_base = pred_base[
        (pred_base.timestamp >= fresh.EVAL_START_2022) &
        (pred_base.timestamp <= fresh.EVAL_END_2022)
    ].copy()
    pred_base.to_csv(out / "fresh_2022_all_s2_scores.csv", index=False)
    print(f"MODEL_PREDICTIONS_2022={len(pred_base):,}", flush=True)

    # Research-only policy patches, identical to current ALL-SESSIONS candidate.
    original_register = p3c.RiskManager.register_trade_result
    def register_no_cooldown(self, is_win, now_utc):
        original_register(self, is_win, now_utc)
        self.state.cooldown_until_utc = None
    p3c.RiskManager.register_trade_result = register_no_cooldown

    original_decide = p3c.decide
    def decide_all_sessions(signal, snapshot, kill_switch_blocks_entries, enabled_sessions=None, live_confirmed=False, ml_state=None, ml_probability_up=None):
        kwargs = dict(
            kill_switch_blocks_entries=kill_switch_blocks_entries,
            enabled_sessions=(snapshot.session.value,),
            live_confirmed=live_confirmed,
            ml_probability_up=ml_probability_up,
        )
        if ml_state is not None:
            kwargs["ml_state"] = ml_state
        return original_decide(signal, snapshot, **kwargs)
    p3c.decide = decide_all_sessions

    def fixed_lot(self, risk_amount, sl_points):
        return FIXED_LOT if self.spec.volume_min <= FIXED_LOT <= self.spec.volume_max else 0.0
    p3c.RiskManager._lot_from_risk = fixed_lot

    rows = []
    for i, threshold in enumerate(THRESHOLDS, 1):
        pred = pred_base.copy()
        pred["s2_threshold"] = float(threshold)
        pred["ml_accept"] = pred["s2_score"] >= float(threshold)
        accepted = int(pred["ml_accept"].sum())
        print(f"[{i}/{len(THRESHOLDS)}] threshold={threshold:.15f} accepted={accepted:,}", flush=True)

        engine = p3c.MLEconomicReplayEngine(
            cfg, spec, p3c.BacktestSettings(initial_balance=1000.0), pred, ml_enabled=True
        )
        result = engine.run(m1, m5, m15, fresh.EVAL_START_2022, fresh.EVAL_END_2022)
        trades = p3c._trades_frame(result, pred, float(threshold), True)
        if trades.empty:
            metrics = {
                "trades": 0, "wins": 0, "losses": 0, "win_rate_pct": 0.0,
                "gross_profit_usd": 0.0, "gross_loss_usd": 0.0,
                "net_profit_usd": 0.0, "profit_factor": 0.0,
                "expectancy_usd": 0.0, "max_drawdown_usd": 0.0,
                "max_drawdown_pct": 0.0,
            }
        else:
            if "volume" in trades.columns and not np.allclose(pd.to_numeric(trades["volume"]), FIXED_LOT):
                raise RuntimeError(f"FIXED_LOT_CONTRACT_BROKEN threshold={threshold}")
            metrics = base.pnl_metrics(trades)
            trades.to_csv(out / f"trades_threshold_{threshold:.6f}.csv", index=False)

        row = {"threshold": float(threshold), "ml_accepted": accepted, **metrics}
        rows.append(row)
        print(
            f"   trades={metrics['trades']} WR={metrics['win_rate_pct']:.2f}% "
            f"PF={metrics['profit_factor']:.4f} exp={metrics['expectancy_usd']:.4f} "
            f"net={metrics['net_profit_usd']:.2f} DD={metrics['max_drawdown_pct']:.2f}%",
            flush=True,
        )

    summary = pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)
    summary.to_csv(out / "threshold_sensitivity_2022.csv", index=False)
    (out / "threshold_sensitivity_2022.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )

    print("\nFINAL 2022 THRESHOLD SENSITIVITY")
    print(summary.to_string(index=False))
    print("\nIMPORTANT=THIS_IS_SENSITIVITY_RESEARCH_NOT_A_NEW_THRESHOLD_SELECTION")
    print("CURRENT_FROZEN_THRESHOLD=0.32696733474731443")
    print("MODEL_RETRAINED=NO")
    print("MAX_POSITIONS=3")
    print("FRESH_2022_DATASET_HASHES=PASS")
    print(f"RUNTIME_SECONDS={time.perf_counter()-t0:.1f}")


if __name__ == "__main__":
    main()
