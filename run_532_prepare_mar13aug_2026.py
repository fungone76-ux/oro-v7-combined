from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import MetaTrader5 as mt5
import pandas as pd

TRAIN_START = pd.Timestamp("2026-03-01T00:00:00Z")
UNSEEN_START = pd.Timestamp("2026-08-14T00:00:00Z")
LABEL_HORIZON = pd.Timedelta(minutes=15)


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build causal Phase3A dataset for 5/3/2 Mar-13Aug 2026")
    ap.add_argument("--top40-root", type=Path, default=Path(r"D:\ORO_SHORT_MEMORY_ALLSESSIONS_FINAL"))
    ap.add_argument("--data-dir", type=Path, default=Path(r"D:\ORO_532_MAR13AUG_2026"))
    ap.add_argument("--out", type=Path, default=Path(r"D:\ORO_532_MAR13AUG_2026\phase3a_532"))
    a = ap.parse_args()

    top = a.top40_root.resolve()
    data = a.data_dir.resolve()
    out = a.out.resolve()
    sys.path.insert(0, str(top))

    from xau_bot.config.settings import BotConfig
    from xau_bot.core.models import SymbolSpec
    from xau_bot.research.ml.feature_builder import build_market_state_features
    from xau_bot.research.ml.candidate_extractor import extract_strategy_candidates
    from xau_bot.research.ml.label_builder import build_candidate_labels

    paths = {
        "M1": data / "XAUUSD_M1_FRESH_MAR13AUG_2026.csv",
        "M5": data / "XAUUSD_M5_FRESH_MAR13AUG_2026.csv",
        "M15": data / "XAUUSD_M15_FRESH_MAR13AUG_2026.csv",
    }
    for name, path in paths.items():
        if not path.exists():
            raise SystemExit(f"MISSING_{name}: {path}")

    if not mt5.initialize():
        raise SystemExit(f"MT5_INIT_FAILED: {mt5.last_error()}")
    try:
        info = mt5.symbol_info("XAUUSD")
        if info is None:
            raise SystemExit(f"MT5_SYMBOL_INFO_FAILED: {mt5.last_error()}")
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
        account = mt5.account_info()
        print(
            f"MT5_SYMBOL_SPEC=PASS symbol={spec.name} point={spec.point} digits={spec.digits} "
            f"tick_size={spec.tick_size} tick_value={spec.tick_value} contract_size={spec.contract_size} "
            f"volume_min={spec.volume_min} volume_step={spec.volume_step}",
            flush=True,
        )
        if account is not None:
            print(f"MT5_ACCOUNT login={account.login} server={account.server}", flush=True)
    finally:
        mt5.shutdown()

    m1, m5, m15 = (load_csv(paths[k]) for k in ("M1", "M5", "M15"))
    cfg = BotConfig()
    market, model_features, _, _ = build_market_state_features(m1, m5, m15, cfg)
    market["timestamp"] = pd.to_datetime(market["timestamp"], utc=True)
    market = market[market["feature_valid"].astype(bool)].reset_index(drop=True)

    extracted = extract_strategy_candidates(
        m1, m5, m15, market, model_features, cfg, spec,
        trades_path=None, baseline_scope_end=None,
    )
    candidates = extracted.candidates.copy()
    candidates["timestamp"] = pd.to_datetime(candidates["timestamp"], utc=True)
    candidates["candidate_entry_time"] = pd.to_datetime(candidates["candidate_entry_time"], utc=True)
    candidates = candidates[
        candidates["feature_valid"].astype(bool)
        & (candidates["timestamp"] >= TRAIN_START)
        & (candidates["timestamp"] < UNSEEN_START)
    ].reset_index(drop=True)

    gaps = pd.DataFrame()
    labels = build_candidate_labels(candidates, m1, gaps, spec)
    candidates = pd.concat([candidates.reset_index(drop=True), labels.reset_index(drop=True)], axis=1)

    required = [
        "label_validity_15m",
        "directional_future_return_15m",
        "directional_MFE_price_15m",
        "directional_MAE_price_15m",
    ]
    missing = [c for c in required if c not in candidates.columns]
    if missing:
        raise RuntimeError(f"CANDIDATE_LABEL_BUILD_FAILED missing={missing}")

    # Hard no-leak guard: the complete 15-minute target horizon must finish before Aug 14.
    horizon_end = candidates["candidate_entry_time"] + LABEL_HORIZON
    no_spill = horizon_end <= UNSEEN_START
    before_guard = len(candidates)
    candidates = candidates[no_spill].copy().reset_index(drop=True)
    excluded_spill = before_guard - len(candidates)

    valid_mask = candidates["label_validity_15m"].astype(str).eq("VALID")
    valid_candidates = candidates[valid_mask].copy().reset_index(drop=True)
    invalid_labels = int((~valid_mask).sum())
    if valid_candidates.empty:
        raise RuntimeError("NO_VALID_TRAINING_CANDIDATES")

    if (valid_candidates["candidate_entry_time"] + LABEL_HORIZON > UNSEEN_START).any():
        raise RuntimeError("AUG14_LABEL_LEAK_GUARD_FAILED")

    out.mkdir(parents=True, exist_ok=True)
    market_path = out / "market_state_mar13aug_2026.parquet"
    all_path = out / "technical_candidates_mar13aug_all.parquet"
    valid_path = out / "technical_candidates_mar13aug_valid.parquet"
    market.to_parquet(market_path, index=False)
    candidates.to_parquet(all_path, index=False)
    valid_candidates.to_parquet(valid_path, index=False)

    manifest = {
        "sequence_target": "5/3/2",
        "training_start": TRAIN_START.isoformat(),
        "unseen_start": UNSEEN_START.isoformat(),
        "label_horizon_minutes": 15,
        "market_rows": int(len(market)),
        "candidate_rows_after_no_spill_guard": int(len(candidates)),
        "valid_training_candidates": int(len(valid_candidates)),
        "invalid_labels": invalid_labels,
        "excluded_for_aug14_label_spill": int(excluded_spill),
        "future_labels_built": True,
        "future_labels_used_as_features": False,
        "aug14_used_in_features": False,
        "aug14_used_in_labels": False,
        "market_path": str(market_path),
        "all_candidates_path": str(all_path),
        "valid_candidates_path": str(valid_path),
        "symbol_spec_source": "MT5 symbol_info(XAUUSD)",
    }
    (out / "PHASE3A_532_MAR13AUG_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("PHASE3A_532_MAR13AUG_PREPARE=PASS")
    print(f"MARKET_ROWS={len(market):,}")
    print(f"CANDIDATES_AFTER_NO_SPILL_GUARD={len(candidates):,}")
    print(f"VALID_TRAINING_CANDIDATES={len(valid_candidates):,}")
    print(f"INVALID_LABELS={invalid_labels:,}")
    print(f"EXCLUDED_FOR_AUG14_LABEL_SPILL={excluded_spill:,}")
    print("LABEL_HORIZON=15m")
    print("AUG14_USED_IN_FEATURES=NO")
    print("AUG14_USED_IN_LABELS=NO")
    print("FUTURE_LABELS_USED_AS_FEATURES=NO")
    print("TRAINING_PERIOD=2026-03-01..2026-08-13")
    print("UNSEEN_START=2026-08-14T00:00:00Z")
    print(f"OUTPUT={out}")


if __name__ == "__main__":
    main()
