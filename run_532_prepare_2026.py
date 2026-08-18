from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build causal Phase3A market/candidates for 5/3/2 2026 tuning")
    ap.add_argument("--top40-root", type=Path, default=Path(r"D:\ORO_SHORT_MEMORY_ALLSESSIONS_FINAL"))
    ap.add_argument("--data-dir", type=Path, default=Path(r"D:\ORO_532_2026_TUNING"))
    ap.add_argument("--out", type=Path, default=Path(r"D:\ORO_532_2026_TUNING\phase3a_532"))
    a = ap.parse_args()

    top = a.top40_root.resolve(); data = a.data_dir.resolve(); out = a.out.resolve()
    sys.path.insert(0, str(top))
    from xau_bot.config.settings import BotConfig
    from xau_bot.research.ml.feature_builder import build_market_state_features
    from xau_bot.research.ml.candidate_extractor import extract_strategy_candidates
    from xau_bot.research.ml import phase3c_economic_replay as p3c

    paths = {
        "M1": data / "XAUUSD_M1_FRESH_532_2026_TUNING.csv",
        "M5": data / "XAUUSD_M5_FRESH_532_2026_TUNING.csv",
        "M15": data / "XAUUSD_M15_FRESH_532_2026_TUNING.csv",
    }
    for k, p in paths.items():
        if not p.exists():
            raise SystemExit(f"MISSING_{k}: {p}")

    # Resolve the symbol metadata from the imported engine root, not from cwd.
    # p3c.METADATA_PATH is relative by design in the original project, so using it
    # directly from another repo/cwd is fragile.
    metadata_path = top / "data" / "historical" / "icmarkets_native" / "manifest.json"
    if not metadata_path.exists():
        raise SystemExit(
            "MISSING_SYMBOL_METADATA: " + str(metadata_path) + "\n"
            "Expected engine metadata under --top40-root."
        )

    m1, m5, m15 = (load_csv(paths[k]) for k in ("M1", "M5", "M15"))
    cfg = BotConfig(); spec = p3c.load_symbol_spec(metadata_path)
    market, model_features, diag, _ = build_market_state_features(m1, m5, m15, cfg)
    market = market[market["feature_valid"].astype(bool)].reset_index(drop=True)
    extracted = extract_strategy_candidates(
        m1, m5, m15, market, model_features, cfg, spec,
        trades_path=None, baseline_scope_end=None,
    )
    candidates = extracted.candidates
    candidates = candidates[candidates["feature_valid"].astype(bool)].reset_index(drop=True)

    start = pd.Timestamp("2026-02-01T00:00:00Z")
    end = pd.Timestamp("2026-06-30T23:59:59Z")
    candidates["timestamp"] = pd.to_datetime(candidates["timestamp"], utc=True)
    tune_candidates = candidates[(candidates.timestamp >= start) & (candidates.timestamp <= end)].copy()
    if tune_candidates.empty:
        raise RuntimeError("NO_TUNING_CANDIDATES_2026_FEB_JUN")

    out.mkdir(parents=True, exist_ok=True)
    market_path = out / "market_state_2026.parquet"
    cand_path = out / "technical_candidates_feb_jun_2026.parquet"
    market.to_parquet(market_path, index=False)
    tune_candidates.to_parquet(cand_path, index=False)

    manifest = {
        "sequence_target": "5/3/2",
        "tuning_start": start.isoformat(),
        "tuning_end": end.isoformat(),
        "market_rows": int(len(market)),
        "candidate_rows_all": int(len(candidates)),
        "candidate_rows_tuning": int(len(tune_candidates)),
        "market_path": str(market_path),
        "candidate_path": str(cand_path),
        "metadata_path": str(metadata_path),
        "future_labels_required": True,
        "lookahead_in_features": False,
    }
    (out / "PHASE3A_532_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("PHASE3A_532_PREPARE=PASS")
    print(f"MARKET_ROWS={len(market):,}")
    print(f"CANDIDATES_ALL={len(candidates):,}")
    print(f"CANDIDATES_TUNING={len(tune_candidates):,}")
    print("TUNING_PERIOD=2026-02-01..2026-06-30")
    print(f"METADATA={metadata_path}")
    print(f"OUTPUT={out}")


if __name__ == "__main__":
    main()
