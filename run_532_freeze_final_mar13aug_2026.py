from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from short_memory_532.dataset import build_tensor_bundle
from short_memory_532.config import CONFIG, MODEL_ARCHITECTURE, RESEARCH_VERSION
from run_532_train_s2_2026 import fit_final, loader, evaluate, sha256
from short_memory.training import M1_FEATURES, M5_FEATURES, M15_FEATURES, STATIC_NUMERIC, STATIC_CATEGORICAL

PHASE3A = Path(r"D:\ORO_532_MAR13AUG_2026\phase3a_532")
OUT = Path(r"D:\ORO_532_MAR13AUG_2026\frozen_demo_532_top15")
FINAL_EPOCHS = 20
TOP_PCT = 15
UNSEEN_START = pd.Timestamp("2026-08-14T00:00:00Z")


def main() -> None:
    market_path = PHASE3A / "market_state_mar13aug_2026.parquet"
    cand_path = PHASE3A / "technical_candidates_mar13aug_valid.parquet"
    for p in (market_path, cand_path):
        if not p.exists():
            raise SystemExit(f"MISSING_INPUT: {p}")

    market = pd.read_parquet(market_path)
    cand = pd.read_parquet(cand_path)
    market["timestamp"] = pd.to_datetime(market["timestamp"], utc=True)
    cand["timestamp"] = pd.to_datetime(cand["timestamp"], utc=True)
    if (market["timestamp"] >= UNSEEN_START).any() or (cand["timestamp"] >= UNSEEN_START).any():
        raise RuntimeError("AUG14_LEAK_IN_FINAL_FREEZE_INPUT")

    bundle = build_tensor_bundle(market, cand)
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 78)
    print("5/3/2 FINAL FREEZE - MAR 01 -> AUG 13 2026")
    print(f"DEVICE={device} | samples={len(bundle.frame):,}")
    print(f"FINAL_EPOCHS={FINAL_EPOCHS} | FROZEN_FILTER=TOP_{TOP_PCT}_PCT")
    print("AUG14_USED=NO")
    print("=" * 78, flush=True)

    model, scalers = fit_final(bundle, device, FINAL_EPOCHS)
    idx = np.arange(len(bundle.frame))
    from run_532_train_s2_2026 import preprocess
    m1, m5, m15, st, _ = preprocess(bundle, idx)
    _, scores = evaluate(model, loader((m1, m5, m15, st), bundle.y_s2, idx, False), device)
    threshold = float(np.quantile(scores.astype(float), 1.0 - TOP_PCT / 100.0))

    OUT.mkdir(parents=True, exist_ok=True)
    model_path = OUT / "selected_532_s2_top15.pt"
    scaler_path = OUT / "scalers_532_s2_top15.joblib"
    torch.save(model.state_dict(), model_path)
    joblib.dump(scalers, scaler_path)

    score_summary = pd.DataFrame({
        "metric": ["min", "p50", "p85_top15_threshold", "p90", "max", "mean"],
        "value": [
            float(np.min(scores)), float(np.quantile(scores, 0.50)), threshold,
            float(np.quantile(scores, 0.90)), float(np.max(scores)), float(np.mean(scores)),
        ],
    })
    score_summary.to_csv(OUT / "training_score_distribution.csv", index=False)

    manifest = {
        "research_version": RESEARCH_VERSION,
        "sequence_lengths": {"m1": 5, "m5": 3, "m15": 2},
        "training_start_utc": "2026-03-01T00:00:00Z",
        "training_end_utc": "2026-08-13T23:59:59Z",
        "unseen_start_utc": "2026-08-14T00:00:00Z",
        "samples": int(len(bundle.frame)),
        "final_epochs": FINAL_EPOCHS,
        "filter": "TOP_15_PCT",
        "top_pct": TOP_PCT,
        "frozen_threshold": threshold,
        "model_architecture": MODEL_ARCHITECTURE,
        "features": {
            "m1": list(M1_FEATURES), "m5": list(M5_FEATURES), "m15": list(M15_FEATURES),
            "static_numeric": list(STATIC_NUMERIC), "static_categorical": list(STATIC_CATEGORICAL),
        },
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "model_sha256": sha256(model_path),
        "scaler_sha256": sha256(scaler_path),
        "fixed_lot": 0.01,
        "max_positions": 3,
        "session_filter": False,
        "cooldown": False,
        "tp_r": 1.5,
        "be_r": 1.0,
        "trailing_r": 1.5,
        "aug14_used_for_training": False,
        "aug14_used_for_threshold_selection": False,
    }
    manifest_path = OUT / "FROZEN_532_TOP15_MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("FINAL_532_FREEZE=PASS")
    print(f"TRAINING_SAMPLES={len(bundle.frame):,}")
    print(f"FINAL_EPOCHS={FINAL_EPOCHS}")
    print(f"TOP15_THRESHOLD={threshold:.9f}")
    print(f"MODEL={model_path}")
    print(f"SCALERS={scaler_path}")
    print("AUG14_USED_FOR_TRAINING=NO")
    print("AUG14_USED_FOR_THRESHOLD_SELECTION=NO")
    print(f"MANIFEST={manifest_path}")


if __name__ == "__main__":
    main()
