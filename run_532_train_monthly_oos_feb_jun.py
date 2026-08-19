from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd

from short_memory_532.config import CONFIG
from short_memory_532.dataset import build_tensor_bundle
from run_532_train_s2_2026 import train_fold, log


MONTHS = ["2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]


def monthly_folds(frame: pd.DataFrame):
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    gap = pd.Timedelta(minutes=CONFIG.purge_minutes + CONFIG.embargo_minutes)
    folds = []
    coverage = []
    for fid, month in enumerate(MONTHS, start=1):
        start = pd.Timestamp(f"{month}-01T00:00:00Z")
        end = start + pd.offsets.MonthBegin(1)
        train_cutoff = start - gap
        tr = np.flatnonzero(ts < train_cutoff)
        va = np.flatnonzero((ts >= start) & (ts < end))
        if len(tr) == 0:
            raise RuntimeError(f"EMPTY_TRAIN_{month}")
        if len(va) == 0:
            raise RuntimeError(f"EMPTY_VALID_{month}")
        folds.append((fid, tr, va, ts.iloc[tr[-1]], ts.iloc[va[0]], ts.iloc[va[-1]]))
        coverage.append({
            "fold": fid,
            "month": month,
            "train_rows": int(len(tr)),
            "valid_rows": int(len(va)),
            "train_last_timestamp": ts.iloc[tr[-1]].isoformat(),
            "valid_first_timestamp": ts.iloc[va[0]].isoformat(),
            "valid_last_timestamp": ts.iloc[va[-1]].isoformat(),
        })
    return folds, pd.DataFrame(coverage)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="5/3/2 monthly expanding walk-forward: Jan seed -> OOS Feb-Jun 2026"
    )
    ap.add_argument(
        "--phase3a",
        type=Path,
        default=Path(r"D:\ORO_532_2026_TUNING\phase3a_532"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(r"D:\ORO_532_2026_TUNING\model_532_fullperiod_oos"),
    )
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    market_path = a.phase3a / "market_state_2026.parquet"
    candidates_path = a.phase3a / "technical_candidates_all_2026.parquet"
    if not market_path.exists():
        raise SystemExit(f"MISSING_MARKET: {market_path}")
    if not candidates_path.exists():
        raise SystemExit(
            f"MISSING_ALL_CANDIDATES: {candidates_path}\n"
            "Run python run_532_prepare_2026.py after git pull."
        )

    market = pd.read_parquet(market_path)
    candidates = pd.read_parquet(candidates_path)
    market["timestamp"] = pd.to_datetime(market["timestamp"], utc=True)
    candidates["timestamp"] = pd.to_datetime(candidates["timestamp"], utc=True)

    # Keep Jan-Jun for this threshold-selection experiment. January is training seed only.
    cutoff = pd.Timestamp("2026-07-01T00:00:00Z")
    candidates = candidates[candidates["timestamp"] < cutoff].copy()
    bundle = build_tensor_bundle(market, candidates)
    log(t0, f"5/3/2 Jan-Jun valid samples={len(bundle.frame):,}")

    folds, coverage = monthly_folds(bundle.frame)
    print("MONTHLY_OOS_CONTRACT=JAN_SEED_THEN_FEB_MAR_APR_MAY_JUN", flush=True)
    print(coverage.to_string(index=False), flush=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DEVICE={device}", flush=True)

    preds = []
    metrics = []
    for fold, month in zip(folds, MONTHS):
        print(f"\n=== OOS MONTH {month} ===", flush=True)
        pred, met, best_epoch = train_fold(bundle, fold, device, t0)
        pred["validation_month"] = month
        met["validation_month"] = month
        met["best_epoch"] = best_epoch
        preds.append(pred)
        metrics.append(met)

    out_pred = pd.concat(preds, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    out_metrics = pd.DataFrame(metrics)

    expected_months = set(MONTHS)
    actual_months = set(pd.to_datetime(out_pred["timestamp"], utc=True).dt.strftime("%Y-%m"))
    if actual_months != expected_months:
        raise RuntimeError(
            f"OOS_MONTH_COVERAGE_FAIL expected={sorted(expected_months)} actual={sorted(actual_months)}"
        )

    pred_path = a.out / "oos_s2_predictions_532_feb_jun_monthly.csv.gz"
    metrics_path = a.out / "oos_s2_fold_metrics_532_feb_jun_monthly.csv"
    coverage_path = a.out / "oos_month_coverage_532.csv"
    out_pred.to_csv(pred_path, index=False, compression="gzip")
    out_metrics.to_csv(metrics_path, index=False)
    coverage.to_csv(coverage_path, index=False)

    print("\nTRAIN_532_FULL_FEB_JUN_OOS=PASS")
    print(f"VALID_SAMPLES_JAN_JUN={len(bundle.frame):,}")
    print(f"OOS_PREDICTIONS_FEB_JUN={len(out_pred):,}")
    print("OOS_MONTHS=2026-02,2026-03,2026-04,2026-05,2026-06")
    print("JANUARY_ROLE=TRAINING_SEED_ONLY")
    print("THRESHOLD_SELECTED=NO")
    print(f"OUTPUT={a.out}")


if __name__ == "__main__":
    main()
