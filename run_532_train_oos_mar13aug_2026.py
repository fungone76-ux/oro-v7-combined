from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd

from short_memory_532.config import CONFIG
from short_memory_532.dataset import build_tensor_bundle
from run_532_train_s2_2026 import (
    evaluate,
    loader,
    log,
    metrics,
    model_class,
    preprocess,
    set_seed,
)

BLOCKS = [
    ("2026-04", pd.Timestamp("2026-04-01T00:00:00Z"), pd.Timestamp("2026-05-01T00:00:00Z")),
    ("2026-05", pd.Timestamp("2026-05-01T00:00:00Z"), pd.Timestamp("2026-06-01T00:00:00Z")),
    ("2026-06", pd.Timestamp("2026-06-01T00:00:00Z"), pd.Timestamp("2026-07-01T00:00:00Z")),
    ("2026-07", pd.Timestamp("2026-07-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
    ("2026-08-01_13", pd.Timestamp("2026-08-01T00:00:00Z"), pd.Timestamp("2026-08-14T00:00:00Z")),
]
TOTAL_FOLDS = len(BLOCKS)
UNSEEN_START = pd.Timestamp("2026-08-14T00:00:00Z")


def expanding_folds(frame: pd.DataFrame):
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    gap = pd.Timedelta(minutes=CONFIG.purge_minutes + CONFIG.embargo_minutes)
    folds = []
    coverage = []
    for fid, (label, start, end) in enumerate(BLOCKS, start=1):
        train_cutoff = start - gap
        tr = np.flatnonzero(ts < train_cutoff)
        va = np.flatnonzero((ts >= start) & (ts < end))
        if len(tr) == 0:
            raise RuntimeError(f"EMPTY_TRAIN_{label}")
        if len(va) == 0:
            raise RuntimeError(f"EMPTY_VALID_{label}")
        if ts.iloc[tr[-1]] >= start:
            raise RuntimeError(f"TRAIN_VALID_OVERLAP_{label}")
        if ts.iloc[va[-1]] >= UNSEEN_START:
            raise RuntimeError(f"AUG14_LEAK_IN_VALIDATION_{label}")
        fold = (fid, tr, va, ts.iloc[tr[-1]], ts.iloc[va[0]], ts.iloc[va[-1]])
        folds.append(fold)
        coverage.append({
            "fold": fid,
            "validation_block": label,
            "train_rows": int(len(tr)),
            "valid_rows": int(len(va)),
            "train_last_timestamp": ts.iloc[tr[-1]].isoformat(),
            "valid_first_timestamp": ts.iloc[va[0]].isoformat(),
            "valid_last_timestamp": ts.iloc[va[-1]].isoformat(),
        })
    return folds, pd.DataFrame(coverage)


def train_fold_5(bundle, fold, device: str, t0: float):
    import torch

    fid, tr, va, *_ = fold
    set_seed(CONFIG.seed + 100 + fid)
    m1, m5, m15, st, _ = preprocess(bundle, tr)
    arrays = (m1, m5, m15, st)
    tl = loader(arrays, bundle.y_s2, tr, True)
    vl = loader(arrays, bundle.y_s2, va, False)

    Model = model_class()
    model = Model(m1.shape[2], m5.shape[2], m15.shape[2], st.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG.learning_rate)
    loss_fn = torch.nn.SmoothL1Loss()

    best = -np.inf
    best_state = None
    best_epoch = 0
    bad = 0
    for ep in range(1, CONFIG.max_epochs + 1):
        model.train()
        for a, b, c, s, y in tl:
            a, b, c, s, y = a.to(device), b.to(device), c.to(device), s.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(a, b, c, s), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        yv, pv = evaluate(model, vl, device)
        sp = metrics(yv, pv)["spearman"]
        log(t0, f"S2 5/3/2 fold {fid}/{TOTAL_FOLDS} epoch {ep} spearman={sp:.5f} best={max(best, sp):.5f}")
        if np.isfinite(sp) and sp > best:
            best = sp
            best_epoch = ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= CONFIG.early_stopping_patience:
                break

    if best_state is None:
        raise RuntimeError(f"NO_MODEL_FOLD_{fid}")
    model.load_state_dict(best_state)
    yv, pv = evaluate(model, vl, device)
    pred = pd.DataFrame({
        "row_index": va,
        "timestamp": bundle.frame.iloc[va]["timestamp"].to_numpy(),
        "y_s2": yv,
        "s2_score": pv,
        "fold": fid,
    })
    met = metrics(yv, pv) | {
        "fold": fid,
        "train_rows": len(tr),
        "valid_rows": len(va),
        "best_epoch": best_epoch,
    }
    return pred, met, best_epoch


def main() -> None:
    ap = argparse.ArgumentParser(
        description="5/3/2 expanding walk-forward: March seed -> OOS Apr, May, Jun, Jul, Aug1-13 2026"
    )
    ap.add_argument(
        "--phase3a",
        type=Path,
        default=Path(r"D:\ORO_532_MAR13AUG_2026\phase3a_532"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(r"D:\ORO_532_MAR13AUG_2026\model_532_oos"),
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    market_path = args.phase3a / "market_state_mar13aug_2026.parquet"
    candidates_path = args.phase3a / "technical_candidates_mar13aug_valid.parquet"
    for p in (market_path, candidates_path):
        if not p.exists():
            raise SystemExit(f"MISSING_INPUT: {p}")

    market = pd.read_parquet(market_path)
    candidates = pd.read_parquet(candidates_path)
    market["timestamp"] = pd.to_datetime(market["timestamp"], utc=True)
    candidates["timestamp"] = pd.to_datetime(candidates["timestamp"], utc=True)
    if (market["timestamp"] >= UNSEEN_START).any():
        raise RuntimeError("AUG14_PRESENT_IN_MARKET_FEATURES")
    if (candidates["timestamp"] >= UNSEEN_START).any():
        raise RuntimeError("AUG14_PRESENT_IN_CANDIDATES")

    bundle = build_tensor_bundle(market, candidates)
    log(t0, f"5/3/2 Mar13Aug valid samples={len(bundle.frame):,}")

    folds, coverage = expanding_folds(bundle.frame)
    print("OOS_CONTRACT=MARCH_SEED_THEN_APR_MAY_JUN_JUL_AUG1_13", flush=True)
    print(coverage.to_string(index=False), flush=True)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"DEVICE={device}", flush=True)

    preds = []
    metrics_rows = []
    best_epochs = []
    for fold, (label, _, _) in zip(folds, BLOCKS):
        print(f"\n=== OOS BLOCK {label} ===", flush=True)
        pred, met, best_epoch = train_fold_5(bundle, fold, device, t0)
        pred["validation_block"] = label
        met["validation_block"] = label
        preds.append(pred)
        metrics_rows.append(met)
        best_epochs.append(best_epoch)

    out_pred = pd.concat(preds, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    out_metrics = pd.DataFrame(metrics_rows)

    ts_pred = pd.to_datetime(out_pred["timestamp"], utc=True)
    if (ts_pred >= UNSEEN_START).any():
        raise RuntimeError("AUG14_PRESENT_IN_OOS_PREDICTIONS")
    if out_pred["row_index"].duplicated().any():
        raise RuntimeError("DUPLICATE_OOS_ROW_INDEX")

    pred_path = args.out / "oos_s2_predictions_532_apr_aug13.csv.gz"
    metrics_path = args.out / "oos_s2_fold_metrics_532_apr_aug13.csv"
    coverage_path = args.out / "oos_coverage_532_apr_aug13.csv"
    out_pred.to_csv(pred_path, index=False, compression="gzip")
    out_metrics.to_csv(metrics_path, index=False)
    coverage.to_csv(coverage_path, index=False)

    median_best_epochs = int(round(float(np.median(best_epochs))))
    print("\nTRAIN_532_MAR13AUG_OOS=PASS")
    print(f"VALID_SAMPLES_MAR13AUG={len(bundle.frame):,}")
    print(f"OOS_PREDICTIONS_APR_AUG13={len(out_pred):,}")
    print("OOS_BLOCKS=2026-04,2026-05,2026-06,2026-07,2026-08-01_13")
    print("MARCH_ROLE=TRAINING_SEED_ONLY")
    print(f"MEDIAN_BEST_EPOCHS={median_best_epochs}")
    print("AUG14_USED_FOR_TRAINING=NO")
    print("AUG14_USED_FOR_VALIDATION=NO")
    print("THRESHOLD_SELECTED=NO")
    print(f"OUTPUT={args.out}")


if __name__ == "__main__":
    main()
