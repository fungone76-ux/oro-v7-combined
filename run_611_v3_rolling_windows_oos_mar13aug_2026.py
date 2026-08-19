from __future__ import annotations

"""6/1/1 V3 rolling-window classifier comparison.

Research only. No live execution.

Changes vs V2:
- same 6/1/1 features and architecture;
- same GOOD target: label_quality_r >= 0.80R;
- same causal 10/20/30 signals-per-market-day rate calibration;
- ONLY the training memory changes: rolling 14, 30, or 45 calendar days
  immediately preceding each OOS block, with the same purge+embargo gap.

This lets us test whether recent-regime training improves robustness without
changing the target, model, entry concept, or rate-selection contract.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from short_memory_611.config import CONFIG
from short_memory_611.dataset import build_tensor_bundle
from run_611_v2_classifier_oos_mar13aug_2026 import (
    GOOD_QUALITY_R,
    TARGET_SIGNALS_PER_DAY,
    loader,
    market_days,
    model_class,
    predict,
    preprocess,
    safe_ap,
    safe_auc,
    set_seed,
    summarize_selected,
    threshold_for_rate,
)

ROOT = Path(r"D:\ORO_611_MAR13AUG_2026")
OUT = ROOT / "v3_rolling_windows_oos"
ROLLING_WINDOWS_DAYS = (14, 30, 45)

BLOCKS = [
    ("2026-04", pd.Timestamp("2026-04-01T00:00:00Z"), pd.Timestamp("2026-05-01T00:00:00Z")),
    ("2026-05", pd.Timestamp("2026-05-01T00:00:00Z"), pd.Timestamp("2026-06-01T00:00:00Z")),
    ("2026-06", pd.Timestamp("2026-06-01T00:00:00Z"), pd.Timestamp("2026-07-01T00:00:00Z")),
    ("2026-07", pd.Timestamp("2026-07-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
    ("2026-08-01_13", pd.Timestamp("2026-08-01T00:00:00Z"), pd.Timestamp("2026-08-14T00:00:00Z")),
]


def log(t0: float, msg: str) -> None:
    print(f"+{time.perf_counter()-t0:7.1f}s | {msg}", flush=True)


def rolling_fold_indices(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, window_days: int):
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    gap = pd.Timedelta(minutes=CONFIG.purge_minutes + CONFIG.embargo_minutes)
    train_end = start - gap
    train_start = train_end - pd.Timedelta(days=window_days)
    tr = np.flatnonzero((ts >= train_start) & (ts < train_end))
    va = np.flatnonzero((ts >= start) & (ts < end))
    if len(tr) == 0 or len(va) == 0:
        raise RuntimeError(
            f"EMPTY_ROLLING_FOLD window={window_days} start={start} train={len(tr)} valid={len(va)}"
        )
    return tr, va, train_start, train_end


def main() -> None:
    import torch

    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = build_tensor_bundle(ROOT)
    frame = bundle.frame.copy().reset_index(drop=True)
    quality = pd.to_numeric(frame["label_quality_r"], errors="coerce").to_numpy(np.float32)
    y = (quality >= GOOD_QUALITY_R).astype(np.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Model = model_class()

    print("=" * 78)
    print("6/1/1 V3 ROLLING-WINDOW CLASSIFIER - CAUSAL OOS")
    print(f"WINDOWS_DAYS={ROLLING_WINDOWS_DAYS}")
    print(f"TARGET=GOOD if label_quality_r >= {GOOD_QUALITY_R:.2f}R")
    print(f"RATE_TARGETS={TARGET_SIGNALS_PER_DAY} signals/market-day")
    print("ONLY_CHANGE_FROM_V2=ROLLING_TRAINING_MEMORY")
    print("RATE_THRESHOLDS=SAME_MODEL_ROLLING_TRAIN_PREDICTIONS_ONLY")
    print("AUG14_USED=NO")
    print("=" * 78)
    print(f"VALID_SAMPLES={len(frame):,} | GOOD_BASE_RATE={float(y.mean()*100):.2f}% | DEVICE={device}")

    predictions_all: list[pd.DataFrame] = []
    fold_metric_rows: list[dict[str, object]] = []
    rate_rows: list[dict[str, object]] = []

    for window_days in ROLLING_WINDOWS_DAYS:
        print(f"\n{'#' * 78}\nROLLING WINDOW = {window_days} DAYS\n{'#' * 78}", flush=True)
        for fid, (label, a, b) in enumerate(BLOCKS, 1):
            tr, va, train_start, train_end = rolling_fold_indices(frame, a, b, window_days)
            tr_days = market_days(frame, tr)
            va_days = market_days(frame, va)
            print(
                f"\n=== W{window_days} OOS {label} | train={len(tr):,} rows/{tr_days} days "
                f"[{train_start.isoformat()} -> {train_end.isoformat()}] | valid={len(va):,}/{va_days} days ===",
                flush=True,
            )

            set_seed(CONFIG.seed + window_days * 100 + fid)
            arrays, _ = preprocess(bundle, tr)
            model = Model(
                arrays[0].shape[2], arrays[1].shape[2], arrays[2].shape[2], arrays[3].shape[1]
            ).to(device)

            pos = float(y[tr].sum())
            neg = float(len(tr) - pos)
            pos_weight = neg / max(pos, 1.0)
            loss_fn = torch.nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device)
            )
            opt = torch.optim.Adam(model.parameters(), lr=CONFIG.learning_rate)
            tl = loader(arrays, y, tr, True)
            vl = loader(arrays, y, va, False)
            tr_eval = loader(arrays, y, tr, False)

            best = -np.inf
            best_state = None
            best_epoch = 0
            bad = 0
            for ep in range(1, CONFIG.max_epochs + 1):
                model.train()
                for x1, x5, x15, xs, yy in tl:
                    x1, x5, x15, xs, yy = (
                        x1.to(device), x5.to(device), x15.to(device), xs.to(device), yy.to(device)
                    )
                    opt.zero_grad(set_to_none=True)
                    logits = model(x1, x5, x15, xs)
                    loss = loss_fn(logits, yy)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()

                yv, pv = predict(model, vl, device)
                ap = safe_ap(yv, pv)
                auc = safe_auc(yv, pv)
                score = ap if np.isfinite(ap) else -np.inf
                log(
                    t0,
                    f"V3 W{window_days} fold {fid}/5 epoch {ep} AP={ap:.5f} AUC={auc:.5f} "
                    f"best_AP={max(best, score):.5f}",
                )
                if score > best:
                    best = score
                    best_epoch = ep
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    bad = 0
                else:
                    bad += 1
                    if bad >= CONFIG.early_stopping_patience:
                        break

            if best_state is None:
                raise RuntimeError(f"NO_MODEL_W{window_days}_{label}")

            model.load_state_dict(best_state)
            yt, pt = predict(model, tr_eval, device)
            yv, pv = predict(model, vl, device)

            fv = frame.iloc[va].copy().reset_index(drop=True)
            fv["good_label"] = yv.astype(np.int8)
            fv["good_probability"] = pv
            fv["window_days"] = window_days
            fv["fold"] = fid
            fv["validation_block"] = label
            predictions_all.append(fv)

            fold_metric_rows.append({
                "window_days": window_days,
                "fold": fid,
                "block": label,
                "train_rows": len(tr),
                "train_days": tr_days,
                "valid_rows": len(va),
                "valid_days": va_days,
                "train_start": train_start.isoformat(),
                "train_end": train_end.isoformat(),
                "best_epoch": best_epoch,
                "train_good_rate_pct": float(yt.mean() * 100.0),
                "valid_good_rate_pct": float(yv.mean() * 100.0),
                "oos_ap": safe_ap(yv, pv),
                "oos_auc": safe_auc(yv, pv),
            })

            for desired in TARGET_SIGNALS_PER_DAY:
                thr = threshold_for_rate(pt, tr_days, desired)
                mask = pv >= thr
                summary = summarize_selected(fv, mask, va_days)
                row = {
                    "window_days": window_days,
                    "fold": fid,
                    "block": label,
                    "desired_signals_per_day": desired,
                    "train_calibrated_probability_threshold": thr,
                    "train_days": tr_days,
                    "valid_days": va_days,
                }
                row.update(summary)
                rate_rows.append(row)
                print(
                    f"W{window_days} RATE {desired:02d}/day | thr={thr:.6f} | OOS={summary['accepted']:,} "
                    f"actual/day={summary['signals_per_market_day']:.2f} | GOOD={summary['good_rate_pct']:.2f}% "
                    f"meanQ={summary['mean_quality_r']:.4f}R | L/S={summary['longs']}/{summary['shorts']} "
                    f"agree/conflict={summary['agreement']}/{summary['conflicts']}",
                    flush=True,
                )

    predictions = pd.concat(predictions_all, ignore_index=True).sort_values(
        ["window_days", "timestamp"]
    )
    fold_metrics = pd.DataFrame(fold_metric_rows)
    rates = pd.DataFrame(rate_rows)

    predictions.to_parquet(OUT / "oos_predictions_611_v3_rolling.parquet", index=False)
    fold_metrics.to_csv(OUT / "oos_fold_metrics_611_v3_rolling.csv", index=False)
    rates.to_csv(OUT / "causal_rate_screen_611_v3_rolling.csv", index=False)

    overall_rows: list[dict[str, object]] = []
    for (window_days, desired), g in rates.groupby(
        ["window_days", "desired_signals_per_day"], sort=True
    ):
        accepted = pd.to_numeric(g["accepted"], errors="coerce").fillna(0).to_numpy(float)
        total_acc = int(accepted.sum())
        total_days = int(pd.to_numeric(g["valid_days"], errors="coerce").fillna(0).sum())
        if accepted.sum() > 0:
            good = float(np.average(pd.to_numeric(g["good_rate_pct"], errors="coerce"), weights=accepted))
            meanq = float(np.average(pd.to_numeric(g["mean_quality_r"], errors="coerce"), weights=accepted))
        else:
            good = float("nan")
            meanq = float("nan")
        overall_rows.append({
            "window_days": int(window_days),
            "desired_signals_per_day": int(desired),
            "accepted": total_acc,
            "market_days": total_days,
            "actual_signals_per_market_day": float(total_acc / total_days) if total_days else float("nan"),
            "weighted_good_rate_pct": good,
            "weighted_mean_quality_r": meanq,
        })

    overall = pd.DataFrame(overall_rows)
    overall.to_csv(OUT / "overall_rate_screen_611_v3_rolling.csv", index=False)

    summary = {
        "method": "611_V3_ROLLING_BINARY_CLASSIFIER",
        "rolling_windows_days": list(ROLLING_WINDOWS_DAYS),
        "good_quality_r_threshold": GOOD_QUALITY_R,
        "target_signal_rates_per_day": list(TARGET_SIGNALS_PER_DAY),
        "valid_samples": int(len(frame)),
        "rate_threshold_contract": "SAME_MODEL_ROLLING_TRAIN_PREDICTIONS_ONLY",
        "only_change_from_v2": "ROLLING_TRAINING_MEMORY",
        "aug14_used": False,
        "deployment_window_selected": False,
        "deployment_threshold_selected": False,
    }
    (OUT / "SUMMARY_611_V3_ROLLING.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n611_V3_ROLLING_OOS=PASS")
    print("AUG14_USED=NO")
    print("DEPLOYMENT_WINDOW_SELECTED=NO")
    print("DEPLOYMENT_THRESHOLD_SELECTED=NO")
    print("\nOVERALL_RATE_SCREEN")
    print(overall.to_string(index=False))
    print("\nBY_BLOCK_RATE_SCREEN")
    cols = [
        "window_days", "block", "desired_signals_per_day",
        "train_calibrated_probability_threshold", "accepted",
        "signals_per_market_day", "good_rate_pct", "mean_quality_r",
        "longs", "shorts", "agreement", "conflicts",
    ]
    print(rates[cols].to_string(index=False))
    print(f"OUTPUT={OUT}")


if __name__ == "__main__":
    main()
