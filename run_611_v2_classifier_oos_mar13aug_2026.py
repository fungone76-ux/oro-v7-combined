from __future__ import annotations

"""6/1/1 V2 binary classifier with causal rate calibration.

Research only. Does not modify or invoke either demo runtime.

Target contract (frozen before this run):
    GOOD = label_quality_r >= 0.80

For each walk-forward fold, probability thresholds for ~10/~20/~30 accepted
signals per market day are calibrated ONLY on training predictions produced by
the same fold model, then applied unchanged to the next OOS block.
"""

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

from short_memory_611.config import CONFIG, MODEL_ARCHITECTURE
from short_memory_611.dataset import build_tensor_bundle

ROOT = Path(r"D:\ORO_611_MAR13AUG_2026")
OUT = ROOT / "v2_classifier_oos"
GOOD_QUALITY_R = 0.80
TARGET_SIGNALS_PER_DAY = (10, 20, 30)


def log(t0: float, msg: str) -> None:
    print(f"+{time.perf_counter()-t0:7.1f}s | {msg}", flush=True)


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scale_seq(x: np.ndarray, tr: np.ndarray):
    flat = x[tr].reshape(-1, x.shape[-1])
    med = np.nanmedian(flat, axis=0)
    sc = StandardScaler().fit(np.where(np.isnan(flat), med, flat))
    all_flat = x.reshape(-1, x.shape[-1])
    out = sc.transform(np.where(np.isnan(all_flat), med, all_flat)).reshape(x.shape).astype(np.float32)
    return sc, out


def preprocess(bundle, tr):
    s1, m1 = scale_seq(bundle.m1, tr)
    s5, m5 = scale_seq(bundle.m5, tr)
    s15, m15 = scale_seq(bundle.m15, tr)
    sn = bundle.static_num
    med = np.nanmedian(sn[tr], axis=0)
    ss = StandardScaler().fit(np.where(np.isnan(sn[tr]), med, sn[tr]))
    st = ss.transform(np.where(np.isnan(sn), med, sn)).astype(np.float32)
    return (m1, m5, m15, st), {"m1": s1, "m5": s5, "m15": s15, "static": ss}


def model_class():
    import torch

    class Model(torch.nn.Module):
        def __init__(self, d1, d5, d15, ds):
            super().__init__()
            self.m1 = torch.nn.LSTM(d1, MODEL_ARCHITECTURE["m1_lstm_hidden"], batch_first=True)
            self.m5 = torch.nn.LSTM(d5, MODEL_ARCHITECTURE["m5_lstm_hidden"], batch_first=True)
            self.m15 = torch.nn.LSTM(d15, MODEL_ARCHITECTURE["m15_lstm_hidden"], batch_first=True)
            merged = (
                MODEL_ARCHITECTURE["m1_lstm_hidden"]
                + MODEL_ARCHITECTURE["m5_lstm_hidden"]
                + MODEL_ARCHITECTURE["m15_lstm_hidden"]
                + ds
            )
            self.head = torch.nn.Sequential(
                torch.nn.Linear(merged, MODEL_ARCHITECTURE["dense"]),
                torch.nn.ReLU(),
                torch.nn.Dropout(MODEL_ARCHITECTURE["dropout"]),
                torch.nn.Linear(MODEL_ARCHITECTURE["dense"], 1),
            )

        def forward(self, a, b, c, s):
            _, (h1, _) = self.m1(a)
            _, (h5, _) = self.m5(b)
            _, (h15, _) = self.m15(c)
            return self.head(torch.cat([h1[-1], h5[-1], h15[-1], s], dim=1)).squeeze(1)

    return Model


def loader(arrays, y, idx, shuffle):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    ds = TensorDataset(
        *(torch.tensor(a[idx], dtype=torch.float32) for a in arrays),
        torch.tensor(y[idx], dtype=torch.float32),
    )
    return DataLoader(
        ds,
        batch_size=CONFIG.batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(CONFIG.seed),
    )


def predict(model, ld, device):
    import torch

    ys, ps = [], []
    model.eval()
    with torch.no_grad():
        for a, b, c, s, y in ld:
            logits = model(a.to(device), b.to(device), c.to(device), s.to(device))
            p = torch.sigmoid(logits)
            ys.append(y.numpy())
            ps.append(p.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def safe_auc(y, p):
    try:
        return float(roc_auc_score(y, p))
    except Exception:
        return float("nan")


def safe_ap(y, p):
    try:
        return float(average_precision_score(y, p))
    except Exception:
        return float("nan")


def make_folds(frame: pd.DataFrame):
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    blocks = [
        ("2026-04", pd.Timestamp("2026-04-01T00:00:00Z"), pd.Timestamp("2026-05-01T00:00:00Z")),
        ("2026-05", pd.Timestamp("2026-05-01T00:00:00Z"), pd.Timestamp("2026-06-01T00:00:00Z")),
        ("2026-06", pd.Timestamp("2026-06-01T00:00:00Z"), pd.Timestamp("2026-07-01T00:00:00Z")),
        ("2026-07", pd.Timestamp("2026-07-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
        ("2026-08-01_13", pd.Timestamp("2026-08-01T00:00:00Z"), pd.Timestamp("2026-08-14T00:00:00Z")),
    ]
    gap = pd.Timedelta(minutes=CONFIG.purge_minutes + CONFIG.embargo_minutes)
    folds = []
    for fid, (label, a, b) in enumerate(blocks, 1):
        tr = np.flatnonzero(ts < (a - gap))
        va = np.flatnonzero((ts >= a) & (ts < b))
        if len(tr) == 0 or len(va) == 0:
            raise RuntimeError(f"EMPTY_FOLD_{label}")
        folds.append((fid, label, tr, va))
    return folds


def market_days(frame: pd.DataFrame, idx: np.ndarray) -> int:
    ts = pd.to_datetime(frame.iloc[idx]["timestamp"], utc=True)
    return int(ts.dt.date.nunique())


def threshold_for_rate(train_probs: np.ndarray, train_days: int, desired_per_day: int) -> float:
    if train_days <= 0 or len(train_probs) == 0:
        return 1.0
    desired = max(1, int(round(desired_per_day * train_days)))
    desired = min(desired, len(train_probs))
    # kth largest probability; causal because only training probabilities are used.
    kth = len(train_probs) - desired
    return float(np.partition(train_probs, kth)[kth])


def summarize_selected(frame: pd.DataFrame, mask: np.ndarray, block_days: int) -> dict[str, float]:
    g = frame.loc[mask].copy()
    if g.empty:
        return {
            "accepted": 0,
            "signals_per_market_day": 0.0,
            "good_rate_pct": float("nan"),
            "mean_quality_r": float("nan"),
            "median_quality_r": float("nan"),
            "longs": 0,
            "shorts": 0,
            "agreement": 0,
            "conflicts": 0,
        }
    yq = pd.to_numeric(g["label_quality_r"], errors="coerce")
    return {
        "accepted": int(len(g)),
        "signals_per_market_day": float(len(g) / block_days) if block_days else float("nan"),
        "good_rate_pct": float((yq >= GOOD_QUALITY_R).mean() * 100.0),
        "mean_quality_r": float(yq.mean()),
        "median_quality_r": float(yq.median()),
        "longs": int((g["direction"].astype(str) == "LONG").sum()),
        "shorts": int((g["direction"].astype(str) == "SHORT").sum()),
        "agreement": int(pd.to_numeric(g["direction_agreement"], errors="coerce").fillna(0).astype(int).sum()),
        "conflicts": int(pd.to_numeric(g["trend_conflict"], errors="coerce").fillna(0).astype(int).sum()),
    }


def main():
    import torch

    t0 = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    bundle = build_tensor_bundle(ROOT)
    frame = bundle.frame.copy().reset_index(drop=True)
    quality = pd.to_numeric(frame["label_quality_r"], errors="coerce").to_numpy(np.float32)
    y = (quality >= GOOD_QUALITY_R).astype(np.float32)
    folds = make_folds(frame)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 78)
    print("6/1/1 V2 BINARY CLASSIFIER - CAUSAL WALK-FORWARD")
    print(f"TARGET=GOOD if label_quality_r >= {GOOD_QUALITY_R:.2f}R")
    print(f"RATE_TARGETS={TARGET_SIGNALS_PER_DAY} signals/market-day")
    print("RATE_THRESHOLDS=calibrated on TRAIN predictions only, then frozen for OOS block")
    print("AUG14_USED=NO")
    print("=" * 78)
    print(f"VALID_SAMPLES={len(frame):,} | GOOD_BASE_RATE={float(y.mean()*100):.2f}% | DEVICE={device}")

    Model = model_class()
    all_preds = []
    fold_metrics = []
    rate_rows = []
    best_epochs = []

    for fid, label, tr, va in folds:
        print(f"\n=== OOS BLOCK {label} ===", flush=True)
        set_seed(CONFIG.seed + 100 + fid)
        arrays, _scalers = preprocess(bundle, tr)
        model = Model(arrays[0].shape[2], arrays[1].shape[2], arrays[2].shape[2], arrays[3].shape[1]).to(device)

        pos = float(y[tr].sum())
        neg = float(len(tr) - pos)
        pos_weight = neg / max(pos, 1.0)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32, device=device))
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
            for a, b, c, s, yy in tl:
                a, b, c, s, yy = a.to(device), b.to(device), c.to(device), s.to(device), yy.to(device)
                opt.zero_grad(set_to_none=True)
                logits = model(a, b, c, s)
                loss = loss_fn(logits, yy)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()

            yv, pv = predict(model, vl, device)
            ap = safe_ap(yv, pv)
            auc = safe_auc(yv, pv)
            score = ap if np.isfinite(ap) else -np.inf
            log(t0, f"V2 fold {fid}/5 epoch {ep} AP={ap:.5f} AUC={auc:.5f} best_AP={max(best, score):.5f}")
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
            raise RuntimeError(f"NO_MODEL_{label}")
        model.load_state_dict(best_state)
        yt, pt = predict(model, tr_eval, device)
        yv, pv = predict(model, vl, device)

        tr_days = market_days(frame, tr)
        va_days = market_days(frame, va)
        fv = frame.iloc[va].copy().reset_index(drop=True)
        fv["good_label"] = yv.astype(np.int8)
        fv["good_probability"] = pv
        fv["fold"] = fid
        fv["validation_block"] = label
        all_preds.append(fv)

        fold_metrics.append({
            "fold": fid,
            "block": label,
            "train_rows": len(tr),
            "valid_rows": len(va),
            "train_days": tr_days,
            "valid_days": va_days,
            "best_epoch": best_epoch,
            "train_good_rate_pct": float(yt.mean() * 100.0),
            "valid_good_rate_pct": float(yv.mean() * 100.0),
            "oos_ap": safe_ap(yv, pv),
            "oos_auc": safe_auc(yv, pv),
        })
        best_epochs.append(best_epoch)

        for desired in TARGET_SIGNALS_PER_DAY:
            thr = threshold_for_rate(pt, tr_days, desired)
            mask = pv >= thr
            summary = summarize_selected(fv, mask, va_days)
            row = {
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
                f"RATE {desired:02d}/day | train_thr={thr:.6f} | OOS accepted={summary['accepted']:,} "
                f"actual/day={summary['signals_per_market_day']:.2f} | GOOD={summary['good_rate_pct']:.2f}% "
                f"meanQ={summary['mean_quality_r']:.4f}R | L/S={summary['longs']}/{summary['shorts']} "
                f"agree/conflict={summary['agreement']}/{summary['conflicts']}",
                flush=True,
            )

    preds = pd.concat(all_preds, ignore_index=True).sort_values("timestamp")
    metrics = pd.DataFrame(fold_metrics)
    rates = pd.DataFrame(rate_rows)
    preds.to_parquet(OUT / "oos_predictions_611_v2.parquet", index=False)
    metrics.to_csv(OUT / "oos_fold_metrics_611_v2.csv", index=False)
    rates.to_csv(OUT / "causal_rate_screen_611_v2.csv", index=False)

    overall_rows = []
    for desired, g in rates.groupby("desired_signals_per_day", sort=True):
        total_acc = int(g["accepted"].sum())
        total_days = int(g["valid_days"].sum())
        # Weighted summaries by accepted count; block rows are already OOS-only.
        w = g["accepted"].to_numpy(float)
        if w.sum() > 0:
            good = float(np.average(g["good_rate_pct"], weights=w))
            meanq = float(np.average(g["mean_quality_r"], weights=w))
        else:
            good = float("nan")
            meanq = float("nan")
        overall_rows.append({
            "desired_signals_per_day": int(desired),
            "accepted": total_acc,
            "market_days": total_days,
            "actual_signals_per_market_day": float(total_acc / total_days) if total_days else float("nan"),
            "weighted_good_rate_pct": good,
            "weighted_mean_quality_r": meanq,
            "longs": int(g["longs"].sum()),
            "shorts": int(g["shorts"].sum()),
            "agreement": int(g["agreement"].sum()),
            "conflicts": int(g["conflicts"].sum()),
        })
    overall = pd.DataFrame(overall_rows)
    overall.to_csv(OUT / "overall_rate_screen_611_v2.csv", index=False)

    med_epoch = int(round(float(np.median(best_epochs))))
    summary_json = {
        "method": "611_V2_BINARY_CLASSIFIER",
        "good_quality_r_threshold": GOOD_QUALITY_R,
        "target_signal_rates_per_day": list(TARGET_SIGNALS_PER_DAY),
        "valid_samples": int(len(frame)),
        "oos_predictions": int(len(preds)),
        "median_best_epochs": med_epoch,
        "rate_threshold_contract": "SAME_FOLD_MODEL_TRAIN_PREDICTIONS_ONLY",
        "aug14_used": False,
        "deployment_threshold_selected": False,
    }
    (OUT / "SUMMARY_611_V2.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    print("\n611_V2_CLASSIFIER_OOS=PASS")
    print(f"VALID_SAMPLES={len(frame):,}")
    print(f"OOS_PREDICTIONS={len(preds):,}")
    print(f"MEDIAN_BEST_EPOCHS={med_epoch}")
    print("AUG14_USED_FOR_TRAINING=NO")
    print("AUG14_USED_FOR_VALIDATION=NO")
    print("DEPLOYMENT_THRESHOLD_SELECTED=NO")
    print("\nOVERALL_CAUSAL_RATE_SCREEN")
    print(overall.to_string(index=False))
    print("\nBY_BLOCK_RATE_SCREEN")
    print(rates.to_string(index=False))
    print(f"OUTPUT={OUT}")


if __name__ == "__main__":
    main()
