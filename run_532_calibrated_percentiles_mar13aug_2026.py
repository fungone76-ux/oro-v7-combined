from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import pandas as pd

from short_memory_532.config import CONFIG
from short_memory_532.dataset import build_tensor_bundle
from run_532_train_s2_2026 import evaluate, loader, log, metrics, model_class, preprocess, set_seed
from run_532_train_oos_mar13aug_2026 import BLOCKS, expanding_folds

PHASE3A = Path(r"D:\ORO_532_MAR13AUG_2026\phase3a_532")
OUT = Path(r"D:\ORO_532_MAR13AUG_2026\percentile_calibration_532")
TOP_PCTS = (10, 15, 20, 25, 30)
UNSEEN_START = pd.Timestamp("2026-08-14T00:00:00Z")


def barrier_stats(g: pd.DataFrame, col: str, prefix: str) -> dict[str, float | int]:
    s = g[col].astype(str)
    wins = int((s == "WIN").sum())
    losses = int((s == "LOSS").sum())
    decisive = wins + losses
    return {
        f"{prefix}_wins": wins,
        f"{prefix}_losses": losses,
        f"{prefix}_decisive_win_rate_pct": wins / decisive * 100.0 if decisive else float("nan"),
    }


def summarize(g: pd.DataFrame, market_days: int) -> dict[str, float | int]:
    y = pd.to_numeric(g["y_s2"], errors="coerce")
    dfr = pd.to_numeric(g["directional_future_return_15m"], errors="coerce")
    out: dict[str, float | int] = {
        "accepted": int(len(g)),
        "trades_per_market_day": float(len(g) / market_days) if market_days else float("nan"),
        "mean_target_quality_r": float(y.mean()) if len(g) else float("nan"),
        "median_target_quality_r": float(y.median()) if len(g) else float("nan"),
        "positive_15m_pct": float((dfr > 0).mean() * 100.0) if len(g) else float("nan"),
        "longs": int((g["direction"].astype(str) == "LONG").sum()),
        "shorts": int((g["direction"].astype(str) == "SHORT").sum()),
    }
    out.update(barrier_stats(g, "barrier_1r_outcome", "r1"))
    out.update(barrier_stats(g, "barrier_1_5r_outcome", "r1_5"))
    return out


def train_fold_with_train_scores(bundle, fold, device: str, t0: float):
    import torch

    fid, tr, va, *_ = fold
    set_seed(CONFIG.seed + 100 + fid)
    m1, m5, m15, st, _ = preprocess(bundle, tr)
    arrays = (m1, m5, m15, st)
    tl = loader(arrays, bundle.y_s2, tr, True)
    tr_eval = loader(arrays, bundle.y_s2, tr, False)
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
        log(t0, f"CAL S2 fold {fid}/5 epoch {ep} spearman={sp:.5f} best={max(best, sp):.5f}")
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
    yt, pt = evaluate(model, tr_eval, device)
    yv, pv = evaluate(model, vl, device)
    return pt.astype(float), pv.astype(float), best_epoch, float(metrics(yv, pv)["spearman"])


def main() -> None:
    market_path = PHASE3A / "market_state_mar13aug_2026.parquet"
    candidates_path = PHASE3A / "technical_candidates_mar13aug_valid.parquet"
    for p in (market_path, candidates_path):
        if not p.exists():
            raise SystemExit(f"MISSING_INPUT: {p}")

    market = pd.read_parquet(market_path)
    candidates = pd.read_parquet(candidates_path)
    market["timestamp"] = pd.to_datetime(market["timestamp"], utc=True)
    candidates["timestamp"] = pd.to_datetime(candidates["timestamp"], utc=True)
    if (market["timestamp"] >= UNSEEN_START).any() or (candidates["timestamp"] >= UNSEEN_START).any():
        raise RuntimeError("AUG14_LEAK_INPUT")

    bundle = build_tensor_bundle(market, candidates)
    folds, coverage = expanding_folds(bundle.frame)
    t0 = time.perf_counter()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 78)
    print("5/3/2 FOLD-LOCAL CAUSAL PERCENTILE CALIBRATION")
    print(f"DEVICE={device} | samples={len(bundle.frame):,}")
    print("CONTRACT=threshold percentile from SAME FOLD MODEL on TRAIN scores -> apply to OOS block")
    print("AUG14_USED=NO")
    print("=" * 78, flush=True)

    market_days = {}
    mt = market["timestamp"]
    for label, start, end in BLOCKS:
        market_days[label] = int(market[(mt >= start) & (mt < end)]["timestamp"].dt.date.nunique())

    rows = []
    accepted_store = []
    fold_meta = []
    for fold, (label, start, end) in zip(folds, BLOCKS):
        fid, tr, va, *_ = fold
        print(f"\n=== CALIBRATE/OOS {label} ===", flush=True)
        train_scores, valid_scores, best_epoch, spearman = train_fold_with_train_scores(bundle, fold, device, t0)
        base = bundle.frame.iloc[va].copy().reset_index(drop=True)
        base["s2_score"] = valid_scores
        base["validation_block"] = label
        fold_meta.append({"fold": fid, "validation_block": label, "train_rows": len(tr), "valid_rows": len(va), "best_epoch": best_epoch, "spearman": spearman})

        for top_pct in TOP_PCTS:
            threshold = float(np.quantile(train_scores, 1.0 - top_pct / 100.0))
            acc = base[base["s2_score"] >= threshold].copy()
            acc["top_pct"] = top_pct
            acc["fold_local_threshold"] = threshold
            accepted_store.append(acc)
            r = {
                "fold": fid,
                "validation_block": label,
                "top_pct": top_pct,
                "fold_local_threshold": threshold,
                "train_rows": len(tr),
                "valid_rows": len(va),
                "market_days": market_days[label],
                "best_epoch": best_epoch,
                "spearman": spearman,
            }
            r.update(summarize(acc, market_days[label]))
            rows.append(r)
            print(
                f"TOP {top_pct:>2}% threshold={threshold:+.6f} accepted={len(acc):4d} "
                f"trades/day={r['trades_per_market_day']:.2f} 1.5R_WR={r['r1_5_decisive_win_rate_pct']:.2f}% "
                f"quality={r['mean_target_quality_r']:+.4f}",
                flush=True,
            )

    detail = pd.DataFrame(rows)
    accepted = pd.concat(accepted_store, ignore_index=True) if accepted_store else pd.DataFrame()
    summary_rows = []
    for top_pct in TOP_PCTS:
        g = accepted[accepted["top_pct"] == top_pct].copy()
        days = sum(market_days.values())
        r = {"top_pct": top_pct, "market_days": days}
        r.update(summarize(g, days))
        summary_rows.append(r)
    summary = pd.DataFrame(summary_rows)

    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "fold_local_percentile_summary.csv", index=False)
    detail.to_csv(OUT / "fold_local_percentile_by_block.csv", index=False)
    pd.DataFrame(fold_meta).to_csv(OUT / "fold_metrics.csv", index=False)
    accepted.to_parquet(OUT / "accepted_candidates.parquet", index=False)

    print("\nFINAL 5/3/2 FOLD-LOCAL PERCENTILE SUMMARY")
    print(summary.to_string(index=False))
    print("\nBY_BLOCK")
    print(detail.to_string(index=False))
    print("CALIBRATION_LOOKAHEAD=NO")
    print("CROSS_MODEL_SCORE_MIXING=NO")
    print("AUG14_USED_FOR_CALIBRATION=NO")
    print("PERCENTILE_SELECTED=NO")
    print(f"OUTPUT={OUT}")


if __name__ == "__main__":
    main()
