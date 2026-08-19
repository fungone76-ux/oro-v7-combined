from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\ORO_611_MAR13AUG_2026")
PRED = ROOT / "model_611_oos" / "oos_predictions_611.parquet"
OUT = ROOT / "oos_percentile_audit_611"
TOP_PCTS = (15, 10, 5, 2, 1)


def summarize(g: pd.DataFrame, market_days: int) -> dict[str, float]:
    y = pd.to_numeric(g["y_s2"], errors="coerce")
    return {
        "accepted": int(len(g)),
        "signals_per_market_day": float(len(g) / market_days) if market_days else np.nan,
        "positive_quality_pct": float((y > 0).mean() * 100.0),
        "mean_quality_r": float(y.mean()),
        "median_quality_r": float(y.median()),
        "p75_quality_r": float(y.quantile(0.75)),
        "p90_quality_r": float(y.quantile(0.90)),
    }


def add_fold_local_percentile(df: pd.DataFrame, top_pct: int) -> pd.DataFrame:
    chunks = []
    for block, g in df.groupby("validation_block", sort=False):
        g = g.copy()
        # Frozen inside each OOS block: choose only by that block's score ranking.
        # This is an audit, not a deployable absolute threshold.
        q = 1.0 - top_pct / 100.0
        threshold = float(pd.to_numeric(g["s2_score"], errors="coerce").quantile(q))
        a = g[pd.to_numeric(g["s2_score"], errors="coerce") >= threshold].copy()
        a["top_pct"] = top_pct
        a["fold_local_threshold"] = threshold
        chunks.append(a)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def main() -> None:
    if not PRED.exists():
        raise SystemExit(f"MISSING_INPUT: {PRED}")
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PRED).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    required = {"timestamp", "validation_block", "direction", "trend_conflict", "direction_agreement", "s2_score", "y_s2"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"MISSING_COLUMNS: {missing}")
    df = df[np.isfinite(pd.to_numeric(df["s2_score"], errors="coerce")) & np.isfinite(pd.to_numeric(df["y_s2"], errors="coerce"))].copy()
    df["context"] = np.where(pd.to_numeric(df["trend_conflict"], errors="coerce").fillna(0).astype(int).eq(1), "CONFLICT", "AGREEMENT")
    df["market_date"] = df["timestamp"].dt.date

    market_days_by_block = df.groupby("validation_block")["market_date"].nunique().to_dict()
    total_market_days = int(df["market_date"].nunique())

    accepted_all = []
    overall_rows = []
    block_rows = []
    direction_rows = []
    context_rows = []
    context_direction_rows = []

    for pct in TOP_PCTS:
        acc = add_fold_local_percentile(df, pct)
        accepted_all.append(acc)
        r = {"top_pct": pct}
        r.update(summarize(acc, total_market_days))
        overall_rows.append(r)

        for block, g in acc.groupby("validation_block", sort=False):
            rr = {"top_pct": pct, "block": block, "market_days": int(market_days_by_block[block]), "threshold": float(g["fold_local_threshold"].iloc[0])}
            rr.update(summarize(g, int(market_days_by_block[block])))
            block_rows.append(rr)
        for direction, g in acc.groupby("direction", sort=False):
            rr = {"top_pct": pct, "direction": direction}
            rr.update(summarize(g, total_market_days))
            direction_rows.append(rr)
        for context, g in acc.groupby("context", sort=False):
            rr = {"top_pct": pct, "context": context}
            rr.update(summarize(g, total_market_days))
            context_rows.append(rr)
        for (context, direction), g in acc.groupby(["context", "direction"], sort=False):
            rr = {"top_pct": pct, "context": context, "direction": direction}
            rr.update(summarize(g, total_market_days))
            context_direction_rows.append(rr)

    overall = pd.DataFrame(overall_rows)
    by_block = pd.DataFrame(block_rows)
    by_direction = pd.DataFrame(direction_rows)
    by_context = pd.DataFrame(context_rows)
    by_context_direction = pd.DataFrame(context_direction_rows)
    accepted = pd.concat(accepted_all, ignore_index=True)

    overall.to_csv(OUT / "overall.csv", index=False)
    by_block.to_csv(OUT / "by_block.csv", index=False)
    by_direction.to_csv(OUT / "by_direction.csv", index=False)
    by_context.to_csv(OUT / "by_context.csv", index=False)
    by_context_direction.to_csv(OUT / "by_context_direction.csv", index=False)
    accepted.to_parquet(OUT / "accepted_fold_local_percentiles.parquet", index=False)

    print("611_OOS_PERCENTILE_AUDIT=PASS")
    print(f"OOS_PREDICTIONS={len(df):,}")
    print(f"MARKET_DAYS={total_market_days}")
    print("SELECTION=FOLD_LOCAL_SCORE_PERCENTILE_AUDIT_ONLY")
    print("ABSOLUTE_DEPLOYMENT_THRESHOLD_SELECTED=NO")
    print("AUG14_USED=NO")
    print("\nOVERALL")
    print(overall.to_string(index=False))
    print("\nBY_BLOCK")
    print(by_block.to_string(index=False))
    print("\nBY_DIRECTION")
    print(by_direction.to_string(index=False))
    print("\nBY_CONTEXT")
    print(by_context.to_string(index=False))
    print("\nBY_CONTEXT_DIRECTION")
    print(by_context_direction.to_string(index=False))
    print(f"OUTPUT={OUT}")


if __name__ == "__main__":
    main()
