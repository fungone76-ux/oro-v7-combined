from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from short_memory_532.dataset import build_tensor_bundle

PHASE3A = Path(r"D:\ORO_532_2026_TUNING\phase3a_532")
MODEL_DIR = Path(r"D:\ORO_532_2026_TUNING\model_532_fullperiod_oos")
OUT = Path(r"D:\ORO_532_2026_TUNING\causal_percentile_screen_532")
TOP_PCTS = (10, 15, 20, 25, 30)
MONTHS = ["2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
TRADE_MONTHS = MONTHS[1:]  # Feb is calibration seed with currently available history.


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


def main() -> None:
    market = pd.read_parquet(PHASE3A / "market_state_2026.parquet")
    cand = pd.read_parquet(PHASE3A / "technical_candidates_all_2026.parquet")
    pred = pd.read_csv(MODEL_DIR / "oos_s2_predictions_532_feb_jun_monthly.csv.gz")
    market["timestamp"] = pd.to_datetime(market["timestamp"], utc=True)
    cand["timestamp"] = pd.to_datetime(cand["timestamp"], utc=True)
    pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True)
    pred["row_index"] = pd.to_numeric(pred["row_index"], errors="raise").astype(int)

    cand = cand[cand["timestamp"] < pd.Timestamp("2026-07-01T00:00:00Z")].copy()
    bundle = build_tensor_bundle(market, cand)
    if pred["row_index"].min() < 0 or pred["row_index"].max() >= len(bundle.frame):
        raise RuntimeError("OOS_ROW_INDEX_OUT_OF_RANGE")
    frame = bundle.frame.iloc[pred["row_index"].to_numpy()].copy().reset_index(drop=True)
    if not (pd.to_datetime(frame["timestamp"], utc=True).reset_index(drop=True) == pred["timestamp"].reset_index(drop=True)).all():
        raise RuntimeError("OOS_TIMESTAMP_ALIGNMENT_FAILED")
    frame["s2_score"] = pred["s2_score"].to_numpy(float)
    frame["y_s2"] = pred["y_s2"].to_numpy(float)
    frame["month"] = frame["timestamp"].dt.strftime("%Y-%m")

    market_days_by_month = {
        m: int(market[market["timestamp"].dt.strftime("%Y-%m").eq(m)]["timestamp"].dt.date.nunique())
        for m in MONTHS
    }

    summary_rows = []
    monthly_rows = []
    accepted_store = []
    for top_pct in TOP_PCTS:
        accepted_parts = []
        for month in TRADE_MONTHS:
            hist_months = [m for m in MONTHS if m < month]
            hist_scores = frame[frame["month"].isin(hist_months)]["s2_score"].astype(float)
            if hist_scores.empty:
                raise RuntimeError(f"NO_CAUSAL_CALIBRATION_HISTORY_{month}")
            threshold = float(hist_scores.quantile(1.0 - top_pct / 100.0))
            cur = frame[frame["month"].eq(month)].copy()
            acc = cur[cur["s2_score"] >= threshold].copy()
            acc["top_pct"] = top_pct
            acc["causal_threshold"] = threshold
            accepted_parts.append(acc)
            row = {"top_pct": top_pct, "month": month, "causal_threshold": threshold,
                   "calibration_months": ",".join(hist_months), "market_days": market_days_by_month[month]}
            row.update(summarize(acc, market_days_by_month[month]))
            monthly_rows.append(row)
        all_acc = pd.concat(accepted_parts, ignore_index=True) if accepted_parts else pd.DataFrame()
        accepted_store.append(all_acc)
        total_days = sum(market_days_by_month[m] for m in TRADE_MONTHS)
        row = {"top_pct": top_pct, "months_traded": ",".join(TRADE_MONTHS), "market_days": total_days}
        row.update(summarize(all_acc, total_days))
        summary_rows.append(row)
        print(f"TOP {top_pct:>2}% -> accepted={len(all_acc):,} trades/day={row['trades_per_market_day']:.2f} "
              f"1.5R_WR={row['r1_5_decisive_win_rate_pct']:.2f}% quality={row['mean_target_quality_r']:.4f}", flush=True)

    summary = pd.DataFrame(summary_rows)
    monthly = pd.DataFrame(monthly_rows)
    accepted = pd.concat(accepted_store, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "causal_percentile_summary.csv", index=False)
    monthly.to_csv(OUT / "causal_percentile_monthly.csv", index=False)
    accepted.to_parquet(OUT / "causal_percentile_accepted.parquet", index=False)

    print("\nFINAL 5/3/2 CAUSAL PERCENTILE SCREEN")
    print(summary.to_string(index=False))
    print("\nMONTHLY")
    print(monthly.to_string(index=False))
    print("FEBRUARY_ROLE=CALIBRATION_ONLY_WITH_CURRENT_HISTORY")
    print("TRADE_MONTHS=2026-03,2026-04,2026-05,2026-06")
    print("LOOKAHEAD_IN_THRESHOLD=NO")
    print("THRESHOLD_SELECTED=NO")
    print(f"OUTPUT={OUT}")


if __name__ == "__main__":
    main()
