from __future__ import annotations

from pathlib import Path

import pandas as pd

from short_memory_532.config import CONFIG
from short_memory_532.dataset import build_tensor_bundle
from run_532_threshold_oos_screen import summarize, family

PHASE3A = Path(r"D:\ORO_532_2026_TUNING\phase3a_532")
MODEL_DIR = Path(r"D:\ORO_532_2026_TUNING\model_532_fullperiod_oos")
OUT = Path(r"D:\ORO_532_2026_TUNING\threshold_fullperiod_oos_screen_532")


def main() -> None:
    market_path = PHASE3A / "market_state_2026.parquet"
    cand_path = PHASE3A / "technical_candidates_all_2026.parquet"
    pred_path = MODEL_DIR / "oos_s2_predictions_532_feb_jun_monthly.csv.gz"
    for p in (market_path, cand_path, pred_path):
        if not p.exists():
            raise SystemExit(f"MISSING_INPUT: {p}")

    market = pd.read_parquet(market_path)
    cand = pd.read_parquet(cand_path)
    market["timestamp"] = pd.to_datetime(market["timestamp"], utc=True)
    cand["timestamp"] = pd.to_datetime(cand["timestamp"], utc=True)
    cand = cand[cand["timestamp"] < pd.Timestamp("2026-07-01T00:00:00Z")].copy()
    bundle = build_tensor_bundle(market, cand)

    pred = pd.read_csv(pred_path)
    pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True)
    pred["row_index"] = pd.to_numeric(pred["row_index"], errors="raise").astype(int)
    if pred["row_index"].duplicated().any():
        raise RuntimeError("DUPLICATE_OOS_ROW_INDEX")
    if pred["row_index"].min() < 0 or pred["row_index"].max() >= len(bundle.frame):
        raise RuntimeError("OOS_ROW_INDEX_OUT_OF_RANGE")

    frame = bundle.frame.iloc[pred["row_index"].to_numpy()].copy().reset_index(drop=True)
    if not (pd.to_datetime(frame["timestamp"], utc=True).reset_index(drop=True) == pred["timestamp"].reset_index(drop=True)).all():
        raise RuntimeError("OOS_TIMESTAMP_ALIGNMENT_FAILED")
    frame["y_s2"] = pred["y_s2"].to_numpy(float)
    frame["s2_score"] = pred["s2_score"].to_numpy(float)
    frame["fold"] = pred["fold"].to_numpy(int)
    frame["validation_month"] = pred["validation_month"].astype(str).to_numpy()
    frame["family"] = frame.apply(family, axis=1)

    total = len(frame)
    expected = ["2026-02", "2026-03", "2026-04", "2026-05", "2026-06"]
    actual = sorted(frame["validation_month"].dropna().unique().tolist())
    if actual != expected:
        raise RuntimeError(f"OOS_MONTH_COVERAGE_FAIL expected={expected} actual={actual}")

    rows = []
    monthly_rows = []
    by_dir_rows = []
    for i, th in enumerate(CONFIG.threshold_grid, 1):
        accepted = frame[frame["s2_score"] >= th].copy()
        row = {"threshold": float(th)}
        row.update(summarize(accepted, total))
        rows.append(row)
        print(
            f"[{i}/{len(CONFIG.threshold_grid)}] threshold={th:.2f} accepted={len(accepted):,} "
            f"quality_mean={row['mean_target_quality_r']:.4f} "
            f"1.5R_WR={row['r1_5_decisive_win_rate_pct']:.2f}%",
            flush=True,
        )
        for month, g in accepted.groupby("validation_month", sort=True):
            mrow = {"threshold": float(th), "month": str(month)}
            mrow.update(summarize(g, total))
            monthly_rows.append(mrow)
        for direction, g in accepted.groupby("direction", sort=False):
            drow = {"threshold": float(th), "direction": str(direction)}
            drow.update(summarize(g, total))
            by_dir_rows.append(drow)

    summary = pd.DataFrame(rows)
    monthly = pd.DataFrame(monthly_rows)
    by_direction = pd.DataFrame(by_dir_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "threshold_fullperiod_oos_summary.csv", index=False)
    monthly.to_csv(OUT / "threshold_fullperiod_oos_monthly.csv", index=False)
    by_direction.to_csv(OUT / "threshold_fullperiod_oos_by_direction.csv", index=False)
    frame.to_parquet(OUT / "oos_feb_jun_candidates_with_scores.parquet", index=False)

    print("\nFINAL 5/3/2 FULL FEB-JUN OOS THRESHOLD SCREEN")
    print(summary.to_string(index=False))
    print("\nMONTHLY")
    print(monthly.to_string(index=False))
    print("\nBY_DIRECTION")
    print(by_direction.to_string(index=False))
    print("THRESHOLD_SELECTED=NO")
    print("MODEL_RETRAINED=NO")
    print("OOS_ONLY=YES")
    print("OOS_MONTHS=2026-02,2026-03,2026-04,2026-05,2026-06")
    print(f"OOS_ROWS={total:,}")
    print(f"OUTPUT={OUT}")


if __name__ == "__main__":
    main()
