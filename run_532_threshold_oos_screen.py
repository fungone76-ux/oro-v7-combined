from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from short_memory_532.config import CONFIG
from short_memory_532.dataset import build_tensor_bundle

PHASE3A = Path(r"D:\ORO_532_2026_TUNING\phase3a_532")
MODEL_DIR = Path(r"D:\ORO_532_2026_TUNING\model_532")
OUT = Path(r"D:\ORO_532_2026_TUNING\threshold_oos_screen_532")


def pct(x: pd.Series) -> float:
    return float(x.mean() * 100.0) if len(x) else float("nan")


def barrier_stats(g: pd.DataFrame, col: str, prefix: str) -> dict[str, float | int]:
    s = g[col].astype(str)
    wins = int((s == "WIN").sum())
    losses = int((s == "LOSS").sum())
    timeout = int((s == "TIMEOUT").sum())
    ambiguous = int((s == "AMBIGUOUS_SAME_BAR").sum())
    invalid = int((s == "INVALID").sum())
    decisive = wins + losses
    return {
        f"{prefix}_wins": wins,
        f"{prefix}_losses": losses,
        f"{prefix}_timeouts": timeout,
        f"{prefix}_ambiguous": ambiguous,
        f"{prefix}_invalid": invalid,
        f"{prefix}_decisive_win_rate_pct": (wins / decisive * 100.0) if decisive else float("nan"),
    }


def family(row: pd.Series) -> str:
    bb = bool(row.get("confirmation_bb_lower_touch", False)) or bool(row.get("confirmation_bb_upper_touch", False))
    sr = bool(row.get("confirmation_support_proximity", False)) or bool(row.get("confirmation_resistance_proximity", False))
    rsi = bool(row.get("confirmation_rsi_rebound", False))
    ema = bool(row.get("confirmation_ema_cross", False))
    pa = bool(row.get("confirmation_price_action", False))
    vol = bool(row.get("confirmation_volume_confirm", False))
    if bb and sr and rsi:
        return "BB_SR_RSI"
    if bb and sr and ema:
        return "BB_SR_EMA"
    if sr and ema and pa:
        return "SR_EMA_PRICE_ACTION"
    if sr and ema:
        return "SR_EMA"
    if ema and pa and vol and not (bb or sr or rsi):
        return "GENERIC_EMA_PA_VOLUME"
    return "OTHER"


def summarize(g: pd.DataFrame, total_oos: int) -> dict[str, float | int]:
    y = pd.to_numeric(g["y_s2"], errors="coerce")
    score = pd.to_numeric(g["s2_score"], errors="coerce")
    dfr = pd.to_numeric(g["directional_future_return_15m"], errors="coerce")
    out: dict[str, float | int] = {
        "accepted": int(len(g)),
        "accepted_pct_oos": float(len(g) / total_oos * 100.0) if total_oos else float("nan"),
        "mean_s2_score": float(score.mean()),
        "median_s2_score": float(score.median()),
        "mean_target_quality_r": float(y.mean()),
        "median_target_quality_r": float(y.median()),
        "positive_15m_pct": pct(dfr > 0),
        "longs": int((g["direction"].astype(str) == "LONG").sum()),
        "shorts": int((g["direction"].astype(str) == "SHORT").sum()),
        "generic_pct": pct(g["family"].eq("GENERIC_EMA_PA_VOLUME")),
        "contextual_extreme_pct": pct(g["family"].isin(["BB_SR_RSI", "BB_SR_EMA"])),
    }
    out.update(barrier_stats(g, "barrier_1r_outcome", "r1"))
    out.update(barrier_stats(g, "barrier_1_5r_outcome", "r1_5"))
    return out


def main() -> None:
    market_path = PHASE3A / "market_state_2026.parquet"
    cand_path = PHASE3A / "technical_candidates_feb_jun_2026.parquet"
    pred_path = MODEL_DIR / "oos_s2_predictions_532.csv.gz"
    for p in (market_path, cand_path, pred_path):
        if not p.exists():
            raise SystemExit(f"MISSING_INPUT: {p}")

    market = pd.read_parquet(market_path)
    cand = pd.read_parquet(cand_path)
    market["timestamp"] = pd.to_datetime(market["timestamp"], utc=True)
    cand["timestamp"] = pd.to_datetime(cand["timestamp"], utc=True)
    bundle = build_tensor_bundle(market, cand)

    pred = pd.read_csv(pred_path)
    pred["timestamp"] = pd.to_datetime(pred["timestamp"], utc=True)
    pred["row_index"] = pd.to_numeric(pred["row_index"], errors="raise").astype(int)
    if pred["row_index"].duplicated().any():
        raise RuntimeError("DUPLICATE_OOS_ROW_INDEX")
    if pred["row_index"].min() < 0 or pred["row_index"].max() >= len(bundle.frame):
        raise RuntimeError("OOS_ROW_INDEX_OUT_OF_RANGE")

    frame = bundle.frame.iloc[pred["row_index"].to_numpy()].copy().reset_index(drop=True)
    frame["oos_timestamp"] = pred["timestamp"].to_numpy()
    if not (pd.to_datetime(frame["timestamp"], utc=True).reset_index(drop=True) == pred["timestamp"].reset_index(drop=True)).all():
        raise RuntimeError("OOS_TIMESTAMP_ALIGNMENT_FAILED")
    frame["y_s2"] = pred["y_s2"].to_numpy(float)
    frame["s2_score"] = pred["s2_score"].to_numpy(float)
    frame["fold"] = pred["fold"].to_numpy(int)
    frame["family"] = frame.apply(family, axis=1)

    thresholds = tuple(CONFIG.threshold_grid)
    rows: list[dict[str, object]] = []
    by_dir_rows: list[dict[str, object]] = []
    total = len(frame)
    for i, th in enumerate(thresholds, 1):
        accepted = frame[frame["s2_score"] >= th].copy()
        row: dict[str, object] = {"threshold": float(th)}
        row.update(summarize(accepted, total))
        rows.append(row)
        print(
            f"[{i}/{len(thresholds)}] threshold={th:.2f} accepted={len(accepted):,} "
            f"quality_mean={row['mean_target_quality_r']:.4f} "
            f"1.5R_WR={row['r1_5_decisive_win_rate_pct']:.2f}%",
            flush=True,
        )
        for direction, g in accepted.groupby("direction", sort=False):
            drow: dict[str, object] = {"threshold": float(th), "direction": str(direction)}
            drow.update(summarize(g, total))
            by_dir_rows.append(drow)

    summary = pd.DataFrame(rows)
    by_direction = pd.DataFrame(by_dir_rows)
    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "threshold_oos_summary.csv", index=False)
    by_direction.to_csv(OUT / "threshold_oos_by_direction.csv", index=False)
    frame.to_parquet(OUT / "oos_candidates_with_scores.parquet", index=False)

    print("\nFINAL 5/3/2 OOS THRESHOLD SCREEN")
    print(summary.to_string(index=False))
    print("\nBY_DIRECTION")
    print(by_direction.to_string(index=False))
    print("THRESHOLD_SELECTED=NO")
    print("MODEL_RETRAINED=NO")
    print("OOS_ONLY=YES")
    print(f"OOS_ROWS={total:,}")
    print(f"OUTPUT={OUT}")


if __name__ == "__main__":
    main()
