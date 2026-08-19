from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CANDIDATES = Path(r"D:\ORO_532_2026_TUNING\phase3a_532\technical_candidates_feb_jun_2026.parquet")
M1_PATH = Path(r"D:\ORO_532_2026_TUNING\XAUUSD_M1_FRESH_532_2026_TUNING.csv")
OUT = Path(r"D:\ORO_532_2026_TUNING\m1_entry_trigger_study")
POINT = 0.01
MAX_TRIGGER_BARS = 5
EVAL_HORIZON_BARS = 15
TP_R = 1.5


@dataclass(frozen=True)
class TriggerResult:
    mode: str
    trigger_idx: int | None
    entry_idx: int | None


def load_m1() -> pd.DataFrame:
    df = pd.read_csv(M1_PATH)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)


def continuous_one_minute(times: pd.Series) -> bool:
    if len(times) <= 1:
        return True
    d = times.diff().dropna().dt.total_seconds().to_numpy()
    return bool(np.all(d == 60.0))


def entry_price(row: pd.Series, direction: str) -> float:
    spread_points = float(row.get("spread", 0.0))
    half_spread = spread_points * POINT / 2.0
    mid = float(row["open"])
    return mid + half_spread if direction == "LONG" else mid - half_spread


def find_color_flip(m1: pd.DataFrame, first_idx: int, direction: str) -> TriggerResult:
    stop = min(first_idx + MAX_TRIGGER_BARS, len(m1))
    for j in range(first_idx, stop):
        row = m1.iloc[j]
        passed = float(row["close"]) > float(row["open"]) if direction == "LONG" else float(row["close"]) < float(row["open"])
        if passed and j + 1 < len(m1):
            return TriggerResult("COLOR_FLIP", j, j + 1)
    return TriggerResult("COLOR_FLIP", None, None)


def find_close_break_prev(m1: pd.DataFrame, first_idx: int, direction: str) -> TriggerResult:
    stop = min(first_idx + MAX_TRIGGER_BARS, len(m1))
    for j in range(first_idx, stop):
        if j <= 0:
            continue
        row = m1.iloc[j]
        prev = m1.iloc[j - 1]
        passed = float(row["close"]) > float(prev["high"]) if direction == "LONG" else float(row["close"]) < float(prev["low"])
        if passed and j + 1 < len(m1):
            return TriggerResult("CLOSE_BREAK_PREV", j, j + 1)
    return TriggerResult("CLOSE_BREAK_PREV", None, None)


def find_rejection_break(m1: pd.DataFrame, first_idx: int, direction: str) -> TriggerResult:
    # Fully causal two-step pattern. A rejection bar first makes a fresh extreme
    # and closes back inside the previous bar's extreme. The following bar must
    # close through the rejection bar in the intended trade direction. Entry is
    # delayed to the NEXT M1 open, so no intrabar future information is used.
    stop = min(first_idx + MAX_TRIGGER_BARS, len(m1))
    for r in range(first_idx, stop):
        if r <= 0 or r + 2 >= len(m1):
            continue
        prev = m1.iloc[r - 1]
        rej = m1.iloc[r]
        confirm = m1.iloc[r + 1]
        if direction == "SHORT":
            rejection = float(rej["high"]) > float(prev["high"]) and float(rej["close"]) < float(prev["high"])
            confirmation = float(confirm["close"]) < float(rej["low"])
        else:
            rejection = float(rej["low"]) < float(prev["low"]) and float(rej["close"]) > float(prev["low"])
            confirmation = float(confirm["close"]) > float(rej["high"])
        if rejection and confirmation:
            return TriggerResult("REJECTION_BREAK", r + 1, r + 2)
    return TriggerResult("REJECTION_BREAK", None, None)


def evaluate_path(m1: pd.DataFrame, entry_idx: int, direction: str, entry: float, stop_distance: float) -> dict[str, object] | None:
    if stop_distance <= 0 or entry_idx + EVAL_HORIZON_BARS >= len(m1):
        return None
    path = m1.iloc[entry_idx:entry_idx + EVAL_HORIZON_BARS + 1].copy()
    if not continuous_one_minute(path["time"]):
        return None

    future = path.iloc[1:]
    highs = future["high"].to_numpy(float)
    lows = future["low"].to_numpy(float)
    final_close = float(future["close"].iloc[-1])

    if direction == "LONG":
        mfe = float(highs.max() - entry)
        mae = float(lows.min() - entry)
        endpoint_r = (final_close - entry) / stop_distance
        tp_price = entry + TP_R * stop_distance
        sl_price = entry - stop_distance
        hit_fn = lambda hi, lo: (hi >= tp_price, lo <= sl_price)
    else:
        mfe = float(entry - lows.min())
        mae = float(entry - highs.max())
        endpoint_r = (entry - final_close) / stop_distance
        tp_price = entry - TP_R * stop_distance
        sl_price = entry + stop_distance
        hit_fn = lambda hi, lo: (lo <= tp_price, hi >= sl_price)

    outcome = "TIMEOUT"
    bars_to_outcome = np.nan
    for k, (hi, lo) in enumerate(zip(highs, lows), start=1):
        hit_tp, hit_sl = hit_fn(hi, lo)
        if hit_tp and hit_sl:
            outcome = "AMBIGUOUS"
            bars_to_outcome = k
            break
        if hit_tp:
            outcome = "WIN_1_5R"
            bars_to_outcome = k
            break
        if hit_sl:
            outcome = "LOSS_1R"
            bars_to_outcome = k
            break

    return {
        "endpoint_r_15m": float(endpoint_r),
        "mfe_r_15m": float(mfe / stop_distance),
        "mae_r_15m": float(mae / stop_distance),
        "barrier_outcome": outcome,
        "bars_to_outcome": bars_to_outcome,
    }


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total_candidates = int(df["sample_id"].nunique())
    for mode, g in df.groupby("mode", sort=False):
        triggered = g[g["triggered"]].copy()
        valid = triggered[triggered["evaluation_valid"]].copy()
        decisive = valid[valid["barrier_outcome"].isin(["WIN_1_5R", "LOSS_1R"])]
        rows.append({
            "mode": mode,
            "candidates": total_candidates,
            "triggered": int(len(triggered)),
            "trigger_rate_pct": float(len(triggered) / total_candidates * 100.0) if total_candidates else np.nan,
            "valid_evaluations": int(len(valid)),
            "avg_entry_delay_min": float(pd.to_numeric(triggered["entry_delay_min"], errors="coerce").mean()) if len(triggered) else np.nan,
            "median_entry_delay_min": float(pd.to_numeric(triggered["entry_delay_min"], errors="coerce").median()) if len(triggered) else np.nan,
            "mean_endpoint_r_15m": float(pd.to_numeric(valid["endpoint_r_15m"], errors="coerce").mean()) if len(valid) else np.nan,
            "median_endpoint_r_15m": float(pd.to_numeric(valid["endpoint_r_15m"], errors="coerce").median()) if len(valid) else np.nan,
            "mean_mfe_r_15m": float(pd.to_numeric(valid["mfe_r_15m"], errors="coerce").mean()) if len(valid) else np.nan,
            "mean_mae_r_15m": float(pd.to_numeric(valid["mae_r_15m"], errors="coerce").mean()) if len(valid) else np.nan,
            "win_1_5r": int((valid["barrier_outcome"] == "WIN_1_5R").sum()) if len(valid) else 0,
            "loss_1r": int((valid["barrier_outcome"] == "LOSS_1R").sum()) if len(valid) else 0,
            "timeout": int((valid["barrier_outcome"] == "TIMEOUT").sum()) if len(valid) else 0,
            "ambiguous": int((valid["barrier_outcome"] == "AMBIGUOUS").sum()) if len(valid) else 0,
            "decisive_win_rate_pct": float((decisive["barrier_outcome"] == "WIN_1_5R").mean() * 100.0) if len(decisive) else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    if not CANDIDATES.exists():
        raise SystemExit(f"MISSING_CANDIDATES: {CANDIDATES}")
    if not M1_PATH.exists():
        raise SystemExit(f"MISSING_M1: {M1_PATH}")

    cand = pd.read_parquet(CANDIDATES)
    cand = cand[cand["label_validity_15m"].astype(str).eq("VALID")].copy()
    cand["timestamp"] = pd.to_datetime(cand["timestamp"], utc=True)
    cand["candidate_entry_time"] = pd.to_datetime(cand["candidate_entry_time"], utc=True)
    m1 = load_m1()
    times = m1["time"].to_numpy(dtype="datetime64[ns]")

    records: list[dict[str, object]] = []
    for n, c in cand.reset_index(drop=True).iterrows():
        direction = str(c["direction"])
        decision_time = pd.Timestamp(c["timestamp"])
        first_idx = int(times.searchsorted(decision_time.to_datetime64(), side="right"))
        if first_idx >= len(m1):
            continue

        modes = [
            TriggerResult("IMMEDIATE_NEXT_M1", None, first_idx),
            find_color_flip(m1, first_idx, direction),
            find_close_break_prev(m1, first_idx, direction),
            find_rejection_break(m1, first_idx, direction),
        ]

        for tr in modes:
            rec: dict[str, object] = {
                "sample_id": c["sample_id"],
                "timestamp": decision_time,
                "direction": direction,
                "setup": c.get("setup", ""),
                "signal_score": c.get("signal_score", np.nan),
                "confirmation_count": c.get("confirmation_count", np.nan),
                "mode": tr.mode,
                "triggered": tr.entry_idx is not None,
                "trigger_time": pd.NaT,
                "entry_time": pd.NaT,
                "entry_delay_min": np.nan,
                "entry_price": np.nan,
                "evaluation_valid": False,
                "endpoint_r_15m": np.nan,
                "mfe_r_15m": np.nan,
                "mae_r_15m": np.nan,
                "barrier_outcome": "NO_TRIGGER",
                "bars_to_outcome": np.nan,
            }
            if tr.entry_idx is None:
                records.append(rec)
                continue

            eidx = int(tr.entry_idx)
            erow = m1.iloc[eidx]
            eprice = float(c["candidate_entry_price"]) if tr.mode == "IMMEDIATE_NEXT_M1" else entry_price(erow, direction)
            rec["entry_time"] = erow["time"]
            rec["entry_price"] = eprice
            rec["entry_delay_min"] = (pd.Timestamp(erow["time"]) - decision_time).total_seconds() / 60.0
            if tr.trigger_idx is not None:
                rec["trigger_time"] = m1.iloc[int(tr.trigger_idx)]["time"]

            ev = evaluate_path(m1, eidx, direction, eprice, float(c["candidate_stop_distance"]))
            if ev is not None:
                rec.update(ev)
                rec["evaluation_valid"] = True
            else:
                rec["barrier_outcome"] = "INVALID_GAP_OR_TAIL"
            records.append(rec)

        if (n + 1) % 1000 == 0:
            print(f"processed={n+1:,}/{len(cand):,}", flush=True)

    detail = pd.DataFrame(records)
    summary = summarize(detail)
    by_direction = []
    for (mode, direction), g in detail.groupby(["mode", "direction"], sort=False):
        sg = summarize(g.assign(sample_id=g["sample_id"]))
        if not sg.empty:
            row = sg.iloc[0].to_dict()
            row["direction"] = direction
            by_direction.append(row)
    by_direction_df = pd.DataFrame(by_direction)

    OUT.mkdir(parents=True, exist_ok=True)
    detail.to_parquet(OUT / "m1_trigger_detail.parquet", index=False)
    summary.to_csv(OUT / "m1_trigger_summary.csv", index=False)
    by_direction_df.to_csv(OUT / "m1_trigger_by_direction.csv", index=False)

    print("\nM1_ENTRY_TRIGGER_STUDY=PASS")
    print(f"VALID_CANDIDATES={len(cand):,}")
    print(f"MAX_TRIGGER_BARS={MAX_TRIGGER_BARS}")
    print(f"EVAL_HORIZON_BARS={EVAL_HORIZON_BARS}")
    print("CAUSAL_ENTRY_RULE=trigger confirmed on closed M1 -> entry next M1 open")
    print("\nSUMMARY")
    print(summary.to_string(index=False))
    print("\nBY_DIRECTION")
    print(by_direction_df.to_string(index=False))
    print(f"OUTPUT={OUT}")


if __name__ == "__main__":
    main()
