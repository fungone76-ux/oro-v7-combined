from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CANDIDATES = Path(r"D:\ORO_532_2026_TUNING\phase3a_532\technical_candidates_feb_jun_2026.parquet")
OUT = Path(r"D:\ORO_532_2026_TUNING\candidate_quality_audit")


def combo_from_row(row: pd.Series, confirmation_cols: list[str]) -> str:
    passed = []
    for col in confirmation_cols:
        if bool(row.get(col, False)):
            passed.append(col.replace("confirmation_", ""))
    return "+".join(sorted(passed)) if passed else "NONE"


def main() -> None:
    if not CANDIDATES.exists():
        raise SystemExit(f"MISSING_CANDIDATES: {CANDIDATES}")

    df = pd.read_parquet(CANDIDATES)
    if df.empty:
        raise SystemExit("EMPTY_CANDIDATES")

    required = [
        "direction", "setup", "signal_score", "confirmation_count",
        "label_validity_15m", "directional_future_return_15m",
        "directional_MFE_price_15m", "directional_MAE_price_15m",
        "atr14",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"MISSING_REQUIRED_COLUMNS: {missing}")

    confirmation_cols = sorted(
        c for c in df.columns
        if c.startswith("confirmation_") and c != "confirmation_count"
    )
    df = df[df["label_validity_15m"].astype(str).eq("VALID")].copy()
    df["confirmation_combo"] = df.apply(
        lambda r: combo_from_row(r, confirmation_cols), axis=1
    )

    atr = pd.to_numeric(df["atr14"], errors="coerce").replace(0, np.nan)
    df["quality_r"] = (
        pd.to_numeric(df["directional_MFE_price_15m"], errors="coerce") / atr
        - (pd.to_numeric(df["directional_MAE_price_15m"], errors="coerce") / atr).abs()
    )
    df["directional_positive_15m"] = (
        pd.to_numeric(df["directional_future_return_15m"], errors="coerce") > 0
    )

    def summarize(g: pd.DataFrame) -> dict[str, float | int]:
        q = pd.to_numeric(g["quality_r"], errors="coerce")
        return {
            "candidates": int(len(g)),
            "pct_total": float(len(g) / len(df) * 100.0),
            "positive_15m_pct": float(g["directional_positive_15m"].mean() * 100.0),
            "mean_quality_r": float(q.mean()),
            "median_quality_r": float(q.median()),
            "mean_signal_score": float(pd.to_numeric(g["signal_score"], errors="coerce").mean()),
            "mean_confirmations": float(pd.to_numeric(g["confirmation_count"], errors="coerce").mean()),
        }

    def grouped(keys: list[str]) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        grouper = keys[0] if len(keys) == 1 else keys
        for key_values, g in df.groupby(grouper, dropna=False, sort=False):
            if len(keys) == 1:
                key_values = (key_values,)
            elif not isinstance(key_values, tuple):
                key_values = (key_values,)
            row = {k: v for k, v in zip(keys, key_values)}
            row.update(summarize(g))
            rows.append(row)
        return pd.DataFrame(rows)

    by_combo = grouped(["direction", "confirmation_combo"])
    by_combo = by_combo.sort_values(
        ["candidates", "mean_quality_r"], ascending=[False, False]
    )

    by_count = grouped(["direction", "confirmation_count"])
    by_count = by_count.sort_values(["direction", "confirmation_count"])

    by_score = grouped(["direction", "signal_score"])
    by_score = by_score.sort_values(["direction", "signal_score"])

    OUT.mkdir(parents=True, exist_ok=True)
    by_combo.to_csv(OUT / "by_confirmation_combo.csv", index=False)
    by_count.to_csv(OUT / "by_confirmation_count.csv", index=False)
    by_score.to_csv(OUT / "by_signal_score.csv", index=False)

    print("CANDIDATE_QUALITY_AUDIT=PASS")
    print(f"VALID_CANDIDATES={len(df):,}")
    print(f"CONFIRMATION_COLUMNS={len(confirmation_cols)}")
    print("TOP_COMBINATIONS")
    print(by_combo.head(20).to_string(index=False))
    print("\nBY_CONFIRMATION_COUNT")
    print(by_count.to_string(index=False))
    print("\nBY_SIGNAL_SCORE")
    print(by_score.to_string(index=False))
    print(f"OUTPUT={OUT}")


if __name__ == "__main__":
    main()
