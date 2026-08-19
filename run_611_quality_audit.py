from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(r"D:\ORO_611_MAR13AUG_2026")
SAMPLES = ROOT / "samples_611_mar13aug.parquet"
OUT = ROOT / "quality_audit_611"


def summarize(g: pd.DataFrame) -> pd.Series:
    q = pd.to_numeric(g["label_quality_r"], errors="coerce")
    endpoint = pd.to_numeric(g["directional_future_return_15m"], errors="coerce")
    mfe = pd.to_numeric(g["directional_MFE_price_15m"], errors="coerce")
    mae = pd.to_numeric(g["directional_MAE_price_15m"], errors="coerce")
    atr = pd.to_numeric(g["atr14_m5"], errors="coerce").replace(0, np.nan)
    mfe_r = mfe / atr
    mae_r = mae.abs() / atr
    return pd.Series({
        "samples": int(len(g)),
        "positive_quality_pct": float((q > 0).mean() * 100.0),
        "mean_quality_r": float(q.mean()),
        "median_quality_r": float(q.median()),
        "p75_quality_r": float(q.quantile(0.75)),
        "p90_quality_r": float(q.quantile(0.90)),
        "endpoint_positive_pct": float((endpoint > 0).mean() * 100.0),
        "mean_mfe_r": float(mfe_r.mean()),
        "mean_mae_r": float(mae_r.mean()),
    })


def main() -> None:
    if not SAMPLES.exists():
        raise SystemExit(f"MISSING_INPUT: {SAMPLES}")
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(SAMPLES)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["month"] = df["timestamp"].dt.strftime("%Y-%m")
    df["context"] = np.where(pd.to_numeric(df["direction_agreement"], errors="coerce").fillna(0).astype(int) == 1, "AGREEMENT", "CONFLICT")

    overall = summarize(df).to_frame().T
    by_context = df.groupby("context", dropna=False).apply(summarize, include_groups=False).reset_index()
    by_direction = df.groupby("direction", dropna=False).apply(summarize, include_groups=False).reset_index()
    by_context_direction = df.groupby(["context", "direction"], dropna=False).apply(summarize, include_groups=False).reset_index()
    by_month_context = df.groupby(["month", "context"], dropna=False).apply(summarize, include_groups=False).reset_index()

    overall.to_csv(OUT / "overall.csv", index=False)
    by_context.to_csv(OUT / "by_context.csv", index=False)
    by_direction.to_csv(OUT / "by_direction.csv", index=False)
    by_context_direction.to_csv(OUT / "by_context_direction.csv", index=False)
    by_month_context.to_csv(OUT / "by_month_context.csv", index=False)

    print("611_QUALITY_AUDIT=PASS")
    print(f"SAMPLES={len(df):,}")
    print("OVERALL")
    print(overall.to_string(index=False))
    print("\nBY_CONTEXT")
    print(by_context.to_string(index=False))
    print("\nBY_DIRECTION")
    print(by_direction.to_string(index=False))
    print("\nBY_CONTEXT_DIRECTION")
    print(by_context_direction.to_string(index=False))
    print("\nBY_MONTH_CONTEXT")
    print(by_month_context.to_string(index=False))
    print(f"OUTPUT={OUT}")


if __name__ == "__main__":
    main()
