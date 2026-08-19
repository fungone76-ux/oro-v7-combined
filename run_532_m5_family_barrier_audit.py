from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CANDIDATES = Path(r"D:\ORO_532_2026_TUNING\phase3a_532\technical_candidates_feb_jun_2026.parquet")
OUT = Path(r"D:\ORO_532_2026_TUNING\m5_family_barrier_audit")


def _b(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].fillna(False).astype(bool)


def classify_family(df: pd.DataFrame) -> pd.Series:
    ema = _b(df, "confirmation_ema_cross")
    pa = _b(df, "confirmation_price_action")
    vol = _b(df, "confirmation_volume_confirm")
    bb = _b(df, "confirmation_bb_lower_touch") | _b(df, "confirmation_bb_upper_touch")
    sr = _b(df, "confirmation_support_proximity") | _b(df, "confirmation_resistance_proximity")
    rsi = _b(df, "confirmation_rsi_rebound")

    family = pd.Series("OTHER", index=df.index, dtype="object")

    # Most specific/context-rich families first.
    family.loc[bb & sr & rsi] = "BB_SR_RSI"
    family.loc[(family == "OTHER") & bb & sr & ema] = "BB_SR_EMA"
    family.loc[(family == "OTHER") & bb & sr] = "BB_SR"
    family.loc[(family == "OTHER") & sr & rsi] = "SR_RSI"
    family.loc[(family == "OTHER") & bb & rsi] = "BB_RSI"
    family.loc[(family == "OTHER") & sr & ema & pa] = "SR_EMA_PRICE_ACTION"
    family.loc[(family == "OTHER") & sr & ema] = "SR_EMA"
    family.loc[(family == "OTHER") & ema & pa & vol] = "GENERIC_EMA_PA_VOLUME"
    family.loc[(family == "OTHER") & ema & pa] = "EMA_PRICE_ACTION"
    family.loc[(family == "OTHER") & ema & vol] = "EMA_VOLUME"
    return family


def _outcome_stats(s: pd.Series, prefix: str) -> dict[str, float | int]:
    x = s.astype(str)
    valid = ~x.isin(["INVALID", "nan", "None"])
    xv = x[valid]
    n = int(len(xv))
    wins = int((xv == "WIN").sum())
    losses = int((xv == "LOSS").sum())
    timeouts = int((xv == "TIMEOUT").sum())
    ambiguous = int((xv == "AMBIGUOUS_SAME_BAR").sum())
    decisive = wins + losses
    return {
        f"{prefix}_valid": n,
        f"{prefix}_wins": wins,
        f"{prefix}_losses": losses,
        f"{prefix}_timeouts": timeouts,
        f"{prefix}_ambiguous": ambiguous,
        f"{prefix}_decisive_win_rate_pct": float(wins / decisive * 100.0) if decisive else np.nan,
    }


def summarize(g: pd.DataFrame, total: int) -> dict[str, float | int]:
    stop = pd.to_numeric(g["candidate_stop_distance"], errors="coerce").replace(0, np.nan)
    mfe_r = pd.to_numeric(g["directional_MFE_price_15m"], errors="coerce") / stop
    mae_r = pd.to_numeric(g["directional_MAE_price_15m"], errors="coerce") / stop
    endpoint = pd.to_numeric(g["directional_future_return_15m"], errors="coerce")
    out: dict[str, float | int] = {
        "candidates": int(len(g)),
        "pct_total": float(len(g) / total * 100.0),
        "mean_signal_score": float(pd.to_numeric(g["signal_score"], errors="coerce").mean()),
        "mean_confirmations": float(pd.to_numeric(g["confirmation_count"], errors="coerce").mean()),
        "positive_15m_pct": float((endpoint > 0).mean() * 100.0),
        "mean_mfe_r_15m": float(mfe_r.mean()),
        "median_mfe_r_15m": float(mfe_r.median()),
        "mean_mae_r_15m": float(mae_r.mean()),
        "median_mae_r_15m": float(mae_r.median()),
        "mean_quality_r": float((mfe_r - mae_r.abs()).mean()),
        "median_quality_r": float((mfe_r - mae_r.abs()).median()),
    }
    out.update(_outcome_stats(g["barrier_1r_outcome"], "r1"))
    out.update(_outcome_stats(g["barrier_1_5r_outcome"], "r1_5"))
    return out


def grouped(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouper = keys[0] if len(keys) == 1 else keys
    for key_values, g in df.groupby(grouper, dropna=False, sort=False):
        if len(keys) == 1:
            key_values = (key_values,)
        elif not isinstance(key_values, tuple):
            key_values = (key_values,)
        row = {k: v for k, v in zip(keys, key_values)}
        row.update(summarize(g, len(df)))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    if not CANDIDATES.exists():
        raise SystemExit(f"MISSING_CANDIDATES: {CANDIDATES}")
    df = pd.read_parquet(CANDIDATES)
    required = [
        "direction", "signal_score", "confirmation_count", "label_validity_15m",
        "candidate_stop_distance", "directional_future_return_15m",
        "directional_MFE_price_15m", "directional_MAE_price_15m",
        "barrier_1r_outcome", "barrier_1_5r_outcome",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"MISSING_REQUIRED_COLUMNS: {missing}")

    df = df[df["label_validity_15m"].astype(str).eq("VALID")].copy()
    df["family"] = classify_family(df)

    by_family = grouped(df, ["family"]).sort_values(["candidates", "r1_5_decisive_win_rate_pct"], ascending=[False, False])
    by_dir_family = grouped(df, ["direction", "family"]).sort_values(["direction", "candidates"], ascending=[True, False])

    # Explicit generic-vs-context comparison.
    generic = df["family"].eq("GENERIC_EMA_PA_VOLUME")
    contextual = df["family"].isin(["BB_SR_RSI", "BB_SR_EMA", "BB_SR", "SR_RSI", "BB_RSI"])
    cohort = pd.Series("OTHER", index=df.index, dtype="object")
    cohort.loc[generic] = "GENERIC"
    cohort.loc[contextual] = "CONTEXTUAL_EXTREME"
    df["cohort"] = cohort
    by_cohort = grouped(df, ["direction", "cohort"]).sort_values(["direction", "cohort"])

    OUT.mkdir(parents=True, exist_ok=True)
    by_family.to_csv(OUT / "by_family.csv", index=False)
    by_dir_family.to_csv(OUT / "by_direction_family.csv", index=False)
    by_cohort.to_csv(OUT / "generic_vs_contextual.csv", index=False)
    df[["sample_id", "timestamp", "direction", "setup", "signal_score", "confirmation_count", "family", "cohort", "barrier_1r_outcome", "barrier_1_5r_outcome"]].to_csv(OUT / "candidate_family_assignment.csv", index=False)

    print("M5_FAMILY_BARRIER_AUDIT=PASS")
    print(f"VALID_CANDIDATES={len(df):,}")
    print("\nBY_FAMILY")
    print(by_family.to_string(index=False))
    print("\nBY_DIRECTION_FAMILY")
    print(by_dir_family.to_string(index=False))
    print("\nGENERIC_VS_CONTEXTUAL")
    print(by_cohort.to_string(index=False))
    print(f"OUTPUT={OUT}")


if __name__ == "__main__":
    main()
