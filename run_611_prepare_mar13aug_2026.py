from __future__ import annotations

"""Prepare the isolated 6/1/1 every-M1 timing research dataset.

This script DOES NOT trade and DOES NOT modify the two demo runtimes.

Causal contract
---------------
At each closed M1 bar:
  * timing context = last 6 closed M1 bars (materialized later by the tensor builder);
  * M5 direction = EMA9 > EMA21 => LONG, EMA9 < EMA21 => SHORT;
  * M15 direction = close > EMA50 => LONG, close < EMA50 => SHORT;
  * a sample exists only when M5 and M15 agree;
  * theoretical entry = next M1 open;
  * 15-minute future path is used only to build labels, never features;
  * no feature or label is allowed to touch 2026-08-14 or later.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from short_memory_611.config import CONFIG, RESEARCH_VERSION

SOURCE = Path(r"D:\ORO_532_MAR13AUG_2026")
OUT = Path(r"D:\ORO_611_MAR13AUG_2026")
UNSEEN_START = pd.Timestamp(CONFIG.unseen_start_utc)
HORIZON = CONFIG.label_horizon_minutes


def load_tf(name: str) -> pd.DataFrame:
    path = SOURCE / f"XAUUSD_{name}_FRESH_MAR13AUG_2026.csv"
    if not path.exists():
        raise SystemExit(f"MISSING_{name}: {path}")
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").drop_duplicates("time").reset_index(drop=True)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift(1)
    return pd.concat(
        [
            (df["high"] - df["low"]).abs(),
            (df["high"] - prev).abs(),
            (df["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)


def add_m1_features(m1: pd.DataFrame) -> pd.DataFrame:
    x = m1.copy()
    x["close_time"] = x["time"] + pd.Timedelta(minutes=1)
    x["range"] = x["high"] - x["low"]
    x["body"] = x["close"] - x["open"]
    x["upper_wick"] = x["high"] - x[["open", "close"]].max(axis=1)
    x["lower_wick"] = x[["open", "close"]].min(axis=1) - x["low"]
    x["return_1"] = x["close"].pct_change()
    x["ema9"] = x["close"].ewm(span=9, adjust=False).mean()
    x["ema21"] = x["close"].ewm(span=21, adjust=False).mean()
    x["atr14"] = true_range(x).ewm(alpha=1 / 14, adjust=False).mean()
    vol = pd.to_numeric(x.get("tick_volume", 0), errors="coerce").fillna(0.0)
    x["tick_volume_z20"] = (vol - vol.rolling(20).mean()) / vol.rolling(20).std(ddof=0).replace(0, np.nan)
    return x


def add_m5_direction(m5: pd.DataFrame) -> pd.DataFrame:
    x = m5.copy()
    x["close_time"] = x["time"] + pd.Timedelta(minutes=5)
    x["ema9"] = x["close"].ewm(span=9, adjust=False).mean()
    x["ema21"] = x["close"].ewm(span=21, adjust=False).mean()
    x["atr14"] = true_range(x).ewm(alpha=1 / 14, adjust=False).mean()
    x["direction_m5"] = np.where(x["ema9"] > x["ema21"], "LONG", np.where(x["ema9"] < x["ema21"], "SHORT", "NONE"))
    return x


def add_m15_direction(m15: pd.DataFrame) -> pd.DataFrame:
    x = m15.copy()
    x["close_time"] = x["time"] + pd.Timedelta(minutes=15)
    x["ema50"] = x["close"].ewm(span=50, adjust=False).mean()
    x["direction_m15"] = np.where(x["close"] > x["ema50"], "LONG", np.where(x["close"] < x["ema50"], "SHORT", "NONE"))
    return x


def build_samples(m1f: pd.DataFrame, m5f: pd.DataFrame, m15f: pd.DataFrame) -> pd.DataFrame:
    # Decision timestamp is the close of each M1 bar.
    base = m1f[["time", "close_time", "close", "atr14"]].copy()
    base = base.rename(columns={"time": "m1_open_time", "close_time": "timestamp", "close": "m1_close", "atr14": "atr14_m1"})
    base = base[base["timestamp"] < UNSEEN_START].sort_values("timestamp")

    m5_ctx = m5f[["close_time", "time", "open", "high", "low", "close", "ema9", "ema21", "atr14", "direction_m5"]].copy()
    m5_ctx = m5_ctx.rename(columns={
        "close_time": "m5_source_close_time",
        "time": "m5_open_time",
        "open": "m5_open",
        "high": "m5_high",
        "low": "m5_low",
        "close": "m5_close",
        "atr14": "atr14_m5",
    }).sort_values("m5_source_close_time")

    m15_ctx = m15f[["close_time", "time", "open", "high", "low", "close", "ema50", "direction_m15"]].copy()
    m15_ctx = m15_ctx.rename(columns={
        "close_time": "m15_source_close_time",
        "time": "m15_open_time",
        "open": "m15_open",
        "high": "m15_high",
        "low": "m15_low",
        "close": "m15_close",
    }).sort_values("m15_source_close_time")

    out = pd.merge_asof(base, m5_ctx, left_on="timestamp", right_on="m5_source_close_time", direction="backward")
    out = pd.merge_asof(out.sort_values("timestamp"), m15_ctx, left_on="timestamp", right_on="m15_source_close_time", direction="backward")

    causal = (
        out["m5_source_close_time"].notna()
        & out["m15_source_close_time"].notna()
        & (out["m5_source_close_time"] <= out["timestamp"])
        & (out["m15_source_close_time"] <= out["timestamp"])
    )
    out = out[causal].copy()
    out = out[(out["direction_m5"] == out["direction_m15"]) & out["direction_m5"].isin(["LONG", "SHORT"])].copy()
    out["direction"] = out["direction_m5"]
    out["candidate_entry_time"] = out["timestamp"]

    # Map entry at next M1 open (which starts exactly at decision timestamp when data are continuous).
    m1_index = m1f.set_index("time", drop=False)
    records: list[dict[str, object]] = []
    excluded_no_entry = 0
    excluded_incomplete_label = 0
    excluded_aug14_spill = 0

    for row in out.itertuples(index=False):
        decision_time = pd.Timestamp(row.timestamp)
        if decision_time not in m1_index.index:
            excluded_no_entry += 1
            continue
        entry_pos_arr = np.flatnonzero(m1f["time"].to_numpy(dtype="datetime64[ns]") == decision_time.to_datetime64())
        if len(entry_pos_arr) == 0:
            excluded_no_entry += 1
            continue
        entry_pos = int(entry_pos_arr[0])
        end_pos = entry_pos + HORIZON - 1
        if end_pos >= len(m1f):
            excluded_incomplete_label += 1
            continue
        future = m1f.iloc[entry_pos : end_pos + 1]
        label_end = pd.Timestamp(future["time"].iloc[-1]) + pd.Timedelta(minutes=1)
        if label_end > UNSEEN_START:
            excluded_aug14_spill += 1
            continue
        # Require exact continuous future M1 path.
        diffs = future["time"].diff().dropna().dt.total_seconds().div(60.0)
        if len(future) != HORIZON or not bool(np.all(np.isclose(diffs.to_numpy(float), 1.0))):
            excluded_incomplete_label += 1
            continue

        entry = float(future["open"].iloc[0])
        direction = str(row.direction)
        if direction == "LONG":
            mfe = float(future["high"].max() - entry)
            mae = float(entry - future["low"].min())
            endpoint = float(future["close"].iloc[-1] - entry)
        else:
            mfe = float(entry - future["low"].min())
            mae = float(future["high"].max() - entry)
            endpoint = float(entry - future["close"].iloc[-1])
        atr = float(row.atr14_m5)
        if not np.isfinite(atr) or atr <= 0:
            continue
        quality_r = mfe / atr - abs(mae / atr)

        rec = row._asdict()
        rec.update({
            "candidate_entry_price": entry,
            "label_end_time": label_end,
            "directional_MFE_price_15m": mfe,
            "directional_MAE_price_15m": mae,
            "directional_future_return_15m": endpoint,
            "label_quality_r": quality_r,
            "label_validity_15m": "VALID",
        })
        records.append(rec)

    samples = pd.DataFrame.from_records(records)
    samples.attrs["excluded_no_entry"] = excluded_no_entry
    samples.attrs["excluded_incomplete_label"] = excluded_incomplete_label
    samples.attrs["excluded_aug14_spill"] = excluded_aug14_spill
    return samples


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m1 = load_tf("M1")
    m5 = load_tf("M5")
    m15 = load_tf("M15")

    m1f = add_m1_features(m1)
    m5f = add_m5_direction(m5)
    m15f = add_m15_direction(m15)
    samples = build_samples(m1f, m5f, m15f)
    if samples.empty:
        raise RuntimeError("NO_611_SAMPLES")

    # Guard: 6 closed M1 bars must exist and be continuous before each sample.
    m1_close_times = m1f["close_time"].to_numpy(dtype="datetime64[ns]")
    valid_six = []
    for ts in pd.to_datetime(samples["timestamp"], utc=True):
        end = int(np.searchsorted(m1_close_times, ts.to_datetime64(), side="right") - 1)
        start = end - CONFIG.m1_length + 1
        ok = start >= 0
        if ok:
            seq = m1_close_times[start : end + 1]
            diffs = np.diff(seq).astype("timedelta64[m]").astype(int)
            ok = len(seq) == CONFIG.m1_length and bool(np.all(diffs == 1)) and seq[-1] <= ts.to_datetime64()
        valid_six.append(ok)
    samples = samples[np.asarray(valid_six, dtype=bool)].reset_index(drop=True)

    if (pd.to_datetime(samples["timestamp"], utc=True) >= UNSEEN_START).any():
        raise RuntimeError("AUG14_PRESENT_IN_611_FEATURE_TIMESTAMPS")
    if (pd.to_datetime(samples["label_end_time"], utc=True) > UNSEEN_START).any():
        raise RuntimeError("AUG14_PRESENT_IN_611_LABELS")
    if (pd.to_datetime(samples["m5_source_close_time"], utc=True) > pd.to_datetime(samples["timestamp"], utc=True)).any():
        raise RuntimeError("M5_LOOKAHEAD")
    if (pd.to_datetime(samples["m15_source_close_time"], utc=True) > pd.to_datetime(samples["timestamp"], utc=True)).any():
        raise RuntimeError("M15_LOOKAHEAD")

    m1f.to_parquet(OUT / "m1_features_611.parquet", index=False)
    m5f.to_parquet(OUT / "m5_direction_611.parquet", index=False)
    m15f.to_parquet(OUT / "m15_direction_611.parquet", index=False)
    samples.to_parquet(OUT / "samples_611_mar13aug.parquet", index=False)

    market_days = int(pd.to_datetime(samples["timestamp"], utc=True).dt.date.nunique())
    longs = int((samples["direction"] == "LONG").sum())
    shorts = int((samples["direction"] == "SHORT").sum())
    per_day = len(samples) / market_days if market_days else float("nan")
    by_month = (
        samples.assign(month=pd.to_datetime(samples["timestamp"], utc=True).dt.strftime("%Y-%m"))
        .groupby("month")
        .agg(samples=("direction", "size"), longs=("direction", lambda s: int((s == "LONG").sum())), shorts=("direction", lambda s: int((s == "SHORT").sum())))
        .reset_index()
    )

    manifest = {
        "research_version": RESEARCH_VERSION,
        "source": str(SOURCE),
        "output": str(OUT),
        "sequence": {"m1": 6, "m5": 1, "m15": 1},
        "decision_frequency": "EVERY_CLOSED_M1",
        "m5_role": "DIRECTION_ONLY_EMA9_VS_EMA21",
        "m15_role": "DIRECTION_ONLY_CLOSE_VS_EMA50",
        "direction_contract": "M5_AND_M15_MUST_AGREE",
        "entry_contract": "DECISION_AFTER_CLOSED_M1_THEN_ENTRY_NEXT_M1_OPEN",
        "label_horizon_minutes": HORIZON,
        "samples": int(len(samples)),
        "market_days": market_days,
        "samples_per_market_day": per_day,
        "longs": longs,
        "shorts": shorts,
        "unseen_start": UNSEEN_START.isoformat(),
        "aug14_used": False,
    }
    (OUT / "PREPARE_611_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    by_month.to_csv(OUT / "samples_611_by_month.csv", index=False)

    print("=" * 78)
    print("6/1/1 M1-TIMING PREPARE - MAR 01 -> AUG 13 2026")
    print("DECISION=EVERY CLOSED M1")
    print("M1=last 6 closed bars | M5=direction EMA9/21 | M15=direction close/EMA50")
    print("ENTRY=next M1 open | LABEL=15m future path, target-only")
    print("=" * 78)
    print("PREPARE_611=PASS")
    print(f"SAMPLES={len(samples):,}")
    print(f"MARKET_DAYS={market_days}")
    print(f"SAMPLES_PER_MARKET_DAY={per_day:.2f}")
    print(f"LONGS={longs:,} SHORTS={shorts:,}")
    print("AUG14_USED_IN_FEATURES=NO")
    print("AUG14_USED_IN_LABELS=NO")
    print("FUTURE_LABELS_USED_AS_FEATURES=NO")
    print("BY_MONTH")
    print(by_month.to_string(index=False))
    print(f"OUTPUT={OUT}")


if __name__ == "__main__":
    main()
