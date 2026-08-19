from __future__ import annotations

"""Prepare the isolated 6/1/1 every-M1 timing research dataset.

RESEARCH ONLY: this script never sends MT5 orders and never modifies the two
running demo bots.

Causal contract
---------------
At every closed M1 bar T:
  * timing context: last 6 closed M1 bars;
  * primary direction: last closed M5, EMA9 vs EMA21;
  * context direction: last closed M15, close vs EMA50;
  * M5/M15 disagreement is KEPT as a sample and encoded as trend_conflict=1;
  * theoretical entry: next M1 open at T;
  * future 15-minute path builds labels only;
  * no feature or label may touch 2026-08-14 or later.
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from short_memory_611.config import CONFIG, RESEARCH_VERSION

SOURCE = Path(r"D:\ORO_532_MAR13AUG_2026")
OUT = Path(r"D:\ORO_611_MAR13AUG_2026")
UNSEEN_START = pd.Timestamp(CONFIG.unseen_start_utc)
HORIZON = int(CONFIG.label_horizon_minutes)


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
        [(df["high"]-df["low"]).abs(), (df["high"]-prev).abs(), (df["low"]-prev).abs()],
        axis=1,
    ).max(axis=1)


def rsi14(close: pd.Series) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def add_m1_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["close_time"] = x["time"] + pd.Timedelta(minutes=1)
    x["range"] = x["high"] - x["low"]
    x["body"] = x["close"] - x["open"]
    x["upper_wick"] = x["high"] - x[["open", "close"]].max(axis=1)
    x["lower_wick"] = x[["open", "close"]].min(axis=1) - x["low"]
    x["return_1"] = x["close"].pct_change()
    x["ema9"] = x["close"].ewm(span=9, adjust=False).mean()
    x["ema21"] = x["close"].ewm(span=21, adjust=False).mean()
    x["ema_gap"] = x["ema9"] - x["ema21"]
    x["rsi14"] = rsi14(x["close"])
    x["atr14"] = true_range(x).ewm(alpha=1/14, adjust=False).mean()
    vol = pd.to_numeric(x.get("tick_volume", 0), errors="coerce").fillna(0.0)
    mean20 = vol.rolling(20).mean()
    std20 = vol.rolling(20).std(ddof=0).replace(0, np.nan)
    x["tick_volume_z20"] = (vol - mean20) / std20
    return x


def add_m5_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["close_time"] = x["time"] + pd.Timedelta(minutes=5)
    x["ema9"] = x["close"].ewm(span=9, adjust=False).mean()
    x["ema21"] = x["close"].ewm(span=21, adjust=False).mean()
    x["ema_gap"] = x["ema9"] - x["ema21"]
    x["atr14"] = true_range(x).ewm(alpha=1/14, adjust=False).mean()
    x["rsi14"] = rsi14(x["close"])
    x["direction_m5"] = np.where(x["ema_gap"] > 0, "LONG", np.where(x["ema_gap"] < 0, "SHORT", "NONE"))
    return x


def add_m15_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    x["close_time"] = x["time"] + pd.Timedelta(minutes=15)
    x["ema50"] = x["close"].ewm(span=50, adjust=False).mean()
    x["trend_gap"] = x["close"] - x["ema50"]
    x["direction_m15"] = np.where(x["trend_gap"] > 0, "LONG", np.where(x["trend_gap"] < 0, "SHORT", "NONE"))
    return x


def exact_six_m1_available(m1f: pd.DataFrame, ts: pd.Timestamp) -> bool:
    closes = m1f["close_time"].to_numpy(dtype="datetime64[ns]")
    end = int(np.searchsorted(closes, ts.to_datetime64(), side="right") - 1)
    start = end - CONFIG.m1_length + 1
    if start < 0:
        return False
    seq = closes[start:end+1]
    if len(seq) != CONFIG.m1_length or seq[-1] > ts.to_datetime64():
        return False
    d = np.diff(seq).astype("timedelta64[m]").astype(int)
    return bool(np.all(d == 1))


def build_samples(m1f: pd.DataFrame, m5f: pd.DataFrame, m15f: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    # One potential sample for every closed M1.
    base = m1f[["time", "close_time", "close", "atr14"]].copy()
    base.columns = ["m1_open_time", "timestamp", "m1_close", "atr14_m1"]
    base = base[base["timestamp"] < UNSEEN_START].sort_values("timestamp")

    m5ctx = m5f[["close_time","time","open","high","low","close","ema9","ema21","ema_gap","atr14","rsi14","direction_m5"]].copy()
    m5ctx = m5ctx.rename(columns={
        "close_time":"m5_source_close_time", "time":"m5_open_time", "open":"m5_open",
        "high":"m5_high", "low":"m5_low", "close":"m5_close", "atr14":"atr14_m5",
        "rsi14":"rsi14_m5", "ema9":"ema9_m5", "ema21":"ema21_m5", "ema_gap":"ema_gap_m5",
    }).sort_values("m5_source_close_time")

    m15ctx = m15f[["close_time","time","open","high","low","close","ema50","trend_gap","direction_m15"]].copy()
    m15ctx = m15ctx.rename(columns={
        "close_time":"m15_source_close_time", "time":"m15_open_time", "open":"m15_open",
        "high":"m15_high", "low":"m15_low", "close":"m15_close", "ema50":"ema50_m15",
        "trend_gap":"trend_gap_m15",
    }).sort_values("m15_source_close_time")

    x = pd.merge_asof(base, m5ctx, left_on="timestamp", right_on="m5_source_close_time", direction="backward")
    x = pd.merge_asof(x.sort_values("timestamp"), m15ctx, left_on="timestamp", right_on="m15_source_close_time", direction="backward")
    x = x[
        x["m5_source_close_time"].notna() & x["m15_source_close_time"].notna()
        & (x["m5_source_close_time"] <= x["timestamp"])
        & (x["m15_source_close_time"] <= x["timestamp"])
        & x["direction_m5"].isin(["LONG","SHORT"])
        & x["direction_m15"].isin(["LONG","SHORT"])
    ].copy()

    # IMPORTANT: disagreement is a feature, not a filter.
    x["direction"] = x["direction_m5"]
    x["direction_agreement"] = (x["direction_m5"] == x["direction_m15"]).astype(np.int8)
    x["trend_conflict"] = (1 - x["direction_agreement"]).astype(np.int8)
    x["m15_same_as_primary"] = x["direction_agreement"]
    x["candidate_entry_time"] = x["timestamp"]

    m1_times = m1f["time"].to_numpy(dtype="datetime64[ns]")
    records: list[dict[str, object]] = []
    stats = {"no_six_m1":0, "no_entry":0, "incomplete_future":0, "aug14_spill":0, "bad_atr":0}

    for row in x.itertuples(index=False):
        ts = pd.Timestamp(row.timestamp)
        if not exact_six_m1_available(m1f, ts):
            stats["no_six_m1"] += 1
            continue
        entry_idx = int(np.searchsorted(m1_times, ts.to_datetime64(), side="left"))
        if entry_idx >= len(m1f) or m1_times[entry_idx] != ts.to_datetime64():
            stats["no_entry"] += 1
            continue
        end_idx = entry_idx + HORIZON - 1
        if end_idx >= len(m1f):
            stats["incomplete_future"] += 1
            continue
        future = m1f.iloc[entry_idx:end_idx+1]
        diffs = future["time"].diff().dropna().dt.total_seconds().div(60.0).to_numpy(float)
        if len(future) != HORIZON or not bool(np.all(np.isclose(diffs, 1.0))):
            stats["incomplete_future"] += 1
            continue
        label_end = pd.Timestamp(future["time"].iloc[-1]) + pd.Timedelta(minutes=1)
        if label_end > UNSEEN_START:
            stats["aug14_spill"] += 1
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
            stats["bad_atr"] += 1
            continue

        rec = row._asdict()
        rec.update({
            "candidate_entry_price": entry,
            "label_end_time": label_end,
            "directional_MFE_price_15m": mfe,
            "directional_MAE_price_15m": mae,
            "directional_future_return_15m": endpoint,
            "label_quality_r": mfe / atr - abs(mae / atr),
            "label_validity_15m": "VALID",
        })
        records.append(rec)

    return pd.DataFrame.from_records(records), stats


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m1f = add_m1_features(load_tf("M1"))
    m5f = add_m5_features(load_tf("M5"))
    m15f = add_m15_features(load_tf("M15"))
    samples, exclusions = build_samples(m1f, m5f, m15f)
    if samples.empty:
        raise RuntimeError("NO_611_SAMPLES")

    ts = pd.to_datetime(samples["timestamp"], utc=True)
    le = pd.to_datetime(samples["label_end_time"], utc=True)
    if (ts >= UNSEEN_START).any() or (le > UNSEEN_START).any():
        raise RuntimeError("AUG14_LEAK_611")
    if (pd.to_datetime(samples["m5_source_close_time"], utc=True) > ts).any():
        raise RuntimeError("M5_LOOKAHEAD_611")
    if (pd.to_datetime(samples["m15_source_close_time"], utc=True) > ts).any():
        raise RuntimeError("M15_LOOKAHEAD_611")

    m1f.to_parquet(OUT / "m1_features_611.parquet", index=False)
    m5f.to_parquet(OUT / "m5_context_611.parquet", index=False)
    m15f.to_parquet(OUT / "m15_context_611.parquet", index=False)
    samples.to_parquet(OUT / "samples_611_mar13aug.parquet", index=False)

    days = int(ts.dt.date.nunique())
    agreement = int(samples["direction_agreement"].sum())
    conflicts = int(samples["trend_conflict"].sum())
    by_month = samples.assign(month=ts.dt.strftime("%Y-%m")).groupby("month").agg(
        samples=("direction","size"),
        longs=("direction", lambda s: int((s=="LONG").sum())),
        shorts=("direction", lambda s: int((s=="SHORT").sum())),
        agreement=("direction_agreement","sum"),
        conflicts=("trend_conflict","sum"),
        mean_quality_r=("label_quality_r","mean"),
    ).reset_index()

    manifest = {
        "research_version": RESEARCH_VERSION,
        "sequence": {"m1":6,"m5":1,"m15":1},
        "decision_frequency":"EVERY_CLOSED_M1",
        "primary_direction":"LAST_CLOSED_M5_EMA9_VS_EMA21",
        "m15_context":"LAST_CLOSED_M15_CLOSE_VS_EMA50",
        "conflict_policy":"KEEP_SAMPLE_AND_ENCODE_TREND_CONFLICT",
        "entry_contract":"DECISION_AT_M1_CLOSE_THEN_NEXT_M1_OPEN",
        "label_horizon_minutes":HORIZON,
        "samples":int(len(samples)),
        "market_days":days,
        "samples_per_market_day":float(len(samples)/days if days else 0.0),
        "direction_agreement_samples":agreement,
        "trend_conflict_samples":conflicts,
        "excluded":exclusions,
        "aug14_used":False,
    }
    (OUT / "PREPARE_611_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    by_month.to_csv(OUT / "samples_611_by_month.csv", index=False)

    print("="*78)
    print("6/1/1 M1-CENTRIC PREPARE | MAR 01 -> AUG 13 2026")
    print("EVERY CLOSED M1 IS ELIGIBLE; M5/M15 CONFLICT IS A FEATURE, NOT A FILTER")
    print("M1=6 timing bars | M5=primary direction | M15=context direction")
    print("ENTRY=next M1 open | LABEL=next 15 minutes, target only")
    print("="*78)
    print("PREPARE_611=PASS")
    print(f"SAMPLES={len(samples):,}")
    print(f"MARKET_DAYS={days}")
    print(f"SAMPLES_PER_MARKET_DAY={len(samples)/days:.2f}")
    print(f"DIRECTION_AGREEMENT={agreement:,}")
    print(f"TREND_CONFLICT_KEPT={conflicts:,}")
    print(f"CONFLICT_RATE_PCT={conflicts/len(samples)*100.0:.2f}")
    print("AUG14_USED_IN_FEATURES=NO")
    print("AUG14_USED_IN_LABELS=NO")
    print("FUTURE_LABELS_USED_AS_FEATURES=NO")
    print("BY_MONTH")
    print(by_month.to_string(index=False))
    print(f"OUTPUT={OUT}")


if __name__ == "__main__":
    main()
