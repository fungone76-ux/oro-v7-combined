from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

OUT = Path(r"D:\oro_top40_shortmem_OLD\unseen_2015_retry")
SYMBOL = "XAUUSD"
START = pd.Timestamp("2014-12-01T00:00:00Z")
END = pd.Timestamp("2016-01-05T23:59:00Z")
EVAL_START = pd.Timestamp("2015-01-01T00:00:00Z")
EVAL_END = pd.Timestamp("2015-12-31T23:59:59Z")
TIMEFRAMES = {
    "M1": (mt5.TIMEFRAME_M1, 1),
    "M5": (mt5.TIMEFRAME_M5, 5),
    "M15": (mt5.TIMEFRAME_M15, 15),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def month_chunks(start: pd.Timestamp, end: pd.Timestamp):
    cur = start
    while cur <= end:
        nxt = (cur + pd.offsets.MonthBegin(1)).normalize()
        chunk_end = min(nxt - pd.Timedelta(seconds=1), end)
        yield cur, chunk_end
        cur = nxt


def normalize_rates(rates) -> pd.DataFrame:
    df = pd.DataFrame(rates)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    cols = [c for c in ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"] if c in df.columns]
    return df[cols].drop_duplicates("time").sort_values("time").reset_index(drop=True)


def download_tf(name: str, timeframe: int, expected_minutes: int) -> pd.DataFrame:
    pieces = []
    print(f"\n=== DOWNLOAD {name} ===", flush=True)
    for i, (a, b) in enumerate(month_chunks(START, END), 1):
        rates = mt5.copy_rates_range(SYMBOL, timeframe, a.to_pydatetime(), b.to_pydatetime())
        rows = 0 if rates is None else len(rates)
        print(f"{name} chunk {i:02d}: {a.date()} -> {b.date()} rows={rows}", flush=True)
        if rates is not None and len(rates):
            pieces.append(normalize_rates(rates))
    if not pieces:
        raise RuntimeError(f"NO_DATA_{name}: {mt5.last_error()}")
    df = pd.concat(pieces, ignore_index=True).drop_duplicates("time").sort_values("time").reset_index(drop=True)

    times = df["time"].to_numpy(dtype="datetime64[m]")
    diffs = np.diff(times).astype("timedelta64[m]").astype(int)
    # Ignore weekend/large market closures for frequency diagnostics.
    intraday = diffs[(diffs > 0) & (diffs < 120)]
    median = float(np.median(intraday)) if len(intraday) else float("nan")
    exact_ratio = float(np.mean(intraday == expected_minutes)) if len(intraday) else 0.0
    print(f"{name} assembled rows={len(df):,} median_intraday_step={median:.2f}m exact_{expected_minutes}m_ratio={exact_ratio:.4f}", flush=True)

    if not np.isfinite(median) or abs(median - expected_minutes) > 0.1 or exact_ratio < 0.90:
        raise RuntimeError(
            f"{name}_TIMEFRAME_INVALID expected={expected_minutes}m exact_ratio={exact_ratio:.4f} median={median}"
        )
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not mt5.initialize():
        raise SystemExit(f"MT5 initialize failed: {mt5.last_error()}")
    try:
        acc = mt5.account_info()
        if acc is None:
            raise SystemExit(f"MT5 account unavailable: {mt5.last_error()}")
        if not mt5.symbol_select(SYMBOL, True):
            raise SystemExit(f"Cannot select {SYMBOL}: {mt5.last_error()}")
        print(f"MT5 connected | login={acc.login} server={acc.server}", flush=True)

        manifest = {
            "symbol": SYMBOL,
            "download_start_utc": START.isoformat(),
            "download_end_utc": END.isoformat(),
            "evaluation_start_utc": EVAL_START.isoformat(),
            "evaluation_end_utc": EVAL_END.isoformat(),
            "evaluation_year": 2015,
            "account_login": int(acc.login),
            "server": str(acc.server),
            "frozen_threshold": 0.32696733474731443,
            "no_retraining": True,
            "no_threshold_recalibration": True,
            "files": {},
        }

        frames = {}
        for name, (tf, minutes) in TIMEFRAMES.items():
            df = download_tf(name, tf, minutes)
            frames[name] = df
            p = OUT / f"XAUUSD_{name}_UNSEEN_2015.csv"
            df.to_csv(p, index=False)
            manifest["files"][name] = {
                "path": str(p),
                "rows": int(len(df)),
                "first_bar_utc": df.time.iloc[0].isoformat(),
                "last_bar_utc": df.time.iloc[-1].isoformat(),
                "sha256": sha256(p),
            }

        m1n, m5n, m15n = map(len, (frames["M1"], frames["M5"], frames["M15"]))
        r5 = m5n / m1n
        r15 = m15n / m1n
        print(f"M5_TO_M1_RATIO={r5:.4f}")
        print(f"M15_TO_M1_RATIO={r15:.4f}")
        # Broad sanity bands because trading-day gaps and broker peculiarities exist.
        if not (0.15 <= r5 <= 0.25 and 0.04 <= r15 <= 0.09):
            raise RuntimeError(f"TIMEFRAME_RATIO_INVALID M5/M1={r5:.4f} M15/M1={r15:.4f}")

        mp = OUT / "UNSEEN_2015_DOWNLOAD_MANIFEST.json"
        mp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print("UNSEEN_2015_CHUNKED_DOWNLOAD=PASS")
        print(f"MANIFEST={mp}")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
