from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import MetaTrader5 as mt5
import numpy as np
import pandas as pd

OUT = Path(r"D:\ORO_532_2026_UNSEEN_JULY")
SYMBOL = "XAUUSD"
# July is the unseen evaluation month. A small August tail is downloaded only
# so labels / position exits near the final July bars can be resolved causally.
START = pd.Timestamp("2026-07-01T00:00:00Z")
END = pd.Timestamp("2026-08-03T06:00:00Z")
EVAL_START = pd.Timestamp("2026-07-01T00:00:00Z")
EVAL_END = pd.Timestamp("2026-07-31T23:59:59Z")
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
        next_month = cur.normalize() + pd.offsets.MonthBegin(1)
        nxt = min(next_month, end + pd.Timedelta(seconds=1))
        if nxt <= cur:
            nxt = min(cur + pd.Timedelta(days=31), end + pd.Timedelta(seconds=1))
        yield cur, nxt - pd.Timedelta(seconds=1)
        cur = nxt


def download_tf(name: str, timeframe: int, expected_minutes: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for i, (a, b) in enumerate(month_chunks(START, END), 1):
        rates = None
        for attempt in range(5):
            rates = mt5.copy_rates_range(
                SYMBOL,
                timeframe,
                a.to_pydatetime(),
                b.to_pydatetime(),
            )
            if rates is not None and len(rates) > 0:
                break
            time.sleep(1.0 + attempt)
        n = 0 if rates is None else len(rates)
        print(f"{name} chunk {i:02d}: {a} -> {b} rows={n:,}", flush=True)
        if rates is not None and len(rates):
            part = pd.DataFrame(rates)
            part["time"] = pd.to_datetime(part["time"], unit="s", utc=True)
            frames.append(part)
        time.sleep(0.1)

    if not frames:
        raise RuntimeError(f"NO_DATA_{name}: {mt5.last_error()}")

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    cols = [
        c for c in [
            "time", "open", "high", "low", "close",
            "tick_volume", "spread", "real_volume",
        ] if c in df.columns
    ]
    df = df[cols]

    delta = df["time"].diff().dt.total_seconds().div(60.0)
    intraday = delta[(delta > 0) & (delta < 180)]
    exact_ratio = (
        float(np.mean(np.isclose(intraday.to_numpy(float), expected_minutes)))
        if len(intraday) else 0.0
    )
    median_step = float(intraday.median()) if len(intraday) else float("nan")
    print(
        f"{name} assembled rows={len(df):,} median_intraday_step={median_step:.2f}m "
        f"exact_{expected_minutes}m_ratio={exact_ratio:.4f}",
        flush=True,
    )
    if exact_ratio < 0.80:
        raise RuntimeError(
            f"{name}_TIMEFRAME_INVALID expected={expected_minutes}m "
            f"exact_ratio={exact_ratio:.4f} median={median_step}"
        )
    return df


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not mt5.initialize():
        raise SystemExit(f"MT5 initialize failed: {mt5.last_error()}")
    account = mt5.account_info()
    if account is None:
        mt5.shutdown()
        raise SystemExit(f"MT5 account unavailable: {mt5.last_error()}")
    if not mt5.symbol_select(SYMBOL, True):
        mt5.shutdown()
        raise SystemExit(f"Cannot select {SYMBOL}: {mt5.last_error()}")

    print(f"MT5 connected | login={account.login} server={account.server}", flush=True)
    manifest: dict[str, object] = {
        "symbol": SYMBOL,
        "download_start_utc": START.isoformat(),
        "download_end_utc": END.isoformat(),
        "evaluation_start_utc": EVAL_START.isoformat(),
        "evaluation_end_utc": EVAL_END.isoformat(),
        "evaluation_month": "2026-07",
        "account_login": int(account.login),
        "server": str(account.server),
        "method": "fresh_mt5_copy_rates_range_july_unseen_with_august_tail",
        "july_used_for_training": False,
        "july_used_for_threshold_selection": False,
        "august_tail_role": "labels_and_exit_resolution_only",
        "files": {},
    }

    try:
        for name, (tf, minutes) in TIMEFRAMES.items():
            print(f"\n=== DOWNLOAD {name} ===", flush=True)
            df = download_tf(name, tf, minutes)
            path = OUT / f"XAUUSD_{name}_FRESH_JULY_UNSEEN_2026.csv"
            df.to_csv(path, index=False)
            eval_rows = int(((df["time"] >= EVAL_START) & (df["time"] <= EVAL_END)).sum())
            manifest["files"][name] = {
                "path": str(path),
                "rows_total": int(len(df)),
                "rows_july_eval": eval_rows,
                "first_bar_utc": df["time"].iloc[0].isoformat(),
                "last_bar_utc": df["time"].iloc[-1].isoformat(),
                "sha256": sha256(path),
            }
    finally:
        mt5.shutdown()

    files = manifest["files"]
    assert isinstance(files, dict)
    m1 = int(files["M1"]["rows_total"])
    m5 = int(files["M5"]["rows_total"])
    m15 = int(files["M15"]["rows_total"])
    ratio5 = m5 / m1 if m1 else 0.0
    ratio15 = m15 / m1 if m1 else 0.0
    manifest["ratios"] = {"m5_to_m1": ratio5, "m15_to_m1": ratio15}
    if not (0.12 <= ratio5 <= 0.28 and 0.035 <= ratio15 <= 0.12):
        raise RuntimeError(
            f"TIMEFRAME_ROW_RATIO_INVALID m5/m1={ratio5:.4f} m15/m1={ratio15:.4f}"
        )

    out_manifest = OUT / "FRESH_JULY_UNSEEN_2026_MANIFEST.json"
    out_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nFRESH_JULY_UNSEEN_2026_DOWNLOAD=PASS")
    print(f"M5_TO_M1_RATIO={ratio5:.4f}")
    print(f"M15_TO_M1_RATIO={ratio15:.4f}")
    print("JULY_USED_FOR_TRAINING=NO")
    print("JULY_USED_FOR_THRESHOLD_SELECTION=NO")
    print(f"MANIFEST={out_manifest}")


if __name__ == "__main__":
    main()
