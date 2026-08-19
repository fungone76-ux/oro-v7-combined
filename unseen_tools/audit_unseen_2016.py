from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED = {"M1": 1, "M5": 5, "M15": 15}


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit coverage/gaps for frozen unseen 2016 dataset")
    ap.add_argument("--data-dir", type=Path, required=True)
    args = ap.parse_args()
    root = args.data_dir.resolve()
    manifest = json.loads((root / "UNSEEN_2016_FROZEN_MANIFEST.json").read_text(encoding="utf-8"))
    files = manifest.get("files", {})

    rows = []
    hard_fail = False
    for tf, step in EXPECTED.items():
        info = files.get(tf, {})
        path = Path(info.get("path", root / f"XAUUSD_{tf}_UNSEEN_2016.csv"))
        if not path.exists():
            raise RuntimeError(f"MISSING_{tf}: {path}")
        df = pd.read_csv(path)
        if "time" not in df.columns:
            raise RuntimeError(f"TIME_COLUMN_MISSING_{tf}")
        t = pd.to_datetime(df["time"], utc=True).sort_values().drop_duplicates().reset_index(drop=True)
        d = t.diff().dropna().dt.total_seconds().div(60.0)
        gaps = d[d > step]
        # Ignore normal weekend closures only for summary; flag intraday discontinuities separately.
        intraday = gaps[gaps < 60 * 24 * 2]
        max_gap = float(gaps.max()) if len(gaps) else 0.0
        first = t.iloc[0]
        last = t.iloc[-1]
        in_2016 = t[(t >= pd.Timestamp("2016-01-01T00:00:00Z")) & (t <= pd.Timestamp("2016-12-31T23:59:59Z"))]
        coverage_days = int(in_2016.dt.normalize().nunique())
        row = {
            "timeframe": tf,
            "rows_total": int(len(t)),
            "rows_2016": int(len(in_2016)),
            "first_utc": first.isoformat(),
            "last_utc": last.isoformat(),
            "trading_dates_2016": coverage_days,
            "gaps_gt_expected": int(len(gaps)),
            "intraday_gaps_lt_2d": int(len(intraday)),
            "max_gap_minutes": max_gap,
        }
        rows.append(row)
        # Relative consistency check: M5 should be about 1/5 of M1 and M15 about 1/15,
        # allowing wide tolerance for broker-history sparsity and session closures.
    frame = pd.DataFrame(rows)
    by = frame.set_index("timeframe")
    m1 = max(int(by.loc["M1", "rows_2016"]), 1)
    r5 = int(by.loc["M5", "rows_2016"]) / m1
    r15 = int(by.loc["M15", "rows_2016"]) / m1
    ratio_ok = 0.12 <= r5 <= 0.30 and 0.035 <= r15 <= 0.12
    if not ratio_ok:
        hard_fail = True

    print("\nUNSEEN 2016 DATA COVERAGE AUDIT")
    print(frame.to_string(index=False))
    print(f"M5_TO_M1_RATIO={r5:.4f}")
    print(f"M15_TO_M1_RATIO={r15:.4f}")
    print(f"TIMEFRAME_RATIO_CHECK={'PASS' if ratio_ok else 'FAIL'}")
    print(f"UNSEEN_2016_DATA_AUDIT={'FAIL' if hard_fail else 'PASS'}")
    if hard_fail:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
