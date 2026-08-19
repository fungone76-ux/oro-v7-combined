import pandas as pd
import pytest

from short_memory.sequences import build_short_memory_window


def frame(start, periods, freq):
    return pd.DataFrame({
        "time": pd.date_range(start, periods=periods, freq=freq, tz="UTC"),
        "value": range(periods),
    })


def test_exact_lengths_and_no_future_rows():
    m1 = frame("2026-01-01 10:00", 30, "1min")
    m5 = frame("2026-01-01 08:00", 40, "5min")
    m15 = frame("2026-01-01 06:00", 30, "15min")
    ts = pd.Timestamp("2026-01-01 10:20", tz="UTC")
    w = build_short_memory_window(m1, m5, m15, ts)
    assert len(w.m1) == 8
    assert len(w.m5) == 5
    assert len(w.m15) == 3
    assert w.m1.time.max() <= ts
    assert w.m5.time.max() <= ts
    assert w.m15.time.max() <= ts


def test_future_rows_are_ignored_not_backfilled():
    m1 = frame("2026-01-01 10:00", 40, "1min")
    m5 = frame("2026-01-01 08:00", 40, "5min")
    m15 = frame("2026-01-01 06:00", 30, "15min")
    ts = pd.Timestamp("2026-01-01 10:20", tz="UTC")
    w = build_short_memory_window(m1, m5, m15, ts)
    assert (w.m1.time > ts).sum() == 0


def test_market_closure_crossing_is_rejected():
    m1 = frame("2026-01-01 10:00", 20, "1min").drop(index=15).reset_index(drop=True)
    m5 = frame("2026-01-01 08:00", 40, "5min")
    m15 = frame("2026-01-01 06:00", 30, "15min")
    with pytest.raises(ValueError, match="CROSSES_MARKET_CLOSURE"):
        build_short_memory_window(m1, m5, m15, pd.Timestamp("2026-01-01 10:19", tz="UTC"))


def test_insufficient_history_is_rejected():
    m1 = frame("2026-01-01 10:00", 7, "1min")
    m5 = frame("2026-01-01 08:00", 20, "5min")
    m15 = frame("2026-01-01 06:00", 20, "15min")
    with pytest.raises(ValueError, match="INSUFFICIENT_HISTORY"):
        build_short_memory_window(m1, m5, m15, pd.Timestamp("2026-01-01 10:06", tz="UTC"))
