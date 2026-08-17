import numpy as np
import pandas as pd

from short_memory.dataset import build_tensor_bundle
from short_memory.training import M1_FEATURES, M5_FEATURES, M15_FEATURES, STATIC_NUMERIC, STATIC_CATEGORICAL


def test_phase3a_adapter_builds_exact_short_memory_shapes():
    n = 181
    ts = pd.date_range('2024-01-01', periods=n, freq='1min', tz='UTC')
    market = pd.DataFrame({'timestamp': ts, 'feature_valid': True})
    for c in M1_FEATURES:
        market[c] = np.arange(n, dtype=float)
    # source-close timestamps are sparse on the M1 state table, exactly as Phase 3A diagnostics.
    market['m5_source_close_time'] = pd.NaT
    market['m15_source_close_time'] = pd.NaT
    for i in range(4, n, 5):
        market.loc[i, 'm5_source_close_time'] = ts[i]
    for i in range(14, n, 15):
        market.loc[i, 'm15_source_close_time'] = ts[i]
    for c in M5_FEATURES:
        market[c] = np.arange(n, dtype=float)
    for c in M15_FEATURES:
        market[c] = np.arange(n, dtype=float)

    ctime = ts[179]
    cand = pd.DataFrame({'timestamp': [ctime], 'feature_valid': [True], 'label_validity_15m': ['VALID'],
                         'directional_future_return_15m': [1.0], 'atr14': [2.0],
                         'directional_MFE_price_15m': [3.0], 'directional_MAE_price_15m': [1.0]})
    for c in STATIC_NUMERIC:
        cand[c] = 1.0
    for c in STATIC_CATEGORICAL:
        cand[c] = 'LONG'

    b = build_tensor_bundle(market, cand)
    assert b.m1.shape == (1, 8, len(M1_FEATURES))
    assert b.m5.shape == (1, 5, len(M5_FEATURES))
    assert b.m15.shape == (1, 3, len(M15_FEATURES))
    assert b.y_s1[0] == 1.0
    assert np.isclose(b.y_s2[0], 1.0)
