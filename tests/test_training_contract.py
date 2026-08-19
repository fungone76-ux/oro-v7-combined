import numpy as np
import pandas as pd

from short_memory.config import CONFIG
from short_memory.training import fit_scalers_train_only, make_walkforward_folds, freeze_oos_thresholds


def test_scalers_are_fit_from_training_values_only():
    train = np.zeros((4, 8, 2), dtype=float)
    valid = np.full((2, 8, 2), 100.0, dtype=float)
    m5 = np.zeros((4, 5, 1), dtype=float)
    m15 = np.zeros((4, 3, 1), dtype=float)
    static = np.zeros((4, 2), dtype=float)
    scalers = fit_scalers_train_only(train, m5, m15, static)
    assert np.allclose(scalers['m1'].mean_, 0.0)
    # A validation distribution shift must not alter the fitted scaler.
    assert not np.allclose(valid.mean(axis=(0, 1)), scalers['m1'].mean_)


def test_walkforward_has_purge_plus_embargo_gap():
    times = pd.Series(pd.date_range('2024-01-01', periods=600, freq='5min', tz='UTC'))
    folds = make_walkforward_folds(times, 4)
    assert len(folds) == 4
    for f in folds:
        assert f.valid_start > f.train_end
        assert f.valid_start - f.train_end >= pd.Timedelta(minutes=CONFIG.purge_minutes + CONFIG.embargo_minutes)


def test_oos_thresholds_are_distribution_quantiles_not_legacy_constants():
    scores = np.linspace(-2.0, 2.0, 1000)
    t = freeze_oos_thresholds(scores)
    assert np.isclose(t['TOP40_SHORT'], np.quantile(scores, 0.60))
    assert np.isclose(t['TOP20_SHORT'], np.quantile(scores, 0.80))
    assert t['TOP40_SHORT'] != 0.28028725385665865
    assert t['TOP20_SHORT'] != 1.234266757965088
