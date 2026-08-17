import pandas as pd

from short_memory.evaluation import trade_metrics, compare_original_vs_short, promotion_verdict


def test_trade_metrics_profitable_sample():
    t = pd.DataFrame({
        'timestamp': pd.to_datetime(['2024-01-01T10:00Z','2024-01-01T11:00Z','2024-01-02T10:00Z','2024-01-02T11:00Z']),
        'pnl': [2.0, -1.0, 2.0, -1.0],
    })
    m = trade_metrics(t)
    assert m['trades'] == 4
    assert m['win_rate'] == 50.0
    assert m['pf'] == 2.0
    assert m['expectancy'] == 0.5
    assert m['trades_per_day'] == 2.0


def test_compare_original_vs_short_has_core_metrics():
    a = {'trades': 100, 'trades_per_day': 10, 'win_rate': 55, 'pf': 1.5, 'net_pnl': 100, 'expectancy': 1, 'max_dd_pct': 5}
    b = {'trades': 120, 'trades_per_day': 12, 'win_rate': 56, 'pf': 1.6, 'net_pnl': 120, 'expectancy': 1.1, 'max_dd_pct': 4}
    out = compare_original_vs_short(a, b)
    assert set(['pf','expectancy','trades_per_day','max_dd_pct']).issubset(set(out.metric))


def test_promotion_gate_rejects_negative_edge():
    assert promotion_verdict({'pf': 0.95, 'expectancy': -0.1, 'max_dd_pct': 3}) == 'REJECT_SHORT_MEMORY'


def test_promotion_gate_promising_requires_pf_and_dd():
    assert promotion_verdict({'pf': 1.35, 'expectancy': 0.1, 'max_dd_pct': 10}) == 'SHORT_MEMORY_PROMISING'
