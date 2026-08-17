from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def max_drawdown_pct(pnls: pd.Series, initial_balance: float = 1000.0) -> tuple[float, float]:
    equity = initial_balance + pnls.cumsum()
    peak = equity.cummax()
    dd_abs = peak - equity
    dd_pct = (dd_abs / peak.replace(0, np.nan)) * 100.0
    return float(dd_abs.max() if len(dd_abs) else 0.0), float(dd_pct.max() if len(dd_pct) else 0.0)


def metrics(df: pd.DataFrame, start: pd.Timestamp | None = None, end: pd.Timestamp | None = None) -> dict:
    pnl = df['fixed_001_net_pnl'].astype(float)
    gp = float(pnl[pnl > 0].sum())
    gl = float(-pnl[pnl < 0].sum())
    pf = float(gp / gl) if gl > 0 else float('inf')
    dd_abs, dd_pct = max_drawdown_pct(pnl)
    days = None
    if start is not None and end is not None:
        days = max((end - start).total_seconds() / 86400.0, 1 / 24)
    return {
        'trades': int(len(df)),
        'wins': int((pnl > 0).sum()),
        'losses': int((pnl < 0).sum()),
        'win_rate_pct': float((pnl > 0).mean() * 100.0) if len(df) else 0.0,
        'gross_profit_usd': gp,
        'gross_loss_usd': gl,
        'net_profit_usd': float(pnl.sum()),
        'profit_factor': pf,
        'expectancy_usd': float(pnl.mean()) if len(df) else 0.0,
        'max_drawdown_usd': dd_abs,
        'max_drawdown_pct': dd_pct,
        'trades_per_calendar_day': float(len(df) / days) if days else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description='Strict no-future fixed 0.01 audit; preserves frozen 3-loss cooldown chronology')
    ap.add_argument('--strict-out', type=Path, required=True, help='Output directory of run_short_memory_c3_strict.py')
    ap.add_argument('--short-out', type=Path, required=True, help='SHORT_MEMORY training/freeze output directory')
    ap.add_argument('--out', type=Path, default=Path('research_output/short_memory_fixed001_strict'))
    args = ap.parse_args()

    strict_out = args.strict_out.resolve(); short_out = args.short_out.resolve(); out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    freeze = json.loads((short_out / 'fixed_001_policy_freeze_manifest.json').read_text(encoding='utf-8'))
    guard = json.loads((strict_out / 'strict_replay_guard.json').read_text(encoding='utf-8'))
    if not freeze.get('strict_no_future_guard') or not guard.get('strict_no_future_guard'):
        raise RuntimeError('STRICT_NO_FUTURE_GUARD_MISSING')
    if float(freeze['fixed_lot']) != 0.01:
        raise RuntimeError('FIXED_LOT_NOT_001')
    if int(freeze['max_consecutive_losses']) != 3 or int(freeze['cooldown_minutes']) != 30:
        raise RuntimeError('COOLDOWN_FREEZE_MISMATCH')

    trades_path = strict_out / 'trades_c3_strict.csv'
    policy_path = strict_out / 'policy_metrics_c3_strict.csv'
    trades = pd.read_csv(trades_path)
    policy = pd.read_csv(policy_path)
    if trades.empty:
        raise RuntimeError('NO_STRICT_TRADES')

    # Fixed-lot rescaling is exact for linear CFD P&L/commission/spread when the execution path is unchanged.
    lot_col = next((c for c in ['volume', 'lot', 'lot_size', 'lots'] if c in trades.columns), None)
    if lot_col is None:
        raise RuntimeError(f'LOT_COLUMN_NOT_FOUND columns={list(trades.columns)}')
    if 'net_pnl' not in trades.columns:
        raise RuntimeError('NET_PNL_COLUMN_NOT_FOUND')
    lot = pd.to_numeric(trades[lot_col], errors='coerce')
    if lot.isna().any() or (lot <= 0).any():
        raise RuntimeError('INVALID_SOURCE_LOT_VALUES')

    # Cooldown depends on win/loss chronology, not P&L magnitude. Positive scaling preserves every sign.
    source_sign = np.sign(pd.to_numeric(trades['net_pnl'], errors='raise').to_numpy())
    scale = 0.01 / lot.to_numpy(float)
    trades['source_lot'] = lot
    trades['fixed_lot'] = 0.01
    trades['fixed_001_net_pnl'] = pd.to_numeric(trades['net_pnl'], errors='raise').to_numpy(float) * scale
    fixed_sign = np.sign(trades['fixed_001_net_pnl'].to_numpy(float))
    if not np.array_equal(source_sign, fixed_sign):
        raise RuntimeError('WIN_LOSS_CHRONOLOGY_CHANGED')

    # If dynamic sizing never triggered a daily stop, fixed 0.01 cannot retroactively alter which trades existed.
    if 'daily_stop_count' in policy.columns and int(policy['daily_stop_count'].fillna(0).sum()) != 0:
        raise RuntimeError('DAILY_STOP_PRESENT_REQUIRES_FULL_FIXED_LOT_REPLAY')

    rows = []
    for seg, g in trades.groupby('segment_id', sort=False):
        prow = policy[policy['segment_id'].astype(str) == str(seg)]
        start = pd.Timestamp(prow.iloc[0]['start']) if not prow.empty and 'start' in prow.columns else None
        end = pd.Timestamp(prow.iloc[0]['end']) if not prow.empty and 'end' in prow.columns else None
        row = {'segment_id': seg, **metrics(g.reset_index(drop=True), start, end)}
        if not prow.empty:
            row['cooldown_count'] = int(prow.iloc[0].get('cooldown_count', 0))
            row['strict_dynamic_trades'] = int(prow.iloc[0].get('trades_opened', len(g)))
        rows.append(row)

    segment_metrics = pd.DataFrame(rows)
    aggregate = metrics(trades.reset_index(drop=True))
    aggregate.update({
        'segments': int(segment_metrics.shape[0]),
        'positive_pf_segments': int((segment_metrics['profit_factor'] > 1).sum()),
        'positive_expectancy_segments': int((segment_metrics['expectancy_usd'] > 0).sum()),
        'fixed_lot': 0.01,
        'max_consecutive_losses': 3,
        'cooldown_minutes': 30,
        'cooldown_chronology_preserved': True,
        'strict_no_future_guard': True,
        'method': 'rescale exact strict executed trade path to 0.01; valid because daily_stop_count=0 and P&L/costs scale linearly with lot',
    })

    trades.to_csv(out / 'trades_fixed001_strict.csv', index=False)
    segment_metrics.to_csv(out / 'segment_metrics_fixed001_strict.csv', index=False)
    (out / 'fixed001_strict_summary.json').write_text(json.dumps(aggregate, indent=2), encoding='utf-8')

    print('\nFINAL FIXED 0.01 STRICT SUMMARY')
    print(segment_metrics.to_string(index=False))
    for k, v in aggregate.items():
        print(f'{k}={v}')
    print('STRICT_NO_FUTURE_GUARD=PASS')
    print('COOLDOWN_AFTER_3_LOSSES=FROZEN')
    print('FIXED_LOT_001=PASS')


if __name__ == '__main__':
    main()
