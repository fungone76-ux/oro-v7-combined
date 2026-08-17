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
        'trades': int(len(df)), 'wins': int((pnl > 0).sum()), 'losses': int((pnl < 0).sum()),
        'win_rate_pct': float((pnl > 0).mean() * 100.0) if len(df) else 0.0,
        'gross_profit_usd': gp, 'gross_loss_usd': gl, 'net_profit_usd': float(pnl.sum()),
        'profit_factor': pf, 'expectancy_usd': float(pnl.mean()) if len(df) else 0.0,
        'max_drawdown_usd': dd_abs, 'max_drawdown_pct': dd_pct,
        'trades_per_calendar_day': float(len(df) / days) if days else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description='Fixed 0.01 STRICT no-future ALL-SESSIONS audit, cooldown disabled')
    ap.add_argument('--strict-out', type=Path, required=True)
    ap.add_argument('--short-out', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('research_output/short_memory_fixed001_strict_nocooldown_allsessions'))
    args = ap.parse_args()
    strict_out=args.strict_out.resolve(); short_out=args.short_out.resolve(); out=args.out.resolve(); out.mkdir(parents=True, exist_ok=True)

    freeze=json.loads((short_out/'fixed_001_policy_freeze_manifest.json').read_text(encoding='utf-8'))
    guard=json.loads((strict_out/'strict_replay_guard.json').read_text(encoding='utf-8'))
    if not freeze.get('strict_no_future_guard') or not guard.get('strict_no_future_guard'): raise RuntimeError('STRICT_NO_FUTURE_GUARD_MISSING')
    if float(freeze['fixed_lot']) != 0.01: raise RuntimeError('FIXED_LOT_NOT_001')
    if freeze.get('loss_cooldown_enabled') is not False or int(freeze.get('cooldown_minutes',-1)) != 0: raise RuntimeError('NO_COOLDOWN_FREEZE_MISMATCH')

    trades=pd.read_csv(strict_out/'trades_c3_strict.csv'); policy=pd.read_csv(strict_out/'policy_metrics_c3_strict.csv')
    if trades.empty: raise RuntimeError('NO_STRICT_TRADES')
    if 'cooldown_count' in policy.columns and int(policy['cooldown_count'].fillna(0).sum()) != 0: raise RuntimeError('COOLDOWN_WAS_ACTIVE_IN_SOURCE_REPLAY')
    if 'daily_stop_count' in policy.columns and int(policy['daily_stop_count'].fillna(0).sum()) != 0: raise RuntimeError('DAILY_STOP_PRESENT_REQUIRES_FULL_FIXED_LOT_REPLAY')

    lot_col=next((c for c in ['volume','lot','lot_size','lots'] if c in trades.columns),None)
    if lot_col is None: raise RuntimeError(f'LOT_COLUMN_NOT_FOUND columns={list(trades.columns)}')
    lot=pd.to_numeric(trades[lot_col],errors='coerce')
    if lot.isna().any() or (lot<=0).any(): raise RuntimeError('INVALID_SOURCE_LOT_VALUES')
    source_pnl=pd.to_numeric(trades['net_pnl'],errors='raise').to_numpy(float)
    trades['source_lot']=lot; trades['fixed_lot']=0.01; trades['fixed_001_net_pnl']=source_pnl*(0.01/lot.to_numpy(float))
    if not np.array_equal(np.sign(source_pnl),np.sign(trades['fixed_001_net_pnl'].to_numpy(float))): raise RuntimeError('WIN_LOSS_CHRONOLOGY_CHANGED')

    segment_rows=[]
    for seg,g in trades.groupby('segment_id',sort=False):
        prow=policy[policy['segment_id'].astype(str)==str(seg)]
        start=pd.Timestamp(prow.iloc[0]['start']) if not prow.empty else None; end=pd.Timestamp(prow.iloc[0]['end']) if not prow.empty else None
        segment_rows.append({'segment_id':seg,**metrics(g.reset_index(drop=True),start,end)})
    segment_metrics=pd.DataFrame(segment_rows)
    if 'session' not in trades.columns: raise RuntimeError(f'SESSION_COLUMN_NOT_FOUND columns={list(trades.columns)}')
    session_metrics=pd.DataFrame([{'session':session,**metrics(g.reset_index(drop=True))} for session,g in trades.groupby('session',sort=False)]).sort_values('trades',ascending=False)
    aggregate=metrics(trades.reset_index(drop=True)); aggregate.update({'segments':int(len(segment_metrics)),'positive_pf_segments':int((segment_metrics.profit_factor>1).sum()),'positive_expectancy_segments':int((segment_metrics.expectancy_usd>0).sum()),'positive_pf_sessions':int((session_metrics.profit_factor>1).sum()),'positive_expectancy_sessions':int((session_metrics.expectancy_usd>0).sum()),'fixed_lot':0.01,'max_positions':3,'loss_cooldown_enabled':False,'cooldown_minutes':0,'session_filter_enabled':False,'strict_no_future_guard':True,'method':'exact linear rescale of STRICT ALL-SESSIONS no-cooldown executed path to fixed 0.01; valid because daily_stop_count=0'})
    trades.to_csv(out/'trades_fixed001_strict_allsessions.csv',index=False); segment_metrics.to_csv(out/'segment_metrics_fixed001_strict_allsessions.csv',index=False); session_metrics.to_csv(out/'session_metrics_fixed001_strict_allsessions.csv',index=False)
    (out/'fixed001_strict_allsessions_summary.json').write_text(json.dumps(aggregate,indent=2),encoding='utf-8')
    print('\nFINAL FIXED 0.01 STRICT ALL-SESSIONS SUMMARY'); print(segment_metrics.to_string(index=False)); print('\nFIXED 0.01 SESSION BREAKDOWN'); print(session_metrics.to_string(index=False))
    for k,v in aggregate.items(): print(f'{k}={v}')
    print('STRICT_NO_FUTURE_GUARD=PASS'); print('SESSION_FILTER=DISABLED'); print('COOLDOWN_AFTER_CONSECUTIVE_LOSSES=DISABLED'); print('FIXED_LOT_001=PASS')

if __name__=='__main__': main()
