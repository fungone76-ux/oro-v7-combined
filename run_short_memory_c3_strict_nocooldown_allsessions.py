from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _print_session_breakdown(out: Path) -> None:
    trades_path = out / 'trades_c3_strict.csv'
    if not trades_path.exists():
        print('SESSION_BREAKDOWN=UNAVAILABLE trades file missing', flush=True)
        return
    trades = pd.read_csv(trades_path)
    if trades.empty or 'session' not in trades.columns:
        print('SESSION_BREAKDOWN=UNAVAILABLE session column missing', flush=True)
        return

    rows = []
    for session, g in trades.groupby('session', dropna=False):
        pnl = pd.to_numeric(g['net_pnl'], errors='coerce').fillna(0.0)
        gp = float(pnl[pnl > 0].sum())
        gl = float(-pnl[pnl < 0].sum())
        rows.append({
            'session': str(session),
            'trades': int(len(g)),
            'wins': int((pnl > 0).sum()),
            'losses': int((pnl < 0).sum()),
            'win_rate_pct': float((pnl > 0).mean() * 100.0),
            'gross_profit_usd_dynamic': gp,
            'gross_loss_usd_dynamic': gl,
            'net_profit_usd_dynamic': float(pnl.sum()),
            'profit_factor': float(gp / gl) if gl > 0 else float('inf'),
            'expectancy_usd_dynamic': float(pnl.mean()),
        })
    frame = pd.DataFrame(rows).sort_values('trades', ascending=False)
    frame.to_csv(out / 'session_breakdown_all_sessions.csv', index=False)
    print('\nALL-SESSIONS BREAKDOWN')
    print(frame.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(
        description='STRICT no-future SHORT_MEMORY 8/5/3 C3 replay with session filter and loss cooldown disabled'
    )
    ap.add_argument('--top40-root', type=Path, required=True)
    ap.add_argument('--short-out', type=Path, required=True)
    ap.add_argument('--policy', choices=['TOP40_SHORT', 'TOP20_SHORT'], default='TOP40_SHORT')
    ap.add_argument('--out', type=Path, default=Path('research_output/short_memory_c3_strict_nocooldown_allsessions'))
    a = ap.parse_args()

    repo = Path(__file__).resolve().parent
    top = a.top40_root.resolve()
    out = a.out.resolve()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(top))
    os.chdir(top)

    from xau_bot.research.ml import phase3c_economic_replay as p3c

    # 1) Disable ONLY the consecutive-loss cooldown veto.
    original_register = p3c.RiskManager.register_trade_result

    def register_trade_result_no_cooldown(self, is_win, now_utc):
        original_register(self, is_win, now_utc)
        if hasattr(self, 'state') and hasattr(self.state, 'cooldown_until_utc'):
            self.state.cooldown_until_utc = None

    p3c.RiskManager.register_trade_result = register_trade_result_no_cooldown

    # 2) Disable ONLY the session membership veto.
    # phase3c imported decide directly, so patch that bound reference. Passing the
    # snapshot's own session as the allowed tuple makes every market session pass,
    # while the original decision engine still enforces kill switch, account mode,
    # news lockout and technical-signal validity exactly as before.
    original_decide = p3c.decide

    def decide_all_sessions(signal, snapshot, kill_switch_blocks_entries, enabled_sessions=None,
                            live_confirmed=False, ml_state=None, ml_probability_up=None):
        kwargs = {
            'kill_switch_blocks_entries': kill_switch_blocks_entries,
            'enabled_sessions': (snapshot.session.value,),
            'live_confirmed': live_confirmed,
            'ml_probability_up': ml_probability_up,
        }
        if ml_state is not None:
            kwargs['ml_state'] = ml_state
        return original_decide(signal, snapshot, **kwargs)

    p3c.decide = decide_all_sessions

    import run_short_memory_c3_strict as strict

    sys.argv = [
        'run_short_memory_c3_strict.py',
        '--top40-root', str(top),
        '--short-out', str(a.short_out.resolve()),
        '--policy', a.policy,
        '--out', str(out),
    ]

    print('ALL_SESSIONS_AUDIT=RESEARCH_ONLY', flush=True)
    print('SESSION_FILTER=DISABLED', flush=True)
    print('COOLDOWN_AFTER_CONSECUTIVE_LOSSES=DISABLED', flush=True)
    print('MAX_POSITIONS=UNCHANGED_3', flush=True)
    print('STRICT_NO_FUTURE_GUARD=REQUIRED', flush=True)
    strict.main()
    _print_session_breakdown(out)
    print('SESSION_FILTER=DISABLED', flush=True)
    print('COOLDOWN_AFTER_CONSECUTIVE_LOSSES=DISABLED', flush=True)
    print('ALL_SESSIONS_AUDIT=COMPLETE', flush=True)


if __name__ == '__main__':
    main()
