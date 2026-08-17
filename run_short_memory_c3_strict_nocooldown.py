from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description='STRICT no-future SHORT_MEMORY C3 replay with consecutive-loss cooldown DISABLED')
    ap.add_argument('--top40-root', type=Path, required=True)
    ap.add_argument('--short-out', type=Path, required=True)
    ap.add_argument('--policy', choices=['TOP40_SHORT', 'TOP20_SHORT'], default='TOP40_SHORT')
    ap.add_argument('--out', type=Path, default=Path('research_output/short_memory_c3_strict_nocooldown'))
    a = ap.parse_args()

    repo = Path(__file__).resolve().parent
    top = a.top40_root.resolve()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(top))
    os.chdir(top)

    # Patch ONLY the consecutive-loss cooldown effect. The canonical risk manager
    # still records win/loss chronology, but cooldown_until_utc is cleared
    # immediately after each result, so it can never veto a subsequent entry.
    from xau_bot.research.ml import phase3c_economic_replay as p3c

    original_register = p3c.RiskManager.register_trade_result

    def register_trade_result_no_cooldown(self, is_win, now_utc):
        original_register(self, is_win, now_utc)
        if hasattr(self, 'state') and hasattr(self.state, 'cooldown_until_utc'):
            self.state.cooldown_until_utc = None

    p3c.RiskManager.register_trade_result = register_trade_result_no_cooldown

    import run_short_memory_c3_strict as strict

    # Reuse the already-audited strict no-future runner. Only the risk-manager
    # cooldown veto is neutralized. Model, threshold, sequences, cutoff, C3
    # execution and max-position logic are unchanged.
    sys.argv = [
        'run_short_memory_c3_strict.py',
        '--top40-root', str(top),
        '--short-out', str(a.short_out.resolve()),
        '--policy', a.policy,
        '--out', str(a.out.resolve()),
    ]
    print('NO_COOLDOWN_POLICY=FROZEN_DISABLED', flush=True)
    strict.main()
    print('COOLDOWN_AFTER_CONSECUTIVE_LOSSES=DISABLED', flush=True)


if __name__ == '__main__':
    main()
