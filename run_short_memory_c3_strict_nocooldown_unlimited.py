from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


RESEARCH_MAX_POSITIONS = 999


def main() -> None:
    ap = argparse.ArgumentParser(
        description='STRICT no-future SHORT_MEMORY C3 research replay with cooldown disabled and position cap effectively removed'
    )
    ap.add_argument('--top40-root', type=Path, required=True)
    ap.add_argument('--short-out', type=Path, required=True)
    ap.add_argument('--policy', choices=['TOP40_SHORT', 'TOP20_SHORT'], default='TOP40_SHORT')
    ap.add_argument('--out', type=Path, default=Path('research_output/short_memory_c3_strict_nocooldown_unlimited'))
    a = ap.parse_args()

    repo = Path(__file__).resolve().parent
    top = a.top40_root.resolve()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(top))
    os.chdir(top)

    from xau_bot.research.ml import phase3c_economic_replay as p3c

    # 1) Disable only the consecutive-loss cooldown veto.
    original_register = p3c.RiskManager.register_trade_result

    def register_trade_result_no_cooldown(self, is_win, now_utc):
        original_register(self, is_win, now_utc)
        if hasattr(self, 'state') and hasattr(self.state, 'cooldown_until_utc'):
            self.state.cooldown_until_utc = None

    p3c.RiskManager.register_trade_result = register_trade_result_no_cooldown

    # 2) Remove the practical 3-position cap for RESEARCH ONLY.
    # We use 999 rather than infinity so every comparison remains integer-safe.
    OriginalBotConfig = p3c.BotConfig

    def UnlimitedBotConfig(*args, **kwargs):
        cfg = OriginalBotConfig(*args, **kwargs)
        try:
            cfg.risk.max_open_positions = RESEARCH_MAX_POSITIONS
        except Exception:
            object.__setattr__(cfg.risk, 'max_open_positions', RESEARCH_MAX_POSITIONS)
        return cfg

    p3c.BotConfig = UnlimitedBotConfig

    # 3) Surface actual concurrency in the segment report.
    original_variant_metrics = p3c._variant_metrics

    def variant_metrics_with_concurrency(segment, variant, result, candidate_decisions, threshold):
        row = original_variant_metrics(segment, variant, result, candidate_decisions, threshold)
        row['max_simultaneous_positions'] = int(result.metrics.get('max_simultaneous_positions', 0))
        row['research_position_cap'] = RESEARCH_MAX_POSITIONS
        row['max_total_lot_at_001'] = float(row['max_simultaneous_positions']) * 0.01
        return row

    p3c._variant_metrics = variant_metrics_with_concurrency

    import run_short_memory_c3_strict as strict

    sys.argv = [
        'run_short_memory_c3_strict.py',
        '--top40-root', str(top),
        '--short-out', str(a.short_out.resolve()),
        '--policy', a.policy,
        '--out', str(a.out.resolve()),
    ]

    print('UNLIMITED_POSITIONS_AUDIT=RESEARCH_ONLY', flush=True)
    print(f'RESEARCH_POSITION_CAP={RESEARCH_MAX_POSITIONS}', flush=True)
    print('COOLDOWN_AFTER_CONSECUTIVE_LOSSES=DISABLED', flush=True)
    strict.main()
    print('UNLIMITED_POSITIONS_AUDIT=COMPLETE', flush=True)


if __name__ == '__main__':
    main()
