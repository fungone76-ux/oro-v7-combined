from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def pick(obj, names):
    for name in names:
        if hasattr(obj, name):
            return name, getattr(obj, name)
    return None, None


def main() -> None:
    ap = argparse.ArgumentParser(description='Freeze SHORT_MEMORY strict fixed-0.01 policy and 3-loss cooldown')
    ap.add_argument('--top40-root', type=Path, required=True)
    ap.add_argument('--short-out', type=Path, required=True)
    ap.add_argument('--policy', choices=['TOP40_SHORT', 'TOP20_SHORT'], default='TOP40_SHORT')
    args = ap.parse_args()

    top = args.top40_root.resolve(); sout = args.short_out.resolve()
    sys.path.insert(0, str(top)); os.chdir(top)
    from xau_bot.research.ml import phase3c_economic_replay as p3c

    threshold_manifest_path = sout / 'short_memory_threshold_freeze_manifest.json'
    threshold_manifest = json.loads(threshold_manifest_path.read_text(encoding='utf-8'))
    if not threshold_manifest.get('strict_no_future_guard'):
        raise RuntimeError('STRICT_NO_FUTURE_GUARD_MISSING')

    cfg = p3c.BotConfig()
    risk = cfg.risk
    maxloss_name, maxloss = pick(risk, ['max_consecutive_losses', 'max_consecutive_loss', 'consecutive_loss_limit'])
    cooldown_name, cooldown = pick(risk, ['cooldown_minutes', 'loss_cooldown_minutes', 'cooldown_after_losses_minutes'])
    maxpos_name, maxpos = pick(risk, ['max_open_positions', 'max_positions'])

    if maxloss is None or int(maxloss) != 3:
        raise RuntimeError(f'COOLDOWN_LOSS_LIMIT_NOT_3: field={maxloss_name} value={maxloss}')
    if cooldown is None or int(cooldown) != 30:
        raise RuntimeError(f'COOLDOWN_MINUTES_NOT_30: field={cooldown_name} value={cooldown}')
    if maxpos is None or int(maxpos) != 3:
        raise RuntimeError(f'MAX_POSITIONS_NOT_3: field={maxpos_name} value={maxpos}')

    payload = {
        'policy_freeze': 'SHORT_MEMORY_V1_STRICT_FIXED_001',
        'sequence_lengths': {'m1': 8, 'm5': 5, 'm15': 3},
        'ml_policy': args.policy,
        's2_threshold': float(threshold_manifest['thresholds'][args.policy]),
        'fixed_lot': 0.01,
        'max_open_positions': 3,
        'max_consecutive_losses': 3,
        'cooldown_minutes': 30,
        'cooldown_behavior': 'FROZEN_FROM_CANONICAL_RISK_MANAGER; no tuning; same accepted-trade chronology must be preserved',
        'strict_no_future_guard': True,
        'final_fit_end_utc': threshold_manifest['final_fit_end_utc'],
        'evaluation_not_before_utc': threshold_manifest['evaluation_not_before_utc'],
        'threshold_manifest_sha256': sha256(threshold_manifest_path),
        's1_model_sha256': threshold_manifest['artifacts']['s1_model_sha256'] if 'artifacts' in threshold_manifest else threshold_manifest.get('artifact_hashes', {}).get('s1_model_sha256'),
        's2_model_sha256': threshold_manifest['artifacts']['s2_model_sha256'] if 'artifacts' in threshold_manifest else threshold_manifest.get('artifact_hashes', {}).get('s2_model_sha256'),
        'rules': [
            'NO threshold retuning',
            'NO sequence-length changes',
            'NO cooldown tuning',
            'NO martingale/grid/averaging down',
            'Fixed 0.01 economic diagnostic only',
        ],
    }
    out = sout / 'fixed_001_policy_freeze_manifest.json'
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print('FIXED_001_POLICY_FREEZE: PASS')
    print(f'fixed_lot={payload["fixed_lot"]}')
    print(f'max_consecutive_losses={payload["max_consecutive_losses"]}')
    print(f'cooldown_minutes={payload["cooldown_minutes"]}')
    print(f'max_open_positions={payload["max_open_positions"]}')
    print(f'strict_no_future_guard={payload["strict_no_future_guard"]}')
    print(f'manifest={out}')


if __name__ == '__main__':
    main()
