from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description='Freeze SHORT_MEMORY strict fixed-0.01 policy with loss cooldown DISABLED')
    ap.add_argument('--short-out', type=Path, required=True)
    ap.add_argument('--policy', choices=['TOP40_SHORT', 'TOP20_SHORT'], default='TOP40_SHORT')
    args = ap.parse_args()

    sout = args.short_out.resolve()
    threshold_manifest_path = sout / 'short_memory_threshold_freeze_manifest.json'
    threshold_manifest = json.loads(threshold_manifest_path.read_text(encoding='utf-8'))
    if not threshold_manifest.get('strict_no_future_guard'):
        raise RuntimeError('STRICT_NO_FUTURE_GUARD_MISSING')

    payload = {
        'policy_freeze': 'SHORT_MEMORY_V1_STRICT_FIXED_001_NO_COOLDOWN',
        'sequence_lengths': {'m1': 8, 'm5': 5, 'm15': 3},
        'ml_policy': args.policy,
        's2_threshold': float(threshold_manifest['thresholds'][args.policy]),
        'fixed_lot': 0.01,
        'max_open_positions': 3,
        'loss_cooldown_enabled': False,
        'max_consecutive_losses_block': None,
        'cooldown_minutes': 0,
        'cooldown_behavior': 'DISABLED_FROZEN_POLICY: consecutive losses never block new entries',
        'strict_no_future_guard': True,
        'final_fit_end_utc': threshold_manifest['final_fit_end_utc'],
        'evaluation_not_before_utc': threshold_manifest['evaluation_not_before_utc'],
        'threshold_manifest_sha256': sha256(threshold_manifest_path),
        's1_model_sha256': threshold_manifest['artifacts']['s1_model_sha256'] if 'artifacts' in threshold_manifest else threshold_manifest.get('artifact_hashes', {}).get('s1_model_sha256'),
        's2_model_sha256': threshold_manifest['artifacts']['s2_model_sha256'] if 'artifacts' in threshold_manifest else threshold_manifest.get('artifact_hashes', {}).get('s2_model_sha256'),
        'rules': [
            'NO threshold retuning',
            'NO sequence-length changes',
            'LOSS COOLDOWN DISABLED',
            'NO martingale/grid/averaging down',
            'Fixed 0.01 economic diagnostic only',
            'STRICT no-future boundary preserved',
        ],
    }
    out = sout / 'fixed_001_policy_freeze_manifest.json'
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print('FIXED_001_POLICY_FREEZE: PASS')
    print(f'fixed_lot={payload["fixed_lot"]}')
    print('loss_cooldown_enabled=False')
    print('cooldown_minutes=0')
    print(f'max_open_positions={payload["max_open_positions"]}')
    print(f'strict_no_future_guard={payload["strict_no_future_guard"]}')
    print(f'manifest={out}')


if __name__ == '__main__':
    main()
