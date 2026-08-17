from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from short_memory.config import CONFIG
from short_memory.dataset import load_phase3a, build_tensor_bundle
from short_memory.preprocess import preprocess_fold
from short_memory.runner import load_frozen_folds, make_loader
from short_memory.training import set_deterministic, torch_model_class


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fit_subset(bundle, idx: np.ndarray, target: str, device: str, epochs: int):
    import torch
    set_deterministic(CONFIG.seed + (1000 if target == 'S2' else 900))
    m1, m5, m15, static, scalers = preprocess_fold(bundle, idx)
    y = bundle.y_s1 if target == 'S1' else bundle.y_s2
    loader = make_loader(m1[idx], m5[idx], m15[idx], static[idx], y[idx], True)
    Model = torch_model_class()
    model = Model(m1.shape[2], m5.shape[2], m15.shape[2], static.shape[1], target).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG.learning_rate)
    loss_fn = torch.nn.BCEWithLogitsLoss() if target == 'S1' else torch.nn.SmoothL1Loss()
    for epoch in range(1, max(1, epochs) + 1):
        model.train()
        for bm1, bm5, bm15, bst, by in loader:
            bm1, bm5, bm15, bst, by = bm1.to(device), bm5.to(device), bm15.to(device), bst.to(device), by.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(bm1, bm5, bm15, bst), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        print(f'{target} strict final epoch {epoch}/{epochs}', flush=True)
    return model, scalers


def main() -> None:
    ap = argparse.ArgumentParser(description='Refit SHORT_MEMORY final models with a hard no-future cutoff')
    ap.add_argument('--market-state', type=Path, required=True)
    ap.add_argument('--candidates', type=Path, required=True)
    ap.add_argument('--short-out', type=Path, required=True)
    ap.add_argument('--splits', type=Path, default=Path('config/walkforward_splits.json'))
    a = ap.parse_args()

    import torch
    out = a.short_out.resolve()
    summary_path = out / 'short_memory_training_summary.json'
    freeze_path = out / 'short_memory_threshold_freeze_manifest.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    freeze = json.loads(freeze_path.read_text(encoding='utf-8'))

    market, candidates = load_phase3a(a.market_state, a.candidates)
    bundle = build_tensor_bundle(market, candidates)
    folds = load_frozen_folds(a.splits)
    development_end = max(f.valid_end for f in folds)
    evaluation_not_before = development_end + pd.Timedelta(minutes=CONFIG.purge_minutes + CONFIG.embargo_minutes)

    ts = pd.to_datetime(bundle.frame['timestamp'], utc=True)
    fit_idx = np.flatnonzero((ts <= development_end).to_numpy())
    excluded_idx = np.flatnonzero((ts > development_end).to_numpy())
    if len(fit_idx) == 0:
        raise RuntimeError('STRICT_FIT_HAS_ZERO_ROWS')
    if pd.Timestamp(ts.iloc[fit_idx].max()) > development_end:
        raise RuntimeError('FUTURE_ROW_ENTERED_STRICT_FIT')

    # Feature contract audit: future outcome/label columns are targets only, never model inputs.
    from short_memory.training import M1_FEATURES, M5_FEATURES, M15_FEATURES, STATIC_NUMERIC, STATIC_CATEGORICAL
    model_features = M1_FEATURES + M5_FEATURES + M15_FEATURES + STATIC_NUMERIC + STATIC_CATEGORICAL
    forbidden_tokens = ('future', 'mfe', 'mae', 'label_', 'target', 'exit_', 'pnl')
    bad = [c for c in model_features if any(tok in c.lower() for tok in forbidden_tokens)]
    if bad:
        raise RuntimeError(f'FUTURE_FEATURE_CONTRACT_VIOLATION={bad}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    epochs = summary['median_best_epochs']
    hashes = {}
    for target in ('S1', 'S2'):
        model, scalers = fit_subset(bundle, fit_idx, target, device, int(epochs[target]))
        mp = out / f'selected_short_memory_{target.lower()}.pt'
        sp = out / f'short_memory_scalers_{target.lower()}.joblib'
        torch.save(model.state_dict(), mp)
        joblib.dump(scalers, sp)
        hashes[f'{target.lower()}_model_sha256'] = sha256(mp)
        hashes[f'{target.lower()}_scaler_sha256'] = sha256(sp)

    freeze['artifact_hashes'] = hashes
    freeze['strict_no_future_guard'] = True
    freeze['final_fit_start_utc'] = pd.Timestamp(ts.iloc[fit_idx].min()).isoformat()
    freeze['final_fit_end_utc'] = development_end.isoformat()
    freeze['evaluation_not_before_utc'] = evaluation_not_before.isoformat()
    freeze['final_fit_rows'] = int(len(fit_idx))
    freeze['rows_excluded_as_future'] = int(len(excluded_idx))
    freeze['feature_future_token_audit'] = 'PASS'
    freeze_path.write_text(json.dumps(freeze, indent=2), encoding='utf-8')

    audit = {
        'STRICT_NO_FUTURE_GUARD': 'PASS',
        'sequence_lengths': {'m1': 8, 'm5': 5, 'm15': 3},
        'development_end_utc': development_end.isoformat(),
        'evaluation_not_before_utc': evaluation_not_before.isoformat(),
        'fit_rows': int(len(fit_idx)),
        'excluded_future_rows': int(len(excluded_idx)),
        'max_timestamp_seen_by_final_fit': pd.Timestamp(ts.iloc[fit_idx].max()).isoformat(),
        'min_timestamp_excluded': None if len(excluded_idx) == 0 else pd.Timestamp(ts.iloc[excluded_idx].min()).isoformat(),
        'future_columns_in_model_inputs': bad,
        'threshold_source': 'existing strictly OOS fold predictions only',
        'artifact_hashes': hashes,
    }
    (out / 'strict_no_future_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')

    print('\nSTRICT NO-FUTURE AUDIT')
    for k, v in audit.items():
        print(f'{k}: {v}')
    print('\nIMPORTANT: previous C3 results produced before this strict refit are INVALIDATED.')


if __name__ == '__main__':
    main()
