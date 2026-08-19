from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from short_memory.dataset import load_phase3a, build_tensor_bundle


def log(t0: float, msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] +{time.perf_counter()-t0:7.1f}s | {msg}', flush=True)


def transform(bundle, scalers):
    def seq(values: np.ndarray, key: str) -> np.ndarray:
        shape = values.shape
        return scalers[key].transform(values.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)
    m1 = seq(bundle.m1, 'm1'); m5 = seq(bundle.m5, 'm5'); m15 = seq(bundle.m15, 'm15')
    sn = scalers['static_numeric'].transform(bundle.static_num).astype(np.float32)
    pieces = []
    for col, values in scalers.get('categories', {}).items():
        cv = bundle.static_cat[col].astype(str).fillna('<NA>')
        for value in values:
            pieces.append((cv == value).astype(float).to_numpy()[:, None])
    sc = np.hstack(pieces).astype(np.float32) if pieces else np.empty((len(bundle.frame), 0), dtype=np.float32)
    return m1, m5, m15, np.hstack([sn, sc]).astype(np.float32)


def load_model(path: Path, dims, device: str):
    import torch
    from short_memory.training import torch_model_class
    Model = torch_model_class()
    model = Model(dims[0], dims[1], dims[2], dims[3], 'S2').to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def predict(model, arrays, device: str) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    m1, m5, m15, st = arrays
    ds = TensorDataset(torch.tensor(m1), torch.tensor(m5), torch.tensor(m15), torch.tensor(st))
    out = []
    with torch.no_grad():
        for bm1, bm5, bm15, bst in DataLoader(ds, batch_size=256, shuffle=False):
            out.append(model(bm1.to(device), bm5.to(device), bm15.to(device), bst.to(device)).cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser(description='Strict no-future SHORT_MEMORY 8/5/3 C3 replay')
    ap.add_argument('--top40-root', type=Path, required=True)
    ap.add_argument('--short-out', type=Path, required=True)
    ap.add_argument('--policy', choices=['TOP40_SHORT', 'TOP20_SHORT'], default='TOP40_SHORT')
    ap.add_argument('--out', type=Path, default=Path('research_output/short_memory_c3_strict'))
    a = ap.parse_args()

    t0 = time.perf_counter(); repo = Path(__file__).resolve().parent
    top = a.top40_root.resolve(); sout = a.short_out.resolve(); out = a.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo)); sys.path.insert(0, str(top)); os.chdir(top)

    import torch
    from xau_bot.research.ml import phase3c_economic_replay as p3c

    manifest = json.loads((sout/'short_memory_threshold_freeze_manifest.json').read_text(encoding='utf-8'))
    if not manifest.get('strict_no_future_guard'):
        raise RuntimeError('STRICT_NO_FUTURE_GUARD_MISSING: run freeze_strict_final.py first')
    cutoff = pd.Timestamp(manifest['final_fit_end_utc'])
    eval_not_before = pd.Timestamp(manifest['evaluation_not_before_utc'])
    threshold = float(manifest['thresholds'][a.policy])

    log(t0, '=== STRICT NO-FUTURE C3 START ===')
    log(t0, f'FINAL FIT CUTOFF={cutoff.isoformat()}')
    log(t0, f'EVALUATION NOT BEFORE={eval_not_before.isoformat()}')
    log(t0, f'POLICY={a.policy} threshold={threshold:.9f}')

    market_path = top/'backtests/PHASE_3A_ML_DATASET/market_state.parquet'
    candidate_path = top/'backtests/PHASE_3A_ML_DATASET/strategy_candidates.parquet'
    market, candidates = load_phase3a(market_path, candidate_path)
    bundle = build_tensor_bundle(market, candidates)
    arrays = transform(bundle, joblib.load(sout/'short_memory_scalers_s2.joblib'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(sout/'selected_short_memory_s2.pt', (arrays[0].shape[2], arrays[1].shape[2], arrays[2].shape[2], arrays[3].shape[1]), device)
    scores = predict(model, arrays, device)

    pred = pd.DataFrame({
        'sample_id': bundle.frame['sample_id'].astype(str),
        'timestamp': pd.to_datetime(bundle.frame['timestamp'], utc=True),
        'direction': bundle.frame['direction'].astype(str),
        'setup': bundle.frame['setup'].astype(str),
        's1_score': np.nan,
        's2_score': scores.astype(float),
        's2_threshold': threshold,
        'ml_accept': scores >= threshold,
    })
    before = len(pred)
    pred = pred[pred.timestamp >= eval_not_before].copy()
    if (pred.timestamp < eval_not_before).any():
        raise RuntimeError('FUTURE_GUARD_BOUNDARY_FAILURE')
    log(t0, f'Predictions after strict boundary={len(pred):,}/{before:,}')

    primary, secondary, raw_eval = p3c.select_economic_segments(pred)
    segments = [s for s in [primary] + secondary if s.start >= eval_not_before]
    dropped = [s.segment_id for s in [primary] + secondary if s.start < eval_not_before]
    if not segments:
        raise RuntimeError('NO_STRICT_POST_CUTOFF_SEGMENTS')
    log(t0, f'Dropped boundary-overlap segments={dropped}')
    log(t0, 'Strict segments=' + ', '.join(s.segment_id for s in segments))

    guard = {
        'strict_no_future_guard': True,
        'final_fit_end_utc': cutoff.isoformat(),
        'evaluation_not_before_utc': eval_not_before.isoformat(),
        'segments_dropped_for_boundary_overlap': dropped,
        'segments_replayed': [s.segment_id for s in segments],
        'model_never_fit_on_timestamp_after': cutoff.isoformat(),
        'evaluation_rows_all_at_or_after': eval_not_before.isoformat(),
    }
    (out/'strict_replay_guard.json').write_text(json.dumps(guard, indent=2), encoding='utf-8')

    cfg = p3c.BotConfig()
    allow_position_override = os.environ.get('SHORT_MEMORY_RESEARCH_POSITION_CAP_OVERRIDE') == '1'
    if cfg.risk.max_open_positions != 3 and not allow_position_override:
        raise RuntimeError('MAX_POSITIONS_CONTRACT_BROKEN')
    if allow_position_override:
        log(t0, f'RESEARCH POSITION CAP OVERRIDE={cfg.risk.max_open_positions}')
    spec = p3c.load_symbol_spec(p3c.METADATA_PATH)
    m1, m5, m15 = p3c.load_csv(p3c.M1_PATH), p3c.load_csv(p3c.M5_PATH), p3c.load_csv(p3c.M15_PATH)

    rows=[]; trade_frames=[]
    for i, seg in enumerate(segments, 1):
        log(t0, f'Replay {seg.segment_id} {i}/{len(segments)}')
        run_m1 = m1[(m1['time'] >= seg.start-pd.Timedelta(days=10)) & (m1['time'] <= seg.end)].reset_index(drop=True)
        run_m5 = m5[(m5['time'] >= seg.start-pd.Timedelta(days=10)) & (m5['time'] <= seg.end)].reset_index(drop=True)
        run_m15 = m15[(m15['time'] >= seg.start-pd.Timedelta(days=15)) & (m15['time'] <= seg.end)].reset_index(drop=True)
        sp = pred[(pred.timestamp >= seg.start) & (pred.timestamp <= seg.end)].copy()
        engine = p3c.MLEconomicReplayEngine(cfg, spec, p3c.BacktestSettings(initial_balance=1000.0), sp, ml_enabled=True)
        result = engine.run(run_m1, run_m5, run_m15, seg.start, seg.end)
        decisions = pd.DataFrame(engine.candidate_decisions)
        row = p3c._variant_metrics(seg, f'STRICT_SHORT_{a.policy}_C3', result, decisions, threshold)
        rows.append(row)
        tf = p3c._trades_frame(result, sp, threshold, True)
        if not tf.empty:
            tf['segment_id']=seg.segment_id; trade_frames.append(tf)
        log(t0, f"{seg.segment_id}: trades={row['trades_opened']} PF={row['profit_factor']} exp={row['expectancy']:.4f} net={row['net_pnl']:.2f} DD={row['max_drawdown_pct']:.2f}%")

    metrics=pd.DataFrame(rows); metrics.to_csv(out/'policy_metrics_c3_strict.csv', index=False)
    trades=pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame(); trades.to_csv(out/'trades_c3_strict.csv', index=False)
    total=int(metrics.trades_opened.sum()); net=float(metrics.net_pnl.sum()); exp=float(np.average(metrics.expectancy, weights=np.maximum(metrics.trades_opened,1)))
    summary={'strict_no_future_guard':True,'policy':a.policy,'threshold':threshold,'final_fit_end_utc':cutoff.isoformat(),'evaluation_not_before_utc':eval_not_before.isoformat(),'total_trades':total,'total_net_pnl_usd':net,'weighted_expectancy_usd':exp,'segments':rows}
    (out/'c3_strict_summary.json').write_text(json.dumps(summary, indent=2, default=str), encoding='utf-8')
    print('\nFINAL STRICT C3 SUMMARY')
    print(metrics.to_string(index=False)); print(f'TOTAL_TRADES={total}'); print(f'TOTAL_NET_PNL_USD={net:.2f}'); print(f'WEIGHTED_EXPECTANCY_USD={exp:.4f}')
    print('STRICT_NO_FUTURE_GUARD=PASS')


if __name__ == '__main__':
    main()
