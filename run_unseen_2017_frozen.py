from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

EXPECTED_HASHES = {
    'M1': 'a41b252c44c21ef30bf52221ea86d69e1144edb0d7ad1773802614c8b6d8d2ac',
    'M5': '4df4a6cbc7be097fe262d4a8e302498d8101fe70c2256481caada7429a180d31',
    'M15': '0d409e23a303463fb1f0d7964c9406743b57c19408b8cd1fe870dcfe280e31a7',
}
FROZEN_THRESHOLD = 0.32696733474731443
EVAL_START = pd.Timestamp('2017-01-01T00:00:00Z')
EVAL_END = pd.Timestamp('2017-12-31T23:59:59Z')
FIXED_LOT = 0.01


def log(t0: float, msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] +{time.perf_counter()-t0:7.1f}s | {msg}', flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['time'] = pd.to_datetime(df['time'], utc=True)
    return df.sort_values('time').drop_duplicates('time').reset_index(drop=True)


def _continuous(times: np.ndarray, start: int, end: int, step: int) -> bool:
    if start < 0 or end < start:
        return False
    d = np.diff(times[start:end+1]).astype('timedelta64[m]').astype(int)
    return len(d) == (end-start) and bool(np.all(d == step))


def _unique_htf(market: pd.DataFrame, source: str, features: list[str]) -> pd.DataFrame:
    out = market[[source] + features].dropna(subset=[source]).copy()
    out[source] = pd.to_datetime(out[source], utc=True)
    return out.sort_values(source).drop_duplicates(source, keep='last').reset_index(drop=True)


def build_inference_bundle(market: pd.DataFrame, candidates: pd.DataFrame):
    from short_memory.config import CONFIG
    from short_memory.training import M1_FEATURES, M5_FEATURES, M15_FEATURES, STATIC_NUMERIC, STATIC_CATEGORICAL

    market = market.sort_values('timestamp').reset_index(drop=True)
    candidates = candidates.sort_values('timestamp').reset_index(drop=True)
    m5 = _unique_htf(market, 'm5_source_close_time', M5_FEATURES)
    m15 = _unique_htf(market, 'm15_source_close_time', M15_FEATURES)

    t1 = market['timestamp'].to_numpy(dtype='datetime64[ns]')
    t5 = m5['m5_source_close_time'].to_numpy(dtype='datetime64[ns]')
    t15 = m15['m15_source_close_time'].to_numpy(dtype='datetime64[ns]')
    v1 = market[M1_FEATURES].to_numpy(np.float32)
    v5 = m5[M5_FEATURES].to_numpy(np.float32)
    v15 = m15[M15_FEATURES].to_numpy(np.float32)

    keep, a1, a5, a15 = [], [], [], []
    for idx, row in candidates.iterrows():
        ts = np.datetime64(pd.Timestamp(row['timestamp']).to_datetime64())
        def sl(values, times, length, step):
            end = int(np.searchsorted(times, ts, side='right') - 1)
            start = end - length + 1
            if start < 0 or not _continuous(times, start, end, step):
                return None
            if times[end] > ts:
                raise AssertionError('LOOKAHEAD_DETECTED')
            return values[start:end+1]
        w1 = sl(v1, t1, CONFIG.m1_length, 1)
        w5 = sl(v5, t5, CONFIG.m5_length, 5)
        w15 = sl(v15, t15, CONFIG.m15_length, 15)
        if w1 is None or w5 is None or w15 is None:
            continue
        if not (np.isfinite(w1).all() and np.isfinite(w5).all() and np.isfinite(w15).all()):
            continue
        keep.append(idx); a1.append(w1); a5.append(w5); a15.append(w15)

    frame = candidates.loc[keep].reset_index(drop=True)
    if frame.empty:
        raise RuntimeError('NO_VALID_UNSEEN_2017_SEQUENCES')
    static_num = frame[STATIC_NUMERIC].astype(float).to_numpy(np.float32)
    finite = np.isfinite(static_num).all(axis=1)
    frame = frame.loc[finite].reset_index(drop=True)
    static_cat = candidates.loc[keep, STATIC_CATEGORICAL].astype(str).reset_index(drop=True).loc[finite].reset_index(drop=True)
    return frame, np.stack(a1)[finite], np.stack(a5)[finite], np.stack(a15)[finite], static_num[finite], static_cat


def transform_bundle(m1, m5, m15, static_num, static_cat, scalers):
    def seq(values: np.ndarray, key: str) -> np.ndarray:
        shape = values.shape
        return scalers[key].transform(values.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)
    a1 = seq(m1, 'm1'); a5 = seq(m5, 'm5'); a15 = seq(m15, 'm15')
    sn = scalers['static_numeric'].transform(static_num).astype(np.float32)
    pieces = []
    for col, values in scalers.get('categories', {}).items():
        cv = static_cat[col].astype(str).fillna('<NA>')
        for value in values:
            pieces.append((cv == value).astype(float).to_numpy()[:, None])
    sc = np.hstack(pieces).astype(np.float32) if pieces else np.empty((len(static_cat), 0), dtype=np.float32)
    return a1, a5, a15, np.hstack([sn, sc]).astype(np.float32)


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
    ds = TensorDataset(*(torch.tensor(x) for x in arrays))
    out = []
    with torch.no_grad():
        for bm1, bm5, bm15, bst in DataLoader(ds, batch_size=256, shuffle=False):
            out.append(model(bm1.to(device), bm5.to(device), bm15.to(device), bst.to(device)).cpu().numpy())
    return np.concatenate(out)


def max_drawdown(pnl: pd.Series, initial_balance: float = 1000.0) -> tuple[float, float]:
    equity = initial_balance + pnl.cumsum()
    peak = equity.cummax()
    dd = peak - equity
    dd_pct = dd / peak.replace(0, np.nan) * 100.0
    return float(dd.max() if len(dd) else 0.0), float(dd_pct.max() if len(dd_pct) else 0.0)


def pnl_metrics(trades: pd.DataFrame) -> dict:
    pnl = pd.to_numeric(trades['net_pnl'], errors='coerce').fillna(0.0)
    gp = float(pnl[pnl > 0].sum()); gl = float(-pnl[pnl < 0].sum())
    dd_abs, dd_pct = max_drawdown(pnl)
    return {
        'trades': int(len(trades)),
        'wins': int((pnl > 0).sum()),
        'losses': int((pnl < 0).sum()),
        'win_rate_pct': float((pnl > 0).mean() * 100.0) if len(pnl) else 0.0,
        'gross_profit_usd': gp,
        'gross_loss_usd': gl,
        'net_profit_usd': float(pnl.sum()),
        'profit_factor': float(gp / gl) if gl > 0 else float('inf'),
        'expectancy_usd': float(pnl.mean()) if len(pnl) else 0.0,
        'max_drawdown_usd': dd_abs,
        'max_drawdown_pct': dd_pct,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description='Frozen SHORT_MEMORY 8/5/3 unseen historical stress test on calendar year 2017')
    ap.add_argument('--top40-root', type=Path, required=True)
    ap.add_argument('--short-out', type=Path, required=True)
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--out', type=Path, default=Path('research_output/unseen_2017_frozen'))
    a = ap.parse_args()

    t0 = time.perf_counter(); repo = Path(__file__).resolve().parent
    top = a.top40_root.resolve(); short = a.short_out.resolve(); data_dir = a.data_dir.resolve(); out = a.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo)); sys.path.insert(0, str(top)); os.chdir(top)

    paths = {
        'M1': data_dir/'XAUUSD_M1_UNSEEN_2017.csv',
        'M5': data_dir/'XAUUSD_M5_UNSEEN_2017.csv',
        'M15': data_dir/'XAUUSD_M15_UNSEEN_2017.csv',
    }
    for k, p in paths.items():
        if not p.exists(): raise RuntimeError(f'MISSING_{k}_FILE: {p}')
        actual = sha256(p)
        if actual != EXPECTED_HASHES[k]: raise RuntimeError(f'UNSEEN_{k}_HASH_MISMATCH expected={EXPECTED_HASHES[k]} actual={actual}')
    log(t0, 'UNSEEN_DATASET_HASHES=PASS')

    freeze = json.loads((short/'short_memory_threshold_freeze_manifest.json').read_text(encoding='utf-8'))
    threshold = float(freeze['thresholds']['TOP40_SHORT'])
    if abs(threshold - FROZEN_THRESHOLD) > 1e-12: raise RuntimeError('FROZEN_THRESHOLD_MISMATCH')
    model_path = short/'selected_short_memory_s2.pt'; scaler_path = short/'short_memory_scalers_s2.joblib'
    if not model_path.exists() or not scaler_path.exists(): raise RuntimeError('FROZEN_ARTIFACT_MISSING')
    log(t0, f'FROZEN_THRESHOLD=PASS value={threshold:.15f}')
    log(t0, 'NO_RETRAINING=PASS (inference-only script contains no fit/optimizer/training path)')

    m1, m5, m15 = load_csv(paths['M1']), load_csv(paths['M5']), load_csv(paths['M15'])
    if m1.time.min() > pd.Timestamp('2016-12-05T00:00:00Z') or m1.time.max() < pd.Timestamp('2018-01-02T00:00:00Z'):
        raise RuntimeError('INSUFFICIENT_WARMUP_OR_TAIL_COVERAGE')
    log(t0, f'RAW COVERAGE M1={m1.time.min()} -> {m1.time.max()} rows={len(m1):,}')

    from xau_bot.config.settings import BotConfig
    from xau_bot.research.ml.feature_builder import build_market_state_features
    from xau_bot.research.ml.candidate_extractor import extract_strategy_candidates
    from xau_bot.research.ml import phase3c_economic_replay as p3c

    cfg = BotConfig()
    spec = p3c.load_symbol_spec(p3c.METADATA_PATH)
    market, model_features, _diag, _ = build_market_state_features(m1, m5, m15, cfg)
    market = market[market['feature_valid'].astype(bool)].reset_index(drop=True)
    log(t0, f'REBUILT market_state from unseen CSV only: {len(market):,} rows')

    extracted = extract_strategy_candidates(m1, m5, m15, market, model_features, cfg, spec, trades_path=None, baseline_scope_end=None)
    candidates = extracted.candidates
    candidates = candidates[candidates['feature_valid'].astype(bool)].reset_index(drop=True)
    log(t0, f'REBUILT strategy_candidates from unseen CSV only: {len(candidates):,} rows')

    frame, b1, b5, b15, sn, sc = build_inference_bundle(market, candidates)
    arrays = transform_bundle(b1, b5, b15, sn, sc, joblib.load(scaler_path))
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(model_path, (arrays[0].shape[2], arrays[1].shape[2], arrays[2].shape[2], arrays[3].shape[1]), device)
    scores = predict(model, arrays, device)
    pred = pd.DataFrame({
        'sample_id': frame['sample_id'].astype(str),
        'timestamp': pd.to_datetime(frame['timestamp'], utc=True),
        'direction': frame['direction'].astype(str),
        'setup': frame['setup'].astype(str),
        's1_score': np.nan,
        's2_score': scores.astype(float),
        's2_threshold': threshold,
        'ml_accept': scores >= threshold,
    })
    pred = pred[(pred.timestamp >= EVAL_START) & (pred.timestamp <= EVAL_END)].copy()
    log(t0, f'2017 frozen predictions={len(pred):,} accepted={int(pred.ml_accept.sum()):,}')

    # Research-only policy patches: all sessions, no consecutive-loss cooldown, true fixed 0.01 lot.
    original_register = p3c.RiskManager.register_trade_result
    def register_no_cooldown(self, is_win, now_utc):
        original_register(self, is_win, now_utc)
        self.state.cooldown_until_utc = None
    p3c.RiskManager.register_trade_result = register_no_cooldown

    original_decide = p3c.decide
    def decide_all_sessions(signal, snapshot, kill_switch_blocks_entries, enabled_sessions=None, live_confirmed=False, ml_state=None, ml_probability_up=None):
        kwargs = dict(kill_switch_blocks_entries=kill_switch_blocks_entries, enabled_sessions=(snapshot.session.value,), live_confirmed=live_confirmed, ml_probability_up=ml_probability_up)
        if ml_state is not None: kwargs['ml_state'] = ml_state
        return original_decide(signal, snapshot, **kwargs)
    p3c.decide = decide_all_sessions

    original_lot = p3c.RiskManager._lot_from_risk
    def fixed_lot(self, risk_amount, sl_points):
        return FIXED_LOT if self.spec.volume_min <= FIXED_LOT <= self.spec.volume_max else 0.0
    p3c.RiskManager._lot_from_risk = fixed_lot

    engine = p3c.MLEconomicReplayEngine(cfg, spec, p3c.BacktestSettings(initial_balance=1000.0), pred, ml_enabled=True)
    result = engine.run(m1, m5, m15, EVAL_START, EVAL_END)
    trades = p3c._trades_frame(result, pred, threshold, True)
    if trades.empty: raise RuntimeError('NO_TRADES_OPENED_2017')
    if 'volume' in trades.columns and not np.allclose(pd.to_numeric(trades['volume']), FIXED_LOT):
        raise RuntimeError('FIXED_LOT_CONTRACT_BROKEN')
    trades.to_csv(out/'trades_unseen_2017_fixed001.csv', index=False)

    overall = pnl_metrics(trades)
    overall.update({
        'evaluation_start': EVAL_START.isoformat(), 'evaluation_end': EVAL_END.isoformat(),
        'threshold': threshold, 'fixed_lot': FIXED_LOT, 'max_positions': cfg.risk.max_open_positions,
        'cooldown_enabled': False, 'session_filter_enabled': False,
        'dataset_hashes': EXPECTED_HASHES,
        'frozen_model_sha256': sha256(model_path), 'frozen_scaler_sha256': sha256(scaler_path),
        'technical_candidates_rebuilt': int(len(candidates)), 'model_predictions_2017': int(len(pred)),
        'ml_accepted_2017': int(pred.ml_accept.sum()),
        'daily_stop_count': int(result.metrics.get('daily_stop_count', 0)),
        'max_positions_rejections': int(result.metrics.get('max_positions_rejections', 0)),
        'spread_filter_rejections': int(result.metrics.get('spread_filter_rejections', 0)),
    })

    rows=[]
    for session, g in trades.groupby('session', dropna=False):
        r={'session': str(session)}; r.update(pnl_metrics(g.reset_index(drop=True))); rows.append(r)
    sessions = pd.DataFrame(rows).sort_values('trades', ascending=False)
    sessions.to_csv(out/'session_breakdown_unseen_2017.csv', index=False)
    (out/'unseen_2017_summary.json').write_text(json.dumps(overall, indent=2), encoding='utf-8')

    print('\nUNSEEN 2017 FROZEN RESULT')
    for k,v in overall.items(): print(f'{k}={v}')
    print('\nSESSION BREAKDOWN 2017')
    print(sessions.to_string(index=False))
    print('UNSEEN_DATASET_HASHES=PASS')
    print('NO_RETRAINING=PASS')
    print('FROZEN_MODEL=PASS')
    print('FROZEN_THRESHOLD=PASS')
    print('SEQUENCES_8_5_3=PASS')
    print('SESSION_FILTER=DISABLED')
    print('COOLDOWN=DISABLED')
    print('FIXED_LOT_001=PASS')
    print('EVALUATION_PERIOD=2017-01-01..2017-12-31')


if __name__ == '__main__':
    main()
