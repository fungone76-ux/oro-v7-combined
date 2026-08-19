"""Walk-forward trainer for TOP40_SHORT_MEMORY V1.

This is research-only. It does not connect to MT5 or send orders.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, brier_score_loss, log_loss,
    mean_absolute_error, mean_squared_error, roc_auc_score,
)

from .config import CONFIG, RESEARCH_VERSION
from .dataset import TensorBundle, load_phase3a, build_tensor_bundle
from .preprocess import preprocess_fold
from .training import Fold, freeze_oos_thresholds, set_deterministic, torch_model_class, write_threshold_freeze


def say(t0: float, msg: str) -> None:
    elapsed = time.perf_counter() - t0
    print(f"[{datetime.now().astimezone().strftime('%H:%M:%S')}] +{elapsed:8.1f}s | {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_frozen_folds(path: Path) -> list[Fold]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload['purge_minutes'] != CONFIG.purge_minutes or payload['embargo_minutes'] != CONFIG.embargo_minutes:
        raise ValueError('WALKFORWARD_GAP_CONTRACT_MISMATCH')
    return [Fold(
        fold=int(x['fold']),
        train_start=pd.Timestamp(x['train_start']), train_end=pd.Timestamp(x['train_end']),
        valid_start=pd.Timestamp(x['valid_start']), valid_end=pd.Timestamp(x['valid_end']),
    ) for x in payload['folds']]


def fold_indices(frame: pd.DataFrame, fold: Fold) -> tuple[np.ndarray, np.ndarray]:
    ts = pd.to_datetime(frame['timestamp'], utc=True)
    tr = np.flatnonzero((ts >= fold.train_start) & (ts <= fold.train_end))
    va = np.flatnonzero((ts >= fold.valid_start) & (ts <= fold.valid_end))
    if len(tr) == 0 or len(va) == 0:
        raise ValueError(f'EMPTY_FOLD_{fold.fold}: train={len(tr)} valid={len(va)}')
    return tr, va


def make_loader(m1, m5, m15, static, y, shuffle: bool):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(
        torch.tensor(m1, dtype=torch.float32), torch.tensor(m5, dtype=torch.float32),
        torch.tensor(m15, dtype=torch.float32), torch.tensor(static, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    gen = torch.Generator().manual_seed(CONFIG.seed)
    return DataLoader(ds, batch_size=CONFIG.batch_size, shuffle=shuffle, generator=gen)


def classification_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    n20 = max(1, int(len(y) * .2)); top = np.argsort(p)[-n20:]
    base = float(np.mean(y)); top_rate = float(np.mean(y[top]))
    return {
        'roc_auc': float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else math.nan,
        'pr_auc': float(average_precision_score(y, p)) if len(np.unique(y)) == 2 else math.nan,
        'log_loss': float(log_loss(y, np.clip(p, 1e-9, 1 - 1e-9), labels=[0, 1])),
        'brier': float(brier_score_loss(y, p)),
        'base_win_rate': base,
        'top20_lift': float(top_rate / base) if base else math.nan,
        'top20_rate': top_rate,
    }


def regression_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    n20 = max(1, int(len(y) * .2)); top = np.argsort(p)[-n20:]; bottom = np.argsort(p)[:n20]
    pearson = float(pd.Series(p).corr(pd.Series(y), method='pearson'))
    spearman = float(pd.Series(p).corr(pd.Series(y), method='spearman'))
    return {
        'mae': float(mean_absolute_error(y, p)),
        'rmse': float(math.sqrt(mean_squared_error(y, p))),
        'r': pearson, 'pearson': pearson, 'spearman': spearman,
        'top20_realized': float(np.mean(y[top])),
        'bottom20_realized': float(np.mean(y[bottom])),
        'overall_realized': float(np.mean(y)),
        'top20_improvement': float(np.mean(y[top]) - np.mean(y)),
    }


def evaluate(model, loader, task: str, device: str) -> tuple[np.ndarray, np.ndarray, float]:
    import torch
    loss_fn = torch.nn.BCEWithLogitsLoss() if task == 'S1' else torch.nn.SmoothL1Loss()
    model.eval(); ys=[]; ps=[]; losses=[]
    with torch.no_grad():
        for m1,m5,m15,st,y in loader:
            m1,m5,m15,st,y = m1.to(device),m5.to(device),m15.to(device),st.to(device),y.to(device)
            out=model(m1,m5,m15,st); losses.append(float(loss_fn(out,y).item()))
            pred=torch.sigmoid(out) if task=='S1' else out
            ys.append(y.cpu().numpy()); ps.append(pred.cpu().numpy())
    return np.concatenate(ys),np.concatenate(ps),float(np.mean(losses))


def train_fold(bundle: TensorBundle, fold: Fold, target: str, device: str, t0: float):
    import torch
    seed = CONFIG.seed + fold.fold + (100 if target == 'S2' else 0)
    set_deterministic(seed)
    tr,va=fold_indices(bundle.frame,fold)
    m1,m5,m15,static,scalers=preprocess_fold(bundle,tr)
    y=bundle.y_s1 if target=='S1' else bundle.y_s2
    train_loader=make_loader(m1[tr],m5[tr],m15[tr],static[tr],y[tr],True)
    valid_loader=make_loader(m1[va],m5[va],m15[va],static[va],y[va],False)
    Model=torch_model_class(); model=Model(m1.shape[2],m5.shape[2],m15.shape[2],static.shape[1],target).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=CONFIG.learning_rate)
    loss_fn=torch.nn.BCEWithLogitsLoss() if target=='S1' else torch.nn.SmoothL1Loss()
    best=-np.inf; best_state=None; bad=0; curves=[]
    tf=time.perf_counter()
    for epoch in range(1,CONFIG.max_epochs+1):
        model.train(); losses=[]
        for bm1,bm5,bm15,bst,by in train_loader:
            bm1,bm5,bm15,bst,by=bm1.to(device),bm5.to(device),bm15.to(device),bst.to(device),by.to(device)
            opt.zero_grad(set_to_none=True); out=model(bm1,bm5,bm15,bst); loss=loss_fn(out,by)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); losses.append(float(loss.item()))
        yv,pv,vloss=evaluate(model,valid_loader,target,device)
        metric=(classification_metrics(yv,pv)['roc_auc'] if target=='S1' else regression_metrics(yv,pv)['spearman'])
        curves.append({'target':target,'fold':fold.fold,'epoch':epoch,'train_loss':float(np.mean(losses)),'validation_loss':vloss,'validation_metric':metric})
        say(t0,f"FASE 4/10 | {target} fold {fold.fold}/4 epoch {epoch}/{CONFIG.max_epochs} | val={metric:.5f} best={max(best,metric):.5f} patience={bad}/{CONFIG.early_stopping_patience}")
        if np.isfinite(metric) and metric>best:
            best=metric; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=CONFIG.early_stopping_patience: break
    if best_state is None: raise RuntimeError(f'NO_VALID_MODEL_STATE {target} fold={fold.fold}')
    model.load_state_dict(best_state)
    yv,pv,_=evaluate(model,valid_loader,target,device)
    met=classification_metrics(yv,pv) if target=='S1' else regression_metrics(yv,pv)
    met.update({'target':target,'fold':fold.fold,'train_rows':len(tr),'valid_rows':len(va),'best_epoch':max(curves,key=lambda r:r['validation_metric'])['epoch'],'runtime_seconds':time.perf_counter()-tf})
    pred=pd.DataFrame({'row_index':va,'timestamp':bundle.frame.iloc[va]['timestamp'].to_numpy(),'target':target,'y':yv,'prediction':pv,'fold':fold.fold})
    return met,curves,model,scalers,pred


def fit_final(bundle: TensorBundle, target: str, device: str, epochs: int):
    """Fit frozen final model on all valid research rows, using only already-selected epoch count."""
    import torch
    set_deterministic(CONFIG.seed + (1000 if target=='S2' else 900))
    idx=np.arange(len(bundle.frame)); m1,m5,m15,static,scalers=preprocess_fold(bundle,idx)
    y=bundle.y_s1 if target=='S1' else bundle.y_s2
    loader=make_loader(m1,m5,m15,static,y,True)
    Model=torch_model_class(); model=Model(m1.shape[2],m5.shape[2],m15.shape[2],static.shape[1],target).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=CONFIG.learning_rate)
    loss_fn=torch.nn.BCEWithLogitsLoss() if target=='S1' else torch.nn.SmoothL1Loss()
    for _ in range(max(1,epochs)):
        model.train()
        for bm1,bm5,bm15,bst,by in loader:
            bm1,bm5,bm15,bst,by=bm1.to(device),bm5.to(device),bm15.to(device),bst.to(device),by.to(device)
            opt.zero_grad(set_to_none=True); loss=loss_fn(model(bm1,bm5,bm15,bst),by); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    return model,scalers


def run(market_state: Path, candidates: Path, out: Path, split_path: Path=Path('config/walkforward_splits.json')) -> dict:
    import torch
    t0=time.perf_counter(); out.mkdir(parents=True,exist_ok=True)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    say(t0,f"=== TOP40 SHORT_MEMORY V1 START | device={device} ===")
    say(t0,"FASE 1/10 | audit input Phase 3A")
    market,cand=load_phase3a(market_state,candidates)
    say(t0,f"Phase3A market={len(market):,} candidates={len(cand):,}")
    say(t0,"FASE 2/10 | build causal tensors M1=8 M5=5 M15=3")
    bundle=build_tensor_bundle(market,cand)
    say(t0,f"Valid short-memory samples={len(bundle.frame):,}")
    folds=load_frozen_folds(split_path)
    say(t0,"FASE 3/10 | frozen original walk-forward splits loaded")
    metrics={'S1':[],'S2':[]}; curves=[]; predictions=[]; best_epochs={}
    for target in ('S1','S2'):
        for fold in folds:
            met,cv,_model,_scalers,pred=train_fold(bundle,fold,target,device,t0)
            metrics[target].append(met); curves.extend(cv); predictions.append(pred)
        best_epochs[target]=int(round(np.median([m['best_epoch'] for m in metrics[target]])))
    say(t0,"FASE 5/10 | aggregate strictly OOS predictions")
    pred=pd.concat(predictions,ignore_index=True).sort_values(['target','timestamp'])
    s2=pred[pred.target=='S2'].copy()
    thresholds=freeze_oos_thresholds(s2.prediction.to_numpy())
    pd.DataFrame(metrics['S1']).to_csv(out/'s1_fold_metrics.csv',index=False)
    pd.DataFrame(metrics['S2']).to_csv(out/'s2_fold_metrics.csv',index=False)
    pd.DataFrame(curves).to_csv(out/'training_curves.csv',index=False)
    pred.to_csv(out/'oos_predictions.csv.gz',index=False,compression='gzip')
    pd.DataFrame({'prediction':s2.prediction}).describe(percentiles=[.1,.2,.4,.6,.8,.9]).to_csv(out/'prediction_distribution.csv')
    say(t0,f"FASE 6/10 | freeze thresholds from OOS only: {thresholds}")
    say(t0,"FASE 7/10 | fit final frozen S1/S2 using median OOS-selected epoch counts")
    hashes={}
    for target in ('S1','S2'):
        model,scalers=fit_final(bundle,target,device,best_epochs[target])
        model_path=out/f'selected_short_memory_{target.lower()}.pt'; scaler_path=out/f'short_memory_scalers_{target.lower()}.joblib'
        torch.save(model.state_dict(),model_path); joblib.dump(scalers,scaler_path)
        hashes[f'{target.lower()}_model_sha256']=sha256(model_path); hashes[f'{target.lower()}_scaler_sha256']=sha256(scaler_path)
    write_threshold_freeze(out/'short_memory_threshold_freeze_manifest.json',thresholds,hashes)
    say(t0,"FASE 8/10 | model/scaler hashes frozen; economic metrics still CLOSED")
    summary={
        'research_version':RESEARCH_VERSION,'sequence_lengths':{'m1':8,'m5':5,'m15':3},
        'samples':len(bundle.frame),'device':device,'thresholds':thresholds,
        'median_best_epochs':best_epochs,'artifact_hashes':hashes,
        'economic_metrics_opened':False,'runtime_seconds':time.perf_counter()-t0,
        'next_step':'economic replay C3 + fixed 0.01 using frozen OOS thresholds',
    }
    (out/'short_memory_training_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    say(t0,"FASE 9/10 | training reports written")
    say(t0,"FASE 10/10 | COMPLETE — economic replay intentionally not opened by trainer")
    return summary
