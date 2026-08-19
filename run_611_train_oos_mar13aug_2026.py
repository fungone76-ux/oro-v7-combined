from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from short_memory_611.config import CONFIG, MODEL_ARCHITECTURE, RESEARCH_VERSION
from short_memory_611.dataset import build_tensor_bundle

ROOT = Path(r"D:\ORO_611_MAR13AUG_2026")
OUT = ROOT / "model_611_oos"


def log(t0: float, msg: str) -> None:
    print(f"+{time.perf_counter()-t0:7.1f}s | {msg}", flush=True)


def set_seed(seed: int) -> None:
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def scale_seq(x: np.ndarray, tr: np.ndarray):
    flat = x[tr].reshape(-1, x.shape[-1])
    med = np.nanmedian(flat, axis=0)
    sc = StandardScaler().fit(np.where(np.isnan(flat), med, flat))
    all_flat = x.reshape(-1, x.shape[-1])
    out = sc.transform(np.where(np.isnan(all_flat), med, all_flat)).reshape(x.shape).astype(np.float32)
    return sc, out


def preprocess(bundle, tr):
    s1, m1 = scale_seq(bundle.m1, tr)
    s5, m5 = scale_seq(bundle.m5, tr)
    s15, m15 = scale_seq(bundle.m15, tr)
    sn = bundle.static_num
    med = np.nanmedian(sn[tr], axis=0)
    ss = StandardScaler().fit(np.where(np.isnan(sn[tr]), med, sn[tr]))
    st = ss.transform(np.where(np.isnan(sn), med, sn)).astype(np.float32)
    return (m1, m5, m15, st), {"m1": s1, "m5": s5, "m15": s15, "static": ss}


def model_class():
    import torch
    class Model(torch.nn.Module):
        def __init__(self, d1, d5, d15, ds):
            super().__init__()
            self.m1 = torch.nn.LSTM(d1, MODEL_ARCHITECTURE["m1_lstm_hidden"], batch_first=True)
            self.m5 = torch.nn.LSTM(d5, MODEL_ARCHITECTURE["m5_lstm_hidden"], batch_first=True)
            self.m15 = torch.nn.LSTM(d15, MODEL_ARCHITECTURE["m15_lstm_hidden"], batch_first=True)
            merged = MODEL_ARCHITECTURE["m1_lstm_hidden"] + MODEL_ARCHITECTURE["m5_lstm_hidden"] + MODEL_ARCHITECTURE["m15_lstm_hidden"] + ds
            self.head = torch.nn.Sequential(
                torch.nn.Linear(merged, MODEL_ARCHITECTURE["dense"]),
                torch.nn.ReLU(), torch.nn.Dropout(MODEL_ARCHITECTURE["dropout"]),
                torch.nn.Linear(MODEL_ARCHITECTURE["dense"], 1),
            )
        def forward(self, a, b, c, s):
            _, (h1, _) = self.m1(a); _, (h5, _) = self.m5(b); _, (h15, _) = self.m15(c)
            return self.head(torch.cat([h1[-1], h5[-1], h15[-1], s], dim=1)).squeeze(1)
    return Model


def loader(arrays, y, idx, shuffle):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    ds = TensorDataset(*(torch.tensor(a[idx], dtype=torch.float32) for a in arrays), torch.tensor(y[idx], dtype=torch.float32))
    return DataLoader(ds, batch_size=CONFIG.batch_size, shuffle=shuffle, generator=torch.Generator().manual_seed(CONFIG.seed))


def evaluate(model, ld, device):
    import torch
    ys=[]; ps=[]; model.eval()
    with torch.no_grad():
        for a,b,c,s,y in ld:
            p=model(a.to(device),b.to(device),c.to(device),s.to(device))
            ys.append(y.numpy()); ps.append(p.cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def spearman(y,p):
    return float(pd.Series(y).corr(pd.Series(p), method="spearman"))


def make_folds(frame: pd.DataFrame):
    ts = pd.to_datetime(frame["timestamp"], utc=True)
    blocks = [
        ("2026-04", pd.Timestamp("2026-04-01T00:00:00Z"), pd.Timestamp("2026-05-01T00:00:00Z")),
        ("2026-05", pd.Timestamp("2026-05-01T00:00:00Z"), pd.Timestamp("2026-06-01T00:00:00Z")),
        ("2026-06", pd.Timestamp("2026-06-01T00:00:00Z"), pd.Timestamp("2026-07-01T00:00:00Z")),
        ("2026-07", pd.Timestamp("2026-07-01T00:00:00Z"), pd.Timestamp("2026-08-01T00:00:00Z")),
        ("2026-08-01_13", pd.Timestamp("2026-08-01T00:00:00Z"), pd.Timestamp("2026-08-14T00:00:00Z")),
    ]
    gap = pd.Timedelta(minutes=CONFIG.purge_minutes + CONFIG.embargo_minutes)
    folds=[]
    for i,(label,a,b) in enumerate(blocks,1):
        tr=np.flatnonzero(ts < (a-gap))
        va=np.flatnonzero((ts >= a) & (ts < b))
        if len(tr)==0 or len(va)==0: raise RuntimeError(f"EMPTY_FOLD_{label}")
        folds.append((i,label,tr,va))
    return folds


def main():
    t0=time.perf_counter(); OUT.mkdir(parents=True, exist_ok=True)
    bundle=build_tensor_bundle(ROOT)
    print(f"6/1/1 valid samples={len(bundle.frame):,}")
    folds=make_folds(bundle.frame)
    rows=[]
    for fid,label,tr,va in folds:
        rows.append({"fold":fid,"block":label,"train_rows":len(tr),"valid_rows":len(va),"train_last":pd.to_datetime(bundle.frame.iloc[tr[-1]]["timestamp"],utc=True).isoformat(),"valid_first":pd.to_datetime(bundle.frame.iloc[va[0]]["timestamp"],utc=True).isoformat(),"valid_last":pd.to_datetime(bundle.frame.iloc[va[-1]]["timestamp"],utc=True).isoformat()})
    print("OOS_CONTRACT=MARCH_SEED_THEN_APR_MAY_JUN_JUL_AUG1_13")
    print(pd.DataFrame(rows).to_string(index=False))

    import torch
    device="cuda" if torch.cuda.is_available() else "cpu"; print(f"DEVICE={device}")
    preds=[]; metrics=[]; best_epochs=[]
    Model=model_class()
    for fid,label,tr,va in folds:
        print(f"\n=== OOS BLOCK {label} ===", flush=True)
        set_seed(CONFIG.seed+fid)
        arrays, scalers=preprocess(bundle,tr)
        model=Model(arrays[0].shape[2],arrays[1].shape[2],arrays[2].shape[2],arrays[3].shape[1]).to(device)
        opt=torch.optim.Adam(model.parameters(),lr=CONFIG.learning_rate); loss_fn=torch.nn.SmoothL1Loss()
        tl=loader(arrays,bundle.y,tr,True); vl=loader(arrays,bundle.y,va,False)
        best=-np.inf; best_state=None; best_epoch=0; bad=0
        for ep in range(1,CONFIG.max_epochs+1):
            model.train()
            for a,b,c,s,y in tl:
                a,b,c,s,y=a.to(device),b.to(device),c.to(device),s.to(device),y.to(device)
                opt.zero_grad(set_to_none=True); loss=loss_fn(model(a,b,c,s),y); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
            yv,pv=evaluate(model,vl,device); sp=spearman(yv,pv)
            log(t0,f"S2 6/1/1 fold {fid}/5 epoch {ep} spearman={sp:.5f} best={max(best,sp):.5f}")
            if np.isfinite(sp) and sp>best:
                best=sp; best_epoch=ep; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; bad=0
            else:
                bad+=1
                if bad>=CONFIG.early_stopping_patience: break
        if best_state is None: raise RuntimeError(f"NO_MODEL_{label}")
        model.load_state_dict(best_state); yv,pv=evaluate(model,vl,device)
        f=bundle.frame.iloc[va].copy().reset_index(drop=True)
        f["y_s2"]=yv; f["s2_score"]=pv; f["fold"]=fid; f["validation_block"]=label
        preds.append(f)
        metrics.append({"fold":fid,"block":label,"train_rows":len(tr),"valid_rows":len(va),"best_epoch":best_epoch,"spearman":spearman(yv,pv)})
        best_epochs.append(best_epoch)

    pred=pd.concat(preds,ignore_index=True).sort_values("timestamp")
    pred.to_parquet(OUT/"oos_predictions_611.parquet",index=False)
    pd.DataFrame(metrics).to_csv(OUT/"oos_fold_metrics_611.csv",index=False)
    med=int(round(float(np.median(best_epochs))))
    summary={"research_version":RESEARCH_VERSION,"samples":len(bundle.frame),"oos_predictions":len(pred),"oos_blocks":[x[1] for x in folds],"march_role":"TRAINING_SEED_ONLY","median_best_epochs":med,"threshold_selected":False,"aug14_used":False}
    (OUT/"TRAINING_611_OOS_SUMMARY.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("\nTRAIN_611_MAR13AUG_OOS=PASS")
    print(f"VALID_SAMPLES={len(bundle.frame):,}")
    print(f"OOS_PREDICTIONS={len(pred):,}")
    print(f"MEDIAN_BEST_EPOCHS={med}")
    print("AUG14_USED_FOR_TRAINING=NO")
    print("AUG14_USED_FOR_VALIDATION=NO")
    print("THRESHOLD_SELECTED=NO")
    print(f"OUTPUT={OUT}")

if __name__=="__main__": main()
