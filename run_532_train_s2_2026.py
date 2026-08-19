from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler

from short_memory_532.config import CONFIG, MODEL_ARCHITECTURE, RESEARCH_VERSION
from short_memory_532.dataset import build_tensor_bundle, TensorBundle532
from short_memory.training import STATIC_CATEGORICAL


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def log(t0: float, msg: str) -> None:
    print(f"+{time.perf_counter()-t0:7.1f}s | {msg}", flush=True)


def fit_seq(values: np.ndarray, train_idx: np.ndarray) -> tuple[StandardScaler, np.ndarray]:
    train_flat = values[train_idx].reshape(-1, values.shape[-1])
    med = np.nanmedian(train_flat, axis=0)
    scaler = StandardScaler().fit(np.where(np.isnan(train_flat), med, train_flat))
    flat = values.reshape(-1, values.shape[-1])
    return scaler, scaler.transform(np.where(np.isnan(flat), med, flat)).reshape(values.shape).astype(np.float32)


def preprocess(bundle: TensorBundle532, train_idx: np.ndarray):
    s1, m1 = fit_seq(bundle.m1, train_idx)
    s5, m5 = fit_seq(bundle.m5, train_idx)
    s15, m15 = fit_seq(bundle.m15, train_idx)
    tr = bundle.static_num[train_idx]
    med = np.nanmedian(tr, axis=0)
    ss = StandardScaler().fit(np.where(np.isnan(tr), med, tr))
    sn = ss.transform(np.where(np.isnan(bundle.static_num), med, bundle.static_num)).astype(np.float32)
    pieces=[]; vocab={}
    for col in STATIC_CATEGORICAL:
        cats=sorted(bundle.static_cat.iloc[train_idx][col].astype(str).fillna("<NA>").unique().tolist())
        vocab[col]=cats; allc=bundle.static_cat[col].astype(str).fillna("<NA>")
        for cat in cats:
            pieces.append((allc == cat).astype(np.float32).to_numpy()[:,None])
    sc=np.hstack(pieces).astype(np.float32) if pieces else np.empty((len(bundle.frame),0),np.float32)
    return m1,m5,m15,np.hstack([sn,sc]).astype(np.float32),{"m1":s1,"m5":s5,"m15":s15,"static_numeric":ss,"categories":vocab}


def model_class():
    import torch
    class Model(torch.nn.Module):
        def __init__(self,d1,d5,d15,ds):
            super().__init__()
            self.m1=torch.nn.LSTM(d1,MODEL_ARCHITECTURE["m1_lstm_hidden"],batch_first=True)
            self.m5=torch.nn.LSTM(d5,MODEL_ARCHITECTURE["m5_lstm_hidden"],batch_first=True)
            self.m15=torch.nn.LSTM(d15,MODEL_ARCHITECTURE["m15_lstm_hidden"],batch_first=True)
            merged=MODEL_ARCHITECTURE["m1_lstm_hidden"]+MODEL_ARCHITECTURE["m5_lstm_hidden"]+MODEL_ARCHITECTURE["m15_lstm_hidden"]+ds
            self.head=torch.nn.Sequential(torch.nn.Linear(merged,MODEL_ARCHITECTURE["dense"]),torch.nn.ReLU(),torch.nn.Dropout(MODEL_ARCHITECTURE["dropout"]),torch.nn.Linear(MODEL_ARCHITECTURE["dense"],1))
        def forward(self,a,b,c,s):
            _,(h1,_)=self.m1(a); _,(h5,_)=self.m5(b); _,(h15,_)=self.m15(c)
            return self.head(torch.cat([h1[-1],h5[-1],h15[-1],s],dim=1)).squeeze(1)
    return Model


def set_seed(seed:int):
    import random, torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def metrics(y,p):
    return {
        "mae":float(mean_absolute_error(y,p)),
        "rmse":float(math.sqrt(mean_squared_error(y,p))),
        "pearson":float(pd.Series(p).corr(pd.Series(y),method="pearson")),
        "spearman":float(pd.Series(p).corr(pd.Series(y),method="spearman")),
    }


def make_folds(frame: pd.DataFrame):
    ts=pd.to_datetime(frame["timestamp"],utc=True).sort_values().reset_index(drop=True)
    n=len(ts); edges=np.linspace(0,n,6,dtype=int); folds=[]
    gap=pd.Timedelta(minutes=CONFIG.purge_minutes+CONFIG.embargo_minutes)
    for i in range(4):
        train_end=ts.iloc[edges[i+1]-1]
        valid_start=ts.iloc[edges[i+1]] + gap
        valid_end=ts.iloc[edges[i+2]-1]
        tr=np.flatnonzero(pd.to_datetime(frame["timestamp"],utc=True) <= train_end)
        va=np.flatnonzero((pd.to_datetime(frame["timestamp"],utc=True) >= valid_start)&(pd.to_datetime(frame["timestamp"],utc=True) <= valid_end))
        if len(tr)==0 or len(va)==0: raise RuntimeError(f"EMPTY_532_FOLD_{i+1}")
        folds.append((i+1,tr,va,train_end,valid_start,valid_end))
    return folds


def loader(arrays,y,idx,shuffle):
    import torch
    from torch.utils.data import TensorDataset,DataLoader
    ds=TensorDataset(*(torch.tensor(x[idx],dtype=torch.float32) for x in arrays),torch.tensor(y[idx],dtype=torch.float32))
    return DataLoader(ds,batch_size=CONFIG.batch_size,shuffle=shuffle,generator=torch.Generator().manual_seed(CONFIG.seed))


def evaluate(model,ld,device):
    import torch
    ys=[]; ps=[]; model.eval()
    with torch.no_grad():
        for a,b,c,s,y in ld:
            p=model(a.to(device),b.to(device),c.to(device),s.to(device))
            ys.append(y.numpy()); ps.append(p.cpu().numpy())
    return np.concatenate(ys),np.concatenate(ps)


def train_fold(bundle, fold, device, t0):
    import torch
    fid,tr,va,*_=fold; set_seed(CONFIG.seed+100+fid)
    m1,m5,m15,st,scalers=preprocess(bundle,tr); arrays=(m1,m5,m15,st)
    tl=loader(arrays,bundle.y_s2,tr,True); vl=loader(arrays,bundle.y_s2,va,False)
    Model=model_class(); model=Model(m1.shape[2],m5.shape[2],m15.shape[2],st.shape[1]).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=CONFIG.learning_rate); loss_fn=torch.nn.SmoothL1Loss()
    best=-np.inf; best_state=None; best_epoch=0; bad=0
    for ep in range(1,CONFIG.max_epochs+1):
        model.train()
        for a,b,c,s,y in tl:
            a,b,c,s,y=a.to(device),b.to(device),c.to(device),s.to(device),y.to(device)
            opt.zero_grad(set_to_none=True); loss=loss_fn(model(a,b,c,s),y); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
        yv,pv=evaluate(model,vl,device); sp=metrics(yv,pv)["spearman"]
        log(t0,f"S2 5/3/2 fold {fid}/4 epoch {ep} spearman={sp:.5f} best={max(best,sp):.5f}")
        if np.isfinite(sp) and sp>best:
            best=sp; best_epoch=ep; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; bad=0
        else:
            bad+=1
            if bad>=CONFIG.early_stopping_patience: break
    if best_state is None: raise RuntimeError(f"NO_MODEL_FOLD_{fid}")
    model.load_state_dict(best_state); yv,pv=evaluate(model,vl,device)
    pred=pd.DataFrame({"row_index":va,"timestamp":bundle.frame.iloc[va]["timestamp"].to_numpy(),"y_s2":yv,"s2_score":pv,"fold":fid})
    met=metrics(yv,pv)|{"fold":fid,"train_rows":len(tr),"valid_rows":len(va),"best_epoch":best_epoch}
    return pred,met,best_epoch


def fit_final(bundle,device,epochs):
    import torch
    idx=np.arange(len(bundle.frame)); set_seed(CONFIG.seed+1000)
    m1,m5,m15,st,scalers=preprocess(bundle,idx); arrays=(m1,m5,m15,st)
    ld=loader(arrays,bundle.y_s2,idx,True); Model=model_class(); model=Model(m1.shape[2],m5.shape[2],m15.shape[2],st.shape[1]).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=CONFIG.learning_rate); loss_fn=torch.nn.SmoothL1Loss()
    for _ in range(max(1,epochs)):
        model.train()
        for a,b,c,s,y in ld:
            a,b,c,s,y=a.to(device),b.to(device),c.to(device),s.to(device),y.to(device)
            opt.zero_grad(set_to_none=True); loss=loss_fn(model(a,b,c,s),y); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
    return model,scalers


def main():
    ap=argparse.ArgumentParser(description="Train new 5/3/2 S2 model using Feb-Jun 2026 walk-forward OOS")
    ap.add_argument("--phase3a",type=Path,default=Path(r"D:\ORO_532_2026_TUNING\phase3a_532"))
    ap.add_argument("--out",type=Path,default=Path(r"D:\ORO_532_2026_TUNING\model_532"))
    a=ap.parse_args(); t0=time.perf_counter(); a.out.mkdir(parents=True,exist_ok=True)
    market=pd.read_parquet(a.phase3a/"market_state_2026.parquet"); cand=pd.read_parquet(a.phase3a/"technical_candidates_feb_jun_2026.parquet")
    market["timestamp"]=pd.to_datetime(market["timestamp"],utc=True); cand["timestamp"]=pd.to_datetime(cand["timestamp"],utc=True)
    bundle=build_tensor_bundle(market,cand); log(t0,f"5/3/2 valid samples={len(bundle.frame):,}")
    folds=make_folds(bundle.frame)
    import torch
    device="cuda" if torch.cuda.is_available() else "cpu"; print(f"DEVICE={device}")
    preds=[]; mets=[]; epochs=[]
    for fold in folds:
        p,m,e=train_fold(bundle,fold,device,t0); preds.append(p); mets.append(m); epochs.append(e)
    pred=pd.concat(preds,ignore_index=True).sort_values("timestamp")
    pred.to_csv(a.out/"oos_s2_predictions_532.csv.gz",index=False,compression="gzip")
    pd.DataFrame(mets).to_csv(a.out/"oos_s2_fold_metrics_532.csv",index=False)
    final_epochs=int(round(np.median(epochs))); model,scalers=fit_final(bundle,device,final_epochs)
    model_path=a.out/"selected_532_s2.pt"; scaler_path=a.out/"scalers_532_s2.joblib"
    torch.save(model.state_dict(),model_path); joblib.dump(scalers,scaler_path)
    summary={"research_version":RESEARCH_VERSION,"sequence_lengths":{"m1":5,"m5":3,"m15":2},"tuning_period":"2026-02-01..2026-06-30","samples":len(bundle.frame),"oos_predictions":len(pred),"median_best_epochs":final_epochs,"model_sha256":sha256(model_path),"scaler_sha256":sha256(scaler_path),"threshold_selected":False,"threshold_grid":list(CONFIG.threshold_grid)}
    (a.out/"TRAINING_532_SUMMARY.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("TRAIN_532_S2=PASS"); print(f"VALID_SAMPLES={len(bundle.frame):,}"); print(f"OOS_PREDICTIONS={len(pred):,}"); print(f"MEDIAN_BEST_EPOCHS={final_epochs}"); print("THRESHOLD_SELECTED=NO"); print(f"OUTPUT={a.out}")

if __name__=="__main__": main()
