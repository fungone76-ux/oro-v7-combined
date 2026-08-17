from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import MetaTrader5 as mt5
import pandas as pd
OUT=Path(r'D:\oro_top40_shortmem_OLD\unseen_2016'); SYMBOL='XAUUSD'
START=datetime(2015,12,1,tzinfo=timezone.utc); END=datetime(2017,1,5,23,59,tzinfo=timezone.utc)
TFS={'M1':mt5.TIMEFRAME_M1,'M5':mt5.TIMEFRAME_M5,'M15':mt5.TIMEFRAME_M15}
def sha256(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
 return h.hexdigest()
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 if not mt5.initialize(): raise SystemExit(f'MT5 initialize failed: {mt5.last_error()}')
 acc=mt5.account_info()
 if acc is None: raise SystemExit(f'MT5 account unavailable: {mt5.last_error()}')
 if not mt5.symbol_select(SYMBOL,True): raise SystemExit(f'Cannot select {SYMBOL}: {mt5.last_error()}')
 manifest={'symbol':SYMBOL,'download_start_utc':START.isoformat(),'download_end_utc':END.isoformat(),'evaluation_year':2016,'account_login':int(acc.login),'server':str(acc.server),'files':{}}
 print(f'MT5 connected | login={acc.login} server={acc.server}')
 for name,tf in TFS.items():
  print(f'Downloading {SYMBOL} {name} ...',flush=True); rates=mt5.copy_rates_range(SYMBOL,tf,START,END)
  if rates is None or len(rates)==0: raise RuntimeError(f'NO_DATA_{name}: {mt5.last_error()}')
  df=pd.DataFrame(rates); df['time']=pd.to_datetime(df['time'],unit='s',utc=True)
  cols=[c for c in ['time','open','high','low','close','tick_volume','spread','real_volume'] if c in df.columns]
  df=df[cols].drop_duplicates('time').sort_values('time').reset_index(drop=True)
  p=OUT/f'XAUUSD_{name}_UNSEEN_2016.csv'; df.to_csv(p,index=False); digest=sha256(p)
  manifest['files'][name]={'path':str(p),'rows':int(len(df)),'first_bar_utc':df.time.iloc[0].isoformat(),'last_bar_utc':df.time.iloc[-1].isoformat(),'sha256':digest}
  print(f'{name}: {len(df):,} bars | {df.time.iloc[0]} -> {df.time.iloc[-1]}'); print(f'SHA256={digest}')
 mp=OUT/'UNSEEN_2016_DOWNLOAD_MANIFEST.json'; mp.write_text(json.dumps(manifest,indent=2),encoding='utf-8'); mt5.shutdown()
 print('UNSEEN_2016_DOWNLOAD=PASS'); print(f'MANIFEST={mp}')
if __name__=='__main__': main()
