from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
FROZEN_THRESHOLD=0.32696733474731443
def sha256(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',type=Path,default=Path(r'D:\oro_top40_shortmem_OLD\unseen_2016')); a=ap.parse_args(); d=a.data_dir.resolve()
 src=d/'UNSEEN_2016_DOWNLOAD_MANIFEST.json'
 if not src.exists(): raise RuntimeError(f'MANIFEST_MISSING: {src}')
 m=json.loads(src.read_text(encoding='utf-8'))
 if int(m.get('evaluation_year',0))!=2016: raise RuntimeError('WRONG_EVALUATION_YEAR')
 files={}
 for tf in ('M1','M5','M15'):
  p=d/f'XAUUSD_{tf}_UNSEEN_2016.csv'
  if not p.exists(): raise RuntimeError(f'MISSING_{tf}')
  digest=sha256(p); expected=m['files'][tf]['sha256']
  if digest!=expected: raise RuntimeError(f'HASH_CHANGED_{tf}')
  files[tf]={'path':str(p),'rows':int(m['files'][tf]['rows']),'first_bar_utc':m['files'][tf]['first_bar_utc'],'last_bar_utc':m['files'][tf]['last_bar_utc'],'sha256':digest}
 frozen={'protocol':'UNSEEN_2016_FROZEN_V1','evaluation_period':'2016-01-01..2016-12-31','dataset_hashes':files,'model_policy':{'sequence_lengths':{'m1':8,'m5':5,'m15':3},'s2_threshold':FROZEN_THRESHOLD,'fixed_lot':0.01,'max_positions':3,'cooldown_enabled':False,'session_filter_enabled':False},'rules':['NO_RETRAINING','NO_THRESHOLD_RECALIBRATION','NO_PARAMETER_TUNING_AFTER_RESULT']}
 out=d/'UNSEEN_2016_FROZEN_MANIFEST.json'; out.write_text(json.dumps(frozen,indent=2),encoding='utf-8')
 print('UNSEEN_2016_HASH_FREEZE=PASS'); print(f'FROZEN_THRESHOLD={FROZEN_THRESHOLD:.15f}'); print('NO_RETRAINING=FROZEN'); print('NO_THRESHOLD_RECALIBRATION=FROZEN'); print(f'MANIFEST={out}')
if __name__=='__main__': main()
