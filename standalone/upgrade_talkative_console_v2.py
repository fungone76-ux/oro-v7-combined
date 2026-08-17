from __future__ import annotations
import argparse
from pathlib import Path

LAUNCHER = r'''from __future__ import annotations
import argparse,re,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
REASONS={"NO_VALID_TECHNICAL_SETUP":"nessun setup tecnico valido","BELOW_THRESHOLD":"punteggio S2 sotto la soglia congelata","SPREAD_TOO_WIDE":"spread troppo ampio","MAX_POSITIONS_REACHED":"limite di 3 posizioni raggiunto","NEWS_LOCKOUT":"finestra news ad alto impatto","DAILY_DRAWDOWN_LIMIT":"limite drawdown giornaliero raggiunto","DAILY_STOP_ACTIVE":"stop giornaliero attivo","LOT_BELOW_MINIMUM":"lotto non eseguibile","ML_PREDICTION_MISSING":"predizione ML non disponibile"}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--check-only',action='store_true'); ap.add_argument('--enable-demo-orders',action='store_true'); a=ap.parse_args()
 cmd=[sys.executable,'-u',str(ROOT/'top40_demo_runtime.py')]
 cmd.append('--check-only' if a.check_only else '--enable-demo-orders' if a.enable_demo_orders else '--once')
 print('='*82); print(' ORO SHORT MEMORY ALL-SESSIONS FINAL - CONSOLE OPERATIVA V2'); print(' 8/5/3 | S2=0.32696733474731443 | lot=0.01 | max pos=3 | cooldown=OFF'); print(' SESSION FILTER=OFF | news/spread/risk guards=ON | DEMO ONLY'); print('='*82,flush=True)
 p=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1); assert p.stdout is not None
 active=False; had_score=False; had_decision=False
 for raw in p.stdout:
  line=raw.rstrip('\r\n'); u=line.upper()
  if 'EVAL_WINDOW LOADING MARKET DATA' in u:
   if not active:
    active=True; had_score=False; had_decision=False; ts=line[:10] if line.startswith('[') else ''
    print(f'{ts} VALUTAZIONE M5 -> analisi dati 8/5/3 in corso...',flush=True)
   continue
  if 'HEARTBEAT' in u:
   if active and not had_decision:
    print('   RISULTATO -> '+('nessun candidato tecnico/ML emesso nella finestra; nessun ordine inviato' if not had_score else 'valutazione conclusa senza ordine'),flush=True); active=False
   print(line,flush=True)
   m=re.search(r'positions=(\d+/\d+).*?spread=([0-9.]+).*?equity=([0-9.]+).*?next_eval_in=([0-9]+)s',line,re.I)
   if m: print(f'   STATO -> posizioni {m.group(1)} | spread {m.group(2)} pt | equity {m.group(3)} | prossima analisi ~{m.group(4)}s',flush=True)
   continue
  print(line,flush=True)
  if 'S2_SCORE' in u:
   had_score=True; m=re.search(r'S2=([-+0-9.]+).*?threshold=([-+0-9.]+)',line,re.I)
   if m:
    s=float(m.group(1)); t=float(m.group(2)); print(f'   ML -> S2={s:.4f} | soglia={t:.4f} | {"PASS" if s>=t else "NO PASS"}',flush=True)
  if 'TECH_CHECK' in u:
   m=re.search(r'action=([^ ]+).*?reason=([^ ]+).*?direction=([^ ]+).*?score=([^ ]+).*?session=([^ ]+).*?news=([^ ]+).*?spread=([^ ]+)',line,re.I)
   if m:
    print(f'   TECNICA -> action={m.group(1)} | dir={m.group(3)} | score={m.group(4)} | session={m.group(5)} | news={m.group(6)} | spread={m.group(7)}',flush=True)
    if m.group(2).upper() in REASONS: print("   PERCHE' -> "+REASONS[m.group(2).upper()],flush=True)
  if 'NO_TRADE' in u or 'NO_ORDER' in u or 'NO ORDER' in u:
   had_decision=True; msg=next((v for k,v in REASONS.items() if k in u),'nessun ordine inviato'); print(f'   DECISIONE -> NO TRADE | {msg}',flush=True); active=False
  if any(k in u for k in ('ORDER_SEND OK','ORDER SENT','TRADE OPEN','OPENED')):
   had_decision=True; print('   DECISIONE -> ORDINE DEMO INVIATO',flush=True); active=False
 return p.wait()
if __name__=='__main__': raise SystemExit(main())
'''
BAT=r'''@echo off
cd /d %~dp0
if not exist .venv\Scripts\python.exe (call INSTALL.bat || exit /b 1)
start "ORO SHORT MEMORY - DEMO" powershell.exe -NoExit -ExecutionPolicy Bypass -Command "Set-Location -LiteralPath '%~dp0'; .\.venv\Scripts\python.exe -u .\talkative_demo_runtime.py --enable-demo-orders"
'''
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--target',type=Path,default=Path(r'D:\ORO_SHORT_MEMORY_ALLSESSIONS_FINAL')); a=ap.parse_args(); dst=a.target.resolve()
 if not (dst/'top40_demo_runtime.py').exists(): raise RuntimeError(f'RUNTIME_NOT_FOUND: {dst}')
 (dst/'talkative_demo_runtime.py').write_text(LAUNCHER,encoding='utf-8'); (dst/'START_DEMO.bat').write_text(BAT,encoding='utf-8')
 print('TALKATIVE_CONSOLE_V2=PASS'); print('TRADING_LOGIC_CHANGED=NO'); print(f'TARGET={dst}')
if __name__=='__main__': main()
