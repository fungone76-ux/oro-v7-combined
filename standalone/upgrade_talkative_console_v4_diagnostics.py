from __future__ import annotations

import argparse
from pathlib import Path

LAUNCHER = r'''from __future__ import annotations
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
THRESHOLD = 0.32696733474731443

REASONS = {
    "SESSION_BLOCKED": "Blocco sessione (non dovrebbe apparire nella ALL-SESSIONS FINAL)",
    "NO_VALID_TECHNICAL_SETUP": "Nessun setup tecnico valido",
    "BELOW_THRESHOLD": "S2 sotto soglia congelata",
    "SPREAD_TOO_WIDE": "Spread troppo ampio",
    "MAX_POSITIONS_REACHED": "Limite massimo posizioni raggiunto",
    "NEWS_LOCKOUT": "Finestra news ad alto impatto",
    "DAILY_DRAWDOWN_LIMIT": "Limite drawdown giornaliero",
    "DAILY_STOP_ACTIVE": "Stop giornaliero attivo",
    "LOT_BELOW_MINIMUM": "Lotto non eseguibile",
    "ML_PREDICTION_MISSING": "Predizione ML mancante",
}


def reason_text(text: str) -> str:
    u = text.upper()
    for key, value in REASONS.items():
        if key in u:
            return value
    return "Nessuna condizione di ingresso completa"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--enable-demo-orders', action='store_true')
    args = ap.parse_args()

    runtime = ROOT / 'top40_demo_runtime.py'
    cmd = [sys.executable, '-u', str(runtime)]
    if args.check_only:
        cmd.append('--check-only')
    elif args.enable_demo_orders:
        cmd.append('--enable-demo-orders')
    else:
        cmd.append('--once')

    print('=' * 88, flush=True)
    print(' ORO SHORT MEMORY ALL-SESSIONS FINAL - DIAGNOSTICA PIPELINE V4', flush=True)
    print(' 8/5/3 | S2=0.32696733474731443 | lot=0.01 | max pos=3 | cooldown=OFF', flush=True)
    print(' SOLO TELEMETRIA: nessuna modifica alla logica di trading', flush=True)
    print('=' * 88, flush=True)

    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    assert p.stdout is not None

    evaluation_open = False
    evaluation_has_candidate = False
    evaluation_has_score = False
    evaluation_has_decision = False
    last_feed = None

    for raw in p.stdout:
        line = raw.rstrip('\r\n')
        upper = line.upper()

        if 'HEARTBEAT' in upper:
            m = re.search(r'positions=(\d+/\d+).*?spread=([0-9.]+).*?equity=([0-9.]+).*?atr_m5_pts=([0-9.]+).*?next_eval_in=([0-9]+)s', line, re.I)
            if m:
                last_feed = {
                    'positions': m.group(1), 'spread': m.group(2), 'equity': m.group(3),
                    'atr': m.group(4), 'next': m.group(5),
                }
                print(line, flush=True)
                print(f"   MT5 FEED -> ATTIVO | posizioni {m.group(1)} | spread {m.group(2)} pt | equity {m.group(3)} | ATR M5 {m.group(4)} | next ~{m.group(5)}s", flush=True)
                if evaluation_open and not evaluation_has_decision:
                    if not evaluation_has_candidate and not evaluation_has_score:
                        print('   PIPELINE -> nessun candidato tecnico sopravvissuto nella finestra', flush=True)
                        print('   ML -> NON CALCOLATO', flush=True)
                        print('   DECISIONE -> NO TRADE', flush=True)
                    print('-' * 88, flush=True)
                    evaluation_open = False
                continue

        if 'EVAL_WINDOW LOADING MARKET DATA' in upper:
            if not evaluation_open:
                evaluation_open = True
                evaluation_has_candidate = False
                evaluation_has_score = False
                evaluation_has_decision = False
                stamp = line.split(']')[0].lstrip('[') if ']' in line else '--:--:--'
                print('', flush=True)
                print('-' * 88, flush=True)
                print(f'[{stamp}] NUOVA VALUTAZIONE M5', flush=True)
                if last_feed:
                    print(f"   MT5 FEED -> ATTIVO | spread {last_feed['spread']} pt | ATR M5 {last_feed['atr']}", flush=True)
                else:
                    print('   MT5 FEED -> in attesa del primo heartbeat', flush=True)
                print('   INPUT -> sequenze richieste M1=8 | M5=5 | M15=3', flush=True)
                print('   PIPELINE -> ricerca candidato tecnico in corso...', flush=True)
            continue

        print(line, flush=True)

        if 'S2_SCORE' in upper:
            evaluation_has_candidate = True
            evaluation_has_score = True
            m = re.search(r'S2=([-+0-9.]+).*?threshold=([-+0-9.]+)', line, re.I)
            if m:
                score = float(m.group(1)); thr = float(m.group(2)); margin = score - thr
                verdict = 'PASS' if score >= thr else 'FAIL'
                sign = '+' if margin >= 0 else ''
                print('   CANDIDATO -> PRESENTE: il candidato e\' arrivato allo scoring ML', flush=True)
                print(f'   ML -> S2={score:.6f} | soglia={thr:.6f} | scostamento={sign}{margin:.6f} | {verdict}', flush=True)

        if 'TECH_CHECK' in upper:
            evaluation_has_candidate = True
            m = re.search(r'action=([^ ]+).*?reason=([^ ]+).*?direction=([^ ]+).*?score=([^ ]+).*?session=([^ ]+).*?news=([^ ]+).*?spread=([^ ]+)', line, re.I)
            if m:
                print(f'   TECNICA -> action={m.group(1)} | direzione={m.group(3)} | score={m.group(4)}', flush=True)
                print(f'   CONTESTO -> sessione={m.group(5)} | news={m.group(6)} | spread={m.group(7)} pt', flush=True)
                if m.group(1).upper() == 'NO_TRADE':
                    print(f"   PERCHE' -> {reason_text(m.group(2))}", flush=True)

        if 'BELOW_THRESHOLD' in upper:
            evaluation_has_decision = True
            print('   DECISIONE -> NO TRADE | S2 sotto soglia congelata', flush=True)
        elif 'NO_TRADE' in upper or 'NO_ORDER' in upper or 'NO ORDER' in upper:
            evaluation_has_decision = True
            print(f'   DECISIONE -> NO TRADE | {reason_text(line)}', flush=True)

        if any(token in upper for token in ('ORDER SENT', 'ORDER_SEND OK', 'TRADE OPEN', 'OPENED BUY', 'OPENED SELL')):
            evaluation_has_decision = True
            print('   DECISIONE -> ORDINE DEMO INVIATO', flush=True)

        if evaluation_open and evaluation_has_decision:
            print('-' * 88, flush=True)
            evaluation_open = False

    return p.wait()


if __name__ == '__main__':
    raise SystemExit(main())
'''

START_BAT = r'''@echo off
cd /d %~dp0
if not exist .venv\Scripts\python.exe (
  echo Ambiente mancante. Eseguo INSTALL.bat...
  call INSTALL.bat || exit /b 1
)
start "ORO SHORT MEMORY - DIAGNOSTICA" powershell.exe -NoExit -ExecutionPolicy Bypass -Command "Set-Location -LiteralPath '%~dp0'; .\.venv\Scripts\python.exe -u .\talkative_demo_runtime.py --enable-demo-orders"
'''


def main() -> None:
    ap = argparse.ArgumentParser(description='Upgrade ONLY explanatory console to safe pipeline diagnostics V4')
    ap.add_argument('--target', type=Path, default=Path(r'D:\ORO_SHORT_MEMORY_ALLSESSIONS_FINAL'))
    a = ap.parse_args()
    target = a.target.resolve()
    runtime = target / 'top40_demo_runtime.py'
    if not runtime.exists():
        raise RuntimeError(f'NOT_AN_ALLSESSIONS_STANDALONE: missing {runtime}')

    # Important: only wrapper and launcher are touched; trading runtime remains byte-for-byte unchanged.
    before = runtime.read_bytes()
    (target / 'talkative_demo_runtime.py').write_text(LAUNCHER, encoding='utf-8')
    (target / 'START_DEMO.bat').write_text(START_BAT, encoding='utf-8')
    after = runtime.read_bytes()
    if before != after:
        raise RuntimeError('TRADING_RUNTIME_CHANGED_UNEXPECTEDLY')

    print('PIPELINE_DIAGNOSTICS_V4=PASS')
    print('TRADING_RUNTIME_CHANGED=NO')
    print('TRADING_LOGIC_CHANGED=NO')
    print('MT5_FEED_STATUS_FROM_HEARTBEAT=ENABLED')
    print('CANDIDATE_STAGE_STATUS=ENABLED')
    print('S2_MARGIN_STATUS=ENABLED')
    print(f'TARGET={target}')


if __name__ == '__main__':
    main()
