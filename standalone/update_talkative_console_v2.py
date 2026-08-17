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

REASONS = {
    "SESSION_BLOCKED": "Blocco sessione (non dovrebbe apparire nella ALL-SESSIONS FINAL)",
    "NO_VALID_TECHNICAL_SETUP": "Nessun setup tecnico valido: le conferme richieste non sono presenti",
    "BELOW_THRESHOLD": "Il punteggio ML S2 e' sotto la soglia congelata",
    "SPREAD_TOO_WIDE": "Spread troppo ampio: ingresso evitato",
    "MAX_POSITIONS_REACHED": "Sono gia' aperte 3 posizioni: nuova esposizione bloccata",
    "NEWS_LOCKOUT": "Finestra news ad alto impatto: ingresso bloccato",
    "DAILY_DRAWDOWN_LIMIT": "Limite di drawdown giornaliero raggiunto",
    "DAILY_STOP_ACTIVE": "Stop giornaliero gia' attivo",
    "LOT_BELOW_MINIMUM": "Sizing non eseguibile al lotto minimo",
    "ML_PREDICTION_MISSING": "Predizione ML mancante per questo candidato",
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

    print('=' * 82, flush=True)
    print(' ORO SHORT MEMORY ALL-SESSIONS FINAL - CONSOLE OPERATIVA V2', flush=True)
    print(' 8/5/3 | S2=0.32696733474731443 | lot=0.01 | max pos=3 | cooldown=OFF', flush=True)
    print(' SESSION FILTER=OFF | news/spread/risk guards=ON | DEMO ONLY', flush=True)
    print('=' * 82, flush=True)

    p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    assert p.stdout is not None

    evaluation_open = False
    evaluation_has_result = False

    for raw in p.stdout:
        line = raw.rstrip('\r\n')
        upper = line.upper()

        # Collapse the noisy repeated EVAL_WINDOW messages into one readable block.
        if 'EVAL_WINDOW LOADING MARKET DATA' in upper:
            if not evaluation_open:
                evaluation_open = True
                evaluation_has_result = False
                stamp = line.split(']')[0].lstrip('[') if ']' in line else '--:--:--'
                print('', flush=True)
                print('-' * 82, flush=True)
                print(f'[{stamp}] NUOVA VALUTAZIONE M5', flush=True)
                print('   ANALISI -> carico mercato e cerco un candidato tecnico causale 8/5/3...', flush=True)
            continue

        # A heartbeat after an evaluation window with no score/decision means no candidate survived.
        if 'HEARTBEAT' in upper and evaluation_open and not evaluation_has_result:
            print('   TECNICA -> nessun candidato tecnico valido trovato in questa finestra', flush=True)
            print('   ML -> S2 non calcolato: senza candidato non c\'e\' nulla da sottoporre al modello', flush=True)
            print('   DECISIONE -> NO TRADE', flush=True)
            print('-' * 82, flush=True)
            evaluation_open = False

        print(line, flush=True)

        if 'HEARTBEAT' in upper:
            m = re.search(r'positions=(\d+/\d+).*?spread=([0-9.]+).*?equity=([0-9.]+).*?next_eval_in=([0-9]+)s', line, re.I)
            if m:
                print(f'   STATO -> posizioni {m.group(1)} | spread {m.group(2)} pt | equity {m.group(3)} | prossima analisi ~{m.group(4)}s', flush=True)

        if 'S2_SCORE' in upper:
            evaluation_has_result = True
            m = re.search(r'S2=([-+0-9.]+).*?threshold=([-+0-9.]+)', line, re.I)
            if m:
                score = float(m.group(1)); thr = float(m.group(2))
                print(f'   ML -> S2={score:.4f} | soglia={thr:.4f} | {"PASS" if score >= thr else "NO PASS"}', flush=True)

        if 'TECH_CHECK' in upper:
            evaluation_has_result = True
            m = re.search(r'action=([^ ]+).*?reason=([^ ]+).*?direction=([^ ]+).*?score=([^ ]+).*?session=([^ ]+).*?news=([^ ]+).*?spread=([^ ]+)', line, re.I)
            if m:
                print(f'   TECNICA -> action={m.group(1)} | direzione={m.group(3)} | score={m.group(4)}', flush=True)
                print(f'   CONTESTO -> sessione={m.group(5)} (ammessa) | news={m.group(6)} | spread={m.group(7)} pt', flush=True)
                if m.group(1).upper() == 'NO_TRADE':
                    print(f'   PERCHE\' -> {reason_text(m.group(2))}', flush=True)

        if 'BELOW_THRESHOLD' in upper:
            evaluation_has_result = True
            print('   DECISIONE -> NO TRADE | S2 sotto soglia congelata', flush=True)
            if evaluation_open:
                print('-' * 82, flush=True)
                evaluation_open = False

        elif 'NO_TRADE' in upper or 'NO_ORDER' in upper or 'NO ORDER' in upper:
            evaluation_has_result = True
            print(f'   DECISIONE -> NO TRADE | {reason_text(line)}', flush=True)
            if evaluation_open:
                print('-' * 82, flush=True)
                evaluation_open = False

        if any(token in upper for token in ('ORDER SENT', 'ORDER_SEND OK', 'TRADE OPEN', 'OPENED BUY', 'OPENED SELL')):
            evaluation_has_result = True
            print('   DECISIONE -> ORDINE DEMO INVIATO', flush=True)
            print('   GESTIONE -> SL/TP e gestione posizione restano quelle del runtime congelato', flush=True)
            if evaluation_open:
                print('-' * 82, flush=True)
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
start "ORO SHORT MEMORY - DEMO" powershell.exe -NoExit -ExecutionPolicy Bypass -Command "Set-Location -LiteralPath '%~dp0'; .\.venv\Scripts\python.exe -u .\talkative_demo_runtime.py --enable-demo-orders"
'''

CHECK_BAT = r'''@echo off
cd /d %~dp0
if not exist .venv\Scripts\python.exe call INSTALL.bat || exit /b 1
.venv\Scripts\python.exe -u talkative_demo_runtime.py --check-only
pause
'''


def main() -> None:
    ap = argparse.ArgumentParser(description='Update ONLY the explanatory console in an existing ALL-SESSIONS FINAL folder')
    ap.add_argument('--target', type=Path, default=Path(r'D:\ORO_SHORT_MEMORY_ALLSESSIONS_FINAL'))
    a = ap.parse_args()
    target = a.target.resolve()
    runtime = target / 'top40_demo_runtime.py'
    if not runtime.exists():
        raise RuntimeError(f'NOT_AN_ALLSESSIONS_STANDALONE: missing {runtime}')
    (target / 'talkative_demo_runtime.py').write_text(LAUNCHER, encoding='utf-8')
    (target / 'START_DEMO.bat').write_text(START_BAT, encoding='utf-8')
    (target / 'CHECK_ONLY.bat').write_text(CHECK_BAT, encoding='utf-8')
    print('TALKATIVE_CONSOLE_V2=PASS')
    print('TRADING_LOGIC_CHANGED=NO')
    print(f'TARGET={target}')


if __name__ == '__main__':
    main()
