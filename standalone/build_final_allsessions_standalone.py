from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def patch_all_sessions(dst: Path) -> None:
    path = dst / "xau_bot" / "strategy" / "decision_engine.py"
    if not path.exists():
        raise RuntimeError(f"missing decision engine: {path}")
    text = path.read_text(encoding="utf-8")
    old = '''    allowed_sessions = enabled_sessions or ("LONDON", "LONDON_NY_OVERLAP", "NEW_YORK")\n    if snapshot.session.value not in allowed_sessions:\n        return TradeDecision(action=TradeAction.NO_TRADE, reason_code="SESSION_BLOCKED",\n                             technical_signal=signal, timestamp_utc=now)\n\n'''
    if old not in text:
        raise RuntimeError("SESSION_FILTER_BLOCK_NOT_FOUND")
    new = '''    # ALL-SESSIONS FINAL: session is telemetry only; it never vetoes an otherwise valid entry.\n    # Weekend/market-closed behavior is still governed by broker data/MT5 availability and other guards.\n    allowed_sessions = enabled_sessions\n\n'''
    path.write_text(text.replace(old, new), encoding="utf-8")


def write_talkative_launcher(dst: Path, runtime_name: str) -> None:
    launcher = r'''from __future__ import annotations
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

REASONS = {
    "SESSION_BLOCKED": "Blocco sessione (NON dovrebbe apparire nella ALL-SESSIONS FINAL)",
    "NO_VALID_TECHNICAL_SETUP": "Nessun setup tecnico valido: il mercato non soddisfa ancora le conferme richieste",
    "BELOW_THRESHOLD": "Il punteggio ML S2 e' sotto la soglia congelata",
    "SPREAD_TOO_WIDE": "Spread troppo ampio: ingresso evitato",
    "MAX_POSITIONS_REACHED": "Ci sono gia' 3 posizioni aperte: nessuna nuova esposizione",
    "NEWS_LOCKOUT": "Finestra news ad alto impatto: ingresso bloccato",
    "DAILY_DRAWDOWN_LIMIT": "Limite di drawdown giornaliero raggiunto",
    "DAILY_STOP_ACTIVE": "Stop giornaliero gia' attivo",
    "LOT_BELOW_MINIMUM": "Sizing non eseguibile al lotto minimo",
    "ML_PREDICTION_MISSING": "Predizione ML mancante per questo candidato",
}

def explain(line: str) -> list[str]:
    u = line.upper()
    out: list[str] = []
    if "HEARTBEAT" in u:
        m = re.search(r"positions=(\d+/\d+).*?spread=([0-9.]+).*?equity=([0-9.]+)", line, re.I)
        if m:
            out.append(f"   STATO -> posizioni {m.group(1)} | spread {m.group(2)} pt | equity {m.group(3)}")
    if "S2_SCORE" in u:
        m = re.search(r"S2=([-+0-9.]+).*?threshold=([-+0-9.]+)", line, re.I)
        if m:
            score=float(m.group(1)); thr=float(m.group(2))
            out.append(f"   ML -> S2 {score:.4f} vs soglia {thr:.4f}: {'PASS' if score >= thr else 'NO PASS'}")
    if "TECH_CHECK" in u:
        m = re.search(r"action=([^ ]+).*?reason=([^ ]+).*?direction=([^ ]+).*?score=([^ ]+).*?session=([^ ]+).*?news=([^ ]+).*?spread=([^ ]+)", line, re.I)
        if m:
            out.append(f"   TECNICA -> action={m.group(1)} direction={m.group(3)} score={m.group(4)} session={m.group(5)} news={m.group(6)} spread={m.group(7)}")
            reason=m.group(2).upper()
            for key, msg in REASONS.items():
                if key in reason:
                    out.append(f"   PERCHE' -> {msg}")
    if "NO_TRADE" in u or "NO ORDER" in u or "NO_ORDER" in u:
        found=False
        for key,msg in REASONS.items():
            if key in u:
                out.append(f"   DECISIONE -> NO TRADE | {msg}"); found=True; break
        if not found:
            out.append("   DECISIONE -> NO TRADE | nessun ordine inviato")
    if any(k in u for k in ["ORDER SENT", "ORDER_SEND OK", "OPENED", "TRADE OPEN", "BUY 0.01", "SELL 0.01"]):
        out.append("   DECISIONE -> ORDINE DEMO INVIATO. La riga originale sopra contiene direzione/prezzo/SL/TP disponibili.")
    return out


def main() -> int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--check-only', action='store_true')
    ap.add_argument('--enable-demo-orders', action='store_true')
    args=ap.parse_args()
    runtime=ROOT/'top40_demo_runtime.py'
    cmd=[sys.executable, '-u', str(runtime)]
    if args.check_only: cmd.append('--check-only')
    elif args.enable_demo_orders: cmd.append('--enable-demo-orders')
    else: cmd.append('--once')
    print('='*78, flush=True)
    print(' ORO SHORT MEMORY ALL-SESSIONS FINAL - CONSOLE OPERATIVA', flush=True)
    print(' 8/5/3 | S2=0.32696733474731443 | lot=0.01 | max pos=3 | cooldown=OFF', flush=True)
    print(' SESSION FILTER=OFF | news/spread/risk guards=ON | DEMO ONLY', flush=True)
    print('='*78, flush=True)
    p=subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert p.stdout is not None
    for raw in p.stdout:
        line=raw.rstrip('\r\n')
        print(line, flush=True)
        for extra in explain(line): print(extra, flush=True)
    return p.wait()

if __name__=='__main__':
    raise SystemExit(main())
'''
    (dst / "talkative_demo_runtime.py").write_text(launcher, encoding="utf-8")
    bat = f'''@echo off\r\ncd /d %~dp0\r\nif not exist .venv\\Scripts\\python.exe (\r\n  echo Ambiente mancante. Eseguo INSTALL.bat...\r\n  call INSTALL.bat || exit /b 1\r\n)\r\nstart "ORO SHORT MEMORY - DEMO" powershell.exe -NoExit -ExecutionPolicy Bypass -Command "Set-Location -LiteralPath '%~dp0'; .\\.venv\\Scripts\\python.exe -u .\\talkative_demo_runtime.py --enable-demo-orders"\r\n'''
    (dst / "START_DEMO.bat").write_text(bat, encoding="utf-8")
    check = '''@echo off\r\ncd /d %~dp0\r\nif not exist .venv\\Scripts\\python.exe call INSTALL.bat || exit /b 1\r\n.venv\\Scripts\\python.exe -u talkative_demo_runtime.py --check-only\r\npause\r\n'''
    (dst / "CHECK_ONLY.bat").write_text(check, encoding="utf-8")


def extend_verifier(dst: Path) -> None:
    path=dst/'verify_standalone.py'
    text=path.read_text(encoding='utf-8')
    extra='''\n# ALL-SESSIONS contract\ndec=(ROOT/"xau_bot/strategy/decision_engine.py").read_text(encoding="utf-8")\nassert "SESSION_BLOCKED" not in dec, "SESSION_FILTER_STILL_ACTIVE"\nassert (ROOT/"talkative_demo_runtime.py").exists()\nprint("SESSION_FILTER=DISABLED")\nprint("TALKATIVE_CONSOLE=PASS")\n'''
    path.write_text(text+extra, encoding='utf-8')


def main() -> None:
    ap=argparse.ArgumentParser(description='Build standalone all-sessions SHORT MEMORY final candidate')
    ap.add_argument('--restore-source', type=Path, default=Path(r'D:\oro_top40_restore_OLD'))
    ap.add_argument('--engine-source', type=Path, default=Path(r'D:\ORO_OLD'))
    ap.add_argument('--short-source', type=Path, default=Path(r'D:\oro_top40_shortmem_OLD'))
    ap.add_argument('--out', type=Path, default=Path(r'D:\ORO_SHORT_MEMORY_ALLSESSIONS_FINAL'))
    a=ap.parse_args()
    here=Path(__file__).resolve().parent
    base=here/'build_final_standalone.py'
    cmd=[sys.executable,str(base),'--restore-source',str(a.restore_source),'--engine-source',str(a.engine_source),'--short-source',str(a.short_source),'--out',str(a.out)]
    subprocess.run(cmd, check=True)
    dst=a.out.resolve()
    patch_all_sessions(dst)
    write_talkative_launcher(dst, 'top40_demo_runtime.py')
    policy_path=dst/'FINAL_POLICY.json'
    policy=json.loads(policy_path.read_text(encoding='utf-8'))
    policy['name']='TOP40_SHORT_MEMORY_ALLSESSIONS_FINAL'
    policy['session_filter_enabled']=False
    policy['research_benchmark']={
        'trades':5351,
        'win_rate_pct':64.2683610540086,
        'profit_factor':2.5611087866108773,
        'expectancy_usd':0.8367146327789192,
        'net_profit_usd':4477.26,
        'max_drawdown_usd':17.21,
        'max_drawdown_pct':0.8925833815964692,
        'positive_pf_segments':12,
        'positive_expectancy_segments':12,
        'positive_pf_sessions':6,
        'positive_expectancy_sessions':6,
    }
    policy_path.write_text(json.dumps(policy,indent=2),encoding='utf-8')
    extend_verifier(dst)
    subprocess.run([sys.executable,str(dst/'verify_standalone.py')],cwd=dst,check=True)
    print('ALLSESSIONS_STANDALONE_BUILD=PASS')
    print(f'OUTPUT={dst}')

if __name__=='__main__':
    main()
