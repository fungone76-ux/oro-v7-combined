from __future__ import annotations

import argparse
from pathlib import Path

AUDIT_LAUNCHER = r'''from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
AUDIT_DIR = ROOT / "audit"
AUDIT_DIR.mkdir(exist_ok=True)
AUDIT_PATH = AUDIT_DIR / "live_parity_events.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit(event: dict) -> None:
    event = {"captured_at_utc": utc_now(), **event}
    with AUDIT_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def parse_stamp(line: str) -> str | None:
    m = re.match(r"\[([^\]]+)\]", line)
    return m.group(1) if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description="TOP40 live/replay parity capture")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--enable-demo-orders", action="store_true")
    args = ap.parse_args()

    runtime = ROOT / "talkative_demo_runtime.py"
    if not runtime.exists():
        runtime = ROOT / "top40_demo_runtime.py"
    if not runtime.exists():
        raise SystemExit("MISSING_RUNTIME")

    cmd = [sys.executable, "-u", str(runtime)]
    if args.check_only:
        cmd.append("--check-only")
    elif args.enable_demo_orders:
        cmd.append("--enable-demo-orders")
    else:
        cmd.append("--once")

    print("=" * 88, flush=True)
    print(" LIVE/REPLAY PARITY AUDIT V1 - TRADING LOGIC UNCHANGED", flush=True)
    print(f" audit -> {AUDIT_PATH}", flush=True)
    print("=" * 88, flush=True)

    emit({"event": "AUDIT_START", "command": cmd, "orders_enabled": bool(args.enable_demo_orders)})

    p = subprocess.Popen(
        cmd,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert p.stdout is not None

    current_eval: dict = {}
    for raw in p.stdout:
        line = raw.rstrip("\r\n")
        print(line, flush=True)
        upper = line.upper()
        stamp = parse_stamp(line)

        if "EVAL_WINDOW" in upper or "VALUTAZIONE M5" in upper:
            current_eval = {"runtime_stamp": stamp}
            emit({"event": "EVAL_WINDOW", "runtime_stamp": stamp, "raw": line})
            continue

        if "S2_SCORE" in upper:
            m = re.search(r"sid=([^ ]+).*?S2=([-+0-9.eE]+).*?threshold=([-+0-9.eE]+)", line, re.I)
            if m:
                score = float(m.group(2))
                threshold = float(m.group(3))
                event = {
                    "event": "S2_SCORE",
                    "runtime_stamp": stamp,
                    "sample_id": m.group(1),
                    "s2": score,
                    "threshold": threshold,
                    "margin": score - threshold,
                    "pass": score >= threshold,
                    "raw": line,
                }
                current_eval.update(event)
                emit(event)
            else:
                emit({"event": "S2_SCORE_UNPARSED", "runtime_stamp": stamp, "raw": line})
            continue

        if "TECH_CHECK" in upper:
            m = re.search(
                r"action=([^ ]+).*?reason=([^ ]+).*?direction=([^ ]+).*?score=([^ ]+).*?session=([^ ]+).*?news=([^ ]+).*?spread=([^ ]+)",
                line,
                re.I,
            )
            if m:
                event = {
                    "event": "TECH_CHECK",
                    "runtime_stamp": stamp,
                    "action": m.group(1),
                    "reason": m.group(2),
                    "direction": m.group(3),
                    "technical_score": m.group(4),
                    "session": m.group(5),
                    "news": m.group(6),
                    "spread": m.group(7),
                    "raw": line,
                }
                current_eval.update(event)
                emit(event)
            else:
                emit({"event": "TECH_CHECK_UNPARSED", "runtime_stamp": stamp, "raw": line})
            continue

        if "POSITION_AUDIT" in upper:
            emit({"event": "POSITION_AUDIT", "runtime_stamp": stamp, "raw": line})
            continue

        if any(tok in upper for tok in ("ORDER SENT", "ORDER_SEND OK", "TRADE OPEN", "OPENED BUY", "OPENED SELL")):
            emit({"event": "ORDER_OPEN", "runtime_stamp": stamp, "context": current_eval, "raw": line})
            continue

        if "NO_TRADE" in upper or "NO ORDER" in upper or "NO_ORDER" in upper:
            emit({"event": "NO_TRADE", "runtime_stamp": stamp, "context": current_eval, "raw": line})
            continue

        if "HEARTBEAT" in upper:
            m = re.search(r"positions=(\d+/\d+).*?spread=([0-9.]+).*?equity=([0-9.]+).*?atr_m5_pts=([0-9.]+)", line, re.I)
            if m:
                emit({
                    "event": "HEARTBEAT",
                    "runtime_stamp": stamp,
                    "positions": m.group(1),
                    "spread": float(m.group(2)),
                    "equity": float(m.group(3)),
                    "atr_m5_pts": float(m.group(4)),
                })

    rc = p.wait()
    emit({"event": "AUDIT_STOP", "returncode": rc})
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
'''

START_BAT = r'''@echo off
cd /d %~dp0
if not exist .venv\Scripts\python.exe (
  echo Ambiente mancante. Eseguo INSTALL.bat...
  call INSTALL.bat || exit /b 1
)
start "ORO SHORT MEMORY - PARITY AUDIT" powershell.exe -NoExit -ExecutionPolicy Bypass -Command "Set-Location -LiteralPath '%~dp0'; .\.venv\Scripts\python.exe -u .\live_replay_parity_runtime.py --enable-demo-orders"
'''


def main() -> None:
    ap = argparse.ArgumentParser(description="Install LIVE/REPLAY parity capture into an existing ALLSESSIONS FINAL folder")
    ap.add_argument("--target", type=Path, required=True)
    args = ap.parse_args()

    root = args.target.resolve()
    if not (root / "top40_demo_runtime.py").exists():
        raise RuntimeError(f"NOT_A_FINAL_RUNTIME: {root}")

    (root / "live_replay_parity_runtime.py").write_text(AUDIT_LAUNCHER, encoding="utf-8")
    (root / "START_DEMO_PARITY_AUDIT.bat").write_text(START_BAT, encoding="utf-8")
    (root / "audit").mkdir(exist_ok=True)

    print("LIVE_REPLAY_PARITY_AUDIT_V1=PASS")
    print("TRADING_RUNTIME_CHANGED=NO")
    print("TRADING_LOGIC_CHANGED=NO")
    print("CAPTURE=timestamp/direction/technical_score/S2/threshold/decision/order/position_audit")
    print(f"TARGET={root}")


if __name__ == "__main__":
    main()
