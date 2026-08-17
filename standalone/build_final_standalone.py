from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

OLD_MODEL_SHA = "ee120a3623375b000c1d65d47debb9c0c3edbb92ee36c978ce3f14578e03035b"
OLD_SCALER_SHA = "73b1548cb5e2e2659b9d332128bf625663ca3eee26f29beaa5ea24b13d5999ac"
THRESHOLD = 0.32696733474731443
FORBIDDEN_ROOTS = [r"D:\\ORO", r"D:\\oro_top40_restore", r"D:\\oro_top40_shortmem"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copytree_clean(src: Path, dst: Path) -> None:
    ignore = shutil.ignore_patterns(
        ".venv", "__pycache__", ".pytest_cache", ".git", "*.log", "*.out.log", "*.err.log",
        "research_output", ".pytest_tmp",
    )
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=ignore)


def replace_text(path: Path, replacements: dict[str, str]) -> None:
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return
    new = text
    for a, b in replacements.items():
        new = new.replace(a, b)
    if new != text:
        path.write_text(new, encoding="utf-8")


def find_by_hash(root: Path, target_hash: str) -> list[Path]:
    hits = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".pt", ".pth", ".joblib", ".pkl", ".pickle"}:
            continue
        try:
            if sha256(p) == target_hash:
                hits.append(p)
        except OSError:
            pass
    return hits


def patch_short_memory_contract(dst: Path, model_hash: str, scaler_hash: str) -> None:
    seq = dst / "xau_bot" / "research" / "ml" / "phase3b2_sequential.py"
    if not seq.exists():
        raise RuntimeError(f"missing {seq}")
    text = seq.read_text(encoding="utf-8")
    text = re.sub(r"M1_LENGTH\s*=\s*60", "M1_LENGTH = 8", text)
    text = re.sub(r"M5_LENGTH\s*=\s*24", "M5_LENGTH = 5", text)
    text = re.sub(r"M15_LENGTH\s*=\s*16", "M15_LENGTH = 3", text)
    seq.write_text(text, encoding="utf-8")

    guard = dst / "xau_bot" / "live" / "causal_inference.py"
    if guard.exists():
        text = guard.read_text(encoding="utf-8")
        text = re.sub(r'EXPECTED_S2_MODEL_SHA256\s*=\s*"[0-9a-f]+"', f'EXPECTED_S2_MODEL_SHA256 = "{model_hash}"', text)
        text = re.sub(r'EXPECTED_S2_SCALER_SHA256\s*=\s*"[0-9a-f]+"', f'EXPECTED_S2_SCALER_SHA256 = "{scaler_hash}"', text)
        text = re.sub(r"TOP40_THRESHOLD\s*=\s*[-+0-9.eE]+", f"TOP40_THRESHOLD = {THRESHOLD!r}", text)
        guard.write_text(text, encoding="utf-8")


def patch_no_cooldown(dst: Path) -> None:
    path = dst / "xau_bot" / "risk" / "risk_manager.py"
    if not path.exists():
        raise RuntimeError(f"missing {path}")
    text = path.read_text(encoding="utf-8")
    # Preserve loss chronology but never create or enforce a cooldown.
    start = text.find("    def register_trade_result(")
    end = text.find("    def in_cooldown(", start)
    end2 = text.find("    def check_daily_drawdown(", end)
    if start < 0 or end < 0 or end2 < 0:
        raise RuntimeError("risk_manager cooldown methods not found")
    replacement = '''    def register_trade_result(self, is_win: bool, now_utc: datetime) -> None:\n        if is_win:\n            self.state.consecutive_losses = 0\n        else:\n            self.state.consecutive_losses += 1\n        self.state.cooldown_until_utc = None\n\n    def in_cooldown(self, now_utc: datetime) -> bool:\n        return False\n\n'''
    text = text[:start] + replacement + text[end2:]
    path.write_text(text, encoding="utf-8")


def write_launchers(dst: Path, runtime_name: str) -> None:
    (dst / "START_DEMO.bat").write_text(
        "@echo off\r\n"
        "cd /d %~dp0\r\n"
        "if not exist .venv\\Scripts\\python.exe (\r\n"
        "  echo Ambiente mancante. Eseguo INSTALL.bat...\r\n"
        "  call INSTALL.bat || exit /b 1\r\n"
        ")\r\n"
        f".venv\\Scripts\\python.exe {runtime_name} --enable-demo-orders\r\n"
        "pause\r\n",
        encoding="utf-8",
    )
    (dst / "CHECK_ONLY.bat").write_text(
        "@echo off\r\ncd /d %~dp0\r\n"
        "if not exist .venv\\Scripts\\python.exe call INSTALL.bat || exit /b 1\r\n"
        f".venv\\Scripts\\python.exe {runtime_name} --check-only\r\n"
        "pause\r\n",
        encoding="utf-8",
    )
    (dst / "INSTALL.bat").write_text(
        "@echo off\r\ncd /d %~dp0\r\n"
        "if not exist .venv\\Scripts\\python.exe python -m venv .venv\r\n"
        ".venv\\Scripts\\python.exe -m pip install --upgrade pip\r\n"
        ".venv\\Scripts\\python.exe -m pip install -r requirements-final.txt\r\n",
        encoding="utf-8",
    )


def write_verifier(dst: Path, model_rel: str, scaler_rel: str) -> None:
    code = f'''from pathlib import Path\nimport hashlib, json, re, sys\nROOT=Path(__file__).resolve().parent\nFORBIDDEN={FORBIDDEN_ROOTS!r}\ndef sha(p):\n h=hashlib.sha256();\n with p.open("rb") as f:\n  for c in iter(lambda:f.read(1024*1024),b""): h.update(c)\n return h.hexdigest()\npolicy=json.loads((ROOT/"FINAL_POLICY.json").read_text(encoding="utf-8"))\nassert sha(ROOT/{model_rel!r})==policy["artifact_hashes"]["s2_model_sha256"]\nassert sha(ROOT/{scaler_rel!r})==policy["artifact_hashes"]["s2_scaler_sha256"]\nseq=(ROOT/"xau_bot/research/ml/phase3b2_sequential.py").read_text(encoding="utf-8")\nassert "M1_LENGTH = 8" in seq and "M5_LENGTH = 5" in seq and "M15_LENGTH = 3" in seq\nrisk=(ROOT/"xau_bot/risk/risk_manager.py").read_text(encoding="utf-8")\nassert "def in_cooldown" in risk and "return False" in risk\nfor p in ROOT.rglob("*"):\n if not p.is_file() or p.suffix.lower() not in {{".py",".json",".bat",".txt",".md",".ini"}}: continue\n try: t=p.read_text(encoding="utf-8")\n except Exception: continue\n for bad in FORBIDDEN:\n  if bad.lower() in t.lower(): raise SystemExit(f"FORBIDDEN_EXTERNAL_REFERENCE {{bad}} in {{p}}")\nprint("STANDALONE_VERIFY=PASS")\nprint("sequence_lengths=8/5/3")\nprint("threshold=",policy["s2_threshold"])\nprint("fixed_lot=",policy["fixed_lot"])\nprint("max_positions=",policy["max_positions"])\nprint("cooldown_enabled=",policy["cooldown_enabled"])\n'''
    (dst / "verify_standalone.py").write_text(code, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build fully standalone TOP40 SHORT MEMORY 8/5/3 demo folder")
    ap.add_argument("--restore-source", type=Path, default=Path(r"D:\oro_top40_restore"))
    ap.add_argument("--engine-source", type=Path, default=Path(r"D:\ORO"))
    ap.add_argument("--short-source", type=Path, default=Path(r"D:\oro_top40_shortmem"))
    ap.add_argument("--out", type=Path, default=Path(r"D:\ORO_SHORT_MEMORY_FINAL"))
    ap.add_argument("--runtime", default="top40_demo_runtime.py")
    a = ap.parse_args()

    restore = a.restore_source.resolve(); engine = a.engine_source.resolve(); short = a.short_source.resolve(); dst = a.out.resolve()
    for p in (restore, engine, short):
        if not p.exists(): raise SystemExit(f"SOURCE_NOT_FOUND: {p}")
    runtime_src = restore / a.runtime
    if not runtime_src.exists(): raise SystemExit(f"RUNTIME_NOT_FOUND: {runtime_src}")

    artifacts = short / "research_output" / "short_memory_v1"
    model = artifacts / "selected_short_memory_s2.pt"
    scaler = artifacts / "short_memory_scalers_s2.joblib"
    manifest = artifacts / "short_memory_threshold_freeze_manifest.json"
    for p in (model, scaler, manifest):
        if not p.exists(): raise SystemExit(f"FROZEN_ARTIFACT_NOT_FOUND: {p}")
    mf = json.loads(manifest.read_text(encoding="utf-8"))
    if not mf.get("strict_no_future_guard"): raise SystemExit("STRICT_NO_FUTURE_GUARD_MISSING")
    if abs(float(mf["thresholds"]["TOP40_SHORT"]) - THRESHOLD) > 1e-12: raise SystemExit("THRESHOLD_MISMATCH")

    if dst.exists(): shutil.rmtree(dst)
    dst.mkdir(parents=True)

    # Start from the proven demo runtime restore tree, then overlay the canonical engine package.
    copytree_clean(restore, dst)
    if (engine / "xau_bot").exists():
        copytree_clean(engine / "xau_bot", dst / "xau_bot")
    if (short / "short_memory").exists():
        copytree_clean(short / "short_memory", dst / "short_memory")

    # Replace exact legacy frozen artifacts wherever the restored runtime expects them.
    model_hits = find_by_hash(dst, OLD_MODEL_SHA)
    scaler_hits = find_by_hash(dst, OLD_SCALER_SHA)
    if not model_hits: raise SystemExit("LEGACY_S2_MODEL_NOT_FOUND_BY_HASH_IN_RESTORE")
    if not scaler_hits: raise SystemExit("LEGACY_S2_SCALER_NOT_FOUND_BY_HASH_IN_RESTORE")
    for p in model_hits: shutil.copy2(model, p)
    for p in scaler_hits: shutil.copy2(scaler, p)

    new_model_hash = sha256(model); new_scaler_hash = sha256(scaler)
    patch_short_memory_contract(dst, new_model_hash, new_scaler_hash)
    patch_no_cooldown(dst)

    # Remove absolute source-root strings from text/config files.
    replacements = {
        str(restore): ".", str(engine): ".", str(short): ".",
        r"D:\oro_top40_restore": ".", r"D:\ORO": ".", r"D:\oro_top40_shortmem": ".",
    }
    for p in dst.rglob("*"):
        if p.is_file() and p.suffix.lower() in {".py", ".json", ".txt", ".md", ".ini", ".bat", ".yaml", ".yml"}:
            replace_text(p, replacements)

    # Keep a canonical local copy of frozen artifacts too.
    local_art = dst / "artifacts"; local_art.mkdir(exist_ok=True)
    shutil.copy2(model, local_art / "selected_short_memory_s2.pt")
    shutil.copy2(scaler, local_art / "short_memory_scalers_s2.joblib")
    shutil.copy2(manifest, local_art / "short_memory_threshold_freeze_manifest.json")

    policy = {
        "name": "TOP40_SHORT_MEMORY_FINAL",
        "sequence_lengths": {"m1": 8, "m5": 5, "m15": 3},
        "s2_threshold": THRESHOLD,
        "fixed_lot": 0.01,
        "max_positions": 3,
        "cooldown_enabled": False,
        "strict_no_future_guard": True,
        "artifact_hashes": {"s2_model_sha256": new_model_hash, "s2_scaler_sha256": new_scaler_hash},
        "research_benchmark": {"trades": 3530, "win_rate_pct": 64.1643059490085, "profit_factor": 2.5588563899868224, "expectancy_usd": 0.837940509915013, "net_profit_usd": 2957.93},
    }
    (dst / "FINAL_POLICY.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")

    (dst / "requirements-final.txt").write_text(
        "MetaTrader5\nnumpy>=1.26,<3\npandas>=2,<3\nscikit-learn==1.8.0\njoblib>=1.3\npyarrow>=15\ntorch==2.11.0\n",
        encoding="utf-8",
    )
    write_launchers(dst, a.runtime)
    write_verifier(dst, "artifacts/selected_short_memory_s2.pt", "artifacts/short_memory_scalers_s2.joblib")
    (dst / "README_FINAL.txt").write_text(
        "TOP40 SHORT MEMORY FINAL - standalone\n\n"
        "1) Eseguire INSTALL.bat una sola volta.\n"
        "2) Eseguire CHECK_ONLY.bat e verificare CHECK OK.\n"
        "3) Eseguire: .venv\\Scripts\\python.exe verify_standalone.py\n"
        "4) Solo dopo usare START_DEMO.bat.\n\n"
        "Dopo STANDALONE_VERIFY=PASS e CHECK OK, il runtime non deve dipendere dalle vecchie cartelle di ricerca.\n",
        encoding="utf-8",
    )

    subprocess.run([sys.executable, str(dst / "verify_standalone.py")], cwd=dst, check=True)
    print(f"STANDALONE_BUILD=PASS\nOUTPUT={dst}\nLEGACY_MODEL_TARGETS_REPLACED={len(model_hits)}\nLEGACY_SCALER_TARGETS_REPLACED={len(scaler_hits)}")


if __name__ == "__main__":
    main()
