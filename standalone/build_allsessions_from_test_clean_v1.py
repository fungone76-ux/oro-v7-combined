from __future__ import annotations

"""Build a clean standalone ALL-SESSIONS demo folder from the STRICT replay contract.

This builder deliberately does NOT copy the previous FINAL standalone folder and
never uses top40_demo_runtime.py.  Sources are limited to:
- canonical xau_bot engine tree (default D:\\ORO),
- frozen short-memory research artifacts (default D:\\oro_top40_shortmem_OLD),
- standalone/top40_allsessions_from_test_v1.py from this repository checkout.

The resulting runtime keeps the test contract:
8/5/3, frozen S2=0.32696733474731443, fixed 0.01, max 3, all sessions,
cooldown off, pending entry executed on the next M1 cycle, immutable initial R.
DEMO ONLY.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

THRESHOLD = 0.32696733474731443
EXPECTED_MODEL_SHA = "fb19d64b58a2df7f51494d479ec1d9c354eb565c2936ecc8ff4f347520b14f50"
EXPECTED_SCALER_SHA = "a74bb603df616cebe1fb3540fe870500fe1b77403afdb971340174e0d965cf40"
LEGACY_MODEL_SHA = "ee120a3623375b000c1d65d47debb9c0c3edbb92ee36c978ce3f14578e03035b"
LEGACY_SCALER_SHA = "73b1548cb5e2e2659b9d332128bf625663ca3eee26f29beaa5ea24b13d5999ac"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copytree_clean(src: Path, dst: Path) -> None:
    ignore = shutil.ignore_patterns(
        ".venv", ".git", "__pycache__", ".pytest_cache", "*.log", "*.out.log", "*.err.log",
        "backtests", "research_output", ".pytest_tmp",
    )
    shutil.copytree(src, dst, dirs_exist_ok=True, ignore=ignore)


def find_by_hash(root: Path, digest: str) -> list[Path]:
    hits: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in {".pt", ".pth", ".joblib", ".pkl", ".pickle"}:
            continue
        try:
            if sha256(p) == digest:
                hits.append(p)
        except OSError:
            pass
    return hits


def patch_sequence_lengths(dst: Path) -> None:
    path = dst / "xau_bot" / "research" / "ml" / "phase3b2_sequential.py"
    text = path.read_text(encoding="utf-8-sig")
    text = re.sub(r"M1_LENGTH\s*=\s*\d+", "M1_LENGTH = 8", text)
    text = re.sub(r"M5_LENGTH\s*=\s*\d+", "M5_LENGTH = 5", text)
    text = re.sub(r"M15_LENGTH\s*=\s*\d+", "M15_LENGTH = 3", text)
    path.write_text(text, encoding="utf-8")


def patch_no_cooldown(dst: Path) -> None:
    path = dst / "xau_bot" / "risk" / "risk_manager.py"
    text = path.read_text(encoding="utf-8-sig")
    start = text.find("    def register_trade_result(")
    mid = text.find("    def in_cooldown(", start)
    end = text.find("    def check_daily_drawdown(", mid)
    if min(start, mid, end) < 0:
        raise RuntimeError("RISK_MANAGER_COOLDOWN_METHODS_NOT_FOUND")
    replacement = '''    def register_trade_result(self, is_win: bool, now_utc) -> None:\n        if is_win:\n            self.state.consecutive_losses = 0\n        else:\n            self.state.consecutive_losses += 1\n        self.state.cooldown_until_utc = None\n\n    def in_cooldown(self, now_utc) -> bool:\n        return False\n\n'''
    path.write_text(text[:start] + replacement + text[end:], encoding="utf-8")


def patch_frozen_guard(dst: Path, model_sha: str, scaler_sha: str) -> None:
    path = dst / "xau_bot" / "live" / "causal_inference.py"
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8-sig")
    text = re.sub(r'EXPECTED_S2_MODEL_SHA256\s*=\s*"[0-9a-f]+"', f'EXPECTED_S2_MODEL_SHA256 = "{model_sha}"', text)
    text = re.sub(r'EXPECTED_S2_SCALER_SHA256\s*=\s*"[0-9a-f]+"', f'EXPECTED_S2_SCALER_SHA256 = "{scaler_sha}"', text)
    text = re.sub(r"TOP40_THRESHOLD\s*=\s*[-+0-9.eE]+", f"TOP40_THRESHOLD = {THRESHOLD!r}", text)
    path.write_text(text, encoding="utf-8")


def write_launchers(dst: Path) -> None:
    (dst / "INSTALL.bat").write_text(
        "@echo off\r\ncd /d %~dp0\r\n"
        "if not exist .venv\\Scripts\\python.exe python -m venv .venv\r\n"
        ".venv\\Scripts\\python.exe -m pip install --upgrade pip\r\n"
        ".venv\\Scripts\\python.exe -m pip install -r requirements-final.txt\r\n",
        encoding="utf-8",
    )
    (dst / "CHECK_ONLY.bat").write_text(
        "@echo off\r\ncd /d %~dp0\r\n"
        "if not exist .venv\\Scripts\\python.exe call INSTALL.bat || exit /b 1\r\n"
        ".venv\\Scripts\\python.exe verify_from_test_clean_v1.py || exit /b 1\r\n"
        ".venv\\Scripts\\python.exe top40_allsessions_from_test_v1.py --check-only\r\n"
        "pause\r\n",
        encoding="utf-8",
    )
    (dst / "START_DEMO.bat").write_text(
        "@echo off\r\ncd /d %~dp0\r\n"
        "if not exist .venv\\Scripts\\python.exe call INSTALL.bat || exit /b 1\r\n"
        ".venv\\Scripts\\python.exe verify_from_test_clean_v1.py || exit /b 1\r\n"
        ".venv\\Scripts\\python.exe top40_allsessions_from_test_v1.py --enable-demo-orders\r\n"
        "pause\r\n",
        encoding="utf-8",
    )


def write_verifier(dst: Path) -> None:
    code = f'''from pathlib import Path\nimport hashlib, json\nROOT=Path(__file__).resolve().parent\nPOLICY=json.loads((ROOT/"FROM_TEST_POLICY.json").read_text(encoding="utf-8"))\ndef sha(p):\n h=hashlib.sha256()\n with p.open("rb") as f:\n  for c in iter(lambda:f.read(1024*1024),b""): h.update(c)\n return h.hexdigest()\nassert POLICY["s2_threshold"]=={THRESHOLD!r}\nassert POLICY["sequence_lengths"]=={{"m1":8,"m5":5,"m15":3}}\nassert POLICY["fixed_lot"]==0.01 and POLICY["max_positions"]==3\nassert POLICY["session_filter_enabled"] is False and POLICY["cooldown_enabled"] is False\nassert sha(ROOT/"artifacts/selected_short_memory_s2.pt")=={EXPECTED_MODEL_SHA!r}\nassert sha(ROOT/"artifacts/short_memory_scalers_s2.joblib")=={EXPECTED_SCALER_SHA!r}\nruntime=(ROOT/"top40_allsessions_from_test_v1.py").read_text(encoding="utf-8")\nassert "top40_demo_runtime" not in runtime\nassert "PENDING" in runtime and "immutable" in runtime.lower()\nassert "S2_THRESHOLD = 0.32696733474731443" in runtime\nseq=(ROOT/"xau_bot/research/ml/phase3b2_sequential.py").read_text(encoding="utf-8")\nassert "M1_LENGTH = 8" in seq and "M5_LENGTH = 5" in seq and "M15_LENGTH = 3" in seq\nrisk=(ROOT/"xau_bot/risk/risk_manager.py").read_text(encoding="utf-8")\nassert "def in_cooldown" in risk and "return False" in risk\nprint("FROM_TEST_CLEAN_VERIFY=PASS")\nprint("legacy_runtime_used=NO")\nprint("entry_contract=M5_ACCEPT_TO_PENDING_TO_NEXT_M1")\nprint("initial_risk_contract=IMMUTABLE")\nprint("sequence=8/5/3")\nprint("threshold=0.32696733474731443")\nprint("fixed_lot=0.01")\nprint("max_positions=3")\nprint("session_filter=OFF")\nprint("cooldown=OFF")\n'''
    (dst / "verify_from_test_clean_v1.py").write_text(code, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine-source", type=Path, default=Path(r"D:\ORO"))
    ap.add_argument("--short-source", type=Path, default=Path(r"D:\oro_top40_shortmem_OLD"))
    ap.add_argument("--out", type=Path, default=Path(r"D:\ORO_ALLSESSIONS_FROM_TEST_CLEAN"))
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    runtime_src = Path(__file__).resolve().parent / "top40_allsessions_from_test_v1.py"
    engine = args.engine_source.resolve()
    short = args.short_source.resolve()
    dst = args.out.resolve()

    if not runtime_src.exists():
        raise SystemExit(f"RUNTIME_SOURCE_MISSING: {runtime_src}")
    if not (engine / "xau_bot").exists():
        raise SystemExit(f"ENGINE_SOURCE_MISSING: {engine / 'xau_bot'}")

    art_src = short / "research_output" / "short_memory_v1"
    model = art_src / "selected_short_memory_s2.pt"
    scaler = art_src / "short_memory_scalers_s2.joblib"
    manifest = art_src / "short_memory_threshold_freeze_manifest.json"
    for p in (model, scaler, manifest):
        if not p.exists():
            raise SystemExit(f"FROZEN_ARTIFACT_MISSING: {p}")
    if sha256(model) != EXPECTED_MODEL_SHA:
        raise SystemExit(f"MODEL_HASH_MISMATCH actual={sha256(model)}")
    if sha256(scaler) != EXPECTED_SCALER_SHA:
        raise SystemExit(f"SCALER_HASH_MISMATCH actual={sha256(scaler)}")
    mf = json.loads(manifest.read_text(encoding="utf-8"))
    if not mf.get("strict_no_future_guard"):
        raise SystemExit("STRICT_NO_FUTURE_GUARD_MISSING")

    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    copytree_clean(engine / "xau_bot", dst / "xau_bot")
    shutil.copy2(runtime_src, dst / "top40_allsessions_from_test_v1.py")

    # Replace the canonical Phase-3 S2 artifact wherever the engine bundle still
    # contains the legacy frozen files. LiveS2Scorer imports those canonical paths.
    model_hits = find_by_hash(dst, LEGACY_MODEL_SHA)
    scaler_hits = find_by_hash(dst, LEGACY_SCALER_SHA)
    if not model_hits:
        raise SystemExit("LEGACY_MODEL_TARGET_NOT_FOUND_IN_ENGINE_COPY")
    if not scaler_hits:
        raise SystemExit("LEGACY_SCALER_TARGET_NOT_FOUND_IN_ENGINE_COPY")
    for p in model_hits:
        shutil.copy2(model, p)
    for p in scaler_hits:
        shutil.copy2(scaler, p)

    patch_sequence_lengths(dst)
    patch_no_cooldown(dst)
    patch_frozen_guard(dst, EXPECTED_MODEL_SHA, EXPECTED_SCALER_SHA)

    artifacts = dst / "artifacts"
    artifacts.mkdir(exist_ok=True)
    shutil.copy2(model, artifacts / "selected_short_memory_s2.pt")
    shutil.copy2(scaler, artifacts / "short_memory_scalers_s2.joblib")
    shutil.copy2(manifest, artifacts / "short_memory_threshold_freeze_manifest.json")

    policy = {
        "name": "TOP40_ALLSESSIONS_FROM_STRICT_TEST_CLEAN_V1",
        "source_contract": "STRICT replay executed-path semantics",
        "legacy_runtime_used": False,
        "sequence_lengths": {"m1": 8, "m5": 5, "m15": 3},
        "s2_threshold": THRESHOLD,
        "fixed_lot": 0.01,
        "max_positions": 3,
        "session_filter_enabled": False,
        "cooldown_enabled": False,
        "entry_semantics": "accepted closed-M5 candidate becomes pending; execute on next M1 cycle",
        "initial_risk_semantics": "immutable for lifetime of position",
        "strict_no_future_guard": True,
        "artifact_hashes": {"s2_model_sha256": EXPECTED_MODEL_SHA, "s2_scaler_sha256": EXPECTED_SCALER_SHA},
        "important_note": "The fixed-0.01 ALL-SESSIONS headline audit linearly rescaled the STRICT replay executed path; this live runtime follows that underlying STRICT decision/entry contract.",
    }
    (dst / "FROM_TEST_POLICY.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    (dst / "requirements-final.txt").write_text(
        "MetaTrader5\nnumpy>=1.26,<3\npandas>=2,<3\nscikit-learn==1.8.0\njoblib>=1.3\npyarrow>=15\ntorch==2.11.0\n",
        encoding="utf-8",
    )
    write_verifier(dst)
    write_launchers(dst)
    (dst / "README_FIRST.txt").write_text(
        "TOP40 ALL-SESSIONS FROM STRICT TEST - CLEAN V1\n\n"
        "This folder is independent from ORO_SHORT_MEMORY_ALLSESSIONS_FINAL.\n"
        "It does not use top40_demo_runtime.py.\n\n"
        "1. INSTALL.bat\n2. CHECK_ONLY.bat\n3. Only after PASS, START_DEMO.bat\n\n"
        "DEMO ONLY. Keep the old FINAL folder closed while validating this build.\n",
        encoding="utf-8",
    )

    # Syntax/structural verification with the current Python; runtime imports are
    # deliberately deferred until the destination venv is installed.
    subprocess.run([sys.executable, "-m", "py_compile", str(dst / "top40_allsessions_from_test_v1.py")], check=True)
    subprocess.run([sys.executable, str(dst / "verify_from_test_clean_v1.py")], cwd=dst, check=True)
    print("CLEAN_FROM_TEST_BUILD=PASS")
    print(f"OUTPUT={dst}")
    print(f"MODEL_TARGETS_REPLACED={len(model_hits)}")
    print(f"SCALER_TARGETS_REPLACED={len(scaler_hits)}")
    print("OLD_FINAL_FOLDER_USED=NO")
    print("LEGACY_RUNTIME_USED=NO")


if __name__ == "__main__":
    main()
