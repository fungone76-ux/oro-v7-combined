from __future__ import annotations

import argparse
import re
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Upgrade ALLSESSIONS FINAL talkative console to show S2 vs frozen threshold with margin")
    ap.add_argument("--target", type=Path, required=True)
    a = ap.parse_args()

    root = a.target.resolve()
    path = root / "talkative_demo_runtime.py"
    if not path.exists():
        raise RuntimeError(f"MISSING_TALKATIVE_RUNTIME: {path}")

    text = path.read_text(encoding="utf-8")

    # Replace any existing S2 explanation branch with a richer one when possible.
    old_pattern = re.compile(
        r'if "S2_SCORE" in u:\n(?:\s+.*\n){1,12}?\s+if m:\n(?:\s+.*\n){1,8}?',
        re.MULTILINE,
    )

    replacement = '''if "S2_SCORE" in u:\n        m = re.search(r"S2=([-+0-9.]+).*?threshold=([-+0-9.]+)", line, re.I)\n        if m:\n            score = float(m.group(1))\n            thr = float(m.group(2))\n            margin = score - thr\n            verdict = "PASS" if score >= thr else "FAIL"\n            sign = "+" if margin >= 0 else ""\n            out.append(\n                f"   ML -> S2={score:.6f} | soglia={thr:.6f} | "\n                f"margine={sign}{margin:.6f} | {verdict}"\n            )\n'''

    new_text, n = old_pattern.subn(replacement, text, count=1)

    if n == 0:
        # Conservative fallback: inject a second explanatory branch before TECH_CHECK.
        marker = '    if "TECH_CHECK" in u:\n'
        if marker not in text:
            raise RuntimeError("TALKATIVE_CONSOLE_STRUCTURE_NOT_RECOGNIZED")
        inject = '''    if "S2_SCORE" in u:\n        m2 = re.search(r"S2=([-+0-9.]+).*?threshold=([-+0-9.]+)", line, re.I)\n        if m2:\n            score = float(m2.group(1))\n            thr = float(m2.group(2))\n            margin = score - thr\n            verdict = "PASS" if score >= thr else "FAIL"\n            sign = "+" if margin >= 0 else ""\n            out.append(f"   ML -> S2={score:.6f} | soglia={thr:.6f} | margine={sign}{margin:.6f} | {verdict}")\n'''
        new_text = text.replace(marker, inject + marker, 1)

    path.write_text(new_text, encoding="utf-8")

    print("TALKATIVE_CONSOLE_V3=PASS")
    print("S2_THRESHOLD_COMPARISON=ENABLED")
    print("TRADING_LOGIC_CHANGED=NO")
    print(f"TARGET={root}")


if __name__ == "__main__":
    main()
