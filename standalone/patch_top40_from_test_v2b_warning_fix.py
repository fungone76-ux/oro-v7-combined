from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Remove Pandas/NumPy timedelta deprecation warnings from TOP40 FROM TEST V2 without changing trading logic")
    ap.add_argument("--target", type=Path, required=True)
    args = ap.parse_args()

    target = args.target.resolve()
    if not target.exists():
        raise SystemExit(f"TARGET_NOT_FOUND: {target}")

    text = target.read_text(encoding="utf-8-sig")
    original = text

    backup = target.with_suffix(target.suffix + ".pre_v2b.bak")
    shutil.copy2(target, backup)

    text = text.replace(
        "from datetime import datetime, timezone",
        "from datetime import datetime, timezone, timedelta",
    )
    text = text.replace('pd.Timedelta("5min")', 'timedelta(minutes=5)')
    text = text.replace("pd.Timedelta('5min')", "timedelta(minutes=5)")
    text = text.replace("pd.Timedelta(minutes=5)", "timedelta(minutes=5)")

    if text == original:
        raise SystemExit("PATCH_NOT_APPLIED: expected timedelta expressions not found or already fixed")

    if "pd.Timedelta(" in text:
        # Be conservative: do not fail on unrelated uses, but report them.
        remaining = text.count("pd.Timedelta(")
    else:
        remaining = 0

    target.write_text(text, encoding="utf-8")

    # Structural guards: trading contract must remain untouched.
    check = target.read_text(encoding="utf-8")
    required = [
        "S2_THRESHOLD = 0.32696733474731443",
        "FIXED_LOT = 0.01",
        "MAX_POSITIONS = 3",
        "PENDING",
        "initial_risk_points",
    ]
    missing = [x for x in required if x not in check]
    if missing:
        shutil.copy2(backup, target)
        raise SystemExit(f"STRUCTURAL_GUARD_FAIL missing={missing}")

    print("TOP40_FROM_TEST_V2B_WARNING_FIX=PASS")
    print("TIMEDELTA_IMPLEMENTATION=datetime.timedelta")
    print("TRADING_LOGIC_CHANGED=NO")
    print("S2_THRESHOLD_CHANGED=NO")
    print("ENTRY_CONTRACT_CHANGED=NO")
    print("POSITION_MANAGEMENT_CHANGED=NO")
    print(f"REMAINING_PD_TIMEDELTA_CALLS={remaining}")
    print(f"TARGET={target}")
    print(f"BACKUP={backup}")


if __name__ == "__main__":
    main()
