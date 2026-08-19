from __future__ import annotations

import argparse
import shutil
from pathlib import Path

MARKER = "ALLSESSIONS_LIVE_PARITY_FIX_V1"


def patch_position_manager(root: Path) -> None:
    path = root / "xau_bot" / "execution" / "position_manager.py"
    if not path.exists():
        raise RuntimeError(f"MISSING_POSITION_MANAGER: {path}")

    text = path.read_text(encoding="utf-8-sig")
    if MARKER in text:
        print("POSITION_MANAGER_FIX=ALREADY_APPLIED")
        return

    old = '''        record = self.position_risk_records.get(position.ticket)\n        if record is not None and record.initial_risk_points > 0:\n            entry_price = record.initial_entry_price\n            risk_points = record.initial_risk_points\n        else:\n            entry_price = position.price_open\n            risk_points = abs(position.price_open - position.sl) / point if position.sl else 0.0\n'''

    new = '''        record = self.position_risk_records.get(position.ticket)\n        if record is not None and record.initial_risk_points > 0:\n            entry_price = record.initial_entry_price\n            risk_points = record.initial_risk_points\n        else:\n            # ALLSESSIONS_LIVE_PARITY_FIX_V1\n            # Never infer INITIAL risk from the CURRENT SL after the position has\n            # been moved to breakeven/trailing. That collapses (e.g. 130 points\n            # -> 2 points) and corrupts every subsequent R-multiple calculation.\n            entry_price = position.price_open\n            rr = float(getattr(self.cfg, "tp_risk_reward", 1.5) or 1.5)\n            if position.tp and rr > 0:\n                # TP is not moved by the position manager. Reconstruct the\n                # original risk from the frozen initial TP distance. This also\n                # survives a process restart when the in-memory risk record is\n                # unavailable.\n                risk_points = abs(position.tp - entry_price) / (rr * point)\n            elif position.sl:\n                # Safe only before any protective SL move. If SL is already at\n                # or beyond entry and TP is unavailable, do NOT fabricate a tiny\n                # initial risk from the protective stop.\n                protective = (\n                    (position.direction == Direction.LONG and position.sl >= entry_price) or\n                    (position.direction == Direction.SHORT and position.sl <= entry_price)\n                )\n                risk_points = 0.0 if protective else abs(entry_price - position.sl) / point\n            else:\n                risk_points = 0.0\n'''

    if old not in text:
        raise RuntimeError("POSITION_MANAGER_FALLBACK_BLOCK_NOT_FOUND")

    backup = path.with_suffix(path.suffix + ".pre_live_parity_fix_v1.bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    print("POSITION_MANAGER_FIX=PASS")


def write_verifier(root: Path) -> None:
    verifier = root / "verify_live_parity_fix_v1.py"
    verifier.write_text(r'''from __future__ import annotations

from types import SimpleNamespace

from xau_bot.config.settings import RiskConfig
from xau_bot.core.models import Direction
from xau_bot.execution.position_manager import PositionManager


class DummyConnector:
    symbol_spec = SimpleNamespace(point=0.01, digits=2)


cfg = RiskConfig()
pm = PositionManager(DummyConnector(), cfg, position_risk_records={})

# SHORT example from the observed failure class: initial risk ~130 points,
# TP stays unchanged, while current SL is already BE+2 points.
pos = SimpleNamespace(
    ticket=999001,
    direction=Direction.SHORT,
    price_open=4366.62,
    sl=4366.60,
    tp=4364.67,  # exactly 1.5 * 1.30 below entry
    open_time=None,
)
state = pm._position_state(pos, 0.01, 2)
assert abs(state.initial_risk_points - 130.0) < 1e-6, state.initial_risk_points
assert state.initial_risk_points != 2.0

# Before BE, fallback remains consistent too.
pos2 = SimpleNamespace(
    ticket=999002,
    direction=Direction.SHORT,
    price_open=4366.62,
    sl=4367.92,
    tp=4364.67,
    open_time=None,
)
state2 = pm._position_state(pos2, 0.01, 2)
assert abs(state2.initial_risk_points - 130.0) < 1e-6, state2.initial_risk_points

print("LIVE_PARITY_FIX_V1=PASS")
print("INITIAL_RISK_BEFORE_BE=130.0")
print("INITIAL_RISK_AFTER_BE=130.0")
print("RISK_COLLAPSE_TO_2_POINTS=BLOCKED")
''', encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fix ALL-SESSIONS live/replay initial-risk parity after BE/trailing")
    ap.add_argument("--target", type=Path, required=True)
    args = ap.parse_args()

    root = args.target.resolve()
    if not (root / "top40_demo_runtime.py").exists():
        raise RuntimeError(f"NOT_AN_ALLSESSIONS_FINAL_FOLDER: {root}")

    patch_position_manager(root)
    write_verifier(root)

    print("ALLSESSIONS_LIVE_PARITY_FIX_V1=PASS")
    print("S2_THRESHOLD_CHANGED=NO")
    print("SEQUENCES_CHANGED=NO")
    print("ENTRY_LOGIC_CHANGED=NO")
    print("SL_TP_POLICY_CHANGED=NO")
    print("FIX=INITIAL_RISK_PERSISTS_AFTER_BREAKEVEN_OR_TRAILING")
    print(f"TARGET={root}")


if __name__ == "__main__":
    main()
