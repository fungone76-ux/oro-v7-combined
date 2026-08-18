from __future__ import annotations

import argparse
from pathlib import Path


MARKER = "POSITION_AUDIT_V1"


def main() -> None:
    ap = argparse.ArgumentParser(description="Add position-management parity diagnostics without changing trading rules")
    ap.add_argument("--target", type=Path, required=True)
    a = ap.parse_args()

    root = a.target.resolve()
    pm = root / "xau_bot" / "execution" / "position_manager.py"
    if not pm.exists():
        raise RuntimeError(f"MISSING_POSITION_MANAGER: {pm}")

    text = pm.read_text(encoding="utf-8-sig")
    if MARKER in text:
        print("POSITION_MANAGEMENT_AUDIT_V1=ALREADY_APPLIED")
        print("TRADING_LOGIC_CHANGED=NO")
        print(f"TARGET={root}")
        return

    old = '''        action = decide_position_action(state, market, self.cfg, atr_points)\n        if action.action != PositionActionType.MOVE_STOP or action.new_sl is None:\n            return\n'''

    new = '''        action = decide_position_action(state, market, self.cfg, atr_points)\n\n        # POSITION_AUDIT_V1: diagnostics only. This does not alter the action.\n        current_price = snapshot.bid if position.direction == Direction.LONG else snapshot.ask\n        current_r = calculate_r_multiple(state, current_price)\n        print(\n            f"POSITION_AUDIT ticket={position.ticket} dir={position.direction.value} "\n            f"entry={state.entry_price:.2f} current={current_price:.2f} "\n            f"initial_risk_pts={state.initial_risk_points:.1f} R={current_r:.4f} "\n            f"be_at_R={self.cfg.breakeven_at_r:.2f} "\n            f"trail_at_R={self.cfg.trailing_activation_r:.2f} "\n            f"be_offset_pts={self.cfg.breakeven_offset_points} "\n            f"trail_atr_mult={self.cfg.trailing_atr_multiplier:.2f} "\n            f"atr_pts={atr_points:.1f} current_sl={state.current_sl:.2f} "\n            f"tp={state.current_tp:.2f} action={action.action.value} "\n            f"reason={action.reason.value} proposed_sl={action.new_sl if action.new_sl is not None else 'NONE'}",\n            flush=True,\n        )\n\n        if action.action != PositionActionType.MOVE_STOP or action.new_sl is None:\n            return\n'''

    if old not in text:
        raise RuntimeError("POSITION_MANAGER_STRUCTURE_NOT_RECOGNIZED")

    pm.write_text(text.replace(old, new, 1), encoding="utf-8")

    print("POSITION_MANAGEMENT_AUDIT_V1=PASS")
    print("LIVE_REPLAY_MATH_VISIBILITY=ENABLED")
    print("TRADING_LOGIC_CHANGED=NO")
    print("SL_TP_RULES_CHANGED=NO")
    print(f"TARGET={root}")


if __name__ == "__main__":
    main()
