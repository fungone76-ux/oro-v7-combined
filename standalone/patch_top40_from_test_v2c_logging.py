from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"PATCH_FAILED_{label}: expected block not found")
    return text.replace(old, new, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Patch TOP40 FROM TEST V2b with concise position logging and explicit close summaries")
    ap.add_argument("--target", type=Path, required=True)
    args = ap.parse_args()

    path = args.target.resolve()
    if not path.exists():
        raise SystemExit(f"TARGET_NOT_FOUND: {path}")

    text = path.read_text(encoding="utf-8-sig")
    original = text
    backup = path.with_suffix(path.suffix + ".pre_v2c.bak")
    shutil.copy2(path, backup)

    text = replace_once(
        text,
        "        self.records: dict[int, PositionRecord] = {}\n        self.last_processed_m1: pd.Timestamp | None = None\n",
        "        self.records: dict[int, PositionRecord] = {}\n        self.last_position_audit_mono: dict[int, float] = {}\n        self.last_processed_m1: pd.Timestamp | None = None\n",
        "AUDIT_THROTTLE_STATE",
    )

    marker = "    def manage_positions(self) -> None:\n"
    close_method = '''    def _log_closed_position(self, rec: PositionRecord) -> None:\n        \"\"\"Print one explicit MT5 history summary when a tracked position disappears.\"\"\"\n        import MetaTrader5 as mt5  # type: ignore\n\n        try:\n            deals = mt5.history_deals_get(position=int(rec.ticket))\n        except Exception as exc:\n            deals = None\n            history_error = f\"{type(exc).__name__}: {exc}\"\n        else:\n            history_error = None\n\n        print(\"=\" * 78, flush=True)\n        print(f\"CHIUSURA POSIZIONE ticket={rec.ticket} | {rec.direction.value}\", flush=True)\n        print(f\"   ENTRY FILL={rec.entry_price} | rischio iniziale={rec.initial_risk_points:.1f} pt | S2={rec.s2_score:.6f}\", flush=True)\n\n        if not deals:\n            print(\"   EXIT/P&L -> non ancora disponibile nella history MT5\", flush=True)\n            if history_error:\n                print(f\"   HISTORY ERROR -> {history_error}\", flush=True)\n            print(\"=\" * 78, flush=True)\n            return\n\n        rows = list(deals)\n        out_entries = {\n            int(getattr(mt5, \"DEAL_ENTRY_OUT\", -101)),\n            int(getattr(mt5, \"DEAL_ENTRY_OUT_BY\", -102)),\n            int(getattr(mt5, \"DEAL_ENTRY_INOUT\", -103)),\n        }\n        closing = [d for d in rows if int(getattr(d, \"entry\", -999)) in out_entries]\n        pool = closing if closing else rows\n        close_deal = max(\n            pool,\n            key=lambda d: int(getattr(d, \"time_msc\", 0) or 0)\n            or int(getattr(d, \"time\", 0) or 0) * 1000,\n        )\n\n        exit_price = float(getattr(close_deal, \"price\", float(\"nan\")))\n        net_pnl = 0.0\n        for d in rows:\n            net_pnl += float(getattr(d, \"profit\", 0.0) or 0.0)\n            net_pnl += float(getattr(d, \"commission\", 0.0) or 0.0)\n            net_pnl += float(getattr(d, \"swap\", 0.0) or 0.0)\n            net_pnl += float(getattr(d, \"fee\", 0.0) or 0.0)\n\n        reason = int(getattr(close_deal, \"reason\", -999))\n        if reason == int(getattr(mt5, \"DEAL_REASON_SL\", -1001)):\n            outcome = \"STOP LOSS\"\n        elif reason == int(getattr(mt5, \"DEAL_REASON_TP\", -1002)):\n            outcome = \"TAKE PROFIT\"\n        else:\n            outcome = \"CHIUSURA MT5\"\n\n        risk_price = rec.initial_risk_points * self.point\n        if risk_price > 0 and exit_price == exit_price:\n            if rec.direction == Direction.LONG:\n                realized_r = (exit_price - rec.entry_price) / risk_price\n            else:\n                realized_r = (rec.entry_price - exit_price) / risk_price\n            r_text = f\"{realized_r:+.3f}R\"\n        else:\n            r_text = \"n/d\"\n\n        duration_text = \"n/d\"\n        try:\n            opened = datetime.fromisoformat(rec.opened_at_iso)\n            close_sec = int(getattr(close_deal, \"time\", 0) or 0)\n            if close_sec > 0:\n                closed = datetime.fromtimestamp(close_sec, tz=timezone.utc)\n                seconds = (closed - opened).total_seconds()\n                if 0 <= seconds <= 7 * 86400:\n                    mins = int(seconds // 60)\n                    secs = int(seconds % 60)\n                    duration_text = f\"{mins}m {secs}s\"\n        except Exception:\n            pass\n\n        print(f\"   EXIT={exit_price} | ESITO={outcome} | R prezzo={r_text}\", flush=True)\n        print(f\"   P&L NETTO MT5={net_pnl:+.2f} | durata={duration_text}\", flush=True)\n        print(\"=\" * 78, flush=True)\n\n'''
    if marker not in text:
        raise SystemExit("PATCH_FAILED_CLOSE_METHOD_MARKER")
    text = text.replace(marker, close_method + marker, 1)

    text = replace_once(
        text,
        '''        for ticket in list(self.records):\n            if ticket not in open_ids:\n                del self.records[ticket]\n''',
        '''        for ticket in list(self.records):\n            if ticket not in open_ids:\n                rec = self.records[ticket]\n                self._log_closed_position(rec)\n                self.last_position_audit_mono.pop(ticket, None)\n                del self.records[ticket]\n''',
        "EXPLICIT_CLOSE_LOG",
    )

    old_audit = '''            print(\n                f\"POSITION_AUDIT ticket={ticket} dir={rec.direction.value} entry={rec.entry_price} \"\n                f\"initial_risk_pts={rec.initial_risk_points:.1f} R={action.r_multiple:.4f} \"\n                f\"current_sl={current_sl} tp={current_tp} action={action.action.value} \"\n                f\"reason={action.reason.value} proposed_sl={action.new_sl}\",\n                flush=True,\n            )\n            if action.action == PositionActionType.MOVE_STOP and action.new_sl is not None:\n                self._modify_sl(ticket, action.new_sl, current_tp)\n'''
    new_audit = '''            now_mono = time.monotonic()\n            last_audit = self.last_position_audit_mono.get(ticket, 0.0)\n            important_action = action.action != PositionActionType.HOLD\n            if important_action or (now_mono - last_audit) >= 10.0:\n                print(\n                    f\"   POSIZIONE #{ticket} {rec.direction.value} | entry={rec.entry_price} | R={action.r_multiple:+.3f} \"\n                    f\"| SL={current_sl} | TP={current_tp} | azione={action.action.value} | motivo={action.reason.value}\",\n                    flush=True,\n                )\n                self.last_position_audit_mono[ticket] = now_mono\n            if action.action == PositionActionType.MOVE_STOP and action.new_sl is not None:\n                print(f\"   GESTIONE -> modifica SL immediata proposta={action.new_sl}\", flush=True)\n                self._modify_sl(ticket, action.new_sl, current_tp)\n'''
    text = replace_once(text, old_audit, new_audit, "THROTTLE_POSITION_AUDIT")

    if text == original:
        raise SystemExit("PATCH_FAILED_NO_CHANGES")

    path.write_text(text, encoding="utf-8")

    check = path.read_text(encoding="utf-8")
    required = [
        "S2_THRESHOLD = 0.32696733474731443",
        "FIXED_LOT = 0.01",
        "MAX_POSITIONS = 3",
        "frozen_risk=risk",
        "Nessun secondo risk-check",
        "CHIUSURA POSIZIONE",
        "last_position_audit_mono",
    ]
    missing = [x for x in required if x not in check]
    if missing:
        shutil.copy2(backup, path)
        raise SystemExit(f"STRUCTURAL_GUARD_FAIL missing={missing}")

    print("TOP40_FROM_TEST_V2C_LOGGING=PASS")
    print("POSITION_AUDIT_INTERVAL_SECONDS=10")
    print("IMPORTANT_POSITION_ACTIONS=IMMEDIATE")
    print("EXPLICIT_CLOSE_SUMMARY=ENABLED")
    print("MT5_HISTORY_PNL=ENABLED")
    print("TRADING_LOGIC_CHANGED=NO")
    print("S2_THRESHOLD_CHANGED=NO")
    print("ENTRY_CONTRACT_CHANGED=NO")
    print("SL_TP_POLICY_CHANGED=NO")
    print("POSITION_MANAGEMENT_RULES_CHANGED=NO")
    print(f"TARGET={path}")
    print(f"BACKUP={backup}")


if __name__ == "__main__":
    main()
