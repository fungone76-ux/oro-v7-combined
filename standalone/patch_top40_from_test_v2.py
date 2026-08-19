from __future__ import annotations

import argparse
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"PATCH_FAILED_{label}: expected block not found")
    return text.replace(old, new, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Patch TOP40 ALL-SESSIONS FROM TEST runtime to strict replay entry parity V2")
    ap.add_argument(
        "--target",
        type=Path,
        default=Path(r"D:\ORO_SHORT_MEMORY_ALLSESSIONS_FINAL\TOP40_ALLSESSIONS_FROM_TEST.py"),
    )
    args = ap.parse_args()
    path = args.target.resolve()
    if not path.exists():
        raise SystemExit(f"TARGET_NOT_FOUND: {path}")

    text = path.read_text(encoding="utf-8")
    original = text

    text = replace_once(
        text,
        "    s2_score: float\n\n\n@dataclass\nclass PositionRecord:",
        "    s2_score: float\n    frozen_risk: Any\n\n\n@dataclass\nclass PositionRecord:",
        "PENDING_FROZEN_RISK_FIELD",
    )

    old_execute = '''        snapshot = self._snapshot(m1, m5, m15, now)\n        # Recreate a minimal signal by regenerating at the original decision bar is\n        # unnecessary here: the replay stores the already-approved risk decision.\n        # We reconstruct only the broker prices around the preserved sl_points.\n        class SignalProxy:\n            direction = pending.direction\n            setup = pending.setup\n            score = pending.signal_score\n\n        # Use RiskManager only to rerun live safety guards at send time; preserve\n        # the historical stop distance from pending after approval.\n        m5_ind = IndicatorSet(m5.rename(columns={"tick_volume": "volume"}), self.cfg.indicators)\n        atr_points_now = float(m5_ind.atr.iloc[-1] / self.point)\n        safety_risk = self.risk_manager.evaluate(SignalProxy(), snapshot, atr_points_now)\n        if not safety_risk.approved:\n            print(f"PENDING_REJECT candidate={pending.candidate_id} reason={safety_risk.reason_code}", flush=True)\n            return\n        safety_risk = replace(safety_risk, sl_points=float(pending.sl_points))\n        risk = self._fixed_lot_risk(safety_risk, pending.direction, snapshot)\n'''
    new_execute = '''        snapshot = self._snapshot(m1, m5, m15, now)\n        # STRICT replay parity: risk was evaluated and approved on the M5 decision.\n        # The NEXT M1 consumes exactly that frozen RiskDecision.  No second\n        # RiskManager.evaluate(), no new ATR, and no extra live-only veto.\n        risk = self._fixed_lot_risk(pending.frozen_risk, pending.direction, snapshot)\n'''
    text = replace_once(text, old_execute, new_execute, "REMOVE_SECOND_RISK_EVALUATION")

    text = replace_once(
        text,
        '''            regime=snapshot.regime.value,\n            s2_score=float(score),\n        )\n''',
        '''            regime=snapshot.regime.value,\n            s2_score=float(score),\n            frozen_risk=risk,\n        )\n''',
        "STORE_FROZEN_RISK",
    )

    text = replace_once(
        text,
        '        latest_m5_close = latest_m5_open + pd.Timedelta(minutes=5)\n',
        '        latest_m5_close = latest_m5_open + pd.Timedelta("5min")\n',
        "TIMEDelta_WARNING",
    )

    old_startup = '''        print("TOP40 ALL-SESSIONS - REPLAY CONTRACT LIVE V1", flush=True)\n        print("SOURCE OF TRUTH: STRICT replay semantics; legacy runtime not used", flush=True)\n        print(f"symbol={self.connector.symbol} magic={MAGIC}", flush=True)\n        print(f"S2 threshold={S2_THRESHOLD:.15f} fixed_lot={FIXED_LOT:.2f} max_positions={MAX_POSITIONS}", flush=True)\n        print("sequence=8/5/3 | session_filter=OFF | cooldown=OFF", flush=True)\n        print(f"orders={'ENABLED_DEMO' if self.orders_enabled else 'DISABLED'}", flush=True)\n        print("ENTRY CONTRACT: accepted M5 candidate -> PENDING -> execute next M1 cycle", flush=True)\n        print("POSITION CONTRACT: immutable initial R -> BE/trailing via shared pure core", flush=True)\n'''
    new_startup = '''        print("TOP40 ALL-SESSIONS - REPLAY CONTRACT LIVE V2", flush=True)\n        print("FONTE: replay STRICT | vecchio runtime NON usato", flush=True)\n        print(f"SIMBOLO={self.connector.symbol} | magic={MAGIC}", flush=True)\n        print(f"ML S2: soglia={S2_THRESHOLD:.6f} | sequenze=8/5/3", flush=True)\n        print(f"RISCHIO: lotto fisso={FIXED_LOT:.2f} | max posizioni={MAX_POSITIONS} | cooldown=OFF", flush=True)\n        print("SESSIONI: ALL-SESSIONS (nessun filtro di sessione)", flush=True)\n        print(f"ORDINI: {'ATTIVI SU DEMO' if self.orders_enabled else 'DISATTIVATI'}", flush=True)\n        print("APERTURA: M5 valida -> tecnica -> S2 -> risk -> PENDING -> apertura M1 successiva", flush=True)\n        print("IMPORTANTE: sulla M1 successiva NON viene rifatto il risk check", flush=True)\n        print("GESTIONE: rischio iniziale immutabile -> BE a 1R -> trailing secondo core del test", flush=True)\n'''
    text = replace_once(text, old_startup, new_startup, "CLEAR_STARTUP_CONSOLE")

    old_m1 = '''        self.last_processed_m1 = latest_m1\n        print(f"M1_CLOSED {latest_m1.isoformat()} positions={self._open_count()}/{MAX_POSITIONS}", flush=True)\n\n        # EXACT replay ordering: pending from previous decision is consumed first,\n'''
    new_m1 = '''        self.last_processed_m1 = latest_m1\n        next_m5_close = pd.Timestamp(m5["time"].iloc[-1]) + pd.Timedelta("5min")\n        if next_m5_close == latest_m1:\n            phase = "M5 CHIUSA -> adesso controllo TECNICA e, se valida, ML S2"\n        else:\n            wait_min = max(0, int((next_m5_close - latest_m1).total_seconds() // 60))\n            phase = f"attendo prossima chiusura M5 (~{wait_min} min)"\n        pending_txt = (\n            f"SI {self.pending.direction.value} S2={self.pending.s2_score:.4f}"\n            if self.pending is not None else "NO"\n        )\n        print(\n            f"[{latest_m1.strftime('%H:%M')}] M1 ricevuta | posizioni={self._open_count()}/{MAX_POSITIONS} "\n            f"| pending={pending_txt} | {phase}",\n            flush=True,\n        )\n\n        # EXACT replay ordering: pending from previous decision is consumed first,\n'''
    text = replace_once(text, old_m1, new_m1, "CLEAR_M1_CONSOLE")

    old_no_trade = '''            print(\n                f"M5_DECISION {now.isoformat()} NO_TRADE tech_dir={signal.direction.value} "\n                f"reason={decision.reason_code}",\n                flush=True,\n            )\n'''
    new_no_trade = '''            print(\n                f"   TECNICA M5 -> NO TRADE | direzione={signal.direction.value} | motivo={decision.reason_code}",\n                flush=True,\n            )\n            print("   DECISIONE -> nessun candidato ML; aspetto la prossima M5", flush=True)\n'''
    text = replace_once(text, old_no_trade, new_no_trade, "CLEAR_NO_TRADE_CONSOLE")

    old_score = '''        print(\n            f"S2_SCORE candidate={cid} dir={signal.direction.value} setup={signal.setup} "\n            f"S2={score:.6f} threshold={S2_THRESHOLD:.6f} delta={margin:+.6f} "\n            f"{'PASS' if passed else 'NO_PASS'} infer_ms={latency_ms:.2f} "\n            f"latest_m1={tensor.latest_m1} latest_m5_close={tensor.latest_m5_close} latest_m15_close={tensor.latest_m15_close}",\n            flush=True,\n        )\n'''
    new_score = '''        print(\n            f"   TECNICA M5 -> CANDIDATO {signal.direction.value} | setup={signal.setup} | conferme={signal.n_confirmations}",\n            flush=True,\n        )\n        print(\n            f"   ML S2 -> valore={score:.6f} | soglia={S2_THRESHOLD:.6f} | scarto={margin:+.6f} "\n            f"| {'PASS' if passed else 'NO PASS'}",\n            flush=True,\n        )\n        print(\n            f"   DATI ML -> M1={tensor.latest_m1} | M5_close={tensor.latest_m5_close} | M15_close={tensor.latest_m15_close} "\n            f"| inferenza={latency_ms:.2f}ms",\n            flush=True,\n        )\n'''
    text = replace_once(text, old_score, new_score, "CLEAR_S2_CONSOLE")

    old_pending = '''        print(\n            f"PENDING_CREATED candidate={cid} dir={signal.direction.value} S2={score:.6f} "\n            f"sl_pts={risk.sl_points:.1f} -> execute on NEXT M1 cycle",\n            flush=True,\n        )\n'''
    new_pending = '''        print(\n            f"   RISK -> PASS | SL={risk.sl_points:.1f} punti | risk approvato sulla M5 e CONGELATO",\n            flush=True,\n        )\n        print(\n            f"   PENDING -> {signal.direction.value} pronto | S2={score:.6f} | apertura alla PROSSIMA M1",\n            flush=True,\n        )\n'''
    text = replace_once(text, old_pending, new_pending, "CLEAR_PENDING_CONSOLE")

    old_exec_print = '''        print(\n            f"PENDING_EXECUTE candidate={pending.candidate_id} dir={pending.direction.value} "\n            f"S2={pending.s2_score:.6f} entry_ref={risk.entry_price} sl_pts={risk.sl_points:.1f} "\n            f"SL={risk.sl_price} TP={risk.tp_price}",\n            flush=True,\n        )\n'''
    new_exec_print = '''        print("-" * 78, flush=True)\n        print(\n            f"APERTURA DA PENDING -> {pending.direction.value} | S2={pending.s2_score:.6f} "\n            f"| entry={risk.entry_price} | SL={risk.sl_price} | TP={risk.tp_price} | lotto={FIXED_LOT:.2f}",\n            flush=True,\n        )\n        print("   Nessun secondo risk-check: uso esattamente il risk approvato sulla M5", flush=True)\n'''
    text = replace_once(text, old_exec_print, new_exec_print, "CLEAR_EXEC_CONSOLE")

    if text == original:
        raise SystemExit("PATCH_FAILED_NO_CHANGES")

    backup = path.with_suffix(path.suffix + ".pre_v2.bak")
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    path.write_text(text, encoding="utf-8")

    print("TOP40_FROM_TEST_V2_PATCH=PASS")
    print("SECOND_RISK_EVALUATION_ON_NEXT_M1=REMOVED")
    print("M5_RISK_DECISION=FROZEN_IN_PENDING")
    print("CONSOLE_CLARITY=UPGRADED")
    print("TIMEDELTA_WARNING_FIX=APPLIED")
    print("S2_THRESHOLD_CHANGED=NO")
    print("TECHNICAL_LOGIC_CHANGED=NO")
    print("SEQUENCES_CHANGED=NO")
    print("POSITION_MANAGEMENT_CHANGED=NO")
    print(f"TARGET={path}")
    print(f"BACKUP={backup}")


if __name__ == "__main__":
    main()
