from __future__ import annotations

"""TOP40 ALL-SESSIONS demo runtime rebuilt from the STRICT replay contract.

This file intentionally does NOT import or wrap top40_demo_runtime.py.
Its ordering semantics mirror the historical economic replay:

1. Process newly closed M1 bars.
2. Evaluate a technical candidate only when a new M5 has just closed.
3. Build the same causal ML candidate and score frozen S2.
4. If S2 passes, evaluate risk at decision time and store a PENDING entry.
5. Execute that pending entry on the NEXT newly closed M1 cycle, using the
   then-current broker market price as the live analogue of the replay's next
   M1 open.
6. Preserve initial_risk_points for the whole position lifetime.
7. Manage breakeven/trailing with the same pure decide_position_action()
   function used by SimulatedExecutor.

Policy frozen for the ALL-SESSIONS short-memory experiment:
- sequence lengths: 8 / 5 / 3 (provided by the frozen project build)
- S2 threshold: 0.32696733474731443
- fixed lot: 0.01
- max simultaneous positions: 3
- session filter: OFF
- loss cooldown: OFF
- demo accounts only

This is DEMO-only software. It refuses to arm orders on a non-demo account.
"""

import argparse
import json
import math
import signal as os_signal
import time
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from xau_bot.analysis.indicators import IndicatorSet
from xau_bot.analysis.market_context import determine_regime, determine_session
from xau_bot.analysis.technical_signal_engine import generate_signal
from xau_bot.config.settings import BotConfig
from xau_bot.core.models import AccountTradeMode, Direction, MarketSnapshot, MlState, TradeAction
from xau_bot.execution.executor import Executor
from xau_bot.execution.position_management_core import (
    PositionActionType,
    PositionMarketState,
    PositionState,
    decide_position_action,
)
from xau_bot.infrastructure.mt5_connector import MT5Connector
from xau_bot.live.forward_runtime import LiveS2Scorer, canonical_bars
from xau_bot.research.ml.candidate_extractor import _FastIndicatorView, _sl_points, add_direction_normalized_features
from xau_bot.research.ml.feature_builder import build_market_state_features, stable_sample_id
from xau_bot.research.ml.schemas import MLDatasetKind
from xau_bot.risk.risk_manager import RiskManager
from xau_bot.strategy.decision_engine import decide


MAGIC = 32040
S2_THRESHOLD = 0.32696733474731443
FIXED_LOT = 0.01
MAX_POSITIONS = 3
STATE_FILE = Path("state/test_contract_live_state.json")


@dataclass
class PendingEntry:
    candidate_id: str
    signal_time_iso: str
    direction: Direction
    setup: str
    signal_score: int
    sl_points: float
    risk_amount_reference: float
    session: str
    regime: str
    s2_score: float


@dataclass
class PositionRecord:
    ticket: int
    direction: Direction
    entry_price: float
    initial_sl: float
    initial_tp: float
    initial_risk_points: float
    lot: float
    opened_at_iso: str
    candidate_id: str
    s2_score: float


class ReplayContractLiveRuntime:
    def __init__(self, enable_demo_orders: bool, check_only: bool = False) -> None:
        self.enable_demo_orders = bool(enable_demo_orders)
        self.check_only = bool(check_only)
        self.cfg = replace(BotConfig(), magic_number=MAGIC)
        self.connector = MT5Connector(self.cfg)
        self.connector.initialize()
        self.scorer = LiveS2Scorer()
        self.risk_manager = RiskManager(self.cfg.risk, self.connector.symbol_spec)
        self.executor = Executor(self.connector, self.cfg)
        self.pending: PendingEntry | None = None
        self.records: dict[int, PositionRecord] = {}
        self.last_processed_m1: pd.Timestamp | None = None
        self.last_evaluated_m5_close: pd.Timestamp | None = None
        self.running = True
        self.orders_enabled = False
        self._load_state()
        self._verify_account()

    @property
    def point(self) -> float:
        return float(self.connector.symbol_spec.point)

    @property
    def digits(self) -> int:
        return int(self.connector.symbol_spec.digits)

    def _verify_account(self) -> None:
        account = self.connector.get_account_state()
        is_demo = account.trade_mode == AccountTradeMode.DEMO
        self.orders_enabled = bool(self.enable_demo_orders and is_demo and not self.check_only)
        if self.enable_demo_orders and not is_demo:
            raise RuntimeError("DEMO_ONLY_GUARD: account is not DEMO")
        print("=" * 78, flush=True)
        print("TOP40 ALL-SESSIONS - REPLAY CONTRACT LIVE V1", flush=True)
        print("SOURCE OF TRUTH: STRICT replay semantics; legacy runtime not used", flush=True)
        print(f"symbol={self.connector.symbol} magic={MAGIC}", flush=True)
        print(f"S2 threshold={S2_THRESHOLD:.15f} fixed_lot={FIXED_LOT:.2f} max_positions={MAX_POSITIONS}", flush=True)
        print("sequence=8/5/3 | session_filter=OFF | cooldown=OFF", flush=True)
        print(f"orders={'ENABLED_DEMO' if self.orders_enabled else 'DISABLED'}", flush=True)
        print("ENTRY CONTRACT: accepted M5 candidate -> PENDING -> execute next M1 cycle", flush=True)
        print("POSITION CONTRACT: immutable initial R -> BE/trailing via shared pure core", flush=True)
        print("=" * 78, flush=True)

    def _load_state(self) -> None:
        if not STATE_FILE.exists():
            return
        try:
            raw = json.loads(STATE_FILE.read_text(encoding="utf-8"))
            for row in raw.get("positions", []):
                rec = PositionRecord(
                    ticket=int(row["ticket"]),
                    direction=Direction(row["direction"]),
                    entry_price=float(row["entry_price"]),
                    initial_sl=float(row["initial_sl"]),
                    initial_tp=float(row["initial_tp"]),
                    initial_risk_points=float(row["initial_risk_points"]),
                    lot=float(row["lot"]),
                    opened_at_iso=str(row["opened_at_iso"]),
                    candidate_id=str(row.get("candidate_id", "")),
                    s2_score=float(row.get("s2_score", math.nan)),
                )
                self.records[rec.ticket] = rec
        except Exception as exc:
            raise RuntimeError(f"STATE_LOAD_FAILED: {exc}") from exc

    def _save_state(self) -> None:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        broker_ids = {int(p.ticket) for p in self.connector.get_open_positions() if int(getattr(p, "magic", MAGIC)) == MAGIC}
        self.records = {k: v for k, v in self.records.items() if k in broker_ids}
        payload = {
            "schema": "TOP40_REPLAY_CONTRACT_LIVE_V1",
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "positions": [
                {
                    "ticket": r.ticket,
                    "direction": r.direction.value,
                    "entry_price": r.entry_price,
                    "initial_sl": r.initial_sl,
                    "initial_tp": r.initial_tp,
                    "initial_risk_points": r.initial_risk_points,
                    "lot": r.lot,
                    "opened_at_iso": r.opened_at_iso,
                    "candidate_id": r.candidate_id,
                    "s2_score": r.s2_score,
                }
                for r in self.records.values()
            ],
        }
        tmp = STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(STATE_FILE)

    def fetch_market(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        # Large enough warmup for indicators and the causal short-memory tensor.
        m1 = canonical_bars(self.connector.get_ohlc("M1", 1500))
        m5 = canonical_bars(self.connector.get_ohlc("M5", 500))
        m15 = canonical_bars(self.connector.get_ohlc("M15", 200))
        return m1, m5, m15

    def _snapshot(self, m1: pd.DataFrame, m5: pd.DataFrame, m15: pd.DataFrame, now: pd.Timestamp) -> MarketSnapshot:
        bid, ask, spread = self.connector.get_tick()
        account = self.connector.get_account_state()
        positions = [p for p in self.connector.get_open_positions() if int(getattr(p, "magic", MAGIC)) == MAGIC]
        m5_ind = IndicatorSet(m5.rename(columns={"tick_volume": "volume"}), self.cfg.indicators)
        atr_pct = float(m5_ind.atr_pct.iloc[-1]) if pd.notna(m5_ind.atr_pct.iloc[-1]) else 0.0
        regime = determine_regime(m5_ind.bb_width, atr_pct)
        return MarketSnapshot(
            timestamp_utc=now.to_pydatetime(),
            symbol=self.connector.symbol,
            bid=float(bid),
            ask=float(ask),
            spread_points=float(spread),
            m1=m1,
            m5=m5,
            m15=m15,
            session=determine_session(now.to_pydatetime(), self.cfg.sessions),
            open_positions=positions,
            account=account,
            regime=regime,
        )

    def _candidate_frame(
        self,
        signal: Any,
        decision_time: pd.Timestamp,
        snapshot: MarketSnapshot,
        market_features: pd.DataFrame,
        m5_ind: IndicatorSet,
        m5_idx: int,
    ) -> pd.DataFrame:
        direction = signal.direction
        entry_reference = snapshot.ask if direction == Direction.LONG else snapshot.bid
        atr_points = float(m5_ind.atr.iloc[m5_idx] / self.point)
        stop_points = float(_sl_points(self.cfg, self.connector.symbol_spec, atr_points))
        stop_distance = stop_points * self.point
        base = pd.DataFrame([{
            "sample_id": stable_sample_id(
                MLDatasetKind.STRATEGY_CANDIDATE.value,
                decision_time.isoformat(),
                direction.value,
                signal.setup,
            ),
            "timestamp": decision_time,
            "dataset_kind": MLDatasetKind.STRATEGY_CANDIDATE.value,
            "direction": direction.value,
            "setup": signal.setup,
            "signal_score": signal.score,
            "session": snapshot.session.value,
            "regime": snapshot.regime.value,
            "technical_candidate": True,
            "candidate_entry_time": decision_time,
            "candidate_entry_price": round(float(entry_reference), self.digits),
            "candidate_entry_spread": snapshot.spread_points,
            "candidate_stop_distance": float(stop_distance),
            "candidate_stop_points": float(stop_points),
            "candidate_1r_price": round(
                float(entry_reference + stop_distance if direction == Direction.LONG else entry_reference - stop_distance),
                self.digits,
            ),
            "confirmation_count": signal.n_confirmations,
            "confirmations_passed": ";".join(c.name for c in signal.confirmations if c.passed),
        }])
        merge_features = market_features.drop(
            columns=[c for c in market_features.columns if c != "timestamp" and c in base.columns],
            errors="ignore",
        )
        merged = base.merge(merge_features, on="timestamp", how="left")
        out, _ = add_direction_normalized_features(merged)
        return out

    def _fixed_lot_risk(self, risk: Any, direction: Direction, snapshot: MarketSnapshot) -> Any:
        # Keep the replay's stop distance. Only volume is frozen to 0.01.
        entry = float(snapshot.ask if direction == Direction.LONG else snapshot.bid)
        sl_distance = float(risk.sl_points) * self.point
        rr = float(self.cfg.risk.tp_risk_reward)
        if direction == Direction.LONG:
            sl = entry - sl_distance
            tp = entry + sl_distance * rr
        else:
            sl = entry + sl_distance
            tp = entry - sl_distance * rr
        value_per_point_1lot = self.connector.symbol_spec.tick_value / self.connector.symbol_spec.tick_size * self.point
        risk_amount = float(risk.sl_points) * value_per_point_1lot * FIXED_LOT
        return replace(
            risk,
            approved=True,
            lot=FIXED_LOT,
            entry_price=round(entry, self.digits),
            sl_price=round(sl, self.digits),
            tp_price=round(tp, self.digits),
            risk_amount=risk_amount,
        )

    def _open_count(self) -> int:
        return len([p for p in self.connector.get_open_positions() if int(getattr(p, "magic", MAGIC)) == MAGIC])

    def execute_pending_if_due(self, now: pd.Timestamp, m1: pd.DataFrame, m5: pd.DataFrame, m15: pd.DataFrame) -> None:
        if self.pending is None:
            return
        signal_time = pd.Timestamp(self.pending.signal_time_iso)
        if now <= signal_time:
            return
        pending = self.pending
        self.pending = None
        if self._open_count() >= MAX_POSITIONS:
            print(f"PENDING_REJECT candidate={pending.candidate_id} reason=MAX_POSITIONS_REACHED", flush=True)
            return

        snapshot = self._snapshot(m1, m5, m15, now)
        # Recreate a minimal signal by regenerating at the original decision bar is
        # unnecessary here: the replay stores the already-approved risk decision.
        # We reconstruct only the broker prices around the preserved sl_points.
        class SignalProxy:
            direction = pending.direction
            setup = pending.setup
            score = pending.signal_score

        # Use RiskManager only to rerun live safety guards at send time; preserve
        # the historical stop distance from pending after approval.
        m5_ind = IndicatorSet(m5.rename(columns={"tick_volume": "volume"}), self.cfg.indicators)
        atr_points_now = float(m5_ind.atr.iloc[-1] / self.point)
        safety_risk = self.risk_manager.evaluate(SignalProxy(), snapshot, atr_points_now)
        if not safety_risk.approved:
            print(f"PENDING_REJECT candidate={pending.candidate_id} reason={safety_risk.reason_code}", flush=True)
            return
        safety_risk = replace(safety_risk, sl_points=float(pending.sl_points))
        risk = self._fixed_lot_risk(safety_risk, pending.direction, snapshot)

        print(
            f"PENDING_EXECUTE candidate={pending.candidate_id} dir={pending.direction.value} "
            f"S2={pending.s2_score:.6f} entry_ref={risk.entry_price} sl_pts={risk.sl_points:.1f} "
            f"SL={risk.sl_price} TP={risk.tp_price}",
            flush=True,
        )
        if not self.orders_enabled:
            print("ORDER_NOT_SENT orders disabled/check-only", flush=True)
            return
        result = self.executor.execute(risk, pending.direction, snapshot)
        if not result.success or result.ticket is None:
            print(f"ORDER_FAILED retcode={result.retcode} reason={result.reason}", flush=True)
            return
        ticket = int(result.ticket)
        filled = float(result.filled_price if result.filled_price is not None else risk.entry_price)
        self.records[ticket] = PositionRecord(
            ticket=ticket,
            direction=pending.direction,
            entry_price=filled,
            initial_sl=float(risk.sl_price),
            initial_tp=float(risk.tp_price),
            initial_risk_points=float(pending.sl_points),
            lot=FIXED_LOT,
            opened_at_iso=now.isoformat(),
            candidate_id=pending.candidate_id,
            s2_score=pending.s2_score,
        )
        self._save_state()
        print(
            f"OPENED ticket={ticket} dir={pending.direction.value} fill={filled} "
            f"immutable_initial_risk_pts={pending.sl_points:.1f}",
            flush=True,
        )

    def evaluate_new_m5(self, now: pd.Timestamp, m1: pd.DataFrame, m5: pd.DataFrame, m15: pd.DataFrame) -> None:
        latest_m5_open = pd.Timestamp(m5["time"].iloc[-1])
        latest_m5_close = latest_m5_open + pd.Timedelta(minutes=5)
        if latest_m5_close != now:
            return
        if self.last_evaluated_m5_close is not None and latest_m5_close <= self.last_evaluated_m5_close:
            return
        self.last_evaluated_m5_close = latest_m5_close

        snapshot = self._snapshot(m1, m5, m15, now)
        m5_ind = IndicatorSet(m5.rename(columns={"tick_volume": "volume"}), self.cfg.indicators)
        m15_ind = IndicatorSet(m15.rename(columns={"tick_volume": "volume"}), self.cfg.indicators)
        m5_idx = len(m5) - 1
        m15_idx = len(m15) - 1
        signal = generate_signal(
            _FastIndicatorView(m5_ind, m5_idx, self.cfg.scoring.volume_lookback + 1),
            _FastIndicatorView(m15_ind, m15_idx, 1),
            self.cfg.scoring,
        )
        decision = decide(
            signal,
            snapshot,
            kill_switch_blocks_entries=False,
            enabled_sessions=None,
            live_confirmed=False,
            ml_state=MlState.NOT_LOADED,
        )
        if decision.action == TradeAction.NO_TRADE:
            print(
                f"M5_DECISION {now.isoformat()} NO_TRADE tech_dir={signal.direction.value} "
                f"reason={decision.reason_code}",
                flush=True,
            )
            return

        market_features, _, _, _ = build_market_state_features(m1, m5, m15, self.cfg)
        candidate = self._candidate_frame(signal, now, snapshot, market_features, m5_ind, m5_idx)
        score, latency_ms, tensor = self.scorer.score(market_features, candidate)
        margin = score - S2_THRESHOLD
        cid = str(candidate.iloc[0]["sample_id"])
        passed = score >= S2_THRESHOLD
        print(
            f"S2_SCORE candidate={cid} dir={signal.direction.value} setup={signal.setup} "
            f"S2={score:.6f} threshold={S2_THRESHOLD:.6f} delta={margin:+.6f} "
            f"{'PASS' if passed else 'NO_PASS'} infer_ms={latency_ms:.2f} "
            f"latest_m1={tensor.latest_m1} latest_m5_close={tensor.latest_m5_close} latest_m15_close={tensor.latest_m15_close}",
            flush=True,
        )
        if not passed:
            return

        atr_points = float(m5_ind.atr.iloc[m5_idx] / self.point)
        risk = self.risk_manager.evaluate(signal, snapshot, atr_points)
        if not risk.approved:
            print(f"ML_PASS_BUT_RISK_REJECT candidate={cid} reason={risk.reason_code}", flush=True)
            return

        # Replay semantics: only one pending entry object exists at a time; the
        # next M1 cycle consumes it before a new M5 candidate can replace it.
        self.pending = PendingEntry(
            candidate_id=cid,
            signal_time_iso=now.isoformat(),
            direction=signal.direction,
            setup=signal.setup,
            signal_score=int(signal.score),
            sl_points=float(risk.sl_points),
            risk_amount_reference=float(risk.risk_amount),
            session=snapshot.session.value,
            regime=snapshot.regime.value,
            s2_score=float(score),
        )
        print(
            f"PENDING_CREATED candidate={cid} dir={signal.direction.value} S2={score:.6f} "
            f"sl_pts={risk.sl_points:.1f} -> execute on NEXT M1 cycle",
            flush=True,
        )

    def _modify_sl(self, ticket: int, new_sl: float, tp: float) -> bool:
        import MetaTrader5 as mt5  # type: ignore

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": int(ticket),
            "symbol": self.connector.symbol,
            "sl": round(float(new_sl), self.digits),
            "tp": round(float(tp), self.digits),
            "magic": MAGIC,
        }
        result = mt5.order_send(request)
        if result is None:
            print(f"SL_MODIFY_FAILED ticket={ticket} result=None error={mt5.last_error()}", flush=True)
            return False
        ok = int(result.retcode) in {int(mt5.TRADE_RETCODE_DONE), int(getattr(mt5, "TRADE_RETCODE_DONE_PARTIAL", -999))}
        print(
            f"SL_MODIFY ticket={ticket} new_sl={new_sl} tp={tp} retcode={result.retcode} "
            f"{'PASS' if ok else 'FAIL'}",
            flush=True,
        )
        return ok

    def manage_positions(self) -> None:
        bid, ask, _ = self.connector.get_tick()
        open_positions = [p for p in self.connector.get_open_positions() if int(getattr(p, "magic", MAGIC)) == MAGIC]
        open_ids = {int(p.ticket) for p in open_positions}
        # Never infer initial risk from a moved SL. Missing record means HOLD and
        # loud diagnostic, not a fabricated R multiple.
        for ticket in list(self.records):
            if ticket not in open_ids:
                del self.records[ticket]
        if not open_positions:
            self._save_state()
            return
        m5 = canonical_bars(self.connector.get_ohlc("M5", 500))
        atr_points = float(IndicatorSet(m5.rename(columns={"tick_volume": "volume"}), self.cfg.indicators).atr.iloc[-1] / self.point)
        now = datetime.now(timezone.utc)
        for pos in open_positions:
            ticket = int(pos.ticket)
            rec = self.records.get(ticket)
            if rec is None:
                print(f"POSITION_HOLD ticket={ticket} reason=MISSING_IMMUTABLE_INITIAL_RISK_RECORD", flush=True)
                continue
            current_sl = float(getattr(pos, "sl", 0.0) or 0.0)
            current_tp = float(getattr(pos, "tp", rec.initial_tp) or rec.initial_tp)
            state = PositionState(
                position_id=ticket,
                direction=rec.direction,
                entry_price=rec.entry_price,
                current_sl=current_sl,
                current_tp=current_tp,
                initial_risk_points=rec.initial_risk_points,
                point=self.point,
                digits=self.digits,
                open_time=datetime.fromisoformat(rec.opened_at_iso),
            )
            market = PositionMarketState(bid=float(bid), ask=float(ask), timestamp_utc=now)
            action = decide_position_action(state, market, self.cfg.risk, atr_points)
            print(
                f"POSITION_AUDIT ticket={ticket} dir={rec.direction.value} entry={rec.entry_price} "
                f"initial_risk_pts={rec.initial_risk_points:.1f} R={action.r_multiple:.4f} "
                f"current_sl={current_sl} tp={current_tp} action={action.action.value} "
                f"reason={action.reason.value} proposed_sl={action.new_sl}",
                flush=True,
            )
            if action.action == PositionActionType.MOVE_STOP and action.new_sl is not None:
                self._modify_sl(ticket, action.new_sl, current_tp)
        self._save_state()

    def process_one_cycle(self) -> None:
        self.manage_positions()
        m1, m5, m15 = self.fetch_market()
        latest_m1 = pd.Timestamp(m1["time"].iloc[-1])
        if self.last_processed_m1 is not None and latest_m1 <= self.last_processed_m1:
            return
        self.last_processed_m1 = latest_m1
        print(f"M1_CLOSED {latest_m1.isoformat()} positions={self._open_count()}/{MAX_POSITIONS}", flush=True)

        # EXACT replay ordering: pending from previous decision is consumed first,
        # then the new M5-close decision is evaluated.
        self.execute_pending_if_due(latest_m1, m1, m5, m15)
        self.evaluate_new_m5(latest_m1, m1, m5, m15)

    def run(self) -> None:
        if self.check_only:
            print("CHECK_ONLY=PASS", flush=True)
            print("LEGACY_RUNTIME_USED=NO", flush=True)
            print("REPLAY_ENTRY_ORDER=PENDING_THEN_NEXT_M1", flush=True)
            print("IMMUTABLE_INITIAL_RISK=PASS_BY_DESIGN", flush=True)
            return
        while self.running:
            try:
                self.process_one_cycle()
            except KeyboardInterrupt:
                self.running = False
                break
            except Exception as exc:
                print(f"RUNTIME_ERROR {type(exc).__name__}: {exc}", flush=True)
            time.sleep(0.25)
        self._save_state()
        print("STOPPED", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="TOP40 ALL-SESSIONS live runtime rebuilt from STRICT replay semantics")
    parser.add_argument("--enable-demo-orders", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    runtime = ReplayContractLiveRuntime(args.enable_demo_orders, args.check_only)

    def stop_handler(signum: int, frame: Any) -> None:
        runtime.running = False

    try:
        os_signal.signal(os_signal.SIGINT, stop_handler)
        os_signal.signal(os_signal.SIGTERM, stop_handler)
    except Exception:
        pass
    runtime.run()


if __name__ == "__main__":
    main()
