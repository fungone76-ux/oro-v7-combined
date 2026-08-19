from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd

REPO_ROOT = Path(r"D:\oro_top40_shortmem_OLD")
ENGINE_ROOT = Path(r"D:\ORO_SHORT_MEMORY_ALLSESSIONS_FINAL")
BASE_RUNTIME_PATH = ENGINE_ROOT / "TOP40_ALLSESSIONS_FROM_TEST.py"
FROZEN_DIR = Path(r"D:\ORO_532_MAR13AUG_2026\frozen_demo_532_top15")
MANIFEST_PATH = FROZEN_DIR / "FROZEN_532_TOP15_MANIFEST.json"
MODEL_PATH = FROZEN_DIR / "selected_532_s2_top15.pt"
SCALER_PATH = FROZEN_DIR / "scalers_532_s2_top15.joblib"
STATE_PATH = ENGINE_ROOT / "state" / "top40_532_top15_live_state.json"

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(ENGINE_ROOT))

from short_memory_532.config import CONFIG
from run_532_train_s2_2026 import model_class
from short_memory.training import M1_FEATURES, M5_FEATURES, M15_FEATURES, STATIC_NUMERIC, STATIC_CATEGORICAL


@dataclass
class TensorAudit:
    latest_m1: pd.Timestamp
    latest_m5_close: pd.Timestamp
    latest_m15_close: pd.Timestamp


def _load_base_runtime():
    if not BASE_RUNTIME_PATH.exists():
        raise SystemExit(f"MISSING_BASE_RUNTIME: {BASE_RUNTIME_PATH}")
    spec = importlib.util.spec_from_file_location("top40_current_local_runtime", BASE_RUNTIME_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("BASE_RUNTIME_IMPORT_SPEC_FAILED")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _window(df: pd.DataFrame, time_col: str, features: list[str], ts: pd.Timestamp, length: int, step_minutes: int) -> tuple[np.ndarray, pd.Timestamp]:
    x = df[[time_col] + features].copy()
    x[time_col] = pd.to_datetime(x[time_col], utc=True)
    x = x[x[time_col] <= ts].dropna(subset=[time_col]).sort_values(time_col).drop_duplicates(time_col, keep="last")
    if len(x) < length:
        raise ValueError(f"INSUFFICIENT_{time_col}_HISTORY have={len(x)} need={length}")
    x = x.tail(length)
    times = x[time_col]
    diffs = times.diff().dropna().dt.total_seconds().div(60.0).to_numpy(float)
    if len(diffs) and not np.allclose(diffs, float(step_minutes)):
        raise ValueError(f"NON_CONTIGUOUS_{time_col}_WINDOW diffs={diffs.tolist()}")
    values = x[features].to_numpy(np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"NONFINITE_{time_col}_FEATURES")
    return values, pd.Timestamp(times.iloc[-1])


class FrozenS2Scorer532:
    def __init__(self) -> None:
        for p in (MANIFEST_PATH, MODEL_PATH, SCALER_PATH):
            if not p.exists():
                raise RuntimeError(f"FROZEN_ARTIFACT_MISSING: {p}")
        self.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if self.manifest.get("sequence_lengths") != {"m1": 5, "m5": 3, "m15": 2}:
            raise RuntimeError(f"BAD_FROZEN_SEQUENCE: {self.manifest.get('sequence_lengths')}")
        if int(self.manifest.get("top_pct", -1)) != 15:
            raise RuntimeError(f"BAD_FROZEN_TOP_PCT: {self.manifest.get('top_pct')}")
        self.threshold = float(self.manifest["frozen_threshold"])
        self.scalers = joblib.load(SCALER_PATH)
        categories = self.scalers["categories"]
        static_dim = len(STATIC_NUMERIC) + sum(len(categories.get(col, [])) for col in STATIC_CATEGORICAL)

        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        Model = model_class()
        self.model = Model(len(M1_FEATURES), len(M5_FEATURES), len(M15_FEATURES), static_dim).to(self.device)
        try:
            state = torch.load(MODEL_PATH, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(MODEL_PATH, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        print(
            f"FROZEN_532_SCORER=PASS device={self.device} threshold={self.threshold:.9f} "
            f"sequence=5/3/2 top15",
            flush=True,
        )

    def _static(self, candidate: pd.DataFrame) -> np.ndarray:
        row = candidate.iloc[0]
        num = np.asarray([float(row[c]) for c in STATIC_NUMERIC], dtype=np.float64).reshape(1, -1)
        if not np.isfinite(num).all():
            raise ValueError("NONFINITE_STATIC_NUMERIC")
        sn = self.scalers["static_numeric"].transform(num).astype(np.float32)
        pieces: list[float] = []
        vocab = self.scalers["categories"]
        for col in STATIC_CATEGORICAL:
            value = str(row[col])
            cats = list(vocab.get(col, []))
            pieces.extend(1.0 if value == cat else 0.0 for cat in cats)
        sc = np.asarray(pieces, dtype=np.float32).reshape(1, -1)
        return np.hstack([sn, sc]).astype(np.float32)

    def score(self, market_features: pd.DataFrame, candidate: pd.DataFrame):
        t0 = time.perf_counter()
        ts = pd.Timestamp(candidate.iloc[0]["timestamp"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")

        market = market_features.copy()
        market["timestamp"] = pd.to_datetime(market["timestamp"], utc=True)
        m5 = market[["m5_source_close_time"] + list(M5_FEATURES)].dropna(subset=["m5_source_close_time"]).copy()
        m15 = market[["m15_source_close_time"] + list(M15_FEATURES)].dropna(subset=["m15_source_close_time"]).copy()

        w1, latest1 = _window(market, "timestamp", list(M1_FEATURES), ts, CONFIG.m1_length, 1)
        w5, latest5 = _window(m5, "m5_source_close_time", list(M5_FEATURES), ts, CONFIG.m5_length, 5)
        w15, latest15 = _window(m15, "m15_source_close_time", list(M15_FEATURES), ts, CONFIG.m15_length, 15)

        x1 = self.scalers["m1"].transform(w1).astype(np.float32)[None, :, :]
        x5 = self.scalers["m5"].transform(w5).astype(np.float32)[None, :, :]
        x15 = self.scalers["m15"].transform(w15).astype(np.float32)[None, :, :]
        xs = self._static(candidate)

        import torch
        with torch.no_grad():
            score = float(self.model(
                torch.tensor(x1, dtype=torch.float32, device=self.device),
                torch.tensor(x5, dtype=torch.float32, device=self.device),
                torch.tensor(x15, dtype=torch.float32, device=self.device),
                torch.tensor(xs, dtype=torch.float32, device=self.device),
            ).item())
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return score, latency_ms, TensorAudit(latest1, latest5, latest15)


def main() -> None:
    ap = argparse.ArgumentParser(description="TOP40 frozen 5/3/2 Top15 demo runtime")
    ap.add_argument("--enable-demo-orders", action="store_true", help="Actually send orders; demo accounts only")
    ap.add_argument("--check-only", action="store_true", help="Load everything and exit without trading")
    args = ap.parse_args()

    if not MANIFEST_PATH.exists():
        raise SystemExit(
            f"FROZEN_MODEL_NOT_FOUND: {MANIFEST_PATH}\n"
            "Run: python run_532_freeze_final_mar13aug_2026.py"
        )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    threshold = float(manifest["frozen_threshold"])

    base = _load_base_runtime()
    base.S2_THRESHOLD = threshold
    base.STATE_FILE = STATE_PATH
    base.LiveS2Scorer = FrozenS2Scorer532

    class Runtime532(base.ReplayContractLiveRuntime):
        def _verify_account(self) -> None:
            account = self.connector.get_account_state()
            is_demo = account.trade_mode == base.AccountTradeMode.DEMO
            self.orders_enabled = bool(self.enable_demo_orders and is_demo and not self.check_only)
            if self.enable_demo_orders and not is_demo:
                raise RuntimeError("DEMO_ONLY_GUARD: account is not DEMO")
            print("=" * 78, flush=True)
            print("TOP40 5/3/2 TOP15 - FROZEN DEMO", flush=True)
            print("TRAINING: 2026-03-01 -> 2026-08-13 | AUG14+ NOT USED", flush=True)
            print(f"SIMBOLO={self.connector.symbol} | magic={base.MAGIC}", flush=True)
            print(f"ML S2: sequenze=5/3/2 | TOP15 threshold={threshold:.9f}", flush=True)
            print(f"RISCHIO: lotto fisso={base.FIXED_LOT:.2f} | max posizioni={base.MAX_POSITIONS} | cooldown=OFF", flush=True)
            print("SESSIONI: ALL-SESSIONS", flush=True)
            print(f"ORDINI: {'ATTIVI SU DEMO' if self.orders_enabled else 'DISATTIVATI / CHECK-ONLY'}", flush=True)
            print("APERTURA: M5 valida -> tecnica -> S2 Top15 -> pending -> M1 successiva", flush=True)
            print("GESTIONE POSIZIONE: runtime locale TOP40 corrente + core BE/trailing condiviso", flush=True)
            print(f"STATE={STATE_PATH}", flush=True)
            print("=" * 78, flush=True)

    runtime = Runtime532(enable_demo_orders=args.enable_demo_orders, check_only=args.check_only)
    runtime.run()


if __name__ == "__main__":
    main()
