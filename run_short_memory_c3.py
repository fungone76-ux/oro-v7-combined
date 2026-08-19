from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def _log(t0: float, msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] +{time.perf_counter()-t0:7.1f}s | {msg}", flush=True)


def _load_short_model(model_path: Path, m1_dim: int, m5_dim: int, m15_dim: int, static_dim: int, device: str):
    import torch
    from short_memory.training import torch_model_class
    Model = torch_model_class()
    model = Model(m1_dim, m5_dim, m15_dim, static_dim, "S2").to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def _transform(bundle, scalers):
    def scale_seq(values: np.ndarray, key: str) -> np.ndarray:
        shape = values.shape
        return scalers[key].transform(values.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)

    m1 = scale_seq(bundle.m1, "m1")
    m5 = scale_seq(bundle.m5, "m5")
    m15 = scale_seq(bundle.m15, "m15")
    static_num = scalers["static_numeric"].transform(bundle.static_num).astype(np.float32)
    pieces = []
    cats = scalers.get("categories", {})
    for col, values in cats.items():
        cv = bundle.static_cat[col].astype(str).fillna("<NA>")
        for value in values:
            pieces.append((cv == value).astype(float).to_numpy()[:, None])
    static_cat = np.hstack(pieces).astype(np.float32) if pieces else np.empty((len(bundle.frame), 0), dtype=np.float32)
    static = np.hstack([static_num, static_cat]).astype(np.float32)
    return m1, m5, m15, static


def _predict(model, arrays, device: str, batch_size: int = 256) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    m1, m5, m15, static = arrays
    dummy = np.zeros(len(m1), dtype=np.float32)
    ds = TensorDataset(
        torch.tensor(m1), torch.tensor(m5), torch.tensor(m15), torch.tensor(static), torch.tensor(dummy)
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    out = []
    with torch.no_grad():
        for bm1, bm5, bm15, bst, _ in loader:
            pred = model(bm1.to(device), bm5.to(device), bm15.to(device), bst.to(device))
            out.append(pred.detach().cpu().numpy())
    return np.concatenate(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="TOP40 SHORT_MEMORY 8/5/3 canonical Phase3C economic replay")
    ap.add_argument("--top40-root", type=Path, required=True, help="Original TOP40 project root, e.g. D:/ORO")
    ap.add_argument("--short-out", type=Path, required=True, help="SHORT_MEMORY training output folder")
    ap.add_argument("--policy", choices=["TOP40_SHORT", "TOP20_SHORT"], default="TOP40_SHORT")
    ap.add_argument("--out", type=Path, default=Path("research_output/short_memory_c3"))
    args = ap.parse_args()

    t0 = time.perf_counter()
    repo_root = Path(__file__).resolve().parent
    top40_root = args.top40_root.resolve()
    short_out = args.short_out.resolve()
    out_dir = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Keep this repo importable after switching cwd to the canonical TOP40 root.
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(top40_root))
    os.chdir(top40_root)

    import torch
    from xau_bot.research.ml import phase3b2_sequential as seq
    from xau_bot.research.ml import phase3c_economic_replay as p3c

    _log(t0, "=== SHORT_MEMORY 8/5/3 CANONICAL C3 START ===")
    _log(t0, f"TOP40 root={top40_root}")
    _log(t0, f"SHORT artifacts={short_out}")

    # Freeze the ONLY structural change versus original Phase3B2.
    seq.M1_LENGTH = 8
    seq.M5_LENGTH = 5
    seq.M15_LENGTH = 3

    manifest = json.loads((short_out / "short_memory_threshold_freeze_manifest.json").read_text(encoding="utf-8"))
    threshold = float(manifest["thresholds"][args.policy])
    _log(t0, f"Frozen policy={args.policy} threshold={threshold:.9f}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _log(t0, f"Device={device}")

    _log(t0, "[1/6] Rebuild exact original Phase3A candidate tensors with lengths 8/5/3")
    market, candidates, _ = seq.load_source_data()
    features = seq.sequence_feature_contract(market, candidates)
    validity = seq.build_sequence_validity(market, candidates, features)
    bundle = seq.build_tensor_bundle(market, candidates, validity, features)
    _log(t0, f"Valid executable ML samples={len(bundle.frame):,}")

    _log(t0, "[2/6] Score ALL candidates with frozen final SHORT_MEMORY S2")
    scalers = joblib.load(short_out / "short_memory_scalers_s2.joblib")
    arrays = _transform(bundle, scalers)
    model = _load_short_model(
        short_out / "selected_short_memory_s2.pt",
        arrays[0].shape[2], arrays[1].shape[2], arrays[2].shape[2], arrays[3].shape[1], device,
    )
    scores = _predict(model, arrays, device)
    pred = pd.DataFrame({
        "sample_id": bundle.frame["sample_id"].astype(str),
        "timestamp": pd.to_datetime(bundle.frame["timestamp"], utc=True),
        "direction": bundle.frame["direction"].astype(str),
        "setup": bundle.frame["setup"].astype(str),
        "s1_score": np.nan,
        "s2_score": scores.astype(float),
        "s2_threshold": threshold,
        "ml_accept": scores >= threshold,
    })
    pred.to_parquet(out_dir / "short_memory_frozen_predictions.parquet", index=False)
    _log(t0, f"Accepted by frozen threshold={int(pred.ml_accept.sum()):,}/{len(pred):,}")

    _log(t0, "[3/6] Select original post-training economic segments WITHOUT performance selection")
    primary, secondary, eval_manifest = p3c.select_economic_segments(pred)
    (out_dir / "economic_eval_manifest.json").write_text(json.dumps(eval_manifest, indent=2), encoding="utf-8")
    segments = [primary] + secondary
    _log(t0, "Segments=" + ", ".join(s.segment_id for s in segments))

    _log(t0, "[4/6] Load canonical ICMarkets M1/M5/M15 and C3 execution engine")
    cfg = p3c.BotConfig()
    if cfg.risk.max_open_positions != 3:
        raise RuntimeError(f"MAX_POSITIONS_CONTRACT_BROKEN={cfg.risk.max_open_positions}")
    spec = p3c.load_symbol_spec(p3c.METADATA_PATH)
    m1, m5, m15 = p3c.load_csv(p3c.M1_PATH), p3c.load_csv(p3c.M5_PATH), p3c.load_csv(p3c.M15_PATH)

    rows = []
    trade_frames = []
    for i, seg in enumerate(segments, start=1):
        _log(t0, f"[5/6] Replay {seg.segment_id} ({i}/{len(segments)}) candidates={seg.candidate_count}")
        run_m1 = m1[(m1["time"] >= seg.start - pd.Timedelta(days=10)) & (m1["time"] <= seg.end)].reset_index(drop=True)
        run_m5 = m5[(m5["time"] >= seg.start - pd.Timedelta(days=10)) & (m5["time"] <= seg.end)].reset_index(drop=True)
        run_m15 = m15[(m15["time"] >= seg.start - pd.Timedelta(days=15)) & (m15["time"] <= seg.end)].reset_index(drop=True)
        seg_pred = pred[(pred.timestamp >= seg.start) & (pred.timestamp <= seg.end)].copy()
        settings = p3c.BacktestSettings(initial_balance=1000.0)
        engine = p3c.MLEconomicReplayEngine(cfg, spec, settings, seg_pred, ml_enabled=True)
        result = engine.run(run_m1, run_m5, run_m15, seg.start, seg.end)
        decisions = pd.DataFrame(engine.candidate_decisions)
        row = p3c._variant_metrics(seg, f"SHORT_{args.policy}_C3", result, decisions, threshold)
        rows.append(row)
        tf = p3c._trades_frame(result, seg_pred, threshold, True)
        if not tf.empty:
            tf["segment_id"] = seg.segment_id
            trade_frames.append(tf)
        _log(t0, f"{seg.segment_id}: trades={row['trades_opened']} PF={row['profit_factor']} exp={row['expectancy']:.4f} net={row['net_pnl']:.2f} DD={row['max_drawdown_pct']:.2f}%")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "policy_metrics_c3.csv", index=False)
    trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    trades.to_csv(out_dir / "trades_c3.csv", index=False)

    # Aggregate only as a transparent descriptive summary; segment rows remain authoritative.
    total_trades = int(metrics.trades_opened.sum()) if not metrics.empty else 0
    total_net = float(metrics.net_pnl.sum()) if not metrics.empty else 0.0
    weighted_exp = float(np.average(metrics.expectancy, weights=np.maximum(metrics.trades_opened, 1))) if not metrics.empty else 0.0
    summary = {
        "policy": args.policy,
        "threshold": threshold,
        "segments": rows,
        "total_trades": total_trades,
        "total_net_pnl_usd": total_net,
        "weighted_expectancy_usd": weighted_exp,
        "economic_replay": "canonical original Phase3C engine / dynamic risk sizing",
        "fixed_001_complete": False,
        "runtime_seconds": time.perf_counter() - t0,
    }
    (out_dir / "c3_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _log(t0, "[6/6] COMPLETE")
    print("\nFINAL C3 SUMMARY")
    print(metrics.to_string(index=False))
    print(f"TOTAL_TRADES={total_trades}")
    print(f"TOTAL_NET_PNL_USD={total_net:.2f}")
    print(f"WEIGHTED_EXPECTANCY_USD={weighted_exp:.4f}")
    print("NOTE: this run uses canonical dynamic risk sizing. Fixed 0.01 audit is intentionally separate.")


if __name__ == "__main__":
    main()
