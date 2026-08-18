# TOP40 Short-Memory 5/3/2 — Methodology 2026

## Objective
Build a new short-memory model with sequence lengths M1=5, M5=3, M15=2 and derive its own S2 threshold. The existing 8/5/3 model and threshold are not reused for calibration.

## Frozen structural rules
- Sequence lengths: M1=5, M5=3, M15=2.
- Model architecture: same architecture used by the current short-memory pipeline.
- Technical candidate generation: unchanged.
- Feature definitions: unchanged.
- Target definitions S1/S2: unchanged.
- Purge: 15 minutes.
- Embargo: 15 minutes.
- Seed: 2303.
- Batch size: 256.
- Max epochs: 30.
- Early stopping patience: 5.
- Learning rate: 0.001.
- Economic replay policy: fixed lot 0.01, max 3 positions, all sessions, cooldown off.
- No martingale, grid, averaging down, or post-loss sizing changes.

## Data contract
Fresh MT5 data already downloaded to D:\ORO_532_2026_TUNING.

Raw download window:
- Warmup start: 2026-01-01 00:00 UTC.
- Threshold-selection window: 2026-02-01 00:00 UTC through 2026-06-30 23:59:59 UTC.
- Tail: through 2026-07-05 23:59 UTC.

The downloaded M1/M5/M15 files and their SHA256 values in FRESH_532_2026_TUNING_DOWNLOAD_MANIFEST.json are the immutable data source for this experiment.

## Phase A — causal 5/3/2 dataset
For every strategy candidate timestamp, build only closed/available bars at or before that timestamp:
- last 5 contiguous M1 bars,
- last 3 contiguous M5 bars,
- last 2 contiguous M15 bars.

A sequence is invalid if it crosses a market closure or contains any timestamp after the candidate time. No forward fill across closures is permitted.

## Phase B — model training
Train a NEW S1 model and a NEW S2 model using the same architecture and optimizer policy as the existing short-memory experiment. Fit preprocessing scalers on training rows only inside each fold.

The 5/3/2 models must be saved under a new output directory and must never overwrite 8/5/3 artifacts.

Required frozen artifacts:
- selected_short_memory_532_s1.pt
- selected_short_memory_532_s2.pt
- short_memory_532_scalers_s1.joblib
- short_memory_532_scalers_s2.joblib
- model/scaler SHA256 manifest
- OOS predictions
- training curves
- fold metrics

## Phase C — threshold selection
Threshold selection is performed only on 5/3/2 S2 predictions. Do not apply the 8/5/3 threshold 0.32696733474731443 as a calibration input.

First-pass thresholds:
- 0.20
- 0.25
- 0.30
- 0.35
- 0.40
- 0.45
- 0.50

Evaluation window for threshold selection:
2026-02-01 through 2026-06-30 only.

For every threshold report:
- ML accepted candidates
- executed trades
- wins
- losses
- win rate
- gross profit
- gross loss
- net profit
- profit factor
- expectancy
- max drawdown USD
- max drawdown percent
- trades/month
- long/short breakdown
- session breakdown

## Threshold decision rule
Do NOT automatically select the row with the highest profit factor or net profit.

Prefer a stable plateau where adjacent thresholds remain profitable and retain sufficient trade count. The selected threshold should balance:
1. profit factor,
2. expectancy,
3. drawdown,
4. trade count,
5. stability across adjacent thresholds,
6. long/short balance,
7. session robustness.

If the apparent optimum is an isolated spike, do not freeze it without a finer local sweep.

## Phase D — fine sweep
After the first seven thresholds, define a narrower grid around the robust region, normally step 0.01 or 0.02. The exact fine grid is chosen only after reviewing Phase C results.

## Phase E — freeze
Once selected, freeze:
- sequence lengths 5/3/2,
- S1/S2 model hashes,
- scaler hashes,
- selected S2 threshold,
- data manifest hashes,
- economic replay rules.

Write a 5/3/2 frozen manifest. No threshold recalibration is allowed after this point for the validation run.

## Phase F — out-of-sample validation
Data used to choose the threshold must not be presented as final unseen evidence.

After freezing the threshold on February-June 2026, validate on a later period excluded from threshold selection (for example July 2026 onward when enough complete data is available). No retraining and no threshold change are allowed during this validation.

## Comparison to 8/5/3
Only after the 5/3/2 threshold is frozen compare 5/3/2 with 8/5/3. The comparison must use the same economic replay policy and the same validation interval. The goal is to determine whether shorter memory improves robustness, not merely whether it can be tuned to a higher in-sample result.

## Experiment identity
Research name: TOP40_SHORT_MEMORY_532_2026_V1
Status at creation: METHODOLOGY_FROZEN_BEFORE_TRAINING
