# TOP40 SHORT MEMORY — Research Protocol V1

## Hypothesis

XAU/USD may react better to a much shorter raw sequence context than the original TOP40 Phase 3B2 representation.

The first experiment freezes exactly one structural change:

- M1: 60 -> **8** bars
- M5: 24 -> **5** bars
- M15: 16 -> **3** bars

No neighboring sequence lengths are tested in V1.

## Preserved baseline

The following remain conceptually identical to the original TOP40 research stack:

- technical candidate engine;
- feature engineering and signed direction features;
- S1 and S2 targets;
- M1/M5/M15 branch architecture (32/16/16 hidden units);
- dense=32, dropout=0.1;
- Adam lr=0.001;
- batch size 256;
- max epochs 30;
- early stopping patience 5;
- seed 2303;
- temporal walk-forward only;
- purge=15m and embargo=15m;
- C3 economic stress;
- fixed lot 0.01 diagnostic;
- max 3 positions;
- no martingale, grid, averaging down or double entry.

## Mandatory causal rules

For candidate timestamp T, all M1/M5/M15 sequence rows must be closed and available at T. No future fill, backfill or target-derived value is allowed. Scalers are fit on training folds only and then applied unchanged to validation/test folds.

## Threshold policy

The old S2 thresholds do not transfer to the new model. TOP20_SHORT and TOP40_SHORT will be derived from OOS score distributions and frozen before final economic metrics are opened.

## Validation order

1. Source/candidate audit.
2. Sequence validity 8/5/3.
3. Unit/causality tests.
4. Walk-forward S1/S2 training.
5. OOS prediction distribution.
6. Freeze TOP20_SHORT/TOP40_SHORT thresholds.
7. Economic replay A0/TOP20_SHORT/TOP40_SHORT under C3.
8. Fixed 0.01 diagnostic.
9. Stability by time, side, session, regime and volatility.
10. Reaction-speed comparison against original 60/24/16 TOP40.

## Promotion gates

SHORT_MEMORY_V1 is not promoted because of higher frequency alone. A promising result requires positive OOS expectancy, PF >= 1.30 as a target gate, no obvious temporal collapse, reasonable drawdown and fixed-0.01 profitability. 2025/2026 must not be described as a pristine holdout because previous project research has already inspected it.

No MT5 execution runtime is created during this phase.
