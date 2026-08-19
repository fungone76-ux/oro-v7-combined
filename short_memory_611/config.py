"""Frozen research configuration for the isolated 6/1/1 M1 timing experiment.

Contract:
- evaluate after every closed M1;
- use the last 6 closed M1 bars for timing;
- use one last-closed M5 for primary direction (EMA9 vs EMA21);
- use one last-closed M15 for directional context (close vs EMA50);
- M5/M15 disagreement NEVER deletes a sample: disagreement is an explicit feature;
- theoretical entry is the next M1 open;
- no live orders are implemented in this package.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ShortMemory611Config:
    m1_length: int = 6
    m5_length: int = 1
    m15_length: int = 1
    label_horizon_minutes: int = 15
    purge_minutes: int = 15
    embargo_minutes: int = 15
    seed: int = 6112026
    batch_size: int = 256
    max_epochs: int = 30
    early_stopping_patience: int = 5
    learning_rate: float = 0.001
    training_start_utc: str = "2026-03-01T00:00:00+00:00"
    training_end_utc: str = "2026-08-13T23:59:59+00:00"
    unseen_start_utc: str = "2026-08-14T00:00:00+00:00"


CONFIG = ShortMemory611Config()

MODEL_ARCHITECTURE = {
    "m1_lstm_hidden": 32,
    "m5_lstm_hidden": 8,
    "m15_lstm_hidden": 8,
    "dense": 32,
    "dropout": 0.1,
}

RESEARCH_VERSION = "TOP40_M1_TIMING_611_MAR13AUG_2026_V2_CONFLICT_AWARE"
METHODOLOGY_STATUS = "FROZEN_BEFORE_FIRST_VALID_611_PREPARE"
