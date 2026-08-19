"""Frozen configuration for TOP40_SHORT_MEMORY_532_2026_V1.

This package is intentionally isolated from short_memory (8/5/3).
Only the sequence lengths change structurally in the first 5/3/2 experiment.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ShortMemory532Config:
    m1_length: int = 5
    m5_length: int = 3
    m15_length: int = 2
    purge_minutes: int = 15
    embargo_minutes: int = 15
    seed: int = 2303
    batch_size: int = 256
    max_epochs: int = 30
    early_stopping_patience: int = 5
    learning_rate: float = 0.001
    fixed_lot_diagnostic: float = 0.01
    max_positions: int = 3
    threshold_grid: tuple[float, ...] = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
    tuning_start_utc: str = "2026-02-01T00:00:00+00:00"
    tuning_end_utc: str = "2026-06-30T23:59:59+00:00"


CONFIG = ShortMemory532Config()

MODEL_ARCHITECTURE = {
    "m1_lstm_hidden": 32,
    "m5_lstm_hidden": 16,
    "m15_lstm_hidden": 16,
    "dense": 32,
    "dropout": 0.1,
}

RESEARCH_VERSION = "TOP40_SHORT_MEMORY_532_2026_V1"
METHODOLOGY_STATUS = "FROZEN_BEFORE_TRAINING"
