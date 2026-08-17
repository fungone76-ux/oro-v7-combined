"""Frozen configuration for TOP40_SHORT_MEMORY V1.

The only structural experiment in V1 is the sequence length triplet 8/5/3.
Do not tune these values inside the first research pass.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ShortMemoryConfig:
    m1_length: int = 8
    m5_length: int = 5
    m15_length: int = 3
    purge_minutes: int = 15
    embargo_minutes: int = 15
    seed: int = 2303
    batch_size: int = 256
    max_epochs: int = 30
    early_stopping_patience: int = 5
    learning_rate: float = 0.001
    fixed_lot_diagnostic: float = 0.01
    max_positions: int = 3


CONFIG = ShortMemoryConfig()

# Original Phase 3B2 architecture is intentionally preserved.
MODEL_ARCHITECTURE = {
    "m1_lstm_hidden": 32,
    "m5_lstm_hidden": 16,
    "m15_lstm_hidden": 16,
    "dense": 32,
    "dropout": 0.1,
}

RESEARCH_VERSION = "SHORT_MEMORY_V1"
