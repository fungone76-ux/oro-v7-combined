from short_memory.config import CONFIG, MODEL_ARCHITECTURE, RESEARCH_VERSION


def test_sequence_lengths_are_frozen_8_5_3():
    assert (CONFIG.m1_length, CONFIG.m5_length, CONFIG.m15_length) == (8, 5, 3)


def test_walkforward_gap_contract():
    assert CONFIG.purge_minutes == 15
    assert CONFIG.embargo_minutes == 15


def test_training_hyperparameters_are_original_phase3b2_baseline():
    assert CONFIG.seed == 2303
    assert CONFIG.batch_size == 256
    assert CONFIG.max_epochs == 30
    assert CONFIG.early_stopping_patience == 5
    assert CONFIG.learning_rate == 0.001
    assert MODEL_ARCHITECTURE == {
        "m1_lstm_hidden": 32,
        "m5_lstm_hidden": 16,
        "m15_lstm_hidden": 16,
        "dense": 32,
        "dropout": 0.1,
    }


def test_execution_research_constraints():
    assert CONFIG.fixed_lot_diagnostic == 0.01
    assert CONFIG.max_positions == 3


def test_version_name_is_stable():
    assert RESEARCH_VERSION == "SHORT_MEMORY_V1"
