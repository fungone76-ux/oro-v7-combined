import json
from pathlib import Path


def provenance():
    return json.loads(Path('SOURCE_PROVENANCE.json').read_text(encoding='utf-8'))


def test_original_commit_is_frozen():
    p = provenance()
    assert p['original_repository'] == 'fungone76-ux/oro'
    assert p['original_commit'] == '0158dab30d2179eedace26fb73618548cd674bb4'


def test_experiment_changes_sequence_lengths_from_60_24_16_to_8_5_3():
    p = provenance()
    assert p['original_sequence_lengths'] == {'m1': 60, 'm5': 24, 'm15': 16}
    assert p['experimental_sequence_lengths'] == {'m1': 8, 'm5': 5, 'm15': 3}


def test_original_frozen_artifact_hashes_are_recorded():
    p = provenance()['original_frozen_artifacts']
    assert p['s2_model_sha256'] == 'ee120a3623375b000c1d65d47debb9c0c3edbb92ee36c978ce3f14578e03035b'
    assert p['s2_scaler_sha256'] == '73b1548cb5e2e2659b9d332128bf625663ca3eee26f29beaa5ea24b13d5999ac'


def test_short_memory_training_does_not_hardcode_legacy_thresholds():
    text = Path('short_memory/training.py').read_text(encoding='utf-8')
    assert '0.28028725385665865' not in text
    assert '1.234266757965088' not in text
