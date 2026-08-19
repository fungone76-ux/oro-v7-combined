import json
from pathlib import Path


def test_walkforward_splits_are_exact_original_phase3b2_contract():
    p = json.loads(Path('config/walkforward_splits.json').read_text(encoding='utf-8'))
    assert p['purge_minutes'] == 15
    assert p['embargo_minutes'] == 15
    assert len(p['folds']) == 4
    assert p['folds'][0]['train_start'] == '2024-08-01T01:00:00+00:00'
    assert p['folds'][0]['valid_start'] == '2024-11-04T15:00:24+00:00'
    assert p['folds'][3]['valid_end'] == '2025-11-21T21:32:00+00:00'
