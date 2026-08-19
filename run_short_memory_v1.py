from __future__ import annotations

import argparse
from pathlib import Path

from short_memory.runner import run


def main() -> None:
    p=argparse.ArgumentParser(description='TOP40 SHORT_MEMORY V1 — 8/5/3 causal walk-forward research')
    p.add_argument('--market-state',type=Path,required=True,help='Original Phase 3A market_state.parquet')
    p.add_argument('--candidates',type=Path,required=True,help='Original Phase 3A strategy_candidates.parquet')
    p.add_argument('--out',type=Path,default=Path('research_output/short_memory_v1'))
    p.add_argument('--splits',type=Path,default=Path('config/walkforward_splits.json'))
    a=p.parse_args()
    summary=run(a.market_state,a.candidates,a.out,a.splits)
    print('\nFINAL TRAINING SUMMARY')
    for k,v in summary.items(): print(f'{k}: {v}')


if __name__=='__main__':
    main()
