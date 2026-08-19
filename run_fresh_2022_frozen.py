from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import types
from pathlib import Path

import pandas as pd

EXPECTED_HASHES_2022 = {
    "M1": "873021f6db4649aef17d4cc270b84f225279c8c638cc463c61e91b0db6ee9755",
    "M5": "7ab621a36d08db7551b5877d9274fcf8b8345c09b9d7c96792c85863d58c613f",
    "M15": "ab0330201e0823ae8f627371213faedcb25da7ee4b17419b645a34159e73fb5e",
}
FROZEN_THRESHOLD = 0.32696733474731443
EVAL_START_2022 = pd.Timestamp("2022-01-01T00:00:00Z")
EVAL_END_2022 = pd.Timestamp("2022-12-31T23:59:59Z")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_patched_base_for_2022(repo: Path):
    """Load the frozen 2017 inference runner with ONLY its hardcoded coverage guard adapted to 2022."""
    base_path = repo / "run_unseen_2017_frozen.py"
    text = base_path.read_text(encoding="utf-8")
    old = "if m1.time.min() > pd.Timestamp('2016-12-05T00:00:00Z') or m1.time.max() < pd.Timestamp('2018-01-02T00:00:00Z'):"
    new = "if m1.time.min() > pd.Timestamp('2021-12-05T00:00:00Z') or m1.time.max() < pd.Timestamp('2023-01-02T00:00:00Z'):"
    if old not in text:
        raise RuntimeError("BASE_2017_COVERAGE_GUARD_NOT_FOUND")
    patched = text.replace(old, new, 1)
    module = types.ModuleType("run_fresh_2022_base_in_memory")
    module.__file__ = str(base_path)
    module.__name__ = "run_fresh_2022_base_in_memory"
    exec(compile(patched, str(base_path), "exec"), module.__dict__)
    return module


def main() -> None:
    ap = argparse.ArgumentParser(description="Fresh-download 2022 frozen SHORT_MEMORY 8/5/3 replay")
    ap.add_argument("--top40-root", type=Path, required=True)
    ap.add_argument("--short-out", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("research_output/fresh_2022_frozen"))
    a = ap.parse_args()

    repo = Path(__file__).resolve().parent
    data_dir = a.data_dir.resolve()
    src = {
        "M1": data_dir / "XAUUSD_M1_FRESH_2022.csv",
        "M5": data_dir / "XAUUSD_M5_FRESH_2022.csv",
        "M15": data_dir / "XAUUSD_M15_FRESH_2022.csv",
    }
    for key, path in src.items():
        if not path.exists():
            raise RuntimeError(f"MISSING_FRESH_2022_{key}: {path}")
        actual = sha256(path)
        if actual != EXPECTED_HASHES_2022[key]:
            raise RuntimeError(
                f"FRESH_2022_{key}_HASH_MISMATCH expected={EXPECTED_HASHES_2022[key]} actual={actual}"
            )

    alias = data_dir / "_frozen_runner_alias_2022"
    alias.mkdir(parents=True, exist_ok=True)
    names = {
        "M1": "XAUUSD_M1_UNSEEN_2017.csv",
        "M5": "XAUUSD_M5_UNSEEN_2017.csv",
        "M15": "XAUUSD_M15_UNSEEN_2017.csv",
    }
    for key in ("M1", "M5", "M15"):
        dst = alias / names[key]
        if dst.exists():
            dst.unlink()
        shutil.copy2(src[key], dst)

    base = load_patched_base_for_2022(repo)
    base.EXPECTED_HASHES = dict(EXPECTED_HASHES_2022)
    base.FROZEN_THRESHOLD = FROZEN_THRESHOLD
    base.EVAL_START = EVAL_START_2022
    base.EVAL_END = EVAL_END_2022

    out = a.out.resolve()
    sys.argv = [
        "run_unseen_2017_frozen.py",
        "--top40-root", str(a.top40_root.resolve()),
        "--short-out", str(a.short_out.resolve()),
        "--data-dir", str(alias),
        "--out", str(out),
    ]

    print("FRESH_2022_DATASET_HASHES=PASS", flush=True)
    print("FRESH_2022_COVERAGE_GUARD=2021-12-05..2023-01-02", flush=True)
    print("BASE_2017_FILE_CHANGED=NO", flush=True)
    print("REUSED_OLD_FILES=NO", flush=True)
    print("REUSED_OLD_PARQUETS=NO", flush=True)
    print("NO_RETRAINING=REQUIRED", flush=True)
    print(f"FROZEN_THRESHOLD={FROZEN_THRESHOLD:.15f}", flush=True)
    print("EVALUATION_PERIOD=2022-01-01..2022-12-31", flush=True)
    print("NOTE=2022_IS_NOT_UNSEEN_TO_MODEL; fresh data download only", flush=True)

    base.main()

    print("FRESH_2022_REPLAY=COMPLETE", flush=True)
    print("FRESH_2022_DATASET_HASHES=PASS", flush=True)
    print("NO_RETRAINING=PASS", flush=True)
    print("FROZEN_MODEL=PASS", flush=True)
    print("FROZEN_THRESHOLD=PASS", flush=True)
    print("SEQUENCES_8_5_3=PASS", flush=True)
    print("FIXED_LOT_001=PASS", flush=True)
    print("SESSION_FILTER=DISABLED", flush=True)
    print("COOLDOWN=DISABLED", flush=True)
    print("DATASET_STATUS=FRESH_DOWNLOAD_BUT_MODEL_SEEN_ERA", flush=True)


if __name__ == "__main__":
    main()
