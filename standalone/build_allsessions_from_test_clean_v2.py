from __future__ import annotations

"""Build CLEAN ALL-SESSIONS from the STRICT-test runtime without a local engine checkout.

This wrapper shallow-clones the canonical engine repository into a temporary
folder, then delegates the actual clean build to build_allsessions_from_test_clean_v1.py.
The previous ORO_SHORT_MEMORY_ALLSESSIONS_FINAL folder is never used.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CANONICAL_ENGINE_REPO = "https://github.com/fungone76-ux/oro.git"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--short-source", type=Path, default=Path(r"D:\oro_top40_shortmem_OLD"))
    ap.add_argument("--out", type=Path, default=Path(r"D:\ORO_ALLSESSIONS_FROM_TEST_CLEAN"))
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    builder_v1 = here / "build_allsessions_from_test_clean_v1.py"
    if not builder_v1.exists():
        raise SystemExit(f"BUILDER_V1_MISSING: {builder_v1}")

    short = args.short_source.resolve()
    if not short.exists():
        raise SystemExit(f"SHORT_SOURCE_MISSING: {short}")

    git = shutil.which("git")
    if git is None:
        raise SystemExit("GIT_NOT_FOUND_IN_PATH")

    with tempfile.TemporaryDirectory(prefix="oro_engine_clean_") as tmp_raw:
        tmp = Path(tmp_raw)
        engine = tmp / "oro"
        print(f"CLONING_CANONICAL_ENGINE={CANONICAL_ENGINE_REPO}", flush=True)
        subprocess.run(
            [git, "clone", "--depth", "1", "--branch", "main", CANONICAL_ENGINE_REPO, str(engine)],
            check=True,
        )
        if not (engine / "xau_bot").exists():
            raise SystemExit(f"CLONED_ENGINE_INVALID: {engine / 'xau_bot'}")
        print("CANONICAL_ENGINE_CLONE=PASS", flush=True)

        cmd = [
            sys.executable,
            str(builder_v1),
            "--engine-source",
            str(engine),
            "--short-source",
            str(short),
            "--out",
            str(args.out.resolve()),
        ]
        subprocess.run(cmd, check=True)

    print("CLEAN_FROM_TEST_REMOTE_ENGINE_BUILD=PASS")
    print(f"OUTPUT={args.out.resolve()}")
    print("ENGINE_SOURCE=CANONICAL_GITHUB_MAIN")
    print("OLD_FINAL_FOLDER_USED=NO")


if __name__ == "__main__":
    main()
