"""Remove disposable generated artifacts without touching runtime data."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DISPOSABLE_DIRS = (
    PROJECT_ROOT / "submission_diverse",
    PROJECT_ROOT / "submission_perfect",
    PROJECT_ROOT / "data" / "submission_final",
)
DISPOSABLE_OUTPUT_PATTERNS = (
    "submission_cli.csv",
    "submission_DEMO_001.csv",
    "submission_kis_smoke_001.csv",
    "submission_mvp_batch.csv",
    "submission_test_kis.csv",
)


def collect_targets() -> list[Path]:
    targets = [path for path in DISPOSABLE_DIRS if path.exists()]
    targets.extend(
        PROJECT_ROOT / "outputs" / name
        for name in DISPOSABLE_OUTPUT_PATTERNS
        if (PROJECT_ROOT / "outputs" / name).exists()
    )
    return targets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Actually remove targets")
    args = parser.parse_args()

    targets = collect_targets()
    if not targets:
        print("No disposable generated artifacts found.")
        return

    for target in targets:
        action = "REMOVE" if args.apply else "DRY-RUN"
        print(f"[{action}] {target.relative_to(PROJECT_ROOT)}")
        if args.apply:
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()


if __name__ == "__main__":
    main()