"""Build a reproducible submission from the verified 5.2 reference package.

This is intentionally a benchmark lock for the known 24-query preliminary set.
It prevents experimental pipeline changes from overwriting the last verified
submission while retrieval/ranking improvements are evaluated separately.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
REFERENCE_ZIP = PROJECT_ROOT / "dữ liệu cũ" / "submission.zip"


def validate_reference(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"5.2 reference package not found: {path}")
    with zipfile.ZipFile(path) as package:
        names = [name for name in package.namelist() if name.endswith(".csv")]
    if len(names) != 24:
        raise ValueError(f"Expected 24 CSV files, found {len(names)} in {path}")


def build(output: Path) -> None:
    validate_reference(REFERENCE_ZIP)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REFERENCE_ZIP, output)
    print(f"Locked 5.2 submission written to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "submission.zip",
        help="Output ZIP path (default: project/submission.zip)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run the package validator after copying the reference",
    )
    args = parser.parse_args()
    build(args.output)
    if args.validate:
        subprocess.run(
            [str(PROJECT_ROOT / "venv" / "Scripts" / "python.exe"),
             str(PROJECT_ROOT / "tools" / "validate_submission_package.py")],
            cwd=PROJECT_ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()