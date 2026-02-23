#!/usr/bin/env python3
"""
Validate ADAM + Indian dataset loading for direct script execution.

Usage:
    python scripts/check_data_loading.py
    python scripts/check_data_loading.py --base-dir /path/to/AEGIS
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from aegis.data_loading import DatasetLoadError, load_dataset_bundle


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate AEGIS dataset folder structure and image/mask pairing."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Repository root. Defaults to automatic detection from package location.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        bundle = load_dataset_bundle(base_dir=args.base_dir)
    except (FileNotFoundError, NotADirectoryError, DatasetLoadError) as exc:
        print(f"[ERROR] Dataset validation failed: {exc}")
        return 1

    print("[OK] Dataset validation passed.")
    print(f"ADAM sessions: {len(bundle.adam_sessions)}")
    print(f"Indian CAR sessions: {len(bundle.indian_car_sessions)}")
    print(f"Indian NCAR images: {len(bundle.indian_ncar_images)}")

    if bundle.adam_sessions:
        print(f"Example ADAM session id: {bundle.adam_sessions[0].session_id}")
    if bundle.indian_car_sessions:
        print(f"Example Indian CAR session id: {bundle.indian_car_sessions[0].session_id}")
    if bundle.indian_ncar_images:
        print(f"Example Indian NCAR file: {bundle.indian_ncar_images[0].name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
