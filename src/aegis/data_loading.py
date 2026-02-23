"""
Dataset loading utilities for AEGIS.

This module is intentionally limited to path resolution and dataset validation.
It does not alter training/evaluation metrics or model parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable, Optional, Sequence

_SUPPORTED_EXTENSIONS = (".nii", ".nii.gz")
_ADAM_DIR = "adam data"
_INDIAN_DIR = "Data_Set"


class DatasetLoadError(ValueError):
    """Raised when dataset folders/files are inconsistent or incomplete."""


@dataclass(frozen=True)
class SessionPair:
    """A single image/mask pair for one session."""

    session_id: str
    image_path: Path
    mask_path: Path


@dataclass(frozen=True)
class DatasetBundle:
    """
    Validated references for both datasets.

    Variables are explicit by phase/session so downstream code can use
    consistent names without guessing.
    """

    adam_sessions: list[SessionPair]
    indian_car_sessions: list[SessionPair]
    indian_ncar_images: list[Path]


def _repo_root_from_here() -> Path:
    # src/aegis/data_loading.py -> repo root
    return Path(__file__).resolve().parents[2]


def _resolve_root(base_dir: Optional[Path | str]) -> Path:
    if base_dir is None:
        return _repo_root_from_here()
    return Path(base_dir).expanduser().resolve()


def _resolve_existing_dir(parent: Path, candidates: Sequence[str], label: str) -> Path:
    for candidate in candidates:
        path = parent / candidate
        if path.exists() and path.is_dir():
            return path.resolve()
    joined = ", ".join(f"'{candidate}'" for candidate in candidates)
    raise FileNotFoundError(
        f"{label} not found under '{parent.resolve()}'. Tried: {joined}."
    )


def _is_supported(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".nii") or name.endswith(".nii.gz")


def _scan_medical_files(folder: Path, label: str) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"{label} directory does not exist: '{folder.resolve()}'.")
    if not folder.is_dir():
        raise NotADirectoryError(
            f"{label} path exists but is not a directory: '{folder.resolve()}'."
        )

    files = sorted(path for path in folder.iterdir() if path.is_file() and _is_supported(path))
    if not files:
        raise DatasetLoadError(
            f"{label} has no supported files in '{folder.resolve()}'. "
            f"Expected extensions: {', '.join(_SUPPORTED_EXTENSIONS)}."
        )
    return files


def _strip_medical_suffix(filename: str) -> str:
    if filename.lower().endswith(".nii.gz"):
        return filename[:-7]
    if filename.lower().endswith(".nii"):
        return filename[:-4]
    return Path(filename).stem


def _canonical_session_id(path: Path) -> str:
    stem = _strip_medical_suffix(path.name)
    normalized = re.sub(r"\s+", "_", stem.strip().lower())
    normalized = normalized.replace("ia_subm_", "ia_sub_")
    normalized = normalized.replace("ia-subm-", "ia-sub-")
    normalized = re.sub(r"^(sub|mask)[-_]*", "", normalized)
    normalized = re.sub(r"[-_]*(mask|masks|orig|image|images)$", "", normalized)
    return normalized


def _build_unique_map(paths: Iterable[Path], label: str) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for path in paths:
        key = _canonical_session_id(path)
        if key in output:
            raise DatasetLoadError(
                f"Duplicate {label} session id '{key}' from "
                f"'{output[key].name}' and '{path.name}'."
            )
        output[key] = path.resolve()
    return output


def _pair_images_and_masks(
    image_paths: Sequence[Path], mask_paths: Sequence[Path], label: str
) -> list[SessionPair]:
    image_map = _build_unique_map(image_paths, f"{label} image")
    mask_map = _build_unique_map(mask_paths, f"{label} mask")

    missing_masks = sorted(session_id for session_id in image_map if session_id not in mask_map)
    missing_images = sorted(session_id for session_id in mask_map if session_id not in image_map)
    if missing_masks or missing_images:
        details: list[str] = []
        if missing_masks:
            details.append(f"missing masks for: {', '.join(missing_masks)}")
        if missing_images:
            details.append(f"missing images for: {', '.join(missing_images)}")
        raise DatasetLoadError(f"{label} pairing mismatch ({'; '.join(details)}).")

    return [
        SessionPair(
            session_id=session_id,
            image_path=image_map[session_id],
            mask_path=mask_map[session_id],
        )
        for session_id in sorted(image_map)
    ]


def load_adam_sessions(base_dir: Optional[Path | str] = None) -> list[SessionPair]:
    """
    Load and validate ADAM sessions from '<repo>/adam data/'.
    """

    root = _resolve_root(base_dir)
    adam_root = _resolve_existing_dir(root, (_ADAM_DIR,), "ADAM dataset root")
    images_dir = _resolve_existing_dir(adam_root, ("orig_images",), "ADAM images directory")
    masks_dir = _resolve_existing_dir(adam_root, ("masks",), "ADAM masks directory")

    image_files = _scan_medical_files(images_dir, "ADAM images")
    mask_files = _scan_medical_files(masks_dir, "ADAM masks")
    return _pair_images_and_masks(image_files, mask_files, label="ADAM")


def load_indian_dataset(base_dir: Optional[Path | str] = None) -> tuple[list[SessionPair], list[Path]]:
    """
    Load and validate the Indian dataset from '<repo>/Data_Set/'.

    Returns:
        (indian_car_sessions, indian_ncar_images)
    """

    root = _resolve_root(base_dir)
    indian_root = _resolve_existing_dir(root, (_INDIAN_DIR,), "Indian dataset root")

    car_root = _resolve_existing_dir(indian_root, ("CAR_Dataset",), "CAR dataset directory")
    car_images_dir = _resolve_existing_dir(
        car_root, ("CAR_images", "CAR_Images"), "CAR images directory"
    )
    car_masks_dir = _resolve_existing_dir(
        car_root, ("CAR_masks", "CAR_Masks"), "CAR masks directory"
    )

    ncar_root = _resolve_existing_dir(indian_root, ("NCAR_Dataset",), "NCAR dataset directory")
    ncar_images_dir = _resolve_existing_dir(
        ncar_root, ("NCAR_images", "NCAR_Images"), "NCAR images directory"
    )

    car_images = _scan_medical_files(car_images_dir, "Indian CAR images")
    car_masks = _scan_medical_files(car_masks_dir, "Indian CAR masks")
    indian_car_sessions = _pair_images_and_masks(car_images, car_masks, label="Indian CAR")

    indian_ncar_images = _scan_medical_files(ncar_images_dir, "Indian NCAR images")
    return indian_car_sessions, [path.resolve() for path in indian_ncar_images]


def load_dataset_bundle(base_dir: Optional[Path | str] = None) -> DatasetBundle:
    """
    Validate and load both datasets using stable variable names.
    """

    adam_sessions = load_adam_sessions(base_dir=base_dir)
    indian_car_sessions, indian_ncar_images = load_indian_dataset(base_dir=base_dir)
    return DatasetBundle(
        adam_sessions=adam_sessions,
        indian_car_sessions=indian_car_sessions,
        indian_ncar_images=indian_ncar_images,
    )
