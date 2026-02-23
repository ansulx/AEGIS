"""Tests for dataset loading and validation helpers."""

from pathlib import Path

import pytest

from aegis.data_loading import DatasetLoadError, load_dataset_bundle, load_indian_dataset


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test")


def test_load_dataset_bundle_success(tmp_path: Path) -> None:
    _touch(tmp_path / "adam data" / "orig_images" / "patient_001.nii")
    _touch(tmp_path / "adam data" / "masks" / "patient_001_mask.nii")

    _touch(tmp_path / "Data_Set" / "CAR_Dataset" / "CAR_images" / "case01.nii.gz")
    _touch(tmp_path / "Data_Set" / "CAR_Dataset" / "CAR_masks" / "case01_mask.nii.gz")
    _touch(tmp_path / "Data_Set" / "NCAR_Dataset" / "NCAR_images" / "healthy01.nii.gz")

    bundle = load_dataset_bundle(base_dir=tmp_path)
    assert len(bundle.adam_sessions) == 1
    assert len(bundle.indian_car_sessions) == 1
    assert len(bundle.indian_ncar_images) == 1


def test_adam_pairing_mismatch_raises(tmp_path: Path) -> None:
    _touch(tmp_path / "adam data" / "orig_images" / "patient_001.nii")
    _touch(tmp_path / "adam data" / "masks" / "patient_002_mask.nii")

    _touch(tmp_path / "Data_Set" / "CAR_Dataset" / "CAR_images" / "case01.nii.gz")
    _touch(tmp_path / "Data_Set" / "CAR_Dataset" / "CAR_masks" / "case01_mask.nii.gz")
    _touch(tmp_path / "Data_Set" / "NCAR_Dataset" / "NCAR_images" / "healthy01.nii.gz")

    with pytest.raises(DatasetLoadError):
        load_dataset_bundle(base_dir=tmp_path)


def test_indian_folder_name_variants_supported(tmp_path: Path) -> None:
    _touch(tmp_path / "Data_Set" / "CAR_Dataset" / "CAR_Images" / "case01.nii.gz")
    _touch(tmp_path / "Data_Set" / "CAR_Dataset" / "CAR_Masks" / "case01_mask.nii.gz")
    _touch(tmp_path / "Data_Set" / "NCAR_Dataset" / "NCAR_Images" / "healthy01.nii.gz")

    car_sessions, ncar_images = load_indian_dataset(base_dir=tmp_path)
    assert len(car_sessions) == 1
    assert len(ncar_images) == 1


def test_indian_realworld_session_prefix_variants(tmp_path: Path) -> None:
    _touch(tmp_path / "Data_Set" / "CAR_Dataset" / "CAR_images" / "ia_sub_001.nii.gz")
    _touch(tmp_path / "Data_Set" / "CAR_Dataset" / "CAR_masks" / "ia_subm_001.nii.gz")
    _touch(tmp_path / "Data_Set" / "NCAR_Dataset" / "NCAR_images" / "Subject_001.nii.gz")

    car_sessions, ncar_images = load_indian_dataset(base_dir=tmp_path)
    assert len(car_sessions) == 1
    assert car_sessions[0].session_id == "ia_sub_001"
    assert len(ncar_images) == 1
