"""Safety tests for main CLI configuration handling."""

import pytest

from main import _resolve_pipeline_config, _validate_run_name


def test_invalid_run_name_rejected() -> None:
    with pytest.raises(ValueError):
        _validate_run_name("miccai 2026 full")


def test_locked_pipeline_override_rejected() -> None:
    baseline = {
        "pipeline": {"mc_samples": 20, "require_gpu": True},
        "safety": {"locked_pipeline_keys": ["mc_samples"]},
    }
    main = {"pipeline": {"mc_samples": 30}}

    with pytest.raises(ValueError):
        _resolve_pipeline_config(baseline, main, allow_cpu_override=False)


def test_protected_section_override_rejected() -> None:
    baseline = {"pipeline": {"mc_samples": 20, "require_gpu": True}}
    main = {"metrics": {"dice": "override-attempt"}}

    with pytest.raises(ValueError):
        _resolve_pipeline_config(baseline, main, allow_cpu_override=False)


def test_allow_cpu_override_changes_require_gpu() -> None:
    baseline = {"pipeline": {"mc_samples": 20, "require_gpu": True}}
    main = {"pipeline": {}}

    config, _ = _resolve_pipeline_config(baseline, main, allow_cpu_override=True)
    assert config.require_gpu is False


def test_training_params_passthrough() -> None:
    baseline = {
        "pipeline": {
            "mc_samples": 20,
            "require_gpu": False,
            "num_epochs": 50,
            "learning_rate": 0.001,
            "feature_size": 24,
        }
    }
    main = {"pipeline": {}}
    config, _ = _resolve_pipeline_config(baseline, main, allow_cpu_override=False)
    assert config.num_epochs == 50
    assert config.learning_rate == 0.001
    assert config.feature_size == 24
