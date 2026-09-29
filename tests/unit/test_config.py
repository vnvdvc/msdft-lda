"""Unit tests for the dataclass configuration models and YAML loading."""

from __future__ import annotations

import os

import pytest

from msref.config import MSRefConfig
from msref.exceptions import InputValidationError

_EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "..", "examples")


@pytest.mark.unit
def test_default_config_validates():
    cfg = MSRefConfig.from_yaml(os.path.join(_EXAMPLES, "default_config.yaml"))
    assert cfg.schema_version == "1.0"
    assert cfg.profile == "production"
    assert cfg.acceptance.tau_projector == pytest.approx(0.020)
    assert cfg.static_leakage.tau_static_max == pytest.approx(0.020)
    # calibration ranges and profiles are preserved
    assert "acceptance.tau_projector" in cfg.calibration_ranges
    assert "development_overrides" in cfg.profiles


@pytest.mark.unit
def test_negative_tolerance_fails():
    d = {
        "static_leakage": {"tau_static_max": -0.1},
        "support_seed": {"captured_average_weight": 0.99},
    }
    cfg = MSRefConfig.from_dict(d)
    with pytest.raises(InputValidationError):
        cfg.validate()


@pytest.mark.unit
def test_probability_out_of_range_fails():
    cfg = MSRefConfig.from_dict({"support_seed": {"captured_average_weight": 1.5}})
    with pytest.raises(InputValidationError):
        cfg.validate()


@pytest.mark.unit
def test_config_hash_deterministic():
    a = MSRefConfig.from_yaml(os.path.join(_EXAMPLES, "default_config.yaml"))
    b = MSRefConfig.from_yaml(os.path.join(_EXAMPLES, "default_config.yaml"))
    assert a.config_hash() == b.config_hash()


@pytest.mark.unit
def test_config_hash_reordered_keys_identical():
    yaml_a = "static_leakage:\n  tau_static_max: 0.02\nsupport_seed:\n  captured_average_weight: 0.99\n"
    yaml_b = "support_seed:\n  captured_average_weight: 0.99\nstatic_leakage:\n  tau_static_max: 0.02\n"
    a = MSRefConfig.from_yaml_string(yaml_a)
    b = MSRefConfig.from_yaml_string(yaml_b)
    assert a.config_hash() == b.config_hash()


@pytest.mark.unit
def test_profile_overrides_applied_without_mutation():
    cfg = MSRefConfig.from_dict(
        {
            "profile": "development",
            "profiles": {
                "development_overrides": {"selected_solver.max_selection_iterations": 15},
                "strict_overrides": {},
            },
        }
    )
    assert cfg.selected_solver.max_selection_iterations == 25  # untouched in source data
    cfg.apply_profile_overrides()
    assert cfg.selected_solver.max_selection_iterations == 15


@pytest.mark.unit
def test_strict_overrides_applied():
    cfg = MSRefConfig.from_dict(
        {
            "profile": "strict",
            "profiles": {
                "strict_overrides": {"acceptance.tau_projector": 0.010},
            },
        }
    )
    cfg.apply_profile_overrides()
    assert cfg.acceptance.tau_projector == pytest.approx(0.010)
