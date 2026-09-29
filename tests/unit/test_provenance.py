"""Unit tests for runtime provenance capture."""

from __future__ import annotations

import pytest

from msref.logging_utils import collect_provenance


@pytest.mark.unit
def test_provenance_keys_present():
    prov = collect_provenance(config_hash="abc123")
    for key in (
        "package_version",
        "python_version",
        "numpy_version",
        "scipy_version",
        "pyscf_version",
        "created_utc",
        "git_commit",
    ):
        assert key in prov
    assert prov["package_version"]
    assert prov["python_version"]
    assert prov["numpy_version"] is not None
    assert prov["scipy_version"] is not None
    assert prov["config_hash"] == "abc123"


@pytest.mark.unit
def test_provenance_pyscf_optional_key():
    # pyscf_version key must exist; its value may be None if not installed.
    prov = collect_provenance()
    assert "pyscf_version" in prov


@pytest.mark.unit
def test_provenance_config_hash_stable():
    a = collect_provenance(config_hash="xyz")
    b = collect_provenance(config_hash="xyz")
    assert a["config_hash"] == b["config_hash"]
