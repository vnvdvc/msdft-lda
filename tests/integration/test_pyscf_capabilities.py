"""Integration test for the PySCF capability probe."""

from __future__ import annotations

import pytest

from msref.backend.pyscf_compat import PROBES, probe_capabilities
from msref.exceptions import BackendCapabilityError

pyscf = pytest.importorskip("pyscf")


@pytest.mark.integration
def test_probe_passes_on_target_pyscf():
    report = probe_capabilities()
    assert report.pyscf_version is not None
    # Core required capabilities must be present on the pinned target.
    for name in ("state_average_mix", "trans_rdm1", "spin_square", "make_rdm12",
                 "nonorthogonal_fci_overlap"):
        assert report.available[name], f"{name} missing: {report.details[name]}"


@pytest.mark.integration
def test_missing_required_raises_named_error():
    # A probe of a capability that does not exist must raise with its name.
    with pytest.raises(BackendCapabilityError) as excinfo:
        probe_capabilities(required=["does_not_exist_capability"])
    assert "does_not_exist_capability" in excinfo.value.context["missing"]


@pytest.mark.integration
def test_probe_all_names_registered():
    assert set(PROBES) == {
        "state_average_mix",
        "trans_rdm1",
        "spin_square",
        "make_rdm12",
        "nonorthogonal_fci_overlap",
        "selected_ci_diagnostic",
        "fix_spin",
    }
