"""H2 release benchmark scientific test (T057)."""

from __future__ import annotations

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")

from msref.benchmarks.h2 import build_h2_ensemble
from msref.config import MSRefConfig
from msref.driver import ensemble_driver


@pytest.mark.scientific
def test_h2_ensemble_release_equilibrium():
    cfg = MSRefConfig()
    cfg.active_space_ensemble.n_active_spaces = 8
    mf, prob = build_h2_ensemble(r=0.74, cfg=cfg)
    fam, evals, report = ensemble_driver(prob)
    # projector and matrix-1RDM are stable at equilibrium
    assert report.max_projector_distance < 1e-4
    assert report.max_gamma_error < 1e-4
    assert report.max_spin_square_error < 1e-5
    # support stays below the benchmark cap
    assert len(fam) <= 50


@pytest.mark.scientific
def test_h2_ensemble_dissociation_static_correlation():
    """At the dissociated limit the two sigma orbitals each carry ~1 electron."""
    cfg = MSRefConfig()
    cfg.active_space_ensemble.n_active_spaces = 8
    mf, prob = build_h2_ensemble(r=4.0, cfg=cfg)
    fam, evals, report = ensemble_driver(prob)
    nominal = prob.teachers[prob.nominal_variant.variant_id]
    diag = np.real(np.diag(nominal.gamma1_parent[0, 0]))
    # fully dissociated singlet: sigma_g and sigma_u each ~1 electron
    assert diag[0] == pytest.approx(1.0, abs=0.1)
    assert diag[1] == pytest.approx(1.0, abs=0.1)
