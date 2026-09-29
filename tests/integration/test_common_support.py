"""Integration tests for common-support multi-variant optimization (T054)."""

from __future__ import annotations

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import gto, scf

from msref import models
from msref.config import MSRefConfig
from msref.benchmarks.common import build_ensemble_problem
from msref.driver import build_ensemble_pool, ensemble_driver, solve_variant_support


def _h2_problem(basis="6-31g", buffer=None):
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis=basis, unit="angstrom", verbose=0)
    mf = scf.RHF(mol).run()
    cfg = MSRefConfig()
    cfg.active_space_ensemble.n_active_spaces = 8
    block = models.StateBlockSpec(block_id="s", spin_S=0.0, ms2=0, nroots=2, weights=[0.5, 0.5])
    return build_ensemble_problem(mf, ncas=2, nelecas=2, state_blocks=[block],
                                  weights=[0.5, 0.5], cfg=cfg, buffer=buffer)


@pytest.mark.integration
def test_common_support_converges():
    prob = _h2_problem()
    fam, evals, report = ensemble_driver(prob)
    assert len(fam) >= 1
    # projector + density are stable (energy may vary with active space)
    assert report.max_projector_distance < 1e-4
    assert report.max_gamma_error < 1e-4
    # all variants represented in the evaluation
    assert len(evals) == len(prob.variants)


@pytest.mark.integration
def test_common_support_family_union():
    prob = _h2_problem()
    pool = build_ensemble_pool(prob)
    fam, evals, report = ensemble_driver(prob)
    # the common support solves every variant with a consistent family set
    for variant in prob.variants:
        ev = solve_variant_support(prob, variant, fam, prob.cfg)
        assert ev is not None
        assert ev.n_determinants >= 1
