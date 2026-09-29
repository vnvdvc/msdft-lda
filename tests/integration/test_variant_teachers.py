"""Integration tests for fixed-orbital CASCI variant teachers (T051)."""

from __future__ import annotations

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import gto, scf

from msref import models
from msref.config import MSRefConfig
from msref.benchmarks.common import build_ensemble_problem
from msref.pyscf_adapter import build_core_and_active, run_variant_casci


@pytest.mark.integration
def test_nominal_variant_matches_casci():
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="6-31g", unit="angstrom", verbose=0)
    mf = scf.RHF(mol).run()
    cfg = MSRefConfig()
    block = models.StateBlockSpec(block_id="s", spin_S=0.0, ms2=0, nroots=1, weights=[1.0])
    prob = build_ensemble_problem(mf, ncas=2, nelecas=2, state_blocks=[block],
                                  weights=[1.0], cfg=cfg)
    nominal = prob.variants[0]
    teacher = prob.teachers[nominal.variant_id]
    # reference: full active-space FCI energy
    e_core, h_act, eri_act = build_core_and_active(
        prob.h1e_parent, prob.eri_parent, nominal.inactive_orbitals, nominal.active_orbitals
    )
    from pyscf import fci
    e_fci, _ = fci.direct_spin1.kernel(h_act, eri_act, len(nominal.active_orbitals),
                                       teacher.nelec_act, nroots=1)
    assert teacher.energies_eh[0] == pytest.approx(float(e_fci) + e_core, abs=1e-8)


@pytest.mark.integration
def test_variant_electron_and_spin():
    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="6-31g", unit="angstrom", verbose=0)
    mf = scf.RHF(mol).run()
    cfg = MSRefConfig()
    cfg.active_space_ensemble.n_active_spaces = 6
    block = models.StateBlockSpec(block_id="s", spin_S=0.0, ms2=0, nroots=2, weights=[0.5, 0.5])
    prob = build_ensemble_problem(mf, ncas=2, nelecas=2, state_blocks=[block],
                                  weights=[0.5, 0.5], cfg=cfg)
    for v in prob.variants:
        t = prob.teachers[v.variant_id]
        assert t.nelec_act == (1, 1)
        assert np.allclose(t.spin_square, 0.0, atol=1e-5)
        # embedded CI vectors sum to unit norm
        for cv in t.ci_vectors:
            assert abs(np.sum(np.abs(cv.coefficients) ** 2) - 1.0) < 1e-8


@pytest.mark.integration
def test_frozen_core_energy_matches_pyscf_casci():
    """Regression: the frozen-core Coulomb/exchange indexing in
    ``build_core_and_active`` must reproduce PySCF's native CASCI total energy.

    This guards against the J=(ii|jj)/K=(ij|ij) vs J=(ij|ij)/K=(ij|ji)
    chemist-notation mix-up that corrupted N2/O2/C2 ensemble energies.
    """
    from pyscf import ao2mo, fci, mcscf

    mol = gto.M(atom="N 0 0 0; N 0 0 1.10", basis="cc-pvdz", unit="angstrom", verbose=0)
    mf = scf.RHF(mol).run()
    ncore = 5
    mc = mcscf.CASCI(mf, 4, 4)
    mc.kernel()

    mo = mf.mo_coeff
    h1e = mo.T @ mf.get_hcore() @ mo
    eri = ao2mo.restore(1, ao2mo.incore.full(mol.intor("int2e"), mo), mo.shape[1])
    e_core, h_act, eri_act = build_core_and_active(h1e, eri, list(range(ncore)), [5, 6, 7, 8])
    e, _ = fci.direct_spin1.kernel(h_act, eri_act, 4, (2, 2), nroots=2)
    total = float(e[0]) + e_core + mol.energy_nuc()
    assert total == pytest.approx(mc.e_tot, abs=1e-8)

