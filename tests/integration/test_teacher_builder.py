"""Integration tests for the spin-mixed SA-CASSCF teacher builder (T011)."""

from __future__ import annotations

import numpy as np
import pytest

from msref import models
from msref.config import MSRefConfig

pyscf = pytest.importorskip("pyscf")
from .helpers import h2_mf, o2_mf


@pytest.mark.integration
def test_same_spin_matches_state_average_i002():
    from pyscf import fci, mcscf
    from msref.pyscf_adapter import build_teacher

    mol, mf = h2_mf()
    cfg = MSRefConfig()
    block = models.StateBlockSpec(block_id="s", spin_S=0.0, ms2=0, nroots=2, weights=[0.5, 0.5])
    res = build_teacher(mf, 2, 2, [block], [0.5, 0.5], mo0=mf.mo_coeff, cfg=cfg)
    assert res.converged
    assert res.state_weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(np.abs(res.spin_square) < 1e-5)

    # reference: ordinary state_average with a direct_spin0 (singlet) solver,
    # which is the "same solver" family (both target pure singlets)
    from pyscf import fci
    mc = mcscf.CASSCF(mf, 2, 2)
    mc.fcisolver = fci.direct_spin0.FCISolver(mol)
    mc = mcscf.state_average_(mc, [0.5, 0.5])
    mc.kernel(mf.mo_coeff)
    assert res.energies_eh[0] == pytest.approx(mc.e_states[0], abs=1e-6)
    assert res.energies_eh[1] == pytest.approx(mc.e_states[1], abs=1e-6)


@pytest.mark.integration
def test_mixed_spin_o2_smoke_i003():
    from msref.pyscf_adapter import build_teacher

    mol, mf = o2_mf()
    cfg = MSRefConfig()
    nocc = mol.nelec[0]
    block_t = models.StateBlockSpec(block_id="t", spin_S=1.0, ms2=0, nroots=1, weights=[1 / 3])
    block_s = models.StateBlockSpec(block_id="s", spin_S=0.0, ms2=0, nroots=2, weights=[1 / 3, 1 / 3])
    res = build_teacher(mf, 2, 2, [block_t, block_s], [1 / 3, 1 / 3, 1 / 3],
                        mo0=mf.mo_coeff, cfg=cfg)
    assert res.converged
    assert np.allclose(res.spin_square, [2.0, 0.0, 0.0], atol=1e-4)
