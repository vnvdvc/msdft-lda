"""Integration tests for teacher RDM extraction (T015)."""

from __future__ import annotations

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import fci

from msref import models
from msref.config import MSRefConfig
from .helpers import h2_mf


@pytest.mark.integration
def test_teacher_rdm_trace_and_hermiticity_i005_i006():
    from msref.pyscf_adapter import build_teacher

    mol, mf = h2_mf()
    cfg = MSRefConfig()
    block = models.StateBlockSpec(block_id="s", spin_S=0.0, ms2=0, nroots=2, weights=[0.5, 0.5])
    res = build_teacher(mf, 2, 2, [block], [0.5, 0.5], mo0=mf.mo_coeff, cfg=cfg)

    g = res.gamma1_parent
    n = g.shape[0]
    assert g.shape == (2, 2, 2, 2)
    # diagonal trace = electron count (2)
    for a in range(n):
        assert np.trace(g[a, a].real) == pytest.approx(2.0, abs=1e-10)
    # hermiticity relation gamma[B,A] = gamma[A,B]^dagger
    for a in range(n):
        for b in range(n):
            assert np.allclose(g[b, a], g[a, b].conj().T, atol=1e-12)


@pytest.mark.integration
def test_teacher_rdm_matches_pyscf():
    from msref.pyscf_adapter import build_teacher
    from .helpers import active_integrals

    mol, mf = h2_mf()
    h1e, eri = active_integrals(mf, norb=2)
    cfg = MSRefConfig()
    block = models.StateBlockSpec(block_id="s", spin_S=0.0, ms2=0, nroots=1, weights=[1.0])
    res = build_teacher(mf, 2, 2, [block], [1.0], mo0=mf.mo_coeff, cfg=cfg)
    # teacher gamma should match direct make_rdm1 on the same CI vector
    from msref.pyscf_adapter import sparse_ci_to_dense
    ci = sparse_ci_to_dense(res.ci_vectors[0].determinant_keys,
                            res.ci_vectors[0].coefficients, 2, (1, 1))
    g_py = fci.direct_spin1.make_rdm1(ci, 2, (1, 1))
    assert np.allclose(res.gamma1_parent[0, 0], g_py, atol=1e-10)
