"""Integration tests for selected 1-RDM / transition 1-RDM (T028)."""

from __future__ import annotations

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import fci

from msref.density import trans_rdm1_selected
from msref.pyscf_adapter import full_ci_vector_to_sparse
from msref.spin_completion import all_determinants
from .helpers import active_integrals, h2_mf


@pytest.mark.integration
def test_selected_rdm_matches_pyscf_i010():
    mol, mf = h2_mf()
    h1e, eri = active_integrals(mf, norb=2)
    e, ci = fci.direct_spin1.kernel(h1e, eri, 2, (1, 1), nroots=2)
    dets = all_determinants(2, 1, 1)

    def dense(civ):
        d = np.zeros(len(dets), dtype=complex)
        keys, coeffs, _ = full_ci_vector_to_sparse(civ, 2, (1, 1), 0.0)
        idx = {k: i for i, k in enumerate(dets)}
        for k, c in zip(keys, coeffs):
            d[idx[k]] = c
        return d

    d0 = dense(ci[0])
    d1 = dense(ci[1])
    # state RDM
    g = trans_rdm1_selected(d0, d0, dets, 2)
    g_py = fci.direct_spin1.make_rdm1(ci[0], 2, (1, 1))
    assert np.allclose(g, g_py, atol=1e-11)
    # transition RDM
    g_t = trans_rdm1_selected(d0, d1, dets, 2)
    g_t_py = fci.direct_spin1.trans_rdm1(ci[0], ci[1], 2, (1, 1))
    assert np.allclose(g_t, g_t_py, atol=1e-11)


@pytest.mark.integration
def test_transition_hermiticity_u025():
    mol, mf = h2_mf()
    h1e, eri = active_integrals(mf, norb=2)
    e, ci = fci.direct_spin1.kernel(h1e, eri, 2, (1, 1), nroots=2)
    dets = all_determinants(2, 1, 1)
    idx = {k: i for i, k in enumerate(dets)}
    d0 = np.zeros(len(dets), dtype=complex)
    d1 = np.zeros(len(dets), dtype=complex)
    for civ, d in ((ci[0], d0), (ci[1], d1)):
        keys, coeffs, _ = full_ci_vector_to_sparse(civ, 2, (1, 1), 0.0)
        for k, c in zip(keys, coeffs):
            d[idx[k]] = c
    g_ab = trans_rdm1_selected(d0, d1, dets, 2)
    g_ba = trans_rdm1_selected(d1, d0, dets, 2)
    assert np.allclose(g_ba, g_ab.conj().T, atol=1e-12)
