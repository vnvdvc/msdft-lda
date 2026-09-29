"""Integration tests for sparse CI extraction (T012)."""

from __future__ import annotations

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import fci

from msref.pyscf_adapter import full_ci_vector_to_sparse, sparse_ci_to_dense
from .helpers import active_integrals, h2_mf


@pytest.mark.integration
def test_sparse_roundtrip_exact_at_zero_screen_i004():
    mol, mf = h2_mf()
    h1e, eri = active_integrals(mf, norb=2)
    e, ci = fci.direct_spin1.kernel(h1e, eri, 2, (1, 1), nroots=2)
    ci0 = ci[0]
    keys, coeffs, frac = full_ci_vector_to_sparse(ci0, 2, (1, 1), screen=0.0)
    assert frac == pytest.approx(1.0, abs=1e-12)
    dense = sparse_ci_to_dense(keys, coeffs, 2, (1, 1))
    assert np.allclose(dense, ci0, atol=1e-12)


@pytest.mark.integration
def test_sparse_norm_tracked():
    mol, mf = h2_mf()
    h1e, eri = active_integrals(mf, norb=2)
    e, ci = fci.direct_spin1.kernel(h1e, eri, 2, (1, 1), nroots=2)
    ci0 = ci[0]
    keys, coeffs, frac_full = full_ci_vector_to_sparse(ci0, 2, (1, 1), screen=0.0)
    keys2, coeffs2, frac_screen = full_ci_vector_to_sparse(ci0, 2, (1, 1), screen=0.5)
    assert frac_screen <= frac_full + 1e-12
