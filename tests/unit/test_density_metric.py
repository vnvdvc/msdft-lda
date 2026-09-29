"""Unit tests for AO density transform and orthogonal metric (T029)."""

from __future__ import annotations

import numpy as np
import pytest

from msref.density import matrix_rdm_error, restore_core_density, symmetric_orthogonal_density
from msref.metrics import projector_distance


@pytest.mark.unit
def test_symmetric_orthogonal_trace_preserved():
    """trace(S^{1/2} P S^{1/2}) == trace(S P) (electron-count invariant)."""
    rng = np.random.default_rng(3)
    n = 4
    x = rng.standard_normal((n, n))
    s = x @ x.T + n * np.eye(n)  # positive definite
    p = np.diag([1.0, 2.0, 3.0, 4.0])
    p_orth = symmetric_orthogonal_density(p, s)
    assert np.trace(p_orth) == pytest.approx(np.trace(s @ p), abs=1e-9)


@pytest.mark.unit
def test_orthogonal_metric_invariant_u026():
    """Unitary rotation in the orthonormal basis leaves the Frobenius metric invariant."""
    s = np.eye(3)
    p = np.diag([1.0, 2.0, 3.0])
    u = np.array([[0.6, -0.8, 0.0], [0.8, 0.6, 0.0], [0.0, 0.0, 1.0]])
    p_rot = u @ p @ u.T
    d1 = symmetric_orthogonal_density(p, s)
    d2 = symmetric_orthogonal_density(p_rot, s)
    assert np.linalg.norm(d1, "fro") == pytest.approx(np.linalg.norm(d2, "fro"), abs=1e-12)


@pytest.mark.unit
def test_restore_core_density():
    n = 2
    c_core = np.array([[1.0], [0.0]])
    p = np.zeros((2, 2))
    p_restored = restore_core_density(p, c_core)
    assert np.allclose(p_restored, 2.0 * c_core @ c_core.T)


@pytest.mark.unit
def test_matrix_rdm_error_u025():
    n = 2
    g_sel = np.zeros((n, n, 2, 2), dtype=complex)
    g_teach = np.zeros((n, n, 2, 2), dtype=complex)
    g_teach[0, 0] = np.eye(2)
    eps, max_block, _ = matrix_rdm_error(g_sel, g_teach, np.array([0.5, 0.5]), nelec=2)
    assert eps > 0.0
    assert max_block > 0.0
