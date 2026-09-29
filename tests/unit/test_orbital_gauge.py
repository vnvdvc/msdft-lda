"""Unit tests for orbital gauge alignment (T013)."""

from __future__ import annotations

import numpy as np
import pytest

from msref.exceptions import OrbitalAlignmentError
from msref.orbital_gauge import (
    align_orbital_gauge,
    fix_phases,
    overlap_matrix,
    principal_angles,
    procrustes_align,
)


def _random_orthogonal(n):
    a = np.random.default_rng(0).standard_normal((n, n))
    q, _ = np.linalg.qr(a)
    return q


@pytest.mark.unit
def test_procrustes_inverts_rotation_u011():
    rng = np.random.default_rng(1)
    n = 4
    c_ref = _random_orthogonal(n)
    q = _random_orthogonal(n)
    c_test = c_ref @ q
    aligned, sigma = procrustes_align(c_ref, np.eye(n), c_test, min_sigma=0.9)
    assert np.allclose(sigma, 1.0, atol=1e-12)
    # aligned should equal c_ref up to a unitary (columns align to c_ref)
    ov = c_ref.T @ aligned
    assert np.allclose(np.abs(np.diag(ov)), 1.0, atol=1e-10)


@pytest.mark.unit
def test_phase_fix_u012():
    n = 3
    c_ref = _random_orthogonal(n)
    signs = np.array([1, -1, 1])
    c_test = c_ref @ np.diag(signs)
    fixed = fix_phases(c_ref, np.eye(n), c_test)
    assert np.allclose(fixed, c_ref, atol=1e-12)


@pytest.mark.unit
def test_low_overlap_raises_u013():
    # reference block spans e1,e2; test block spans e1,e3 -> one near-orthogonal direction
    c_ref = np.array([[1.0, 0], [0, 1.0], [0, 0]])
    c_test = np.array([[1.0, 0], [0, 0], [0, 1.0]])
    with pytest.raises(OrbitalAlignmentError):
        procrustes_align(c_ref, np.eye(3), c_test, min_sigma=0.9)


@pytest.mark.unit
def test_principal_angles_identical():
    c = _random_orthogonal(3)
    theta = principal_angles(c, np.eye(3), c)
    # arccos near 1 is ill-conditioned; allow 1e-6
    assert np.allclose(theta, 0.0, atol=1e-6)


@pytest.mark.unit
def test_overlap_matrix():
    n = 3
    c_ref = _random_orthogonal(n)
    c_test = _random_orthogonal(n)
    s = np.eye(n)
    m = overlap_matrix(c_ref, s, c_test)
    assert m.shape == (n, n)
    assert np.allclose(m, c_ref.T @ s @ c_test)
