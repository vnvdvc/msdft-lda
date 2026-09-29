"""Orbital gauge alignment: Procrustes rotation, phase fixing, principal angles.

For Tier-I fixed-orbital variants there is no orbital reoptimization, so the
parent gauge is exact and no CI transformation is required.  These utilities are
still used to (a) establish a deterministic gauge and (b) later, in Tier II, to
align reoptimized orbitals to the parent reference.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .exceptions import OrbitalAlignmentError


def overlap_matrix(
    c_ref: np.ndarray, s_ao: np.ndarray, c_test: np.ndarray
) -> np.ndarray:
    """Return ``M = C_ref^dagger S_AO C_test`` (orbital overlap matrix)."""
    return np.asarray(c_ref).conj().T @ np.asarray(s_ao) @ np.asarray(c_test)


def _svd_procrustes(m: np.ndarray):
    u, s, vh = np.linalg.svd(m, full_matrices=False)
    r = vh.conj().T @ u.conj().T
    return s, r


def procrustes_align(
    c_ref: np.ndarray,
    s_ao: np.ndarray,
    c_test: np.ndarray,
    min_sigma: float = 0.90,
):
    """Align a test orbital block to a reference block.

    Returns ``(aligned_C_test, singular_values)``.  Raises
    ``OrbitalAlignmentError`` if ``min(sigma) < min_sigma``.
    """
    c_ref = np.asarray(c_ref)
    c_test = np.asarray(c_test)
    m = overlap_matrix(c_ref, s_ao, c_test)
    sigma, r = _svd_procrustes(m)
    if float(np.min(sigma)) < min_sigma:
        raise OrbitalAlignmentError(
            "orbital alignment unresolved: low overlap",
            {"min_sigma": float(np.min(sigma)), "threshold": min_sigma},
        )
    aligned = c_test @ r
    return aligned, sigma


def fix_phases(
    c_ref: np.ndarray,
    s_ao: np.ndarray,
    c_test: np.ndarray,
    tol: float = 1e-8,
) -> np.ndarray:
    """Fix orbital phases so each column has non-negative real overlap with ref.

    If a column has a (numerically) zero overlap magnitude it is left unchanged
    rather than choosing an arbitrary phase.
    """
    c_ref = np.asarray(c_ref)
    c_test = np.asarray(c_test).copy()
    ov = overlap_matrix(c_ref, s_ao, c_test)  # [n, n]
    diag = np.diag(ov)
    for p in range(c_test.shape[1]):
        op = diag[p]
        mag = abs(op)
        if mag > tol and op.real < 0:
            c_test[:, p] *= -1
        elif mag > tol and abs(op.imag) > tol:
            # rotate so overlap is real and positive
            phase = op / mag
            c_test[:, p] /= phase
    return c_test


def principal_angles(
    c_ref: np.ndarray, s_ao: np.ndarray, c_test: np.ndarray
) -> np.ndarray:
    """Return principal angles (radians) between two orbital subspaces."""
    m = overlap_matrix(c_ref, s_ao, c_test)
    sigma = np.linalg.svd(m, compute_uv=False)
    sigma = np.clip(sigma, 0.0, 1.0)
    return np.arccos(sigma)


def align_orbital_gauge(
    c_ref: np.ndarray,
    s_ao: np.ndarray,
    c_test: np.ndarray,
    min_sigma: float = 0.90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full gauge alignment: Procrustes + phase fix.

    Returns ``(aligned_C_test, singular_values, principal_angles_rad)``.
    """
    aligned, sigma = procrustes_align(c_ref, s_ao, c_test, min_sigma=min_sigma)
    aligned = fix_phases(c_ref, s_ao, aligned)
    theta = np.arccos(np.clip(sigma, 0.0, 1.0))
    return aligned, sigma, theta


__all__ = [
    "overlap_matrix",
    "procrustes_align",
    "fix_phases",
    "principal_angles",
    "align_orbital_gauge",
]
