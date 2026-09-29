"""Fidelity/stability metrics: projector, energy, density, family, spin (T032/T033)."""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

from .exceptions import InputValidationError


def principal_angles(y_ref: np.ndarray, y_test: np.ndarray) -> np.ndarray:
    """Principal angles (radians) between two orthonormal column frames."""
    y_ref = _orthonormalize(y_ref)
    y_test = _orthonormalize(y_test)
    o = y_ref.conj().T @ y_test
    sigma = np.linalg.svd(o, compute_uv=False)
    sigma = np.clip(sigma, 0.0, 1.0)
    return np.arccos(sigma)


def projector_distance(
    y_ref: np.ndarray, y_test: np.ndarray
) -> tuple[float, float, np.ndarray]:
    """Return ``(dP_op, dP_frobenius, singular_values)``.

    ``dP_op = sqrt(1 - sigma_min^2)`` and ``dP_fro = sqrt(sum(1 - sigma_i^2))``.
    """
    y_ref = _orthonormalize(y_ref)
    y_test = _orthonormalize(y_test)
    o = y_ref.conj().T @ y_test
    sigma = np.linalg.svd(o, compute_uv=False)
    sigma = np.clip(sigma, 0.0, 1.0)
    if float(np.min(sigma)) > 1.0 + 1e-8:
        raise InputValidationError("overlap singular value exceeds 1 (normalization bug)")
    d_op = float(np.sqrt(1.0 - float(np.min(sigma)) ** 2))
    d_fro = float(np.sqrt(np.sum(1.0 - sigma ** 2)))
    return d_op, d_fro, sigma


def _orthonormalize(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    q, _ = np.linalg.qr(y)
    return q


def hellinger_distance(p: Mapping, q: Mapping) -> float:
    """Hellinger distance between two non-negative distributions (normalized)."""
    keys = set(p) | set(q)
    pv = np.array([p.get(k, 0.0) for k in keys], dtype=float)
    qv = np.array([q.get(k, 0.0) for k in keys], dtype=float)
    pv = pv / pv.sum() if pv.sum() > 0 else pv
    qv = qv / qv.sum() if qv.sum() > 0 else qv
    return float(np.linalg.norm(np.sqrt(pv) - np.sqrt(qv)) / np.sqrt(2.0))


def weighted_jaccard(p: Mapping, q: Mapping) -> float:
    """Weighted Jaccard similarity ``sum min / sum max`` (in ``[0,1]``)."""
    keys = set(p) | set(q)
    num = sum(min(p.get(k, 0.0), q.get(k, 0.0)) for k in keys)
    den = sum(max(p.get(k, 0.0), q.get(k, 0.0)) for k in keys)
    if den == 0.0:
        return 1.0
    return float(num / den)


def spin_square_error(ss: float, spin_s: float) -> float:
    """Return ``|<S^2> - S(S+1)|``."""
    return abs(float(ss) - spin_s * (spin_s + 1.0))


def sa_energy_error(
    state_weights: Sequence[float], e_selected: np.ndarray, e_teacher: np.ndarray
) -> float:
    """State-averaged energy error ``|sum_A w_A (E_sel - E_teach)|``."""
    w = np.asarray(state_weights, dtype=float)
    return float(abs(np.dot(w, np.asarray(e_selected) - np.asarray(e_teacher))))


def sa_energy_span(e_selected: np.ndarray, state_weights: Sequence[float]) -> float:
    """State-averaged energy (single value)."""
    return float(np.dot(np.asarray(state_weights), np.asarray(e_selected)))


__all__ = [
    "principal_angles",
    "projector_distance",
    "hellinger_distance",
    "weighted_jaccard",
    "spin_square_error",
    "sa_energy_error",
    "sa_energy_span",
]
