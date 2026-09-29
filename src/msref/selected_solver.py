"""Selected-space eigensolver and root recovery (T027).

Diagonalizes the selected Hamiltonian (dense path) and recovers target roots by
spin/symmetry/overlap tracking rather than energy order.  Root normalization and
phase fixing are deterministic.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .exceptions import SelectedDiagonalizationError, SelectedRootMissingError
from .models import DeterminantKey


def solve_selected_sector(
    hamiltonian: np.ndarray,
    nroots: int,
    root_buffer: int = 2,
    eig_tol: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """Dense diagonalization; return ``(energies, coefficients)``.

    ``coefficients`` has shape ``[n_det, n_requested_roots]`` with the lowest
    ``nroots + root_buffer`` eigenvalues.
    """
    n = hamiltonian.shape[0]
    if nroots > n:
        raise SelectedRootMissingError(
            "requested roots exceed selected space dimension",
            {"n_det": n, "nroots": nroots, "root_buffer": root_buffer},
        )
    try:
        evals, evecs = np.linalg.eigh(hamiltonian)
    except np.linalg.LinAlgError as exc:
        raise SelectedDiagonalizationError("dense eigensolve failed", {"error": str(exc)}) from exc
    take = min(nroots + root_buffer, n)
    return evals[:take], evecs[:, :take]


def normalize_and_phase_fix(
    coeffs: np.ndarray, reference: Optional[np.ndarray] = None
) -> np.ndarray:
    """Normalize columns and fix phases deterministically.

    With a ``reference`` vector, each column's phase is fixed by positive real
    overlap; otherwise the phase is fixed by the largest-magnitude coefficient.
    """
    coeffs = np.asarray(coeffs, dtype=complex).copy()
    n, k = coeffs.shape
    for j in range(k):
        v = coeffs[:, j]
        norm = np.linalg.norm(v)
        if norm < 1e-15:
            raise SelectedDiagonalizationError("degenerate zero-norm eigenvector")
        v = v / norm
        if reference is not None:
            ov = float(np.vdot(reference, v))
            phase = ov / abs(ov) if abs(ov) > 1e-12 else 1.0
        else:
            imax = int(np.argmax(np.abs(v)))
            phase = v[imax] / abs(v[imax])
        coeffs[:, j] = v / phase
    return coeffs


def match_roots_by_overlap(
    selected_coeffs: np.ndarray,
    determinant_keys: Sequence[DeterminantKey],
    reference_vectors: Sequence[dict],
    min_overlap: float = 0.50,
) -> np.ndarray:
    """Assign selected eigenvectors to reference states by overlap.

    ``reference_vectors`` are ``{DeterminantKey: coeff}`` mappings in the same
    determinant basis.  Returns a permutation ``perm`` such that selected root
    ``perm[s]`` corresponds to reference state ``s``.
    """
    n_ref = len(reference_vectors)
    n_sel = selected_coeffs.shape[1]
    index = {k: i for i, k in enumerate(determinant_keys)}

    def dense_of(ref: dict) -> np.ndarray:
        v = np.zeros(len(determinant_keys), dtype=complex)
        for k, c in ref.items():
            i = index.get(k)
            if i is not None:
                v[i] = c
        return v

    ov = np.abs(
        np.array([[np.vdot(dense_of(ref), selected_coeffs[:, j])
                   for j in range(n_sel)] for ref in reference_vectors])
    )
    from scipy.optimize import linear_sum_assignment

    row, col = linear_sum_assignment(-ov)
    perm = np.full(n_ref, -1, dtype=int)
    for r, c in zip(row, col):
        perm[r] = c
    for r in range(n_ref):
        if perm[r] < 0 or ov[r, perm[r]] < min_overlap:
            raise SelectedRootMissingError(
                "target root not found in selected space",
                {"reference_state": r, "best_overlap": float(ov[r].max())},
            )
    return perm


def phase_fix_to_reference(
    selected_coeffs: np.ndarray,
    determinant_keys: Sequence[DeterminantKey],
    reference_vectors: Sequence[dict],
    perm: np.ndarray,
) -> np.ndarray:
    """Align each selected column's global phase to its matched reference state.

    ``perm[s]`` is the selected column index matching reference state ``s``.
    This makes transition (off-diagonal) 1-RDMs phase-consistent with the
    reference gauge.
    """
    out = np.asarray(selected_coeffs, dtype=complex).copy()
    index = {k: i for i, k in enumerate(determinant_keys)}
    for s, ref in enumerate(reference_vectors):
        col = int(perm[s])
        ref_dense = np.zeros(len(determinant_keys), dtype=complex)
        for k, c in ref.items():
            i = index.get(k)
            if i is not None:
                ref_dense[i] = c
        ov = np.vdot(ref_dense, out[:, col])
        if abs(ov) > 1e-12:
            out[:, col] *= ov / abs(ov)
    return out


__all__ = [
    "solve_selected_sector",
    "normalize_and_phase_fix",
    "match_roots_by_overlap",
    "phase_fix_to_reference",
]
