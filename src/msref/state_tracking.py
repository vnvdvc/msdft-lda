"""State overlap, root tracking, and cluster alignment.

Roots are tracked by overlap (and spin/symmetry labels), never by energy order
alone.  Near-degenerate clusters are aligned by Procrustes rotation; ambiguity
raises ``StateTrackingError``.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

import numpy as np

from .exceptions import StateTrackingError
from .models import DeterminantKey, SectorKey, SparseStateVector


def sparse_vector_dict(vec: SparseStateVector) -> dict[DeterminantKey, complex]:
    """Return ``{determinant_key: coefficient}`` for a sparse state vector."""
    return {k: c for k, c in zip(vec.determinant_keys, vec.coefficients)}


def sparse_overlap(
    a: SparseStateVector | Mapping[DeterminantKey, complex],
    b: SparseStateVector | Mapping[DeterminantKey, complex],
) -> complex:
    """Return ``<a|b>`` for two sparse vectors over a common determinant gauge."""
    da = a if isinstance(a, dict) else sparse_vector_dict(a)
    db = b if isinstance(b, dict) else sparse_vector_dict(b)
    if len(da) > len(db):
        # iterate the smaller (b) side: <a|b> = conj(<b|a>)
        inner = sum(np.conj(c) * da.get(k, 0j) for k, c in db.items())
        return np.conjugate(inner)
    return sum(np.conj(c) * db.get(k, 0j) for k, c in da.items())


def overlap_matrix(
    states_a: Sequence[SparseStateVector],
    states_b: Sequence[SparseStateVector],
) -> np.ndarray:
    """Return ``O[A,B] = <Psi_A|Psi_B>`` (complex)."""
    da = [sparse_vector_dict(v) for v in states_a]
    db = [sparse_vector_dict(v) for v in states_b]
    o = np.zeros((len(states_a), len(states_b)), dtype=complex)
    for i, va in enumerate(da):
        for j, vb in enumerate(db):
            o[i, j] = sparse_overlap(va, vb)
    return o


def _cost_from_overlap(o: np.ndarray) -> np.ndarray:
    return 1.0 - np.abs(o) ** 2


def hungarian_assignment(
    o: np.ndarray, hard_constraints: Optional[list[tuple[int, int]]] = None
) -> tuple[list[int], list[tuple[int, int]]]:
    """Assign rows to columns minimizing ``1 - |overlap|^2``.

    ``hard_constraints`` is a list of ``(row, col)`` pairs that are forbidden
    (cost set to a large penalty).  Returns ``(row_to_col, ambiguous_pairs)``.
    """
    from scipy.optimize import linear_sum_assignment

    n, m = o.shape
    cost = _cost_from_overlap(o)
    if hard_constraints:
        big = 1e6
        for i, j in hard_constraints:
            if 0 <= i < n and 0 <= j < m:
                cost[i, j] = big
    row, col = linear_sum_assignment(cost)
    row_to_col = [-1] * n
    for i, j in zip(row, col):
        row_to_col[i] = j
    ambiguous = _detect_ambiguity(o, row_to_col)
    return row_to_col, ambiguous


def _detect_ambiguity(
    o: np.ndarray, row_to_col: Sequence[int], ambiguity_tol: float = 0.05
) -> list[tuple[int, int]]:
    """Return pairs where two roots compete within the ambiguity tolerance."""
    ambiguous = []
    n = o.shape[0]
    for i in range(n):
        j = row_to_col[i]
        if j < 0:
            continue
        mag = np.abs(o[i]) ** 2
        order = np.argsort(-mag)
        if len(order) > 1 and (mag[order[0]] - mag[order[1]]) < ambiguity_tol:
            ambiguous.append((i, j))
    return ambiguous


def track_roots(
    ref_states: Sequence[SparseStateVector],
    new_states: Sequence[SparseStateVector],
    min_overlap: float = 0.50,
    ambiguity_tol: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Track ``new_states`` to ``ref_states``.

    Returns ``(permutation, rotated_new_states, overlaps)`` where
    ``permutation[i]`` is the reference index matched to ``new_states[i]``.
    Raises ``StateTrackingError`` on unresolved matching.
    """
    o = overlap_matrix(new_states, ref_states)  # [n_new, n_ref]
    # sector/spin constraints: only match within the same sector key.
    constraints = []
    for i, va in enumerate(new_states):
        for j, vb in enumerate(ref_states):
            if va.sector_key != vb.sector_key:
                constraints.append((i, j))
    row_to_col, ambiguous = hungarian_assignment(o, hard_constraints=constraints)
    for i, j in enumerate(row_to_col):
        if j < 0 or abs(o[i, j]) < min_overlap:
            raise StateTrackingError(
                "root tracking unresolved: overlap below threshold",
                {"row": i, "overlap": abs(o[i, j]) if j >= 0 else None},
            )
    if ambiguous:
        raise StateTrackingError(
            "root tracking ambiguous: competing overlaps",
            {"ambiguous": ambiguous},
        )
    return np.asarray(row_to_col, dtype=int), o, o


def procrustes_cluster(
    states_a: Sequence[SparseStateVector],
    states_b: Sequence[SparseStateVector],
) -> tuple[np.ndarray, np.ndarray]:
    """Procrustes-align a degenerate cluster of ``states_b`` to ``states_a``.

    Both lists must share the same ordered determinant basis (same sector and
    keys); otherwise a dense projection is built.  Returns
    ``(rotated_states_matrix, singular_values)`` where ``rotated_states_matrix``
    has shape ``[n_det, n_state]``.
    """
    if len(states_a) != len(states_b):
        raise StateTrackingError("cluster rank mismatch", {"a": len(states_a), "b": len(states_b)})

    keys = states_a[0].determinant_keys
    na = len(keys)
    ya = np.zeros((na, len(states_a)), dtype=complex)
    yb = np.zeros((na, len(states_b)), dtype=complex)
    index = {k: i for i, k in enumerate(keys)}
    for s, (va, vb) in enumerate(zip(states_a, states_b)):
        for k, c in zip(va.determinant_keys, va.coefficients):
            ya[index[k], s] = c
        for k, c in zip(vb.determinant_keys, vb.coefficients):
            if k in index:
                yb[index[k], s] = c
    o = ya.conj().T @ yb  # [n, n]
    u, sigma, vh = np.linalg.svd(o, full_matrices=False)
    r = vh.conj().T @ u.conj().T
    rotated = yb @ r
    return rotated, sigma


__all__ = [
    "sparse_vector_dict",
    "sparse_overlap",
    "overlap_matrix",
    "hungarian_assignment",
    "track_roots",
    "procrustes_cluster",
]
