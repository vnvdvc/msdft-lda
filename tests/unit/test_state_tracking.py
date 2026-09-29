"""Unit tests for state overlap and root tracking (T014)."""

from __future__ import annotations

import numpy as np
import pytest

from msref.exceptions import StateTrackingError
from msref.models import DeterminantKey, SparseStateVector, StateID
from msref.state_tracking import (
    hungarian_assignment,
    overlap_matrix,
    procrustes_cluster,
    sparse_overlap,
    track_roots,
)


def _vec(keys, coeffs, spin_s=0.0, block="b"):
    sid = StateID(name=f"{block}_r0", block_id=block, spin_S=spin_s, ms2=0)
    return SparseStateVector(state_id=sid, sector_key=(1, 1, None),
                             determinant_keys=tuple(keys), coefficients=np.array(coeffs, dtype=complex))


@pytest.mark.unit
def test_sparse_overlap():
    k = [DeterminantKey(1, 2), DeterminantKey(2, 1), DeterminantKey(4, 4)]
    a = {k[0]: 1.0, k[1]: 1j}
    b = {k[0]: 0.5, k[2]: 0.5}
    ov = sparse_overlap(a, b)
    assert np.isclose(ov, 0.5, atol=1e-12)


@pytest.mark.unit
def test_hungarian_swap_u014():
    o = np.array([[0.1, 0.9], [0.9, 0.1]])  # swapped -> assignment recovers identity
    row_to_col, amb = hungarian_assignment(o)
    assert row_to_col == [1, 0]


@pytest.mark.unit
def test_track_roots_u015_degenerate():
    k = [DeterminantKey(1, 2), DeterminantKey(2, 1)]
    ref = [_vec(k, [1, 0]), _vec(k, [0, 1])]
    # random unitary rotation within the 2-state cluster
    u = np.array([[np.cos(0.3), -np.sin(0.3)], [np.sin(0.3), np.cos(0.3)]])
    new = [
        _vec(k, [u[0, 0], u[0, 1]]),
        _vec(k, [u[1, 0], u[1, 1]]),
    ]
    perm, ov, _ = track_roots(ref, new)
    # rotated states map 1:1 (any permutation is valid for a degenerate cluster)
    assert sorted(perm) == [0, 1]


@pytest.mark.unit
def test_track_roots_unresolved():
    k1 = [DeterminantKey(1, 2)]
    k2 = [DeterminantKey(4, 8)]
    ref = [_vec(k1, [1.0])]
    new = [_vec(k2, [1.0])]  # no overlap
    with pytest.raises(StateTrackingError):
        track_roots(ref, new, min_overlap=0.5)


@pytest.mark.unit
def test_procrustes_cluster_u015():
    k = [DeterminantKey(1, 2), DeterminantKey(2, 1), DeterminantKey(4, 4)]
    ref = [_vec(k, [1, 0, 0]), _vec(k, [0, 1, 0])]
    u = np.array([[0.8, -0.6], [0.6, 0.8]])
    new = [_vec(k, [u[0, 0], u[0, 1], 0]), _vec(k, [u[1, 0], u[1, 1], 0])]
    rotated, sigma = procrustes_cluster(ref, new)
    assert np.allclose(sigma, 1.0, atol=1e-12)


@pytest.mark.unit
def test_cluster_rank_mismatch_u016():
    k = [DeterminantKey(1, 2), DeterminantKey(2, 1)]
    ref = [_vec(k, [1, 0]), _vec(k, [0, 1])]
    new = [_vec(k, [1, 0])]
    with pytest.raises(StateTrackingError):
        procrustes_cluster(ref, new)
