"""Unit tests for family pool statistics and support seed (T030/T031)."""

from __future__ import annotations

import numpy as np
import pytest

from msref.config_pool import (
    aggregate_family_pool,
    captured_average,
    seed_support,
    variant_family_stats,
)
from msref.models import DeterminantKey, SparseStateVector, SpatialOccupationKey, StateID


def _vec(occ_keys, coeffs, block="b"):
    sid = StateID(name=f"{block}_r0", block_id=block, spin_S=0.0, ms2=0)
    return SparseStateVector(state_id=sid, sector_key=(1, 1, None),
                             determinant_keys=tuple(occ_keys), coefficients=np.array(coeffs, dtype=complex))


@pytest.mark.unit
def test_family_stats_weight_normalization():
    k1 = DeterminantKey(0b0011, 0b0011)  # orbitals 0,1 doubly occupied
    k2 = DeterminantKey(0b0110, 0b0110)  # orbitals 1,2 doubly occupied
    state = _vec([k1, k2], [1.0, 0.0])
    q, by_state = variant_family_stats([state], [1.0], norb=4)
    assert sum(q.values()) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.unit
def test_aggregate_pool_invariant_under_state_rotation_u003():
    # two degenerate states; family qbar must be invariant to their unitary rotation
    k = [DeterminantKey(0b11, 0b11), DeterminantKey(0b1100, 0b1100)]
    s1 = _vec(k, [1, 0])
    s2 = _vec(k, [0, 1])
    q1, bs1 = variant_family_stats([s1, s2], [0.5, 0.5], norb=4)

    u = np.array([[0.6, 0.8], [0.8, -0.6]])
    r1 = _vec(k, [u[0, 0], u[0, 1]])
    r2 = _vec(k, [u[1, 0], u[1, 1]])
    q2, bs2 = variant_family_stats([r1, r2], [0.5, 0.5], norb=4)
    for f in set(q1) | set(q2):
        assert q1.get(f, 0.0) == pytest.approx(q2.get(f, 0.0), abs=1e-12)


@pytest.mark.unit
def test_seed_support_basic():
    fam_a = SpatialOccupationKey((2, 0, 0, 0))
    fam_b = SpatialOccupationKey((1, 1, 0, 0))
    pool = aggregate_family_pool(
        [{fam_a: 0.9, fam_b: 0.1}],
        [[{fam_a: 0.9, fam_b: 0.1}]],
        [1.0], tau_persist=1e-5,
    )
    from msref.config import MSRefConfig
    cfg = MSRefConfig()
    cfg.support_seed.captured_average_weight = 0.99
    cfg.support_seed.minimum_captured_weight_per_state = 0.5
    sel = seed_support(pool, [], cfg)
    assert fam_a in sel


@pytest.mark.unit
def test_captured_average():
    fam_a = SpatialOccupationKey((2, 0, 0, 0))
    pool = aggregate_family_pool([{fam_a: 0.8}], [[{fam_a: 0.8}]], [1.0], 1e-5)
    assert captured_average([pool[fam_a]]) == pytest.approx(0.8)
