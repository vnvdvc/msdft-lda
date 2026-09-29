"""Unit tests for family mapping and spin completion (T022/T023)."""

from __future__ import annotations

import itertools
from collections import defaultdict
from math import comb

import pytest

from msref.backend.slater_condon_reference import annihilate, create, is_occupied
from msref.models import DeterminantKey, SpatialOccupationKey
from msref.spin_completion import (
    all_determinants,
    enumerate_family,
    family_determinants,
    family_of,
    materialize_support,
)


def _bitstrings(norb, nelec):
    return [sum(1 << p for p in occ) for occ in itertools.combinations(range(norb), nelec)]


@pytest.mark.unit
def test_family_mapping_u006():
    norb = 4
    fam = SpatialOccupationKey((1, 1, 2, 0))
    # orbitals 0,1 singly occupied (one alpha, one beta), orbital 2 doubly
    for a, b in ((0b001, 0b010), (0b010, 0b001)):
        det = DeterminantKey(a | 0b100, b | 0b100)
        assert family_of(det, norb) == fam


@pytest.mark.unit
def test_family_enumeration_count_u007():
    fam = SpatialOccupationKey((2, 1, 1, 0, 1))  # 1 double, 3 single
    for na, nb in [(3, 2), (2, 3), (4, 1), (1, 4)]:
        dets = enumerate_family(fam, na, nb)
        expected = comb(3, na - 1)  # choose (na - 1) of the 3 singly-occupied for alpha
        assert len(dets) == expected, (na, nb, len(dets), expected)


@pytest.mark.unit
def test_family_enumeration_incompatible():
    fam = SpatialOccupationKey((2, 2, 0))
    assert enumerate_family(fam, 1, 1) == ()


@pytest.mark.unit
def test_spin_closure_u008():
    """Apply S^2 to each family determinant; the image stays inside the family."""
    norb = 4
    na = nb = 2
    fam = SpatialOccupationKey((2, 1, 1, 0))
    dets = enumerate_family(fam, na, nb)
    assert len(dets) == 2

    for ket in dets:
        for out in _s2_image(ket, norb):
            # the spin-flip image must retain the same spatial occupancy pattern
            assert family_of(out, norb) == fam


@pytest.mark.unit
def test_family_union_spin_closure_u009():
    norb = 4
    na = nb = 2
    fams = (SpatialOccupationKey((2, 1, 1, 0)), SpatialOccupationKey((1, 1, 1, 1)))
    all_dets = set()
    for f in fams:
        all_dets.update(enumerate_family(f, na, nb))
    for ket in all_dets:
        for out in _s2_image(ket, norb):
            assert family_of(out, norb) in fams


def _s2_image(ket: DeterminantKey, norb: int):
    """Return the set of determinants with nonzero ``S^- S^+`` coupling to ``ket``."""
    out = set()
    # S^- S^+ = sum_{pq} a_pbeta^+ a_palpha a_qalpha^+ a_qbeta
    # applied right-to-left: a_qbeta, a_qalpha^+, a_palpha, a_pbeta^+
    for q in range(norb):
        if not is_occupied(ket.beta, q):
            continue
        t1, s1 = annihilate(ket.beta, q)  # a_qbeta
        for _ in range(1):  # a_qalpha^+ : create alpha at q
            r = create(ket.alpha, q)
            if r is None:
                continue
            new_alpha, s2 = r
            for p in range(norb):
                if not is_occupied(new_alpha, p):
                    continue
                t2, s3 = annihilate(new_alpha, p)  # a_palpha
                r2 = create(t1, p)  # a_pbeta^+
                if r2 is None:
                    continue
                new_beta, s4 = r2
                if s1 * s2 * s3 * s4 != 0:
                    out.add(DeterminantKey(t2, new_beta))
    return out


@pytest.mark.unit
def test_materialize_support():
    fams = (SpatialOccupationKey((2, 0, 0)), SpatialOccupationKey((1, 1, 0)))
    support = materialize_support(fams, [(2, 1, None), (1, 2, None)], max_support_determinants=100)
    assert len(support.families) == 2
    for ss in support.sector_support.values():
        assert len(ss.index) == len(ss.determinant_keys)


@pytest.mark.unit
def test_all_determinants_count():
    dets = all_determinants(4, 2, 2)
    assert len(dets) == comb(4, 2) * comb(4, 2) == 36
