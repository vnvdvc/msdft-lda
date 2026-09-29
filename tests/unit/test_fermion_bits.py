"""Unit tests for fermionic bit operations (T024)."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from msref.backend.slater_condon_reference import (
    annihilate,
    count_occupied_between,
    create,
    excitation_rank,
    is_occupied,
    popcount,
)
from msref.models import DeterminantKey


def _all_bitstrings(norb, nelec):
    return [sum(1 << p for p in occ) for occ in itertools.combinations(range(norb), nelec)]


@pytest.mark.unit
def test_bit_count_u001():
    for norb in range(1, 8):
        for nelec in range(norb + 1):
            for bits in _all_bitstrings(norb, nelec):
                assert popcount(bits) == nelec


@pytest.mark.unit
def test_create_annihilate_roundtrip_u002():
    """Create/annihilate are mutual inverses with unit round-trip sign."""
    norb = 4
    for nelec in range(norb + 1):
        for bits in _all_bitstrings(norb, nelec):
            for p in range(norb):
                # a_p^+ a_p |D> = n_p |D>
                res = annihilate(bits, p)
                if is_occupied(bits, p):
                    assert res is not None
                    nb, s1 = res
                    res2 = create(nb, p)
                    assert res2 is not None and res2[0] == bits
                    assert s1 * res2[1] == 1
                else:
                    assert res is None


@pytest.mark.unit
def test_create_annihilate_adjoint_u002():
    """Creation is the adjoint of annihilation across electron-number sectors."""
    norb = 4
    strs_n = _all_bitstrings(norb, 2)
    strs_n1 = _all_bitstrings(norb, 1)
    idx_n = {b: i for i, b in enumerate(strs_n)}
    idx_n1 = {b: i for i, b in enumerate(strs_n1)}
    for p in range(norb):
        A = np.zeros((len(strs_n1), len(strs_n)))  # <i|a_p|j>, i in N-1, j in N
        C = np.zeros((len(strs_n), len(strs_n1)))  # <i|a_p^+|j>, i in N, j in N-1
        for j, bits in enumerate(strs_n):
            res = annihilate(bits, p)
            if res is not None:
                A[idx_n1[res[0]], j] = res[1]
        for j, bits in enumerate(strs_n1):
            res = create(bits, p)
            if res is not None:
                C[idx_n[res[0]], j] = res[1]
        # a_p^+ = a_p^dagger  =>  C = A^T (real)
        assert np.allclose(C, A.T)


@pytest.mark.unit
def test_excitation_rank_u003():
    dets = [DeterminantKey(a, b)
            for a in _all_bitstrings(4, 2) for b in _all_bitstrings(4, 2)]
    for bra in dets:
        for ket in dets:
            da, db = excitation_rank(bra, ket)
            assert da % 2 == 0 and db % 2 == 0


@pytest.mark.unit
def test_count_occupied_between():
    # bits: orbitals 0,2,5 occupied -> 0b100101 = 37
    bits = (1 << 0) | (1 << 2) | (1 << 5)
    assert count_occupied_between(bits, 0, 5) == 1  # only orbital 2 between
    assert count_occupied_between(bits, 0, 2) == 0
    assert count_occupied_between(bits, 2, 5) == 0


@pytest.mark.unit
def test_is_occupied():
    assert is_occupied(0b101, 0)
    assert not is_occupied(0b101, 1)
    assert is_occupied(0b101, 2)
