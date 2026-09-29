"""Unit tests for external connected-family generation (T034)."""

from __future__ import annotations

import numpy as np
import pytest

from msref.backend.slater_condon_reference import h_element
from msref.models import DeterminantKey
from msref.residual import (
    connected_external_families,
    generate_connected_external,
    generate_doubles,
    generate_singles,
)
from msref.spin_completion import all_determinants, family_of


def _tiny_eri(norb=3):
    rng = np.random.default_rng(7)
    eri = rng.standard_normal((norb, norb, norb, norb))
    eri = (eri + eri.transpose(1, 0, 2, 3) + eri.transpose(0, 1, 3, 2)
           + eri.transpose(1, 0, 3, 2)) / 4.0
    return eri


@pytest.mark.unit
def test_singles_spin_conserving():
    det = DeterminantKey(0b001, 0b010)  # alpha occ 0, beta occ 1
    singles = generate_singles(det, 3)
    for s in singles:
        assert s.alpha.bit_count() == 1 and s.beta.bit_count() == 1


@pytest.mark.unit
def test_doubles_spin_conserving():
    det = DeterminantKey(0b011, 0b100)  # alpha occ 0,1 ; beta occ 2
    doubles = generate_doubles(det, 4)
    for d in doubles:
        assert d.alpha.bit_count() == 2 and d.beta.bit_count() == 1


@pytest.mark.unit
def test_external_generation_equals_nonzero_connections_u():
    """For a small CAS, generated singles+doubles equal all nonzero H connections."""
    norb = 3
    na = nb = 1
    h1e = np.eye(norb) * 0.5
    eri = _tiny_eri(norb)
    det = DeterminantKey(0b001, 0b001)
    conn = generate_singles(det, norb) | generate_doubles(det, norb)
    # brute-force: any determinant with nonzero H coupling must be in conn
    full = all_determinants(norb, na, nb)
    for other in full:
        if other == det:
            continue
        if abs(h_element(other, det, h1e, eri)) > 1e-12:
            assert other in conn, f"missed coupling to {other}"


@pytest.mark.unit
def test_connected_external_families():
    sel = {DeterminantKey(0b001, 0b001), DeterminantKey(0b010, 0b010)}
    fams = connected_external_families(sel, norb=3)
    assert isinstance(fams, tuple)
    assert all(len(f.occ) == 3 for f in fams)
