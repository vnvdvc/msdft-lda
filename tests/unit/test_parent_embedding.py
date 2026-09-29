"""Unit tests for parent determinant embedding (T021)."""

from __future__ import annotations

import numpy as np
import pytest

from msref.config_pool import embed_variant_det
from msref.models import ActiveSpaceVariant, DeterminantKey, ParentOrbitalSpace, Role


def _parent():
    return ParentOrbitalSpace(
        parent_id="p",
        mo_coeff_ao=np.eye(6),
        ao_overlap=np.eye(6),
        orbital_ids=tuple(f"o{i}" for i in range(6)),
        always_core=(0,),
        comparison_orbitals=(1, 2, 3, 4, 5),
    )


def _variant(inactive, active):
    roles = {0: Role.FROZEN_CORE}
    for i in inactive:
        roles[i] = Role.INACTIVE
    for i in active:
        roles[i] = Role.ACTIVE
    return ActiveSpaceVariant(
        variant_id="v", parent_id="p", roles=roles,
        active_orbitals=tuple(sorted(active)), inactive_orbitals=tuple(sorted(inactive)),
    )


@pytest.mark.unit
def test_embedding_u004():
    parent = _parent()
    var = _variant(inactive=(1,), active=(2, 3))
    # active bits: alpha occ at active position 0 (orbital 2), beta at position 1 (orbital 3)
    det = embed_variant_det(parent, var, active_alpha_bits=0b01, active_beta_bits=0b10)
    # comparison positions: orbital1->pos0, orbital2->pos1, orbital3->pos2, ...
    # inactive orbital1 -> alpha/beta bit at pos0 set
    # active orbital2 (active pos0, alpha set) -> parent pos1 alpha
    # active orbital3 (active pos1, beta set) -> parent pos2 beta
    assert det.alpha == 0b0011  # bits 0 (inactive), 1 (active alpha)
    assert det.beta == 0b0101   # bits 0 (inactive), 2 (active beta)


@pytest.mark.unit
def test_embedding_collision_u005():
    """Same physical determinant from two fixed-orbital variants -> identical key."""
    parent = _parent()
    # variant A: orbitals 2,3 active, orbital 1 inactive
    va = _variant(inactive=(1,), active=(2, 3))
    da = embed_variant_det(parent, va, active_alpha_bits=0b01, active_beta_bits=0b10)
    # variant B: same physical occupation, all three comparison orbitals active
    vb = _variant(inactive=(), active=(1, 2, 3))
    # orbital1 doubly occupied, orbital2 alpha, orbital3 beta
    # -> active alpha bits = orbitals {1,2}, active beta bits = orbitals {1,3}
    db = embed_variant_det(parent, vb, active_alpha_bits=0b011, active_beta_bits=0b101)
    assert da == db
