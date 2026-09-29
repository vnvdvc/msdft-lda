"""Unit tests for core data models (no PySCF)."""

from __future__ import annotations

import numpy as np
import pytest

from msref.exceptions import InputValidationError
from msref.models import (
    ActiveSpaceVariant,
    DeterminantKey,
    ParentOrbitalSpace,
    Role,
    SparseStateVector,
    SpatialOccupationKey,
    StateBlockSpec,
    StateID,
    validate_state_weights,
)


@pytest.mark.unit
def test_determinant_key_order():
    a = DeterminantKey(0, 1)
    b = DeterminantKey(1, 0)
    c = DeterminantKey(0, 1)
    assert a == c
    assert (a < b) or (b < a)
    assert hash(a) == hash(c)


@pytest.mark.unit
def test_determinant_key_negative_fails():
    with pytest.raises(InputValidationError):
        DeterminantKey(-1, 0)


@pytest.mark.unit
def test_spatial_occupation_key():
    f = SpatialOccupationKey((2, 1, 0))
    assert f.nelec == 3
    assert f.norb == 3
    with pytest.raises(InputValidationError):
        SpatialOccupationKey((3, 0, 0))


@pytest.mark.unit
def test_state_block_spec_validation():
    b = StateBlockSpec(block_id="s", spin_S=0.0, ms2=0, nroots=2, weights=[0.5, 0.5])
    assert len(b.weights) == b.nroots
    with pytest.raises(InputValidationError):
        StateBlockSpec(block_id="s", spin_S=0.0, ms2=0, nroots=2, weights=[0.5])
    with pytest.raises(InputValidationError):
        StateBlockSpec(block_id="s", spin_S=0.0, ms2=0, nroots=2, weights=[-0.5, 1.5])


@pytest.mark.unit
def test_parent_orbital_space_duplicate_ids_fail():
    with pytest.raises(InputValidationError):
        ParentOrbitalSpace(
            parent_id="p",
            mo_coeff_ao=np.eye(2),
            ao_overlap=np.eye(2),
            orbital_ids=("a", "a"),
            always_core=(0,),
        )


@pytest.mark.unit
def test_parent_orbital_space_role_disjoint():
    with pytest.raises(InputValidationError):
        ParentOrbitalSpace(
            parent_id="p",
            mo_coeff_ao=np.eye(3),
            ao_overlap=np.eye(3),
            orbital_ids=("a", "b", "c"),
            always_core=(0,),
            comparison_orbitals=(0, 1),
        )


@pytest.mark.unit
def test_active_space_variant_roles():
    v = ActiveSpaceVariant(
        variant_id="v0",
        parent_id="p",
        roles={0: Role.INACTIVE, 1: Role.ACTIVE, 2: Role.EXTERNAL},
        active_orbitals=(1,),
        inactive_orbitals=(0,),
    )
    assert v.roles[1] is Role.ACTIVE
    with pytest.raises(InputValidationError):
        ActiveSpaceVariant(
            variant_id="bad",
            parent_id="p",
            roles={1: Role.INACTIVE},
            active_orbitals=(1,),
        )


@pytest.mark.unit
def test_sparse_state_vector_uniqueness():
    sid = StateID(name="s0", block_id="b", spin_S=0.0, ms2=0)
    with pytest.raises(InputValidationError):
        SparseStateVector(
            state_id=sid,
            sector_key=(1, 1, None),
            determinant_keys=(DeterminantKey(1, 0), DeterminantKey(1, 0)),
            coefficients=np.array([0.5, 0.5], dtype=complex),
        )


@pytest.mark.unit
def test_validate_state_weights():
    validate_state_weights([0.5, 0.5])
    with pytest.raises(InputValidationError):
        validate_state_weights([0.5, 0.4])
    with pytest.raises(InputValidationError):
        validate_state_weights([-0.1, 1.1])
