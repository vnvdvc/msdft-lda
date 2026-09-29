"""Unit tests for role maps and electron accounting (T020)."""

from __future__ import annotations

import numpy as np
import pytest

from msref.active_ensemble import count_role, derive_sector_electron_counts, validate_role_map
from msref.exceptions import InputValidationError
from msref.models import ActiveSpaceVariant, ParentOrbitalSpace, Role


def _parent(nmo=6):
    return ParentOrbitalSpace(
        parent_id="p",
        mo_coeff_ao=np.eye(nmo),
        ao_overlap=np.eye(nmo),
        orbital_ids=tuple(f"o{i}" for i in range(nmo)),
        always_core=(0,),
        comparison_orbitals=(1, 2, 3, 4, 5),
    )


def _variant(roles):
    act = tuple(sorted(i for i, r in roles.items() if r is Role.ACTIVE))
    inact = tuple(sorted(i for i, r in roles.items() if r is Role.INACTIVE))
    return ActiveSpaceVariant(
        variant_id="v", parent_id="p", roles=roles,
        active_orbitals=act, inactive_orbitals=inact,
    )


@pytest.mark.unit
def test_electron_accounting():
    parent = _parent()
    # roles: 1,2 INACTIVE (doubly occ), 3,4 ACTIVE, 5 EXTERNAL
    roles = {0: Role.FROZEN_CORE, 1: Role.INACTIVE, 2: Role.INACTIVE,
             3: Role.ACTIVE, 4: Role.ACTIVE, 5: Role.EXTERNAL}
    var = _variant(roles)
    # total (5 alpha, 5 beta): core 1, so comparison has 4 alpha + 4 beta;
    # 2 inactive -> 2 active alpha + 2 active beta
    (na_cmp, nb_cmp), (na_act, nb_act) = derive_sector_electron_counts(parent, var, (5, 5))
    assert (na_cmp, nb_cmp) == (4, 4)
    assert (na_act, nb_act) == (2, 2)


@pytest.mark.unit
def test_count_role():
    roles = {1: Role.INACTIVE, 2: Role.INACTIVE, 3: Role.ACTIVE}
    var = _variant(roles)
    assert count_role(var, Role.INACTIVE) == 2
    assert count_role(var, Role.ACTIVE) == 1


@pytest.mark.unit
def test_invalid_role_map_missing():
    parent = _parent(3)
    roles = {0: Role.FROZEN_CORE, 1: Role.ACTIVE}  # orbital 2 missing
    var = _variant(roles)
    with pytest.raises(InputValidationError):
        validate_role_map(var, parent, (2, 2))


@pytest.mark.unit
def test_protected_core_removed():
    parent = _parent(3)
    roles = {0: Role.ACTIVE, 1: Role.ACTIVE, 2: Role.ACTIVE}  # core (0) removed
    var = _variant(roles)
    with pytest.raises(InputValidationError):
        validate_role_map(var, parent, (2, 2))


@pytest.mark.unit
def test_negative_active_electrons():
    parent = _parent()
    roles = {0: Role.FROZEN_CORE, 1: Role.INACTIVE, 2: Role.INACTIVE,
             3: Role.INACTIVE, 4: Role.INACTIVE, 5: Role.INACTIVE}
    var = _variant(roles)
    with pytest.raises(InputValidationError):
        derive_sector_electron_counts(parent, var, (3, 3))
