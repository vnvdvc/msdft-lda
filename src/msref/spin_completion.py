"""Spatial-occupation family mapping and spin-complete enumeration.

The MVP selection unit is a *spatial-occupation family*: for a determinant
``(alpha, beta)``, the family key is the per-orbital occupancy ``occ[p] in
{0,1,2}``.  A family is the set of all spin assignments of its singly occupied
orbitals consistent with a fixed ``(N_alpha, N_beta)`` sector.  This span is
invariant under ``S^2``, so a union of complete families gives spin-pure
eigenstates of the spin-free selected Hamiltonian.
"""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence

from .exceptions import ReferenceNotCompact
from .models import (
    DeterminantKey,
    SectorSupport,
    SelectedSupport,
    SpatialOccupationKey,
)


def family_of(det: DeterminantKey, norb: int) -> SpatialOccupationKey:
    """Map a determinant to its spatial-occupation family over ``norb`` orbitals."""
    occ = []
    for p in range(norb):
        a = (det.alpha >> p) & 1
        b = (det.beta >> p) & 1
        occ.append(a + b)
    return SpatialOccupationKey(tuple(occ))


def enumerate_family(
    family: SpatialOccupationKey, na: int, nb: int
) -> tuple[DeterminantKey, ...]:
    """Enumerate every determinant in a family consistent with ``(na, nb)``.

    Returns an empty tuple if the family is incompatible with the sector.
    """
    double = [p for p, o in enumerate(family.occ) if o == 2]
    single = [p for p, o in enumerate(family.occ) if o == 1]

    nas = na - len(double)
    nbs = nb - len(double)
    if nas < 0 or nbs < 0 or nas + nbs != len(single):
        return ()

    base = sum(1 << p for p in double)
    out = []
    for alpha_single in combinations(single, nas):
        alpha_set = set(alpha_single)
        a = base
        b = base
        for p in single:
            if p in alpha_set:
                a |= 1 << p
            else:
                b |= 1 << p
        out.append(DeterminantKey(a, b))
    out.sort()
    return tuple(out)


def family_determinants(
    families: Sequence[SpatialOccupationKey], na: int, nb: int
) -> tuple[DeterminantKey, ...]:
    """Union of determinant realizations of ``families`` in one sector."""
    dets = set()
    for family in families:
        dets.update(enumerate_family(family, na, nb))
    return tuple(sorted(dets))


def materialize_support(
    families: Sequence[SpatialOccupationKey],
    sectors: Sequence[tuple[int, int, object]],
    max_support_determinants: int,
    protected: frozenset[SpatialOccupationKey] = frozenset(),
    generation: int = 0,
) -> SelectedSupport:
    """Materialize a family support into per-sector determinant lists."""
    fam_tuple = tuple(sorted(set(families)))
    sector_support = {}
    for sector in sectors:
        na, nb, wfnsym = sector
        keys = family_determinants(fam_tuple, na, nb)
        if len(keys) > max_support_determinants:
            raise ReferenceNotCompact(
                "support determinant count exceeds guardrail",
                {"sector": sector, "n_det": len(keys), "max": max_support_determinants},
            )
        sector_support[sector] = SectorSupport(
            sector=sector,
            determinant_keys=keys,
            index={k: i for i, k in enumerate(keys)},
        )
    return SelectedSupport(
        families=fam_tuple,
        sector_support=sector_support,
        protected=protected,
        generation=generation,
    )


def all_determinants(norb: int, na: int, nb: int) -> tuple[DeterminantKey, ...]:
    """Enumerate the full ``(na, nb)`` determinant sector (lexicographic)."""
    from itertools import combinations

    dets = []
    for alpha_occ in combinations(range(norb), na):
        a = sum(1 << p for p in alpha_occ)
        for beta_occ in combinations(range(norb), nb):
            b = sum(1 << p for p in beta_occ)
            dets.append(DeterminantKey(a, b))
    return tuple(dets)


__all__ = [
    "family_of",
    "enumerate_family",
    "family_determinants",
    "materialize_support",
    "all_determinants",
]
