"""Active-space role maps and electron accounting (T020, T050).

A parent comparison orbital can be inactive (doubly occupied), active, or
external in each variant.  Always-core orbitals are factored out of the CI
problem but their density contribution is restored later.
"""

from __future__ import annotations

from itertools import combinations
from typing import Optional, Sequence

import numpy as np

from .exceptions import InputValidationError
from .models import ActiveSpaceVariant, ParentOrbitalSpace, Role, StateBlockSpec


def count_role(variant: ActiveSpaceVariant, role: Role) -> int:
    """Number of parent orbitals with a given role in the variant."""
    return sum(1 for r in variant.roles.values() if r is role)


def derive_sector_electron_counts(
    parent: ParentOrbitalSpace,
    variant: ActiveSpaceVariant,
    nelec_total: tuple[int, int],
):
    """Derive ``((na_cmp, nb_cmp), (na_act, nb_act))`` for a variant.

    ``nelec_total`` is the full-molecule ``(nalpha, nbeta)`` used for the block.
    """
    nalpha_total, nbeta_total = int(nelec_total[0]), int(nelec_total[1])
    ncore = len(parent.always_core)
    na_cmp = nalpha_total - ncore
    nb_cmp = nbeta_total - ncore

    ninactive = count_role(variant, Role.INACTIVE)
    na_act = na_cmp - ninactive
    nb_act = nb_cmp - ninactive

    if na_act < 0 or nb_act < 0:
        raise InputValidationError(
            "negative active electron count",
            {"na_act": na_act, "nb_act": nb_act, "ninactive": ninactive},
        )
    nact = len(variant.active_orbitals)
    if na_act > nact or nb_act > nact:
        raise InputValidationError(
            "too many active electrons for active orbital count",
            {"na_act": na_act, "nb_act": nb_act, "nactive": nact},
        )
    return (na_cmp, nb_cmp), (na_act, nb_act)


def validate_role_map(
    variant: ActiveSpaceVariant, parent: ParentOrbitalSpace, nelec_total: tuple[int, int]
) -> None:
    """Validate that a variant role map conserves electrons and protected orbitals."""
    # every parent orbital must have a role
    for idx in range(len(parent.orbital_ids)):
        if idx not in variant.roles:
            raise InputValidationError("missing role for orbital", {"orbital": idx})
    # protected core orbitals must remain FROZEN_CORE
    for idx in parent.always_core:
        if variant.roles.get(idx) is not Role.FROZEN_CORE:
            raise InputValidationError("protected core orbital removed", {"orbital": idx})
    # electron accounting must be consistent
    derive_sector_electron_counts(parent, variant, nelec_total)


__all__ = [
    "count_role",
    "derive_sector_electron_counts",
    "validate_role_map",
    "generate_active_space_variants",
]


def generate_active_space_variants(
    nominal_active: Sequence[int],
    doubly_occupied: Sequence[int],
    virtual: Sequence[int],
    nelec_comparison: tuple[int, int],
    cfg,
) -> list[ActiveSpaceVariant]:
    """Generate the deterministic fixed-orbital active-space ensemble (T050).

    The ensemble is built from a nominal active set plus add / exchange / remove
    operations over the doubly-occupied ("core") and virtual ("buffer")
    comparison orbitals.  Each variant carries a full role map and its derived
    active electron count, so ``derive_sector_electron_counts`` stays the single
    source of truth for accounting.

    Parameters
    ----------
    nominal_active:
        parent-orbital indices of the nominal active space.
    doubly_occupied:
        parent-orbital indices that are doubly occupied in the reference and are
        *not* in the nominal active space (promoting one adds two electrons).
    virtual:
        parent-orbital indices that are empty in the reference (buffer).
    nelec_comparison:
        ``(nalpha, nbeta)`` electron count of the full comparison space.

    Returns
    -------
    list of :class:`ActiveSpaceVariant`, capped at ``n_active_spaces``.
    """
    ase = cfg.active_space_ensemble
    nominal = tuple(sorted(set(int(i) for i in nominal_active)))
    d_occ = [int(i) for i in doubly_occupied if int(i) not in nominal]
    virt = [int(i) for i in virtual if int(i) not in nominal]
    na_cmp, nb_cmp = int(nelec_comparison[0]), int(nelec_comparison[1])

    candidate_sets: dict[tuple[int, ...], str] = {}
    candidate_sets[nominal] = "nominal"

    max_add = ase.max_add_orbitals

    def _register(active: tuple[int, ...], rule: str) -> None:
        key = tuple(sorted(set(active)))
        if key not in candidate_sets:
            candidate_sets[key] = rule

    # add *virtual/buffer* orbitals only (no electron-count change); the
    # doubly-occupied orbitals stay frozen (inactive) in every variant.
    for nadd in range(1, max_add + 1):
        for combo in combinations(virt, nadd):
            _register(nominal + tuple(combo), f"add{nadd}")

    # exchange: swap one active orbital for one virtual/buffer orbital
    if ase.allow_exchange:
        for a in nominal:
            for b in virt:
                _register(tuple(x for x in nominal if x != a) + (b,), "exchange")

    # remove (optional)
    if ase.allow_remove:
        for a in nominal:
            _register(tuple(x for x in nominal if x != a), "remove")

    # random subsets (seedable)
    if ase.n_random_variants > 0:
        rng = np.random.default_rng(cfg.seed if cfg.seed is not None else 0)
        pool = virt
        n_avail = min(len(pool), max_add)
        for _ in range(ase.n_random_variants * 2):  # oversample, dedup below
            k = int(rng.integers(0, n_avail + 1)) if pool else 0
            chosen = tuple(int(pool[i]) for i in rng.choice(len(pool), size=k, replace=False)) if pool else ()
            _register(nominal + chosen, "random")

    # deterministic order: nominal first, then lexicographic by active set
    ordered = sorted(candidate_sets.keys(), key=lambda s: (0 if s == nominal else 1, s))
    ordered = ordered[: ase.n_active_spaces]

    variants = []
    for idx, active in enumerate(ordered):
        rule = candidate_sets[active]
        active = tuple(sorted(set(active)))
        roles: dict[int, Role] = {i: Role.ACTIVE for i in active}
        for i in d_occ:
            if i not in active:
                roles[i] = Role.INACTIVE
        for i in virt:
            if i not in active:
                roles[i] = Role.EXTERNAL
        ninactive = sum(1 for r in roles.values() if r is Role.INACTIVE)
        na_act = na_cmp - ninactive
        nb_act = nb_cmp - ninactive
        if na_act < 0 or nb_act < 0 or na_act > len(active) or nb_act > len(active):
            continue  # invalid electron accounting for this variant
        variants.append(
            ActiveSpaceVariant(
                variant_id=f"{rule}_{idx:03d}",
                parent_id="parent",
                roles=roles,
                active_orbitals=active,
                inactive_orbitals=tuple(i for i, r in roles.items() if r is Role.INACTIVE),
                nelecas_by_sector={(na_cmp, nb_cmp, None): (na_act, nb_act)},
                generation_rule=rule,
                perturbation_seed=cfg.seed,
                weight=1.0,
            )
        )
    return variants
