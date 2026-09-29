"""Configuration/family pool: parent embedding, family statistics, support seed.

Stage F of the master plan: embed teacher determinants into the parent gauge,
aggregate state-averaged family weights, and seed the initial common support.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from .exceptions import InputValidationError, ParentEmbeddingError, ReferenceNotCompact
from .models import (
    ActiveSpaceVariant,
    DeterminantKey,
    FamilyStats,
    ParentOrbitalSpace,
    Role,
    SpatialOccupationKey,
    SparseStateVector,
)
from .spin_completion import family_of


# ---------------------------------------------------------------------------
# T021 -- parent determinant embedding
# ---------------------------------------------------------------------------

def embed_variant_det(
    parent: ParentOrbitalSpace,
    variant: ActiveSpaceVariant,
    active_alpha_bits: int,
    active_beta_bits: int,
) -> DeterminantKey:
    """Embed a variant CAS determinant into the full parent comparison bitstring."""
    alpha = 0
    beta = 0

    parentpos = {
        orbital_index: compact_pos
        for compact_pos, orbital_index in enumerate(parent.comparison_orbitals)
    }

    for orb in variant.inactive_orbitals:
        if orb in parentpos:
            p = parentpos[orb]
            alpha |= 1 << p
            beta |= 1 << p

    for aidx, orb in enumerate(variant.active_orbitals):
        if orb not in parentpos:
            raise ParentEmbeddingError("active orbital absent from comparison space", {"orbital": orb})
        p = parentpos[orb]
        if (active_alpha_bits >> aidx) & 1:
            alpha |= 1 << p
        if (active_beta_bits >> aidx) & 1:
            beta |= 1 << p

    key = DeterminantKey(alpha, beta)
    return key


def embed_sparse_vector(
    parent: ParentOrbitalSpace,
    variant: ActiveSpaceVariant,
    vec: SparseStateVector,
) -> SparseStateVector:
    """Embed every determinant of a sparse vector into the parent gauge."""
    keys = tuple(
        embed_variant_det(parent, variant, det.alpha, det.beta)
        for det in vec.determinant_keys
    )
    return SparseStateVector(
        state_id=vec.state_id,
        sector_key=vec.sector_key,
        determinant_keys=keys,
        coefficients=vec.coefficients.copy(),
    )


# ---------------------------------------------------------------------------
# T030 -- family pool statistics
# ---------------------------------------------------------------------------

def family_weights_for_state(
    state_vec: SparseStateVector, norb: int
) -> dict[SpatialOccupationKey, float]:
    """Aggregate ``|c|^2`` per spatial-occupation family for one state."""
    out: dict[SpatialOccupationKey, float] = defaultdict(float)
    for det, c in zip(state_vec.determinant_keys, state_vec.coefficients):
        out[family_of(det, norb)] += float(abs(c) ** 2)
    return dict(out)


def variant_family_stats(
    aligned_states: Sequence[SparseStateVector],
    state_weights: Sequence[float],
    norb: int,
):
    """Per-variant family weights.

    Returns ``(q, by_state)`` where ``q[f]`` is the state-averaged family weight
    and ``by_state[A][f]`` is the per-state family weight.
    """
    by_state = [family_weights_for_state(v, norb) for v in aligned_states]
    families = set()
    for bs in by_state:
        families.update(bs.keys())
    q: dict[SpatialOccupationKey, float] = {}
    for f in families:
        q[f] = sum(w * bs.get(f, 0.0) for w, bs in zip(state_weights, by_state))
    return q, by_state


def aggregate_family_pool(
    variant_q: Sequence[dict[SpatialOccupationKey, float]],
    variant_by_state: Sequence[Sequence[dict[SpatialOccupationKey, float]]],
    variant_weights: Sequence[float],
    tau_persist: float,
) -> dict[SpatialOccupationKey, FamilyStats]:
    """Aggregate family statistics over variants.

    ``qbar`` is the ensemble-averaged family weight; ``persistence`` is the
    ensemble-weighted fraction of variants where the family exceeds
    ``tau_persist``; ``max_state_weight`` protects low-weight roots.
    """
    families = set()
    for q in variant_q:
        families.update(q.keys())
    nvariant = len(variant_q)
    nstate = max((len(bs) for bs in variant_by_state), default=0)

    stats: dict[SpatialOccupationKey, FamilyStats] = {}
    for f in families:
        qv = tuple(vq.get(f, 0.0) for vq in variant_q)
        qbar = sum(w * qa for w, qa in zip(variant_weights, qv))
        persistence = sum(
            w * (1.0 if qa > tau_persist else 0.0)
            for w, qa in zip(variant_weights, qv)
        )
        q_by_state = np.zeros((nvariant, nstate))
        for a in range(nvariant):
            for s in range(nstate):
                q_by_state[a, s] = variant_by_state[a][s].get(f, 0.0)
        max_state = float(q_by_state.max()) if q_by_state.size else 0.0
        stats[f] = FamilyStats(
            family=f,
            q_by_variant=qv,
            q_by_state_variant=q_by_state,
            qbar=qbar,
            persistence=persistence,
            max_state_weight=max_state,
        )
    return stats


# ---------------------------------------------------------------------------
# T031 -- common support seed
# ---------------------------------------------------------------------------

def captured_average(stats: Sequence[FamilyStats]) -> float:
    return float(sum(s.qbar for s in stats))


def seed_support(
    pool: Mapping[SpatialOccupationKey, FamilyStats],
    protected: Sequence[SpatialOccupationKey],
    cfg,
) -> tuple[SpatialOccupationKey, ...]:
    """Select the initial common family support from the pool."""
    selected = set(protected)

    order = sorted(
        pool.values(),
        key=lambda s: (-s.qbar, -s.max_state_weight, -s.persistence, s.family.occ),
    )

    def criteria_met(sel: set[SpatialOccupationKey]) -> bool:
        sel_stats = [pool[f] for f in sel if f in pool]
        avg = captured_average(sel_stats)
        # per-state-variant captured weight: min over (variant, state)
        per_state_min = _min_state_variant_captured(sel, pool)
        return (
            avg >= cfg.support_seed.captured_average_weight
            and per_state_min >= cfg.support_seed.minimum_captured_weight_per_state
            and len(sel) >= cfg.support_seed.min_support_families
        )

    for stat in order:
        if criteria_met(selected):
            break
        selected.add(stat.family)

    if not criteria_met(selected):
        raise ReferenceNotCompact(
            "seed criteria impossible from stored teacher pool",
            {"n_pool": len(pool)},
        )
    return tuple(sorted(selected))


def _min_state_variant_captured(
    sel: set[SpatialOccupationKey], pool: Mapping[SpatialOccupationKey, FamilyStats]
) -> float:
    captured = None
    for f in sel:
        st = pool.get(f)
        if st is None:
            continue
        q = st.q_by_state_variant
        if captured is None:
            captured = q.copy()
        else:
            captured = captured + q
    if captured is None:
        return 0.0
    return float(captured.min())


def active_occupancy_of(
    family: SpatialOccupationKey, variant: ActiveSpaceVariant
) -> Optional[tuple[int, ...]]:
    """Return the active-orbital occupancy sub-tuple, or None if incompatible.

    A parent family is compatible with a variant iff every inactive orbital has
    occupancy 2 and every external (non-active, non-inactive) orbital has
    occupancy 0.
    """
    occ = family.occ
    active = variant.active_orbitals
    inactive = set(variant.inactive_orbitals)
    for orb in variant.inactive_orbitals:
        if orb >= len(occ) or occ[orb] != 2:
            return None
    # external orbitals (not active, not inactive) must be empty
    for orb in range(len(occ)):
        if orb in active or orb in inactive:
            continue
        if occ[orb] != 0:
            return None
    return tuple(occ[orb] for orb in active)


def enumerate_variant_family(
    parent: ParentOrbitalSpace,
    variant: ActiveSpaceVariant,
    family: SpatialOccupationKey,
    na_act: int,
    nb_act: int,
) -> tuple[DeterminantKey, ...]:
    """Enumerate parent-embedded determinants of a family for one variant."""
    active_occ = active_occupancy_of(family, variant)
    if active_occ is None:
        return ()
    active_family = SpatialOccupationKey(active_occ)
    dets_act = enumerate_family(active_family, na_act, nb_act)
    out = []
    for d in dets_act:
        out.append(embed_variant_det(parent, variant, d.alpha, d.beta))
    return tuple(out)


def variant_family_determinants(
    parent: ParentOrbitalSpace,
    variant: ActiveSpaceVariant,
    families: Sequence[SpatialOccupationKey],
    na_act: int,
    nb_act: int,
) -> tuple[DeterminantKey, ...]:
    """Union of parent-embedded determinants of ``families`` for one variant."""
    dets = set()
    for f in families:
        dets.update(enumerate_variant_family(parent, variant, f, na_act, nb_act))
    return tuple(sorted(dets))


__all__ = [
    "embed_variant_det",
    "embed_sparse_vector",
    "family_weights_for_state",
    "variant_family_stats",
    "aggregate_family_pool",
    "captured_average",
    "seed_support",
    "active_occupancy_of",
    "enumerate_variant_family",
    "variant_family_determinants",
]
