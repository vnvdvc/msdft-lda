"""Backward family pruning with exact trial recheck (T040)."""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

from .models import FamilyStats, SelectedSupport, SpatialOccupationKey


def removal_priority(
    support: SelectedSupport,
    pool: Mapping[SpatialOccupationKey, FamilyStats],
) -> list[SpatialOccupationKey]:
    """Order removable families by lowest impact first."""
    families = [f for f in support.families if f not in support.protected]
    key = lambda f: (  # noqa: E731
        pool[f].qbar if f in pool else 0.0,
        pool[f].max_state_weight if f in pool else 0.0,
        pool[f].persistence if f in pool else 0.0,
        f.occ,
    )
    return sorted(families, key=key)


def prune_support(
    support: SelectedSupport,
    pool: Mapping[SpatialOccupationKey, FamilyStats],
    evaluate: Callable[[SelectedSupport], bool],
    safety_factor: float = 0.90,
) -> SelectedSupport:
    """Remove redundant families while every mandatory criterion stays safe.

    ``evaluate(support)`` must return ``True`` when the support passes *all*
    mandatory criteria with the pruning safety margin already applied.  Each
    accepted deletion is re-checked exactly (no caching assumptions).
    """
    current = support
    candidates = removal_priority(current, pool)
    for f in candidates:
        trial_families = tuple(x for x in current.families if x != f)
        trial = SelectedSupport(
            families=trial_families,
            sector_support={},  # will be rematerialized by evaluate
            protected=current.protected,
            generation=current.generation + 1,
        )
        if evaluate(trial):
            current = trial
    return current


__all__ = ["removal_priority", "prune_support"]
