"""Core data records for msref.

This module contains *pure* records with no PySCF imports.  Conventions follow
``DATA_CONTRACTS.yaml``:

* orbital indices are zero-based;
* energies are Hartree internally (angles radians);
* CI determinant keys use parent-orbital bit positions ``(alpha, beta)``;
* matrix 1-RDM follows PySCF: ``gamma[p,q] = <q^dagger p>``;
* state weights sum to one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Sequence, Union

import numpy as np

from .exceptions import InputValidationError

# ---------------------------------------------------------------------------
# Small, hashable, ordered key types
# ---------------------------------------------------------------------------

# (nalpha, nbeta, wfnsym) over the parent comparison space.
SectorKey = tuple[int, int, Optional[Union[str, int]]]


@dataclass(frozen=True, order=True)
class DeterminantKey:
    """A determinant as a pair of integer bitstrings over parent orbitals."""

    alpha: int
    beta: int

    def __post_init__(self) -> None:
        if self.alpha < 0 or self.beta < 0:
            raise InputValidationError("DeterminantKey bits must be non-negative")


@dataclass(frozen=True, order=True)
class SpatialOccupationKey:
    """A spatial-occupation family: occ[p] in {0, 1, 2} per parent orbital."""

    occ: tuple[int, ...]

    def __post_init__(self) -> None:
        if any(o not in (0, 1, 2) for o in self.occ):
            raise InputValidationError("spatial occupation values must be 0, 1, or 2")

    @property
    def norb(self) -> int:
        return len(self.occ)

    @property
    def nelec(self) -> int:
        return sum(self.occ)


class Role(Enum):
    """Orbital role within an active-space variant."""

    FROZEN_CORE = "FROZEN_CORE"
    INACTIVE = "INACTIVE"
    ACTIVE = "ACTIVE"
    EXTERNAL = "EXTERNAL"
    FROZEN_EXTERNAL = "FROZEN_EXTERNAL"


# ---------------------------------------------------------------------------
# State specification and identity
# ---------------------------------------------------------------------------

@dataclass
class StateBlockSpec:
    """A spin/symmetry block of roots for the teacher solver."""

    block_id: str
    spin_S: float
    ms2: int  # 2 * M_S
    nroots: int
    weights: list[float]
    wfnsym: Optional[Union[str, int]] = None
    # total alpha/beta counts for the full molecule (used for electron accounting)
    nelec_total: Optional[tuple[int, int]] = None

    def __post_init__(self) -> None:
        if self.nroots < 1:
            raise InputValidationError("nroots must be >= 1")
        if len(self.weights) != self.nroots:
            raise InputValidationError("len(weights) must equal nroots")
        if any(w < 0 for w in self.weights):
            raise InputValidationError("state weights must be non-negative")
        if self.spin_S < 0:
            raise InputValidationError("spin_S must be >= 0")


@dataclass(frozen=True)
class StateID:
    """An immutable, descriptive identity for a target state."""

    name: str
    block_id: str
    spin_S: float
    ms2: int
    wfnsym: Optional[Union[str, int]] = None
    cluster_id: Optional[str] = None
    root_ordinal: int = 0

    def __post_init__(self) -> None:
        if self.root_ordinal < 0:
            raise InputValidationError("root_ordinal must be >= 0")


# ---------------------------------------------------------------------------
# Orbital space and active-space variants
# ---------------------------------------------------------------------------

@dataclass
class ParentOrbitalSpace:
    """Ordered parent orbital gauge shared by all active-space variants."""

    parent_id: str
    mo_coeff_ao: np.ndarray  # [nao, nmo], complex-capable
    ao_overlap: np.ndarray  # [nao, nao]
    orbital_ids: tuple[str, ...]
    orbsym: tuple[Optional[int], ...] = ()
    anchor_labels: tuple[Optional[str], ...] = ()
    always_core: tuple[int, ...] = ()
    comparison_orbitals: tuple[int, ...] = ()
    always_external: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        nmo = self.mo_coeff_ao.shape[1]
        if len(self.orbital_ids) != nmo:
            raise InputValidationError("len(orbital_ids) must equal nmo")
        if len(set(self.orbital_ids)) != len(self.orbital_ids):
            raise InputValidationError("orbital_ids must be unique")
        roles = set(self.always_core) | set(self.comparison_orbitals) | set(self.always_external)
        if len(roles) != (
            len(self.always_core) + len(self.comparison_orbitals) + len(self.always_external)
        ):
            raise InputValidationError("orbital role groups must be disjoint")


@dataclass
class ActiveSpaceVariant:
    """One active-space definition: a role for every parent orbital."""

    variant_id: str
    parent_id: str
    roles: dict[int, Role]  # parent orbital index -> Role
    active_orbitals: tuple[int, ...]
    inactive_orbitals: tuple[int, ...] = ()
    nelecas_by_sector: dict[SectorKey, tuple[int, int]] = field(default_factory=dict)
    generation_rule: str = "nominal"
    perturbation_seed: Optional[int] = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        if self.weight < 0:
            raise InputValidationError("variant weight must be >= 0")
        for idx, role in self.roles.items():
            if idx < 0:
                raise InputValidationError("orbital indices must be non-negative")
        # active orbitals must be labelled ACTIVE in roles
        for idx in self.active_orbitals:
            if self.roles.get(idx) is not Role.ACTIVE:
                raise InputValidationError("active_orbitals must be labelled ACTIVE")


# ---------------------------------------------------------------------------
# Sparse state vectors
# ---------------------------------------------------------------------------

@dataclass
class SparseStateVector:
    """A CI vector represented sparsely over parent determinant keys."""

    state_id: StateID
    sector_key: SectorKey
    determinant_keys: tuple[DeterminantKey, ...]
    coefficients: np.ndarray  # complex128, shape [ndet]

    def __post_init__(self) -> None:
        self.coefficients = np.asarray(self.coefficients, dtype=complex)
        if self.coefficients.ndim != 1:
            raise InputValidationError("coefficients must be 1-D")
        if len(self.coefficients) != len(self.determinant_keys):
            raise InputValidationError("coefficients and determinant_keys must match length")
        if len(set(self.determinant_keys)) != len(self.determinant_keys):
            raise InputValidationError("determinant_keys must be unique")

    def as_dict(self) -> dict[DeterminantKey, complex]:
        return {k: c for k, c in zip(self.determinant_keys, self.coefficients)}


# ---------------------------------------------------------------------------
# Teacher result
# ---------------------------------------------------------------------------

@dataclass
class TeacherResult:
    """Normalized output of a spin-mixed SA-CASSCF teacher."""

    variant_id: str
    parent_id: str
    mo_coeff_ao: np.ndarray
    energies_eh: np.ndarray  # [nstate]
    state_ids: list[StateID]
    state_weights: np.ndarray  # [nstate]
    spin_square: np.ndarray  # [nstate]
    ci_vectors: list[SparseStateVector]
    gamma1_parent: np.ndarray  # [nstate, nstate, nparent, nparent]
    converged: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def nstate(self) -> int:
        return len(self.state_ids)


# ---------------------------------------------------------------------------
# Selected support and result
# ---------------------------------------------------------------------------

@dataclass
class SectorSupport:
    """Determinant realization of the family support in one spin sector."""

    sector: SectorKey
    determinant_keys: tuple[DeterminantKey, ...]
    index: dict[DeterminantKey, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.index:
            self.index = {k: i for i, k in enumerate(self.determinant_keys)}

    def __len__(self) -> int:
        return len(self.determinant_keys)


@dataclass
class SelectedSupport:
    """A common family support materialized per spin sector."""

    families: tuple[SpatialOccupationKey, ...]
    sector_support: dict[SectorKey, SectorSupport]
    protected: frozenset[SpatialOccupationKey] = frozenset()
    generation: int = 0
    support_hash: str = ""

    @property
    def nfamilies(self) -> int:
        return len(self.families)


@dataclass
class SelectedResult:
    """Rediagonalized compact reference in the selected space."""

    support: SelectedSupport
    energies_eh: np.ndarray  # [nstate]
    state_ids: list[StateID]
    coeffs_by_sector: dict[SectorKey, np.ndarray] = field(default_factory=dict)
    gamma1_parent: Optional[np.ndarray] = None
    spin_square: Optional[np.ndarray] = None
    symmetry_labels: Optional[list] = None
    convergence: bool = False


# ---------------------------------------------------------------------------
# Statistics / scoring / reports
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FamilyStats:
    """Aggregate family statistics over variants and states."""

    family: SpatialOccupationKey
    q_by_variant: tuple[float, ...]
    q_by_state_variant: np.ndarray  # [nvariant, nstate]
    qbar: float
    persistence: float
    max_state_weight: float


@dataclass
class ExternalFamilyScore:
    """Static-leakage score for one external spatial-occupation family."""

    family_key: SpatialOccupationKey
    determinant_keys: tuple[DeterminantKey, ...]
    v_by_state: np.ndarray  # [nstate]
    denominator_by_state: np.ndarray  # [nstate]
    t_by_state: np.ndarray  # [nstate]
    static_score_max: float
    static_score_rms: float
    intruder: bool = False
    pt2_eh: float = 0.0
    mandatory: bool = False


@dataclass
class StabilityReport:
    """Machine-readable certification report with exact threshold rationale."""

    passed: bool
    failed_criteria: list[str]
    max_projector_distance: float = 0.0
    max_gamma_error: float = 0.0
    max_gamma_block_error: float = 0.0
    sa_energy_span_eh: float = 0.0
    max_external_amplitude: float = 0.0
    max_external_family_rms: float = 0.0
    max_spin_square_error: float = 0.0
    family_hellinger_max: float = 0.0
    weighted_jaccard_min: float = 0.0
    n_families: int = 0
    n_determinants_by_sector: dict[SectorKey, int] = field(default_factory=dict)
    compression_ratio_by_sector: dict[SectorKey, float] = field(default_factory=dict)
    active_space_variant_results: list = field(default_factory=list)
    thresholds_resolved: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        return {
            "passed": self.passed,
            "failed_criteria": self.failed_criteria,
            "max_projector_distance": self.max_projector_distance,
            "max_gamma_error": self.max_gamma_error,
            "max_gamma_block_error": self.max_gamma_block_error,
            "sa_energy_span_eh": self.sa_energy_span_eh,
            "max_external_amplitude": self.max_external_amplitude,
            "max_external_family_rms": self.max_external_family_rms,
            "max_spin_square_error": self.max_spin_square_error,
            "family_hellinger_max": self.family_hellinger_max,
            "weighted_jaccard_min": self.weighted_jaccard_min,
            "n_families": self.n_families,
            "n_determinants_by_sector": {
                str(k): v for k, v in self.n_determinants_by_sector.items()
            },
            "compression_ratio_by_sector": {
                str(k): v for k, v in self.compression_ratio_by_sector.items()
            },
            "active_space_variant_results": self.active_space_variant_results,
            "thresholds_resolved": self.thresholds_resolved,
            "provenance": self.provenance,
        }


# ---------------------------------------------------------------------------
# Invariant helpers
# ---------------------------------------------------------------------------

def validate_state_weights(weights: Sequence[float], tol: float = 1e-12) -> None:
    """Raise unless the weights are non-negative and sum to one within ``tol``."""
    w = np.asarray(weights, dtype=float)
    if np.any(w < 0):
        raise InputValidationError("state weights must be non-negative")
    if abs(float(w.sum()) - 1.0) > tol:
        raise InputValidationError(
            f"state weights must sum to one, got {float(w.sum())!r}"
        )


__all__ = [
    "SectorKey",
    "DeterminantKey",
    "SpatialOccupationKey",
    "Role",
    "StateBlockSpec",
    "StateID",
    "ParentOrbitalSpace",
    "ActiveSpaceVariant",
    "SparseStateVector",
    "TeacherResult",
    "SectorSupport",
    "SelectedSupport",
    "SelectedResult",
    "FamilyStats",
    "ExternalFamilyScore",
    "StabilityReport",
    "validate_state_weights",
]
