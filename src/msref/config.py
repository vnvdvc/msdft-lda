"""Typed configuration models and YAML loading for msref.

The canonical defaults live in ``DEFAULT_CONFIG.yaml`` (mirrored at
``examples/default_config.yaml``).  This module mirrors those defaults as
stdlib dataclasses and provides:

* ``from_yaml`` / ``from_dict`` loading;
* profile override application without mutating source data;
* cross-field validation;
* a deterministic ``config_hash`` over the canonical resolved config.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass, field, fields
from typing import Any, Optional

from .exceptions import InputValidationError


def _build(cls, data: dict[str, Any]):
    """Instantiate ``cls`` from a dict, ignoring unknown keys and applying defaults."""
    known = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in known}
    return cls(**kwargs)


@dataclass
class RuntimeConfig:
    dtype: str = "float64"
    complex_enabled: bool = True
    deterministic: bool = True
    max_memory_mb: int = 8000
    n_processes: int = 1
    blas_threads_per_process: int = 1
    checkpoint_every_iteration: bool = True
    fail_on_warning_categories: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d):  # pragma: no cover - trivial wrapper
        return _build(cls, d)


@dataclass
class BackendConfig:
    name: str = "pyscf"
    target_version_family: str = "2.14.x"
    require_capabilities: list[str] = field(default_factory=list)
    allow_private_api: bool = False
    private_api_requires_guard: bool = True

    @classmethod
    def from_dict(cls, d):
        return _build(cls, d)


@dataclass
class TeacherConfig:
    casscf_energy_tol_eh: float = 1.0e-9
    casscf_grad_tol: float = 3.0e-5
    max_macro_cycles: int = 100
    max_micro_cycles: Optional[int] = None
    spin_penalty_shift_eh: float = 0.2
    spin_square_abs_tol: float = 1.0e-5
    state_weight_sum_tol: float = 1.0e-12
    require_converged: bool = True
    canonicalize_after_casscf: bool = False

    @classmethod
    def from_dict(cls, d):
        return _build(cls, d)


@dataclass
class OrbitalGaugeConfig:
    mode: str = "anchored_procrustes"
    preserve_symmetry_blocks: bool = True
    phase_fix: str = "max_reference_overlap"
    orbital_match_min_sigma: float = 0.90
    orbital_match_warn_sigma: float = 0.97
    degeneracy_tie_tol: float = 1.0e-7
    deterministic_tie_break: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d):
        return _build(cls, d)


@dataclass
class StateTrackingConfig:
    cluster_mode: str = "automatic_with_user_overrides"
    state_match_min_overlap: float = 0.50
    state_match_ambiguity_tol: float = 0.05
    cluster_gap_threshold_eh: float = 0.005
    use_hungarian_assignment: bool = True
    use_procrustes_for_clusters: bool = True
    root_buffer: int = 2
    max_root_buffer: int = 8
    forbid_energy_only_fallback: bool = True

    @classmethod
    def from_dict(cls, d):
        return _build(cls, d)


@dataclass
class ActiveSpaceEnsembleConfig:
    tier: str = "fixed_orbital"
    n_active_spaces: int = 12
    require_all_variants: bool = True
    minimum_success_fraction_development: float = 0.90
    max_add_orbitals: int = 2
    allow_remove: bool = False
    allow_exchange: bool = True
    n_random_variants: int = 4
    rotation_angle_rad: float = 0.05
    preserve_symmetry_blocks: bool = True
    match_local_character: bool = True
    max_variant_active_orbitals: Optional[int] = None
    variant_weights: str = "uniform"

    @classmethod
    def from_dict(cls, d):
        return _build(cls, d)


@dataclass
class TeacherCIStorageConfig:
    coefficient_io_screen: float = 1.0e-6
    minimum_captured_norm_per_state: float = 0.999999
    sparse_storage: bool = True

    @classmethod
    def from_dict(cls, d):
        return _build(cls, d)


@dataclass
class SupportSeedConfig:
    selection_unit: str = "spatial_occupation_family"
    q_persistence_threshold: float = 1.0e-5
    captured_average_weight: float = 0.995
    minimum_captured_weight_per_state: float = 0.99
    minimum_persistence_preference: float = 0.50
    persistence_is_hard_filter: bool = False
    min_support_families: int = 1
    protect_user_reference_families: bool = True

    @classmethod
    def from_dict(cls, d):
        return _build(cls, d)


@dataclass
class SpinCompletionConfig:
    enabled: bool = True
    mandatory: bool = True
    verify_after_each_solve: bool = True
    spin_square_abs_tol: float = 1.0e-5
    max_determinant_expansion_factor_warn: float = 20.0
    max_determinant_expansion_factor_fail: float = 200.0

    @classmethod
    def from_dict(cls, d):
        return _build(cls, d)


@dataclass
class SelectedSolverConfig:
    reference_dense_max_determinants: int = 5000
    explicit_sparse_max_determinants: int = 50000
    eigensolver: str = "auto"
    eig_tol_eh: float = 1.0e-10
    eig_max_cycles: int = 100
    eig_max_subspace: int = 30
    eig_lindep: float = 1.0e-12
    numerical_preconditioner_shift_eh: float = 1.0e-6
    max_support_families: int = 2000
    max_support_determinants: int = 20000
    add_batch_size: int = 4
    max_selection_iterations: int = 25

    @classmethod
    def from_dict(cls, d):
        return _build(cls, d)


@dataclass
class StaticLeakageConfig:
    denominator_mode: str = "en_diagonal"
    tau_static_max: float = 0.020
    tau_static_rms: float = 0.030
    delta_intruder_eh: float = 0.050
    tau_coupling_eh: float = 1.0e-4
    delta_numeric_eh: float = 1.0e-10
    max_external_candidates: int = 2000000
    external_prescreen_enabled: bool = False
    pt2_energy_used_for_selection: bool = False
    store_top_external_families: int = 100

    @classmethod
    def from_dict(cls, d):
        return _build(cls, d)


@dataclass
class AcceptanceConfig:
    tau_projector: float = 0.020
    tau_projector_frobenius: Optional[float] = None
    tau_gamma: float = 1.0e-3
    tau_gamma_block_max: float = 3.0e-3
    gamma_block_weighting: str = "state_weight_product"
    tau_E_SA_eh: float = 5.0e-4
    energy_is_mandatory: bool = True
    tau_excitation_gap_ev: float = 0.020
    excitation_gap_is_mandatory: bool = False
    tau_family_hellinger: float = 0.050
    family_distribution_is_mandatory: bool = False
    minimum_weighted_jaccard: float = 0.90
    support_metric_is_mandatory: bool = False
    spin_is_mandatory: bool = True
    symmetry_is_mandatory: bool = True
    static_leakage_is_mandatory: bool = True
    compactness_is_mandatory: bool = True
    require_all_active_space_variants: bool = True

    @classmethod
    def from_dict(cls, d):
        return _build(cls, d)


@dataclass
class RepairConfig:
    density_candidate_pool: int = 24
    density_add_batch_size: int = 1
    projector_candidate_pool: int = 24
    projector_add_batch_size: int = 1
    exact_trial_density_leverage: bool = True
    max_repair_rounds: int = 20

    @classmethod
    def from_dict(cls, d):
        return _build(cls, d)


@dataclass
class PruningConfig:
    enabled: bool = True
    safety_factor: float = 0.90
    exact_recheck: bool = True
    batch_removal: bool = False
    cache_trials: bool = True

    @classmethod
    def from_dict(cls, d):
        return _build(cls, d)


@dataclass
class MetricsConfig:
    ao_metric: str = "symmetric_orthogonal"
    matrix_rdm_norm: str = "frobenius"
    matrix_rdm_normalize_by_electrons: bool = True
    configuration_distance: str = "hellinger"
    support_similarity: str = "weighted_jaccard"
    report_pt2_tail: bool = True
    report_individual_state_overlaps: bool = True
    report_principal_angles: bool = True

    @classmethod
    def from_dict(cls, d):
        return _build(cls, d)


@dataclass
class IOConfig:
    hdf5_schema_version: str = "1.0"
    output_h5: str = "reference.msref.h5"
    output_json: str = "reference.msref.json"
    write_iteration_history: bool = True
    write_teacher_sparse_ci: bool = True
    store_mo_coefficients: bool = True
    store_ao_overlap: bool = True
    store_matrix_rdm: bool = True
    hash_major_arrays: bool = True

    @classmethod
    def from_dict(cls, d):
        return _build(cls, d)


@dataclass
class BenchmarkCap:
    max_support_families: int = 2000
    max_support_determinants: int = 20000

    @classmethod
    def from_dict(cls, d):
        return _build(cls, d)


@dataclass
class BenchmarkCaps:
    h2: BenchmarkCap = field(default_factory=BenchmarkCap)
    n2: BenchmarkCap = field(default_factory=BenchmarkCap)
    o2: BenchmarkCap = field(default_factory=BenchmarkCap)
    c2: BenchmarkCap = field(default_factory=BenchmarkCap)

    @classmethod
    def from_dict(cls, d):
        if not isinstance(d, dict):
            return cls()
        return cls(
            h2=BenchmarkCap.from_dict(d.get("h2", {})),
            n2=BenchmarkCap.from_dict(d.get("n2", {})),
            o2=BenchmarkCap.from_dict(d.get("o2", {})),
            c2=BenchmarkCap.from_dict(d.get("c2", {})),
        )


@dataclass
class MSRefConfig:
    """Top-level resolved configuration."""

    schema_version: str = "1.0"
    profile: str = "production"
    seed: int = 20260818
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    backend: BackendConfig = field(default_factory=BackendConfig)
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    orbital_gauge: OrbitalGaugeConfig = field(default_factory=OrbitalGaugeConfig)
    state_tracking: StateTrackingConfig = field(default_factory=StateTrackingConfig)
    active_space_ensemble: ActiveSpaceEnsembleConfig = field(
        default_factory=ActiveSpaceEnsembleConfig
    )
    teacher_ci_storage: TeacherCIStorageConfig = field(default_factory=TeacherCIStorageConfig)
    support_seed: SupportSeedConfig = field(default_factory=SupportSeedConfig)
    spin_completion: SpinCompletionConfig = field(default_factory=SpinCompletionConfig)
    selected_solver: SelectedSolverConfig = field(default_factory=SelectedSolverConfig)
    static_leakage: StaticLeakageConfig = field(default_factory=StaticLeakageConfig)
    calibration_ranges: dict[str, list[float]] = field(default_factory=dict)
    acceptance: AcceptanceConfig = field(default_factory=AcceptanceConfig)
    repair: RepairConfig = field(default_factory=RepairConfig)
    pruning: PruningConfig = field(default_factory=PruningConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    io: IOConfig = field(default_factory=IOConfig)
    profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    benchmark_caps: BenchmarkCaps = field(default_factory=BenchmarkCaps)

    # -- loading -----------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "MSRefConfig":
        cfg = cls(
            schema_version=d.get("schema_version", "1.0"),
            profile=d.get("profile", "production"),
            seed=d.get("seed", 20260818),
            runtime=RuntimeConfig.from_dict(d.get("runtime", {})),
            backend=BackendConfig.from_dict(d.get("backend", {})),
            teacher=TeacherConfig.from_dict(d.get("teacher", {})),
            orbital_gauge=OrbitalGaugeConfig.from_dict(d.get("orbital_gauge", {})),
            state_tracking=StateTrackingConfig.from_dict(d.get("state_tracking", {})),
            active_space_ensemble=ActiveSpaceEnsembleConfig.from_dict(
                d.get("active_space_ensemble", {})
            ),
            teacher_ci_storage=TeacherCIStorageConfig.from_dict(
                d.get("teacher_ci_storage", {})
            ),
            support_seed=SupportSeedConfig.from_dict(d.get("support_seed", {})),
            spin_completion=SpinCompletionConfig.from_dict(d.get("spin_completion", {})),
            selected_solver=SelectedSolverConfig.from_dict(d.get("selected_solver", {})),
            static_leakage=StaticLeakageConfig.from_dict(d.get("static_leakage", {})),
            calibration_ranges=dict(d.get("calibration_ranges", {})),
            acceptance=AcceptanceConfig.from_dict(d.get("acceptance", {})),
            repair=RepairConfig.from_dict(d.get("repair", {})),
            pruning=PruningConfig.from_dict(d.get("pruning", {})),
            metrics=MetricsConfig.from_dict(d.get("metrics", {})),
            io=IOConfig.from_dict(d.get("io", {})),
            profiles={k: dict(v) for k, v in d.get("profiles", {}).items()},
            benchmark_caps=BenchmarkCaps.from_dict(d.get("benchmark_caps", {})),
        )
        return cfg

    @classmethod
    def from_yaml(cls, path: str) -> "MSRefConfig":
        """Load and validate configuration from a YAML file."""
        import yaml  # local import to keep config loadable without pyyaml at import time

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not isinstance(data, dict):
            raise InputValidationError("configuration YAML must be a mapping", {"path": path})
        cfg = cls.from_dict(data)
        cfg.apply_profile_overrides()
        cfg.validate()
        return cfg

    @classmethod
    def from_yaml_string(cls, text: str) -> "MSRefConfig":
        import yaml

        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise InputValidationError("configuration YAML must be a mapping")
        cfg = cls.from_dict(data)
        cfg.apply_profile_overrides()
        cfg.validate()
        return cfg

    # -- profiles ----------------------------------------------------------

    def apply_profile_overrides(self) -> None:
        """Apply dotted-key profile overrides without mutating source data."""
        override_map = {
            "development": self.profiles.get("development_overrides", {}),
            "strict": self.profiles.get("strict_overrides", {}),
        }
        overrides = override_map.get(self.profile, {})
        for dotted_key, value in overrides.items():
            self._set_dotted(dotted_key, value)

    def _set_dotted(self, dotted_key: str, value: Any) -> None:
        parts = dotted_key.split(".")
        obj: Any = self
        for part in parts[:-1]:
            if not hasattr(obj, part):
                raise InputValidationError(
                    f"unknown configuration path {dotted_key!r}", {"path": dotted_key}
                )
            obj = getattr(obj, part)
        leaf = parts[-1]
        if not hasattr(obj, leaf):
            raise InputValidationError(
                f"unknown configuration path {dotted_key!r}", {"path": dotted_key}
            )
        setattr(obj, leaf, value)

    # -- validation --------------------------------------------------------

    def validate(self) -> None:
        """Validate thresholds, probabilities, and cross-field constraints."""

        def _nonneg(name: str, value: Any) -> None:
            if value is None:
                return
            if value < 0:
                raise InputValidationError(f"{name} must be non-negative", {"value": value})

        def _prob(name: str, value: Any) -> None:
            if value is None:
                return
            if not (0.0 <= value <= 1.0):
                raise InputValidationError(
                    f"{name} must be in [0, 1]", {"value": value}
                )

        s = self.support_seed
        _prob("support_seed.captured_average_weight", s.captured_average_weight)
        _prob("support_seed.minimum_captured_weight_per_state",
              s.minimum_captured_weight_per_state)
        _prob("support_seed.minimum_persistence_preference",
              s.minimum_persistence_preference)
        _nonneg("support_seed.q_persistence_threshold", s.q_persistence_threshold)

        sc = self.static_leakage
        _nonneg("static_leakage.tau_static_max", sc.tau_static_max)
        _nonneg("static_leakage.tau_static_rms", sc.tau_static_rms)
        _nonneg("static_leakage.delta_intruder_eh", sc.delta_intruder_eh)
        _nonneg("static_leakage.tau_coupling_eh", sc.tau_coupling_eh)
        _nonneg("static_leakage.delta_numeric_eh", sc.delta_numeric_eh)

        a = self.acceptance
        _nonneg("acceptance.tau_projector", a.tau_projector)
        _nonneg("acceptance.tau_gamma", a.tau_gamma)
        _nonneg("acceptance.tau_gamma_block_max", a.tau_gamma_block_max)
        _nonneg("acceptance.tau_E_SA_eh", a.tau_E_SA_eh)

        t = self.teacher
        _nonneg("teacher.casscf_energy_tol_eh", t.casscf_energy_tol_eh)
        _nonneg("teacher.casscf_grad_tol", t.casscf_grad_tol)
        _nonneg("teacher.spin_square_abs_tol", t.spin_square_abs_tol)

        sp = self.spin_completion
        _nonneg("spin_completion.spin_square_abs_tol", sp.spin_square_abs_tol)

        p = self.pruning
        if not (0.0 < p.safety_factor <= 1.0):
            raise InputValidationError(
                "pruning.safety_factor must be in (0, 1]", {"value": p.safety_factor}
            )

        if self.profile not in ("production", "development", "strict"):
            raise InputValidationError(f"unknown profile {self.profile!r}")

        if self.seed is None and self.profile == "production":
            raise InputValidationError("production profile requires a deterministic seed")

    # -- serialization / hashing ------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Return a nested plain-dict representation (dataclasses -> dict)."""
        return _to_dict(self)

    def canonical_json(self) -> str:
        """Deterministic JSON over the canonical resolved config."""
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=_json_default,
        )

    def config_hash(self) -> str:
        """SHA-256 of the canonical resolved configuration (hex digest)."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def resolved_thresholds(self) -> dict[str, Any]:
        """Flatten the resolved, serialization-relevant thresholds for reports."""
        a = self.acceptance
        sc = self.static_leakage
        return {
            "tau_projector": a.tau_projector,
            "tau_gamma": a.tau_gamma,
            "tau_gamma_block_max": a.tau_gamma_block_max,
            "tau_E_SA_eh": a.tau_E_SA_eh,
            "tau_static_max": sc.tau_static_max,
            "tau_static_rms": sc.tau_static_rms,
            "delta_intruder_eh": sc.delta_intruder_eh,
            "tau_coupling_eh": sc.tau_coupling_eh,
            "spin_square_abs_tol": self.spin_completion.spin_square_abs_tol,
            "max_support_families": self.selected_solver.max_support_families,
            "max_support_determinants": self.selected_solver.max_support_determinants,
        }


def _json_default(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return _to_dict(obj)
    if isinstance(obj, (list, tuple)):
        return list(obj)
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


def _to_dict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj):
        return {f.name: _to_dict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, dict):
        return {str(k): _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    return obj


__all__ = [
    "MSRefConfig",
    "RuntimeConfig",
    "BackendConfig",
    "TeacherConfig",
    "OrbitalGaugeConfig",
    "StateTrackingConfig",
    "ActiveSpaceEnsembleConfig",
    "TeacherCIStorageConfig",
    "SupportSeedConfig",
    "SpinCompletionConfig",
    "SelectedSolverConfig",
    "StaticLeakageConfig",
    "AcceptanceConfig",
    "RepairConfig",
    "PruningConfig",
    "MetricsConfig",
    "IOConfig",
    "BenchmarkCap",
    "BenchmarkCaps",
]
