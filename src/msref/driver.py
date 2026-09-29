"""Finite-state selection driver (T037-T041).

Orchestrates seed -> solve -> static promotion -> density/projector repair ->
prune -> certify.  This module contains no low-level math; it composes the
selected Hamiltonian, solver, density, residual, metrics, and pruning modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

import numpy as np

from .config import MSRefConfig
from .config_pool import aggregate_family_pool, seed_support, variant_family_stats
from .density import matrix_rdm_error, trans_rdm1_selected
from .exceptions import (
    DensityCriterionUnreachable,
    ProjectorCriterionUnreachable,
    ReferenceNotCompact,
    SelectionNonConvergence,
    StateTrackingError,
)
from .metrics import projector_distance, spin_square_error
from .models import (
    DeterminantKey,
    ExternalFamilyScore,
    FamilyStats,
    SelectedSupport,
    SpatialOccupationKey,
    SparseStateVector,
    StabilityReport,
    StateID,
)
from .residual import connected_external_families, score_external_family
from .selected_hamiltonian import build_selected_hamiltonian
from .selected_solver import (
    match_roots_by_overlap,
    normalize_and_phase_fix,
    phase_fix_to_reference,
    solve_selected_sector,
)
from .spin_completion import all_determinants, materialize_support


class DriverAction(str, Enum):
    SEED = "SEED"
    SOLVE = "SOLVE"
    PROMOTE_STATIC = "PROMOTE_STATIC"
    REPAIR_DENSITY = "REPAIR_DENSITY"
    REPAIR_PROJECTOR = "REPAIR_PROJECTOR"
    REPAIR_ENERGY = "REPAIR_ENERGY"
    PRUNE = "PRUNE"
    CERTIFY = "CERTIFY"
    DONE = "DONE"
    FAIL = "FAIL"


@dataclass
class SelectionProblem:
    """Single-variant, single-sector selection problem."""

    norb: int
    nelec: tuple[int, int]
    h1e: np.ndarray
    eri: np.ndarray
    teacher_states: list[SparseStateVector]
    state_weights: np.ndarray
    state_ids: list[StateID]
    cfg: MSRefConfig
    full_basis: tuple[DeterminantKey, ...] = ()
    spin_S: float = 0.0

    def __post_init__(self) -> None:
        if not self.full_basis:
            self.full_basis = all_determinants(self.norb, self.nelec[0], self.nelec[1])
        self.state_weights = np.asarray(self.state_weights, dtype=float)

    @property
    def sector(self):
        return (self.nelec[0], self.nelec[1], None)

    @property
    def nstate(self) -> int:
        return len(self.teacher_states)


@dataclass
class Evaluated:
    """Result of solving a support and evaluating all metrics."""

    support: SelectedSupport
    energies: np.ndarray
    coeffs: np.ndarray  # [n_support_det, nstate]
    support_keys: tuple[DeterminantKey, ...]
    gamma: np.ndarray
    spin_square: np.ndarray
    dP_op: float
    dP_fro: float
    gamma_error: float
    gamma_block_max: float
    spin_error: float
    external_scores: list[ExternalFamilyScore]
    mandatory_families: tuple[SpatialOccupationKey, ...]

    @property
    def passes(self) -> bool:
        cfg = None
        return False  # replaced by explicit check below


def _teacher_dicts(problem: SelectionProblem) -> list[dict]:
    return [{k: c for k, c in zip(v.determinant_keys, v.coefficients)}
            for v in problem.teacher_states]


def _teacher_dense(problem: SelectionProblem) -> np.ndarray:
    """[n_full, nstate] dense teacher states over the full basis."""
    index = {k: i for i, k in enumerate(problem.full_basis)}
    y = np.zeros((len(problem.full_basis), problem.nstate), dtype=complex)
    for s, v in enumerate(problem.teacher_states):
        for k, c in zip(v.determinant_keys, v.coefficients):
            y[index[k], s] = c
    return y


def _selected_dense(problem: SelectionProblem, support_keys, coeffs) -> np.ndarray:
    """[n_full, nstate] dense selected states over the full basis."""
    index = {k: i for i, k in enumerate(problem.full_basis)}
    y = np.zeros((len(problem.full_basis), coeffs.shape[1]), dtype=complex)
    for j, k in enumerate(support_keys):
        i = index.get(k)
        if i is not None:
            y[i, :] = coeffs[j, :]
    return y


def _gamma_from_states(problem: SelectionProblem, states_dense: np.ndarray) -> np.ndarray:
    """Matrix 1-RDM [A,B,p,q] from dense state columns over the full basis."""
    n = states_dense.shape[1]
    gamma = np.zeros((n, n, problem.norb, problem.norb), dtype=complex)
    for a in range(n):
        for b in range(n):
            gamma[a, b] = trans_rdm1_selected(
                states_dense[:, a], states_dense[:, b], problem.full_basis, problem.norb
            )
    return gamma


def _spin_square_selected(problem: SelectionProblem, support_keys, coeffs) -> np.ndarray:
    from .pyscf_adapter import sparse_ci_to_dense
    from pyscf import fci

    out = np.zeros(coeffs.shape[1])
    for s in range(coeffs.shape[1]):
        dense = sparse_ci_to_dense(support_keys, coeffs[:, s], problem.norb, problem.nelec)
        ss, _ = fci.spin_square(dense, problem.norb, problem.nelec)
        out[s] = ss
    return out


def solve_support(problem: SelectionProblem, support: SelectedSupport) -> Evaluated:
    """Solve a support, rediagonalize, match roots, and evaluate all metrics."""
    ss = support.sector_support[problem.sector]
    keys = ss.determinant_keys
    ham = build_selected_hamiltonian(
        keys, problem.h1e, problem.eri, max_memory_mb=problem.cfg.runtime.max_memory_mb
    )
    nroots = problem.nstate
    evals, evecs = solve_selected_sector(
        ham, nroots, problem.cfg.state_tracking.root_buffer, problem.cfg.selected_solver.eig_tol_eh
    )
    perm = match_roots_by_overlap(evecs, keys, _teacher_dicts(problem),
                                  problem.cfg.state_tracking.state_match_min_overlap)
    evecs = normalize_and_phase_fix(evecs)
    evecs = phase_fix_to_reference(evecs, keys, _teacher_dicts(problem), perm)

    coeffs = evecs[:, perm]
    energies = evals[perm]

    gamma_sel = _gamma_from_states(problem, _selected_dense(problem, keys, coeffs))
    gamma_teach = _gamma_from_states(problem, _teacher_dense(problem))

    allowed = np.ones((nroots, nroots), dtype=bool)
    gamma_err, gamma_block, _ = matrix_rdm_error(
        gamma_sel, gamma_teach, problem.state_weights, problem.nelec[0] + problem.nelec[1],
        allowed=allowed, mode=problem.cfg.acceptance.gamma_block_weighting,
    )

    d_op, d_fro, _ = projector_distance(
        _teacher_dense(problem), _selected_dense(problem, keys, coeffs)
    )

    spin = _spin_square_selected(problem, keys, coeffs)
    spin_err = max(spin_square_error(float(s), problem.spin_S) for s in spin)

    external = scan_static_leakage(problem, keys, coeffs, energies)
    mandatory = tuple(sorted({sc.family_key for sc in external if sc.mandatory}))

    return Evaluated(
        support=support,
        energies=energies,
        coeffs=coeffs,
        support_keys=keys,
        gamma=gamma_sel,
        spin_square=spin,
        dP_op=d_op,
        dP_fro=d_fro,
        gamma_error=gamma_err,
        gamma_block_max=gamma_block,
        spin_error=spin_err,
        external_scores=external,
        mandatory_families=mandatory,
    )


def scan_static_leakage(
    problem: SelectionProblem,
    support_keys: tuple[DeterminantKey, ...],
    coeffs: np.ndarray,
    energies: np.ndarray,
) -> list[ExternalFamilyScore]:
    """Score all connected external families for static leakage."""
    external_families = connected_external_families(support_keys, problem.norb)
    scores = []
    for f in external_families:
        sc = score_external_family(
            f, problem.nelec[0], problem.nelec[1],
            support_keys, coeffs, energies, problem.state_weights,
            problem.h1e, problem.eri, problem.cfg,
        )
        scores.append(sc)
    return scores


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------

def _candidate_families(problem: SelectionProblem, support: SelectedSupport,
                        pool: dict[SpatialOccupationKey, FamilyStats], top_k: int):
    unselected = [f for f in pool if f not in support.families]
    unselected.sort(key=lambda f: (-pool[f].qbar, -pool[f].max_state_weight, f.occ))
    return unselected[:top_k]


def density_repair(problem: SelectionProblem, support: SelectedSupport,
                   eval_now: Evaluated, pool, cfg) -> tuple[SelectedSupport, list]:
    """Trial-add families that reduce the matrix-1RDM error."""
    candidates = _candidate_families(problem, support, pool, cfg.repair.density_candidate_pool)
    best: list[tuple[float, SpatialOccupationKey]] = []
    for f in candidates:
        trial = _add_families(problem, support, [f])
        try:
            e = solve_support(problem, trial)
        except Exception:  # noqa: BLE001 - skip invalid trials
            continue
        leverage = (eval_now.gamma_error - e.gamma_error)
        best.append((leverage, f))
    best.sort(key=lambda x: -x[0])
    added = [f for lev, f in best[:cfg.repair.density_add_batch_size] if lev > 0]
    if not added:
        raise DensityCriterionUnreachable("no family reduces matrix-1RDM error")
    return _add_families(problem, support, added), added


def projector_repair(problem: SelectionProblem, support: SelectedSupport,
                     eval_now: Evaluated, pool, cfg) -> tuple[SelectedSupport, list]:
    """Trial-add families that improve the state-projector distance."""
    candidates = _candidate_families(problem, support, pool, cfg.repair.projector_candidate_pool)
    best: list[tuple[float, SpatialOccupationKey]] = []
    for f in candidates:
        trial = _add_families(problem, support, [f])
        try:
            e = solve_support(problem, trial)
        except Exception:  # noqa: BLE001
            continue
        best.append((eval_now.dP_op - e.dP_op, f))
    best.sort(key=lambda x: -x[0])
    added = [f for lev, f in best[:cfg.repair.projector_add_batch_size] if lev > 0]
    if not added:
        raise ProjectorCriterionUnreachable("no family improves projector distance")
    return _add_families(problem, support, added), added


def _add_families(problem: SelectionProblem, support: SelectedSupport,
                  families: Sequence[SpatialOccupationKey]) -> SelectedSupport:
    new_families = tuple(sorted(set(support.families) | set(families)))
    return materialize_support(
        new_families, [problem.sector],
        problem.cfg.selected_solver.max_support_determinants,
        protected=support.protected, generation=support.generation + 1,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _passes(problem: SelectionProblem, ev: Evaluated) -> list[str]:
    a = problem.cfg.acceptance
    failed = []
    if a.spin_is_mandatory and ev.spin_error > problem.cfg.spin_completion.spin_square_abs_tol:
        failed.append("spin")
    if ev.dP_op > a.tau_projector:
        failed.append("projector")
    if ev.gamma_error > a.tau_gamma:
        failed.append("matrix_1rdm")
    if ev.gamma_block_max > a.tau_gamma_block_max:
        failed.append("matrix_1rdm_block")
    if a.static_leakage_is_mandatory and ev.mandatory_families:
        failed.append("static_leakage")
    if a.compactness_is_mandatory and len(ev.support.families) > problem.cfg.selected_solver.max_support_families:
        failed.append("compactness")
    return failed


def build_pool(problem: SelectionProblem) -> dict[SpatialOccupationKey, FamilyStats]:
    """Build the family pool from the (single-variant) teacher states."""
    q, by_state = variant_family_stats(
        problem.teacher_states, problem.state_weights, problem.norb
    )
    return aggregate_family_pool(
        [q], [by_state], [1.0], problem.cfg.support_seed.q_persistence_threshold
    )


def selection_driver(
    problem: SelectionProblem,
    protected: Sequence[SpatialOccupationKey] = (),
    initial_families: Optional[Sequence[SpatialOccupationKey]] = None,
):
    """Run the selection loop and return ``(Evaluated, history, report)``.

    ``initial_families`` (if given) is used as the starting support instead of
    the pool seed, enabling checkpoint/resume.
    """
    cfg = problem.cfg
    pool = build_pool(problem)

    if initial_families is not None:
        families = tuple(sorted(set(initial_families)))
    else:
        families = seed_support(pool, protected, cfg)
    support = materialize_support(
        families, [problem.sector], cfg.selected_solver.max_support_determinants,
        protected=frozenset(protected),
    )

    history: list[dict] = []

    for iteration in range(cfg.selected_solver.max_selection_iterations):
        ev = solve_support(problem, support)
        action = DriverAction.SOLVE.value
        failed = _passes(problem, ev)

        history.append({
            "iteration": iteration,
            "action": action,
            "n_families": len(support.families),
            "n_determinants": len(ev.support_keys),
            "dP_op": ev.dP_op,
            "gamma_error": ev.gamma_error,
            "spin_error": ev.spin_error,
            "n_mandatory": len(ev.mandatory_families),
        })

        if ev.mandatory_families:
            support = _add_families(problem, support, ev.mandatory_families)
            history[-1]["action"] = DriverAction.PROMOTE_STATIC.value
            continue

        if "matrix_1rdm" in failed or "matrix_1rdm_block" in failed:
            support, _ = density_repair(problem, support, ev, pool, cfg)
            history[-1]["action"] = DriverAction.REPAIR_DENSITY.value
            continue

        if "projector" in failed:
            support, _ = projector_repair(problem, support, ev, pool, cfg)
            history[-1]["action"] = DriverAction.REPAIR_PROJECTOR.value
            continue

        if "spin" in failed:
            # spin failure on a spin-complete family support indicates a bug;
            # promote the remaining families to restore closure is not meaningful,
            # so surface it loudly.
            raise SelectionNonConvergence(
                "spin purity not restored on family-complete support",
                {"spin_error": ev.spin_error},
            )

        if "compactness" in failed:
            raise ReferenceNotCompact(
                "support exceeds family guardrail",
                {"n_families": len(support.families)},
            )

        # all mandatory criteria pass -> prune and certify
        support = _prune(problem, support, ev, pool)
        final = solve_support(problem, support)
        final_failed = _passes(problem, final)
        if not final_failed:
            history.append({
                "iteration": iteration,
                "action": DriverAction.CERTIFY.value,
                "n_families": len(support.families),
                "dP_op": final.dP_op,
                "gamma_error": final.gamma_error,
                "n_mandatory": len(final.mandatory_families),
            })
            report = build_report(problem, final)
            return final, history, report

        # prune-induced failure should not happen; fall through to iterate
        support = final.support

    raise SelectionNonConvergence("selection did not converge within iteration budget")


def _prune(problem: SelectionProblem, support: SelectedSupport, ev: Evaluated, pool):
    """Backward pruning with exact recheck (single family at a time)."""
    if not problem.cfg.pruning.enabled:
        return support
    from .pruning import removal_priority

    current = support
    for f in removal_priority(current, pool):
        trial = _add_families_removing(problem, current, f)
        if trial is None:
            continue
        try:
            e = solve_support(problem, trial)
        except Exception:  # noqa: BLE001 - skip invalid trials
            continue
        if _passes_with_safety(problem, e):
            current = trial
    return current


def _passes_with_safety(problem: SelectionProblem, ev: Evaluated) -> bool:
    if not _passes(problem, ev):
        return True
    # safety margin: all criteria must be within safety_factor of threshold
    a = problem.cfg.acceptance
    sf = problem.cfg.pruning.safety_factor
    return (
        ev.dP_op < sf * a.tau_projector
        and ev.gamma_error < sf * a.tau_gamma
        and ev.gamma_block_max < sf * a.tau_gamma_block_max
        and ev.spin_error < sf * problem.cfg.spin_completion.spin_square_abs_tol
        and not ev.mandatory_families
    )


def _add_families_removing(problem, support, f):
    new_families = tuple(x for x in support.families if x != f)
    if not new_families:
        return None  # an empty support is never valid
    try:
        return materialize_support(
            new_families, [problem.sector],
            problem.cfg.selected_solver.max_support_determinants,
            protected=support.protected, generation=support.generation + 1,
        )
    except ReferenceNotCompact:
        return None


def build_report(problem: SelectionProblem, ev: Evaluated) -> StabilityReport:
    """Produce a machine-readable stability report."""
    failed = _passes(problem, ev)
    return StabilityReport(
        passed=not failed,
        failed_criteria=failed,
        max_projector_distance=ev.dP_op,
        max_gamma_error=ev.gamma_error,
        max_gamma_block_error=ev.gamma_block_max,
        max_external_amplitude=max((s.static_score_max for s in ev.external_scores), default=0.0),
        max_external_family_rms=max((s.static_score_rms for s in ev.external_scores), default=0.0),
        max_spin_square_error=ev.spin_error,
        n_families=len(ev.support.families),
        n_determinants_by_sector={problem.sector: len(ev.support_keys)},
        thresholds_resolved=problem.cfg.resolved_thresholds(),
    )


__all__ = [
    "DriverAction",
    "SelectionProblem",
    "Evaluated",
    "solve_support",
    "scan_static_leakage",
    "build_pool",
    "selection_driver",
    "build_report",
    "EnsembleProblem",
    "VariantEvaluation",
    "align_variant_states",
    "build_ensemble_pool",
    "solve_variant_support",
    "ensemble_driver",
    "build_stability_report",
]


# ---------------------------------------------------------------------------
# M4 -- active-space ensemble (T050-T055)
# ---------------------------------------------------------------------------

@dataclass
class EnsembleProblem:
    """Fixed-orbital active-space ensemble problem."""

    parent: object  # ParentOrbitalSpace
    variants: list  # list[ActiveSpaceVariant]
    teachers: dict  # variant_id -> VariantTeacher
    state_weights: np.ndarray
    h1e_parent: np.ndarray
    eri_parent: np.ndarray
    cfg: MSRefConfig

    @property
    def nominal_variant(self):
        return self.variants[0]

    @property
    def nparent(self) -> int:
        return len(self.parent.comparison_orbitals)

    @property
    def nstate(self) -> int:
        return len(self.teachers[self.nominal_variant.variant_id].state_ids)


@dataclass
class VariantEvaluation:
    """Common-support evaluation for one active-space variant."""

    variant_id: str
    energies: np.ndarray  # total (incl. core), matched to nominal order
    gamma_error: float
    gamma_block_max: float
    dP_op: float
    spin_error: float
    mandatory_families: tuple  # parent SpatialOccupationKey
    n_determinants: int
    support_families: tuple


def _spin_constraint_matrix(states_a, states_b):
    """Forbid matching states of different total spin."""
    import numpy as _np

    c = []
    for a in states_a:
        row = []
        for b in states_b:
            row.append(a.state_id.spin_S != b.state_id.spin_S)
        c.append(row)
    return _np.asarray(c, dtype=bool)


def align_variant_states(problem: EnsembleProblem) -> dict:
    """Align every variant's roots to the nominal variant (T052).

    Returns ``{variant_id: perm}`` where ``perm[v]`` is the nominal state index
    matched to variant state ``v``.
    """
    from .state_tracking import hungarian_assignment, overlap_matrix

    nominal = problem.teachers[problem.nominal_variant.variant_id]
    aligns = {problem.nominal_variant.variant_id: list(range(len(nominal.state_ids)))}
    for variant in problem.variants[1:]:
        teacher = problem.teachers[variant.variant_id]
        o = overlap_matrix(teacher.ci_vectors, nominal.ci_vectors)  # [n_v, n_n]
        hard = [(i, j) for i in range(o.shape[0]) for j in range(o.shape[1])
                if teacher.state_ids[i].spin_S != nominal.state_ids[j].spin_S]
        perm, ambiguous = hungarian_assignment(o, hard_constraints=hard)
        if ambiguous:
            raise StateTrackingError(
                "cross-variant state alignment ambiguous",
                {"variant": variant.variant_id, "ambiguous": ambiguous},
            )
        for i, j in enumerate(perm):
            if j < 0 or abs(o[i, j]) < problem.cfg.state_tracking.state_match_min_overlap:
                raise StateTrackingError(
                    "cross-variant state alignment unresolved",
                    {"variant": variant.variant_id, "row": i},
                )
        aligns[variant.variant_id] = perm
    return aligns


def build_ensemble_pool(problem: EnsembleProblem) -> dict:
    """Aggregate family statistics over states and active-space variants (T053)."""
    from .config_pool import aggregate_family_pool, variant_family_stats

    norb = problem.nparent
    v_q = []
    v_by_state = []
    v_weights = [v.weight for v in problem.variants]
    total = sum(v_weights)
    v_weights = [w / total for w in v_weights]
    for variant in problem.variants:
        teacher = problem.teachers[variant.variant_id]
        q, by_state = variant_family_stats(teacher.ci_vectors, problem.state_weights, norb)
        v_q.append(q)
        v_by_state.append(by_state)
    return aggregate_family_pool(
        v_q, v_by_state, v_weights, problem.cfg.support_seed.q_persistence_threshold
    )


def _active_keys_of(problem, variant, families):
    from .config_pool import active_occupancy_of
    from .spin_completion import enumerate_family as _enf

    na_act, nb_act = problem.teachers[variant.variant_id].nelec_act
    keys = set()
    for f in families:
        occ = active_occupancy_of(f, variant)
        if occ is None:
            continue
        keys.update(_enf(SpatialOccupationKey(occ), na_act, nb_act))
    return tuple(sorted(keys))


def solve_variant_support(
    problem: EnsembleProblem, variant, families, cfg: MSRefConfig
) -> Optional[VariantEvaluation]:
    """Solve a common family support under one variant's Hamiltonian."""
    from pyscf import fci as _fci
    from .pyscf_adapter import sparse_ci_to_dense

    teacher = problem.teachers[variant.variant_id]
    na_act, nb_act = teacher.nelec_act
    active_keys = _active_keys_of(problem, variant, families)
    if not active_keys:
        return None
    nactive = len(teacher.active_orbitals)

    ham = build_selected_hamiltonian(
        active_keys, teacher.h_act, teacher.eri_act, max_memory_mb=cfg.runtime.max_memory_mb
    )
    nroots = len(teacher.state_ids)
    evals, evecs = solve_selected_sector(
        ham, nroots, cfg.state_tracking.root_buffer, cfg.selected_solver.eig_tol_eh
    )

    # embed selected active states to the parent gauge for matching + metrics
    parent_keys = tuple(
        embed_variant_det_for(problem.parent, variant, k.alpha, k.beta) for k in active_keys
    )
    teacher_dicts = [
        {kk: cc for kk, cc in zip(v.determinant_keys, v.coefficients)}
        for v in teacher.ci_vectors
    ]
    try:
        perm = match_roots_by_overlap(
            evecs, parent_keys, teacher_dicts, cfg.state_tracking.state_match_min_overlap
        )
    except Exception:  # noqa: BLE001 - root not found => support too small
        return None
    evecs = phase_fix_to_reference(evecs, parent_keys, teacher_dicts, perm)
    coeffs = evecs[:, perm]
    energies = evals[perm] + teacher.e_core

    # union determinant basis for projector comparison
    union = sorted(set(parent_keys) | {kk for v in teacher.ci_vectors for kk in v.determinant_keys})
    y_ref = _dense_over(teacher.ci_vectors, union)
    y_test = _dense_over_basis(coeffs, parent_keys, union)
    d_op, _d_fro, _sigma = projector_distance(y_ref, y_test)

    # selected matrix 1-RDM in parent basis
    nstate = len(teacher.state_ids)
    nparent = problem.nparent
    gamma_sel = np.zeros((nstate, nstate, nparent, nparent), dtype=complex)
    for a in range(nstate):
        for b in range(nstate):
            gamma_sel[a, b] = trans_rdm1_selected(coeffs[:, a], coeffs[:, b], parent_keys, nparent)

    nelec_total = na_act + nb_act + 2 * len(variant.inactive_orbitals)
    allowed = np.ones((nstate, nstate), dtype=bool)
    gamma_err, gamma_block, _ = matrix_rdm_error(
        gamma_sel, teacher.gamma1_parent, problem.state_weights, nelec_total,
        allowed=allowed, mode=cfg.acceptance.gamma_block_weighting,
    )

    # spin via active-space dense expansion
    spin = np.zeros(nstate)
    for s in range(nstate):
        act_keys = [_active_of(problem, variant, kk) for kk in parent_keys]
        dense_act = sparse_ci_to_dense(act_keys, coeffs[:, s], nactive, (na_act, nb_act))
        spin[s] = float(_fci.spin_square(dense_act, nactive, (na_act, nb_act))[0])
    spin_err = max(
        spin_square_error(float(spin[s]), teacher.state_ids[s].spin_S) for s in range(nstate)
    )

    return VariantEvaluation(
        variant_id=variant.variant_id,
        energies=energies,
        gamma_error=gamma_err,
        gamma_block_max=gamma_block,
        dP_op=d_op,
        spin_error=spin_err,
        mandatory_families=(),
        n_determinants=len(active_keys),
        support_families=tuple(f.occ for f in families),
    )


def embed_variant_det_for(parent, variant, alpha, beta):
    from .config_pool import embed_variant_det as _evd

    return _evd(parent, variant, alpha, beta)


def _active_of(problem, variant, parent_key):
    active = variant.active_orbitals
    a = 0
    b = 0
    for aidx, orb in enumerate(active):
        if (parent_key.alpha >> orb) & 1:
            a |= 1 << aidx
        if (parent_key.beta >> orb) & 1:
            b |= 1 << aidx
    return DeterminantKey(a, b)


def _dense_over(states, basis_keys):
    index = {k: i for i, k in enumerate(basis_keys)}
    y = np.zeros((len(basis_keys), len(states)), dtype=complex)
    for s, v in enumerate(states):
        for k, c in zip(v.determinant_keys, v.coefficients):
            i = index.get(k)
            if i is not None:
                y[i, s] = c
    return y


def _dense_over_basis(coeffs, parent_keys, union):
    index = {k: i for i, k in enumerate(union)}
    y = np.zeros((len(union), coeffs.shape[1]), dtype=complex)
    for j, k in enumerate(parent_keys):
        y[index[k], :] = coeffs[j, :]
    return y


def _dominant_families(problem, variant, families):
    """Return the top family (per state) of a variant not already in the support."""
    from .config_pool import family_weights_for_state

    norb = problem.nparent
    teacher = problem.teachers[variant.variant_id]
    selected = set(families)
    out = set()
    for s, v in enumerate(teacher.ci_vectors):
        fw = family_weights_for_state(v, norb)
        for f, _w in sorted(fw.items(), key=lambda kv: -kv[1]):
            if f not in selected:
                out.add(f)
                break
    return out


def ensemble_driver(
    problem: EnsembleProblem,
    protected=(),
):
    """Optimize one common family support passing all active-space variants (T054)."""
    cfg = problem.cfg
    pool = build_ensemble_pool(problem)

    families = seed_support(pool, protected, cfg)
    supports = [families]

    best = None
    for iteration in range(cfg.selected_solver.max_selection_iterations):
        families = supports[-1]
        evals = []
        missing = set()
        for variant in problem.variants:
            ev = solve_variant_support(problem, variant, families, cfg)
            if ev is not None:
                evals.append(ev)
            else:
                # support too small for this variant -> add its dominant families
                missing.update(_dominant_families(problem, variant, families))

        if missing:
            families = tuple(sorted(set(families) | missing))
            supports.append(families)
            continue

        if len(evals) != len(problem.variants):
            raise SelectionNonConvergence(
                "common support cannot represent all active-space variants",
                {"solved": len(evals), "n_variants": len(problem.variants)},
            )

        # union of mandatory static promotions across variants (worst-case)
        mandatory = set()
        for ev in evals:
            mandatory.update(ev.mandatory_families)

        worst_gamma = max((e.gamma_error for e in evals), default=0.0)
        worst_dP = max((e.dP_op for e in evals), default=0.0)
        worst_spin = max((e.spin_error for e in evals), default=0.0)

        if mandatory:
            families = tuple(sorted(set(families) | set(mandatory)))
            supports.append(families)
            continue

        # try density/projector repair using worst-case candidates
        repaired = False
        if worst_gamma > cfg.acceptance.tau_gamma:
            new_fam = _ensemble_repair(problem, families, pool, evals, mode="density")
            if new_fam is not None and new_fam != families:
                supports.append(new_fam)
                repaired = True
        if not repaired and worst_dP > cfg.acceptance.tau_projector:
            new_fam = _ensemble_repair(problem, families, pool, evals, mode="projector")
            if new_fam is not None and new_fam != families:
                supports.append(new_fam)
                repaired = True
        if repaired:
            continue

        # prune only if all variants retain safety margin
        pruned = _ensemble_prune(problem, families, pool, cfg)
        if pruned != families:
            supports.append(pruned)
            continue

        best = evals
        break

    if best is None:
        raise SelectionNonConvergence("ensemble common support did not converge")

    report = build_stability_report(problem, best, families, pool)
    return families, best, report


def _ensemble_repair(problem, families, pool, evals, mode):
    """Trial-add families improving the worst-case metric."""
    candidates = [f for f in pool if f not in set(families)]
    candidates.sort(key=lambda f: (-pool[f].qbar, -pool[f].max_state_weight, f.occ))
    candidates = candidates[: problem.cfg.repair.density_candidate_pool]
    best_improve = 0.0
    best_fam = None
    for f in candidates:
        trial = tuple(sorted(set(families) | {f}))
        worst = _worst_metric(problem, trial, mode)
        if best_fam is None or worst < best_improve:
            best_improve = worst
            best_fam = f
    if best_fam is None:
        return None
    return tuple(sorted(set(families) | {best_fam}))


def _worst_metric(problem, families, mode):
    worst = 0.0
    for variant in problem.variants:
        ev = solve_variant_support(problem, variant, families, problem.cfg)
        if ev is None:
            return 1e9
        val = ev.gamma_error if mode == "density" else ev.dP_op
        worst = max(worst, val)
    return worst


def _ensemble_prune(problem, families, pool, cfg):
    current = tuple(families)
    for f in sorted(current):
        trial = tuple(x for x in current if x != f)
        if not trial:
            continue
        ok = True
        for variant in problem.variants:
            try:
                ev = solve_variant_support(problem, variant, trial, cfg)
            except Exception:  # noqa: BLE001 - root not found -> family essential
                ok = False
                break
            if ev is None:
                ok = False
                break
            if not _ensemble_variant_passes(problem, ev, cfg):
                ok = False
                break
        if ok:
            current = trial
    return current


def _ensemble_variant_passes(problem, ev, cfg):
    a = cfg.acceptance
    sf = cfg.pruning.safety_factor
    return (
        ev.dP_op < sf * a.tau_projector
        and ev.gamma_error < sf * a.tau_gamma
        and ev.gamma_block_max < sf * a.tau_gamma_block_max
        and ev.spin_error < sf * cfg.spin_completion.spin_square_abs_tol
    )


def build_stability_report(
    problem: EnsembleProblem, evals: list, families, pool
) -> StabilityReport:
    """Produce the machine-readable ensemble stability report (T055)."""
    cfg = problem.cfg
    worst = {
        "dP": max((e.dP_op for e in evals), default=0.0),
        "gamma": max((e.gamma_error for e in evals), default=0.0),
        "gamma_block": max((e.gamma_block_max for e in evals), default=0.0),
        "spin": max((e.spin_error for e in evals), default=0.0),
    }
    # state-averaged energy oscillation across variants (not the raw root span)
    sa_energies = [
        float(np.dot(problem.state_weights, e.energies)) for e in evals
    ]
    energy_span = (max(sa_energies) - min(sa_energies)) if sa_energies else 0.0
    failed = []
    a = cfg.acceptance
    if worst["dP"] > a.tau_projector:
        failed.append("projector")
    if worst["gamma"] > a.tau_gamma:
        failed.append("matrix_1rdm")
    if worst["spin"] > cfg.spin_completion.spin_square_abs_tol:
        failed.append("spin")
    if energy_span > a.tau_E_SA_eh and a.energy_is_mandatory:
        failed.append("sa_energy")
    if len(families) > cfg.selected_solver.max_support_families:
        failed.append("compactness")

    return StabilityReport(
        passed=not failed,
        failed_criteria=failed,
        max_projector_distance=worst["dP"],
        max_gamma_error=worst["gamma"],
        max_gamma_block_error=worst["gamma_block"],
        sa_energy_span_eh=energy_span,
        max_spin_square_error=worst["spin"],
        n_families=len(families),
        n_determinants_by_sector={},
        active_space_variant_results=[
            {
                "variant_id": e.variant_id,
                "energies": [float(x) for x in e.energies],
                "sa_energy": float(np.dot(problem.state_weights, e.energies)),
                "gamma_error": e.gamma_error,
                "dP_op": e.dP_op,
                "spin_error": e.spin_error,
                "n_determinants": e.n_determinants,
            }
            for e in evals
        ],
        thresholds_resolved=cfg.resolved_thresholds(),
    )
