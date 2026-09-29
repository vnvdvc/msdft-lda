"""PySCF adapter: all direct PySCF interaction lives in this module.

Conventions enforced here (see ``DATA_CONTRACTS.yaml``):

* orbital indices are zero-based;
* energies in Hartree;
* CI determinant keys use parent-orbital bit positions ``(alpha, beta)`` with
  bit ``p`` corresponding to orbital ``p`` (matches PySCF ``cistring``);
* matrix 1-RDM follows PySCF: ``gamma[p,q] = <q^dagger p>``;
* cross-sector (different spin / particle-number) transition blocks are zero
  with an explicit reason code.

Spin-mixed state averaging is realized with a *common* ``M_S = 0`` determinant
representation across blocks: every block uses ``direct_spin1`` and spin is
targeted with ``fix_spin_(ss=S(S+1))``, which adds a spin penalty without
changing the ``(nalpha, nbeta)`` sector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from pyscf import fci, mcscf

from . import models
from .config import MSRefConfig
from .config_pool import embed_variant_det
from .exceptions import (
    InputValidationError,
    SpinContaminationError,
    TeacherConvergenceError,
)
from .models import DeterminantKey, SectorKey, SparseStateVector, StateID, TeacherResult


@dataclass
class SectorMetadata:
    """Normalized spin/symmetry/particle-number metadata for one state block."""

    block_id: str
    spin_S: float
    ms2: int
    wfnsym: Optional[object]
    nalpha: int
    nbeta: int

    @property
    def sector_key(self) -> SectorKey:
        return (self.nalpha, self.nbeta, self.wfnsym)


# ---------------------------------------------------------------------------
# Determinant-string basis helpers (public PySCF ``cistring`` interface)
# ---------------------------------------------------------------------------

def determinant_string_basis(norb: int, nelec: tuple[int, int]):
    """Return (alpha_bitstrings, beta_bitstrings) in PySCF FCI address order."""
    from pyscf.fci import cistring

    na, nb = int(nelec[0]), int(nelec[1])
    n_a = cistring.num_strings(norb, na)
    n_b = cistring.num_strings(norb, nb)
    addrs_a = np.arange(n_a, dtype=np.int32)
    addrs_b = np.arange(n_b, dtype=np.int32)
    str_a = cistring.addrs2str(norb, na, addrs_a)
    str_b = cistring.addrs2str(norb, nb, addrs_b)
    return str_a, str_b


def full_ci_vector_to_sparse(
    fcivec: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
    screen: float = 0.0,
):
    """Convert a dense PySCF FCI vector to determinant-key sparse form.

    Returns ``(keys, coefficients, captured_norm_fraction)``.
    """
    fcivec = np.asarray(fcivec)
    str_a, str_b = determinant_string_basis(norb, nelec)
    n_a, n_b = len(str_a), len(str_b)
    if fcivec.ndim == 1:
        fcivec = fcivec.reshape(n_a, n_b)
    mask = np.abs(fcivec) > screen
    total = float(np.sum(np.abs(fcivec) ** 2))
    captured = float(np.sum(np.abs(fcivec[mask]) ** 2)) if np.any(mask) else 0.0
    keys = [
        DeterminantKey(int(str_a[i]), int(str_b[j]))
        for i, j in zip(*np.nonzero(mask))
    ]
    coeffs = fcivec[mask].astype(complex)
    order = np.argsort(keys)
    keys = [keys[k] for k in order]
    coeffs = coeffs[order]
    fraction = captured / total if total > 0 else 0.0
    return tuple(keys), coeffs, fraction


def sparse_ci_to_dense(
    keys: Sequence[DeterminantKey],
    coeffs: np.ndarray,
    norb: int,
    nelec: tuple[int, int],
) -> np.ndarray:
    """Inverse of :func:`full_ci_vector_to_sparse` (PySCF FCI array layout)."""
    from pyscf.fci import cistring

    na, nb = int(nelec[0]), int(nelec[1])
    n_a = cistring.num_strings(norb, na)
    n_b = cistring.num_strings(norb, nb)
    dense = np.zeros((n_a, n_b), dtype=complex)
    for key, c in zip(keys, coeffs):
        ia = cistring.str2addr(norb, na, int(key.alpha))
        ib = cistring.str2addr(norb, nb, int(key.beta))
        dense[ia, ib] = c
    # PySCF FCI kernels (make_rdm1/spin_square) assume real CI vectors; return a
    # real array whenever the coefficients are real so those kernels stay exact.
    if np.allclose(dense.imag, 0.0):
        dense = dense.real
    return dense


# ---------------------------------------------------------------------------
# T010 -- state-block solver factory
# ---------------------------------------------------------------------------

def make_solver_for_block(
    mol, block: models.StateBlockSpec, cfg: MSRefConfig
) -> tuple[object, SectorMetadata]:
    """Construct a ``direct_spin1`` FCI solver targeting the block spin.

    Returns ``(solver, sector_metadata)``.
    """
    solver = fci.direct_spin1.FCISolver(mol)
    solver.nroots = block.nroots
    if block.wfnsym is not None:
        solver.wfnsym = block.wfnsym

    # Target spin via a spin penalty in the common M_S=0 determinant sector.
    ss_target = block.spin_S * (block.spin_S + 1.0)
    if block.spin_S > 0 or block.ms2 == 0:
        fci.addons.fix_spin_(solver, shift=cfg.teacher.spin_penalty_shift_eh, ss=ss_target)

    # nelec from ms2 = 2*M_S = nalpha - nbeta (total active electrons = nalpha+nbeta).
    nalpha = block.nelec_total[0] if block.nelec_total is not None else None
    nbeta = block.nelec_total[1] if block.nelec_total is not None else None
    meta = SectorMetadata(
        block_id=block.block_id,
        spin_S=block.spin_S,
        ms2=block.ms2,
        wfnsym=block.wfnsym,
        nalpha=nalpha,
        nbeta=nbeta,
    )
    return solver, meta


def _sector_key_of(meta: SectorMetadata) -> SectorKey:
    return (meta.nalpha, meta.nbeta, meta.wfnsym)


# ---------------------------------------------------------------------------
# T011 -- spin-mixed SA-CASSCF teacher builder
# ---------------------------------------------------------------------------

def build_teacher(
    mf,
    ncas: int,
    nelecas: int,
    state_blocks: Sequence[models.StateBlockSpec],
    weights: Sequence[float],
    mo0: Optional[np.ndarray] = None,
    cfg: Optional[MSRefConfig] = None,
) -> TeacherResult:
    """Build and run a spin-mixed SA-CASSCF teacher, returning a TeacherResult."""
    cfg = cfg or MSRefConfig()

    # Validate weights across the flattened solver/root order.
    flat_weights = list(weights)
    nroots = sum(b.nroots for b in state_blocks)
    if len(flat_weights) != nroots:
        raise InputValidationError(
            f"expected {nroots} weights, got {len(flat_weights)}"
        )
    if abs(sum(flat_weights) - 1.0) > cfg.teacher.state_weight_sum_tol:
        raise InputValidationError("state weights must sum to one")

    mc = mcscf.CASSCF(mf, ncas, nelecas)
    solvers = []
    metas = []
    for block in state_blocks:
        solver, meta = make_solver_for_block(mf.mol, block, cfg)
        solvers.append(solver)
        metas.append(meta)

    mcscf.state_average_mix_(mc, solvers, flat_weights)

    mc.conv_tol = cfg.teacher.casscf_energy_tol_eh
    mc.conv_tol_grad = cfg.teacher.casscf_grad_tol
    mc.max_cycle_macro = cfg.teacher.max_macro_cycles
    if cfg.teacher.max_micro_cycles is not None:
        mc.max_cycle_micro = cfg.teacher.max_micro_cycles

    mc.kernel(mo0)

    if cfg.teacher.require_converged and not mc.converged:
        raise TeacherConvergenceError(
            "teacher SA-CASSCF did not converge", {"e_tot": getattr(mc, "e_tot", None)}
        )

    return extract_teacher(mc, ncas, nelecas, state_blocks, metas, flat_weights, cfg)


def _nelec_for_ms(nelecas: int, ms2: int) -> tuple[int, int]:
    nalpha = (nelecas + ms2) // 2
    nbeta = (nelecas - ms2) // 2
    if nalpha + nbeta != nelecas:
        raise InputValidationError(f"nelecas={nelecas} incompatible with ms2={ms2}")
    return nalpha, nbeta


def extract_teacher(
    mc,
    ncas: int,
    nelecas: int,
    state_blocks: Sequence[models.StateBlockSpec],
    metas: Sequence[SectorMetadata],
    flat_weights: Sequence[float],
    cfg: MSRefConfig,
) -> TeacherResult:
    """Extract a normalized TeacherResult from a converged SA-CASSCF object."""
    ci_list = list(mc.ci)
    energies = np.asarray(list(mc.e_states), dtype=float)
    nstate = len(ci_list)

    state_ids: list[StateID] = []
    ci_vectors: list[SparseStateVector] = []
    spin_square: list[float] = []

    # sector (nalpha,nbeta) per state, from the block ms2
    state_sectors: list[tuple[int, int]] = []
    state_wfnsym: list[Optional[object]] = []
    root_ordinal = 0
    for block, meta in zip(state_blocks, metas):
        na, nb = _nelec_for_ms(nelecas, block.ms2)
        for r in range(block.nroots):
            state_sectors.append((na, nb))
            state_wfnsym.append(block.wfnsym)
            state_ids.append(
                StateID(
                    name=f"{block.block_id}_root{r}",
                    block_id=block.block_id,
                    spin_S=block.spin_S,
                    ms2=block.ms2,
                    wfnsym=block.wfnsym,
                    root_ordinal=root_ordinal,
                )
            )
            root_ordinal += 1

    for i, ci in enumerate(ci_list):
        na, nb = state_sectors[i]
        keys, coeffs, fraction = full_ci_vector_to_sparse(
            ci, ncas, (na, nb), screen=cfg.teacher_ci_storage.coefficient_io_screen
        )
        if fraction < cfg.teacher_ci_storage.minimum_captured_norm_per_state:
            raise InputValidationError(
                "sparse teacher storage dropped too much norm",
                {"state": i, "captured_norm": fraction},
            )
        ci_vectors.append(
            SparseStateVector(
                state_id=state_ids[i],
                sector_key=(na, nb, state_wfnsym[i]),
                determinant_keys=keys,
                coefficients=coeffs,
            )
        )
        ss, _multip = fci.spin_square(ci, ncas, (na, nb))
        spin_square.append(float(ss))
        _validate_spin(state_ids[i], ss, cfg)

    gamma1 = _build_teacher_gamma(ci_list, ncas, state_sectors, state_wfnsym)

    return TeacherResult(
        variant_id="nominal",
        parent_id="active",
        mo_coeff_ao=np.asarray(mc.mo_coeff),
        energies_eh=energies,
        state_ids=state_ids,
        state_weights=np.asarray(flat_weights, dtype=float),
        spin_square=np.asarray(spin_square, dtype=float),
        ci_vectors=ci_vectors,
        gamma1_parent=gamma1,
        converged=bool(mc.converged),
        metadata={"e_tot": float(getattr(mc, "e_tot", np.nan))},
    )


def _validate_spin(state_id: StateID, ss: float, cfg: MSRefConfig) -> None:
    target = state_id.spin_S * (state_id.spin_S + 1.0)
    if abs(ss - target) > cfg.teacher.spin_square_abs_tol:
        raise SpinContaminationError(
            f"state {state_id.name} has <S^2>={ss:.6f}, expected {target:.6f}",
            {"state": state_id.name, "s2": ss, "target": target},
        )


def _build_teacher_gamma(
    ci_list: Sequence[np.ndarray],
    ncas: int,
    state_sectors: Sequence[tuple[int, int]],
    state_wfnsym: Sequence[Optional[object]],
) -> np.ndarray:
    """Build the full matrix 1-RDM [A,B,p,q] with forbidden blocks zeroed."""
    nstate = len(ci_list)
    gamma = np.zeros((nstate, nstate, ncas, ncas), dtype=complex)
    for a in range(nstate):
        na_a, nb_a = state_sectors[a]
        gamma[a, a] = fci.direct_spin1.make_rdm1(ci_list[a], ncas, (na_a, nb_a))
        for b in range(a):
            na_b, nb_b = state_sectors[b]
            same_sector = (na_a, nb_a) == (na_b, nb_b) and state_wfnsym[a] == state_wfnsym[b]
            if same_sector:
                g = fci.direct_spin1.trans_rdm1(ci_list[a], ci_list[b], ncas, (na_a, nb_a))
                gamma[a, b] = g
                gamma[b, a] = g.conj().T
            else:
                # spin/sector-forbidden block: zero (reason encoded in metadata)
                gamma[a, b] = 0.0
                gamma[b, a] = 0.0
    return gamma


# ---------------------------------------------------------------------------
# T015 -- RDM and spin-square extraction wrappers
# ---------------------------------------------------------------------------

def make_state_rdm1(ci: np.ndarray, norb: int, nelec: tuple[int, int]) -> np.ndarray:
    """State 1-RDM, ``gamma[p,q] = <q^dagger p>``."""
    return fci.direct_spin1.make_rdm1(ci, norb, nelec)


def trans_rdm1(ci_bra: np.ndarray, ci_ket: np.ndarray, norb: int,
               nelec: tuple[int, int]) -> np.ndarray:
    """Transition 1-RDM, ``gamma[p,q] = <bra|q^dagger p|ket>``."""
    return fci.direct_spin1.trans_rdm1(ci_bra, ci_ket, norb, nelec)


def spin_square_of(ci: np.ndarray, norb: int, nelec: tuple[int, int]) -> tuple[float, float]:
    """Return ``(<S^2>, 2S+1)`` for a CI vector."""
    return fci.spin_square(ci, norb, nelec)


@dataclass
class VariantTeacher:
    """Fixed-orbital CASCI result for one active-space variant, parent-embedded."""

    variant_id: str
    active_orbitals: tuple[int, ...]
    energies_eh: np.ndarray  # [nstate]
    state_ids: list
    ci_vectors: list  # list[SparseStateVector] embedded to parent gauge
    gamma1_parent: np.ndarray  # [nstate, nstate, nparent, nparent]
    spin_square: np.ndarray  # [nstate]
    e_core: float
    nelec_act: tuple[int, int] = (0, 0)
    h_act: Optional[np.ndarray] = None
    eri_act: Optional[np.ndarray] = None


def build_core_and_active(
    h1e: np.ndarray, eri: np.ndarray, inactive: Sequence[int], active: Sequence[int]
):
    """Return ``(e_core, h_act, eri_act)`` for a frozen-core active Hamiltonian.

    ``h1e``/``eri`` are in the parent MO basis; ``inactive`` orbitals are doubly
    occupied (frozen), ``active`` orbitals carry the CI problem.
    """
    inactive = [int(i) for i in inactive]
    active = [int(i) for i in active]
    h_act = np.array(h1e[np.ix_(active, active)], dtype=float)
    e_core = 0.0
    for i in inactive:
        e_core += 2.0 * float(h1e[i, i].real)
    for i in inactive:
        for j in inactive:
            # chemist notation (pq|rs); J=(ii|jj), K=(ij|ij)
            e_core += float((2.0 * eri[i, i, j, j] - eri[i, j, i, j]).real)
    for pi, p in enumerate(active):
        for qi, q in enumerate(active):
            for i in inactive:
                # core-active Fock contribution: 2(pq|ii) - (pi|iq)
                h_act[pi, qi] += 2.0 * eri[p, q, i, i] - eri[p, i, i, q]
    eri_act = np.array(eri[np.ix_(active, active, active, active)], dtype=float)
    return e_core, h_act, eri_act


def run_variant_casci(
    mf,
    parent: "models.ParentOrbitalSpace",
    variant: "models.ActiveSpaceVariant",
    state_blocks: Sequence["models.StateBlockSpec"],
    weights: Sequence[float],
    cfg: MSRefConfig,
) -> VariantTeacher:
    """Run fixed-orbital CASCI for one variant and embed CI vectors to the parent.

    All variants share the same parent MO gauge (``mf.mo_coeff``); only the
    active/inactive role partition changes.
    """
    from pyscf import ao2mo

    mol = mf.mol
    mo = mf.mo_coeff
    h1e = mo.T @ mf.get_hcore() @ mo
    eri = ao2mo.restore(1, ao2mo.incore.full(mol.intor("int2e"), mo), mo.shape[1])

    active = list(variant.active_orbitals)
    inactive = list(variant.inactive_orbitals)
    nactive = len(active)
    nmo = mo.shape[1]

    # active electron count (assume a single Ms=0 sector)
    na_cmp = sum(1 for r in variant.roles.values() if r in (models.Role.ACTIVE, models.Role.INACTIVE))
    ninactive = len(inactive)
    # derive from comparison electron count stored in nelecas_by_sector
    na_act = nb_act = None
    for (_na_cmp, _nb_cmp, _wfnsym), (_na, _nb) in variant.nelecas_by_sector.items():
        na_act, nb_act = _na, _nb
    if na_act is None:
        na_act = na_cmp - ninactive
        nb_act = na_cmp - ninactive

    e_core, h_act, eri_act = build_core_and_active(h1e, eri, inactive, active)

    # run FCI per state block with spin targeting
    energies = []
    spin_sq = []
    ci_dense = []
    state_ids = []
    state_sectors = []
    root_ordinal = 0
    for block in state_blocks:
        solver, _meta = make_solver_for_block(mol, block, cfg)
        e, c = solver.kernel(h_act, eri_act, nactive, (na_act, nb_act))
        if block.nroots == 1:
            e_list = [e]
            c_list = [c]
        else:
            e_list = list(e)
            c_list = list(c)
        for r in range(block.nroots):
            energies.append(e_list[r])
            ci_dense.append(c_list[r])
            ss, _ = fci.spin_square(c_list[r], nactive, (na_act, nb_act))
            spin_sq.append(float(ss))
            state_sectors.append((na_act, nb_act))
            state_ids.append(
                StateID(
                    name=f"{block.block_id}_root{r}",
                    block_id=block.block_id,
                    spin_S=block.spin_S,
                    ms2=block.ms2,
                    wfnsym=block.wfnsym,
                    root_ordinal=root_ordinal,
                )
            )
            root_ordinal += 1
        _meta  # unused

    # embed CI vectors to the parent determinant gauge
    embedded = []
    for s, c in enumerate(ci_dense):
        na_s, nb_s = state_sectors[s]
        keys_act, coeffs, frac = full_ci_vector_to_sparse(
            c, nactive, (na_s, nb_s), screen=cfg.teacher_ci_storage.coefficient_io_screen
        )
        parent_keys = tuple(
            embed_variant_det(parent, variant, int(k.alpha), int(k.beta)) for k in keys_act
        )
        embedded.append(
            SparseStateVector(
                state_id=state_ids[s],
                sector_key=(na_s, nb_s, None),
                determinant_keys=parent_keys,
                coefficients=coeffs,
            )
        )

    # parent matrix 1-RDM (core restored on diagonal blocks)
    nstate = len(embedded)
    gamma = np.zeros((nstate, nstate, nmo, nmo), dtype=complex)
    for a in range(nstate):
        for i in inactive:
            gamma[a, a, i, i] = 2.0
        for b in range(nstate):
            same_sector = state_sectors[a] == state_sectors[b]
            if same_sector:
                g = fci.direct_spin1.trans_rdm1(ci_dense[a], ci_dense[b], nactive, state_sectors[a])
            else:
                g = np.zeros((nactive, nactive))
            for pi, p in enumerate(active):
                for qi, q in enumerate(active):
                    gamma[a, b, p, q] = g[pi, qi]

    return VariantTeacher(
        variant_id=variant.variant_id,
        active_orbitals=tuple(active),
        energies_eh=np.asarray(energies, dtype=float) + e_core,
        state_ids=state_ids,
        ci_vectors=embedded,
        gamma1_parent=gamma,
        spin_square=np.asarray(spin_sq, dtype=float),
        e_core=e_core,
        nelec_act=(na_act, nb_act),
        h_act=h_act,
        eri_act=eri_act,
    )


__all__ = [
    "SectorMetadata",
    "VariantTeacher",
    "determinant_string_basis",
    "full_ci_vector_to_sparse",
    "sparse_ci_to_dense",
    "make_solver_for_block",
    "build_teacher",
    "extract_teacher",
    "build_core_and_active",
    "run_variant_casci",
    "make_state_rdm1",
    "trans_rdm1",
    "spin_square_of",
]
