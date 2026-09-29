"""Shared helpers for the M4 active-space ensemble benchmarks."""

from __future__ import annotations

import numpy as np

from ..active_ensemble import generate_active_space_variants
from ..models import ParentOrbitalSpace
from ..pyscf_adapter import run_variant_casci


def setup_parent(mf) -> ParentOrbitalSpace:
    """Build the fixed parent orbital gauge (all MOs as comparison orbitals)."""
    nmo = mf.mo_coeff.shape[1]
    return ParentOrbitalSpace(
        parent_id="parent",
        mo_coeff_ao=np.asarray(mf.mo_coeff),
        ao_overlap=np.asarray(mf.get_ovlp()),
        orbital_ids=tuple(f"mo{i}" for i in range(nmo)),
        orbsym=tuple([None] * nmo),
        always_core=(),
        comparison_orbitals=tuple(range(nmo)),
        always_external=(),
    )


def nominal_partition(mf, ncas: int, nelecas: int, buffer: list | None = None):
    """Return ``(nominal_active, doubly_occupied, virtual)`` orbital index lists.

    Uses the PySCF convention: the first ``ncore`` MOs are frozen doubly
    occupied, the next ``ncas`` are the nominal active space, the rest virtual.
    ``buffer`` optionally restricts the virtual (buffer) orbitals to a
    chemically-matched subset.
    """
    nelec = mf.mol.nelec  # (nalpha, nbeta)
    ncore = (sum(nelec) - nelecas) // 2
    nominal_active = list(range(ncore, ncore + ncas))
    doubly_occupied = list(range(ncore))
    virtual = list(range(ncore + ncas, mf.mo_coeff.shape[1]))
    if buffer is not None:
        virtual = [b for b in buffer if b in virtual]
    return nominal_active, doubly_occupied, virtual


def build_ensemble_problem(
    mf,
    ncas: int,
    nelecas: int,
    state_blocks,
    weights,
    cfg,
    buffer: list | None = None,
    nelec_comparison: tuple | None = None,
):
    """Build the fixed-orbital ensemble problem for a molecule."""
    from ..driver import EnsembleProblem
    from pyscf import ao2mo

    mol = mf.mol
    parent = setup_parent(mf)
    nominal_active, d_occ, virt = nominal_partition(mf, ncas, nelecas, buffer=buffer)
    if nelec_comparison is None:
        nelec_comparison = mol.nelec
    variants = generate_active_space_variants(
        nominal_active, d_occ, virt, (nelec_comparison[0], nelec_comparison[1]), cfg
    )

    teachers = {}
    for v in variants:
        teachers[v.variant_id] = run_variant_casci(mf, parent, v, state_blocks, weights, cfg)

    mo = mf.mo_coeff
    h1e = mo.T @ mf.get_hcore() @ mo
    eri = ao2mo.restore(1, ao2mo.incore.full(mol.intor("int2e"), mo), mo.shape[1])

    return EnsembleProblem(
        parent=parent,
        variants=variants,
        teachers=teachers,
        state_weights=np.asarray(weights, dtype=float),
        h1e_parent=h1e,
        eri_parent=eri,
        cfg=cfg,
    )


__all__ = ["setup_parent", "nominal_partition", "build_ensemble_problem", "ensemble_ci_analysis"]


def ensemble_ci_analysis(prob):
    """Extract per-variant CI-vector / density data for the report.

    Returns a list of dicts (one per variant) with energies, state-averaged
    energy, dominant configuration-family weights, and the ground-state matrix
    1-RDM diagonal occupations.
    """
    from ..config_pool import family_weights_for_state
    from ..spin_completion import family_of

    norb = prob.nparent
    rows = []
    for variant in prob.variants:
        teacher = prob.teachers[variant.variant_id]
        # state-averaged family weights
        fam_weight = {}
        for s, v in enumerate(teacher.ci_vectors):
            fw = family_weights_for_state(v, norb)
            for f, w in fw.items():
                fam_weight[f] = fam_weight.get(f, 0.0) + prob.state_weights[s] * w
        top = sorted(fam_weight.items(), key=lambda kv: -kv[1])[:6]
        rdm_diag = np.real(np.diag(teacher.gamma1_parent[0, 0]))
        rows.append(
            {
                "variant_id": variant.variant_id,
                "active_orbitals": list(variant.active_orbitals),
                "nelec_act": list(teacher.nelec_act),
                "energies": [float(x) for x in teacher.energies_eh],
                "sa_energy": float(np.dot(prob.state_weights, teacher.energies_eh)),
                "spin_square": [float(x) for x in teacher.spin_square],
                "top_families": [
                    {"occ": list(f.occ), "weight": float(w)} for f, w in top
                ],
                "rdm_diag": [float(x) for x in rdm_diag],
            }
        )
    return rows
