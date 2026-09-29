"""H2 release benchmark: exact sanity + dissociation (T057)."""

from __future__ import annotations

import numpy as np
from pyscf import gto, scf

from .. import models
from ..config import MSRefConfig
from .common import build_ensemble_problem, ensemble_ci_analysis

H2_GEOMETRIES = [0.60, 0.74, 1.00, 1.50, 2.00, 3.00, 4.00]


def build_h2(r=0.74, basis="sto-3g"):
    mol = gto.M(atom=f"H 0 0 0; H 0 0 {r}", basis=basis, unit="angstrom", verbose=0)
    mf = scf.RHF(mol).run()
    return mol, mf


def singlet_teacher(mf, cfg=None):
    """CAS(2,2) two-singlet teacher smoke calculation (M1)."""
    from ..pyscf_adapter import build_teacher

    cfg = cfg or MSRefConfig()
    block = models.StateBlockSpec(block_id="singlet", spin_S=0.0, ms2=0,
                                  nroots=2, weights=[0.5, 0.5])
    return build_teacher(mf, 2, 2, [block], [0.5, 0.5], mo0=mf.mo_coeff, cfg=cfg)


def build_h2_ensemble(r=0.74, basis="cc-pvdz", cfg=None):
    """Build the H2 fixed-orbital active-space ensemble problem.

    Nominal CAS(2,2) sigma_g/sigma_u pair; buffer = the next sigma_g/sigma_u
    correlating orbitals.
    """
    cfg = cfg or MSRefConfig()
    mol = gto.M(atom=f"H 0 0 0; H 0 0 {r}", basis=basis, unit="angstrom", verbose=0)
    mf = scf.RHF(mol).run()
    # sigma buffer orbitals are the first few virtuals (2sg, 2su, 3sg, 3su)
    ncore = (mol.nelec[0] + mol.nelec[1] - 2) // 2
    buffer = list(range(ncore + 2, min(ncore + 6, mf.mo_coeff.shape[1])))
    block = models.StateBlockSpec(block_id="singlet", spin_S=0.0, ms2=0,
                                  nroots=2, weights=[0.5, 0.5])
    prob = build_ensemble_problem(mf, ncas=2, nelecas=2, state_blocks=[block],
                                  weights=[0.5, 0.5], cfg=cfg, buffer=buffer)
    return mf, prob


def run_h2_ensemble(r=0.74, basis="cc-pvdz", cfg=None):
    """Run the ensemble driver and return (prob, families, evals, report, analysis)."""
    from ..driver import ensemble_driver

    mf, prob = build_h2_ensemble(r=r, basis=basis, cfg=cfg)
    fam, evals, report = ensemble_driver(prob)
    analysis = ensemble_ci_analysis(prob)
    return prob, fam, evals, report, analysis


def h2_release_scan(cfg=None):
    """Run the H2 geometry grid; return a list of per-geometry result dicts."""
    cfg = cfg or MSRefConfig()
    out = []
    for r in H2_GEOMETRIES:
        prob, fam, evals, report, analysis = run_h2_ensemble(r=r, cfg=cfg)
        out.append(
            {
                "r": r,
                "n_variants": len(prob.variants),
                "n_families": len(fam),
                "passed": report.passed,
                "failed": report.failed_criteria,
                "max_dP": report.max_projector_distance,
                "max_gamma": report.max_gamma_error,
                "sa_energy_span": report.sa_energy_span_eh,
                "analysis": analysis,
            }
        )
    return out


__all__ = ["H2_GEOMETRIES", "build_h2_ensemble", "run_h2_ensemble", "h2_release_scan"]
