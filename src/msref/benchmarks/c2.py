"""C2 release benchmark: dense multireference state manifold (T060)."""

from __future__ import annotations

from pyscf import gto, scf

from .. import models
from ..config import MSRefConfig
from .common import build_ensemble_problem, ensemble_ci_analysis


def build_c2_ensemble(r=1.24, basis="cc-pvdz", cfg=None):
    cfg = cfg or MSRefConfig()
    mol = gto.M(atom=f"C 0 0 0; C 0 0 {r}", basis=basis, unit="angstrom", verbose=0)
    mf = scf.RHF(mol).run()
    # CAS(6,6) valence p-block; low singlet manifold + buffer
    ncore = (mol.nelec[0] + mol.nelec[1] - 4) // 2
    nmo = mf.mo_coeff.shape[1]
    buffer = list(range(ncore + 4, min(nmo, ncore + 8)))
    block = models.StateBlockSpec(block_id="singlet", spin_S=0.0, ms2=0,
                                  nroots=3, weights=[1 / 3, 1 / 3, 1 / 3])
    prob = build_ensemble_problem(mf, ncas=4, nelecas=4, state_blocks=[block],
                                  weights=[1 / 3, 1 / 3, 1 / 3], cfg=cfg,
                                  buffer=buffer)
    return mf, prob


def run_c2_ensemble(r=1.24, basis="cc-pvdz", cfg=None):
    from ..driver import ensemble_driver

    mf, prob = build_c2_ensemble(r=r, basis=basis, cfg=cfg)
    fam, evals, report = ensemble_driver(prob)
    return prob, fam, evals, report, ensemble_ci_analysis(prob)


__all__ = ["build_c2_ensemble", "run_c2_ensemble"]
