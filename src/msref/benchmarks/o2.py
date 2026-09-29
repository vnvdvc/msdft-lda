"""O2 mixed-spin release benchmark (T059)."""

from __future__ import annotations

from pyscf import gto, scf

from .. import models
from ..config import MSRefConfig
from .common import build_ensemble_problem, ensemble_ci_analysis


def build_o2_ensemble(r=1.21, basis="cc-pvdz", cfg=None):
    cfg = cfg or MSRefConfig()
    cfg.active_space_ensemble.allow_exchange = False  # keep the pi* pair intact
    # ROHF triplet reference correctly identifies the pi* (pi_g) frontier orbitals
    mol = gto.M(atom=f"O 0 0 0; O 0 0 {r}", basis=basis, unit="angstrom",
                spin=2, verbose=0)
    mf = scf.ROHF(mol).run()
    # CAS(2,2) pi* (X3Sg-, a1Dg, b1Sg+); Ms=0 representation => 8 alpha, 8 beta
    ncore = (16 - 2) // 2  # 7 doubly occupied
    nmo = mf.mo_coeff.shape[1]
    # buffer = the next virtual orbitals above pi* (add-virtual variants)
    buffer = list(range(ncore + 2, min(nmo, ncore + 4)))
    block_t = models.StateBlockSpec(block_id="triplet", spin_S=1.0, ms2=0,
                                    nroots=1, weights=[1 / 3])
    block_s = models.StateBlockSpec(block_id="singlet", spin_S=0.0, ms2=0,
                                    nroots=2, weights=[1 / 3, 1 / 3])
    prob = build_ensemble_problem(mf, ncas=2, nelecas=2,
                                  state_blocks=[block_t, block_s],
                                  weights=[1 / 3, 1 / 3, 1 / 3], cfg=cfg,
                                  buffer=buffer, nelec_comparison=(8, 8))
    return mf, prob


def run_o2_ensemble(r=1.21, basis="cc-pvdz", cfg=None):
    from ..driver import ensemble_driver

    mf, prob = build_o2_ensemble(r=r, basis=basis, cfg=cfg)
    fam, evals, report = ensemble_driver(prob)
    return prob, fam, evals, report, ensemble_ci_analysis(prob)


__all__ = ["build_o2", "mixed_spin_teacher", "build_o2_ensemble", "run_o2_ensemble"]


def build_o2(r=1.2, basis="cc-pvdz"):
    mol = gto.M(atom=f"O 0 0 0; O 0 0 {r}", basis=basis, unit="angstrom",
                spin=0, verbose=0)
    mf = scf.RHF(mol).run()
    return mol, mf


def mixed_spin_teacher(mf, cfg=None):
    """Triplet X3Sg- plus two singlets (a1Dg, b1Sg+) over CAS(2,2) in pi*."""
    from ..pyscf_adapter import build_teacher

    cfg = cfg or MSRefConfig()
    block_t = models.StateBlockSpec(block_id="triplet", spin_S=1.0, ms2=0,
                                    nroots=1, weights=[1 / 3])
    block_s = models.StateBlockSpec(block_id="singlet", spin_S=0.0, ms2=0,
                                    nroots=2, weights=[1 / 3, 1 / 3])
    return build_teacher(mf, 2, 2, [block_t, block_s], [1 / 3, 1 / 3, 1 / 3],
                         mo0=mf.mo_coeff, cfg=cfg)
