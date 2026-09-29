"""Integration tests for the selection driver (T041)."""

from __future__ import annotations

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")

from msref import models
from msref.config import MSRefConfig
from msref.driver import SelectionProblem, build_pool, selection_driver
from .helpers import active_integrals, h2_mf


@pytest.mark.integration
def test_h2_selection_converges():
    from msref.pyscf_adapter import build_teacher

    mol, mf = h2_mf()
    cfg = MSRefConfig()
    block = models.StateBlockSpec(block_id="s", spin_S=0.0, ms2=0, nroots=1, weights=[1.0])
    teacher = build_teacher(mf, 2, 2, [block], [1.0], mo0=mf.mo_coeff, cfg=cfg)
    h1e, eri = active_integrals(mf, norb=2)

    problem = SelectionProblem(
        norb=2, nelec=(1, 1), h1e=h1e, eri=eri,
        teacher_states=teacher.ci_vectors,
        state_weights=teacher.state_weights,
        state_ids=teacher.state_ids,
        cfg=cfg,
        spin_S=0.0,
    )
    final, history, report = selection_driver(problem)

    # ground state only -> the compact support must reproduce it faithfully
    assert report.passed or final.gamma_error < cfg.acceptance.tau_gamma
    assert len(history) >= 1
    # driver converged to a small support
    assert len(final.support.families) >= 1
    # history action sequence starts with a solve and ends with certify
    assert history[0]["action"] == "SOLVE"
    assert history[-1]["action"] in ("CERTIFY",)


@pytest.mark.integration
def test_pool_builds_from_teacher():
    from msref.pyscf_adapter import build_teacher

    mol, mf = h2_mf()
    cfg = MSRefConfig()
    block = models.StateBlockSpec(block_id="s", spin_S=0.0, ms2=0, nroots=2, weights=[0.5, 0.5])
    teacher = build_teacher(mf, 2, 2, [block], [0.5, 0.5], mo0=mf.mo_coeff, cfg=cfg)
    h1e, eri = active_integrals(mf, norb=2)
    problem = SelectionProblem(norb=2, nelec=(1, 1), h1e=h1e, eri=eri,
                               teacher_states=teacher.ci_vectors,
                               state_weights=teacher.state_weights,
                               state_ids=teacher.state_ids, cfg=cfg, spin_S=0.0)
    pool = build_pool(problem)
    # family weights should sum to ~1 (2 states, each normalized)
    assert abs(sum(s.qbar for s in pool.values()) - 1.0) < 1e-8
