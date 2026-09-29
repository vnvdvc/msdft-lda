"""Integration tests for density/projector repair and pruning (T038-T040)."""

from __future__ import annotations

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")

import msref.driver as D
from msref import models
from msref.config import MSRefConfig
from msref.config_pool import seed_support
from msref.models import SpatialOccupationKey
from msref.pyscf_adapter import build_teacher
from msref.spin_completion import materialize_support
from .helpers import active_integrals, h2_mf


def _problem():
    mol, mf = h2_mf()
    cfg = MSRefConfig()
    block = models.StateBlockSpec(block_id="s", spin_S=0.0, ms2=0, nroots=1, weights=[1.0])
    teacher = build_teacher(mf, 2, 2, [block], [1.0], mo0=mf.mo_coeff, cfg=cfg)
    h1e, eri = active_integrals(mf, norb=2)
    return mol, mf, cfg, D.SelectionProblem(
        norb=2, nelec=(1, 1), h1e=h1e, eri=eri,
        teacher_states=teacher.ci_vectors, state_weights=teacher.state_weights,
        state_ids=teacher.state_ids, cfg=cfg, spin_S=0.0,
    )


@pytest.mark.integration
def test_pruning_preserves_mandatory_criteria_a005():
    mol, mf, cfg, problem = _problem()
    final, history, report = D.selection_driver(problem)
    assert report.passed
    # every remaining family is locally irreducible: removing any single family
    # must violate some criterion, fail to find the target root, or hit the
    # pruning safety margin
    pool = D.build_pool(problem)
    for f in final.support.families:
        trial = D._add_families_removing(problem, final.support, f)
        if trial is None:
            continue
        try:
            ev = D.solve_support(problem, trial)
        except Exception:  # noqa: BLE001 - root not found -> family essential
            continue
        assert not D._passes_with_safety(problem, ev), f"family {f.occ} is redundant"


def _dominant_family_only(pool, cfg):
    fam = seed_support(pool, [], cfg)
    # keep only the highest-qbar family (drop the tail families)
    top = sorted(fam, key=lambda f: -pool[f].qbar)[:1]
    return tuple(top)


@pytest.mark.integration
def test_density_repair_reduces_error():
    mol, mf, cfg, problem = _problem()
    pool = D.build_pool(problem)
    too_small = _dominant_family_only(pool, cfg)
    support = materialize_support(too_small, [problem.sector], cfg.selected_solver.max_support_determinants)
    ev = D.solve_support(problem, support)
    if ev.gamma_error > cfg.acceptance.tau_gamma:
        repaired, added = D.density_repair(problem, support, ev, pool, cfg)
        ev2 = D.solve_support(problem, repaired)
        assert ev2.gamma_error < ev.gamma_error
        assert len(added) >= 1


@pytest.mark.integration
def test_projector_repair_reduces_distance():
    mol, mf, cfg, problem = _problem()
    pool = D.build_pool(problem)
    too_small = _dominant_family_only(pool, cfg)
    support = materialize_support(too_small, [problem.sector], cfg.selected_solver.max_support_determinants)
    ev = D.solve_support(problem, support)
    if ev.dP_op > cfg.acceptance.tau_projector:
        repaired, added = D.projector_repair(problem, support, ev, pool, cfg)
        ev2 = D.solve_support(problem, repaired)
        assert ev2.dP_op < ev.dP_op
        assert len(added) >= 1
