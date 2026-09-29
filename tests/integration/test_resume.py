"""Integration test for deterministic checkpoint/resume (T042)."""

from __future__ import annotations

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")

import msref.driver as D
from msref import models
from msref.config import MSRefConfig
from msref.io import dumps_json
from msref.models import SpatialOccupationKey
from msref.pyscf_adapter import build_teacher
from .helpers import active_integrals, h2_mf


def _serialize_families(families):
    return dumps_json([list(f.occ) for f in families])


def _deserialize_families(text):
    import json
    return tuple(SpatialOccupationKey(tuple(occ)) for occ in json.loads(text))


@pytest.mark.integration
def test_resume_from_seed_matches_full_run_i017():
    mol, mf = h2_mf()
    cfg = MSRefConfig()
    block = models.StateBlockSpec(block_id="s", spin_S=0.0, ms2=0, nroots=1, weights=[1.0])
    teacher = build_teacher(mf, 2, 2, [block], [1.0], mo0=mf.mo_coeff, cfg=cfg)
    h1e, eri = active_integrals(mf, norb=2)
    problem = D.SelectionProblem(norb=2, nelec=(1, 1), h1e=h1e, eri=eri,
                                 teacher_states=teacher.ci_vectors,
                                 state_weights=teacher.state_weights,
                                 state_ids=teacher.state_ids, cfg=cfg, spin_S=0.0)

    final_full, _, report_full = D.selection_driver(problem)

    # "resume" from the seed (skip re-seeding) must give the identical support
    seed = __import__("msref.config_pool", fromlist=["seed_support"]).seed_support(
        D.build_pool(problem), [], cfg
    )
    final_resumed, _, report_resumed = D.selection_driver(problem, initial_families=seed)

    assert _serialize_families(final_full.support.families) == _serialize_families(
        final_resumed.support.families
    )
    assert report_full.passed == report_resumed.passed
    assert np.allclose(final_full.energies, final_resumed.energies, atol=1e-10)
