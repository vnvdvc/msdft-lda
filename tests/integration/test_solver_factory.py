"""Integration tests for the state-block solver factory (T010)."""

from __future__ import annotations

import numpy as np
import pytest

from msref import models
from msref.config import MSRefConfig
from msref.pyscf_adapter import make_solver_for_block, spin_square_of

pyscf = pytest.importorskip("pyscf")


@pytest.mark.integration
def test_singlet_and_triplet_solvers_s2():
    from pyscf import gto
    from .helpers import h2_mf

    mol, mf = h2_mf()
    cfg = MSRefConfig()

    block_s = models.StateBlockSpec(block_id="s", spin_S=0.0, ms2=0, nroots=1, weights=[1.0])
    solver_s, meta_s = make_solver_for_block(mol, block_s, cfg)
    assert meta_s.spin_S == 0.0

    block_t = models.StateBlockSpec(block_id="t", spin_S=1.0, ms2=0, nroots=1, weights=[1.0])
    solver_t, meta_t = make_solver_for_block(mol, block_t, cfg)
    assert meta_t.spin_S == 1.0

    # both solvers are direct_spin1-family and target the requested spin
    assert solver_s.nroots == 1
    assert solver_t.nroots == 1


@pytest.mark.integration
def test_solver_returns_sector_metadata():
    from .helpers import h2_mf

    mol, mf = h2_mf()
    cfg = MSRefConfig()
    block = models.StateBlockSpec(block_id="s", spin_S=0.0, ms2=0, nroots=2, weights=[0.5, 0.5])
    solver, meta = make_solver_for_block(mol, block, cfg)
    assert meta.block_id == "s"
    assert meta.ms2 == 0
    assert solver.nroots == 2
