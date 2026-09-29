"""Scientific smoke benchmarks for the teacher (T016)."""

from __future__ import annotations

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")

from msref.benchmarks.h2 import build_h2, singlet_teacher
from msref.benchmarks.o2 import build_o2, mixed_spin_teacher


@pytest.mark.scientific
def test_h2_teacher_smoke():
    mol, mf = build_h2()
    res = singlet_teacher(mf)
    assert res.converged
    assert np.abs(res.spin_square[0]) < 1e-5
    # H2 ground-state energy in STO-3G at equilibrium (~ -1.137 Eh for CASCI)
    assert res.energies_eh[0] < -1.0


@pytest.mark.scientific
def test_o2_mixed_spin_teacher_smoke():
    mol, mf = build_o2()
    res = mixed_spin_teacher(mf)
    assert res.converged
    # triplet X3Sg- (S^2=2), two singlets (S^2=0)
    assert np.allclose(res.spin_square, [2.0, 0.0, 0.0], atol=1e-4)
    # triplet is the ground state -> lowest energy
    assert res.energies_eh[0] == pytest.approx(min(res.energies_eh), abs=1e-8)
