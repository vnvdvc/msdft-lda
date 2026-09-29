"""Integration tests for the selected Hamiltonian and solver (T026/T027)."""

from __future__ import annotations

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")
from pyscf import fci

from msref.selected_hamiltonian import build_selected_hamiltonian
from msref.selected_solver import solve_selected_sector
from msref.spin_completion import all_determinants
from .helpers import active_integrals, h2_mf


@pytest.mark.integration
def test_full_support_matches_fci_i008():
    mol, mf = h2_mf()
    h1e, eri = active_integrals(mf, norb=2)
    dets = all_determinants(2, 1, 1)
    ham = build_selected_hamiltonian(dets, h1e, eri)
    # hermiticity
    assert np.linalg.norm(ham - ham.conj().T) < 1e-12
    evals = np.sort(np.linalg.eigvalsh(ham).real)
    e_fci, _ = fci.direct_spin1.kernel(h1e, eri, 2, (1, 1), nroots=len(dets))
    assert np.allclose(evals, np.sort(e_fci), atol=1e-11)


@pytest.mark.integration
def test_random_subset_matches_fci_action_i009():
    mol, mf = h2_mf()
    norb = 2
    h1e, eri = active_integrals(mf, norb=norb)
    dets = all_determinants(norb, 1, 1)
    full_ham = build_selected_hamiltonian(dets, h1e, eri)
    subset = dets[:3]
    sub_ham = build_selected_hamiltonian(subset, h1e, eri)
    idx = [dets.index(d) for d in subset]
    assert np.allclose(sub_ham, full_ham[np.ix_(idx, idx)], atol=1e-12)


@pytest.mark.integration
def test_solver_roots_and_buffer():
    mol, mf = h2_mf()
    h1e, eri = active_integrals(mf, norb=2)
    dets = all_determinants(2, 1, 1)
    ham = build_selected_hamiltonian(dets, h1e, eri)
    evals, evecs = solve_selected_sector(ham, nroots=2, root_buffer=2)
    assert evals.shape == (4,)
    assert np.allclose(evals, np.sort(np.linalg.eigvalsh(ham).real)[:4], atol=1e-10)
