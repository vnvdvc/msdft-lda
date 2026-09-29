"""Shared helpers for integration tests (tiny H2/O2 model systems)."""

from __future__ import annotations

import numpy as np
from pyscf import ao2mo, gto, scf


def h2_mf(basis="sto-3g", r=0.74):
    mol = gto.M(atom=f"H 0 0 0; H 0 0 {r}", basis=basis, unit="angstrom", verbose=0)
    mf = scf.RHF(mol).run()
    return mol, mf


def active_integrals(mf, norb=None):
    """Return ``(h1e, eri)`` transformed to the MO basis (first ``norb`` orbitals)."""
    mo = mf.mo_coeff
    if norb is not None:
        mo = mo[:, :norb]
    h1e = mo.T @ mf.get_hcore() @ mo
    eri = ao2mo.restore(1, ao2mo.incore.full(mf.mol.intor("int2e"), mo), mo.shape[1])
    return h1e, eri


def o2_mf(basis="cc-pvdz", r=1.2):
    mol = gto.M(atom=f"O 0 0 0; O 0 0 {r}", basis=basis, unit="angstrom",
                spin=0, verbose=0)
    mf = scf.RHF(mol).run()
    return mol, mf
