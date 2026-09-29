# -*- coding: utf-8 -*-
"""Cached one- and two-electron integral contractions.

The range-separated cache deliberately keeps the physical ``omega`` parameter
separate from PySCF's ``omega=0`` sentinel.  In PySCF, zero selects the full
Coulomb kernel, whereas RS-LMDA needs zero to mean a zero long-range kernel.
"""
from __future__ import annotations

import math

import pyscf.gto
import pyscf.pbc.gto

import torch
from torch import Tensor


def _range_eri(mol: pyscf.gto.Mole, omega: float) -> object:
    """Evaluate molecular ERIs for the requested signed range kernel."""
    with mol.with_range_coulomb(omega):
        return mol.intor("int2e", aosym="s1")


class RangeSeparatedIntegralCache(torch.nn.Module):
    """Cache full, long-range, and complementary short-range AO ERIs.

    ``omega`` is a physical non-negative range parameter in inverse bohr.
    ``omega=0`` returns an exactly zero long-range tensor and a full Coulomb
    short-range tensor.  ``omega=inf`` returns the opposite endpoint.  For a
    finite positive value, both kernels are evaluated directly so the short
    range remainder does not suffer cancellation when it is small.
    """

    def __init__(self, mol: pyscf.gto.Mole, omega: float):
        super().__init__()
        if isinstance(mol, pyscf.pbc.gto.cell.Cell):
            raise NotImplementedError("RangeSeparatedIntegralCache currently supports molecules only")
        if not math.isfinite(omega) and omega != math.inf:
            raise ValueError(f"omega must be finite and non-negative or inf, got {omega!r}")
        if omega < 0.0:
            raise ValueError(f"omega must be non-negative, got {omega!r}")
        self.mol = mol
        self.omega = float(omega)
        nao = mol.nao_nr()
        if omega == 0.0:
            eri_full = mol.intor("int2e", aosym="s1")
            eri_lr = eri_full * 0.0
            eri_sr = eri_full
        elif omega == math.inf:
            eri_full = mol.intor("int2e", aosym="s1")
            eri_lr = eri_full
            eri_sr = eri_full * 0.0
        else:
            eri_lr = _range_eri(mol, omega)
            eri_sr = _range_eri(mol, -omega)
        self.register_buffer("eri_long_range_ao", torch.from_numpy(eri_lr).to(dtype=torch.double))
        self.register_buffer("eri_short_range_ao", torch.from_numpy(eri_sr).to(dtype=torch.double))
        self.register_buffer(
            "eri_full_ao",
            (self.eri_long_range_ao + self.eri_short_range_ao).detach().clone(),
        )
        if self.eri_long_range_ao.shape != (nao, nao, nao, nao):
            raise RuntimeError("PySCF returned an unexpected ERI shape")

    def eri_ao(self, component: str = "long_range") -> Tensor:
        """Return a cached AO ERI tensor for ``long_range`` or ``short_range``."""
        aliases = {
            "long": "long_range",
            "lr": "long_range",
            "short": "short_range",
            "sr": "short_range",
            "full": "full",
        }
        component = aliases.get(component, component)
        if component == "long_range":
            return self.eri_long_range_ao
        if component == "short_range":
            return self.eri_short_range_ao
        if component == "full":
            return self.eri_full_ao
        raise ValueError(f"Unknown ERI component {component!r}")

    def eri_mo(self, mo_coeff: Tensor, component: str = "long_range") -> Tensor:
        """Transform the selected ERI component to the live MO basis."""
        eri_ao = self.eri_ao(component).to(dtype=mo_coeff.dtype, device=mo_coeff.device)
        # Contract one AO index at a time.  A single five-operand einsum can
        # choose a path that materializes an eight-index intermediate; for an
        # aug-cc-pVDZ H2 calculation that transient exceeds an A800's memory.
        transformed = torch.einsum("abcd,ap->pbcd", eri_ao, mo_coeff)
        transformed = torch.einsum("pbcd,bq->pqcd", transformed, mo_coeff)
        transformed = torch.einsum("pqcd,cr->pqrd", transformed, mo_coeff)
        return torch.einsum("pqrd,ds->pqrs", transformed, mo_coeff)


class OneElectronIntegralCache(torch.nn.Module):
    """Cache fixed AO one-electron integrals and contract them in the MO basis."""

    def __init__(self, mol: pyscf.gto.Mole, intor: str | None = None):
        super().__init__()
        self.mol = mol
        self.intor = intor
        if intor is None:
            if isinstance(mol, pyscf.pbc.gto.cell.Cell):
                hcore_ao = mol.pbc_intor("int1e_kin") + mol.pbc_intor("int1e_nuc")
            else:
                hcore_ao = mol.intor_symmetric("int1e_kin") + mol.intor_symmetric("int1e_nuc")
        elif isinstance(mol, pyscf.pbc.gto.cell.Cell):
            hcore_ao = mol.pbc_intor(intor)
        else:
            hcore_ao = mol.intor_symmetric(intor)
        self.register_buffer("integrals_ao", torch.from_numpy(hcore_ao).to(dtype=torch.double))

    def integrals_mo(self, mo_coeff: Tensor) -> Tensor:
        hcore_ao = self.integrals_ao.to(dtype=mo_coeff.dtype, device=mo_coeff.device)
        return torch.einsum("ap,ab,bq->pq", mo_coeff, hcore_ao, mo_coeff)

    def matrix_elements_from_gamma(self, gamma_mo: Tensor, mo_coeff: Tensor) -> Tensor:
        """Contract spin-traced transition 1-RDMs with cached MO integrals."""
        hcore_mo = self.integrals_mo(mo_coeff)
        gamma_total = torch.einsum("ss...->...", gamma_mo)
        return torch.einsum("pq,pqij->ij", hcore_mo, gamma_total)
