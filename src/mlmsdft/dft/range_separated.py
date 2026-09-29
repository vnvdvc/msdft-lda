# -*- coding: utf-8 -*-
"""Differentiable range-separated MSDFT components."""
from __future__ import annotations

import torch
from torch import Tensor

from mlmsdft.dft.integrals import RangeSeparatedIntegralCache


class LongRangeInteractionFunctional(torch.nn.Module):
    """Contract a target-state transition 2-RDM with long-range ERIs.

    The transition 2-RDM follows
    ``Gamma[stpqrs,I,J] = <I|a†(p,s) a†(q,t) a(s,t) a(r,s)|J>``.
    With chemist ERIs ``(pq|rs)``, the electronic interaction matrix is
    ``1/2 * einsum('(pq|rs), Gamma')``.  The MO transformation remains in the
    Torch graph, so orbital and target-state derivatives are preserved.
    """

    def __init__(self, mol, omega: float):
        super().__init__()
        self.integrals = RangeSeparatedIntegralCache(mol, omega)

    @property
    def omega(self) -> float:
        return self.integrals.omega

    def forward(self, msmd) -> Tensor:
        gamma = msmd.transition_2rdm_mo()
        eri_mo = self.integrals.eri_mo(msmd.orbital_coefficients(), "long_range")
        return 0.5 * torch.einsum("pqrs,abpqrsij->ij", eri_mo, gamma)
