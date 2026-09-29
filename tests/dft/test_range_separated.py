#!/usr/bin/env python
# coding: utf-8
"""Regression tests for the first range-separated LMDA implementation slice."""

import unittest

import numpy
import pyscf.gto
import torch

from mlmsdft.dft.density import TargetStateMultistateMatrixDensityCAS
from mlmsdft.dft.hamiltonian import HamiltonianTargetStateLMDA
from mlmsdft.dft.hartree import HartreeFunctionalShortRangeAO
from mlmsdft.dft.integrals import RangeSeparatedIntegralCache
from mlmsdft.dft.range_separated import LongRangeInteractionFunctional
from mlmsdft.dft.spin import SpinType
from mlmsdft.dft.xc import complementary_sr_lda_unpolarized, lda_xc_dirac_chachiyo_unpolarized


class TestRangeSeparatedIntegrals(unittest.TestCase):
    @staticmethod
    def molecule():
        return pyscf.gto.M(
            atom="H 0 0 -0.35; H 0 0 0.35",
            basis="sto-3g",
            charge=0,
            spin=0,
            verbose=0,
        )

    def test_long_plus_short_partition_and_endpoints(self):
        mol = self.molecule()
        full = torch.from_numpy(mol.intor("int2e", aosym="s1"))
        cache = RangeSeparatedIntegralCache(mol, 0.4)
        torch.testing.assert_close(cache.eri_ao("long") + cache.eri_ao("short"), full, atol=2e-12, rtol=2e-12)

        zero = RangeSeparatedIntegralCache(mol, 0.0)
        torch.testing.assert_close(zero.eri_ao("long"), torch.zeros_like(full))
        torch.testing.assert_close(zero.eri_ao("short"), full)

        infinite = RangeSeparatedIntegralCache(mol, numpy.inf)
        torch.testing.assert_close(infinite.eri_ao("long"), full)
        torch.testing.assert_close(infinite.eri_ao("short"), torch.zeros_like(full))

    def test_short_range_hartree_is_differentiable(self):
        mol = self.molecule()
        functional = HartreeFunctionalShortRangeAO(mol, 0.4)
        density = torch.eye(mol.nao_nr(), dtype=torch.double).unsqueeze(-1).unsqueeze(-1)
        density = density.expand(-1, -1, 2, 2).clone().requires_grad_(True)
        value = functional(density).sum()
        value.backward()
        self.assertTrue(torch.isfinite(density.grad).all())

    def test_complementary_sr_lda_limits_and_derivative(self):
        density = torch.diag(torch.tensor([0.1, 0.2], dtype=torch.double)).requires_grad_(True)
        zero = complementary_sr_lda_unpolarized(density, 0.0)
        finite = complementary_sr_lda_unpolarized(density, 0.4)
        infinite = complementary_sr_lda_unpolarized(density, numpy.inf)
        self.assertTrue(torch.isfinite(zero).all())
        self.assertTrue(torch.isfinite(finite).all())
        torch.testing.assert_close(infinite, torch.zeros_like(infinite))
        finite.sum().backward()
        self.assertTrue(torch.isfinite(density.grad).all())


class TestLongRangeTransitionInteraction(unittest.TestCase):
    def test_closed_shell_determinant_and_orbital_gradient(self):
        mol = pyscf.gto.M(
            atom="H 0 0 -0.35; H 0 0 0.35",
            basis="sto-3g",
            charge=0,
            spin=0,
            verbose=0,
        )
        msmd = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol,
            norb=2,
            nelec=2,
            target_states=1,
            spin_symmetry=True,
            spin_type=SpinType.UNPOLARIZED,
            guess="hcore",
        )
        functional = LongRangeInteractionFunctional(mol, 0.4)
        interaction = functional(msmd)
        gamma = msmd.transition_2rdm_mo()
        self.assertEqual(tuple(gamma.shape), (2, 2, 2, 2, 2, 2, 1, 1))

        eri_mo = functional.integrals.eri_mo(msmd.orbital_coefficients(), "long")
        expected = eri_mo[0, 0, 0, 0]
        torch.testing.assert_close(interaction[0, 0], expected, atol=2e-12, rtol=2e-12)

        # Independent AO density contraction for a doubly occupied spatial
        # orbital: E_ee = 1/2 Tr[P J] - 1/4 Tr[P K] with P=2|0><0|.
        eri_ao = functional.integrals.eri_ao("long")
        coeff = msmd.orbital_coefficients().detach()
        density = 2.0 * coeff[:, :1] @ coeff[:, :1].T
        J = torch.einsum("abcd,cd->ab", eri_ao, density)
        K = torch.einsum("abcd,bd->ac", eri_ao, density)
        independent = 0.5 * torch.einsum("ab,ab", density, J) - 0.25 * torch.einsum("ac,ac", density, K)
        torch.testing.assert_close(interaction[0, 0].detach(), independent, atol=2e-12, rtol=2e-12)

        msmd.zero_grad()
        interaction.sum().backward()
        self.assertTrue(torch.isfinite(msmd.orbital_rotation_params.grad).all())

    def test_target_state_hamiltonian_range_endpoint(self):
        mol = pyscf.gto.M(
            atom="H 0 0 -0.35; H 0 0 0.35",
            basis="sto-3g",
            charge=0,
            spin=0,
            verbose=0,
        )
        msmd = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, norb=2, nelec=2, target_states=1,
            spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess="hcore",
        )
        full_range = HamiltonianTargetStateLMDA(
            mol,
            grid_level=1,
            omega=0.0,
            short_range_exchange_correlation_functional=lda_xc_dirac_chachiyo_unpolarized,
        )
        H = full_range(msmd)
        self.assertTrue(torch.isfinite(H).all())
        self.assertGreater(float(H[0, 0]), -10.0)

        zero_sr = HamiltonianTargetStateLMDA(
            mol,
            grid_level=1,
            omega=0.4,
            short_range_exchange_correlation_functional=lambda density, *_: complementary_sr_lda_unpolarized(density, 0.4),
        )
        H_lr = zero_sr(msmd)
        self.assertTrue(torch.isfinite(H_lr).all())
