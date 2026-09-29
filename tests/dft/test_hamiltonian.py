#!/usr/bin/env python
# coding: utf-8
import numpy
import importlib.util
import pathlib
import pyscf.gto
import torch
from torch import Tensor
import torch.linalg
import torch.testing
from tqdm import tqdm
import types
import unittest
import warnings

from mlmsdft.dft.density import MultistateMatrixDensity
from mlmsdft.dft.density import MultistateMatrixDensityCAS
from mlmsdft.dft.density import MultistateMatrixDensityKohnSham
from mlmsdft.dft.density import TargetStateMultistateMatrixDensityCAS
from mlmsdft.dft.hamiltonian import Hamiltonian
from mlmsdft.dft.hamiltonian import HamiltonianSemilocal
from mlmsdft.dft.hamiltonian import HamiltonianTargetStateLMDA
from mlmsdft.dft.hamiltonian import noncollinear_channel_densities
from mlmsdft.dft.hamiltonian import minimize_subspace_energy
from mlmsdft.dft.hartree_gpu4pyscf import HartreeFunctionalGpu4PySCFDFAO
from mlmsdft.dft.hartree_gpu4pyscf import gpu4pyscf_environment_probe
from mlmsdft.dft.spin import SpinType
from mlmsdft.dft.spin import concat_spin_blocks, spin_trace
from mlmsdft.dft.xc import lda_c_chachiyo, lda_x_dirac
from mlmsdft.dft.pure import (
    LDA,
)
from mlmsdft.optim.torch_optimizer import WrappedOptimizer
import mlmsdft.nn.functional as MF
import mlmsdft.dft.hamiltonian as hamiltonian_mod

from dft.fixture import FixtureMixin
# Need to change name of class to avoid running unittest on it.
from dft.test_density import TestMultistateMatrixDensityCAS as MultistateMatrixDensityCASTest

kinetic_functionals = [
    # Here `None` means that the kinetic energy matrix is not computed from
    # the density but from the wavefunctions.
    None,
]

correlation_functionals = {
    "lda": [lda_c_chachiyo],
}

exchange_functionals = {
    "lda": [lda_x_dirac],
}

# Classes for instantiating pure composite functionals.
# These functionals have to be intantiated with one dummy argument,
# which is not used, so that they have the same interface as the
# hybrid functionals
pure_xc_functionals = {
    "LDA": LDA,
}


# This member function is dynamically attached to an instance of HamiltonianSemilocal
# for testing purposes
def _matrix_elements_single_chunk(self: HamiltonianSemilocal, msmd: MultistateMatrixDensity) -> Tensor:
    """
    Compute the Hamiltonian matrix H[D(r)]ᵢⱼ in the basis of electronic states at a
    given matrix density D(r). This function is a copy of HamiltonianSemilocal.matrix_elements(...)
    with the only difference that the integration over grid points is performed in a single
    without splitting the grid into chunks.

    :param msmd: multistate matrix density
    :type msmd: :class:`~.MultistateMatrixDensity`

    :return hamiltonian: Hamiltonian matrix Hᵢⱼ
    :rtype hamiltonian: Tensor of shape (Nstate,Nstate)
    """
    old_grid_chunks = self.grid_chunks
    old_cache = self._ao_grid_cache
    old_cache_key = self._ao_grid_cache_key
    try:
        self.grid_chunks = 1
        self._ao_grid_cache = None
        self._ao_grid_cache_key = None
        return HamiltonianSemilocal.matrix_elements(self, msmd)
    finally:
        self.grid_chunks = old_grid_chunks
        self._ao_grid_cache = old_cache
        self._ao_grid_cache_key = old_cache_key

    # Check that the same geometry and basis is used for defining
    # the Hamiltonian and the matrix density.
    assert id(self.mol) == id(msmd.mol)
    # Use AO representation to compute external potential,
    # Hartree matrices, exact exchange matrices and kinetic energy
    # if there is not kinetic density functional.
    spin_dm = msmd.density_matrices_ao()
    # sum over spins, Dᵦᵧᵢⱼ = Dᵅᵅᵦᵧᵢⱼ + Dᵝᵝᵦᵧᵢⱼ
    dm = torch.einsum('ss...->...', spin_dm)
    # electron-nuclei attraction
    Ven = self.nuclear_ao(dm)
    # Hartree-part of electron-electron repulsion
    J = self.hartree_ao(dm)

    # nuclear repulsion energy between ions is the same
    # for all electronic states
    Vnn = self.mol.get_enuc() * torch.eye(msmd.number_of_states).to(
        dtype=Ven.dtype, device=Ven.device)

    # Hamiltonian matrix (without T, X and C)
    H = Vnn + Ven + J

    if self.kinetic is None:
        # compute T from atomic orbital representation of matrix density
        T = self.kinetic_ao(dm)
        # Add kinetic energy to Hamiltonian, if the kinetic energy is calculated
        # with an orbital-free functional it is added further down.
        H = H + T

    # Exact exchange is calculated from density matrices in AO basis.
    if self.exact_exchange is not None:
        match self.spin_type:
            case SpinType.UNPOLARIZED:
                # Compute exact exchange matrix from total charge density.
                # Assuming that the spin-up and spin-down densities are the same,
                # Dᵅᵅ(r,r') = Dᵝᵝ(r,r') = D(r,r')/2,
                # the exact exchange is calculated as
                # K[Dᵅᵅ(r,r')]+K[Dᵝᵝ(r,r')] = 2*K[D(r,r')/2]ᵢⱼ.
                K = 2.0*self.exact_exchange(dm/2.0)
            case SpinType.POLARIZED:
                # Densities for spin-up and spin-down contribute separately to the
                # exchange.
                #   K = ax * K[Dᵅᵅ(r,r')]ᵢⱼ + ax * K[Dᵝᵝ(r,r')]ᵢⱼ
                #     = ax * (
                #       -1/2 ∑ₖ ∫∫' Dᵅᵅᵢₖ(r,r') Dᵅᵅₖⱼ(r',r)/|r-r'| +
                #       -1/2 ∑ₖ ∫∫' Dᵝᵝᵢₖ(r,r') Dᵝᵝₖⱼ(r',r)/|r-r'| )
                K = (
                    # K[Dᵅᵅ]ᵢⱼ
                    self.exact_exchange(spin_dm[0,0,...]) +
                    # K[Dᵝᵝ]ᵢⱼ
                    self.exact_exchange(spin_dm[1,1,...])
                )
            case SpinType.INVARIANT:
                # Stack spin blocks into a (2N)*(2N) supermatrix:
                # (Dᵅᵅ Dᵅᵝ)
                # (Dᵝᵅ Dᵝᵝ)
                super_spin_dm = concat_spin_blocks(spin_dm)
                # The exact exchange matrix functional operates on the supermatrix
                # to get a (2N)*(2N) exchange matrix
                # (Exxᵅᵅ Exxᵅᵝ)          (Dᵅᵅ Dᵅᵝ)
                # (           ) = Exx { (       ) }
                # (Exxᵝᵅ Exxᵝᵝ )          (Dᵝᵅ Dᵝᵝ)
                # By spin-tracing out the off-diagonal block we obtain the N*N exchange
                # matrix summed over spins:
                #                                  (Dᵅᵅ Dᵅᵝ)
                # Exxᵅᵅ + Exxᵝᵝ = spin_trace( Exx { (       ) } )
                #                                  (Dᵝᵅ  Dᵝᵝ)
                K = spin_trace(self.exact_exchange(super_spin_dm))
            case SpinType.INVARIANT_MIX:
                # Exact exchange from spin matrix density.
                super_spin_dm = concat_spin_blocks(spin_dm)
                K_inv = spin_trace(self.exact_exchange(super_spin_dm))
                # Exact exchange from total charge matrix density
                K_unpol = 2.0*self.exact_exchange(dm/2.0)
                # Average of unpolarized and invariant treatment.
                K = 0.5 * (K_unpol + K_inv)
            case _:
                raise ValueError(f"`spin_type` must be instance of SpinType, got {self.spin_type}")
        H = H + K

    # evaluate semilocal kinetic, exchange and correlation functionals
    # Inputs are D(r) and ∇D(r) on the integration grid.
    # NOTE: ∇²D is only calculated if it is needed by any of the functionals.
    spin_D, grad_spin_D, lapl_spin_D = msmd.evaluate(
        self.grids.coords, need_laplacian=self.need_laplacian)

    # Sum over spin
    D = torch.einsum('ss...->...', spin_D)
    grad_D = torch.einsum('ss...->...', grad_spin_D)
    # If ∇²D is not needed by any functional, it is set to None.
    if lapl_spin_D is not None:
        lapl_D = torch.einsum('ss...->...', lapl_spin_D)
    else:
        lapl_D = None

    # kinetic (t), exchange (x) and correlation (c) energy densities
    if self.kinetic is not None:
        t = self.kinetic(D, grad_D, lapl_D)

    if self.exchange is not None:
        match self.spin_type:
            case SpinType.UNPOLARIZED:
                # The spin-restricted/unpolarized version of the exchange functional assumes
                # that the spin-up and spin-down parts of the matrix density are the same,
                # D(r) = Dᵅ(r)+Dᵝ(r) = 2 Dᵅ(r)
                # Therefore, one can reconstruct the spin matrix density from the
                # charge matrix density,
                #   Dᵅ(r) = D(r)/2
                # and similarly for the gradient
                #   ∇Dᵅ(r) = ∇D(r)/2
                # and the Laplacian
                #   ∇²Dᵅ(r) = ∇²D(r)/2
                # Since, the spin matrix densities for up and down are the same,
                # the exchange energy is only calculated once and then multiplied by two.
                # xed[Dᵅ(r),Dᵝ(r)] = xed[Dᵅ(r)] + xed[Dᵝ(r)] = 2 xed[Dᵅ(r)]
                if lapl_D is None:
                    # Functional does not depend on ∇²D(r)/2
                    x = 2.0 * self.exchange(D/2.0, grad_D/2.0, None)
                else:
                    x = 2.0 * self.exchange(D/2.0, grad_D/2.0, lapl_D/2.0)
            case SpinType.POLARIZED:
                # Compute the exchange energy for spin-up and spin-down electrons
                # separately and add it,
                # xed[Dᵅᵅ(r),Dᵝᵝ(r)] = xed[Dᵅᵅ(r)] + xed[Dᵝᵝ(r)]
                if lapl_spin_D is None:
                    # Functional does not depend on ∇²Dᵅᵅ(r) or ∇²Dᵝᵝ(r)
                    x = (
                        # xed(        Dᵅᵅ(r),          ∇Dᵅᵅ(r)   ) +
                        self.exchange(spin_D[0,0,...], grad_spin_D[0,0,...], None) +
                        # xed(        Dᵝᵝ(r),           ∇Dᵝᵝ(r)    )
                        self.exchange(spin_D[1,1,...], grad_spin_D[1,1,...], None)
                    )
                else:
                    x = (
                        # xed(        Dᵅᵅ(r),          ∇Dᵅᵅ(r),              ∇²Dᵅᵅ(r)  ) +
                        self.exchange(spin_D[0,0,...], grad_spin_D[0,0,...], lapl_spin_D[0,0,...]) +
                        # xed(        Dᵝᵝ(r),           ∇Dᵝᵝ(r),               ∇²Dᵝᵝ(r)  )
                        self.exchange(spin_D[1,1,...], grad_spin_D[1,1,...], lapl_spin_D[1,1,...])
                    )
            case SpinType.INVARIANT:
                # Stack spin blocks into a (2N)*(2N) supermatrix:
                # (Dᵅᵅ Dᵅᵝ)
                # (Dᵝᵅ  Dᵝᵝ)
                super_spin_D = concat_spin_blocks(spin_D)
                # similarly for gradient
                # (∇Dᵅᵅ ∇Dᵅᵝ)
                # (∇Dᵝᵅ ∇Dᵝᵝ)
                super_grad_spin_D = concat_spin_blocks(grad_spin_D)
                # and for Laplacian
                # (∇²Dᵅᵅ ∇²Dᵅᵝ)
                # (∇²Dᵝᵅ ∇²Dᵝᵝ)  if available
                if lapl_spin_D is None:
                    super_lapl_spin_D = None
                else:
                    super_lapl_spin_D = concat_spin_blocks(lapl_spin_D)
                # The exchange energy density matrix functional operates on the supermatrix
                # to get a (2N)*(2N) matrix with the exchange energy density
                # (xedᵅᵅ(r) xedᵅᵝ(r))         (Dᵅᵅ(r) Dᵅᵝ(r))
                # (                 ) = xed[ (             ) ]
                # (xedᵝᵅ(r)  xedᵝᵝ(r))         (Dᵝᵅ(r)  Dᵝᵝ(r))
                # By spin-tracing out the off-diagonal block we obtain the N*N matrix for
                # the exchange energy density:
                #                                       (Dᵅᵅ(r) Dᵅᵝ(r))
                # xedᵅᵅ(r) + xedᵝᵝ(r) = spin_trace( xed[ (             ) ] )
                #                                       (Dᵝᵅ(r)  Dᵝᵝ(r))
                x = spin_trace(
                    self.exchange(super_spin_D, super_grad_spin_D, super_lapl_spin_D)
                )
            case SpinType.INVARIANT_MIX:
                # Stack spin blocks into a (2N)*(2N) supermatrix:
                super_spin_D = concat_spin_blocks(spin_D)
                # similarly for gradient
                super_grad_spin_D = concat_spin_blocks(grad_spin_D)
                # and for Laplacian if available
                if lapl_spin_D is None:
                    super_lapl_spin_D = None
                else:
                    super_lapl_spin_D = concat_spin_blocks(lapl_spin_D)
                # Exchange from spin matrix density
                x_inv = spin_trace(
                    self.exchange(super_spin_D, super_grad_spin_D, super_lapl_spin_D)
                )
                # Exchange from charge matrix density
                if lapl_D is None:
                    # Functional does not depend on ∇²D(r)/2
                    x_unpol = 2.0 * self.exchange(D/2.0, grad_D/2.0, None)
                else:
                    x_unpol = 2.0 * self.exchange(D/2.0, grad_D/2.0, lapl_D/2.0)
                # Average of unpolarized and invariant treatment.
                x = 0.5 * (x_unpol + x_inv)
            case _:
                raise ValueError(f"`spin_type` must be instance of SpinType, got {self.spin_type}")
    else:
        x = torch.zeros_like(D, dtype=D.dtype, device=D.device)

    if self.correlation is not None:
        # Some correlation functions operate on the total charge density, while
        # others expect separate matrix densities for spin-up and spin-down.
        if getattr(self.correlation, "need_spin_density", False):
            # Spin-polarized correlation functionals mix the spin-up and spin-down
            # densities in a complicated way that breaks the rotational invariance.
            c = self.correlation(
                # Only keep the same-spin blocks
                # Dᵅᵅ(r) and Dᵝᵝ(r)
                torch.einsum('ss...->s...', spin_D),
                # ∇Dᵅᵅ and ∇Dᵝᵝ
                torch.einsum('ss...->s...', grad_spin_D),
                # ∇²Dᵅᵅ and ∇²Dᵝᵝ is available
                None if lapl_spin_D is None else torch.einsum('ss...->s...', lapl_spin_D),
                spin_polarized=True
            )
        else:
            c = self.correlation(D, grad_D, lapl_D)
    else:
        c = torch.zeros_like(D, dtype=D.dtype, device=D.device)

    # integrate energy densities
    weights = torch.from_numpy(self.grids.weights).to(dtype=D.dtype, device=D.device)
    # reshape weights so that they can be multiplied with matrices
    # using the broadcasting rules
    #   weights -> (...,1,1)
    #   t          (...,n,n)
    weights = weights.unsqueeze(-1).unsqueeze(-1)
    X = torch.sum(x * weights, 0)
    C = torch.sum(c * weights, 0)

    # Add exchange and correlation energy to Hamiltonian
    H = H + X + C

    if self.kinetic is not None:
        T = torch.sum(t * weights, 0)
        # Add kinetic energy from orbital-free functional
        H = H + T

    return H


class TestHamiltonian(unittest.TestCase, FixtureMixin):
    def test_noncollinear_channel_densities_reduce_to_collinear_spin_blocks(self):
        Daa = torch.tensor([
            [[1.2, 0.1], [0.1, 0.8]],
            [[0.7, 0.0], [0.0, 0.5]],
        ], dtype=torch.double)
        Dbb = torch.tensor([
            [[0.4, 0.05], [0.05, 0.3]],
            [[0.2, 0.0], [0.0, 0.1]],
        ], dtype=torch.double)
        spin_D = torch.zeros((2, 2) + Daa.size(), dtype=torch.double)
        spin_D[0, 0, ...] = Daa
        spin_D[1, 1, ...] = Dbb

        D_plus, D_minus, spin_direction = noncollinear_channel_densities(
            spin_D, spin_vector_regularization=1.0e-10)

        torch.testing.assert_close(D_plus, Daa, rtol=1.0e-12, atol=1.0e-12)
        torch.testing.assert_close(D_minus, Dbb, rtol=1.0e-12, atol=1.0e-12)
        self.assertIsNone(spin_direction)

    def test_target_state_one_electron_mo_matches_ao_backend(self):
        cases = [
            (self.create_test_molecules()["hydrogen molecule"], 2, 2, 2, "hydrogen molecule"),
            (self.create_test_molecules()["lithium hydride"], 4, 4, 3, "lithium hydride"),
            (pyscf.gto.M(atom="C 0 0 0; C 0 0 1.25", basis="sto-3g", spin=0), 2, 2, 2, "carbon dimer"),
        ]
        for mol, norb, nelec, target_states, molecule in cases:
            with self.subTest(molecule=molecule):
                msmd = TargetStateMultistateMatrixDensityCAS.from_guess(
                    mol, norb, nelec, target_states=target_states,
                    spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess="hcore")
                hamiltonian = HamiltonianTargetStateLMDA(mol, hartree_backend="ao")
                one_mo = hamiltonian.one_electron_matrix(msmd)
                dm = torch.einsum('ss...->...', msmd.density_matrices_ao())
                one_ao = hamiltonian_mod.NuclearFunctionalAO(mol)(dm) + hamiltonian_mod.KineticFunctionalAO(mol)(dm)
                torch.testing.assert_close(one_mo, one_ao, rtol=1.0e-9, atol=1.0e-9)

    def test_target_state_lmda_hamiltonian_matches_semilocal_h2_full_space(self):
        mol = self.create_test_molecules()["hydrogen molecule"]
        dense = MultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess="hcore")
        target = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, target_states=dense.number_of_states,
            spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess="hcore")
        reference = HamiltonianSemilocal(
            mol,
            exchange_functional=lda_x_dirac,
            correlation_functional=lda_c_chachiyo,
            spin_type=SpinType.UNPOLARIZED,
            grid_level=1,
            grid_chunks=2,
        )
        target_hamiltonian = HamiltonianTargetStateLMDA(
            mol,
            exchange_functional=lda_x_dirac,
            correlation_functional=lda_c_chachiyo,
            spin_type=SpinType.UNPOLARIZED,
            grid_level=1,
            grid_chunks=2,
            hartree_backend="ao",
        )
        torch.testing.assert_close(
            target_hamiltonian(target), reference(dense), rtol=1.0e-8, atol=1.0e-8)

    def test_stiefel_target_hamiltonian_zero_params_matches_full_rotation(self):
        mol = self.create_test_molecules()["hydrogen molecule"]
        full = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, target_states=2,
            spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess="hcore",
            target_parameterization="full_rotation")
        stiefel = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, target_states=2,
            spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess="hcore",
            target_parameterization="stiefel_k")
        hamiltonian = HamiltonianTargetStateLMDA(
            mol,
            exchange_functional=lda_x_dirac,
            correlation_functional=lda_c_chachiyo,
            spin_type=SpinType.UNPOLARIZED,
            grid_level=1,
            grid_chunks=2,
            hartree_backend="ao",
        )

        torch.testing.assert_close(
            hamiltonian(stiefel), hamiltonian(full), rtol=1.0e-10, atol=1.0e-10)

    def test_target_state_noncollinear_lmda_hamiltonian_matches_semilocal_h2_full_space(self):
        mol = self.create_test_molecules()["hydrogen molecule"]
        dense = MultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, spin_symmetry=False, spin_type=SpinType.NONCOLLINEAR, guess="hcore")
        target = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, target_states=dense.number_of_states,
            spin_symmetry=False, spin_type=SpinType.NONCOLLINEAR, guess="hcore")
        reference = HamiltonianSemilocal(
            mol,
            exchange_functional=lda_x_dirac,
            correlation_functional=lda_c_chachiyo,
            spin_type=SpinType.NONCOLLINEAR,
            grid_level=1,
            grid_chunks=2,
        )
        target_hamiltonian = HamiltonianTargetStateLMDA(
            mol,
            exchange_functional=lda_x_dirac,
            correlation_functional=lda_c_chachiyo,
            spin_type=SpinType.NONCOLLINEAR,
            grid_level=1,
            grid_chunks=2,
            hartree_backend="ao",
        )
        torch.testing.assert_close(
            target_hamiltonian(target), reference(dense), rtol=1.0e-8, atol=1.0e-8)

    def test_target_state_noncollinear_rejects_fused_xc_functional(self):
        mol = self.create_test_molecules()["hydrogen molecule"]
        xc_functional = LDA(mol)
        with self.assertRaises(NotImplementedError):
            HamiltonianTargetStateLMDA(
                mol,
                exchange_functional=None,
                correlation_functional=None,
                exchange_correlation_functional=xc_functional.exchange_correlation,
                spin_type=SpinType.NONCOLLINEAR,
            )

    def test_target_state_noncollinear_xc_autograd(self):
        mol = self.create_test_molecules()["hydrogen molecule"]
        msmd = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, target_states=2,
            spin_symmetry=False, spin_type=SpinType.NONCOLLINEAR, guess="hcore")
        hamiltonian = HamiltonianTargetStateLMDA(
            mol,
            spin_type=SpinType.NONCOLLINEAR,
            grid_level=1,
            grid_chunks=2,
            hartree_backend="ao",
        )
        energy = torch.trace(hamiltonian.exchange_correlation_matrix(msmd))
        energy.backward()

        self.assertIsNotNone(msmd.target_rotation_params.grad)
        self.assertIsNotNone(msmd.orbital_rotation_params.grad)
        self.assertFalse(msmd.target_rotation_params.grad.isnan().any())
        self.assertFalse(msmd.orbital_rotation_params.grad.isnan().any())

    def test_torch_lbfgs_optimizer_smoke_preserves_noncollinear_target_orthonormality(self):
        mol = self.create_test_molecules()["hydrogen molecule"]
        msmd = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, target_states=2,
            spin_symmetry=False, spin_type=SpinType.NONCOLLINEAR, guess="hcore")
        hamiltonian = HamiltonianTargetStateLMDA(
            mol,
            spin_type=SpinType.NONCOLLINEAR,
            grid_level=1,
            grid_chunks=2,
            hartree_backend="ao",
        )
        energies, optimized = minimize_subspace_energy(
            hamiltonian,
            msmd,
            optimizer="torch_lbfgs",
            maxiter=2,
            gtol=1.0e-8,
        )

        self.assertEqual(energies.size(), torch.Size([msmd.number_of_states]))
        self.assertFalse(energies.isnan().any())
        self.assertLess(optimized.target_orthonormality_error().item(), 1.0e-10)
        self.assertLess(optimized.orbital_orthonormality_error().item(), 1.0e-8)

    def test_gpu4pyscf_hartree_backend_fallback_matches_ao(self):
        mol = self.create_test_molecules()["hydrogen molecule"]
        msmd = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, target_states=2,
            spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess="hcore")
        dm = torch.einsum('ss...->...', msmd.density_matrices_ao())
        fallback = HartreeFunctionalGpu4PySCFDFAO(mol, allow_fallback=True)
        reference = hamiltonian_mod.HartreeFunctionalAO(mol)
        if not fallback.using_fallback:
            self.skipTest("GPU4PySCF is installed; fallback path is not active.")
        torch.testing.assert_close(fallback(dm), reference(dm))

    def test_gpu4pyscf_environment_probe_reports_required_keys(self):
        mol = self.create_test_molecules()["hydrogen molecule"]
        info = gpu4pyscf_environment_probe(mol)
        for key in [
            "gpu4pyscf_available",
            "cupy_available",
            "torch_cuda_available",
            "df_build_succeeded",
            "torch_version",
        ]:
            self.assertIn(key, info)

    def test_gpu4pyscf_direct_path_rejects_cpu_without_fallback(self):
        mol = self.create_test_molecules()["hydrogen molecule"]
        msmd = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, target_states=2,
            spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess="hcore")
        dm = torch.einsum('ss...->...', msmd.density_matrices_ao())
        backend = HartreeFunctionalGpu4PySCFDFAO(mol, allow_fallback=False, dfobj=object())
        with self.assertRaises(ValueError):
            backend(dm)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for direct GPU4PySCF DLPack test")
    def test_gpu4pyscf_direct_path_fake_builder_autograd(self):
        try:
            import cupy
        except ImportError:
            self.skipTest("CuPy is required for direct GPU4PySCF DLPack test")
        mol = self.create_test_molecules()["hydrogen molecule"]
        msmd = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, target_states=2,
            spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess="hcore")
        dm = torch.einsum('ss...->...', msmd.density_matrices_ao()).detach().cuda().requires_grad_(True)

        def fake_j_builder(dm_batch):
            return dm_batch

        backend = HartreeFunctionalGpu4PySCFDFAO(
            mol, allow_fallback=False, dfobj=object(), j_builder=fake_j_builder)
        energy = torch.trace(backend(dm))
        energy.backward()
        self.assertIsNotNone(dm.grad)
        self.assertFalse(dm.grad.isnan().any())

    def test_target_state_hartree_backward_finite_difference(self):
        mol = self.create_test_molecules()["hydrogen molecule"]
        msmd = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, target_states=2,
            spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess="hcore")
        dm = torch.einsum('ss...->...', msmd.density_matrices_ao()).detach().requires_grad_(True)
        hartree = hamiltonian_mod.HartreeFunctionalAO(mol)

        def scalar_energy(x):
            return torch.trace(hartree(x))

        energy = scalar_energy(dm)
        energy.backward()
        idx = (0, 0, 0, 0)
        h = 1.0e-5
        dm_plus = dm.detach().clone()
        dm_minus = dm.detach().clone()
        dm_plus[idx] += h
        dm_minus[idx] -= h
        finite_diff = (scalar_energy(dm_plus) - scalar_energy(dm_minus)) / (2 * h)
        torch.testing.assert_close(dm.grad[idx], finite_diff, rtol=1.0e-4, atol=1.0e-5)

    def test_torch_lbfgs_optimizer_smoke_preserves_target_orthonormality(self):
        mol = self.create_test_molecules()["hydrogen molecule"]
        msmd = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, target_states=2,
            spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess="hcore")
        hamiltonian = HamiltonianTargetStateLMDA(mol, hartree_backend="ao")
        energies, optimized = minimize_subspace_energy(
            hamiltonian,
            msmd,
            optimizer="torch_lbfgs",
            maxiter=2,
            gtol=1.0e-8,
        )

        self.assertEqual(energies.size(), torch.Size([msmd.number_of_states]))
        self.assertLess(optimized.target_orthonormality_error().item(), 1.0e-10)
        self.assertLess(optimized.orbital_orthonormality_error().item(), 1.0e-8)
        self.assertEqual(optimized.optimizer_telemetry["numpy_parameter_transfers"], 0)
        self.assertEqual(optimized.optimizer_telemetry["numpy_gradient_transfers"], 0)
        self.assertGreater(optimized.optimizer_telemetry["closure_calls"], 0)

    def test_wrapped_optimizer_reports_numpy_transfers(self):
        param = torch.nn.Parameter(torch.tensor([0.25], dtype=torch.double))
        optimizer = WrappedOptimizer([param], maxiter=5, gtol=1.0e-12)

        def closure():
            optimizer.zero_grad()
            return torch.sum(param * param)

        optimizer.step(closure)
        self.assertGreater(optimizer.telemetry.numpy_parameter_transfers, 0)
        self.assertGreater(optimizer.telemetry.numpy_gradient_transfers, 0)

    def test_phase_f_validation_artifact_writer_smoke(self):
        script_path = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "target_state_phase_f_validation.py"
        spec = importlib.util.spec_from_file_location("target_state_phase_f_validation", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = {
            "molecule": "H2",
            "coordinate_label": "R_HH_Angstrom",
            "coordinate_value": 0.7,
            "geometry_id": "H2_R_0.700",
            "charge": 0,
            "spin": 0,
            "basis": "sto-3g",
            "active_space": "CAS(2,2)",
            "target_states": 2,
            "functional": "LMDA(LDA_X + LDA_C_CHACHIYO)",
            "grid_level": 1,
            "hartree_backend": "ao",
            "optimizer": "torch_lbfgs",
            "device": "cpu",
            "runtime_seconds": 0.01,
            "energies_hartree": [-1.0, -0.5],
            "excitations_ev": [0.0, 13.6],
            "diagnostics": {
                "target_orthonormality_error": 0.0,
                "orbital_orthonormality_error": 0.0,
            },
            "optimizer_telemetry": {},
        }
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = module.write_artifacts([result], pathlib.Path(tmpdir))
            for path in paths:
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)

    def test_ao_grid_cache_uses_lmda_derivative_level(self):
        """LDA/LMDA Hamiltonians should cache AO values with deriv=0 only."""
        mol = self.create_test_molecules()["hydrogen molecule"]
        msmd = MultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, guess="hcore", spin_symmetry=False, spin_type=SpinType.POLARIZED)
        hamiltonian = HamiltonianSemilocal(
            mol,
            exchange_functional=lda_x_dirac,
            correlation_functional=lda_c_chachiyo,
            spin_type=SpinType.POLARIZED,
            grid_level=1,
            grid_chunks=2,
        )
        self.assertFalse(hamiltonian.need_gradient)
        self.assertFalse(hamiltonian.need_laplacian)

        spin_dm = msmd.density_matrices_ao()
        original_eval_ao = hamiltonian_mod.numint.eval_ao

        def eval_ao_spy(*args, **kwargs):
            return original_eval_ao(*args, **kwargs)

        with unittest.mock.patch.object(hamiltonian_mod.numint, "eval_ao", side_effect=eval_ao_spy) as patched_eval_ao:
            batches_first = hamiltonian._get_ao_grid_cache(dtype=spin_dm.dtype, device=spin_dm.device)
            batches_second = hamiltonian._get_ao_grid_cache(dtype=spin_dm.dtype, device=spin_dm.device)

        self.assertIs(batches_first, batches_second)
        self.assertEqual(patched_eval_ao.call_count, hamiltonian.grid_chunks)
        for call in patched_eval_ao.call_args_list:
            self.assertEqual(call.kwargs["deriv"], 0)
        for batch in batches_first:
            self.assertIsNone(batch.grad_ao_value)
            self.assertIsNone(batch.lapl_ao_value)

    def test_auto_grid_chunks_respects_memory_budget(self):
        mol = self.create_test_molecules()["hydrogen molecule"]
        msmd = MultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, guess="hcore", spin_symmetry=False, spin_type=SpinType.POLARIZED)
        hamiltonian = HamiltonianSemilocal(
            mol,
            spin_type=SpinType.POLARIZED,
            grid_level=1,
            grid_chunks="auto",
        )
        dtype = torch.double
        device = torch.device("cpu")
        bytes_per_grid = hamiltonian._estimate_bytes_per_grid_point(dtype, msmd.number_of_states)
        ngrids = len(hamiltonian.grids.coords)

        with unittest.mock.patch.object(hamiltonian, "_available_memory_bytes", return_value=bytes_per_grid):
            chunks_low_memory = hamiltonian._resolve_grid_chunks(dtype, device, msmd)
        with unittest.mock.patch.object(hamiltonian, "_available_memory_bytes", return_value=bytes_per_grid * ngrids * 2):
            chunks_high_memory = hamiltonian._resolve_grid_chunks(dtype, device, msmd)

        self.assertGreaterEqual(chunks_low_memory, 1)
        self.assertEqual(chunks_high_memory, 1)
        self.assertGreaterEqual(chunks_low_memory, chunks_high_memory)

    def test_auto_grid_chunks_hamiltonian_matches_explicit_chunks(self):
        mol = self.create_test_molecules()["hydrogen molecule"]
        msmd = MultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, guess="hcore", spin_symmetry=False, spin_type=SpinType.POLARIZED)
        hamiltonian_ref = HamiltonianSemilocal(
            mol,
            spin_type=SpinType.POLARIZED,
            grid_level=1,
            grid_chunks=2,
        )
        hamiltonian_auto = HamiltonianSemilocal(
            mol,
            spin_type=SpinType.POLARIZED,
            grid_level=1,
            grid_chunks="auto",
            max_grid_points_per_chunk=max(1, len(hamiltonian_ref.grids.coords) // 2),
        )
        H_ref = hamiltonian_ref(msmd)
        H_auto = hamiltonian_auto(msmd)
        self.assertIsNotNone(hamiltonian_auto.resolved_grid_chunks)
        self.assertGreaterEqual(hamiltonian_auto.resolved_grid_chunks, 1)
        torch.testing.assert_close(H_auto, H_ref)

    def test_minimize_kohn_sham_energy_h2(self):
        """
        Minimize energy of a Kohn-Sham determinant starting from
        a matrix density with core hamiltonian guess.
        """
        # Test molecule
        mol = pyscf.gto.M(
            atom = 'H 0 0 -0.35; H 0 0 0.35',
            basis = '6-31g',
            charge = 0,
            spin = 0)

        msmd = MultistateMatrixDensityKohnSham.from_guess(mol, guess="hcore")
        for kinetic_functional in kinetic_functionals:
            with self.subTest(kinetic_functional=str(kinetic_functional)):
                hamiltonian = HamiltonianSemilocal(mol, kinetic_functional=kinetic_functional)
                self.check_minimize_subspace_energy(msmd, hamiltonian)

    def test_minimize_cas_energy_h2(self):
        """
        Minimize subspace energy of a (2,2) complete active space starting from
        a matrix density with core hamiltonian guess.

        TODO: This long test should be broken up in many smaller tests.
        """
        # Test molecule
        mol = pyscf.gto.M(
            atom = 'H 0 0 -0.35; H 0 0 0.35',
            basis = '6-31g',
            charge = 0,
            spin = 0)

        devices = ['cpu']
        if torch.cuda.is_available():
            # Run tests of GPU, too
            devices.append('cuda')
        else:
            print("CUDA not available, tests are only run on CPU.")

        # Run tests of available devices.
        for device in devices:
            for spin_symmetry in [True, False]:
                for max_level in [1, numpy.inf]:
                    # Loop over available kinetic funtional
                    for kinetic_functional in [None]:
                        # Name of kinetic functional
                        if kinetic_functional is None:
                            kinetic_functional_name = "AO"
                        else:
                            kinetic_functional_name = kinetic_functional.__class__.__name__
                        # Loop over types of xc-functionals
                        for xc_type in ["lda"]:
                            # Loop over available xc-functionals
                            for exchange_functional in exchange_functionals[xc_type]:
                                for correlation_functional in correlation_functionals[xc_type]:
                                    # Whether to calculated the exchange energy from the total
                                    # density, separately for each spin type or from the spin supermatrix.
                                    for spin_type in SpinType:
                                        with self.subTest(
                                            device=device,
                                            spin_symmetry=spin_symmetry,
                                            max_level=max_level,
                                            t=kinetic_functional_name,
                                            x=exchange_functional.__name__,
                                            c=correlation_functional.__name__,
                                            spin_type=spin_type
                                        ):
                                            msmd = MultistateMatrixDensityCAS.from_guess(
                                                mol, 2, 2,
                                                spin_symmetry=spin_symmetry,
                                                spin_type=spin_type,
                                                max_level=max_level,
                                                guess="hcore"
                                            )
                                            hamiltonian = HamiltonianSemilocal(
                                                mol,
                                                kinetic_functional=kinetic_functional,
                                                exchange_functional=exchange_functional,
                                                correlation_functional=correlation_functional,
                                                spin_type=spin_type
                                            )
                                            self.check_minimize_subspace_energy(msmd, hamiltonian, device=device)

    def test_minimize_cas_energy_h2_reduced_subspace_average(self):
        """
        Minimize subspace energy of a (2,2) complete active space.

        The number of states to average over is varied.
        """
        # Test molecule
        mol = pyscf.gto.M(
            atom = 'H 0 0 -0.35; H 0 0 0.35',
            basis = '6-31g',
            charge = 0,
            spin = 0)

        for spin_symmetry in [True, False]:
            for max_level in [1, numpy.inf]:
                for state_average in [1,2,3,4]:
                    for spin_type in [SpinType.UNPOLARIZED]:
                        with self.subTest(
                            spin_symmetry=spin_symmetry,
                            max_level=max_level,
                            state_average=state_average,
                            spin_type=spin_type,
                        ):
                            msmd = MultistateMatrixDensityCAS.from_guess(
                                mol, 2, 2,
                                spin_symmetry=spin_symmetry,
                                spin_type=spin_type,
                                max_level=max_level,
                                guess="hcore"
                            )
                            xc_functional = LDA(mol)
                            hamiltonian = HamiltonianSemilocal(
                                mol,
                                exchange_functional = xc_functional.exchange,
                                correlation_functional = xc_functional.correlation,
                                spin_type = spin_type
                            )
                            self.check_minimize_subspace_energy(
                                msmd, hamiltonian,
                                state_average=min(state_average, msmd.number_of_states)
                            )

    def check_minimize_subspace_energy(
        self,
        msmd: MultistateMatrixDensity,
        hamiltonian: Hamiltonian,
        state_average=None,
        device='cpu'
    ):
        """
        Solve for lowest few electronic states by minimizing the subspace energy
        and check that at the minimum the gradient on the parameters of the
        matrix density is zero.

        :param state_average: How many states should be averaged over?
            None means to include all states in the subspace
        :type state_average: int or None
        """
        # Move matrix density to CPU or GPU.
        msmd.to(device=device)

        # Minimize the state-averaged energy
        energies, msmd = minimize_subspace_energy(hamiltonian, msmd,
            state_average=state_average,
            # show how energy and gradient norm decreases during minimization.
            debug=1
        )

        # check that Hamiltonian is diagonal in the basis of eigenstates
        H = hamiltonian(msmd)
        torch.testing.assert_close(H, torch.diag(energies))

        # Check that the matrix density is a stationary point, i.e. that the
        # gradients on the parameters vanish.
        msmd.zero_grad()
        H = hamiltonian(msmd)
        # Average energy of all electronic states, weighing states with their spin multiplicity
        # E = 1/N tr(H) = 1/N ∑ᵢ H[D(r)]ᵢᵢ
        #   = ∑ᵢ (2 Sᵢ + 1) H[D(r)]ᵢᵢ / (∑ⱼ (2 Sⱼ + 1))
        # In INVARIANT calculations, all components of the multiplet are present,
        # and the state weights are set to 1.
        with torch.no_grad():
            # The weight is an integer, it does not require gradients.
            weights = msmd.state_weights()
        subspace_energy = MF.trace_average(H, weights=weights, subspace_dim=state_average)
        # compute gradients
        subspace_energy.backward()

        # Check that gradients are close to zero.
        for param in msmd.parameters():
            if torch.isnan(param.grad).all():
                # `param` is most likely a constant. Taken the gradient of a constant gives NaN
                warnings.warn(
                    "Gradients on parameter are all NaN. The parameter is probably a constant."
                )
            else:
                torch.testing.assert_close(param.grad, torch.zeros_like(param.grad), atol=1.0e-5, rtol=1.0e-5)

    def check_lda_kohn_sham_energy_vs_pyscf(self, mol):
        """
        For a single Kohn-Sham state the same total energy should be obtained as with pyscf's RKS.
        """
        assert mol.spin == 0, (
            "MultistateMatrixDensityKohnSham only works for closed-shell singlet states"
        )
        # Minimize energy of a Kohn-Sham determinant starting from a Hcore guess matrix density.
        msmd = MultistateMatrixDensityKohnSham.from_guess(mol, guess="hcore")
        hamiltonian = HamiltonianSemilocal(
            mol,
            # compute kinetic energy from wavefunctions
            kinetic_functional = None,
            exchange_functional = lda_x_dirac,
            correlation_functional = lda_c_chachiyo
        )

        # Minimize the state-averaged energy. There is only a single state, so this is the same
        # as minimizing the ground state energy.
        energies, msmd = minimize_subspace_energy(hamiltonian, msmd)

        kohn_sham_energy = energies[0].detach().numpy()

        # pyscf should give the same energy.
        rks = pyscf.dft.RKS(mol)
        rks.xc = 'LDA_X,LDA_C_CHACHIYO'
        rks.verbose = 0
        rks.kernel()
        kohn_sham_energy_pyscf = rks.e_tot

        self.assertAlmostEqual(kohn_sham_energy, kohn_sham_energy_pyscf, places=6)

    def test_minimize_subspace_energy_can_return_convergence_info(self):
        mol = self.create_test_molecules()['hydrogen molecule']
        msmd = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, target_states=2,
            spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess='hcore')
        hamiltonian = HamiltonianTargetStateLMDA(mol, hartree_backend="ao")

        energies, optimized, info = minimize_subspace_energy(
            hamiltonian,
            msmd,
            optimizer="torch_lbfgs",
            maxiter=1,
            convergence={"gtol": 1.0e6, "ftol": 1.0e6},
            return_info=True,
        )

        self.assertEqual(len(energies), optimized.number_of_states)
        self.assertIn("optimizer_convergence", info)
        self.assertIn("optimizer_telemetry", info)
        self.assertIn("final_grad_norm", info["optimizer_convergence"])
        self.assertIn("history_tail", info["optimizer_convergence"])

    def test_lda_kohn_sham_energy_vs_pyscf(self):
        """
        Compare Kohn-Sham energies for LDA functionals with pyscf
        """
        for name, mol in tqdm(self.create_test_molecules_closed_shell().items()):
            with self.subTest(molecule=name):
                self.check_lda_kohn_sham_energy_vs_pyscf(mol)

    def check_composite_pure_xc_functional_vs_pyscf(
        self, mol, xc_functional_class, xc_code, spin_type=SpinType.UNPOLARIZED):
        """
        For a single Kohn-Sham state the same total energy should be obtained as with pyscf's RKS.
        """
        assert mol.spin == 0, (
            "MultistateMatrixDensityKohnSham only works for closed-shell singlet states"
        )
        xc_functional = xc_functional_class(mol)
        # Pure functionals do not have any exact exchange.
        self.assertEqual(xc_functional.exact_exchange, None)

        # Minimize energy of a Kohn-Sham determinant starting from a Hcore guess matrix density.
        msmd = MultistateMatrixDensityKohnSham.from_guess(mol, guess="hcore")
        hamiltonian = HamiltonianSemilocal(
            mol,
            # compute kinetic energy from wavefunctions
            kinetic_functional = None,
            exchange_functional = xc_functional.exchange,
            correlation_functional = xc_functional.correlation,
            exact_exchange_functional = xc_functional.exact_exchange,
            spin_type = spin_type
        )

        # Minimize the state-averaged energy. There is only a single state, so this is the same
        # as minimizing the ground state energy.
        energies, msmd = minimize_subspace_energy(hamiltonian, msmd)

        kohn_sham_energy = energies[0].detach().numpy()

        if 'MGGA' in xc_code:
            # For BR89LYP we just check that the code runs through without errors.
            # We cannot compare the results with pyscf, since Laplacians are not implemented
            # for Meta GGAs in pyscf.
            return
        # pyscf should give the same energy.
        rks = pyscf.dft.RKS(mol)
        rks.xc = xc_code
        rks.verbose = 0
        rks.kernel()
        kohn_sham_energy_pyscf = rks.e_tot

        self.assertAlmostEqual(kohn_sham_energy, kohn_sham_energy_pyscf, places=6)

    def test_fused_pure_xc_functional_matches_separate_xc_path(self):
        mol = pyscf.gto.M(
            atom='H 0 0 -0.35; H 0 0 0.35',
            basis='6-31g',
            charge=0,
            spin=0)
        msmd = MultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, spin_symmetry=False, spin_type=SpinType.UNPOLARIZED, guess='hcore')
        xc_functional = LDA(mol)
        h_separate = HamiltonianSemilocal(
            mol,
            exchange_functional=xc_functional.exchange,
            correlation_functional=xc_functional.correlation,
            spin_type=SpinType.UNPOLARIZED)
        h_fused = HamiltonianSemilocal(
            mol,
            exchange_functional=None,
            correlation_functional=None,
            exchange_correlation_functional=xc_functional.exchange_correlation,
            spin_type=SpinType.UNPOLARIZED)
        torch.testing.assert_close(
            h_fused(msmd), h_separate(msmd), rtol=1.0e-8, atol=1.0e-8)

    def test_fused_pure_xc_functional_rejects_spin_resolved_paths(self):
        mol = pyscf.gto.M(
            atom='H 0 0 -0.35; H 0 0 0.35',
            basis='6-31g',
            charge=0,
            spin=0)
        xc_functional = LDA(mol)
        for spin_type in [SpinType.POLARIZED, SpinType.NONCOLLINEAR]:
            with self.subTest(spin_type=spin_type):
                msmd = MultistateMatrixDensityCAS.from_guess(
                    mol, 2, 2, spin_symmetry=False, spin_type=spin_type, guess='hcore')
                h_fused = HamiltonianSemilocal(
                    mol,
                    exchange_functional=None,
                    correlation_functional=None,
                    exchange_correlation_functional=xc_functional.exchange_correlation,
                    spin_type=spin_type)
                with self.assertRaises(NotImplementedError):
                    h_fused(msmd)

    def test_composite_pure_xc_functional_vs_pyscf(self):
        """
        Compare Kohn-Sham energies for pure composite xc-functional with pyscf
        """
        mol = self.create_test_molecules()['hydrogen molecule']
        for spin_type in [SpinType.UNPOLARIZED, SpinType.POLARIZED, SpinType.NONCOLLINEAR]:
            for xc_functional_class, xc_code in [
                    (LDA, 'LDA_X,LDA_C_CHACHIYO'),
            ]:
                # For a single, closed-shell state the spin-polarized and unpolarized
                # calculations should give the same results.
                with self.subTest(
                    xc_functional=xc_functional_class.__name__,
                    spin_type=spin_type,
                ):
                    self.check_composite_pure_xc_functional_vs_pyscf(
                        mol, xc_functional_class, xc_code, spin_type=spin_type)

    def test_noncollinear_matches_polarized_collinear_limit(self):
        mol = pyscf.gto.M(
            atom='H 0 0 -0.35; H 0 0 0.35',
            basis='6-31g',
            charge=0,
            spin=0)
        msmd = MultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, spin_symmetry=False, spin_type=SpinType.POLARIZED, guess='hcore')
        xc_functional = LDA(mol)
        h_pol = HamiltonianSemilocal(
            mol,
            exchange_functional=xc_functional.exchange,
            correlation_functional=xc_functional.correlation,
            spin_type=SpinType.POLARIZED)
        h_noncol = HamiltonianSemilocal(
            mol,
            exchange_functional=xc_functional.exchange,
            correlation_functional=xc_functional.correlation,
            spin_type=SpinType.NONCOLLINEAR)
        H_pol = h_pol(msmd)
        H_noncol = h_noncol(msmd)
        torch.testing.assert_close(H_noncol, H_pol, rtol=1.0e-8, atol=1.0e-8)

    def test_deprecated_invariant_spin_types_raise(self):
        mol = pyscf.gto.M(
            atom='H 0 0 -0.35; H 0 0 0.35',
            basis='6-31g',
            charge=0,
            spin=0)
        for spin_type in [SpinType.INVARIANT, SpinType.INVARIANT_MIX]:
            with self.subTest(spin_type=spin_type):
                msmd = MultistateMatrixDensityCAS.from_guess(
                    mol, 2, 2, spin_symmetry=False, spin_type=spin_type, guess='hcore')
                hamiltonian = HamiltonianSemilocal(mol, spin_type=spin_type)
                with self.assertRaises(NotImplementedError):
                    hamiltonian(msmd)

    def check_basis_transformation_cas(self, msmd: MultistateMatrixDensityCAS):
        """
        Check that after transforming the matrix density with eigenvectors
        of the Hamiltonian, the Hamiltonian matrix becomes diagonal.
        """
        hamiltonian = HamiltonianSemilocal(msmd.mol)

        # Diagonalize Hamiltonian ...
        H = hamiltonian(msmd)
        eigvals, eigvecs = torch.linalg.eigh(H)

        # ... and transform optimized matrix density to the basis of the eigenstates
        msmd.basis_transformation(eigvecs.T)

        # Hamiltonian after basis transformation should be diagonal.
        H = hamiltonian(msmd)
        eigvals, eigvecs = torch.linalg.eigh(H)
        torch.testing.assert_close(H, torch.diag(eigvals))

    def test_basis_transformation_cas(self):
        """ Check that matrix density transforms as expected under a change of basis """
        # Test molecule
        mol = pyscf.gto.M(
            atom = 'H 0 0 -0.35; H 0 0 0.35',
            basis = '6-31g',
            charge = 0,
            spin = 0)

        for spin_symmetry in [True, False]:
            for spin_type in SpinType:
                for max_level in [1, 2, numpy.inf]:
                    with self.subTest(
                        spin_symmetry=spin_symmetry,
                        spin_type=spin_type,
                        max_level=max_level
                    ):
                        msmd = MultistateMatrixDensityCASTest.create_random_matrix_density_cas(
                            mol, 2, 2,
                            spin_symmetry=spin_symmetry, spin_type=spin_type, max_level=max_level
                        )
                        self.check_basis_transformation_cas(msmd)

    def test_spin_invariant_multiplet_degeneracy_lda(self):
        """
        Check that all components of the triplet states in the CAS(2,2) of H2
        are degenerate but different in energy from the open-shell singlet state:

            E(S0) ≠ E(S1) ≠ E(S2) ≠ E(T1,Sz=-1) = E(T1,Sz=0) = E(T1,Sz=+1)
        """
        # Hydrogen molecule
        mol = pyscf.gto.M(
            atom = 'H 0 0 -0.35; H 0 0 0.35',
            basis = '6-31g',
            charge = 0,
            spin = 0)

        msmd = MultistateMatrixDensityCASTest.create_random_matrix_density_cas(
            # 2 electrons in 2 orbitals gives rise to 3 singlet states and 3 triplet states
            mol, 2, 2,
            # any spin S²
            spin_symmetry=False,
            # any spin projection Sz=-S,...,+S
            spin_type=SpinType.INVARIANT,
        )

        xc_functional = LDA(msmd.mol)
        hamiltonian = HamiltonianSemilocal(
            msmd.mol,
            # exchange, correlation and exact exchange parts of hybrid functional.
            exchange_functional = xc_functional.exchange,
            correlation_functional = xc_functional.correlation,
            exact_exchange_functional = xc_functional.exact_exchange,
            spin_type = SpinType.INVARIANT
        )

        # Diagonalize Hamiltonian ...
        H = hamiltonian(msmd)
        eigvals, eigvecs = torch.linalg.eigh(H)

        # ... and transform optimized matrix density to the basis of the eigenstates
        msmd.basis_transformation(eigvecs.T)

        # Check that the energy levels show the correct degeneracies due to spin multiplets
        s2 = msmd.spin_s2_expectation().detach().cpu().numpy()
        s2 = numpy.round(s2, decimals=2)
        energies = eigvals.detach().cpu().numpy()
        # Check for degeneracy within 1.0e-8
        energies = numpy.round(energies, decimals=8)
        # The energies of the three singlet states should be all different
        singlet_energies = energies[s2 == 0.0]
        self.assertEqual( len(numpy.unique(singlet_energies)), 3 )
        # The triplet energies should be all the same
        triplet_energies = energies[s2 == 2.0]
        self.assertEqual( len(numpy.unique(triplet_energies)), 1 )
        # The triplet energies should be different from any singlet energy
        # E(S0) ≠ E(S1) ≠ E(S2) ≠ E(T1,Sz=-1) = E(T1,Sz=0) = E(T1,Sz=+1)
        self.assertEqual( len(numpy.unique(energies)), 4 )

    def check_matrix_elements_chunking(
        self,
        hamiltonian_ref: HamiltonianSemilocal,
        msmd: MultistateMatrixDensity
    ):
        """
        The Hamiltonian matrix elements are computed by splitting the integration
        grid into chunks so that each chunk fits into memory. This test checks
        that the final H matrix does not depend on the number of chunks.
        """
        # overwrite method for computing matrix elements
        hamiltonian_ref.matrix_elements = types.MethodType(
            _matrix_elements_single_chunk, hamiltonian_ref)
        # reference Hamiltonian
        H_ref = hamiltonian_ref(msmd)

        for chunks in [1,2,3,4]:
            # copy of hamiltonian_ref with chunking
            hamiltonian = HamiltonianSemilocal(
                hamiltonian_ref.mol,
                kinetic_functional = hamiltonian_ref.kinetic,
                exchange_functional = hamiltonian_ref.exchange,
                correlation_functional = hamiltonian_ref.correlation,
                grid_level = hamiltonian_ref.grid_level,
                grid_chunks = chunks
            )
            H = hamiltonian(msmd)
            torch.testing.assert_close(H, H_ref)

    def test_matrix_elements_chunking(self):
        """
        Test that hamiltonian does not depend on how many chunks are used to compute it.
        """
        mol = self.create_test_molecules()["hydrogen molecule"]

        devices = ['cpu']
        if torch.cuda.is_available():
            # Run tests of GPU, too
            devices.append('cuda')
        else:
            print("CUDA not available, tests are only run on CPU.")

        for device in devices:
            # Random matrix density
            for msmd in tqdm(self.create_random_matrix_densities(mol)):
                msmd.to(device)
                for kinetic_functional in kinetic_functionals:
                    with self.subTest(device=device, kinetic_functional=str(kinetic_functional)):
                        hamiltonian = HamiltonianSemilocal(mol, kinetic_functional=kinetic_functional)
                        self.check_matrix_elements_chunking(hamiltonian, msmd)


if __name__ == "__main__":
    unittest.main()
