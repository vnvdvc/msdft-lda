#!/usr/bin/env python
# coding: utf-8
from abc import ABC, abstractmethod
import numpy
import numpy.linalg as la
import numpy.testing
import tempfile
import pyscf.dft

import torch
from torch import Size
from torch.autograd import gradcheck
import torch.linalg
import torch.testing
from tqdm import tqdm
import unittest
import warnings
from unittest import mock

from mlmsdft.dft.active_space import ActiveSpaceError
from mlmsdft.dft.density import AOGridBatch
from mlmsdft.dft.density import antisymmetric_matrix
from mlmsdft.dft.density import MultistateMatrixDensity
from mlmsdft.dft.density import MultistateMatrixDensityCAS
from mlmsdft.dft.density import MultistateMatrixDensityKohnSham
from mlmsdft.dft.density import TargetStateMultistateMatrixDensityCAS
import mlmsdft.dft.density as density_mod
from mlmsdft.dft.density import orbital_guess
from mlmsdft.dft.density import reorder_active_orbitals
from mlmsdft.dft.spin import SpinType

from nn.test_functional import random_orthogonal_matrix
from nn.test_functional import random_tensor

from dft.fixture import FixtureMixin


class BaseTestMultistateMatrixDensity(FixtureMixin, ABC):
    @abstractmethod
    def create_random_matrix_density(self, mol):
        """
        Multistate matrix density with random parameters.
        """
        pass

    def check_integrals(self, msmd: MultistateMatrixDensity):
        """
        check that the state density integrates to the correct number of electrons
        and that the transition density integrates to 0.

        :param msmd: The multistate matrix density to be tested.
        :type msmd: MultistateMatrixDensity
        """
        # integration grid
        grids = pyscf.dft.gen_grid.Grids(msmd.mol)
        grids.level = 8
        grids.build()

        D, _, _ = msmd.evaluate(grids.coords)
        weights = torch.from_numpy(grids.weights).to(
            dtype=D.dtype, device=D.device)

        # Integral over space and spin ∑_s ∫ Dˢˢ(r) dr
        integrals = torch.einsum('r,ssrij->ij', weights, D)

        nstate = msmd.number_of_states
        for i in range(0, nstate):
            for j in range(0, nstate):
                with self.subTest(i=i, j=j):
                    if i == j:
                        # State densities should integrate to the number of electrons.
                        self.assertAlmostEqual(msmd.number_of_electrons, integrals[i,i].item(), places=3)
                    else:
                        # Integrating the transition density, just gives the overlap between
                        # the states, which should be zero for different eigenstates.
                        self.assertAlmostEqual(0.0, integrals[i,j].item(), places=5)

    def test_integrals(self):
        """ Check integrals of D(r) for all test molecules """
        for name, mol in tqdm(self.create_test_systems().items()):
            with self.subTest(molecule=name, guess="random"):
                # Random matrix density
                msmd = self.create_random_matrix_density(mol)
                self.check_integrals(msmd)

    def check_gradient_and_laplacian(self, mol):
        """
        compare analytical gradients ∇D(r) and the Laplacian ∇²D(r)
        with numerical ones from finite differences
        """
        # Example density.
        msmd = self.create_random_matrix_density(mol)
        # Gradients are checked at random coordinates.
        ncoord = 100
        # The hardcoded seed ensures that the same random numbers are used
        # every time the test is run. Otherwise the test fails occasionally
        # when the threshold is is too tight.
        random_number_generator = numpy.random.default_rng(seed=2345)
        coords = 5.0*(random_number_generator.random((ncoord,3)) - 0.5)

        # Wrapper function that converts torch Tensors to numpy arrays.
        def msmd_evaluate(coords):
            tensors = msmd.evaluate(coords)
            tensors = [t.detach().numpy() for t in tensors]
            # D, grad_D, lapl_D
            return tensors

        # Analytical gradients and Laplacian of D
        D, grad_D, lapl_D = msmd_evaluate(coords)

        # Numerical gradients of D and tr(D)
        grad_D_numerical = numpy.zeros_like(grad_D)
        lapl_D_numerical = numpy.zeros_like(lapl_D)

        # dD/dx = [D(x+h) - D(x-h)]/(2 h)
        h = 0.001
        for xyz in [0,1,2]:
            # unit vector in the x,y or z-direction
            unit_vector = numpy.zeros(3)
            unit_vector[xyz] = 1.0

            # D(r+h*e_x)
            D_plus, _, _ = msmd_evaluate(coords + h*unit_vector)
            # D(r-h*e_x)
            D_minus, _, _ = msmd_evaluate(coords - h*unit_vector)

            # finite difference gradient
            grad_D_numerical[:,:,:,xyz,:,:] = (D_plus - D_minus)/(2*h)

            # Add finite difference approximation for second derivative to
            # numerical Laplacian.
            lapl_D_numerical += (D_plus - 2*D + D_minus)/pow(h,2)

        # Compare analytical and numerical gradients
        with self.subTest("gradient of D(r)"):
            # relative error |∇D-∇D(numerical)|/|∇D(numerical)|
            relative_error = (
                la.norm(grad_D - grad_D_numerical)/la.norm(grad_D_numerical))
            self.assertLess(relative_error, 1.0e-3)
            numpy.testing.assert_almost_equal(grad_D, grad_D_numerical, decimal=2)
        with self.subTest("Laplacian of D(r)"):
            # relative error |∇²D-∇²D(numerical)|/|∇²D(numerical)|
            relative_error = (
                la.norm(lapl_D - lapl_D_numerical)/la.norm(lapl_D_numerical))
            self.assertLess(relative_error, 1.0e-3)
            numpy.testing.assert_almost_equal(lapl_D, lapl_D_numerical, decimal=2)

    def test_gradient_and_laplacian(self):
        """ Compare numerical and analytical gradient ∇D(r) and Laplacian ∇²D(r) for all test molecules """
        for name, mol in tqdm(self.create_test_molecules().items()):
            with self.subTest(molecule=name):
                self.check_gradient_and_laplacian(mol)

    def check_densities_on_grid(self, msmd):
        """
        Compare values of (transition) densities on a grid with pyscf
        """
        # Densities are checked at random coordinates.
        ncoord = 100
        # The hardcoded seed ensures that the same random numbers are used
        # every time the test is run. Otherwise the test fails occasionally
        # when the threshold is is too tight.
        random_number_generator = numpy.random.default_rng(seed=567)
        coords = 2.0*(random_number_generator.random((ncoord,3)) - 0.5)

        # matrix density D(r)
        D, _, _ = msmd.evaluate(coords)
        # convert torch tensors to numpy array
        D = D.detach().numpy()

        # construct matrix density in AO basis, Dᵦᵧᵢⱼ
        dm = msmd.density_matrices_ao().detach().numpy()
        # Atomic orbitals on grid.
        ao_value = pyscf.dft.numint.eval_ao(msmd.mol, coords, deriv=0)

        nstate = msmd.number_of_states
        for i in range(0, nstate):
            for j in range(0, nstate):
                # loop over electronic spin s
                for s in [0,1]:
                    # loop over electronic spin t
                    for t in [0,1]:
                        with self.subTest(i=i, j=j, s=s, t=t):
                            # (transition) density matrix computed using pyscf
                            Dij_ref = pyscf.dft.numint.eval_rho(
                                msmd.mol, ao_value, dm[s,t,:,:,i,j],
                                xctype='LDA'
                            )
                            numpy.testing.assert_allclose(D[s,t,:,i,j], Dij_ref)

    def test_densities_on_grid(self):
        """ Compare D(r) with pyscf's implementation """
        for name, mol in tqdm(self.create_test_systems().items()):
            with self.subTest(molecule=name, guess="random"):
                # Random matrix density
                msmd = self.create_random_matrix_density(mol)
                self.check_densities_on_grid(msmd)

    def check_integrals_density_matrices_ao(self, msmd):
        """
        Check that the density matrices in the AO basis are normalized properly.

            ∑ₛ ∫ Dˢˢᵢⱼ(r) dr = ∑ᵦ ∑ᵧ ∑ₛ Dˢˢᵦᵧᵢⱼ ∫ 𝛘ᵦ(r) 𝛘ᵧ(r) = ∑ᵦ ∑ᵧ Dᵦᵧᵢⱼ Sᵦᵧ = δᵢⱼ
        """
        # construct Dˢᵗᵦᵧᵢⱼ
        dm = msmd.density_matrices_ao()
        # AO overlap matrix S
        overlap_matrix = msmd.mol.intor_symmetric('int1e_ovlp')
        S = torch.from_numpy(overlap_matrix).to(dtype=dm.dtype, device=dm.device)

        # Integrals over space ∑ₛ ∫ Dˢˢᵢⱼ(r) dr = ∑ᵦ ∑ᵧ Dᵦᵧᵢⱼ Sᵦᵧ
        integrals = torch.einsum('ssmnij,mn->ij', dm, S)

        nstate = msmd.number_of_states
        for i in range(0, nstate):
            for j in range(0, nstate):
                with self.subTest(i=i, j=j):
                    if i == j:
                        # State densities should integrate to the number of electrons.
                        self.assertAlmostEqual(msmd.number_of_electrons, integrals[i,i].item(), places=3)
                    else:
                        # Integrating the transition density, just gives the overlap between
                        # the states, which should be zero for different eigenstates.
                        self.assertAlmostEqual(0.0, integrals[i,j].item(), places=5)

    def test_integrals_density_matrices_ao(self):
        """
        Check that the density matrices in the AO basis are normalized properly.

            ∑ₛ ∫ Dˢˢᵢⱼ(r) dr = ∑ᵦ ∑ᵧ ∑ₛ Dˢˢᵦᵧᵢⱼ ∫ 𝛘ᵦ(r) 𝛘ᵧ(r) = ∑ᵦ ∑ᵧ Dᵦᵧᵢⱼ Sᵦᵧ = δᵢⱼ
        """
        # Choose test system
        mol = self.create_test_systems()['silicon crystal']
        # Random matrix density
        msmd = self.create_random_matrix_density(mol)
        self.check_integrals_density_matrices_ao(msmd)

    def check_basis_transformation(self, mol):
        """
        Check that the matrix density parameters X are transformed
        correctly, so that L D[X](r) Lᵀ = D[X'].
        """
        # Random matrix density D
        msmd = self.create_random_matrix_density(mol)
        # Random orthogonal transformation L in the electronic state space.
        L = random_orthogonal_matrix(msmd.number_of_states)

        # Evaluate matrix density at random positions.
        ncoord = 100
        # The hardcoded seed ensures that the same random numbers are used
        # every time the test is run. Otherwise the test fails occasionally
        # when the threshold is is too tight.
        random_number_generator = numpy.random.default_rng(seed=857)
        coords = 5.0*(random_number_generator.random((ncoord,3)) - 0.5)
        # Evaluate Dˢ[X](r)
        D, _, _ = msmd.evaluate(coords)
        # Transform matrix density, Dˢᵗ'(r) = L Dˢᵗ(r) Lᵀ
        Dp = torch.einsum('ik,strkl,lj->strij', L, D, L.T)

        # Transform parameters of matrix density, in-place.
        msmd.basis_transformation(L)
        # Evaluate D[X'](r)
        DXp, _, _ = msmd.evaluate(coords)

        # Check that  D[X'] = L D[X](r) Lᵀ
        torch.testing.assert_close(DXp, Dp)

    def test_basis_transformation(self):
        """
        Check that transforming matrix density parameters gives
        the same result as transforming the matrix density.
        """
        for name, mol in tqdm(self.create_test_systems().items()):
            with self.subTest(molecule=name):
                self.check_basis_transformation(mol)

    def test_identify_device(self):
        """ Check that we can identify on which device the parameters of the matrix density are"""
        devices = ['cpu']
        if torch.cuda.is_available():
            # Run tests of GPU, too
            devices.append('cuda')
        else:
            print("CUDA not available, tests are only run on CPU.")

        # Test molecule
        mol = self.create_test_molecules()['hydrogen molecule']
        # Random matrix density
        msmd = self.create_random_matrix_density(mol)

        for device in devices:
            with self.subTest(device=device):
                msmd.to(device)
                self.assertEqual(msmd.device.type, device)


class TestMultistateMatrixDensityKohnSham(unittest.TestCase, BaseTestMultistateMatrixDensity):
    """ Test Kohn-Sham density """
    def create_random_matrix_density(self, mol):
        """
        Multistate matrix density with random parameters.
        """
        msmd = MultistateMatrixDensityKohnSham.from_guess(mol, guess="random")
        return msmd

    def test_antisymmetric_matrix(self):
        """ Check creation of antisymmetric matrix works """
        for n in range(1, 5):
            nrot = (n*(n-1))//2
            elements = torch.rand(nrot)
            A = antisymmetric_matrix(elements, n)
            # Check that A = -Aᵀ
            torch.testing.assert_close(A, -A.T)
            # Check individual matrix elements
            k = 0
            for i in range(0, n):
                self.assertEqual(A[i,i], 0.0)
                for j in range(i+1, n):
                    self.assertEqual(A[i,j], elements[k])
                    self.assertEqual(A[j,i], -elements[k])
                    k += 1

    def test_from_guess(self):
        """
        Test factory method for creating initial guess of matrix density.
        """
        for name, mol in tqdm(self.create_test_systems().items()):
            for guess in ["random", "hcore", "rohf"]:
                with self.subTest(molecule=name, guess=guess):
                    msmd = MultistateMatrixDensityKohnSham.from_guess(mol, guess=guess)
                    self.check_integrals_density_matrices_ao(msmd)
                    # Check that orbital rotations are registered as parameters.
                    parameters = list(msmd.parameters())
                    self.assertGreater(len(parameters), 0)
                    # Check that parameters require gradients
                    for name, param in msmd.named_parameters():
                        if name == "orbital_rotation_params":
                            self.assertEqual(param.requires_grad, True)
                        else:
                            self.assertEqual(param.requires_grad, False)

    def check_autograd(self, mol):
        """
        Compare analytical and numerical gradients of D(r) with respect to
        parameters of matrix density.
        """
        # Gradients are checked at random coordinates.
        ncoord = 100
        # The hardcoded seed ensures that the same random numbers are used
        # every time the test is run. Otherwise the test fails occasionally
        # when the threshold is is too tight.
        random_number_generator = numpy.random.default_rng(seed=789)
        coords = 5.0*(random_number_generator.random((ncoord,3)) - 0.5)

        msmd_0 = MultistateMatrixDensityKohnSham.from_guess(mol, guess="random")
        orbital_coefficients = msmd_0.orbital_coefficients()

        def wrapper_function(orbital_rotation_params):
            msmd = MultistateMatrixDensityKohnSham(
                mol, orbital_coefficients, orbital_rotation_params
            )
            # (D, grad_D, lapl_D)
            outputs = msmd.evaluate(coords)
            return outputs

        nao, nmo = orbital_coefficients.size()
        # Random orbital rotation parameters.
        nrot = (nmo*(nmo-1))//2
        orbital_rotation_params = random_tensor(Size([nrot]))
        orbital_rotation_params.requires_grad_(True)

        # Compare analytical gradients computed by backpropagation
        # finite difference approximation.
        inputs = (orbital_rotation_params,)
        gradcheck(wrapper_function, inputs)

    def test_autograd(self):
        """
        Check that automatic differentiation works for D(U;r) w/r/t U.
        """
        for name, mol in tqdm(self.create_test_systems().items()):
            with self.subTest(molecule=name):
                self.check_autograd(mol)

    def test_autograd_density_matrices_ao(self):
        """
        Compare analytical and numerical gradients of Dᵦᵧᵢⱼ with respect to
        parameters of matrix density.
        """
        # Choose test system
        mol = self.create_test_systems()['water']

        msmd_0 = MultistateMatrixDensityKohnSham.from_guess(mol, guess="random")
        orbital_coefficients = msmd_0.orbital_coefficients()

        def wrapper_function(orbital_rotation_params):
            msmd = MultistateMatrixDensityKohnSham(
                mol, orbital_coefficients, orbital_rotation_params
            )
            # density matrices Dᵦᵧᵢⱼ in AO basis
            output = msmd.density_matrices_ao()
            return output

        nao, nmo = orbital_coefficients.size()
        # Random orbital rotation parameters.
        nrot = (nmo*(nmo-1))//2
        orbital_rotation_params = random_tensor(Size([nrot]))
        orbital_rotation_params.requires_grad_(True)

        # Compare analytical gradients computed by backpropagation
        # finite difference approximation.
        inputs = (orbital_rotation_params,)
        gradcheck(wrapper_function, inputs)

    def test_spin_multiplicity(self):
        """ Check spin multiplicity of ground state is retrieved correctly. """
        # singlet
        mol = self.create_test_systems()['water']
        msmd = MultistateMatrixDensityKohnSham.from_guess(mol, guess="random")
        multiplicity = msmd.spin_multiplicity()
        self.assertEqual(multiplicity.tolist(), [1])
        # triplet
        mol = self.create_test_systems()['water']
        mol.spin = 2
        mol.build()
        msmd = MultistateMatrixDensityKohnSham.from_guess(mol, guess="random")
        multiplicity = msmd.spin_multiplicity()
        self.assertEqual(multiplicity.tolist(), [3])
        # Since only one Sz component of a spin multiple is calculated, its
        # energy has to be weighted by the spin multiplicity.
        weights = msmd.state_weights()
        self.assertEqual(multiplicity.tolist(), weights.tolist())

class TestMultistateMatrixDensityCAS(unittest.TestCase, BaseTestMultistateMatrixDensity):
    """ Test complete active space (CAS) matrix density """
    @classmethod
    def create_random_matrix_density(cls, mol):
        """
        Multistate matrix density with random parameters and default active space.
        """
        if mol.tot_electrons() % 2 == 0:
            norb = 2
            nelec = 2
        else:
            norb = 2
            nelec = 1
        return cls.create_random_matrix_density_cas(mol, norb, nelec, spin_symmetry=True)

    def test_evaluate_derivative_flags(self):
        """Check that evaluate requests only the AO derivatives it needs."""
        mol = self.create_test_molecules()["hydrogen molecule"]
        msmd = self.create_random_matrix_density(mol)
        coords = numpy.array([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.2],
            [-0.2, 0.3, 0.1],
        ])

        original_eval_ao = density_mod.numint.eval_ao

        def eval_ao_spy(*args, **kwargs):
            return original_eval_ao(*args, **kwargs)

        with mock.patch.object(density_mod.numint, "eval_ao", side_effect=eval_ao_spy) as patched_eval_ao:
            D0, grad_D0, lapl_D0 = msmd.evaluate(
                coords, need_gradient=False, need_laplacian=False)
            self.assertEqual(patched_eval_ao.call_args.kwargs["deriv"], 0)
            self.assertIsNone(grad_D0)
            self.assertIsNone(lapl_D0)

            D1, grad_D1, lapl_D1 = msmd.evaluate(
                coords, need_gradient=True, need_laplacian=False)
            self.assertEqual(patched_eval_ao.call_args.kwargs["deriv"], 1)
            self.assertIsNotNone(grad_D1)
            self.assertIsNone(lapl_D1)

            D2, grad_D2, lapl_D2 = msmd.evaluate(
                coords, need_gradient=False, need_laplacian=True)
            self.assertEqual(patched_eval_ao.call_args.kwargs["deriv"], 2)
            self.assertIsNotNone(grad_D2)
            self.assertIsNotNone(lapl_D2)

        torch.testing.assert_close(D0, D1)
        torch.testing.assert_close(D0, D2)

    def test_evaluate_from_ao_batch(self):
        """Check cached AO grid batches reproduce direct evaluate outputs."""
        mol = self.create_test_molecules()["hydrogen molecule"]
        msmd = self.create_random_matrix_density(mol)
        coords = numpy.array([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.2],
            [-0.2, 0.3, 0.1],
        ])
        dm_ao = msmd.density_matrices_ao()

        ao_value_all = pyscf.dft.numint.eval_ao(mol, coords, deriv=2)
        ao_value_all = torch.from_numpy(ao_value_all).to(dtype=dm_ao.dtype, device=dm_ao.device)
        batch = AOGridBatch(
            weights=None,
            ao_value=ao_value_all[0,:,:],
            grad_ao_value=ao_value_all[1:4,:,:],
            lapl_ao_value=ao_value_all[4,:,:] + ao_value_all[7,:,:] + ao_value_all[9,:,:],
        )

        direct = msmd.evaluate(
            coords, dm_ao=dm_ao, need_gradient=True, need_laplacian=True)
        cached = msmd.evaluate_from_ao_batch(
            batch, dm_ao=dm_ao, need_gradient=True, need_laplacian=True)

        for direct_tensor, cached_tensor in zip(direct, cached):
            torch.testing.assert_close(cached_tensor, direct_tensor)

    @classmethod
    def create_random_matrix_density_cas(
        cls, mol, norb: int, nelec: int,
        spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, max_level=numpy.inf
    ):
        """
        Multistate matrix density with random parameters.
        """
        # Only the MO coefficients are chosen randomly,
        # whereas the initial state coefficients, the orbital rotation
        # and state rotation matrices are set to the identity.
        msmd = MultistateMatrixDensityCAS.from_guess(
            mol, norb, nelec,
            spin_symmetry=spin_symmetry, spin_type=spin_type, max_level=max_level, guess="random"
        )
        # In order to obtain a truely random matrix density, the initial
        # state coefficients and orbital and state rotations have to be
        # randomized as well.

        # random initial state coefficients
        nstate = msmd.number_of_states
        # random antisymmetric orbital rotation matrix.
        R_ci = torch.randn(Size([nstate, nstate]), dtype=torch.double)
        # antisymmetrize,  R_ciᵀ = -R_ci
        R_ci = 0.5 * (R_ci - R_ci.T)
        state_coefficients = torch.matrix_exp(R_ci)

        # random orbital and state rotations
        orbital_rotation_params = torch.nn.Parameter(
            data=torch.randn_like(msmd.orbital_rotation_params).to(dtype=torch.double),
            requires_grad=True
        )
        state_rotation_params = torch.nn.Parameter(
            data=torch.randn_like(msmd.state_rotation_params).to(dtype=torch.double),
            requires_grad=True
        )

        msmd_random = MultistateMatrixDensityCAS(
            msmd.mol,
            msmd.active_space.norb,
            msmd.active_space.nelec,
            # MO coefficients are already random
            msmd.orbital_coefficients(),
            orbital_rotation_params,
            state_coefficients,
            state_rotation_params,
            spin_symmetry=msmd.spin_symmetry,
            spin_type=msmd.spin_type,
            max_level=msmd.active_space.max_level
        )

        return msmd_random

    def active_spaces_for_testing(self, mol, norb_max=4):
        """
        enumerate some active space (norb, nelec) for which tests are run.
        """
        # Assume nmo == nao
        nmo = mol.nao_nr()
        nelec = mol.tot_electrons()
        for norb in range(1, min(nmo, norb_max)+1):
            for neleca in range(1, min(nelec, norb)+1):
                for nelecb in range(0, min(neleca, norb, nelec-neleca)+1):
                    # number of orbitals that are always doubly occupied
                    ndouble = (nelec - (neleca+nelecb))//2
                    if norb > nmo - ndouble:
                        # not enough orbitals for active space
                        continue
                    unpaired_electrons = neleca-nelecb
                    if mol.spin == unpaired_electrons:
                        # specify only total number of electrons
                        yield norb, neleca+nelecb
                        # specify up and down electron separately
                        yield norb, (neleca, nelecb)

    def test_from_guess(self):
        """
        Test factory method for creating initial guess of matrix density.
        """
        for name, mol in tqdm(self.create_test_systems().items()):
            for (norb, nelec) in self.active_spaces_for_testing(mol):
                for guess in ["random", "hcore", "rohf"]:
                    with self.subTest(molecule=name, norb=norb, nelec=nelec, guess=guess):
                        msmd = MultistateMatrixDensityCAS.from_guess(mol, norb=norb, nelec=nelec, guess=guess)
                        self.check_integrals_density_matrices_ao(msmd)
                        # Check that orbital rotations are registered as parameters.
                        parameters = list(msmd.parameters())
                        self.assertGreater(len(parameters), 0)
                        # Check that parameters require gradients
                        for param_name, param in msmd.named_parameters():
                            if param_name == "orbital_rotation_params":
                                self.assertEqual(param.requires_grad, True)
                            elif param_name == "state_rotation_params":
                                self.assertEqual(param.requires_grad, True)
                            else:
                                self.assertEqual(param.requires_grad, False)

    def check_autograd(self, mol, norb: int, nelec: int):
        """
        Compare analytical and numerical gradients of D(r) with respect to
        parameters of matrix density.
        """
        # Gradients are checked at random coordinates.
        ncoord = 100
        # The hardcoded seed ensures that the same random numbers are used
        # every time the test is run. Otherwise the test fails occasionally
        # when the threshold is is too tight.
        random_number_generator = numpy.random.default_rng(seed=789)
        coords = 5.0*(random_number_generator.random((ncoord,3)) - 0.5)

        msmd_0 = self.create_random_matrix_density_cas(mol, norb, nelec)
        orbital_coefficients = msmd_0.orbital_coefficients()
        state_coefficients = msmd_0.state_coefficients()

        def wrapper_function(orbital_rotation_params, state_rotation_params):
            msmd = MultistateMatrixDensityCAS(
                mol, norb, nelec,
                orbital_coefficients, orbital_rotation_params,
                state_coefficients, state_rotation_params
            )
            # (D, grad_D, lapl_D)
            outputs = msmd.evaluate(coords)
            return outputs

        nao, nmo = orbital_coefficients.size()
        # Random orbital rotation parameters.
        nrot_mo = (nmo*(nmo-1))//2
        orbital_rotation_params = random_tensor(Size([nrot_mo]))
        orbital_rotation_params.requires_grad_(True)

        nstate, _ = state_coefficients.size()
        nrot_ci = (nstate*(nstate-1))//2
        state_rotation_params = random_tensor(Size([nrot_ci]))
        state_rotation_params.requires_grad_(True)

        # Compare analytical gradients computed by backpropagation
        # finite difference approximation.
        inputs = (orbital_rotation_params, state_rotation_params)
        gradcheck(wrapper_function, inputs)

    def test_autograd(self):
        """
        Check that automatic differentiation works for D(U_mo,U_ci;r) w/r/t U_mo and U_ci.
        """
        for name, mol in tqdm(self.create_test_molecules_minimal().items()):
            for (norb, nelec) in self.active_spaces_for_testing(mol, norb_max=2):
                with self.subTest(molecule=name, norb=norb, nelec=nelec):
                    self.check_autograd(mol, norb, nelec)

    def test_autograd_density_matrices_ao(self):
        """
        Compare analytical and numerical gradients of Dᵦᵧᵢⱼ with respect to
        parameters of matrix density.
        """
        # Choose test system
        mol = self.create_test_systems()['water']

        # active space
        norb, nelec = 2, 2
        msmd_0 = self.create_random_matrix_density_cas(mol, norb, nelec)
        orbital_coefficients = msmd_0.orbital_coefficients()
        state_coefficients = msmd_0.state_coefficients()

        def wrapper_function(orbital_rotation_params, state_rotation_params):
            msmd = MultistateMatrixDensityCAS(
                mol, norb, nelec,
                orbital_coefficients, orbital_rotation_params,
                state_coefficients, state_rotation_params
            )
            # density matrices Dᵦᵧᵢⱼ in AO basis
            outputs = msmd.density_matrices_ao()
            return outputs

        nao, nmo = orbital_coefficients.size()
        # Random orbital rotation parameters.
        nrot_mo = (nmo*(nmo-1))//2
        orbital_rotation_params = random_tensor(Size([nrot_mo]))
        orbital_rotation_params.requires_grad_(True)

        nstate, _ = state_coefficients.size()
        nrot_ci = (nstate*(nstate-1))//2
        state_rotation_params = random_tensor(Size([nrot_ci]))
        state_rotation_params.requires_grad_(True)

        # Compare analytical gradients computed by backpropagation
        # finite difference approximation.
        inputs = (orbital_rotation_params, state_rotation_params)
        gradcheck(wrapper_function, inputs)

    def test_wrong_active_space_raises_error(self):
        """
        Check that an error is raised if the active space is empty or
        does not contain any state with the desired spin.
        """
        mol = self.create_test_molecules()['water']

        # 1) norb < nelec
        with self.assertRaises(ActiveSpaceError) as err:
            MultistateMatrixDensityCAS.from_guess(mol, 2, 6)
        # Check error message
        self.assertIn('is empty', str(err.exception))

        # 2) no singlet state possible with two spin-up electrons
        with self.assertRaises(ActiveSpaceError) as err:
            MultistateMatrixDensityCAS.from_guess(mol, 2, (2,0))
        # Check error message
        self.assertIn('no states with total spin', str(err.exception))

        mol = self.create_test_molecules()['hydrogen atom']
        # 3) More active orbitals than possible
        with self.assertRaises(ActiveSpaceError) as err:
            MultistateMatrixDensityCAS.from_guess(mol, 4, 1)
        # Check error message
        self.assertIn('More active orbitals', str(err.exception))

        # 4) more active electrons than total number of electrons
        with self.assertRaises(ActiveSpaceError) as err:
            MultistateMatrixDensityCAS.from_guess(mol, 3, 3)
        # Check error message
        self.assertIn('More active electrons', str(err.exception))

        # 5) Odd total number of electrons but even number of active electrons
        mol = self.create_test_molecules()['lithium atom']
        with self.assertRaises(ActiveSpaceError) as err:
            MultistateMatrixDensityCAS.from_guess(mol, 2, 2)

    def test_determinant_coefficients_h2(self):
        """
        Check coefficients of singlet and triplet states of the hydrogen molecule
        for the (2,2) CAS, which are determined by symmetry.
        """
        mol = self.create_test_molecules()['hydrogen molecule']
        # Singlet
        mol.spin = 0
        msmd = MultistateMatrixDensityCAS.from_guess(mol, 2, 2)
        ci_coeff_det = msmd.determinant_coefficients().detach().numpy()
        occ_labels = msmd.occupation_labels()
        self.assertListEqual(occ_labels, ['2.', 'ab', 'ba', '.2'])
        # There are three singlet states.
        self.assertEqual(msmd.number_of_states, 3)
        for istate in range(0, msmd.number_of_states):
            # Since eigenvectors of S² with the same eigenvalue are degenerate,
            # they can come in any order.
            if numpy.round(ci_coeff_det[0,istate], 0) == 1.0:
                # HF ground state 2.
                numpy.testing.assert_allclose(
                    # remove global sign by taking absolute value
                    abs(ci_coeff_det[:,istate]), numpy.array([1.0, 0.0, 0.0, 0.0])
                )
            elif numpy.round(ci_coeff_det[3,istate], 0) == 1.0:
                # doubly excited state .2
                numpy.testing.assert_allclose(
                    # remove global sign by taking absolute value
                    abs(ci_coeff_det[:,istate]), numpy.array([0.0, 0.0, 0.0, 1.0])
                )
            else:
                # singly excited state
                numpy.testing.assert_allclose(
                    abs(ci_coeff_det[:,istate]), numpy.array([0.0, 1.0, 1.0, 0.0])/numpy.sqrt(2.0)
                )

        # Triplet
        mol.spin = 2
        msmd = MultistateMatrixDensityCAS.from_guess(mol, 2, 2)
        ci_coeff_det = msmd.determinant_coefficients().detach().numpy()
        occ_labels = msmd.occupation_labels()
        self.assertListEqual(occ_labels, ['2.', 'ab', 'ba', '.2'])
        # There is only a singlet triplet state.
        self.assertEqual(msmd.number_of_states, 1)
        self.assertAlmostEqual(ci_coeff_det[1,0], -ci_coeff_det[2,0])

    def test_determinant_coefficients_h2_cis(self):
        """
        Check coefficients of singlet and triplet states of the hydrogen molecule
        for the (2,2) CAS, which are determined by symmetry.
        Only the HF and singly excited determinants are included (max_level=1).
        """
        mol = self.create_test_molecules()['hydrogen molecule']
        # Singlet
        mol.spin = 0
        msmd = MultistateMatrixDensityCAS.from_guess(mol, 2, 2, max_level=1)
        ci_coeff_det = msmd.determinant_coefficients().detach().numpy()
        occ_labels = msmd.occupation_labels()
        self.assertListEqual(occ_labels, ['2.', 'ab', 'ba'])
        # There are two singlet states.
        self.assertEqual(msmd.number_of_states, 2)
        for istate in range(0, msmd.number_of_states):
            # Since eigenvectors of S² with the same eigenvalue are degenerate,
            # they can come in any order.
            if numpy.round(ci_coeff_det[0,istate], 0) == 1.0:
                # HF ground state 2.
                numpy.testing.assert_allclose(
                    # remove global sign by taking absolute value
                    abs(ci_coeff_det[:,istate]), numpy.array([1.0, 0.0, 0.0])
                )
            else:
                # singly excited state
                numpy.testing.assert_allclose(
                    abs(ci_coeff_det[:,istate]), numpy.array([0.0, 1.0, 1.0])/numpy.sqrt(2.0)
                )

        # Triplet
        mol.spin = 2
        msmd = MultistateMatrixDensityCAS.from_guess(mol, 2, 2, max_level=1)
        ci_coeff_det = msmd.determinant_coefficients().detach().numpy()
        occ_labels = msmd.occupation_labels()
        self.assertListEqual(occ_labels, ['2.', 'ab', 'ba'])
        # There is only a singlet triplet state.
        self.assertEqual(msmd.number_of_states, 1)
        self.assertAlmostEqual(ci_coeff_det[1,0], -ci_coeff_det[2,0])

    def test_string_representation(self):
        """ Check that __repr__ and __str__ methods work """
        mol = self.create_test_molecules()['hydrogen molecule']
        msmd = MultistateMatrixDensityCAS.from_guess(mol, 2, 2)
        repr(msmd)
        str(msmd)
        msmd = MultistateMatrixDensityCAS.from_guess(mol, 2, 2, spin_symmetry=False)
        repr(msmd)
        str(msmd)
        msmd = MultistateMatrixDensityCAS.from_guess(mol, 2, 2, spin_symmetry=False, max_level=1)
        repr(msmd)
        str(msmd)

    def test_spin_s2_expectation(self):
        """
        check expectation values of total spin operator S²
        """
        # H2
        mol = self.create_test_molecules()['hydrogen molecule']
        # singlet states
        mol.spin = 0
        msmd = MultistateMatrixDensityCAS.from_guess(mol, 2, 2)
        s2 = msmd.spin_s2_expectation()
        self.assertEqual(set(torch.round(s2, decimals=2).tolist()), set([0.0, 0.0, 0.0]))
        # triplet state
        mol.spin = 2
        msmd = MultistateMatrixDensityCAS.from_guess(mol, 2, 2)
        s2 = msmd.spin_s2_expectation()
        self.assertEqual(set(torch.round(s2, decimals=2).tolist()), set([2.0]))
        # without spin symmetry
        msmd = MultistateMatrixDensityCAS.from_guess(mol, 2, 2, spin_symmetry=False)
        s2 = msmd.spin_s2_expectation()
        self.assertEqual(set(torch.round(s2, decimals=2).tolist()), set([0.0, 0.0, 0.0, 2.0]))
        # same with max_level=2 (HF, singles and doubles)
        msmd = MultistateMatrixDensityCAS.from_guess(mol, 2, 2, spin_symmetry=False, max_level=2)
        s2 = msmd.spin_s2_expectation()
        self.assertEqual(set(torch.round(s2, decimals=2).tolist()), set([0.0, 0.0, 0.0, 2.0]))

        # only HF and singly excited configurations (singlets)
        mol.spin = 0
        msmd = MultistateMatrixDensityCAS.from_guess(mol, 2, 2, spin_symmetry=True, max_level=1)
        s2 = msmd.spin_s2_expectation()
        self.assertEqual(set(torch.round(s2, decimals=2).tolist()), set([0.0, 0.0]))
        # only HF and singly excited configurations (singlets and triplet)
        msmd = MultistateMatrixDensityCAS.from_guess(mol, 2, 2, spin_symmetry=False, max_level=1)
        s2 = msmd.spin_s2_expectation()
        self.assertEqual(set(torch.round(s2, decimals=2).tolist()), set([0.0, 0.0, 2.0]))

    def test_spin_s2_expectation_invariant(self):
        """
        check expectation values of total spin operator S² when all Sz components are
        included in the subspace.
        """
        # H2
        mol = self.create_test_molecules()['hydrogen molecule']
        # singlet states
        mol.spin = 0
        msmd = MultistateMatrixDensityCAS.from_guess(mol, 2, 2, spin_type=SpinType.INVARIANT)
        s2 = msmd.spin_s2_expectation()
        self.assertEqual(set(torch.round(s2, decimals=2).tolist()), set([0.0, 0.0, 0.0]))
        # triplet state
        mol.spin = 2
        msmd = MultistateMatrixDensityCAS.from_guess(mol, 2, 2, spin_type=SpinType.INVARIANT)
        s2 = msmd.spin_s2_expectation()
        self.assertEqual(set(torch.round(s2, decimals=2).tolist()), set([2.0, 2.0, 2.0]))
        # without spin symmetry
        msmd = MultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, spin_symmetry=False, spin_type=SpinType.INVARIANT)
        s2 = msmd.spin_s2_expectation()
        self.assertEqual(set(torch.round(s2, decimals=2).tolist()), set([0.0, 0.0, 0.0, 2.0, 2.0, 2.0]))
        # same with max_level=2 (HF, singles and doubles)
        msmd = MultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, spin_symmetry=False, max_level=2, spin_type=SpinType.INVARIANT)
        s2 = msmd.spin_s2_expectation()
        self.assertEqual(set(torch.round(s2, decimals=2).tolist()), set([0.0, 0.0, 0.0, 2.0, 2.0, 2.0]))

        # only HF and singly excited configurations (singlets)
        mol.spin = 0
        msmd = MultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, spin_symmetry=True, max_level=1, spin_type=SpinType.INVARIANT)
        s2 = msmd.spin_s2_expectation()
        self.assertEqual(set(torch.round(s2, decimals=2).tolist()), set([0.0, 0.0]))
        # only HF and singly excited configurations (singlets and triplet)
        msmd = MultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, spin_symmetry=False, max_level=1, spin_type=SpinType.INVARIANT)
        s2 = msmd.spin_s2_expectation()
        self.assertEqual(set(torch.round(s2, decimals=2).tolist()), set([0.0, 0.0, 2.0, 2.0, 2.0]))

    def test_spin_s2_expectation_invariant_mix(self):
        """
        check expectation values of total spin operator S² when all Sz components are
        included in the subspace.
        """
        # H2
        mol = self.create_test_molecules()['hydrogen molecule']
        # singlet states
        mol.spin = 0
        msmd = MultistateMatrixDensityCAS.from_guess(mol, 2, 2, spin_type=SpinType.INVARIANT_MIX)
        s2 = msmd.spin_s2_expectation()
        self.assertEqual(set(torch.round(s2, decimals=2).tolist()), set([0.0, 0.0, 0.0]))
        # triplet state
        mol.spin = 2
        msmd = MultistateMatrixDensityCAS.from_guess(mol, 2, 2, spin_type=SpinType.INVARIANT_MIX)
        s2 = msmd.spin_s2_expectation()
        self.assertEqual(set(torch.round(s2, decimals=2).tolist()), set([2.0, 2.0, 2.0]))
        # without spin symmetry
        msmd = MultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, spin_symmetry=False, spin_type=SpinType.INVARIANT_MIX)
        s2 = msmd.spin_s2_expectation()
        self.assertEqual(set(torch.round(s2, decimals=2).tolist()), set([0.0, 0.0, 0.0, 2.0, 2.0, 2.0]))
        # same with max_level=2 (HF, singles and doubles)
        msmd = MultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, spin_symmetry=False, max_level=2, spin_type=SpinType.INVARIANT_MIX)
        s2 = msmd.spin_s2_expectation()
        self.assertEqual(set(torch.round(s2, decimals=2).tolist()), set([0.0, 0.0, 0.0, 2.0, 2.0, 2.0]))

        # only HF and singly excited configurations (singlets)
        mol.spin = 0
        msmd = MultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, spin_symmetry=True, max_level=1, spin_type=SpinType.INVARIANT_MIX)
        s2 = msmd.spin_s2_expectation()
        self.assertEqual(set(torch.round(s2, decimals=2).tolist()), set([0.0, 0.0]))
        # only HF and singly excited configurations (singlets and triplet)
        msmd = MultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, spin_symmetry=False, max_level=1, spin_type=SpinType.INVARIANT_MIX)
        s2 = msmd.spin_s2_expectation()
        self.assertEqual(set(torch.round(s2, decimals=2).tolist()), set([0.0, 0.0, 2.0, 2.0, 2.0]))

    def test_spin_multiplicity(self):
        """
        check spin multiplicities 2*S+1
        """
        # H2
        mol = self.create_test_molecules()['hydrogen molecule']
        # singlet states
        mol.spin = 0
        msmd = MultistateMatrixDensityCAS.from_guess(mol, 2, 2)
        multiplicity = msmd.spin_multiplicity()
        self.assertEqual(multiplicity.tolist(), [1, 1, 1])
        # Since only one Sz component of a spin multiple is calculated, its
        # energy has to be weighted by the spin multiplicity.
        self.assertEqual(multiplicity.tolist(), msmd.state_weights().tolist())
        # triplet state
        mol.spin = 2
        msmd = MultistateMatrixDensityCAS.from_guess(mol, 2, 2)
        multiplicity = msmd.spin_multiplicity()
        self.assertEqual(multiplicity.tolist(), [3])
        self.assertEqual(multiplicity.tolist(), msmd.state_weights().tolist())
        # without spin symmetry
        msmd = MultistateMatrixDensityCAS.from_guess(mol, 2, 2, spin_symmetry=False)
        multiplicity = msmd.spin_multiplicity()
        self.assertEqual(multiplicity.tolist(), [1, 1, 1, 3])
        self.assertEqual(multiplicity.tolist(), msmd.state_weights().tolist())
        # same with max_level=2 (HF, singles and doubles)
        msmd = MultistateMatrixDensityCAS.from_guess(mol, 2, 2, spin_symmetry=False, max_level=2)
        multiplicity = msmd.spin_multiplicity()
        self.assertEqual(multiplicity.tolist(), [1, 1, 1, 3])
        self.assertEqual(multiplicity.tolist(), msmd.state_weights().tolist())

        # only HF and singly excited configurations (singlets)
        mol.spin = 0
        msmd = MultistateMatrixDensityCAS.from_guess(mol, 2, 2, spin_symmetry=True, max_level=1)
        multiplicity = msmd.spin_multiplicity()
        self.assertEqual(multiplicity.tolist(), [1, 1])
        self.assertEqual(multiplicity.tolist(), msmd.state_weights().tolist())
        # only HF and singly excited configurations (singlets and triplet)
        msmd = MultistateMatrixDensityCAS.from_guess(mol, 2, 2, spin_symmetry=False, max_level=1)
        multiplicity = msmd.spin_multiplicity()
        self.assertEqual(multiplicity.tolist(), [1, 1, 3])
        self.assertEqual(multiplicity.tolist(), msmd.state_weights().tolist())

    def test_spin_multiplicity_invariant(self):
        """
        If all Sz components are included, there is no need to reweight the states by their
        spin multiplicity.
        """
        # H2
        mol = self.create_test_molecules()['hydrogen molecule']
        # singlet states
        mol.spin = 0
        msmd = MultistateMatrixDensityCAS.from_guess(mol, 2, 2, spin_type=SpinType.INVARIANT)
        multiplicity = msmd.spin_multiplicity()
        self.assertEqual(multiplicity.tolist(), [1, 1, 1])
        # triplet state
        mol.spin = 2
        msmd = MultistateMatrixDensityCAS.from_guess(mol, 2, 2, spin_type=SpinType.INVARIANT)
        multiplicity = msmd.spin_multiplicity()
        self.assertEqual(multiplicity.tolist(), [3, 3, 3])
        # Since all Sz components of a multiple are explictly included, the weights of the
        # states are no longer the spin multiplicities (2*S+1) but always 1.
        self.assertEqual(msmd.state_weights().tolist(), [1] * len(multiplicity))
        # without spin symmetry
        msmd = MultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, spin_symmetry=False, spin_type=SpinType.INVARIANT)
        multiplicity = msmd.spin_multiplicity()
        self.assertEqual(multiplicity.tolist(), [1, 1, 1,  3, 3, 3])
        self.assertEqual(msmd.state_weights().tolist(), [1] * len(multiplicity))

    def test_spin_multiplicity_invariant_mix(self):
        """
        If all Sz components are included, there is no need to reweight the states by their
        spin multiplicity.
        """
        # H2
        mol = self.create_test_molecules()['hydrogen molecule']
        # singlet states
        mol.spin = 0
        msmd = MultistateMatrixDensityCAS.from_guess(mol, 2, 2, spin_type=SpinType.INVARIANT_MIX)
        multiplicity = msmd.spin_multiplicity()
        self.assertEqual(multiplicity.tolist(), [1, 1, 1])
        # triplet state
        mol.spin = 2
        msmd = MultistateMatrixDensityCAS.from_guess(mol, 2, 2, spin_type=SpinType.INVARIANT_MIX)
        multiplicity = msmd.spin_multiplicity()
        self.assertEqual(multiplicity.tolist(), [3, 3, 3])
        # Since all Sz components of a multiple are explictly included, the weights of the
        # states are no longer the spin multiplicities (2*S+1) but always 1.
        self.assertEqual(msmd.state_weights().tolist(), [1] * len(multiplicity))
        # without spin symmetry
        msmd = MultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, spin_symmetry=False, spin_type=SpinType.INVARIANT_MIX)
        multiplicity = msmd.spin_multiplicity()
        self.assertEqual(multiplicity.tolist(), [1, 1, 1,  3, 3, 3])
        self.assertEqual(msmd.state_weights().tolist(), [1] * len(multiplicity))

    def test_orbital_guess(self):
        """
        check all ways to pass the initial guess for the molecular orbital coefficients.
        """
        # H2
        mol = self.create_test_molecules()['hydrogen molecule']
        # number of basis functions
        nao = mol.nao_nr()
        # Create MOs from scratch
        for guess in ["random", "hcore", "rohf"]:
            with self.subTest(guess=guess):
                mo_coeff = orbital_guess(mol, guess=guess)
                self.assertEqual(mo_coeff.shape, (nao,nao))
        # Use existing MOs from DFT calculation
        roks = pyscf.dft.ROKS(mol)
        # silent
        roks.verbose = 0
        roks.kernel()
        mo_coeff = orbital_guess(mol, roks.mo_coeff)
        # Check that orbitals are returned as guess.
        numpy.testing.assert_allclose(mo_coeff, roks.mo_coeff)

    def test_orbital_guess_raises_exceptions(self):
        """
        Check that orbital_guess raises exceptions for wrong inputs.
        """
        # H2
        mol = self.create_test_molecules()['hydrogen molecule']

        # 1) Unknown method to generate guess
        with self.assertRaises(NotImplementedError) as err:
            orbital_guess(mol, guess="roks")
        # Check error message
        self.assertIn('Initial guess', str(err.exception))

        # 2) torch tensor instead of numpy array
        mo_coeff = torch.zeros((3,3))
        with self.assertRaises(ValueError) as err:
            orbital_guess(mol, guess=mo_coeff)
        # Check error message
        self.assertIn('Initial guess must be str or 2D', str(err.exception))

        # 3) numpy array but with wrong dimensions
        mo_coeff = numpy.zeros((3,2))
        with self.assertRaises(ValueError) as err:
            orbital_guess(mol, guess=mo_coeff)
        # Check error message
        self.assertIn('matrix', str(err.exception))

    def test_reorder_active_orbitals(self):
        """
        Orbitals can be reordered such that the active orbitals become the HOMO and LUMO.
        """
        mol = self.create_test_molecules()['water']
        # generate MOs of water
        mol.basis = '6-31g'
        roks = pyscf.dft.ROKS(mol)
        roks.verbose = 0
        roks.kernel()
        # Active space should consist of 2 electrons in HOMO-1 and LUMO
        nelec = 2
        for active_orbitals in [
            [3, 5],
            ["HOMO-1", "LUMO"],
            ["H-1", "L"],
            [3, "LUMO"]
        ]:
            with self.subTest(active_orbitals=active_orbitals):
                # Reorder active space
                mo_coeff = reorder_active_orbitals(mol, roks.mo_coeff, active_orbitals, nelec)
                # Check that orbitals are still orthonormal
                overlap = mol.intor_symmetric('int1e_ovlp')
                nao = overlap.shape[0]
                numpy.testing.assert_allclose(
                    mo_coeff.T @ overlap @ mo_coeff, numpy.eye(nao), atol=1.0e-12)
                # Check that new HOMO is old HOMO-1
                numpy.testing.assert_allclose(mo_coeff[:,4], roks.mo_coeff[:,3], atol=1.0e-12)
                # Check that new LUMO is old LUMO
                numpy.testing.assert_allclose(mo_coeff[:,5], roks.mo_coeff[:,5], atol=1.0e-12)


class TestTargetStateMultistateMatrixDensityCAS(unittest.TestCase, FixtureMixin):
    """Stage 1 target-state CAS density tests."""

    def test_h2_full_target_space_matches_dense_density_matrices_ao(self):
        mol = self.create_test_molecules()['hydrogen molecule']
        dense = MultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess='hcore')
        target = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, target_states=dense.number_of_states,
            spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess='hcore')

        self.assertEqual(target.number_of_states, dense.number_of_states)
        self.assertEqual(target.number_of_csfs, dense.number_of_states)
        torch.testing.assert_close(target.density_matrices_ao(), dense.density_matrices_ao())

    def test_sparse_gamma_matches_dense_csf_contraction_cas44(self):
        mol = self.create_test_molecules()['lithium hydride']
        target = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 4, 4, target_states=3,
            spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess='hcore')

        with torch.no_grad():
            target.target_rotation_params.copy_(torch.linspace(
                -0.03, 0.04, target.target_rotation_params.numel(), dtype=torch.double))

        dense = MultistateMatrixDensityCAS.from_guess(
            mol, 4, 4, spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess='hcore')
        coeff = target.target_coefficients()
        gamma_dense = torch.einsum('...st,si,tj->...ij', dense.dm_mo_spin, coeff, coeff)
        gamma_sparse = target.transition_1rdm_mo()

        torch.testing.assert_close(gamma_sparse, gamma_dense, rtol=1.0e-10, atol=1.0e-10)

    def test_target_one_body_table_does_not_call_dense_matrix_density_mo(self):
        mol = self.create_test_molecules()['hydrogen molecule']
        reference = MultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess='hcore')
        with mock.patch.object(
                density_mod.ActiveSpace,
                "matrix_density_mo",
                side_effect=AssertionError("dense matrix_density_mo should not be called")):
            target = TargetStateMultistateMatrixDensityCAS(
                mol, 2, 2, target_states=2,
                orbital_coefficients=reference.orbital_coefficients(),
                orbital_rotation_params=reference.orbital_rotation_params,
                det_to_csf=reference.det_to_csf,
                s2_matrix=reference.s2_matrix,
                active_space=reference.active_space,
                spin_symmetry=True,
                spin_type=SpinType.UNPOLARIZED,
            )

        self.assertGreater(target.one_body_values.numel(), 0)
        self.assertEqual(target.one_body_shape, (2, 2, target.nmo, target.nmo, target.ndet, target.ndet))

    def test_target_from_guess_does_not_call_dense_from_guess(self):
        mol = self.create_test_molecules()['hydrogen molecule']
        with mock.patch.object(
                density_mod.MultistateMatrixDensityCAS,
                "from_guess",
                side_effect=AssertionError("target from_guess must not call dense from_guess")):
            target = TargetStateMultistateMatrixDensityCAS.from_guess(
                mol, 2, 2, target_states=2,
                spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess='hcore')

        self.assertEqual(target.number_of_states, 2)
        self.assertEqual(target.number_of_csfs, 2)
        self.assertGreater(target.one_body_values.numel(), 0)

    def test_target_direct_from_guess_matches_dense_spin_basis_small_system(self):
        mol = self.create_test_molecules()['lithium hydride']
        dense = MultistateMatrixDensityCAS.from_guess(
            mol, 4, 4, spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess='hcore')
        target = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 4, 4, target_states=3,
            spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess='hcore')

        self.assertEqual(target.number_of_determinants, dense.number_of_determinants)
        self.assertEqual(target.number_of_csfs, dense.number_of_states)
        torch.testing.assert_close(target.det_to_csf, dense.det_to_csf)
        torch.testing.assert_close(target.s2_matrix, dense.s2_matrix)
        torch.testing.assert_close(target.orbital_coefficients(), dense.orbital_coefficients())

    def test_sparse_gamma_matches_dense_all_spin_blocks_noncollinear(self):
        mol = self.create_test_molecules()['hydrogen molecule']
        target = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, target_states=4,
            spin_symmetry=False, spin_type=SpinType.NONCOLLINEAR, guess='hcore')

        with torch.no_grad():
            target.target_rotation_params.copy_(torch.linspace(
                -0.04, 0.05, target.target_rotation_params.numel(), dtype=torch.double))

        dense = MultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, spin_symmetry=False, spin_type=SpinType.NONCOLLINEAR, guess='hcore')
        coeff = target.target_coefficients()
        gamma_dense = torch.einsum('...st,si,tj->...ij', dense.dm_mo_spin, coeff, coeff)
        gamma_sparse = target.transition_1rdm_mo()

        torch.testing.assert_close(gamma_sparse, gamma_dense, rtol=1.0e-10, atol=1.0e-10)

    def test_mo_grid_density_matches_ao_grid_density(self):
        mol = self.create_test_molecules()['hydrogen molecule']
        target = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, target_states=2,
            spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess='hcore')
        coords = numpy.array([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.2],
            [-0.2, 0.3, 0.1],
        ])
        ao_value = pyscf.dft.numint.eval_ao(mol, coords, deriv=0)
        ao_value = torch.from_numpy(ao_value).to(dtype=torch.double)
        batch = AOGridBatch(weights=None, ao_value=ao_value)

        D_mo, grad_D_mo, lapl_D_mo = target.evaluate_from_mo_grid_batch(batch)
        D_ao, grad_D_ao, lapl_D_ao = target.evaluate_from_ao_batch(
            batch, need_gradient=False, need_laplacian=False)

        torch.testing.assert_close(D_mo, D_ao, rtol=1.0e-10, atol=1.0e-10)
        self.assertIsNone(grad_D_mo)
        self.assertIsNone(lapl_D_mo)
        self.assertIsNone(grad_D_ao)
        self.assertIsNone(lapl_D_ao)

    def test_noncollinear_target_density_uses_full_spin_block_space(self):
        mol = self.create_test_molecules()['hydrogen molecule']
        dense = MultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, spin_symmetry=False, spin_type=SpinType.NONCOLLINEAR, guess='hcore')
        target = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, target_states=2,
            spin_symmetry=False, spin_type=SpinType.NONCOLLINEAR, guess='hcore')

        self.assertEqual(target.spin_type, SpinType.NONCOLLINEAR)
        self.assertEqual(target.number_of_determinants, dense.ndet)
        self.assertEqual(target.number_of_csfs, dense.number_of_states)
        self.assertEqual(target.transition_1rdm_mo().size()[:4], Size([2, 2, target.nmo, target.nmo]))

    def test_noncollinear_mo_grid_density_matches_ao_grid_density(self):
        mol = self.create_test_molecules()['hydrogen molecule']
        target = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, target_states=2,
            spin_symmetry=False, spin_type=SpinType.NONCOLLINEAR, guess='hcore')
        coords = numpy.array([
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.2],
            [-0.2, 0.3, 0.1],
        ])
        ao_value = pyscf.dft.numint.eval_ao(mol, coords, deriv=0)
        ao_value = torch.from_numpy(ao_value).to(dtype=torch.double)
        batch = AOGridBatch(weights=None, ao_value=ao_value)

        D_mo, grad_D_mo, lapl_D_mo = target.evaluate_from_mo_grid_batch(batch)
        D_ao, grad_D_ao, lapl_D_ao = target.evaluate_from_ao_batch(
            batch, need_gradient=False, need_laplacian=False)

        self.assertEqual(D_mo.size(), Size([2, 2, len(coords), target.number_of_states, target.number_of_states]))
        torch.testing.assert_close(D_mo, D_ao, rtol=1.0e-10, atol=1.0e-10)
        self.assertIsNone(grad_D_mo)
        self.assertIsNone(lapl_D_mo)
        self.assertIsNone(grad_D_ao)
        self.assertIsNone(lapl_D_ao)

    def test_target_density_autograd(self):
        mol = self.create_test_molecules()['hydrogen molecule']
        target0 = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, target_states=2,
            spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess='hcore')
        coords = numpy.array([[0.0, 0.0, 0.0], [0.2, -0.1, 0.1]])
        ao_value = torch.from_numpy(pyscf.dft.numint.eval_ao(mol, coords, deriv=0)).to(dtype=torch.double)
        batch = AOGridBatch(weights=None, ao_value=ao_value)

        def wrapper_function(orbital_rotation_params, target_rotation_params):
            target = TargetStateMultistateMatrixDensityCAS(
                mol,
                2,
                2,
                2,
                target0.mo_coeff_guess,
                orbital_rotation_params,
                target0.det_to_csf,
                target0.s2_matrix,
                target0.active_space,
                spin_symmetry=True,
                spin_type=SpinType.UNPOLARIZED,
                target_rotation_params=target_rotation_params,
            )
            D, _, _ = target.evaluate_from_mo_grid_batch(batch)
            return D

        orbital_rotation_params = random_tensor(target0.orbital_rotation_params.size())
        orbital_rotation_params.requires_grad_(True)
        target_rotation_params = random_tensor(target0.target_rotation_params.size())
        target_rotation_params.requires_grad_(True)
        gradcheck(wrapper_function, (orbital_rotation_params, target_rotation_params))

    def test_target_and_orbital_orthonormality_invariants(self):
        mol = self.create_test_molecules()['hydrogen molecule']
        target = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, target_states=2,
            spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess='hcore')

        self.assertLess(target.target_orthonormality_error().item(), 1.0e-12)
        self.assertLess(target.orbital_orthonormality_error().item(), 1.0e-10)
        with torch.no_grad():
            target.target_rotation_params.add_(torch.linspace(
                -0.02, 0.03, target.target_rotation_params.numel(), dtype=torch.double))
            target.orbital_rotation_params.add_(torch.linspace(
                -0.01, 0.02, target.orbital_rotation_params.numel(), dtype=torch.double))
        self.assertLess(target.target_orthonormality_error().item(), 1.0e-12)
        self.assertLess(target.orbital_orthonormality_error().item(), 1.0e-10)

    def test_mo_grid_evaluator_does_not_persist_orbital_pair_grid_density(self):
        mol = self.create_test_molecules()['hydrogen molecule']
        target = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, target_states=2,
            spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess='hcore')
        coords = numpy.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.0]])
        ao_value = torch.from_numpy(pyscf.dft.numint.eval_ao(mol, coords, deriv=0)).to(dtype=torch.double)
        D, _, _ = target.evaluate_from_mo_grid_batch(AOGridBatch(weights=None, ao_value=ao_value))

        self.assertEqual(D.size(), Size([2, 2, len(coords), target.number_of_states, target.number_of_states]))
        self.assertIsNone(target._last_orbital_grid_density)
        for name in vars(target):
            self.assertNotIn('pqg', name.lower())
            self.assertNotIn('orbital_pair_grid', name.lower())

    def test_stage4_invariant_diagnostics_and_restart(self):
        mol = self.create_test_molecules()['hydrogen molecule']
        target = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, target_states=2,
            spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess='hcore')
        with torch.no_grad():
            target.target_rotation_params.add_(torch.linspace(
                -0.02, 0.03, target.target_rotation_params.numel(), dtype=torch.double))
            target.orbital_rotation_params.add_(torch.linspace(
                -0.01, 0.02, target.orbital_rotation_params.numel(), dtype=torch.double))

        diagnostics = target.check_invariants(fail=True)
        self.assertLess(diagnostics['target_orthonormality_error'], 1.0e-10)
        self.assertLess(diagnostics['orbital_orthonormality_error'], 1.0e-8)
        state = target.restart_state()

        restored = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, target_states=2,
            spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess='hcore')
        restored.load_restart_state(state)
        torch.testing.assert_close(restored.target_rotation_params, target.target_rotation_params)
        torch.testing.assert_close(restored.orbital_rotation_params, target.orbital_rotation_params)
        self.assertLess(restored.check_invariants()['target_orthonormality_error'], 1.0e-10)

    def test_versioned_restart_file_roundtrip_and_incompatibility_rejection(self):
        mol = self.create_test_molecules()['hydrogen molecule']
        target = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, target_states=2,
            spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess='hcore')
        with torch.no_grad():
            target.target_rotation_params.add_(torch.linspace(
                -0.01, 0.02, target.target_rotation_params.numel(), dtype=torch.double))
        with tempfile.TemporaryDirectory() as tmpdir:
            restart_path = f"{tmpdir}/target_restart.json"
            target.save_restart_file(restart_path)
            restored = TargetStateMultistateMatrixDensityCAS.from_guess(
                mol, 2, 2, target_states=2,
                spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess='hcore')
            restored.load_restart_file(restart_path)
            torch.testing.assert_close(restored.target_rotation_params, target.target_rotation_params)

            incompatible = restored.restart_state()
            incompatible["target_states"] = 1
            with self.assertRaises(ValueError):
                restored.load_restart_state(incompatible)

    def test_enforce_spin_symmetry_rejects_cross_spin_target_rotations(self):
        mol = self.create_test_molecules()['hydrogen molecule']
        target = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, target_states=4,
            spin_symmetry=False, spin_type=SpinType.UNPOLARIZED, guess='hcore')
        target.spin_symmetry = True
        with torch.no_grad():
            target.target_rotation_params.add_(0.05)
        with self.assertRaises(RuntimeError):
            target.enforce_spin_symmetry()

    def test_stage4_spin_mixing_warning_for_cross_spin_targets(self):
        mol = self.create_test_molecules()['hydrogen molecule']
        target = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, target_states=4,
            spin_symmetry=False, spin_type=SpinType.UNPOLARIZED, guess='hcore')
        target.spin_symmetry = True
        with torch.no_grad():
            target.target_rotation_params.add_(0.05)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            defect = target.warn_if_spin_mixed()
        self.assertIsNotNone(defect)
        self.assertTrue(any('mix spin sectors' in str(item.message) for item in caught))

    def test_state_diagnostics_reports_spin_dominant_csfs_and_root_overlap(self):
        mol = self.create_test_molecules()['hydrogen molecule']
        target = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, target_states=2,
            spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess='hcore')
        reference = target.target_coefficients().detach().clone()
        with torch.no_grad():
            target.target_rotation_params.add_(torch.linspace(
                -0.01, 0.02, target.target_rotation_params.numel(), dtype=torch.double))

        diagnostics = target.state_diagnostics(reference_target_coefficients=reference, dominant_count=2)

        self.assertEqual(len(diagnostics["spin_s2"]), target.number_of_states)
        self.assertEqual(len(diagnostics["spin_multiplicity"]), target.number_of_states)
        self.assertEqual(len(diagnostics["dominant_csfs"]), target.number_of_states)
        self.assertEqual(len(diagnostics["dominant_csfs"][0]), 2)
        self.assertEqual(len(diagnostics["reference_overlap_abs"]), target.number_of_states)
        self.assertGreaterEqual(diagnostics["dominant_csfs"][0][0]["weight"], 0.0)

    def test_restart_resume_preserves_hamiltonian_and_diagnostics(self):
        mol = self.create_test_molecules()['hydrogen molecule']
        target = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, target_states=2,
            spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess='hcore')
        with torch.no_grad():
            target.target_rotation_params.add_(torch.linspace(
                -0.01, 0.02, target.target_rotation_params.numel(), dtype=torch.double))
            target.orbital_rotation_params.add_(torch.linspace(
                -0.02, 0.01, target.orbital_rotation_params.numel(), dtype=torch.double))
        with tempfile.TemporaryDirectory() as tmpdir:
            restart_path = f"{tmpdir}/target_restart.json"
            target.save_restart_file(restart_path)
            resumed = TargetStateMultistateMatrixDensityCAS.from_guess(
                mol, 2, 2, target_states=2,
                spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess='hcore')
            resumed.load_restart_file(restart_path)

        torch.testing.assert_close(resumed.target_coefficients(), target.target_coefficients())
        torch.testing.assert_close(resumed.orbital_coefficients(), target.orbital_coefficients())
        self.assertEqual(
            resumed.state_diagnostics()["spin_multiplicity"],
            target.state_diagnostics()["spin_multiplicity"],
        )

    def test_stiefel_zero_params_matches_full_rotation_target_coefficients(self):
        mol = self.create_test_molecules()['lithium hydride']
        full = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 4, 4, target_states=3,
            spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess='hcore',
            target_parameterization="full_rotation")
        stiefel = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 4, 4, target_states=3,
            spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess='hcore',
            target_parameterization="stiefel_k")

        torch.testing.assert_close(stiefel.target_coefficients(), full.target_coefficients())
        torch.testing.assert_close(stiefel.density_matrices_ao(), full.density_matrices_ao())

    def test_stiefel_target_parameter_count(self):
        mol = self.create_test_molecules()['lithium hydride']
        target = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 4, 4, target_states=3,
            spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess='hcore',
            target_parameterization="stiefel_k")

        self.assertEqual(
            target.target_rotation_params.size(),
            Size([target.number_of_csfs - target.number_of_states, target.number_of_states]),
        )
        self.assertEqual(
            target.target_rotation_params.numel(),
            (target.number_of_csfs - target.number_of_states) * target.number_of_states,
        )

    def test_stiefel_target_orthonormality_invariant(self):
        mol = self.create_test_molecules()['lithium hydride']
        target = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 4, 4, target_states=3,
            spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess='hcore',
            target_parameterization="stiefel_k")
        with torch.no_grad():
            target.target_rotation_params.copy_(torch.linspace(
                -0.03, 0.04, target.target_rotation_params.numel(), dtype=torch.double
            ).reshape_as(target.target_rotation_params))

        self.assertLess(target.target_orthonormality_error().item(), 1.0e-12)

    def test_stiefel_target_density_autograd(self):
        mol = self.create_test_molecules()['hydrogen molecule']
        target0 = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, target_states=2,
            spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess='hcore',
            target_parameterization="stiefel_k")
        coords = numpy.array([[0.0, 0.0, 0.0], [0.2, -0.1, 0.1]])
        ao_value = torch.from_numpy(pyscf.dft.numint.eval_ao(mol, coords, deriv=0)).to(dtype=torch.double)
        batch = AOGridBatch(weights=None, ao_value=ao_value)

        def wrapper_function(orbital_rotation_params, target_rotation_params):
            target = TargetStateMultistateMatrixDensityCAS(
                mol,
                2,
                2,
                2,
                target0.mo_coeff_guess,
                orbital_rotation_params,
                target0.det_to_csf,
                target0.s2_matrix,
                target0.active_space,
                spin_symmetry=True,
                spin_type=SpinType.UNPOLARIZED,
                target_rotation_params=target_rotation_params,
                target_parameterization="stiefel_k",
            )
            D, _, _ = target.evaluate_from_mo_grid_batch(batch)
            return D

        orbital_rotation_params = random_tensor(target0.orbital_rotation_params.size())
        orbital_rotation_params.requires_grad_(True)
        target_rotation_params = random_tensor(target0.target_rotation_params.size())
        target_rotation_params.requires_grad_(True)
        gradcheck(wrapper_function, (orbital_rotation_params, target_rotation_params))

    def test_stiefel_restart_roundtrip_and_cross_mode_rejection(self):
        mol = self.create_test_molecules()['hydrogen molecule']
        stiefel = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, target_states=2,
            spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess='hcore',
            target_parameterization="stiefel_k")
        with torch.no_grad():
            stiefel.target_rotation_params.add_(0.01)

        state = stiefel.restart_state()
        restored = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, target_states=2,
            spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess='hcore',
            target_parameterization="stiefel_k")
        restored.load_restart_state(state)

        torch.testing.assert_close(restored.target_rotation_params, stiefel.target_rotation_params)
        torch.testing.assert_close(restored.target_coefficients(), stiefel.target_coefficients())

        full = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, target_states=2,
            spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess='hcore',
            target_parameterization="full_rotation")
        with self.assertRaises(ValueError):
            full.load_restart_state(state)

    def test_stiefel_basis_transformation_resets_rectangular_params(self):
        mol = self.create_test_molecules()['hydrogen molecule']
        target = TargetStateMultistateMatrixDensityCAS.from_guess(
            mol, 2, 2, target_states=2,
            spin_symmetry=True, spin_type=SpinType.UNPOLARIZED, guess='hcore',
            target_parameterization="stiefel_k")
        old_shape = target.target_rotation_params.size()
        L = torch.eye(target.number_of_states, dtype=torch.double)

        target.basis_transformation(L)

        self.assertEqual(target.target_rotation_params.size(), old_shape)
        self.assertLess(target.target_orthonormality_error().item(), 1.0e-12)


if __name__ == "__main__":
    unittest.main()
