#!/usr/bin/env python
# coding: utf-8
from mlmsdft.nn import functional
from mlmsdft.nn.functional import ScalarFunction

import torch
from torch.autograd import gradcheck
from torch import Size, Tensor
import torch.linalg
import torch.nn.functional
import torch.testing

from typing import Callable
import unittest


def random_tensor(size: Size, scale=5.0) -> Tensor:
    """
    Create a random tensor with elements sampled from a normal
    distribution with mean 0.0 and standard distribution `scale`.

    :param size: size of random tensor
    :type size: Size
    """
    tensor = scale * torch.randn(size, dtype=torch.double)
    return tensor

def random_symmetric_matrix_tensor(size: Size, n=1, scale=5.0) -> Tensor:
    """
    Create a random tensor with elements sampled
    uniformly from the range [-scale, scale].

    The matrix dimensions (n,n) are appended to the size tuple
    before creating the tensor.

    For example, size=(3,4) and n=2 creates a tensor of size (3,4,2,2).

    The tensor is symmetric in the last two indices,
    i.e. X[...,i,j] = X[...,j,i].

    :param size: size for the batch dimensions
    :type size: int, tuple or Size

    :param n: matrix dimension
    :type n: int

    :return X: symmetric tensor of size (...,n,n) where dimensions
        for ... are specified using `size`.
    :rtype X: Tensor
    """
    if type(size) is int:
        size = torch.Size([size])
    elif type(size) is tuple:
        size = torch.Size(size)
    # Add matrix dimensions.
    size_nn = size + torch.Size([n,n])
    # Create random tensor.
    X = scale * torch.randn(size_nn, dtype=torch.double)
    # Symmetrize tensor in the last dimension
    # X(sym) = 1/2 (X + Xᵀ)
    X_sym = 0.5 * (X + torch.transpose(X, -1, -2))

    return X_sym

def random_orthogonal_matrix(n: int) -> Tensor:
    """
    create a random orthogonal matrix L of dimension n.

    :param n: matrix dimension
    :type n: int > 0

    :return L: orthogonal matrix
    :rtype L: Tensor of size (n,n)
    """
    # Create a random skew-symmetric matrix Xᵀ = -X
    scale = 2.0
    X = scale * torch.randn(n, n, dtype=torch.double)
    # Antisymmetrize X
    X = 0.5 * (X - torch.transpose(X, 0, 1))
    # The Lie algebra of the orthogonal group consists of skew-symmetric matrices.
    L_ortho = torch.linalg.matrix_exp(X)

    return L_ortho


def random_positive_symmetric_matrix_tensor(
    size: Size,
    n=1,
    scale=5.0,
    eps=0.0,
    singular=0
) -> Tensor:
    """
    Create a random batch of symmetric, positive definite matrices.

    The matrix dimensions (n,n) are appended to the size tuple
    before creating the tensor.

    For example, size=(3,4) and n=2 creates a tensor of size (3,4,2,2).

    The tensor is symmetric in the last two indices,
    i.e. X[...,i,j] = X[...,j,i]
    and has only non-negative eigenvalues.

    :param size: size for the batch dimensions
    :type size: int, tuple or Size

    :param n: matrix dimension
    :type n: int

    :param scale: standard distribution of random elements
    :param scale: float

    :param eps: A small positive number is added to the eigenvalues
        of X to ensure it is strictly positive.
    :type eps: float >= 0.0

    :param singular: randomly select `singular` eigenvalues and set them
        to a tiny value in order to obtain an almost singular matrix
    :type singlar: int < n

    :return X: symmetric, positive-definite tensor of size (...,n,n)
        where dimensions for ... are specified using `size`.
    :rtype X: Tensor
    """
    if type(size) is int:
        size = torch.Size([size])
    elif type(size) is tuple:
        size = torch.Size(size)
    # Add matrix dimensions.
    # dimension for random eigenvalues
    size_n = size + torch.Size([n])
    # dimension for random eigenvectors
    size_nn = size + torch.Size([n,n])

    # 1) Random eigenvectors U
    # Create a random skew-symmetric matrix Mᵀ = -M
    M = scale * torch.randn(*size_nn, dtype=torch.double)
    # Antisymmetrize M
    M = 0.5 * (M - torch.transpose(M, -2, -1))
    # The Lie algebra of the orthogonal group consists of skew-symmetric matrices.
    U = torch.linalg.matrix_exp(M)

    # 2) Random positive eigenvalues
    eigvals = torch.pow(torch.randn(*size_n, dtype=torch.double), 2) + eps

    if singular > 0:
        # Randomly set some of the eigenvalues to `eps` (such that 1+`eps`==1)
        singular_indices = torch.randperm(n)[:singular]
        eigvals[...,singular_indices] = 10.0 * torch.finfo(eigvals.dtype).eps

    # 3) The symmetric, positive-definite matrix is computed
    # from its eigendecomposition
    # X = U.Λ.U(t)ᵀ
    X = torch.einsum('...ik,...k,...jk->...ij', U, eigvals, U)

    # Verify that X is symmetric.
    torch.testing.assert_close(X, X.transpose(-2,-1))
    # Verify that X is positive-definite
    eigvals_test, _ = torch.linalg.eigh(X)
    assert torch.min(eigvals_test) > 0.0

    return X

def random_positive_symmetric_matrix(
    n=1,
    scale=5.0,
    eps=0.0
) -> Tensor:
    """
    Create a random symmetric, positive-definite n x n matrix

    :param n: matrix dimension
    :type n: int

    :param scale: standard distribution of random elements
    :param scale: float

    :param eps: A small positive number is added to the eigenvalues
        of X to ensure it is strictly positive.
    :type eps: float > 0.0

    :return X: symmetric, positive-definite matrix of size (n,n)
    :rtype X: Tensor

    """
    X = random_positive_symmetric_matrix_tensor(
        torch.Size([1]), n=n, scale=scale, eps=eps
    ).reshape(n,n)

    return X


def check_autograd(
    function: Callable,
    size: Size,
    n: int,
    *other_args,
    positive_definite = False,
    scale = 1.0
):
    """
    check consistency between forward and backward propagation
    for matrix functions.

    :param function: matrix function
    :param size: size of tensor (without matrix dimensions)
    :param n: matrix dimension (...,n,n)

    :param other_args: other positional arguments that are passed
        the matrix function, F(X, *other_args)
    :type other_args: anything except Tensor

    :param positive_definite: Some functions such as log are only
        uniquely defined for positive-definite inputs.
        If True, function is tested only on positive-definite X.
    :type positive_definite: True

    :param scale: Random arguments are scaled by this positive number
    :type scale: float > 0
    """
    # Closure to hide additional arguments to function.
    def wrapper_function(X):
        # Matrix functions can only process symmetric matrices.
        # Symmetrize tensor in the last dimension.
        # X(sym) = 1/2 (X + Xᵀ)
        # Transposition is a differentiable operation, so no problem here.
        X = 0.5 * (X + torch.transpose(X, -1, -2))
        fX = function(X, *other_args)
        return fX

    # Random input tensor with matrix dimensions.
    if positive_definite:
        X = scale * random_positive_symmetric_matrix_tensor(size, n=n, eps=1.0e-8)
    else:
        X = scale * random_symmetric_matrix_tensor(size, n=n)
    X.requires_grad_(True)
    # Compare analytical gradients computed by backpropagation
    # finite difference approximation.
    gradcheck(wrapper_function, X)


def check_transformation_property(
    function: Callable,
    size: Size,
    n: int,
    *other_args,
    positive_definite = False,
    scale = 1.0
):
    """
    Check that the matrix function F(X) transforms like

        F(L X L^{-1}) = L F(X) L^{-1}

    under an orthogonal basis transformation L, where L^{-1} = Lᵀ.

    :param function: matrix function F(X)

    :param size: size of tensor (without matrix dimensions)

    :param n: matrix dimension
    :type n: int > 0

    :param other_args: other positional arguments that are passed
        the matrix function, F(X, *other_args)
    :type other_args: anything except Tensor

    :param positive_definite: Some functions such as log are only
        uniquely defined for positive-definite inputs.
        If True, function is tested only on positive-definite X.
    :type positive_definite: bool

    :param scale: random numbers are scaled by this factor
    :type scale: float > 0
    """
    # Random input tensor with matrix dimensions.
    if positive_definite:
        X = scale * random_positive_symmetric_matrix_tensor(size, n=n, eps=1.0e-8)
    else:
        X = scale * random_symmetric_matrix_tensor(size, n=n)
    # Random orthogonal transformation matrix.
    L = random_orthogonal_matrix(n)
    Lt = torch.transpose(L, 0, 1)
    # Check  that L really is orthogonal, i.e. L Lᵀ = 1
    Id = torch.matmul(L, Lt)
    torch.testing.assert_close(Id, torch.eye(n, dtype=L.dtype))

    # Compute F(X)
    F_X = function(X, *other_args)
    # transform input argument, X' = L X Lᵀ
    Xp = torch.einsum('ia,...ab,bj->...ij', L, X, Lt)
    # transform output, F' = L F(X) Lᵀ
    Fp = torch.einsum('ia,...ab,bj->...ij', L, F_X, Lt)
    # Compute F of transformed output, F(L X Lᵀ)
    F_Xp = function(Xp, *other_args)

    # Check that F' = L F(X) Lᵀ = F(X') = F(L X Lᵀ)
    torch.testing.assert_close(Fp, F_Xp)


class TestScalarFunction(unittest.TestCase):
    """ check implementation of scalar functions and their derivatives """

    def _check_function_values(
        self,
        func: ScalarFunction,
        func_ref,
        *other_args,
        positive_definite=False
    ):
        """
        compare the value of a scalar function with pytorch's
        implementation as a reference.
        """
        x = random_tensor((2,2,1))
        if positive_definite:
            x = torch.abs(x)
        f = func.value(x, *other_args)
        f_ref = func_ref(x, *other_args)
        torch.testing.assert_close(f, f_ref, equal_nan=True)

    def _check_function_derivatives1(
        self,
        func: ScalarFunction,
        func_ref,
        *other_args,
        positive_definite=False
    ):
        """
        compare the 1st derivative value of a scalar function with
        pytorch's implementation as a reference.
        """
        x = random_tensor((2,2,1))
        if positive_definite:
            x = torch.abs(x)
        # Compute the gradient using the analytic expression.
        grad = func.derivative1(x, *other_args)
        # Now the gradient is computed using back-propagation on the
        # pytorch implementation.
        x.requires_grad_(True)
        f_ref = func_ref(x, *other_args)
        f_ref.backward(torch.ones_like(x))
        grad_ref = x.grad
        # Verify that both gradients agree.
        torch.testing.assert_close(grad, grad_ref, equal_nan=True)

    def test_exp(self):
        self._check_function_values(functional.Exp, torch.exp)
        self._check_function_derivatives1(functional.Exp, torch.exp)


# List of matrix functions that should be tested and
# the corresponding scalar torch functions.
# Some of them such as pow(x,m) can take additional parameters.
matrix_function_tests = [
    # Each test is a tuple of the form
    # (matrix function, torch scalar function, other positional arguments, needs positive inputs?)
    (functional.exp,  torch.exp,                         [],     False),
]

def _zero_noncommuting_elements(X: Tensor, w: Tensor) -> Tensor:
    """
    Given a square (n x n) matrix X and an n-dimensional vector,
    set elements of X to zero such that X and W=diag(w) commute.

    To ensure that X and W commute, i.e. wᵢ Xᵢⱼ = Xᵢⱼ wⱼ,
    we must have Xᵢⱼ=0 if wᵢ≠wⱼ.

    :param X: batch of square matrices
    :type X: Tensor of shape (...,n,n)

    :param w: elements wᵢ of diagonal matrix Wᵢⱼ = wᵢ δᵢⱼ
    :type w: Tensor of shape (n)

    :return X: X is modified in place.
    """
    # Set all off-diagonal elements of X between i and j belonging
    # to different weight groups to zero.
    n = len(w)
    for i in range(0, n):
        for j in range(0, n):
            if w[i] != w[j]:
                X[...,i,j] = 0.0
    return X

class TestMatrixFunction(unittest.TestCase):
    def test_level0_spectral_descriptor_n1_zero_corrections(self):
        X = torch.tensor([[[2.0]]], dtype=torch.double)
        params = functional.Level0LMDAParameters(A_x=1.0, A_c=1.0, gamma_x=2.0, gamma_c=2.0)
        torch.testing.assert_close(
            functional.lmda_level0_exchange(X, params),
            functional.lmda_level0_exchange(X, functional.Level0LMDAParameters()),
        )
        torch.testing.assert_close(
            functional.lmda_level0_correlation(X, params),
            functional.lmda_level0_correlation(X, functional.Level0LMDAParameters()),
        )

    def test_level0_analytic_baseline_uses_original_eigenvalues(self):
        eigenvalues = torch.tensor([0.4, 0.4 + 5.0e-11, 1.2], dtype=torch.double)
        X = torch.diag(eigenvalues)
        exchange = functional.lmda_level0_exchange(X)
        expected_values = -functional.Cx_Spin * eigenvalues.pow(4.0 / 3.0)
        torch.testing.assert_close(exchange, torch.diag(expected_values))
        self.assertNotEqual(float(exchange[0, 0]), float(exchange[1, 1]))

    def test_level0_zero_amplitude_matches_baseline(self):
        X = random_positive_symmetric_matrix_tensor((2,), n=3, eps=1.0e-4)
        params = functional.Level0LMDAParameters(A_x=0.0, A_c=0.0, beta_x=0.7, gamma_x=3.0, gamma_c=2.0)
        torch.testing.assert_close(
            functional.lmda_level0_exchange(X, params),
            functional.lmda_level0_exchange(X, functional.Level0LMDAParameters()),
        )
        torch.testing.assert_close(
            functional.lmda_level0_correlation(X, params),
            functional.lmda_level0_correlation(X, functional.Level0LMDAParameters()),
        )

    def test_level0_functionals_are_covariant(self):
        params = functional.Level0LMDAParameters(A_x=0.2, beta_x=0.8, gamma_x=0.5, A_c=0.1, b_c=0.7, gamma_c=0.4)
        check_transformation_property(functional.lmda_level0_exchange, torch.Size([2]), 3, params, positive_definite=True)
        check_transformation_property(functional.lmda_level0_correlation, torch.Size([2]), 3, params, positive_definite=True)

    def test_level0_noncollinear_exchange_matches_collinear_channels(self):
        Daa = torch.diag_embed(torch.tensor([[2.0, 1.5, 1.0], [1.7, 1.2, 0.8]], dtype=torch.double))
        Dbb = torch.diag_embed(torch.tensor([[0.4, 0.3, 0.2], [0.5, 0.2, 0.1]], dtype=torch.double))
        zero = torch.zeros_like(Daa)
        pauli = torch.stack((Daa + Dbb, zero, zero, Daa - Dbb), dim=0)
        result = functional.lmda_level0_noncollinear_exchange(pauli)
        reference = functional.lmda_level0_exchange(Daa) + functional.lmda_level0_exchange(Dbb)
        torch.testing.assert_close(result, reference)

    def test_level0_noncollinear_exchange_is_state_covariant(self):
        generator = torch.tensor(
            [
                [0.0, 0.3, -0.2, 0.1],
                [-0.3, 0.0, 0.4, -0.1],
                [0.2, -0.4, 0.0, 0.5],
                [-0.1, 0.1, -0.5, 0.0],
            ],
            dtype=torch.double,
        )
        full_rotation = torch.linalg.matrix_exp(generator)
        full_density = full_rotation @ torch.diag(
            torch.tensor([0.2, 0.5, 1.0, 1.4], dtype=torch.double)
        ) @ full_rotation.T
        pauli = self._pauli_from_full_spin_density(full_density)
        state_rotation = torch.tensor(
            [[torch.cos(torch.tensor(0.37)), -torch.sin(torch.tensor(0.37))],
             [torch.sin(torch.tensor(0.37)), torch.cos(torch.tensor(0.37))]],
            dtype=torch.double,
        )
        transformed_pauli = torch.stack(
            tuple(state_rotation @ channel @ state_rotation.T for channel in pauli), dim=0
        )
        full_state_rotation = torch.kron(torch.eye(2, dtype=torch.double), state_rotation)
        torch.testing.assert_close(
            functional.noncollinear_spin_density(transformed_pauli),
            full_state_rotation @ full_density @ full_state_rotation.T,
        )
        params = functional.Level0LMDAParameters(A_x=0.2, beta_x=0.8, gamma_x=0.5)
        result = functional.lmda_level0_noncollinear_exchange(pauli, params)
        transformed = functional.lmda_level0_noncollinear_exchange(transformed_pauli, params)
        torch.testing.assert_close(transformed, state_rotation @ result @ state_rotation.T)

    def test_level0_noncollinear_exchange_is_spin_rotation_invariant(self):
        full_density = torch.tensor(
            [
                [1.1, 0.1, 0.2, -0.1],
                [0.1, 0.9, 0.05, 0.15],
                [0.2, 0.05, 0.8, 0.0],
                [-0.1, 0.15, 0.0, 0.7],
            ],
            dtype=torch.double,
        )
        spin_angle = torch.tensor(0.29, dtype=torch.double)
        spin_rotation = torch.tensor(
            [
                [torch.cos(spin_angle), -torch.sin(spin_angle)],
                [torch.sin(spin_angle), torch.cos(spin_angle)],
            ],
            dtype=torch.double,
        )
        full_spin_rotation = torch.kron(spin_rotation, torch.eye(2, dtype=torch.double))
        pauli = self._pauli_from_full_spin_density(full_density)
        rotated_pauli = self._pauli_from_full_spin_density(
            full_spin_rotation @ full_density @ full_spin_rotation.T
        )
        params = functional.Level0LMDAParameters(A_x=0.2, beta_x=0.8, gamma_x=0.5)
        torch.testing.assert_close(
            functional.lmda_level0_noncollinear_exchange(rotated_pauli, params),
            functional.lmda_level0_noncollinear_exchange(pauli, params),
        )

    def test_level0_noncollinear_fitted_parameter_gradients_are_finite(self):
        D_alpha = torch.diag(torch.tensor([1.4, 0.8], dtype=torch.double))
        D_beta = torch.diag(torch.tensor([0.3, 0.5], dtype=torch.double))
        zero = torch.zeros_like(D_alpha)
        pauli = torch.stack((D_alpha + D_beta, zero, zero, D_alpha - D_beta))
        fitted = {
            name: torch.tensor(value, dtype=torch.double, requires_grad=True)
            for name, value in {
                "A_x": 0.2,
                "beta_x": 0.8,
                "gamma_x": 0.5,
                "chi_width_x": 0.05,
            }.items()
        }
        result = functional.lmda_level0_noncollinear_exchange(
            pauli, functional.Level0LMDAParameters(**fitted)
        )
        result.sum().backward()
        for name, parameter in fitted.items():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(bool(torch.isfinite(parameter.grad)))

    def test_level0_noncollinear_exchange_rejects_scalar_carrier(self):
        X = random_positive_symmetric_matrix_tensor((2,), n=3, eps=1.0e-4)
        with self.assertRaises(ValueError):
            functional.lmda_level0_noncollinear_exchange(X)

    @staticmethod
    def _pauli_from_full_spin_density(full_density):
        n = full_density.size(-1) // 2
        alpha = full_density[..., :n, :n]
        beta = full_density[..., n:, n:]
        alpha_beta = full_density[..., :n, n:]
        beta_alpha = full_density[..., n:, :n]
        return torch.stack(
            (
                alpha + beta,
                alpha_beta + beta_alpha,
                alpha_beta - beta_alpha,
                alpha - beta,
            ),
            dim=0,
        )

    def test_level0_fitted_tensor_parameter_gradients_are_finite(self):
        X = torch.diag_embed(
            torch.tensor([[0.2, 0.7, 1.4], [0.3, 0.8, 1.6]], dtype=torch.double)
        )
        fitted = {
            name: torch.tensor(value, dtype=torch.double, requires_grad=True)
            for name, value in {
                "A_x": 0.2,
                "beta_x": 0.8,
                "gamma_x": 0.5,
                "chi_width_x": 0.05,
                "A_c": 0.1,
                "b_c": 0.7,
                "gamma_c": 0.4,
                "chi_width_c": 0.06,
            }.items()
        }
        params = functional.Level0LMDAParameters(**fitted)
        loss = functional.lmda_level0_exchange(X, params).sum() + functional.lmda_level0_correlation(X, params).sum()
        loss.backward()
        for name, parameter in fitted.items():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(bool(torch.isfinite(parameter.grad)))

    def test_vectorized_divided_differences_match_loop_reference(self):
        L = torch.tensor(
            [[0.2, 0.2 + 1.0e-13, 1.7], [0.1, 0.5, 3.0]],
            dtype=torch.double,
        )
        fL = functional.Exp.value(L)
        epsilon = 1.0e-12

        eigval_derivs_ref = torch.zeros(L.size() + L.size()[-1:], dtype=L.dtype)
        for a in range(L.size(-1)):
            La = L[...,a]
            fLa = fL[...,a]
            for b in range(L.size(-1)):
                Lb = L[...,b]
                fLb = fL[...,b]
                same = torch.abs(La - Lb) < epsilon
                Yab = torch.zeros(L.size()[:-1], dtype=L.dtype)
                Yab[same] = functional.Exp.derivative1(0.5 * (La[same] + Lb[same]))
                Yab[~same] = (fLa[~same] - fLb[~same]) / (La[~same] - Lb[~same])
                eigval_derivs_ref[...,a,b] = Yab

        La = L.unsqueeze(-1)
        Lb = L.unsqueeze(-2)
        fLa = fL.unsqueeze(-1)
        fLb = fL.unsqueeze(-2)
        dL = La - Lb
        same = torch.abs(dL) < epsilon
        safe_dL = torch.where(same, torch.ones_like(dL), dL)
        eigval_derivs_vec = torch.where(
            same,
            functional.Exp.derivative1(0.5 * (La + Lb)),
            (fLa - fLb) / safe_dL,
        )

        torch.testing.assert_close(eigval_derivs_vec, eigval_derivs_ref)

    def test_matrix_dimension_raises_exceptions(self):
        """
        Verify that an exception is raised, if the input tensor to a
        matrix function does not have the size (...,n,n).
        """
        x = random_tensor((3,3,2))
        with self.assertRaises(ValueError) as err:
            functional.exp(x)
        # Check error message
        self.assertEqual(
            'The tensor X has to be of size (...,n,n) '
            'so that it can be intepreted as a batch of n x n matrices.\n'
            'The given input of size torch.Size([3, 3, 2]) does not fit that pattern.',
            str(err.exception))

    def test_autograd(self):
        # Loop over matrix functions that need to be tested.
        for func, _, other_args, posdef in matrix_function_tests:
            # Loop over matrix dimensions.
            for n in [1,2,3]:
                size = Size([2,1])
                with self.subTest(function=func, dimension=n, other_args=other_args):
                    check_autograd(func, size, n, *other_args, positive_definite=posdef)

    def test_transformation_property(self):
        # Loop over matrix functions that need to be tested.
        for func, _, other_args, posdef in matrix_function_tests:
            # Loop over size of tensor (excluding matrix dimensions)
            for size in [torch.Size([1]), torch.Size([2,3])]:
                # Loop over matrix dimensions.
                for n in [1,2,3]:
                    with self.subTest(function=func, size=size, dimension=n, other_args=other_args):
                        check_transformation_property(
                            func, size, n, *other_args, positive_definite=posdef)

    def check_scalar_function(
        self,
        function: Callable,
        function_ref: Callable,
        *other_args,
        positive_definite = False
    ):
        """ For 1 x 1 matrices, the matrix function should be the
        same result as pytorch's scalar function.
        Some functions such as log are only uniquely defined for positive-definite inputs.
        """
        # batch of 1 x 1 matrices, (...,1,1)
        if positive_definite:
            X = random_positive_symmetric_matrix_tensor((3,), n=1)
        else:
            X = random_symmetric_matrix_tensor((3,), n=1)
        F = function(X, *other_args)
        F_ref = function_ref(X, *other_args)
        torch.testing.assert_close(F, F_ref)

    def test_matrix_functions_on_1x1_matrices(self):
        # Loop over matrix functions that need to be tested.
        for func, func_ref, other_args, posdef in matrix_function_tests:
            with self.subTest(function=func):
                self.check_scalar_function(func, func_ref, *other_args, positive_definite=posdef)

    def test_matrix_exponential(self):
        """ Check implementation of matrix exponential against pytorch. """
        X = random_symmetric_matrix_tensor((2,3), n=2)
        expX = functional.exp(X)
        expX_ref = torch.linalg.matrix_exp(X)
        torch.testing.assert_close(expX, expX_ref)

    def test_trace_average(self):
        """ Check trace average over matrix dimensions. """
        size = torch.Size([2,3])
        for n in [1,2,3]:
            with self.subTest(dimension=n):
                X = random_symmetric_matrix_tensor(size, n=n)
                trX = functional.trace_average(X)
                # Check that the trace has shape (...,1,1)
                self.assertEqual(trX.size()[:2], size)
                self.assertEqual(trX.size()[-2:], torch.Size([1,1]))

                # compute trace-average by averaging over diagonal elements
                # of matrix dimensions.
                trX_ref = torch.zeros(size, dtype=X.dtype)
                for i in range(0, n):
                    trX_ref += 1.0/n * X[...,i,i]
                torch.testing.assert_close(trX[...,0,0], trX_ref)

    def test_subspace_trace_average(self):
        """ Check trace average over a m-dimensional subspace (m <= n)"""
        size = torch.Size([2,3])
        for n in [2,3]:
            for m in [1,n-1,n]:
                with self.subTest(dimension=n, subspace_dim=m):
                    X = random_symmetric_matrix_tensor(size, n=n)
                    trX = functional.trace_average(X, subspace_dim=m)
                    # Check that the trace has shape (...,1,1)
                    self.assertEqual(trX.size()[:2], size)
                    self.assertEqual(trX.size()[-2:], torch.Size([1,1]))

                    # compute trace-average by averaging over the lowest m
                    # eigenvalues of the matrix
                    trX_ref = torch.zeros(size, dtype=X.dtype)
                    # Diagonalize X, X = U.diag(x).U⁻¹
                    x = torch.linalg.eigvalsh(X)
                    # Average over lowest m eigenvalues
                    for i in range(0, m):
                        trX_ref += 1.0/m * x[...,i]
                    torch.testing.assert_close(trX[...,0,0], trX_ref)

    def test_trace_average_with_weights(self):
        """ Check weighted trace average over matrix dimensions . """
        size = torch.Size([2,3])
        for n in [1,2,3]:
            with self.subTest(dimension=n):
                # random weights
                weights = torch.randint(1, 3, Size([n]))
                X = random_symmetric_matrix_tensor(size, n=n)
                trX = functional.trace_average(X, weights=weights)
                # Check that the trace has shape (...,1,1)
                self.assertEqual(trX.size()[:2], size)
                self.assertEqual(trX.size()[-2:], torch.Size([1,1]))

                # compute trace-average by averaging over diagonal elements
                # of matrix dimensions.
                trX_ref = torch.zeros(size, dtype=X.dtype)
                for i in range(0, n):
                    trX_ref += weights[i] * X[...,i,i] / torch.sum(weights)
                torch.testing.assert_close(trX[...,0,0], trX_ref)

    def _random_symmetric_matrix_tensor_commuting_diagonal(self, size, n, k):
        """
        Create a random (...,n,n) matrix X and a diagonal matrix W
        with k distinct elements, such that X and W commute,
        [X,W]=0

        :param size: the remaining dimensions (denoted by ...)
        :type size: instance of Size

        :param n: matrix dimension
        :type n: int

        :param k: number of distinct numbers on the diagonal of W
        :type k: int <= n

        :return X: random symmetric matrix
        :rtype X: Tensor of shape (...,n,n)

        :return weights: weight vector wᵢ, diagonal elements of W, Wᵢⱼ = wᵢ δᵢⱼ
        :rtype weights Tensor of shape (n,)
        """
        X = random_symmetric_matrix_tensor(size, n=n)
        # k distinct values for weights
        unique_weights = torch.rand(k, dtype=torch.double)
        # Assign the distinct weights randomly to the n positions in the weight vector.
        index_groups = torch.chunk(torch.randperm(n), k)
        weights = torch.zeros(n, dtype=torch.double)
        for kk,indices_k in enumerate(index_groups):
            weights[indices_k] = unique_weights[kk]

        # To ensure that X and W commute, i.e. wᵢ Xᵢⱼ = Xᵢⱼ wⱼ,
        # we must have Xᵢⱼ=0 if wᵢ≠wⱼ.
        # Set all off-diagonal elements of X between i and j belonging
        # to different weight groups to zero.
        X = _zero_noncommuting_elements(X, weights)

        # Check that X and W = diag(weights) commute
        WX = torch.einsum('i,...ij->...ij', weights, X)
        XW = torch.einsum('...ij,j->...ij', X, weights)
        torch.testing.assert_close(WX, XW)

        return X, weights

    def test_subspace_trace_average_with_weights(self):
        """
        Check trace average over a m-dimensional subspace (m <= n)
        when the eigenvalues are given different weights.
        """
        size = torch.Size([2,3])
        for n in [2,3,5]:
            for m in [1,n-1,n]:
                for k in range(1, n+1):
                    with self.subTest(dimension=n, subspace_dim=m, distinct_weights=k):
                        # random (n x n) matrix X and weights
                        X, weights = self._random_symmetric_matrix_tensor_commuting_diagonal(size, n, k)

                        trX = functional.trace_average(X, weights=weights, subspace_dim=m)
                        # Check that the trace has shape (...,1,1)
                        self.assertEqual(trX.size()[:-2], size)
                        self.assertEqual(trX.size()[-2:], torch.Size([1,1]))

                        # compute trace-average as a weighted average over the lowest m
                        # eigenvalues of the matrix.
                        trX_ref = torch.zeros(size, dtype=X.dtype)
                        # Diagonalize X, X = U.diag(x).U⁻¹
                        x, U = torch.linalg.eigh(X)
                        # U should also diagonalize W, it only reorders the weights
                        # diag(w) = U⁻¹.W.U = U^T.W.U
                        w = torch.einsum('...ai,a,...ai->...i', U, weights, U)
                        # Sort the eigenvalues w*x
                        wx_sorted, _ = torch.sort(w*x, dim=-1)
                        # Sum of all weights
                        weight_sum = torch.sum(w, axis=-1)
                        # Sum of the lowest m weighted eigenvalues.
                        for i in range(0, m):
                            trX_ref += n/m * (1.0/weight_sum) * wx_sorted[...,i]

                        # Compare
                        torch.testing.assert_close(trX[...,0,0], trX_ref)

    def test_subspace_trace_average_autograd(self):
        """ Check analytical gradients if trace includes only lowest m < n eigenvalues """
        size = torch.Size([3,2])
        # Loop over dimension of matrix X
        for n in [2,3,5]:
            # Loop over dimension of subspace that should be included in the trace.
            for m in [1,n-1,n]:
                with self.subTest(dimension=n, subspace_dim=m):
                    # Trace of m-dimensional subspace
                    def func(X):
                        trX = functional.trace_average(X, subspace_dim=m)
                        return trX

                    # Compare analytic and numerical gradients
                    check_autograd(func, size, n)

    def test_subspace_trace_average_with_weights_autograd(self):
        """
        Check analytical gradients if trace includes only lowest m < n eigenvalues
        with different weights.
        """
        size = torch.Size([3,2])
        # Loop over dimension of matrix X
        for n in [2,3,5]:
            # k <= n is the number of distinct values for weights
            for k in range(1,n+1):
                unique_weights = torch.rand(k, dtype=torch.double)
                # Assign the distinct weights randomly to the n positions in the weight vector.
                index_groups = torch.chunk(torch.randperm(n), k)
                weights = torch.zeros(n, dtype=torch.double)
                for kk,indices_k in enumerate(index_groups):
                    weights[indices_k] = unique_weights[kk]

                # Loop over dimension of subspace that should be included in the trace.
                for m in [1,n-1,n]:
                    with self.subTest(dimension=n, subspace_dim=m):
                        # Trace of m-dimensional subspace
                        def func(X):
                            # Autograd changes all elements of X, however, some elements have
                            # the stay zero for X to commute with the weight matrix W. Therefore
                            # these elements are explicitly set to zero again.
                            X = _zero_noncommuting_elements(X, weights)
                            trX = functional.trace_average(X, weights=weights, subspace_dim=m)
                            return trX

                        # Compare analytic and numerical gradients
                        check_autograd(func, size, n)

    def test_weighted_trace_average_raises_exception(self):
        """ Check that an error is raised if the weight vector does not have n elements. """
        X = random_symmetric_matrix_tensor((3,1), n=2)
        weights = torch.rand(3)
        with self.assertRaises(ValueError) as err:
            functional.trace_average(X, weights=weights)
        # Check error message
        self.assertEqual(
            "Optional input tensor `weights` has to be of size (n,) "
            "with weights for each of the diagonal elements of an (n,n) matrix.\n"
            "n=2, but `weights` has size torch.Size([3]) .",
            str(err.exception))

    def test_subspace_trace_average_raises_exception(self):
        """ Check that an error is raised if the subspace dimension is outside of the valid range """
        X = random_symmetric_matrix_tensor((3,1), n=2)
        with self.assertRaises(ValueError) as err:
            functional.trace_average(X, subspace_dim=0)
        # Check error message
        self.assertEqual(
            "0 < `subspace_dim` <= n is the number of lowest eigenvalues to include.\n"
            "n=2, but subspace_dim=0",
            str(err.exception))


if __name__ == "__main__":
    unittest.main()
