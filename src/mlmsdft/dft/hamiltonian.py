# -*- coding: utf-8 -*-
from abc import ABC, abstractmethod
from collections.abc import Callable
import numpy

import pyscf.dft
import pyscf.gto
from pyscf.dft import numint

from torch import Tensor
import torch.linalg
import torch.nn
import torch.optim

from mlmsdft.dft.density import AOGridBatch
from mlmsdft.dft.density import MultistateMatrixDensity
from mlmsdft.dft.density import TargetStateMultistateMatrixDensityCAS
from mlmsdft.dft.hartree import HartreeFunctionalAO
from mlmsdft.dft.hartree import HartreeFunctionalShortRangeAO
from mlmsdft.dft.hartree_gpu4pyscf import HartreeFunctionalGpu4PySCFDFAO
from mlmsdft.dft.integrals import OneElectronIntegralCache
from mlmsdft.dft.range_separated import LongRangeInteractionFunctional
from mlmsdft.dft.kinetic import KineticFunctionalAO
from mlmsdft.dft.nuclear import NuclearFunctionalAO
from mlmsdft.dft.spin import SpinType
from mlmsdft.dft.spin import concat_spin_blocks, spin_trace
from mlmsdft.dft.spin import check_not_deprecated_spin_type
from mlmsdft.dft.spin import spin_blocks_to_pauli_channels
from mlmsdft.dft.xc import lda_x_dirac, lda_c_chachiyo
# Replace scalar functions by matrix functionals.
import mlmsdft.nn.functional as MF
# wrapper around optimizer written in numpy
from mlmsdft.optim.torch_optimizer import WrappedOptimizer
from mlmsdft.optim.torch_optimizer import TorchLBFGSOptimizer


def robust_symmetric_eigh(matrix: Tensor, epsilon: float) -> tuple[Tensor, Tensor]:
    try:
        return torch.linalg.eigh(matrix)
    except torch.linalg.LinAlgError:
        n = matrix.size(-1)
        identity = torch.eye(n, dtype=matrix.dtype, device=matrix.device)
        base_jitter = max(epsilon * epsilon, 1.0e-14)
        for scale in [1.0, 1.0e2, 1.0e4, 1.0e6, 1.0e8]:
            try:
                return torch.linalg.eigh(matrix + (base_jitter * scale) * identity)
            except torch.linalg.LinAlgError:
                continue
        return torch.linalg.eigh(torch.nan_to_num(matrix) + (base_jitter * 1.0e10) * identity)


def matrix_inverse_square_root(matrix: Tensor, epsilon: float) -> Tensor:
    n = matrix.size(-1)
    identity = torch.eye(n, dtype=matrix.dtype, device=matrix.device)
    regularized = matrix @ matrix + (epsilon * epsilon) * identity
    eigvals, eigvecs = robust_symmetric_eigh(regularized, epsilon)
    inv_sqrt = torch.rsqrt(torch.clamp(eigvals, min=epsilon * epsilon))
    return torch.einsum('...ia,...a,...ja->...ij', eigvecs, inv_sqrt, eigvecs)


def matrix_square_root_from_square(matrix_square: Tensor, epsilon: float) -> Tensor:
    eigvals, eigvecs = robust_symmetric_eigh(matrix_square, epsilon)
    sqrt_eigvals = torch.sqrt(torch.clamp(eigvals, min=0.0) + epsilon * epsilon)
    return torch.einsum('...ia,...a,...ja->...ij', eigvecs, sqrt_eigvals, eigvecs)


def noncollinear_channel_densities(
        spin_D: Tensor,
        spin_vector_regularization: float,
        need_spin_vector: bool = False) -> tuple[Tensor, Tensor, Tensor | None]:
    pauli_D = spin_blocks_to_pauli_channels(spin_D)
    D0 = pauli_D[0]
    spin_vector = pauli_D[1:4]
    if spin_vector_regularization > 0.0:
        spin_vector_norm = torch.max(torch.abs(spin_vector)).detach()
        if spin_vector_norm <= spin_vector_regularization:
            D_half = 0.5 * D0
            spin_direction = None
            if need_spin_vector:
                spin_direction = torch.zeros_like(spin_vector)
            return D_half, D_half, spin_direction
    Q2 = sum(Dj @ Dj for Dj in spin_vector)
    Q = matrix_square_root_from_square(Q2, spin_vector_regularization)
    D_plus = 0.5 * (D0 + Q)
    D_minus = 0.5 * (D0 - Q)

    spin_direction = None
    if need_spin_vector:
        Q_inv = matrix_inverse_square_root(Q, spin_vector_regularization)
        spin_direction = torch.stack(
            [0.5 * (Dj @ Q_inv + Q_inv @ Dj) for Dj in spin_vector],
            dim=0,
        )

    return D_plus, D_minus, spin_direction


def evaluate_noncollinear_lmda_xc(
        spin_D: Tensor,
        D: Tensor,
        grad_D: Tensor | None,
        lapl_D: Tensor | None,
        exchange: Callable,
        correlation: Callable,
        exchange_correlation: Callable,
        spin_vector_regularization: float,
        need_spin_vector: bool = False) -> Tensor:
    D_plus, D_minus, _spin_direction = noncollinear_channel_densities(
        spin_D,
        spin_vector_regularization=spin_vector_regularization,
        need_spin_vector=need_spin_vector,
    )
    if exchange_correlation is not None:
        raise NotImplementedError(
            "Fused exchange-correlation callables are only supported for "
            "SpinType.UNPOLARIZED. Use separate exchange and correlation "
            "callables for noncollinear calculations."
        )
    x = 0.0
    c = 0.0
    if exchange is not None:
        x = exchange(D_plus, None, None) + exchange(D_minus, None, None)
    if correlation is not None:
        c = correlation(D, grad_D, lapl_D)
    return x + c


class Hamiltonian(ABC):
    """
    Abstract base class
    """
    @abstractmethod
    def matrix_elements(self, msmd: MultistateMatrixDensity) -> Tensor:
        """
        Implement the Hamiltonian functional H[D(r)] that maps the
        matrix density to the Hamiltonian in the subspace of electronic states.
        """
        pass


class HamiltonianSemilocal(Hamiltonian):
    def __init__(
            self,
            mol: pyscf.gto,
            kinetic_functional: Callable = None,
            exchange_functional: Callable = lda_x_dirac,
            correlation_functional: Callable = lda_c_chachiyo,
            exact_exchange_functional: Callable = None,
            exchange_correlation_functional: Callable = None,
            spin_type: SpinType = SpinType.UNPOLARIZED,
            need_spin_vector: bool = False,
            spin_vector_regularization: float = 1.0e-10,
            grid_level: int = 3,
            grid_chunks: int | str | None = 1,
            target_memory_fraction: float | None = None,
            max_grid_points_per_chunk: int | None = None,
            min_grid_points_per_chunk: int = 1,
        ):
        """
        Hamiltonian with semilocal kinetic, exchange and correlation functionals.

        :param mol: molecule with atomic coordinates and basis set
        :type mol: pyscf.gto.Mole

        :param kinetic_functional, exchange_functional, correlation_functional:
            Semilocal kinetic, exchange and correlation functionals.
            If kinetic_functional is None, the kinetic energy is calculated from
            the wavefunction.
        :type kinetic_functional, exchange_functional, correlation_functional:
            Callables taking three tensors with the matrix density D(r), its gradient ∇D(r)
            and possibly also the Laplacian ∇²D as arguments

        :param exact_exchange_functional: Functional for exact (Hartree-Fock) exchange.
            The exact exchange is a non-local functional that depends on the density field
            Dᵢⱼ(r,r') = ∑ᵣ ∑ₛ Dᵣₛ,ᵢⱼ 𝛘ᵣ(r) 𝛘ₛ(r') (D=Dᵅᵅ+Dᵝᵝ,Dᵅᵅ,Dᵝᵝ or supermatrix)
        :type exact_exchange_functional: Callable taking the matrix density in the
            atomic orbitals (AO) basis, Dᵣₛ,ᵢⱼ, where r and s are AO indices and
            i,j are electronic states. If None, no exact exchange is added.

        :param spin_type: Determines how the electronic spin degrees of the matrix
            density are treated when constructing the Hamiltonian. The options are:
            UNPOLARIZED: The Hamiltonian is constructed from the spin-traced matrix density,
                Dᵢⱼ=Dᵅᵢⱼ(r)+Dᵝᵢⱼ(r). The (exact) exchange is calculated as 2*X[D/2]ᵢⱼ.
            POLARIZED: The (exact) exchange part of the Hamiltonian is computed separately for the
                spin-up and spin-down matrix densities, Xᵢⱼ = X[Dᵅ]ᵢⱼ + X[Dᵝ]ᵢⱼ.
            INVARIANT:
                The (exact) exchange part of the Hamiltonian is computed from the (2N)x(2N)
                supermatrix containing the NxN same-spin blocks Dᵅᵅ and Dᵝᵝ on the diagonal and
                the mixed-spin blocks Dᵅᵝ and Dᵝᵅ on the off-diagonal:
                Xᵢⱼ=spin_trace(X[(Dᵅᵅ Dᵅᵝ \\ Dᵝᵅ Dᵝ)])ᵢⱼ
                The resulting exchange energy matrix is invariant under rotations of the
                electronic spins, provided that all components of a spin multiplet (Sz=-S,...,S)
                are included in the subspace.
            INVARIANT_MIX:
                The (exact) exchange part of the Hamiltonian is the average of 50% of the
                unpolarized and 50% of the invariant exchange parts,
                Xᵢⱼ=1/2 { 2 X[D/2]ᵢⱼ +  spin_trace(X[(Dᵅᵅ Dᵅᵝ \\ Dᵝᵅ Dᵝ)])ᵢⱼ }
        :type spin_type: SpinType

        :param grid_level: The level (3-8) controls the number of grid points
           in the integration grid.
        :type grid_level: int

        :param grid_chunks: If there is not enough memory (on the CPU or GPU)
            to hold all arrays, the grid is divided into `grid_chunks` parts,
            the functionals are evaluated on the smaller chunks of the
            grid and are summed into the Hamiltonian matrix at the end. Use
            "auto" or None to choose a chunk count from a conservative memory
            estimate.
        """
        self.grid_level = grid_level
        # Coordinate grid is divided in `grid_chunks` chunks to reduce memory footprint.
        if isinstance(grid_chunks, int):
            assert grid_chunks >= 1, "`grid_chunks` should be an integer >= 1"
        elif grid_chunks is not None and grid_chunks != "auto":
            raise ValueError("`grid_chunks` should be an integer >= 1, 'auto', or None")
        self.grid_chunks = grid_chunks
        self.target_memory_fraction = target_memory_fraction
        if min_grid_points_per_chunk < 1:
            raise ValueError("`min_grid_points_per_chunk` should be >= 1")
        if max_grid_points_per_chunk is not None and max_grid_points_per_chunk < min_grid_points_per_chunk:
            raise ValueError("`max_grid_points_per_chunk` should be >= `min_grid_points_per_chunk`")
        self.min_grid_points_per_chunk = min_grid_points_per_chunk
        self.max_grid_points_per_chunk = max_grid_points_per_chunk
        self._resolved_grid_chunks = None

        self.mol = mol
        # generate a multicenter integration grid
        self.grids = pyscf.dft.gen_grid.Grids(mol)
        self.grids.level = grid_level
        self.grids.build()
        # The external potential energy and the Hartree-part
        # of the electron-electron repulsion are calculated
        # using the AO representation of the matrix density.
        self.nuclear_ao = NuclearFunctionalAO(mol)
        self.hartree_ao = HartreeFunctionalAO(mol)
        # If no orbital-free kinetic functional is provided (kinetic_functional = None)
        # the kinetic energy is computed from the wavefunctions
        self.kinetic_ao = KineticFunctionalAO(mol)
        # Kinetic, correlation and exchange energies are
        # computed on a real-space grid.
        self.kinetic = kinetic_functional
        self.exchange = exchange_functional
        self.correlation = correlation_functional
        self.exchange_correlation = exchange_correlation_functional
        # Exact exchange (EXX)
        self.exact_exchange = exact_exchange_functional
        # Should the exchange energy be calculated from
        # (i) the total matrix density Dᵢⱼ=Dᵅᵢⱼ+Dᵝᵢⱼ as
        #   Xᵢⱼ = 2 * X[D/2]ᵢⱼ  if spin_type == UNPOLARIZED
        # or
        # (ii) for each spin type separately,
        #   Xᵢⱼ = X[Dᵅ]ᵢⱼ + X[Dᵝ]ᵢⱼ  if spin_type == POLARIZED
        # or
        # (iii) from the supermatrix
        #                       (Dᵅᵅᵢⱼ Dᵅᵝᵢⱼ)
        #   Xᵢⱼ = spin_trace{ X[(           )] }
        #                       (Dᵝᵅᵢⱼ  Dᵝᵝᵢⱼ)
        # if spin_type == INVARIANT
        # or
        # (iv) as an equal mix of the unpolarized and invariant exchange energies
        #   Xᵢⱼ = 1/2 * (X^{unpol} + X^{inv})
        #                                              (Dᵅᵅ Dᵅᵝ)
        #       = 1/2 * ( 2 * X[D/2]ᵢⱼ + spin_trace{ X[(       )] }ᵢⱼ )
        #                                              (Dᵝᵅ  Dᵝᵝ)
        # if spin_type == INVARIANT_MIX
        # ?
        if spin_type not in SpinType:
            raise ValueError(f"`spin_type` must be instance of SpinType, got {spin_type}")
        self.spin_type = spin_type
        self.need_spin_vector = need_spin_vector
        self.spin_vector_regularization = spin_vector_regularization
        # Does any of the functionals need the Laplacian of the density?
        self.need_laplacian = (
            getattr(self.kinetic, "need_laplacian", False) or
            getattr(self.exchange, "need_laplacian", False) or
            getattr(self.correlation, "need_laplacian", False) or
            getattr(self.exchange_correlation, "need_laplacian", False)
        )
        # Does any of the functionals need the gradient of the density?
        # A Laplacian request also requires AO gradients for the 2∇χᵦ·∇χᵧ term.
        self.need_gradient = (
            getattr(self.kinetic, "need_gradient", False) or
            getattr(self.exchange, "need_gradient", False) or
            getattr(self.correlation, "need_gradient", False) or
            getattr(self.exchange_correlation, "need_gradient", False) or
            self.need_laplacian
        )
        self._ao_grid_cache = None
        self._ao_grid_cache_key = None

    def __call__(self, msmd: MultistateMatrixDensity) -> Tensor:
        return self.matrix_elements(msmd)

    @staticmethod
    def _scale_optional(value: Tensor | None, factor: float) -> Tensor | None:
        if value is None:
            return None
        return value * factor

    def _ao_derivative_level(self) -> int:
        if self.need_laplacian:
            return 2
        if self.need_gradient:
            return 1
        return 0

    def _auto_grid_chunks_requested(self) -> bool:
        return self.grid_chunks is None or self.grid_chunks == "auto"

    def _ao_cache_factor(self) -> int:
        if self.need_laplacian:
            return 5
        if self.need_gradient:
            return 4
        return 1

    def _matrix_channel_factor(self) -> int:
        if self.spin_type == SpinType.POLARIZED:
            return 4
        if self.spin_type == SpinType.NONCOLLINEAR:
            return 6
        return 3

    def _memory_safety_factor(self) -> float:
        if self.need_laplacian:
            return 8.0
        if self.need_gradient:
            return 6.0
        return 4.0

    def _estimate_bytes_per_grid_point(self, dtype: torch.dtype, nstate: int) -> int:
        element_size = torch.empty((), dtype=dtype).element_size()
        nao = self.mol.nao_nr()
        ao_values = nao * self._ao_cache_factor()
        matrix_values = self._matrix_channel_factor() * nstate * nstate
        eig_values = 3 * nstate
        return int(element_size * self._memory_safety_factor() * (ao_values + matrix_values + eig_values))

    def _available_memory_bytes(self, device: torch.device) -> int | None:
        if device.type == "cuda" and torch.cuda.is_available():
            free_bytes, _total_bytes = torch.cuda.mem_get_info(device)
            fraction = self.target_memory_fraction if self.target_memory_fraction is not None else 0.35
            return int(free_bytes * fraction)

        fraction = self.target_memory_fraction if self.target_memory_fraction is not None else 0.50
        try:
            import psutil
            return int(psutil.virtual_memory().available * fraction)
        except ImportError:
            return int(2 * 1024**3 * fraction)

    def _resolve_grid_chunks(self, dtype: torch.dtype, device: torch.device, msmd: MultistateMatrixDensity = None) -> int:
        if not self._auto_grid_chunks_requested():
            return self.grid_chunks

        if msmd is None:
            # The state dimension is needed for memory-aware chunking. If no
            # matrix density is available, keep a conservative deterministic fallback.
            return 1

        ngrids = len(self.grids.coords)
        if ngrids <= 1:
            return 1

        bytes_per_grid = max(1, self._estimate_bytes_per_grid_point(dtype, msmd.number_of_states))
        available_bytes = self._available_memory_bytes(device)
        if available_bytes is None or available_bytes <= 0:
            return 1

        points_per_chunk = max(1, available_bytes // bytes_per_grid)
        points_per_chunk = min(points_per_chunk, ngrids)
        points_per_chunk = max(points_per_chunk, self.min_grid_points_per_chunk)
        if self.max_grid_points_per_chunk is not None:
            points_per_chunk = min(points_per_chunk, self.max_grid_points_per_chunk)
        chunks = int(numpy.ceil(ngrids / points_per_chunk))
        return max(1, chunks)

    @property
    def resolved_grid_chunks(self) -> int | None:
        return self._resolved_grid_chunks

    def _get_ao_grid_cache(self, dtype: torch.dtype, device: torch.device, msmd: MultistateMatrixDensity = None):
        resolved_grid_chunks = self._resolve_grid_chunks(dtype, device, msmd=msmd)
        self._resolved_grid_chunks = resolved_grid_chunks
        cache_key = (
            dtype, device, self.grid_chunks, resolved_grid_chunks,
            self.need_gradient, self.need_laplacian, self.target_memory_fraction,
            self.max_grid_points_per_chunk, self.min_grid_points_per_chunk,
        )
        if self._ao_grid_cache_key == cache_key and self._ao_grid_cache is not None:
            return self._ao_grid_cache

        deriv = self._ao_derivative_level()
        batches = []
        for coords, weights in zip(
                numpy.array_split(self.grids.coords, resolved_grid_chunks),
                numpy.array_split(self.grids.weights, resolved_grid_chunks)):
            ao_value_all = numint.eval_ao(self.mol, coords, deriv=deriv)
            ao_value_all = torch.from_numpy(ao_value_all).to(dtype=dtype, device=device)
            weights = torch.from_numpy(weights).to(dtype=dtype, device=device)

            if deriv == 0:
                ao_value = ao_value_all
                grad_ao_value = None
                lapl_ao_value = None
            else:
                ao_value = ao_value_all[0,:,:]
                grad_ao_value = ao_value_all[1:4,:,:]
                if self.need_laplacian:
                    lapl_ao_value = ao_value_all[4,:,:] + ao_value_all[7,:,:] + ao_value_all[9,:,:]
                else:
                    lapl_ao_value = None

            batches.append(AOGridBatch(
                weights=weights,
                ao_value=ao_value,
                grad_ao_value=grad_ao_value,
                lapl_ao_value=lapl_ao_value,
            ))

        self._ao_grid_cache_key = cache_key
        self._ao_grid_cache = batches
        return batches

    @staticmethod
    def _matrix_inverse_square_root(matrix: Tensor, epsilon: float) -> Tensor:
        return matrix_inverse_square_root(matrix, epsilon)

    @staticmethod
    def _matrix_square_root_from_square(matrix_square: Tensor, epsilon: float) -> Tensor:
        return matrix_square_root_from_square(matrix_square, epsilon)

    def _noncollinear_channel_densities(self, spin_D: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        return noncollinear_channel_densities(
            spin_D,
            spin_vector_regularization=self.spin_vector_regularization,
            need_spin_vector=self.need_spin_vector,
        )

    def _evaluate_spin_unpolarized_xc(self, D: Tensor, grad_D: Tensor | None, lapl_D: Tensor | None) -> Tensor:
        if self.exchange_correlation is not None:
            return self.exchange_correlation(D, grad_D, lapl_D)
        x = 0.0
        c = 0.0
        if self.exchange is not None:
            x = 2.0 * self.exchange(
                D/2.0, self._scale_optional(grad_D, 0.5),
                None if lapl_D is None else lapl_D/2.0)
        if self.correlation is not None:
            c = self.correlation(D, grad_D, lapl_D)
        return x + c

    def _evaluate_spin_polarized_xc(self, spin_D: Tensor, grad_spin_D: Tensor | None, lapl_spin_D: Tensor | None, D: Tensor, grad_D: Tensor | None, lapl_D: Tensor | None) -> Tensor:
        if self.exchange_correlation is not None:
            raise NotImplementedError(
                "Fused exchange-correlation callables are only supported for "
                "SpinType.UNPOLARIZED. Use separate exchange and correlation "
                "callables for spin-polarized calculations."
            )
        x = 0.0
        c = 0.0
        if self.exchange is not None:
            if lapl_spin_D is None:
                x = (
                    self.exchange(spin_D[0,0,...], None if grad_spin_D is None else grad_spin_D[0,0,...], None) +
                    self.exchange(spin_D[1,1,...], None if grad_spin_D is None else grad_spin_D[1,1,...], None)
                )
            else:
                x = (
                    self.exchange(spin_D[0,0,...], grad_spin_D[0,0,...], lapl_spin_D[0,0,...]) +
                    self.exchange(spin_D[1,1,...], grad_spin_D[1,1,...], lapl_spin_D[1,1,...])
                )
        if self.correlation is not None:
            if getattr(self.correlation, "need_spin_density", False):
                c = self.correlation(
                    torch.einsum('ss...->s...', spin_D),
                    None if grad_spin_D is None else torch.einsum('ss...->s...', grad_spin_D),
                    None if lapl_spin_D is None else torch.einsum('ss...->s...', lapl_spin_D),
                    spin_polarized=True,
                )
            else:
                c = self.correlation(D, grad_D, lapl_D)
        return x + c

    def _evaluate_noncollinear_xc(self, spin_D: Tensor, D: Tensor, grad_D: Tensor | None, lapl_D: Tensor | None) -> Tensor:
        return evaluate_noncollinear_lmda_xc(
            spin_D,
            D,
            grad_D,
            lapl_D,
            exchange=self.exchange,
            correlation=self.correlation,
            exchange_correlation=self.exchange_correlation,
            spin_vector_regularization=self.spin_vector_regularization,
            need_spin_vector=self.need_spin_vector,
        )

    def matrix_elements(self, msmd: MultistateMatrixDensity) -> Tensor:
        """
        Compute the Hamiltonian matrix H[D(r)]ᵢⱼ in the basis of electronic states at a
        given matrix density D(r).

        :param msmd: multistate matrix density
        :type msmd: :class:`~.MultistateMatrixDensity`

        :return hamiltonian: Hamiltonian matrix Hᵢⱼ
        :rtype hamiltonian: Tensor of shape (Nstate,Nstate)
        """
        # Check that the same geometry and basis is used for defining
        # the Hamiltonian and the matrix density.
        assert id(self.mol) == id(msmd.mol)
        check_not_deprecated_spin_type(self.spin_type)
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
                    K = 2.0*self.exact_exchange(dm/2.0)
                case SpinType.POLARIZED:
                    K = self.exact_exchange(spin_dm[0,0,...]) + self.exact_exchange(spin_dm[1,1,...])
                case SpinType.NONCOLLINEAR:
                    raise NotImplementedError(
                        "Exact exchange for SpinType.NONCOLLINEAR is not implemented yet."
                    )
                case _:
                    raise ValueError(f"`spin_type` must be instance of SpinType, got {self.spin_type}")
            H = H + K

        # Loop over cached chunks of grid points and associated integration weights.
        for ao_grid_batch in self._get_ao_grid_cache(dtype=spin_dm.dtype, device=spin_dm.device, msmd=msmd):
            # evaluate semilocal kinetic, exchange and correlation functionals
            # Inputs are D(r), and optionally ∇D(r) and ∇²D(r), on the integration grid.
            spin_D, grad_spin_D, lapl_spin_D = msmd.evaluate_from_ao_batch(
                ao_grid_batch,
                dm_ao=spin_dm,
                need_gradient=self.need_gradient,
                need_laplacian=self.need_laplacian)

            # Sum over spin
            D = torch.einsum('ss...->...', spin_D)
            if grad_spin_D is not None:
                grad_D = torch.einsum('ss...->...', grad_spin_D)
            else:
                grad_D = None
            # If ∇²D is not needed by any functional, it is set to None.
            if lapl_spin_D is not None:
                lapl_D = torch.einsum('ss...->...', lapl_spin_D)
            else:
                lapl_D = None

            # kinetic (t) and exchange-correlation (xc) energy densities
            if self.kinetic is not None:
                t = self.kinetic(D, grad_D, lapl_D)

            if self.spin_type == SpinType.UNPOLARIZED:
                xc = self._evaluate_spin_unpolarized_xc(D, grad_D, lapl_D)
            elif self.spin_type == SpinType.POLARIZED:
                xc = self._evaluate_spin_polarized_xc(spin_D, grad_spin_D, lapl_spin_D, D, grad_D, lapl_D)
            elif self.spin_type == SpinType.NONCOLLINEAR:
                xc = self._evaluate_noncollinear_xc(spin_D, D, grad_D, lapl_D)
            else:
                raise ValueError(f"Unsupported spin_type {self.spin_type}")

            # integrate energy densities
            weights = ao_grid_batch.weights
            # reshape weights so that they can be multiplied with matrices
            # using the broadcasting rules
            #   weights -> (...,1,1)
            #   t          (...,n,n)
            weights = weights.unsqueeze(-1).unsqueeze(-1)
            XC = torch.sum(xc * weights, 0)

            # Add exchange and correlation energy to Hamiltonian
            H = H + XC

            if self.kinetic is not None:
                T = torch.sum(t * weights, 0)
                # Add kinetic energy from orbital-free functional
                H = H + T

        # Check for NaN's
        if H.isnan().any():
            print("Hamiltonian")
            print(H)
            raise RuntimeError(
                "Hamiltonian contains NaN's, there is probably a bug in any of the functionals."
            )

        return H


class HamiltonianTargetStateLMDA(Hamiltonian):
    """Target-state Hamiltonian components with cached MO one-electron terms."""

    def __init__(
        self,
        mol: pyscf.gto.Mole,
        exchange_functional: Callable = lda_x_dirac,
        correlation_functional: Callable = lda_c_chachiyo,
        exchange_correlation_functional: Callable = None,
        spin_type: SpinType = SpinType.UNPOLARIZED,
        grid_level: int = 3,
        grid_chunks: int = 1,
        hartree_backend: str = "ao",
        auxbasis=None,
        allow_hartree_fallback: bool = True,
        spin_vector_regularization: float = 1.0e-10,
        omega: float | None = None,
        short_range_exchange_correlation_functional: Callable = None,
    ):
        self.mol = mol
        self.exchange = exchange_functional
        self.correlation = correlation_functional
        self.exchange_correlation = exchange_correlation_functional
        self.short_range_exchange_correlation = short_range_exchange_correlation_functional
        self.spin_type = spin_type
        if spin_type not in SpinType:
            raise ValueError(f"`spin_type` must be instance of SpinType, got {spin_type}")
        if spin_type not in {SpinType.UNPOLARIZED, SpinType.NONCOLLINEAR}:
            raise NotImplementedError(
                "HamiltonianTargetStateLMDA currently supports SpinType.UNPOLARIZED "
                "and SpinType.NONCOLLINEAR only."
            )
        if spin_type == SpinType.NONCOLLINEAR and exchange_correlation_functional is not None:
            raise NotImplementedError(
                "Fused exchange-correlation callables are only supported for "
                "SpinType.UNPOLARIZED. Use separate exchange and correlation "
                "callables for noncollinear calculations."
            )
        self.spin_vector_regularization = spin_vector_regularization
        if not isinstance(grid_chunks, int) or grid_chunks < 1:
            raise ValueError("HamiltonianTargetStateLMDA grid_chunks must be an integer >= 1.")
        self.grid_level = grid_level
        self.grid_chunks = grid_chunks
        self.grids = pyscf.dft.gen_grid.Grids(mol)
        self.grids.level = grid_level
        self.grids.build()
        self.one_electron_cache = OneElectronIntegralCache(mol)
        self.omega = omega
        self.range_separated = omega is not None
        if self.range_separated:
            if omega < 0.0:
                raise ValueError(f"omega must be non-negative, got {omega!r}")
            if exchange_correlation_functional is not None and short_range_exchange_correlation_functional is not None:
                raise ValueError(
                    "Pass the range-separated scalar functional as "
                    "short_range_exchange_correlation_functional; "
                    "exchange_correlation_functional is reserved for full-range mode."
                )
            if short_range_exchange_correlation_functional is None:
                raise ValueError(
                    "Range-separated mode requires a complementary short-range "
                    "exchange-correlation functional."
                )
            self.exchange_correlation = short_range_exchange_correlation_functional
            self.long_range_interaction = LongRangeInteractionFunctional(mol, omega)
        else:
            self.long_range_interaction = None
        self.hartree_backend = hartree_backend
        if self.range_separated:
            if hartree_backend != "ao":
                raise NotImplementedError(
                    "Range-separated Hartree currently uses the direct AO ERI backend."
                )
            self.hartree = HartreeFunctionalShortRangeAO(mol, omega)
        elif hartree_backend == "ao":
            self.hartree = HartreeFunctionalAO(mol)
        elif hartree_backend == "gpu4pyscf_df":
            self.hartree = HartreeFunctionalGpu4PySCFDFAO(
                mol, auxbasis=auxbasis, allow_fallback=allow_hartree_fallback)
        else:
            raise ValueError("hartree_backend must be 'ao' or 'gpu4pyscf_df'")

    def __call__(self, msmd: TargetStateMultistateMatrixDensityCAS) -> Tensor:
        return self.matrix_elements(msmd)

    def one_electron_matrix(self, msmd: TargetStateMultistateMatrixDensityCAS) -> Tensor:
        gamma = msmd.transition_1rdm_mo()
        return self.one_electron_cache.matrix_elements_from_gamma(gamma, msmd.orbital_coefficients())

    def hartree_matrix(self, msmd: TargetStateMultistateMatrixDensityCAS) -> Tensor:
        spin_dm = msmd.density_matrices_ao()
        dm = torch.einsum('ss...->...', spin_dm)
        return self.hartree(dm)

    def long_range_interaction_matrix(self, msmd: TargetStateMultistateMatrixDensityCAS) -> Tensor:
        """Return the explicit long-range two-electron matrix, or zeros."""
        if self.long_range_interaction is None:
            return torch.zeros(
                (msmd.number_of_states, msmd.number_of_states),
                dtype=msmd.orbital_rotation_params.dtype,
                device=msmd.orbital_rotation_params.device,
            )
        return self.long_range_interaction(msmd)

    def exchange_correlation_matrix(self, msmd: TargetStateMultistateMatrixDensityCAS) -> Tensor:
        gamma_mo = msmd.transition_1rdm_mo()
        Hxc = torch.zeros(
            (msmd.number_of_states, msmd.number_of_states),
            dtype=gamma_mo.dtype,
            device=gamma_mo.device,
        )
        for coords, weights in zip(
                numpy.array_split(self.grids.coords, self.grid_chunks),
                numpy.array_split(self.grids.weights, self.grid_chunks)):
            ao_value = numint.eval_ao(self.mol, coords, deriv=0)
            ao_value = torch.from_numpy(ao_value).to(dtype=gamma_mo.dtype, device=gamma_mo.device)
            weights = torch.from_numpy(weights).to(dtype=gamma_mo.dtype, device=gamma_mo.device)
            spin_D, _grad_spin_D, _lapl_spin_D = msmd.evaluate_from_mo_grid_batch(
                AOGridBatch(weights=weights, ao_value=ao_value), gamma_mo=gamma_mo)
            D = torch.einsum('ss...->...', spin_D)
            if self.spin_type == SpinType.UNPOLARIZED and self.exchange_correlation is not None:
                xc = self.exchange_correlation(D, None, None)
            elif self.spin_type == SpinType.UNPOLARIZED:
                xc = 0.0
                if self.exchange is not None:
                    xc = xc + 2.0 * self.exchange(D / 2.0, None, None)
                if self.correlation is not None:
                    xc = xc + self.correlation(D, None, None)
            elif self.spin_type == SpinType.NONCOLLINEAR:
                xc = evaluate_noncollinear_lmda_xc(
                    spin_D,
                    D,
                    None,
                    None,
                    exchange=self.exchange,
                    correlation=self.correlation,
                    exchange_correlation=self.exchange_correlation,
                    spin_vector_regularization=self.spin_vector_regularization,
                )
            else:
                raise ValueError(f"Unsupported spin_type {self.spin_type}")
            Hxc = Hxc + torch.sum(xc * weights.unsqueeze(-1).unsqueeze(-1), dim=0)
        return Hxc

    def matrix_elements(self, msmd: TargetStateMultistateMatrixDensityCAS) -> Tensor:
        assert id(self.mol) == id(msmd.mol)
        Vnn = self.mol.get_enuc() * torch.eye(
            msmd.number_of_states,
            dtype=msmd.orbital_rotation_params.dtype,
            device=msmd.orbital_rotation_params.device,
        )
        return (
            Vnn +
            self.one_electron_matrix(msmd) +
            self.hartree_matrix(msmd) +
            self.exchange_correlation_matrix(msmd) +
            self.long_range_interaction_matrix(msmd)
        )


def minimize_subspace_energy(
    hamiltonian: Hamiltonian,
    msmd: MultistateMatrixDensity,
    state_average=None,
    optimizer="wrapped",
    enforce_invariants=True,
    convergence=None,
    return_info=False,
    **optimizer_kwds
):
    """
    Find the matrix density of the lowest few electronic states by
    minimizing the staged-averaged energy.

    :param state_average: number of states M to average over
        If `state_average` is None, the average extends over all states (M=N)
    :type state_average: int <= N or None
    """
    optimizer_name = optimizer
    if optimizer_name in ["wrapped", "numpy", "bfgs"]:
        optimizer = WrappedOptimizer(msmd.parameters(), **optimizer_kwds)
    elif optimizer_name == "torch_lbfgs":
        optimizer = TorchLBFGSOptimizer(msmd.parameters(), **optimizer_kwds)
    else:
        raise ValueError("optimizer must be 'wrapped' or 'torch_lbfgs'")
    # Enable all parameters for optimization.
    for param in msmd.parameters():
        param.requires_grad = True
    if enforce_invariants and hasattr(msmd, "enforce_invariants"):
        msmd.enforce_invariants()

    # objective function
    convergence = convergence or {}
    closure_history = []

    def _grad_norm():
        parts = []
        for param in msmd.parameters():
            if param.grad is not None:
                parts.append(torch.linalg.norm(param.grad.detach()))
        if not parts:
            return 0.0
        return float(torch.linalg.norm(torch.stack(parts)).detach().cpu())

    def closure():
        optimizer.zero_grad()
        # compute objective function, that is the subspace
        # energy to minimize.
        H = hamiltonian(msmd)
        # Average energy of the lowest M electronic states to get the subspace energy
        #
        #  E = 1/M ∑ₐ λₐ   a=1,...,M   λ(1) <= λ(2) <= ...<= λ(M) < ... <= λ(N)
        #
        # where λₐ are the eigenvalues of the Hamiltonian H[D(r)].
        #
        # If M=`state_average` is not specified (None), all states are included in the average,
        #
        #   E = 1/N tr(H) = 1/N ∑ᵢ H[D(r)]ᵢᵢ
        #
        # NOTE: In POLARIZED/UNPOLARIZED calculations,
        # the subspace only includes wavefunctions with one value of Sz.
        # Therefore, if the subspace contains states with different spins Sᵢ,
        # the state energies have to be weighted with the spin multiplicity, 2*Sᵢ+1.
        # The total number of states, counting all components of the spin multiplets,
        # is given by
        #
        #   N =  ∑ᵢ (2*Sᵢ+1)
        #
        # and the average energy of the subspace is
        #
        #   E = 1/N ∑ᵢ (2*Sᵢ+1) H[D(r)]ᵢᵢ
        #
        # In INVARIANT calculations, all components of the multiplet are present,
        # and the state weights are set to 1.
        with torch.no_grad():
            # The weight is an integer, it does not require gradients.
            weights = msmd.state_weights()
        subspace_energy = MF.trace_average(H, weights=weights, subspace_dim=state_average)

        # There is no need to do a backward pass at this point.
        # The optimizer will do that if needed. The line searches
        # to find the best step length do not require gradients.

        if return_info or convergence:
            subspace_energy.backward(retain_graph=True)
            closure_history.append({
                "loss": float(subspace_energy.detach().cpu()),
                "grad_norm": _grad_norm(),
            })
            optimizer.zero_grad()
        return subspace_energy

    # Find stationary point of subspace energy
    optimizer.step(closure)
    msmd.optimizer_telemetry = optimizer.telemetry.as_dict()
    loss_delta = None
    if len(closure_history) >= 2:
        loss_delta = abs(closure_history[-1]["loss"] - closure_history[-2]["loss"])
    final_grad_norm = closure_history[-1]["grad_norm"] if closure_history else None
    final_loss = closure_history[-1]["loss"] if closure_history else None
    gtol = convergence.get("gtol")
    ftol = convergence.get("ftol")
    converged_grad = None if gtol is None or final_grad_norm is None else final_grad_norm <= gtol
    converged_loss = None if ftol is None or loss_delta is None else loss_delta <= ftol
    converged_checks = [value for value in (converged_grad, converged_loss) if value is not None]
    msmd.optimizer_convergence = {
        "closure_count": len(closure_history),
        "final_loss": final_loss,
        "final_grad_norm": final_grad_norm,
        "loss_delta": loss_delta,
        "gtol": gtol,
        "ftol": ftol,
        "converged_grad": converged_grad,
        "converged_loss": converged_loss,
        "converged": all(converged_checks) if converged_checks else None,
        "history_tail": closure_history[-10:],
    }
    if enforce_invariants and hasattr(msmd, "enforce_invariants"):
        msmd.enforce_invariants()

    # Diagonalize Hamiltonian at stationary point ...
    H = hamiltonian(msmd)
    eigvals, eigvecs = torch.linalg.eigh(H)

    # ... and transform optimized matrix density to the basis of the eigenstates
    # D(r) -> Uᵀ D(r) U
    msmd.basis_transformation(eigvecs.T)
    if enforce_invariants and hasattr(msmd, "enforce_invariants"):
        msmd.enforce_invariants()

    if return_info:
        return eigvals, msmd, {
            "optimizer_telemetry": msmd.optimizer_telemetry,
            "optimizer_convergence": msmd.optimizer_convergence,
        }
    return eigvals, msmd
