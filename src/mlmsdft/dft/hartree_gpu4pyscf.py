# -*- coding: utf-8 -*-
"""Optional GPU4PySCF density-fitted Hartree backend.

The class keeps the same public interface as `HartreeFunctionalAO`.  If
GPU4PySCF is unavailable and `allow_fallback=True`, it delegates to the current
CPU PySCF AO-J backend so callers can make fallback behavior explicit.
"""
from __future__ import annotations

import importlib.util
from typing import Callable, Iterable

import pyscf.gto
import torch
from torch import Tensor
from torch.autograd import Function
from torch.autograd.function import once_differentiable

from mlmsdft.dft.hartree import HartreeFunctionalAO


def gpu4pyscf_environment_probe(mol: pyscf.gto.Mole, auxbasis=None) -> dict:
    """Return a non-throwing summary of the optional GPU4PySCF runtime."""
    info = {
        "gpu4pyscf_available": importlib.util.find_spec("gpu4pyscf") is not None,
        "cupy_available": importlib.util.find_spec("cupy") is not None,
        "torch_cuda_available": torch.cuda.is_available(),
        "df_build_succeeded": False,
        "error": None,
    }
    try:
        import pyscf
        info["pyscf_version"] = getattr(pyscf, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - defensive probe path
        info["pyscf_version_error"] = str(exc)
    info["torch_version"] = torch.__version__
    if info["cupy_available"]:
        try:
            import cupy
            info["cupy_version"] = getattr(cupy, "__version__", "unknown")
            info["cupy_cuda_device_count"] = cupy.cuda.runtime.getDeviceCount()
        except Exception as exc:  # pragma: no cover - depends on CUDA runtime
            info["cupy_error"] = str(exc)
    if info["gpu4pyscf_available"]:
        try:
            import gpu4pyscf
            import gpu4pyscf.df.df
            info["gpu4pyscf_version"] = getattr(gpu4pyscf, "__version__", "unknown")
            gpu4pyscf.df.df.DF(mol, auxbasis=auxbasis).build()
            info["df_build_succeeded"] = True
        except Exception as exc:  # pragma: no cover - depends on optional package
            info["error"] = str(exc)
    return info


def _require_cupy():
    if importlib.util.find_spec("cupy") is None:
        raise ImportError("cupy is required for the direct GPU4PySCF DF-J backend.")
    import cupy
    return cupy


def torch_cuda_to_cupy(tensor: Tensor):
    """Convert a CUDA torch tensor to a CuPy array through DLPack."""
    if not tensor.is_cuda:
        raise ValueError("Direct GPU4PySCF DF-J requires CUDA torch tensors; use explicit fallback for CPU tensors.")
    cupy = _require_cupy()
    tensor = tensor.contiguous()
    return cupy.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))


def cupy_to_torch_cuda(array, *, dtype: torch.dtype, device: torch.device) -> Tensor:
    """Convert a CuPy array to a CUDA torch tensor through DLPack."""
    tensor = torch.from_dlpack(array)
    return tensor.to(dtype=dtype, device=device)


def _upper_triangular_density_list(density_matrices_ao: Tensor) -> tuple[list[tuple[int, int]], Tensor]:
    nbasis_a, nbasis_b, nstate_i, nstate_j = density_matrices_ao.size()
    if nbasis_a != nbasis_b or nstate_i != nstate_j:
        raise ValueError(
            "density_matrices_ao must have shape (nao, nao, nstate, nstate), "
            f"got {tuple(density_matrices_ao.size())}."
        )
    pairs = []
    dms = []
    for i in range(nstate_i):
        for j in range(i, nstate_i):
            pairs.append((i, j))
            dms.append(density_matrices_ao[:, :, i, j])
    return pairs, torch.stack(dms, dim=0)


def _coerce_vj_sequence(vjs) -> Iterable:
    if isinstance(vjs, tuple):
        vjs = vjs[0]
    return vjs


def _hartree_from_potentials(density_matrices_ao: Tensor, hartree_potentials_ao: Tensor) -> Tensor:
    return 0.5 * torch.einsum('bgik,bgkj->ij', density_matrices_ao, hartree_potentials_ao)


def _gradient_from_potentials(grad_output: Tensor, hartree_potentials_ao: Tensor) -> Tensor:
    return 0.5 * (
        torch.einsum('mj,bgjn->bgmn', grad_output, hartree_potentials_ao) +
        torch.einsum('bgmj,jn->bgmn', hartree_potentials_ao, grad_output))


def _build_df_j_potentials(density_matrices_ao: Tensor, dfobj, j_builder: Callable = None) -> Tensor:
    pairs, dm_stack = _upper_triangular_density_list(density_matrices_ao)
    cp_dm_stack = torch_cuda_to_cupy(dm_stack)
    if j_builder is None:
        def j_builder(dm_batch):
            return dfobj.get_jk(dm_batch, with_j=True, with_k=False)
    vjs = _coerce_vj_sequence(j_builder(cp_dm_stack))
    vj_batch = cupy_to_torch_cuda(vjs, dtype=density_matrices_ao.dtype, device=density_matrices_ao.device)
    if vj_batch.size(0) != len(pairs):
        raise RuntimeError(
            "GPU4PySCF DF-J returned an unexpected number of J matrices: "
            f"{vj_batch.size(0)} != {len(pairs)}."
        )
    nbasis, _, nstate, _ = density_matrices_ao.size()
    hartree_potentials = torch.zeros(
        (nbasis, nbasis, nstate, nstate),
        dtype=density_matrices_ao.dtype,
        device=density_matrices_ao.device,
    )
    for pair_index, (i, j) in enumerate(pairs):
        vj = vj_batch[pair_index]
        hartree_potentials[:, :, i, j] = vj
        if i != j:
            hartree_potentials[:, :, j, i] = vj
    return hartree_potentials


class _HartreeFunctionalGpu4PySCFDFAO(Function):
    @staticmethod
    def forward(ctx, density_matrices_ao: Tensor, dfobj, j_builder: Callable = None) -> Tensor:
        hartree_potentials_ao = _build_df_j_potentials(density_matrices_ao, dfobj, j_builder=j_builder)
        ctx.save_for_backward(hartree_potentials_ao)
        return _hartree_from_potentials(density_matrices_ao, hartree_potentials_ao)

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        hartree_potentials_ao, = ctx.saved_tensors
        return _gradient_from_potentials(grad_output, hartree_potentials_ao), None, None


class HartreeFunctionalGpu4PySCFDFAO(torch.nn.Module):
    """Hartree backend selector for GPU4PySCF DF-J with explicit CPU fallback."""

    def __init__(
        self,
        mol: pyscf.gto.Mole,
        auxbasis=None,
        allow_fallback: bool = False,
        dfobj=None,
        j_builder: Callable = None,
    ):
        super().__init__()
        self.mol = mol
        self.auxbasis = auxbasis
        self.allow_fallback = allow_fallback
        self.backend_name = "gpu4pyscf_df"
        self._fallback = HartreeFunctionalAO(mol)
        self._j_builder = j_builder
        self._gpu4pyscf_available = importlib.util.find_spec("gpu4pyscf") is not None or dfobj is not None
        self._dfobj = dfobj
        if self._dfobj is not None:
            pass
        elif self._gpu4pyscf_available:
            import gpu4pyscf.df.df
            self._dfobj = gpu4pyscf.df.df.DF(mol, auxbasis=auxbasis).build()
        elif not allow_fallback:
            raise ImportError(
                "gpu4pyscf is not available. Install GPU4PySCF or construct "
                "HartreeFunctionalGpu4PySCFDFAO(..., allow_fallback=True)."
            )

    @property
    def using_fallback(self) -> bool:
        return self._dfobj is None

    def forward(self, density_matrices_ao: Tensor) -> Tensor:
        if self._dfobj is None:
            return self._fallback(density_matrices_ao)
        if not density_matrices_ao.is_cuda:
            if self.allow_fallback:
                return self._fallback(density_matrices_ao)
            raise ValueError(
                "Direct GPU4PySCF DF-J requires CUDA density matrices. "
                "Pass allow_fallback=True to use the AO PySCF backend on CPU."
            )
        return _HartreeFunctionalGpu4PySCFDFAO.apply(
            density_matrices_ao, self._dfobj, self._j_builder)
