# coding: utf-8
"""Cluster-invariant spectral utilities for fixed matrix densities."""
from dataclasses import dataclass

import torch
from torch import Tensor

__all__ = [
    "CarrierDomainError",
    "ClusteredSpectrum",
    "cluster_chi",
    "clustered_spectral_descriptors",
    "noncollinear_exchange_carriers",
    "noncollinear_spin_density",
    "reconstruct_cluster_matrix",
    "reconstruct_cluster_matrix_compact",
    "reconstruct_spectral_matrix",
    "robust_symmetric_eigh",
    "spin_partial_trace",
]


class CarrierDomainError(ValueError):
    """Raised when a density or carrier is outside the PSD domain."""


@dataclass(frozen=True)
class ClusteredSpectrum:
    """Fixed-slot clustered representation of a symmetric PSD matrix."""

    eigenvalues: Tensor
    eigenvectors: Tensor
    cluster_id: Tensor
    cluster_projectors: Tensor
    multiplicity: Tensor
    rho_cluster: Tensor
    rho: Tensor
    s_cluster: Tensor
    eta: Tensor
    slot_active: Tensor
    model_supported: Tensor
    psd_projection_count: Tensor
    max_cluster_diameter: Tensor


def _check_square_matrix(matrix: Tensor, name: str) -> None:
    if matrix.ndim < 2 or matrix.size(-2) != matrix.size(-1):
        raise ValueError(f"{name} must have shape (..., N, N)")


def _symmetry_tolerance(matrix: Tensor) -> Tensor:
    if not matrix.dtype.is_floating_point:
        raise TypeError("spectral matrix inputs must have a floating-point dtype")
    scale = torch.maximum(
        matrix.abs().amax(dim=(-2, -1), keepdim=True),
        torch.ones((), dtype=matrix.dtype, device=matrix.device),
    )
    return 100.0 * torch.finfo(matrix.dtype).eps * scale


def _validate_symmetry(matrix: Tensor, name: str, *, antisymmetric: bool = False) -> Tensor:
    _check_square_matrix(matrix, name)
    if not bool(torch.all(torch.isfinite(matrix))):
        raise CarrierDomainError(f"{name} contains non-finite values")
    transpose = matrix.transpose(-1, -2)
    residual = matrix + transpose if antisymmetric else matrix - transpose
    if bool(torch.any(residual.abs() > _symmetry_tolerance(matrix))):
        kind = "antisymmetric" if antisymmetric else "symmetric"
        raise CarrierDomainError(f"{name} must be {kind}")
    return 0.5 * (matrix - transpose) if antisymmetric else 0.5 * (matrix + transpose)


def robust_symmetric_eigh(matrix: Tensor) -> tuple[Tensor, Tensor]:
    """Diagonalize a finite symmetric matrix without concealing invalid input."""
    symmetric = _validate_symmetry(matrix, "matrix")
    return torch.linalg.eigh(symmetric)


def reconstruct_spectral_matrix(values: Tensor, vectors: Tensor) -> Tensor:
    """Reconstruct ``U diag(values) U.T`` for a batched spectrum."""
    return torch.einsum("...ia,...a,...ja->...ij", vectors, values, vectors)


def _project_psd(
    matrix: Tensor,
    *,
    name: str,
    psd_atol: float,
    psd_rtol: float,
    psd_unit_floor: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    values, vectors = robust_symmetric_eigh(matrix)
    scale = torch.maximum(
        values.abs().amax(dim=-1, keepdim=True),
        torch.as_tensor(psd_unit_floor, dtype=values.dtype, device=values.device),
    )
    tolerance = psd_atol + psd_rtol * scale
    if bool(torch.any(values < -tolerance)):
        raise CarrierDomainError(f"{name} is materially indefinite")
    projected = torch.where(values < 0.0, torch.zeros_like(values), values)
    projection_count = torch.sum(values < 0.0, dim=-1)
    return projected, vectors, projection_count, reconstruct_spectral_matrix(projected, vectors)


def clustered_spectral_descriptors(
    matrix: Tensor,
    *,
    cluster_rtol: float = 1.0e-10,
    cluster_atol: float = 1.0e-14,
    density_floor: float = 1.0e-14,
    cluster_density_floor: float = 1.0e-14,
    psd_atol: float = 1.0e-14,
    psd_rtol: float = 1.0e-10,
    psd_unit_floor: float = 1.0e-14,
) -> ClusteredSpectrum:
    """Build fixed-slot, diameter-bounded descriptors for a PSD matrix."""
    _check_square_matrix(matrix, "matrix")
    if matrix.size(-1) == 0:
        raise ValueError("matrix dimension N must be positive")
    with torch.no_grad():
        values, vectors, projection_count, _ = _project_psd(
            matrix,
            name="matrix",
            psd_atol=psd_atol,
            psd_rtol=psd_rtol,
            psd_unit_floor=psd_unit_floor,
        )
        n = values.size(-1)
        scale = torch.maximum(
            values.abs().amax(dim=-1, keepdim=True),
            torch.as_tensor(density_floor, dtype=values.dtype, device=values.device),
        )
        threshold = cluster_atol + cluster_rtol * scale

        cluster_id = torch.zeros_like(values, dtype=torch.long)
        cluster_start = values[..., 0]
        for index in range(1, n):
            new_cluster = values[..., index] - cluster_start > threshold.squeeze(-1)
            cluster_id[..., index] = cluster_id[..., index - 1] + new_cluster.to(torch.long)
            cluster_start = torch.where(new_cluster, values[..., index], cluster_start)

        slot_shape = (1,) * (values.ndim - 1) + (n, 1)
        slots = torch.arange(n, device=values.device).reshape(slot_shape)
        cluster_mask = cluster_id.unsqueeze(-2) == slots
        multiplicity = cluster_mask.sum(dim=-1)
        slot_active = multiplicity > 0
        mask = cluster_mask.to(values.dtype)
        cluster_projectors = torch.einsum(
            "...ik,...ck,...jk->...cij", vectors, mask, vectors
        )
        rho_cluster = torch.sum(mask * values.unsqueeze(-2), dim=-1) / multiplicity.clamp_min(1)
        rho = torch.sum(multiplicity.to(values.dtype) * rho_cluster, dim=-1, keepdim=True)
        total_supported = rho > density_floor
        safe_rho = rho.clamp_min(density_floor)
        s_cluster = torch.where(total_supported, rho_cluster / safe_rho, 0.0)
        model_supported = slot_active & total_supported & (rho_cluster > cluster_density_floor)

        if n == 1:
            eta = torch.zeros_like(rho)
        else:
            second_moment = torch.sum(
                multiplicity.to(values.dtype) * s_cluster.square(), dim=-1, keepdim=True
            )
            eta = (n * second_moment - 1.0) / (n - 1.0)
            eta = torch.where(total_supported, eta.clamp(0.0, 1.0), 0.0)

        positive_inf = torch.full_like(values.unsqueeze(-2), torch.inf)
        negative_inf = torch.full_like(values.unsqueeze(-2), -torch.inf)
        cluster_min = torch.where(cluster_mask, values.unsqueeze(-2), positive_inf).amin(dim=-1)
        cluster_max = torch.where(cluster_mask, values.unsqueeze(-2), negative_inf).amax(dim=-1)
        diameters = torch.where(slot_active, cluster_max - cluster_min, 0.0)
        max_cluster_diameter = diameters.amax(dim=-1)

    return ClusteredSpectrum(
        eigenvalues=values,
        eigenvectors=vectors,
        cluster_id=cluster_id,
        cluster_projectors=cluster_projectors,
        multiplicity=multiplicity,
        rho_cluster=rho_cluster,
        rho=rho,
        s_cluster=s_cluster,
        eta=eta,
        slot_active=slot_active,
        model_supported=model_supported,
        psd_projection_count=projection_count,
        max_cluster_diameter=max_cluster_diameter,
    )


def cluster_chi(
    s_cluster: Tensor,
    multiplicity: Tensor,
    slot_active: Tensor,
    total_supported: Tensor,
    width: float | Tensor,
) -> Tensor:
    """Evaluate multiplicity-aware near-degeneracy descriptors in ``[0, 1]``."""
    n = s_cluster.size(-1)
    if n == 1:
        return torch.zeros_like(s_cluster)
    width_tensor = torch.as_tensor(width, dtype=s_cluster.dtype, device=s_cluster.device)
    if not bool(torch.all(torch.isfinite(width_tensor))) or bool(torch.any(width_tensor <= 0.0)):
        raise ValueError("width must be finite and positive")
    gaps = s_cluster.unsqueeze(-1) - s_cluster.unsqueeze(-2)
    while width_tensor.ndim < gaps.ndim:
        width_tensor = width_tensor.unsqueeze(-1)
    width_squared = width_tensor.square()
    kernel = width_squared / (gaps.square() + width_squared)
    weighted = torch.sum(kernel * multiplicity.to(s_cluster.dtype).unsqueeze(-2), dim=-1) - 1.0
    chi = (weighted / (n - 1.0)).clamp(0.0, 1.0)
    supported = total_supported
    if supported.ndim == chi.ndim - 1:
        supported = supported.unsqueeze(-1)
    return torch.where(slot_active & supported, chi, 0.0)


def reconstruct_cluster_matrix(
    values: Tensor,
    projectors: Tensor,
    model_supported: Tensor,
) -> Tensor:
    """Reconstruct one scalar value per supported cluster projector."""
    supported_values = torch.where(model_supported, values, torch.zeros_like(values))
    return torch.einsum("...c,...cij->...ij", supported_values, projectors)


def reconstruct_cluster_matrix_compact(
    cluster_values: Tensor,
    eigenvectors: Tensor,
    cluster_id: Tensor,
    model_supported: Tensor,
) -> Tensor:
    """Reconstruct a cluster correction without explicit cluster projectors."""
    eigenchannel_values = torch.gather(cluster_values, -1, cluster_id)
    eigenchannel_supported = torch.gather(model_supported, -1, cluster_id)
    eigenchannel_values = torch.where(
        eigenchannel_supported, eigenchannel_values, torch.zeros_like(eigenchannel_values)
    )
    return reconstruct_spectral_matrix(eigenchannel_values, eigenvectors)


def _validated_pauli_channels(pauli_density: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if pauli_density.ndim < 3 or pauli_density.size(0) != 4:
        raise ValueError(
            "pauli_density must have shape (4, ..., N, N) with D0,D1,D2_imag_coeff,D3 channels"
        )
    D0, D1, D2_coeff, D3 = pauli_density
    if D1.shape != D0.shape or D2_coeff.shape != D0.shape or D3.shape != D0.shape:
        raise ValueError("all Pauli channels must have the same shape")
    return (
        _validate_symmetry(D0, "D0"),
        _validate_symmetry(D1, "D1"),
        _validate_symmetry(D2_coeff, "D2_imag_coeff", antisymmetric=True),
        _validate_symmetry(D3, "D3"),
    )


def noncollinear_spin_density(
    pauli_density: Tensor,
    *,
    psd_atol: float = 1.0e-14,
    psd_rtol: float = 1.0e-10,
    psd_unit_floor: float = 1.0e-14,
    return_diagnostics: bool = False,
):
    """Construct the real symmetric PSD full-spin density matrix."""
    D0, D1, D2_coeff, D3 = _validated_pauli_channels(pauli_density)
    upper = torch.cat((D0 + D3, D1 + D2_coeff), dim=-1)
    lower = torch.cat((D1 - D2_coeff, D0 - D3), dim=-1)
    spin_density = 0.5 * torch.cat((upper, lower), dim=-2)
    _, _, projection_count, projected = _project_psd(
        spin_density,
        name="full spin density",
        psd_atol=psd_atol,
        psd_rtol=psd_rtol,
        psd_unit_floor=psd_unit_floor,
    )
    if not return_diagnostics:
        return projected
    return projected, {
        "psd_projection_count": projection_count,
        "spin_density_psd_projection_count": projection_count,
    }


def spin_partial_trace(matrix: Tensor) -> Tensor:
    """Trace a full-spin matrix over its two spin blocks."""
    _check_square_matrix(matrix, "matrix")
    dimension = matrix.size(-1)
    if dimension % 2 != 0:
        raise ValueError("full-spin matrix dimension must be even")
    n = dimension // 2
    return matrix[..., :n, :n] + matrix[..., n:, n:]


def noncollinear_exchange_carriers(
    pauli_density: Tensor,
    *,
    spin_square_psd_atol: float = 1.0e-14,
    spin_square_psd_rtol: float = 1.0e-10,
    spin_square_unit_floor: float = 1.0e-14,
    carrier_psd_atol: float = 1.0e-14,
    carrier_psd_rtol: float = 1.0e-10,
    carrier_unit_floor: float = 1.0e-14,
    return_diagnostics: bool = False,
):
    """Construct PSD exchange carriers in the real Pauli-record convention."""
    D0, D1, D2_coeff, D3 = _validated_pauli_channels(pauli_density)

    spin_square = D1 @ D1 - D2_coeff @ D2_coeff + D3 @ D3
    spin_values, spin_vectors, spin_count, _ = _project_psd(
        spin_square,
        name="spin-square matrix",
        psd_atol=spin_square_psd_atol,
        psd_rtol=spin_square_psd_rtol,
        psd_unit_floor=spin_square_unit_floor,
    )
    Q = reconstruct_spectral_matrix(torch.sqrt(spin_values), spin_vectors)
    D_plus_raw = 0.5 * (D0 + Q)
    D_minus_raw = 0.5 * (D0 - Q)
    _, _, plus_count, D_plus = _project_psd(
        D_plus_raw,
        name="D_plus carrier",
        psd_atol=carrier_psd_atol,
        psd_rtol=carrier_psd_rtol,
        psd_unit_floor=carrier_unit_floor,
    )
    _, _, minus_count, D_minus = _project_psd(
        D_minus_raw,
        name="D_minus carrier",
        psd_atol=carrier_psd_atol,
        psd_rtol=carrier_psd_rtol,
        psd_unit_floor=carrier_unit_floor,
    )
    if not return_diagnostics:
        return D_plus, D_minus, D0
    diagnostics = {
        "spin_square_psd_projection_count": spin_count,
        "carrier_plus_psd_projection_count": plus_count,
        "carrier_minus_psd_projection_count": minus_count,
    }
    return D_plus, D_minus, D0, diagnostics
