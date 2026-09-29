"""Explicit selected Hamiltonian builder over arbitrary determinant sets (T026).

For small supports the Hamiltonian matrix is assembled explicitly (dense) and is
the correctness oracle.  Element evaluation uses the reference Slater-Condon
backend; Hermiticity is checked before diagonalization.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from .backend.slater_condon_reference import h_element
from .exceptions import InternalConsistencyError, ReferenceNotCompact
from .models import DeterminantKey, SectorSupport


def estimate_dense_bytes(n_det: int, complex_dtype: bool = True) -> int:
    """Estimate dense matrix storage in bytes."""
    per = 16 if complex_dtype else 8
    return per * n_det * n_det


def build_selected_hamiltonian(
    determinant_keys: Sequence[DeterminantKey],
    h1e: np.ndarray,
    eri: np.ndarray,
    hermiticity_tol: float = 1e-12,
    max_memory_mb: Optional[int] = None,
) -> np.ndarray:
    """Assemble the dense Hermitian selected Hamiltonian matrix.

    ``h1e`` is ``[norb, norb]``, ``eri`` is ``[norb]*4`` in PySCF convention.
    """
    keys = list(determinant_keys)
    n = len(keys)
    if n == 0:
        raise ReferenceNotCompact("empty determinant set")
    if max_memory_mb is not None:
        bytes_ = estimate_dense_bytes(n)
        if bytes_ > max_memory_mb * 1024 * 1024:
            raise ReferenceNotCompact(
                "dense selected Hamiltonian exceeds memory guard",
                {"n_det": n, "bytes": bytes_, "max_memory_mb": max_memory_mb},
            )

    h = np.zeros((n, n), dtype=complex)
    for i in range(n):
        ki = keys[i]
        h[i, i] = h_element(ki, ki, h1e, eri)
        for j in range(i):
            x = h_element(ki, keys[j], h1e, eri)
            h[i, j] = x
            h[j, i] = np.conj(x)

    herm = np.linalg.norm(h - h.conj().T)
    if herm > hermiticity_tol:
        raise InternalConsistencyError(
            "selected Hamiltonian is not Hermitian", {"hermiticity_error": float(herm)}
        )
    return h


def build_sector_hamiltonian(
    sector: SectorSupport, h1e: np.ndarray, eri: np.ndarray, **kwargs
) -> np.ndarray:
    """Build the Hamiltonian for a single ``SectorSupport``."""
    return build_selected_hamiltonian(sector.determinant_keys, h1e, eri, **kwargs)


__all__ = [
    "estimate_dense_bytes",
    "build_selected_hamiltonian",
    "build_sector_hamiltonian",
]
