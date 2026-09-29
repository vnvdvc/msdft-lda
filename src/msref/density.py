"""Density transforms and matrix-1RDM metrics.

The matrix 1-RDM is a first-class scientific target: ``gamma[p,q] = <q^dagger p>``
(PySCF convention), stored as ``gamma[A,B,p,q]`` with the Hermiticity relation
``gamma[B,A] = gamma[A,B]^dagger``.

AO comparisons are performed after symmetric orthogonalization so a raw
non-orthogonal AO Frobenius norm is never treated as basis invariant.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .exceptions import InputValidationError, OrbitalAlignmentError


def symmetric_orthogonal_density(
    p_ao: np.ndarray, s_ao: np.ndarray, eps: float = 1e-12
) -> np.ndarray:
    """Return ``S^{1/2} P_AO S^{1/2}`` (orthonormalized AO density)."""
    s_ao = np.asarray(s_ao)
    vals, vecs = np.linalg.eigh(s_ao)
    if float(np.min(vals)) < eps:
        raise OrbitalAlignmentError("AO overlap nearly singular", {"min_eig": float(np.min(vals))})
    shalf = (vecs * np.sqrt(vals)) @ vecs.conj().T
    return shalf @ np.asarray(p_ao) @ shalf


def restore_core_density(p_ao: np.ndarray, c_core: np.ndarray) -> np.ndarray:
    """Add the doubly-occupied core contribution ``2 C_core C_core^dagger``."""
    return np.asarray(p_ao) + 2.0 * np.asarray(c_core) @ np.asarray(c_core).conj().T


def ao_matrix_density(
    c_comparison: np.ndarray,
    gamma_parent: np.ndarray,
    always_core: np.ndarray,
) -> np.ndarray:
    """Build the full AO matrix density ``[A,B,nao,nao]`` from parent 1-RDMs.

    Diagonal blocks include the restored always-core contribution; transition
    blocks (``A != B``) carry only the active/comparison contribution (the core
    term is multiplied by ``<A|B> = 0``).
    """
    c = np.asarray(c_comparison)
    nstate = gamma_parent.shape[0]
    nao = c.shape[0]
    p_ao = np.einsum("pi,ABpq,qj->ABij", c, gamma_parent, c.conj())
    # einsum above: P_AB = C gamma_AB C^dagger
    core = 2.0 * always_core @ always_core.conj().T if always_core.shape[0] else 0.0
    for a in range(nstate):
        p_ao[a, a] = p_ao[a, a] + core
    return p_ao


def matrix_rdm_error(
    g_sel: np.ndarray,
    g_teacher: np.ndarray,
    state_weights: np.ndarray,
    nelec: int,
    allowed: Optional[np.ndarray] = None,
    mode: str = "state_weight_product",
):
    """Compute the aggregate and max-block matrix-1RDM error.

    Returns ``(epsilon_gamma, max_block_error, block_errors)``.
    """
    n = g_sel.shape[0]
    w = np.asarray(state_weights, dtype=float)
    if allowed is None:
        allowed = np.ones((n, n), dtype=bool)
    allowed = np.asarray(allowed, dtype=bool)

    if mode == "state_weight_product":
        nu = np.outer(w, w)
    elif mode == "uniform_matrix_blocks":
        count = int(np.count_nonzero(allowed))
        if count == 0:
            raise InputValidationError("no allowed matrix blocks")
        nu = np.where(allowed, 1.0 / count, 0.0)
    else:
        raise InputValidationError(f"unknown gamma_block_weighting {mode!r}")

    block_errors = {}
    num = 0.0
    for a in range(n):
        for b in range(n):
            if not allowed[a, b]:
                continue
            d = g_sel[a, b] - g_teacher[a, b]
            e = float(np.linalg.norm(d, ord="fro"))
            block_errors[(a, b)] = e
            num += nu[a, b] * e * e

    eps = float(np.sqrt(num)) / nelec
    max_block = max(block_errors.values()) if block_errors else 0.0
    return eps, max_block, block_errors


def trans_rdm1_selected(
    bra_coeff: np.ndarray,
    ket_coeff: np.ndarray,
    determinant_keys,
    norb: int,
) -> np.ndarray:
    """Exact transition 1-RDM ``gamma[p,q] = <bra|q^dagger p|ket>``.

    Computed directly from an arbitrary selected determinant list (no FCI-array
    shape assumption), using fermionic bit-operation signs.  ``bra_coeff`` and
    ``ket_coeff`` are length-``ndet`` complex arrays over ``determinant_keys``.
    """
    from .backend.slater_condon_reference import annihilate, create, is_occupied
    from .models import DeterminantKey

    index = {k: i for i, k in enumerate(determinant_keys)}
    gamma = np.zeros((norb, norb), dtype=complex)

    for j, ket in enumerate(determinant_keys):
        cj = ket_coeff[j]
        if abs(cj) == 0:
            continue
        # alpha contribution
        for p in range(norb):
            if not is_occupied(ket.alpha, p):
                continue
            tmp, s1 = annihilate(ket.alpha, p)
            for q in range(norb):
                res = create(tmp, q)
                if res is None:
                    continue
                out_a, s2 = res
                i = index.get(DeterminantKey(out_a, ket.beta))
                if i is not None:
                    gamma[p, q] += np.conj(bra_coeff[i]) * cj * (s1 * s2)
        # beta contribution
        for p in range(norb):
            if not is_occupied(ket.beta, p):
                continue
            tmp, s1 = annihilate(ket.beta, p)
            for q in range(norb):
                res = create(tmp, q)
                if res is None:
                    continue
                out_b, s2 = res
                i = index.get(DeterminantKey(ket.alpha, out_b))
                if i is not None:
                    gamma[p, q] += np.conj(bra_coeff[i]) * cj * (s1 * s2)
    return gamma


def state_rdm1_selected(coeffs: np.ndarray, determinant_keys, norb: int) -> np.ndarray:
    """Diagonal state 1-RDM from a selected determinant vector."""
    return trans_rdm1_selected(coeffs, coeffs, determinant_keys, norb)


__all__ = [
    "symmetric_orthogonal_density",
    "restore_core_density",
    "ao_matrix_density",
    "matrix_rdm_error",
    "trans_rdm1_selected",
    "state_rdm1_selected",
]
