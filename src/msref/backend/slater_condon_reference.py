"""Reference fermionic bit operations and Slater-Condon Hamiltonian elements.

This is the *correctness oracle* for the selected-space Hamiltonian.  It uses
explicit second-quantized operator application on ``(alpha, beta)`` bitstrings
over spatial orbitals, in PySCF's integral convention::

    H = sum_pq h1e[p,q] E_pq + 1/2 sum_pqrs eri[p,q,r,s] e_{pqrs}

with ``E_pq = sum_sigma a_p,sigma^dagger a_q,sigma`` and
``e_{pqrs} = sum_{sigma,tau} a_p,sigma^dagger a_r,tau^dagger a_s,tau a_q,sigma``.
``eri[p,q,r,s] = (pq|rs)`` is PySCF's chemist-notation two-electron integral.
"""

from __future__ import annotations

from typing import Optional

from ..models import DeterminantKey

# ---------------------------------------------------------------------------
# Fermionic bit operations (T024)
# ---------------------------------------------------------------------------

def is_occupied(bits: int, p: int) -> bool:
    return bool((bits >> p) & 1)


def popcount(bits: int) -> int:
    return bits.bit_count()


def count_occupied_between(bits: int, p: int, q: int) -> int:
    """Number of occupied orbitals strictly between ``p`` and ``q``."""
    lo, hi = (p, q) if p < q else (q, p)
    mask = ((1 << hi) - 1) ^ ((1 << (lo + 1)) - 1)
    return popcount(bits & mask)


def annihilate(bits: int, p: int) -> Optional[tuple[int, int]]:
    """Apply ``a_p``; return ``(new_bits, sign)`` or ``None`` if unoccupied."""
    if not is_occupied(bits, p):
        return None
    n_before = popcount(bits & ((1 << p) - 1))
    sign = -1 if (n_before & 1) else 1
    return bits & ~(1 << p), sign


def create(bits: int, p: int) -> Optional[tuple[int, int]]:
    """Apply ``a_p^dagger``; return ``(new_bits, sign)`` or ``None`` if occupied."""
    if is_occupied(bits, p):
        return None
    n_before = popcount(bits & ((1 << p) - 1))
    sign = -1 if (n_before & 1) else 1
    return bits | (1 << p), sign


def excitation_rank(bra: DeterminantKey, ket: DeterminantKey) -> tuple[int, int]:
    """Return ``(alpha_diff_bits, beta_diff_bits)`` (each is ``2 * rank``)."""
    return popcount(bra.alpha ^ ket.alpha), popcount(bra.beta ^ ket.beta)


def _single_bit(x: int) -> int:
    return x.bit_length() - 1


# ---------------------------------------------------------------------------
# Single-spin one-body transitions (internal)
# ---------------------------------------------------------------------------

def _one_body_transition(bits1: int, bits2: int, norb: int) -> list[tuple[int, int, float]]:
    """Return ``[(p, q, value)]`` for ``gamma[p,q] = <bits1| a_p^dagger a_q |bits2>``."""
    d = bits1 ^ bits2
    if d == 0:
        return [(p, p, 1.0) for p in range(norb) if is_occupied(bits1, p)]
    if popcount(d) == 2:
        q = d & bits2  # occupied in ket only -> annihilated
        p = d & bits1  # occupied in bra only -> created
        q_idx = _single_bit(q)
        p_idx = _single_bit(p)
        sign = _excitation_sign(bits2, q_idx, p_idx)
        return [(p_idx, q_idx, float(sign))]
    return []


def _excitation_sign(bits: int, q: int, p: int) -> int:
    tmp, s1 = annihilate(bits, q)
    _, s2 = create(tmp, p)
    return s1 * s2


def _one_body(bits1: int, bits2: int, h1e) -> float:
    norb = h1e.shape[0]
    return float(sum(h1e[p, q] * v for p, q, v in _one_body_transition(bits1, bits2, norb)))


def _two_body_same(bits1: int, bits2: int, eri) -> float:
    """``<bits1| sum_pqrs eri[p,q,r,s] a_p^+ a_r^+ a_s a_q |bits2>`` (same spin)."""
    norb = eri.shape[0]
    total = 0.0
    occ = [p for p in range(norb) if is_occupied(bits2, p)]
    for q in occ:
        t1, s1 = annihilate(bits2, q)
        for s in occ:
            res = annihilate(t1, s)
            if res is None:
                continue
            t2, s2 = res
            for r in range(norb):
                res3 = create(t2, r)
                if res3 is None:
                    continue
                t3, s3 = res3
                for p in range(norb):
                    res4 = create(t3, p)
                    if res4 is None:
                        continue
                    out, s4 = res4
                    if out == bits1:
                        total += eri[p, q, r, s] * (s1 * s2 * s3 * s4)
    return float(total)


def _two_body_opposite(a1: int, a2: int, b1: int, b2: int, eri) -> float:
    """``sum_pqrs eri[p,q,r,s] gamma_alpha[p,q] gamma_beta[r,s]`` (alpha-beta)."""
    norb = eri.shape[0]
    ta = _one_body_transition(a1, a2, norb)
    tb = _one_body_transition(b1, b2, norb)
    total = 0.0
    for p, q, va in ta:
        for r, s, vb in tb:
            total += eri[p, q, r, s] * va * vb
    return float(total)


# ---------------------------------------------------------------------------
# Full spin-free Hamiltonian element (T025)
# ---------------------------------------------------------------------------

def h_element(bra: DeterminantKey, ket: DeterminantKey, h1e, eri) -> complex:
    """Return ``<bra|H|ket>`` for the spin-free Hamiltonian (PySCF convention)."""
    a1, b1 = bra.alpha, bra.beta
    a2, b2 = ket.alpha, ket.beta
    da = popcount(a1 ^ a2)
    db = popcount(b1 ^ b2)
    if da & 1 or db & 1:
        return 0.0j  # incompatible particle number
    ra, rb = da >> 1, db >> 1
    if ra + rb > 2:
        return 0.0j

    total = 0.0

    # one-body: alpha term needs beta identical, beta term needs alpha identical
    if db == 0:
        total += _one_body(a1, a2, h1e)
    if da == 0:
        total += _one_body(b1, b2, h1e)

    # two-body: same-spin (1/2 factor each) + opposite-spin (full, by symmetry)
    if db == 0:
        total += 0.5 * _two_body_same(a1, a2, eri)
    if da == 0:
        total += 0.5 * _two_body_same(b1, b2, eri)
    total += _two_body_opposite(a1, a2, b1, b2, eri)

    return complex(total, 0.0)


def diagonal_element(det: DeterminantKey, h1e, eri) -> float:
    """Return ``<det|H|det>`` (real)."""
    return float(h_element(det, det, h1e, eri).real)


__all__ = [
    "is_occupied",
    "popcount",
    "count_occupied_between",
    "annihilate",
    "create",
    "excitation_rank",
    "h_element",
    "diagonal_element",
]
