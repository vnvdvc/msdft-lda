"""External connected-family generation, couplings, and EN amplitudes (T034-T036)."""

from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

import numpy as np

from .backend.slater_condon_reference import (
    annihilate,
    create,
    diagonal_element,
    h_element,
    is_occupied,
)
from .models import DeterminantKey, ExternalFamilyScore, SpatialOccupationKey
from .spin_completion import enumerate_family, family_of


# ---------------------------------------------------------------------------
# T034 -- external connected determinant generation
# ---------------------------------------------------------------------------

def generate_singles(det: DeterminantKey, norb: int) -> set[DeterminantKey]:
    """All spin-conserving single excitations of ``det``."""
    out = set()
    for p in range(norb):
        if not is_occupied(det.alpha, p):
            continue
        tmp, _ = annihilate(det.alpha, p)
        for q in range(norb):
            res = create(tmp, q)
            if res is not None:
                out.add(DeterminantKey(res[0], det.beta))
    for p in range(norb):
        if not is_occupied(det.beta, p):
            continue
        tmp, _ = annihilate(det.beta, p)
        for q in range(norb):
            res = create(tmp, q)
            if res is not None:
                out.add(DeterminantKey(det.alpha, res[0]))
    return out


def generate_doubles(det: DeterminantKey, norb: int) -> set[DeterminantKey]:
    """All spin-conserving double excitations of ``det``."""
    out = set()
    occ_a = [p for p in range(norb) if is_occupied(det.alpha, p)]
    occ_b = [p for p in range(norb) if is_occupied(det.beta, p)]

    # alpha-alpha
    for i, p in enumerate(occ_a):
        t1 = annihilate(det.alpha, p)[0]
        for r in occ_a[i + 1:]:
            t2 = annihilate(t1, r)[0]
            for q in range(norb):
                res = create(t2, q)
                if res is None:
                    continue
                for s in range(norb):
                    res2 = create(res[0], s)
                    if res2 is not None:
                        out.add(DeterminantKey(res2[0], det.beta))
    # beta-beta
    for i, p in enumerate(occ_b):
        t1 = annihilate(det.beta, p)[0]
        for r in occ_b[i + 1:]:
            t2 = annihilate(t1, r)[0]
            for q in range(norb):
                res = create(t2, q)
                if res is None:
                    continue
                for s in range(norb):
                    res2 = create(res[0], s)
                    if res2 is not None:
                        out.add(DeterminantKey(det.alpha, res2[0]))
    # alpha-beta
    for p in occ_a:
        ta = annihilate(det.alpha, p)[0]
        for q in range(norb):
            ra = create(ta, q)
            if ra is None:
                continue
            for r in occ_b:
                tb = annihilate(det.beta, r)[0]
                for s in range(norb):
                    rb = create(tb, s)
                    if rb is not None:
                        out.add(DeterminantKey(ra[0], rb[0]))
    return out


def generate_connected_external(
    selected: Iterable[DeterminantKey],
    norb: int,
    symmetry_filter=None,
) -> set[DeterminantKey]:
    """All external determinants connected to ``selected`` by singles/doubles."""
    sel = set(selected)
    external = set()
    for det in sel:
        for ext in generate_singles(det, norb):
            if ext not in sel and (symmetry_filter is None or symmetry_filter(ext)):
                external.add(ext)
        for ext in generate_doubles(det, norb):
            if ext not in sel and (symmetry_filter is None or symmetry_filter(ext)):
                external.add(ext)
    return external


def connected_external_families(
    selected: Iterable[DeterminantKey],
    norb: int,
    symmetry_filter=None,
) -> tuple[SpatialOccupationKey, ...]:
    """Map connected external determinants to their spatial-occupation families."""
    external = generate_connected_external(selected, norb, symmetry_filter)
    families = {family_of(det, norb) for det in external}
    return tuple(sorted(families))


# ---------------------------------------------------------------------------
# T035 -- external coupling and diagonal
# ---------------------------------------------------------------------------

def external_coupling(
    external_det: DeterminantKey,
    selected_keys: Sequence[DeterminantKey],
    selected_coeffs: np.ndarray,
    h1e: np.ndarray,
    eri: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return ``(v_by_state, H_mumu)`` where ``v_A = <mu|H|Psi_A>``."""
    nstate = selected_coeffs.shape[1]
    v = np.zeros(nstate, dtype=complex)
    for j, detj in enumerate(selected_keys):
        hij = h_element(external_det, detj, h1e, eri)
        if hij != 0:
            v += hij * selected_coeffs[j, :]
    hmm = diagonal_element(external_det, h1e, eri)
    return v, hmm


# ---------------------------------------------------------------------------
# T036 -- EN amplitudes and intruder policy
# ---------------------------------------------------------------------------

def signed_floor(delta: float, delta_numeric: float) -> float:
    """Numerically floor a denominator without changing its sign."""
    return math.copysign(max(abs(delta), delta_numeric), delta)


def score_external_family(
    family: SpatialOccupationKey,
    na: int,
    nb: int,
    selected_keys: Sequence[DeterminantKey],
    selected_coeffs: np.ndarray,
    energies: np.ndarray,
    weights: np.ndarray,
    h1e: np.ndarray,
    eri: np.ndarray,
    cfg,
) -> ExternalFamilyScore:
    """Compute static-leakage score and intruder flag for one external family."""
    ext_dets = enumerate_family(family, na, nb)
    nstate = len(energies)
    t2sum = np.zeros(nstate)
    maxabs = 0.0
    pt2 = 0.0
    intruder = False
    v_by_state = np.zeros(nstate, dtype=complex)
    denom_by_state = np.zeros(nstate)
    t_by_state = np.zeros(nstate, dtype=complex)

    for mu in ext_dets:
        hmm = diagonal_element(mu, h1e, eri)
        v, _ = external_coupling(mu, selected_keys, selected_coeffs, h1e, eri)
        for a in range(nstate):
            delta = energies[a] - hmm
            v_by_state[a] += v[a]
            denom_by_state[a] += delta
            if abs(delta) < cfg.static_leakage.delta_intruder_eh and abs(v[a]) > cfg.static_leakage.tau_coupling_eh:
                intruder = True
            den = signed_floor(delta, cfg.static_leakage.delta_numeric_eh)
            t = v[a] / den
            t_by_state[a] += t
            t2sum[a] += abs(t) ** 2
            maxabs = max(maxabs, abs(t))
            pt2 += weights[a] * (abs(v[a]) ** 2 / den)

    rms = float(np.sqrt(float(np.dot(weights, t2sum))))
    mandatory = intruder or maxabs > cfg.static_leakage.tau_static_max or rms > cfg.static_leakage.tau_static_rms
    return ExternalFamilyScore(
        family_key=family,
        determinant_keys=ext_dets,
        v_by_state=v_by_state,
        denominator_by_state=denom_by_state,
        t_by_state=t_by_state,
        static_score_max=maxabs,
        static_score_rms=rms,
        intruder=bool(intruder),
        pt2_eh=float(pt2),
        mandatory=bool(mandatory),
    )


__all__ = [
    "generate_singles",
    "generate_doubles",
    "generate_connected_external",
    "connected_external_families",
    "external_coupling",
    "signed_floor",
    "score_external_family",
]
