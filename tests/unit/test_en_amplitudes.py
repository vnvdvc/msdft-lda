"""Unit tests for EN amplitudes and intruder policy (T036)."""

from __future__ import annotations

import numpy as np
import pytest

from msref.config import MSRefConfig
from msref.models import DeterminantKey, SpatialOccupationKey
from msref.residual import score_external_family, signed_floor


def _cfg():
    return MSRefConfig()


@pytest.mark.unit
def test_signed_floor():
    assert signed_floor(0.03, 1e-10) == 0.03
    assert signed_floor(-0.03, 1e-10) == -0.03
    assert signed_floor(0.0, 1e-10) == 1e-10
    assert signed_floor(1e-15, 1e-10) == 1e-10


@pytest.mark.unit
def test_normal_denominator_u020():
    # single selected determinant, one external family with a clean denominator
    h1e = np.zeros((2, 2))
    eri = np.zeros((2, 2, 2, 2))
    sel_keys = (DeterminantKey(0b01, 0b01),)
    sel_coeffs = np.array([[1.0]])
    energies = np.array([-1.0])
    weights = np.array([1.0])
    fam = SpatialOccupationKey((1, 1))
    cfg = _cfg()
    cfg.static_leakage.delta_intruder_eh = 0.01
    sc = score_external_family(fam, 1, 1, sel_keys, sel_coeffs, energies, weights, h1e, eri, cfg)
    # no coupling (zero eri) -> all scores zero, not mandatory
    assert sc.static_score_max == 0.0
    assert not sc.intruder


@pytest.mark.unit
def test_intruder_detection_u021():
    """Small denominator + significant coupling -> intruder flag."""
    h1e = np.zeros((2, 2))
    h1e[0, 1] = h1e[1, 0] = 5.0  # nonzero coupling for the beta single excitation
    eri = np.zeros((2, 2, 2, 2))
    sel_keys = (DeterminantKey(0b01, 0b01),)
    sel_coeffs = np.array([[1.0]])
    energies = np.array([-1.0])
    weights = np.array([1.0])
    fam = SpatialOccupationKey((1, 1))
    cfg = _cfg()
    cfg.static_leakage.delta_intruder_eh = 100.0  # force intruder (everything near-degenerate)
    cfg.static_leakage.tau_coupling_eh = 0.01
    sc = score_external_family(fam, 1, 1, sel_keys, sel_coeffs, energies, weights, h1e, eri, cfg)
    assert sc.intruder
    assert sc.mandatory


@pytest.mark.unit
def test_harmless_small_denominator_u022():
    """Small denominator but negligible coupling -> no intruder."""
    h1e = np.zeros((2, 2))
    eri = np.zeros((2, 2, 2, 2))
    sel_keys = (DeterminantKey(0b01, 0b01),)
    sel_coeffs = np.array([[1.0]])
    energies = np.array([0.0])
    weights = np.array([1.0])
    fam = SpatialOccupationKey((1, 1))
    cfg = _cfg()
    cfg.static_leakage.delta_intruder_eh = 100.0
    cfg.static_leakage.tau_coupling_eh = 0.01
    sc = score_external_family(fam, 1, 1, sel_keys, sel_coeffs, energies, weights, h1e, eri, cfg)
    assert not sc.intruder
