"""Unit tests for projector, energy, density, and family metrics (T032/T033)."""

from __future__ import annotations

import numpy as np
import pytest

from msref.metrics import (
    hellinger_distance,
    projector_distance,
    spin_square_error,
    weighted_jaccard,
)


@pytest.mark.unit
def test_projector_identical_u024():
    y = np.array([[1, 0], [0, 1], [0, 0]], dtype=float)
    d_op, d_fro, _ = projector_distance(y, y)
    assert d_op == pytest.approx(0.0, abs=1e-12)
    assert d_fro == pytest.approx(0.0, abs=1e-12)


@pytest.mark.unit
def test_projector_orthogonal_u024():
    y1 = np.array([[1.0], [0.0]])
    y2 = np.array([[0.0], [1.0]])
    d_op, d_fro, _ = projector_distance(y1, y2)
    assert d_op == pytest.approx(1.0, abs=1e-12)
    assert d_fro == pytest.approx(1.0, abs=1e-12)


@pytest.mark.unit
def test_projector_cluster_rotation_invariant_u015():
    ref = np.array([[1.0, 0], [0, 1.0]])
    u = np.array([[0.7, -0.7], [0.7, 0.7]])  # ~orthogonal-ish, will be QR'd
    test = ref @ u
    d_op, d_fro, _ = projector_distance(ref, test)
    assert d_op == pytest.approx(0.0, abs=1e-10)


@pytest.mark.unit
def test_hellinger_u017():
    assert hellinger_distance({"a": 1.0}, {"a": 1.0}) == pytest.approx(0.0, abs=1e-12)
    assert hellinger_distance({"a": 1.0}, {"b": 1.0}) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.unit
def test_weighted_jaccard_u018():
    assert weighted_jaccard({"a": 1.0}, {"a": 1.0}) == pytest.approx(1.0)
    assert weighted_jaccard({"a": 1.0}, {"b": 1.0}) == pytest.approx(0.0)


@pytest.mark.unit
def test_spin_square_error():
    assert spin_square_error(2.0, 1.0) == pytest.approx(0.0)
    assert spin_square_error(0.75, 0.5) == pytest.approx(0.0)
    assert spin_square_error(0.0, 1.0) == pytest.approx(2.0)
