"""Unit tests for the typed exception hierarchy."""

from __future__ import annotations

import json

import pytest

import msref.exceptions as ex


ALL_EXCEPTIONS = [
    ex.MSRefError,
    ex.InputValidationError,
    ex.BackendCapabilityError,
    ex.TeacherConvergenceError,
    ex.SpinContaminationError,
    ex.SymmetryMismatchError,
    ex.OrbitalAlignmentError,
    ex.StateTrackingError,
    ex.ParentEmbeddingError,
    ex.SelectedDiagonalizationError,
    ex.SelectedRootMissingError,
    ex.IntruderOverflowError,
    ex.DensityCriterionUnreachable,
    ex.ProjectorCriterionUnreachable,
    ex.ReferenceNotCompact,
    ex.SelectionNonConvergence,
    ex.CheckpointCompatibilityError,
    ex.InternalConsistencyError,
]


@pytest.mark.unit
@pytest.mark.parametrize("cls", ALL_EXCEPTIONS)
def test_exception_imports_and_subclasses(cls):
    assert issubclass(cls, ex.MSRefError)
    err = cls("boom", {"k": 1})
    assert err.message == "boom"


@pytest.mark.unit
def test_to_dict_roundtrips_json():
    err = ex.StateTrackingError("ambiguous roots", {"overlap": 0.31, "states": ["a", "b"]})
    d = err.to_dict()
    # Must be JSON-serializable
    json.dumps(d)
    assert d["type"] == "StateTrackingError"
    assert d["context"]["overlap"] == 0.31
    assert d["context"]["states"] == ["a", "b"]


@pytest.mark.unit
def test_default_context_empty():
    err = ex.ReferenceNotCompact("too big")
    assert err.context == {}
