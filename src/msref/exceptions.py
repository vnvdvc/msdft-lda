"""Typed exception hierarchy for msref.

Every exception carries structured, JSON-compatible context so that failures can
be serialized into deterministic reports without parsing human-readable logs.
Low-level algorithms must not catch ``MSRefError`` unless they are adding
context and re-raising a more specific type.
"""

from __future__ import annotations

from typing import Any, Optional


class MSRefError(Exception):
    """Base class for all msref failures.

    Parameters
    ----------
    message:
        Human-readable description.
    context:
        Optional structured, JSON-compatible payload (dict/list/str/number).
    """

    def __init__(self, message: str = "", context: Optional[dict] = None) -> None:
        super().__init__(message)
        self.message = message
        self.context = context if context is not None else {}

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of this failure."""
        return {
            "type": type(self).__name__,
            "message": self.message,
            "context": self.context,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}({self.message!r}, context={self.context!r})"


class InputValidationError(MSRefError):
    """Invalid user input, configuration, or data model."""


class BackendCapabilityError(MSRefError):
    """A required upstream (PySCF) capability is missing or has drifted."""


class TeacherConvergenceError(MSRefError):
    """The teacher SA-CASSCF calculation failed to converge."""


class SpinContaminationError(MSRefError):
    """A computed state has <S^2> outside the configured tolerance."""


class SymmetryMismatchError(MSRefError):
    """A computed state carries a symmetry label inconsistent with its target."""


class OrbitalAlignmentError(MSRefError):
    """Orbital gauge alignment is ambiguous or below the matching threshold."""


class StateTrackingError(MSRefError):
    """Root/cluster tracking between calculations could not be resolved."""


class ParentEmbeddingError(MSRefError):
    """A variant CAS determinant could not be embedded in the parent gauge."""


class SelectedDiagonalizationError(MSRefError):
    """The selected-space eigensolver failed (convergence or internal state)."""


class SelectedRootMissingError(MSRefError):
    """A target root was not found within the selected root buffer."""


class IntruderOverflowError(MSRefError):
    """Too many intruder configurations to promote under the support guard."""


class DensityCriterionUnreachable(MSRefError):
    """Matrix-1RDM criterion cannot be met by available trial additions."""


class ProjectorCriterionUnreachable(MSRefError):
    """State-projector criterion cannot be met by available trial additions."""


class ReferenceNotCompact(MSRefError):
    """Selected support exceeded a configured size guardrail."""


class SelectionNonConvergence(MSRefError):
    """The selection driver exhausted its iteration budget."""


class CheckpointCompatibilityError(MSRefError):
    """A checkpoint schema/version is incompatible with this package."""


class InternalConsistencyError(MSRefError):
    """An invariant that must hold by construction was violated."""


__all__ = [
    "MSRefError",
    "InputValidationError",
    "BackendCapabilityError",
    "TeacherConvergenceError",
    "SpinContaminationError",
    "SymmetryMismatchError",
    "OrbitalAlignmentError",
    "StateTrackingError",
    "ParentEmbeddingError",
    "SelectedDiagonalizationError",
    "SelectedRootMissingError",
    "IntruderOverflowError",
    "DensityCriterionUnreachable",
    "ProjectorCriterionUnreachable",
    "ReferenceNotCompact",
    "SelectionNonConvergence",
    "CheckpointCompatibilityError",
    "InternalConsistencyError",
]
