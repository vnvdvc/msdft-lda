"""PySCF capability probe and version guard.

All private/undocumented PySCF interaction is isolated here.  The probe verifies
at runtime that the required public interfaces exist with the expected shape so
that upstream API drift fails loudly (``BackendCapabilityError``) instead of
silently producing wrong science.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ..exceptions import BackendCapabilityError


def _pyscf_version() -> Optional[str]:
    try:
        import pyscf

        return getattr(pyscf, "__version__", None)
    except Exception:
        return None


def _has_state_average_mix() -> tuple[bool, str]:
    try:
        from pyscf.mcscf import addons

        ok = all(hasattr(addons, n) for n in ("state_average_mix", "state_average_mix_"))
        return ok, f"pyscf.mcscf.addons.state_average_mix{'/' if ok else ''}_"
    except Exception as exc:  # noqa: BLE001
        return False, f"import failed: {exc}"


def _has_trans_rdm1() -> tuple[bool, str]:
    try:
        from pyscf.fci import direct_spin1

        ok = callable(getattr(direct_spin1, "trans_rdm1", None))
        return ok, "pyscf.fci.direct_spin1.trans_rdm1"
    except Exception as exc:  # noqa: BLE001
        return False, f"import failed: {exc}"


def _has_spin_square() -> tuple[bool, str]:
    try:
        from pyscf import fci

        ok = callable(getattr(fci, "spin_square", None))
        return ok, "pyscf.fci.spin_square"
    except Exception as exc:  # noqa: BLE001
        return False, f"import failed: {exc}"


def _has_make_rdm12() -> tuple[bool, str]:
    try:
        from pyscf.fci import direct_spin1

        ok = callable(getattr(direct_spin1, "make_rdm12", None))
        return ok, "pyscf.fci.direct_spin1.make_rdm12"
    except Exception as exc:  # noqa: BLE001
        return False, f"import failed: {exc}"


def _has_nonorthogonal_fci_overlap() -> tuple[bool, str]:
    try:
        from pyscf.fci import addons

        fn = getattr(addons, "overlap", None)
        if fn is None:
            return False, "pyscf.fci.addons.overlap missing"
        sig = inspect.signature(fn)
        ok = "s" in sig.parameters
        return ok, "pyscf.fci.addons.overlap(bra, ket, norb, nelec, s=...)"
    except Exception as exc:  # noqa: BLE001
        return False, f"import failed: {exc}"


def _has_selected_ci() -> tuple[bool, str]:
    try:
        from pyscf.fci import selected_ci

        cls = getattr(selected_ci, "SelectedCI", None)
        if cls is None:
            return False, "pyscf.fci.selected_ci.SelectedCI missing"
        attrs = ("ci_coeff_cutoff", "select_cutoff", "conv_tol", "start_tol", "tol_decay_rate")
        ok = all(hasattr(cls, a) for a in attrs)
        return ok, "pyscf.fci.selected_ci.SelectedCI (diagnostic only)"
    except Exception as exc:  # noqa: BLE001
        return False, f"import failed: {exc}"


def _has_fix_spin() -> tuple[bool, str]:
    try:
        from pyscf.mcscf import casci

        ok = callable(getattr(casci.CASCI, "fix_spin_", None))
        return ok, "pyscf.mcscf.casci.CASCI.fix_spin_"
    except Exception as exc:  # noqa: BLE001
        return False, f"import failed: {exc}"


#: name -> probe returning (available, detail)
PROBES: dict[str, Callable[[], tuple[bool, str]]] = {
    "state_average_mix": _has_state_average_mix,
    "trans_rdm1": _has_trans_rdm1,
    "spin_square": _has_spin_square,
    "make_rdm12": _has_make_rdm12,
    "nonorthogonal_fci_overlap": _has_nonorthogonal_fci_overlap,
    "selected_ci_diagnostic": _has_selected_ci,
    "fix_spin": _has_fix_spin,
}


@dataclass
class CapabilityReport:
    """Result of the PySCF capability probe."""

    pyscf_version: Optional[str] = None
    available: dict[str, bool] = field(default_factory=dict)
    details: dict[str, str] = field(default_factory=dict)

    def missing_required(self, required: list[str]) -> list[str]:
        return [name for name in required if not self.available.get(name, False)]

    def assert_required(self, required: list[str]) -> None:
        """Raise ``BackendCapabilityError`` if any required capability is absent."""
        missing = self.missing_required(required)
        if missing:
            raise BackendCapabilityError(
                "missing required PySCF capabilities",
                {
                    "pyscf_version": self.pyscf_version,
                    "missing": missing,
                    "details": {m: self.details.get(m) for m in missing},
                },
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "pyscf_version": self.pyscf_version,
            "available": self.available,
            "details": self.details,
        }


def probe_capabilities(required: Optional[list[str]] = None) -> CapabilityReport:
    """Probe PySCF interfaces and return a ``CapabilityReport``.

    If ``required`` is given, any missing required capability raises
    ``BackendCapabilityError``.
    """
    report = CapabilityReport(pyscf_version=_pyscf_version())
    for name, probe in PROBES.items():
        ok, detail = probe()
        report.available[name] = ok
        report.details[name] = detail
    if required:
        report.assert_required(required)
    return report


__all__ = ["CapabilityReport", "probe_capabilities", "PROBES"]
