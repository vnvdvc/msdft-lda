"""HDF5 checkpoint I/O, array hashing, and JSON report serialization.

The checkpoint layout follows ``DATA_CONTRACTS.yaml``.  This module provides the
schema skeleton (``/meta``, ``/molecule``, ``/orbitals``) plus helpers used by
later milestones for teachers/selection/final groups.  Writes are atomic via a
temp-file + rename so an interrupted run never corrupts a previous checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from typing import Any, Optional

import numpy as np

from .exceptions import CheckpointCompatibilityError, InputValidationError

HDF5_SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Array hashing
# ---------------------------------------------------------------------------

def array_hash(arr: np.ndarray) -> str:
    """SHA-256 of the canonical byte representation of an array."""
    a = np.ascontiguousarray(arr)
    h = hashlib.sha256()
    h.update(a.dtype.str.encode("ascii"))
    h.update(a.shape.__repr__().encode("ascii"))
    h.update(a.tobytes())
    return h.hexdigest()


def _major_version(schema_version: str) -> str:
    return schema_version.split(".")[0]


# ---------------------------------------------------------------------------
# Atomic writing helper
# ---------------------------------------------------------------------------

def atomic_write(target: str, writer, **kwargs: Any) -> str:
    """Call ``writer(handle, **kwargs)`` writing atomically to ``target``.

    ``writer`` must accept an open file-like/h5 handle as its first argument.
    """
    target = os.path.abspath(target)
    directory = os.path.dirname(target) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp_msref_", suffix=".h5")
    os.close(fd)
    try:
        writer(tmp, **kwargs)
        os.replace(tmp, target)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return target


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------

def dumps_json(obj: Any, **kwargs: Any) -> str:
    """Deterministic JSON serialization with a stable default for numpy."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, default=_json_default, **kwargs)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (set, frozenset)):
        return sorted(obj, key=str)
    if isinstance(obj, complex):
        return {"real": obj.real, "imag": obj.imag}
    raise TypeError(f"not JSON serializable: {type(obj)!r}")


# ---------------------------------------------------------------------------
# HDF5 schema skeleton
# ---------------------------------------------------------------------------

class CheckpointWriter:
    """Versioned HDF5 writer."""

    def __init__(self, path: str, schema_version: str = HDF5_SCHEMA_VERSION) -> None:
        self.path = path
        self.schema_version = schema_version

    def _write(self, tmp_path: str, data: dict[str, Any]) -> None:
        import h5py  # lazy import

        with h5py.File(tmp_path, "w") as f:
            f.attrs["schema_version"] = self.schema_version
            _write_meta(f, data["meta"])
            if "molecule" in data:
                _write_molecule(f, data["molecule"])
            if "orbitals" in data:
                _write_orbitals(f, data["orbitals"])

    def write(self, data: dict[str, Any]) -> str:
        """Write ``data`` (with keys meta/molecule/orbitals) atomically."""
        return atomic_write(self.path, self._write, data=data)


class CheckpointReader:
    """Versioned HDF5 reader with schema compatibility enforcement."""

    def __init__(self, path: str, schema_version: str = HDF5_SCHEMA_VERSION) -> None:
        self.path = path
        self.schema_version = schema_version

    def read(self) -> dict[str, Any]:
        import h5py  # lazy import

        if not os.path.exists(self.path):
            raise InputValidationError("checkpoint not found", {"path": self.path})
        with h5py.File(self.path, "r") as f:
            stored = f.attrs.get("schema_version", "0.0")
            if _major_version(str(stored)) > _major_version(self.schema_version):
                raise CheckpointCompatibilityError(
                    f"checkpoint schema {stored!r} is newer than supported "
                    f"{self.schema_version!r}",
                    {"stored": str(stored), "supported": self.schema_version},
                )
            meta = _read_meta(f)
            out: dict[str, Any] = {"meta": meta}
            if "molecule" in f:
                out["molecule"] = _read_molecule(f)
            if "orbitals" in f:
                out["orbitals"] = _read_orbitals(f)
            return out


def _write_meta(f, meta: dict[str, Any]) -> None:
    g = f.create_group("meta")
    for key, value in meta.items():
        if isinstance(value, str):
            g.attrs[key] = value
        elif isinstance(value, (int, float, bool)):
            g.attrs[key] = value
        else:
            g.attrs[key] = json.dumps(value, sort_keys=True, default=_json_default)


def _read_meta(f) -> dict[str, Any]:
    g = f["meta"]
    out: dict[str, Any] = {}
    for key, value in g.attrs.items():
        out[key] = _try_json(value)
    return out


def _try_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    return value


def _write_molecule(f, mol: dict[str, Any]) -> None:
    g = f.create_group("molecule")
    g.attrs["charge"] = int(mol["charge"])
    g.attrs["spin"] = int(mol["spin"])
    g.attrs["symmetry"] = str(mol.get("symmetry", ""))
    g.attrs["atom"] = json.dumps(mol["atom"])
    g.attrs["basis_json"] = json.dumps(mol.get("basis_json", {}))


def _read_molecule(f) -> dict[str, Any]:
    g = f["molecule"]
    return {
        "atom": json.loads(g.attrs["atom"]),
        "charge": int(g.attrs["charge"]),
        "spin": int(g.attrs["spin"]),
        "basis_json": json.loads(g.attrs["basis_json"]),
        "symmetry": str(g.attrs.get("symmetry", "")),
    }


def _write_orbitals(f, orb: dict[str, Any]) -> None:
    g = f.create_group("orbitals")
    g.create_dataset("ao_overlap", data=np.asarray(orb["ao_overlap"]))
    g.create_dataset("parent_mo_coeff", data=np.asarray(orb["parent_mo_coeff"]))
    g.attrs["orbital_ids"] = json.dumps(orb.get("orbital_ids", []))
    g.attrs["orbsym"] = json.dumps(orb.get("orbsym", []))


def _read_orbitals(f) -> dict[str, Any]:
    g = f["orbitals"]
    return {
        "ao_overlap": np.asarray(g["ao_overlap"]),
        "parent_mo_coeff": np.asarray(g["parent_mo_coeff"]),
        "orbital_ids": json.loads(g.attrs["orbital_ids"]),
        "orbsym": json.loads(g.attrs["orbsym"]),
    }


__all__ = [
    "HDF5_SCHEMA_VERSION",
    "array_hash",
    "atomic_write",
    "dumps_json",
    "CheckpointWriter",
    "CheckpointReader",
]
