"""Unit tests for the HDF5 schema skeleton and array hashing."""

from __future__ import annotations

import numpy as np
import pytest

from msref.exceptions import CheckpointCompatibilityError
from msref.io import CheckpointReader, CheckpointWriter, array_hash, dumps_json


@pytest.mark.unit
def test_array_hash_deterministic_and_shape_sensitive():
    a = np.arange(12.0).reshape(3, 4)
    b = np.arange(12.0).reshape(4, 3)
    assert array_hash(a) == array_hash(a.copy())
    assert array_hash(a) != array_hash(b)


@pytest.mark.unit
def test_checkpoint_roundtrip(tmp_path):
    path = str(tmp_path / "run.h5")
    meta = {"schema_version": "1.0", "package_version": "0.1.0", "config_hash": "abc"}
    mol = {
        "atom": [["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, 0.74]]],
        "charge": 0,
        "spin": 0,
        "basis_json": {"H": "cc-pvdz"},
        "symmetry": "",
    }
    orb = {
        "ao_overlap": np.eye(3),
        "parent_mo_coeff": np.eye(3, dtype=complex),
        "orbital_ids": ["o0", "o1", "o2"],
        "orbsym": [None, None, None],
    }
    CheckpointWriter(path).write({"meta": meta, "molecule": mol, "orbitals": orb})

    out = CheckpointReader(path).read()
    assert out["meta"]["config_hash"] == "abc"
    assert out["molecule"]["charge"] == 0
    assert out["molecule"]["atom"][0][0] == "H"
    assert np.allclose(out["orbitals"]["ao_overlap"], np.eye(3))
    assert out["orbitals"]["orbital_ids"] == ["o0", "o1", "o2"]


@pytest.mark.unit
def test_newer_major_schema_fails(tmp_path):
    path = str(tmp_path / "run.h5")
    import h5py

    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = "2.0"
    with pytest.raises(CheckpointCompatibilityError):
        CheckpointReader(path, schema_version="1.0").read()


@pytest.mark.unit
def test_dumps_json_numpy_compatible():
    s = dumps_json({"arr": np.array([1, 2, 3]), "val": np.float64(2.5)})
    assert "[1, 2, 3]" in s
    assert "2.5" in s
