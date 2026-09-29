"""Unit tests for the deterministic active-space variant generator (T050)."""

from __future__ import annotations

import pytest

from msref.active_ensemble import derive_sector_electron_counts, generate_active_space_variants
from msref.config import MSRefConfig


def _cfg(**over):
    cfg = MSRefConfig()
    cfg.active_space_ensemble.n_active_spaces = over.get("n", 12)
    cfg.active_space_ensemble.max_add_orbitals = over.get("max_add", 2)
    cfg.active_space_ensemble.allow_remove = over.get("allow_remove", False)
    return cfg


@pytest.mark.unit
def test_nominal_variant_present():
    cfg = _cfg(n=4)
    vs = generate_active_space_variants((2, 3), [0, 1], [4, 5, 6], (2, 2), cfg)
    ids = [v.variant_id for v in vs]
    assert any("nominal" in i for i in ids)
    assert vs[0].active_orbitals == (2, 3)


@pytest.mark.unit
def test_deterministic_same_seed():
    cfg1 = _cfg(n=10)
    cfg2 = _cfg(n=10)
    v1 = generate_active_space_variants((0, 1), [], [2, 3, 4, 5, 6, 7], (1, 1), cfg1)
    v2 = generate_active_space_variants((0, 1), [], [2, 3, 4, 5, 6, 7], (1, 1), cfg2)
    assert [(v.variant_id, v.active_orbitals) for v in v1] == \
           [(v.variant_id, v.active_orbitals) for v in v2]


@pytest.mark.unit
def test_add_and_exchange_generated():
    cfg = _cfg(n=20)
    vs = generate_active_space_variants((0, 1), [], [2, 3, 4], (1, 1), cfg)
    rules = {v.generation_rule for v in vs}
    assert "add1" in rules
    assert "exchange" in rules or "add2" in rules


@pytest.mark.unit
def test_all_variants_pass_electron_accounting():
    cfg = _cfg(n=20)
    vs = generate_active_space_variants((2, 3), [0, 1], [4, 5, 6], (3, 3), cfg)
    for v in vs:
        na_inact = len(v.inactive_orbitals)
        # na_act = na_cmp - n_inactive
        for (_a, _b, _w), (na, nb) in v.nelecas_by_sector.items():
            assert na == 3 - na_inact
            assert nb == 3 - na_inact
            assert 0 <= na <= len(v.active_orbitals)
            assert 0 <= nb <= len(v.active_orbitals)


@pytest.mark.unit
def test_capped_at_n_active_spaces():
    cfg = _cfg(n=5)
    vs = generate_active_space_variants((0, 1), [], list(range(2, 20)), (1, 1), cfg)
    assert len(vs) <= 5
