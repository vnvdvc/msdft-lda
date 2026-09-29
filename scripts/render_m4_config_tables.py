"""Per-state configuration-weight tables (N2/O2/C2).

Rows = distinct configurations, identified *across active spaces* via a
hole/particle label relative to the ground-state reference configuration.
Columns = active-space variants. Cells = the configuration's weight (sum |c|^2
over the spatial-occupation family) in that variant's frozen-core CASCI.

A configuration from a larger active space is the *same* physical configuration
as one from a smaller active space when, in the full parent-orbital basis, they
have identical occupancies (extra orbitals empty). We therefore reconstruct every
family in the full MO basis, then label it by holes/particles relative to the
reference.
"""

from __future__ import annotations

import os

# Deterministic BLAS: the RHF SCF is non-reproducible under multi-threaded
# OpenBLAS, and the near-degenerate configuration weights amplify that noise.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from collections import defaultdict

import numpy as np
from pyscf import gto, scf

from msref import models
from msref.config import MSRefConfig
from msref.config_pool import family_weights_for_state
from msref.benchmarks.c2 import build_c2_ensemble
from msref.benchmarks.common import build_ensemble_problem
from msref.benchmarks.n2 import build_n2_ensemble
from msref.benchmarks.o2 import build_o2_ensemble

TOP_N = 10
OUT = "/Users/vnvdvc/Documents/Agents/MSDFT/MS-SCF/msref_m4_config_tables.md"


def _build_o2_rich(cfg):
    """O2 with an expanded virtual buffer (same as the stability report)."""
    cfg.active_space_ensemble.allow_exchange = False
    mol = gto.M(atom="O 0 0 0; O 0 0 1.21", basis="cc-pvdz", unit="angstrom",
                spin=2, verbose=0)
    mf = scf.ROHF(mol).run()
    ncore = 7
    nmo = mf.mo_coeff.shape[1]
    buffer = list(range(ncore + 2, min(nmo, ncore + 6)))
    block_t = models.StateBlockSpec(block_id="triplet", spin_S=1.0, ms2=0,
                                    nroots=1, weights=[1 / 3])
    block_s = models.StateBlockSpec(block_id="singlet", spin_S=0.0, ms2=0,
                                    nroots=2, weights=[1 / 3, 1 / 3])
    prob = build_ensemble_problem(mf, ncas=2, nelecas=2,
                                  state_blocks=[block_t, block_s],
                                  weights=[1 / 3, 1 / 3, 1 / 3], cfg=cfg,
                                  buffer=buffer, nelec_comparison=(8, 8))
    return mf, prob


# molecule -> (name, builder, [(state label, state index), ...], name_mode)
MOL = {
    "n2": ("N₂", lambda c: build_n2_ensemble(r=1.10, cfg=c),
           [("S0", 0), ("S1", 1)], "homo_lumo"),
    "o2": ("O₂", _build_o2_rich,
           [("T1 (X³Σg⁻)", 0), ("S0 (a¹Δg)", 1), ("S1 (b¹Σg⁺)", 2)], "index"),
    "c2": ("C₂", lambda c: build_c2_ensemble(r=1.24, cfg=c),
           [("S0", 0), ("S1", 1), ("S2", 2)], "index"),
}


def _orb_names(name_mode, ref_occ, nmo):
    names = {}
    if name_mode == "homo_lumo":
        occ = [i for i in range(nmo) if ref_occ[i] > 0]
        virt = [i for i in range(nmo) if ref_occ[i] == 0]
        for k, i in enumerate(reversed(occ)):
            names[i] = "ho" if k == 0 else f"ho-{k}"
        for k, i in enumerate(virt):
            names[i] = "lo" if k == 0 else f"lo+{k}"
    else:
        for i in range(nmo):
            names[i] = f"mo{i}"
    return names


def _label(occ, ref_occ, names):
    h = defaultdict(int)
    p = defaultdict(int)
    for i in range(len(occ)):
        d = occ[i] - ref_occ[i]
        if d < 0:
            h[i] = -d
        elif d > 0:
            p[i] = d
    if not h and not p:
        return "ref"

    def fmt(d):
        parts = []
        for i in sorted(d):
            nm = names[i]
            parts.append(f"{nm}×{d[i]}" if d[i] > 1 else nm)
        return ",".join(parts)

    return f"h[{fmt(h)}] p[{fmt(p)}]"


def _variant_header(variant, nominal_active):
    if variant.generation_rule == "nominal":
        return "nominal"
    act = set(variant.active_orbitals)
    nom = set(nominal_active)
    added = sorted(act - nom)
    removed = sorted(nom - act)
    if removed:
        return "swap " + ",".join(map(str, removed)) + "→" + ",".join(map(str, added))
    return "+" + ",".join(map(str, added))


def analyze(name, name_mode, build_fn, state_labels, cfg):
    mf, prob = build_fn(cfg)
    nmo = prob.nparent
    nominal = prob.teachers[prob.nominal_variant.variant_id]

    # reference = dominant family of the ground state (state index 0) in nominal
    ref_fw = family_weights_for_state(nominal.ci_vectors[0], nmo)
    ref_occ = max(ref_fw.items(), key=lambda kv: kv[1])[0].occ
    names = _orb_names(name_mode, ref_occ, nmo)

    # collect full-occupancy family weights per (variant, state)
    data = {}  # (variant_id, state_idx) -> {full_occ(tuple): weight}
    for variant in prob.variants:
        t = prob.teachers[variant.variant_id]
        for s in range(len(t.state_ids)):
            fw = family_weights_for_state(t.ci_vectors[s], nmo)
            top = dict(sorted(((k.occ, w) for k, w in fw.items()),
                              key=lambda kv: -kv[1])[:TOP_N])
            data[(variant.variant_id, s)] = top

    variant_ids = [v.variant_id for v in prob.variants]
    headers = {v.variant_id: _variant_header(v, prob.nominal_variant.active_orbitals)
               for v in prob.variants}

    tables = []
    for (slabel, sidx) in state_labels:
        # label -> {variant_id: weight}
        by_label = defaultdict(dict)
        for variant in prob.variants:
            vid = variant.variant_id
            for occ, w in data[(vid, sidx)].items():
                by_label[_label(occ, ref_occ, names)][vid] = w
        # sort rows by peak weight (desc), then by label
        rows = sorted(by_label.items(),
                      key=lambda kv: (-max(kv[1].values()), kv[0]))
        rows = rows[:TOP_N]
        tables.append((slabel, rows, headers, variant_ids))
    return {
        "name": name,
        "ref_occ": ref_occ,
        "names": names,
        "nmo": nmo,
        "nominal_active": prob.nominal_variant.active_orbitals,
        "tables": tables,
        "state_labels": state_labels,
    }


def _ref_desc(name_mode, ref_occ, names, nominal_active):
    occ_str = [(names.get(i, str(i)), ref_occ[i]) for i in range(len(ref_occ))
               if ref_occ[i] > 0 and i >= nominal_active[0]]
    return ", ".join(f"{n}={v}" for n, v in occ_str)


def build():
    L = []
    a = L.append

    a("# MSREF — Configuration-weight tables per state (N₂ / O₂ / C₂)\n")
    a("Each table lists the **dominant configurations** of one state, identified *across* "
      "active-space computations by a **hole/particle label** relative to the ground-state "
      "reference configuration. A configuration in a larger active space (e.g. `[2,2,0,0,0,0]` "
      "on 6 orbitals) is the same configuration as `[2,2,0,0]` on 4 orbitals when their full "
      "MO occupancies coincide (the extra orbitals are empty). Weights are Σ|c|² within the "
      "spatial-occupation family, per frozen-core CASCI.\n")
    a("**Label convention.** `ref` = the ground-state reference configuration; otherwise "
      "`h[...] p[...]` lists the orbitals that lost electrons (holes) and gained electrons "
      "(particles) relative to the reference, with `×n` for multiplicity. For N₂ the labels use "
      "HOMO/LUMO-relative names (`ho` = HOMO, `lo` = LUMO; e.g. `h[ho-1×2] p[lo+1×2]` = double "
      "excitation HOMO-1 → LUMO+1); for O₂ and C₂ (open-shell references) the labels use absolute "
      "orbital indices `mo{i}`. `–` = the configuration is absent from that variant's dominant set.\n")
    a("**Method & caveats.** Weights are from a *deterministic* run (single-threaded BLAS); the "
      "RHF SCF is otherwise non-reproducible under multi-threading and the near-degenerate "
      "configurations amplify that noise. Two further caveats: (i) energies are fixed-orbital "
      "(RHF/ROHF MOs frozen across variants); (ii) **for (near-)degenerate states the per-state "
      "config assignment is basis-dependent** — this affects O₂'s two singlet roots (a¹Δg / "
      "b¹Σg⁺, degenerate in the minimal CAS) and C₂'s low singlet manifold, whose roots mix "
      "arbitrarily. For those, the *state-averaged* weight over the manifold is the robust "
      "quantity, and the per-state rows should be read with that in mind.\n")

    MOL_NOTES = {
        "o2": "\n> **Degenerate singlet manifold.** a¹Δg and b¹Σg⁺ are degenerate in the minimal "
              "CAS(2,2), so their individual config rows are basis-dependent; the open-shell ⟨1,1⟩ "
              "and the ionic ⟨2,0⟩/⟨0,2⟩ character redistributes freely between them as the active "
              "space changes.\n",
        "c2": "\n> **Near-degenerate singlet manifold.** S0/S1/S2 mix strongly (⟨2,1,1,0⟩ vs "
              "⟨1,2,1,0⟩), so individual-state config weights are sensitive to the (fixed) RHF "
              "orbital gauge; read the state-averaged picture rather than any single root.\n",
    }

    for key in ["n2", "o2", "c2"]:
        molname, build_fn, state_labels, name_mode = MOL[key]
        cfg = MSRefConfig()
        if key == "o2":
            cfg.active_space_ensemble.allow_exchange = False
        r = analyze(molname, name_mode, build_fn, state_labels, cfg)
        a(f"## {molname}\n")
        if key in MOL_NOTES:
            a(MOL_NOTES[key])
        a(f"- **Reference configuration** (ground state; nominal active space "
          f"{list(r['nominal_active'])}, occupancy): "
          f"{_ref_desc(name_mode, r['ref_occ'], r['names'], r['nominal_active'])}\n")
        for (slabel, rows, headers, variant_ids) in r["tables"]:
            a(f"### {molname} — {slabel}\n")
            a("| configuration | " + " | ".join(headers[v] for v in variant_ids) + " |")
            a("|---|" + "---|" * len(variant_ids))
            for label, by_var in rows:
                cells = []
                for v in variant_ids:
                    w = by_var.get(v)
                    cells.append(f"{w:.4f}" if w is not None else "–")
                a(f"| {label} | " + " | ".join(cells) + " |")
            a("")

    with open(OUT, "w") as f:
        f.write("\n".join(L))
    print(f"wrote {OUT} ({len(L)} lines)")


if __name__ == "__main__":
    build()
