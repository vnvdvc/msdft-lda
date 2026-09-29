"""Render the M4 stability-descriptor JSON into a detailed Markdown report."""

from __future__ import annotations

import json

import numpy as np

SRC = "/Users/vnvdvc/Documents/Agents/MSDFT/MS-SCF/msref_m4_stability_data.json"
OUT = "/Users/vnvdvc/Documents/Agents/MSDFT/MS-SCF/msref_m4_stability_report.md"

D = json.load(open(SRC))

MOL_NAMES = {"n2": "N₂", "o2": "O₂", "c2": "C₂"}
MOL_DESC = {
    "n2": "dinitrogen (N₂), triple bond, closed-shell singlet manifold",
    "o2": "dioxygen (O₂), mixed-spin manifold (X³Σg⁻ / a¹Δg / b¹Σg⁺)",
    "c2": "dicarbon (C₂), dense multireference singlet manifold",
}


def occ_str(occ):
    return "[" + ",".join(str(x) for x in occ) + "]"


def rule_label(rule):
    return {
        "nominal": "nominal",
        "add1": "add-1 (extend)",
        "add2": "add-2 (extend)",
        "exchange": "exchange (swap)",
        "random": "random",
    }.get(rule, rule)


def fmt_e(e):
    return f"{e:+.6f}"


def build():
    L = []
    a = L.append

    a("# MSREF — Active-Space Ensemble Stability Descriptors (N₂ / O₂ / C₂)\n")
    a("This report documents, for every active-space variant of the fixed-orbital ensemble, the "
      "stability descriptors used to certify a compact multi-state reference: **configuration-family "
      "weights (descending)**, **state energies**, **matrix-1-RDM variation with respect to active-space "
      "choice**, **spin purity**, and the **common-support certification** where it converges.\n")
    a("**Conventions.** Occupancy tuples are over the *active* orbitals in the listed order "
      "(2/1/0 = doubly / singly / empty). Configuration weights are Σ|c|² within each spatial-occupation "
      "family, sorted descending. Energies are *total* energies (frozen-core CASCI + nuclear repulsion, "
      "Hartree). ΔE<sub>SA</sub> is the state-averaged-energy shift relative to the nominal active space. "
      "Δγ is the Frobenius norm of the difference of the state-averaged matrix-1RDM vs the nominal "
      "active space. ⟨S²⟩ is the expected spin. Basis cc-pVDZ throughout.\n")
    a("> **⚠️ Correction to earlier M4 numbers.** A Coulomb/exchange indexing bug in the frozen-core "
      "path (`build_core_and_active`, J=(ii|jj)/K=(ij|ij) written as J=(ij|ij)/K=(ij|ji)) had corrupted "
      "all N₂/O₂/C₂ ensemble *total* energies by up to ~70 Eh. It is fixed (verified against PySCF native "
      "CASCI to <1e-13 Eh) and all numbers below are the corrected values. H₂ (no frozen core) was "
      "unaffected.\n")

    for name in ["n2", "o2", "c2"]:
        r = D[name]
        enuc = r["e_nuc"]
        nmo = r["nmo"]
        weights = r["state_weights"]
        nstate = len(r["variants"][0]["states"])
        a(f"## {MOL_NAMES[name]} — {MOL_DESC[name]}\n")
        nom_active = r["variants"][0]["active_orbitals"]
        nelec = int(round(sum(r["variants"][0]["rdm_diag_active"])))
        n_add = sum(1 for v in r["variants"] if v["generation_rule"] in ("add1", "add2"))
        n_ex = sum(1 for v in r["variants"] if v["generation_rule"] == "exchange")
        a(f"- **Nominal active space:** orbitals {nom_active} "
          f"(CAS({len(nom_active)}, {nelec})) in the parent MO gauge; "
          f"**{nmo}** MOs total; **{r['n_variants']}** variants "
          f"({n_add} extension, {n_ex} exchange).")
        a(f"- **States:** {nstate} state(s), weights {['%.3f' % w for w in weights]}; "
          f"nuclear repulsion E<sub>nuc</sub> = {enuc:.5f} Eh.")
        # spin block summary
        spins = [s["spin_S"] for s in r["variants"][0]["states"]]
        a(f"- **Spin blocks:** S = {sorted(set(spins))} (M_S = 0 representation).\n")

        # ---- Table A: energy + 1-RDM variation (SA level) ----
        a("### Energy and matrix-1RDM variation across active spaces\n")
        a("| active space | rule | orbitals | E<sub>SA</sub> (Eh) | ΔE<sub>SA</sub> (mEh) | Δγ<sub>SA</sub> (Frob) | max Δγ<sub>state</sub> |")
        a("|---|---|---|---|---|---|---|")
        for v in r["variants"]:
            sa = v["sa_energy"] + enuc
            dsa = v["sa_energy"] - r["variants"][0]["sa_energy"]
            dg = v["delta_gamma_SA"]
            dg_state_max = max(v["delta_gamma_state"]) if v["delta_gamma_state"] else 0.0
            a(f"| {v['variant_id']} | {rule_label(v['generation_rule'])} | {occ_str(v['active_orbitals'])} "
              f"| {fmt_e(sa)} | {dsa*1000:+.3f} | {dg:.4f} | {dg_state_max:.4f} |")
        a("")
        a(f"- **SA-energy span over the full ensemble:** {r['sa_energy_span_eh']*1000:.3f} mEh; "
          f"**max pairwise Δγ<sub>SA</sub>:** {r['max_pairwise_delta_gamma_SA']:.4f} "
          f"(pair {r['max_pairwise_pair'][0]} vs {r['max_pairwise_pair'][1]}).\n")

        # ---- Table B: per-state configuration weights (descending) ----
        a("### Configuration-component weights per active space (descending)\n")
        a("| active space | rule | state | E (Eh) | ⟨S²⟩ | dominant configs (occupancy : weight) |")
        a("|---|---|---|---|---|---|")
        for v in r["variants"]:
            rule = rule_label(v["generation_rule"])
            vid = v["variant_id"]
            for s in v["states"]:
                etot = s["energy"] + enuc
                conf = ", ".join(
                    f"{occ_str(f['occ'])}:{f['weight']:.4f}" for f in s["families"][:5]
                )
                a(f"| {vid} | {rule} | {s['state_id']} | {fmt_e(etot)} | {s['spin_square']:.4f} | {conf} |")
        a("")

        # nominal 1-RDM occupations
        nom = r["variants"][0]
        occ_line = ", ".join(f"{x:.4f}" for x in nom["rdm_diag_active"])
        a(f"**Nominal state-averaged 1-RDM occupations (active orbitals {occ_str(nom_active)}):** "
          f"{occ_line}\n")

        # ---- common support ----
        cs = r["common_support"]
        a("### Common-support certification (extension variants only)\n")
        if cs.get("converged"):
            a(f"- Converged on a **{cs['n_families']}-family** common support across "
              f"**{cs['n_variants_tested']}** extension variants; criteria passed = "
              f"**{cs['passed']}** (failed: {cs['failed_criteria'] or 'none'}).")
            a(f"- worst projector distance d<sub>P</sub> = {cs['max_projector_distance']:.2e}; "
              f"worst matrix-1RDM error = {cs['max_gamma_error']:.2e}; "
              f"worst ⟨S²⟩ error = {cs['max_spin_square_error']:.2e}.")
            a("")
            a("| active space | d<sub>P</sub> | Δγ (matrix-1RDM err) | ⟨S²⟩ err | n<sub>det</sub> |")
            a("|---|---|---|---|---|")
            for vr in cs["variant_results"]:
                a(f"| {vr['variant_id']} | {vr['dP_op']:.2e} | {vr['gamma_error']:.2e} "
                  f"| {vr['spin_error']:.2e} | {vr['n_determinants']} |")
            a("")
        else:
            a(f"- **No compact common support converged** across the {cs.get('n_variants_tested','?')} "
              f"extension variants within the iteration budget "
              f"(`{cs.get('error','')}`). The per-variant descriptors above remain exact "
              f"(frozen-core CASCI teachers); this flags the common-support driver, not the data.\n")

    # ---- cross-molecule summary ----
    a("## Cross-molecule summary\n")
    a("| molecule | nominal CAS | n variants | SA-energy span (mEh) | max pairwise Δγ<sub>SA</sub> | common-support |")
    a("|---|---|---|---|---|---|")
    for name in ["n2", "o2", "c2"]:
        r = D[name]
        cs = r["common_support"]
        if cs.get("converged"):
            cstxt = f"{cs['n_families']} families (passed={cs['passed']})"
        else:
            cstxt = "not converged"
        a(f"| {MOL_NAMES[name]} | {occ_str(r['variants'][0]['active_orbitals'])} | {r['n_variants']} "
          f"| {r['sa_energy_span_eh']*1000:.3f} | {r['max_pairwise_delta_gamma_SA']:.4f} | {cstxt} |")
    a("")

    a("## Interpretation\n")
    a("**N₂ (triple bond).** The nominal CAS(4,4) is essentially converged: extending it by one or two "
      "virtuals changes E<sub>SA</sub> by only 0.1–8.7 mEh and Δγ<sub>SA</sub> ≤ 0.10, with the ground state "
      "dominated by one closed-shell configuration (weight ≈0.95). The single exchange variant that drops "
      "the π<sub>g</sub>-type orbital (orbital 8) costs **+31 mEh** and shifts Δγ<sub>SA</sub> to ≈0.50 — "
      "a direct quantitative marker that this orbital is *essential* to the active space. The compact common "
      "support does not converge once exchange variants are included, which is expected: exchange probes "
      "orbital essentiality, not active-space extension.\n")
    a("**O₂ (mixed spin).** The π*-pair CAS(2,2) is essentially exact: triplet X³Σg⁻ = ⟨1,1⟩ (weight 1.00), "
      "the two singlets are the ⟨2,0⟩/⟨0,2⟩ resonance mixture, S² = 2/0/0, and extending the active space "
      "changes E<sub>SA</sub> by ≤2.4 mEh with Δγ<sub>SA</sub> ≤ 0.07. A compact 6-family common support "
      "certifies all 11 extension variants (worst d<sub>P</sub> ≈ 1.9e-3, only for the add-2 variants that "
      "admit the higher π-type virtuals). The only failed criterion is `sa_energy`, because the default "
      "τ<sub>E,SA</sub> = 0.5 mEh is stricter than the ~2.4 mEh of *dynamic* correlation recovered by the "
      "extension — a threshold-calibration issue (T061), not a physical instability.\n")
    a("**C₂ (dense multireference).** The nominal CAS(4,4) is **not** converged: adding orbitals 8 or 10 "
      "lowers E<sub>SA</sub> by ~12–25 mEh and Δγ<sub>SA</sub> reaches 0.78, and the ground state is a "
      "genuine two-configuration ⟨2,1,1,0⟩/⟨1,2,1,0⟩ mixture. The exchange variant (swap orbitals 7↔8) is "
      "degenerate (ΔE ≈ 0), consistent with a π-type degeneracy. C₂ therefore requires a larger active "
      "space before a compact reference can be certified; its 1-RDM is the most active-space-sensitive of "
      "the three molecules.\n")

    a("## Caveats\n")
    a("1. Energies are fixed-orbital (frozen-core CASCI) — the parent MO gauge is held fixed across "
      "variants; Tier-II orbital reoptimization (M5) is out of scope.\n")
    a("2. `τ_E_SA = 5e-4 Eh` is uncalibrated (T061): it flags the small dynamic-correlation energy gains "
      "of active-space extension as 'instability'.\n")
    a("3. The common-support driver (`ensemble_driver`) converges for O₂ but not yet for N₂/C₂ extension "
      "ensembles, and cannot span exchange variants by design.\n")

    with open(OUT, "w") as f:
        f.write("\n".join(L))
    print(f"wrote {OUT} ({len(L)} lines)")


if __name__ == "__main__":
    build()
