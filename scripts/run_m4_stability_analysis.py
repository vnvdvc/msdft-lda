"""Generate per-active-space stability descriptors for the N2/O2/C2 ensembles.

For each active-space variant of a molecule this extracts:
  * per-state configuration-family weights (descending |c|^2), energy, spin^2;
  * state-averaged (SA) energy and its variation across the ensemble;
  * the SA matrix-1RDM and its variation vs the nominal active space
    (Frobenius norm, per-state and state-averaged, plus max pairwise);
  * the common-support certification (ensemble_driver), if it converges.

Output: a JSON file (and a rendered Markdown block) suitable for a report.
"""

from __future__ import annotations

import json
import sys
import traceback

import numpy as np

sys.path.insert(0, "/Users/vnvdvc/Documents/Agents/MSDFT/msdft-lda/src")

from msref.config import MSRefConfig
from msref.config_pool import family_weights_for_state
from msref.driver import align_variant_states, build_ensemble_pool, ensemble_driver
from msref.models import SpatialOccupationKey
from msref import models

from pyscf import gto, scf  # noqa: E402

TOP_K = 6


def _json_default(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _active_occ(family: SpatialOccupationKey, active_orbitals):
    return tuple(family.occ[o] for o in active_orbitals)


def _fro(a, b):
    return float(np.linalg.norm(a - b, ord="fro"))


def _sa_rdm(teacher, weights):
    """State-averaged 1-RDM (diagonal blocks, real part)."""
    nstate = len(teacher.state_ids)
    acc = np.zeros(teacher.gamma1_parent.shape[2:], dtype=float)
    for s in range(nstate):
        acc += weights[s] * np.real(teacher.gamma1_parent[s, s])
    return acc


def _alignment(prob):
    """Overlap-based state alignment vs nominal; fall back to identity."""
    nominal = prob.teachers[prob.nominal_variant.variant_id]
    out = {}
    for variant in prob.variants:
        t = prob.teachers[variant.variant_id]
        if variant.variant_id == prob.nominal_variant.variant_id:
            out[variant.variant_id] = list(range(len(t.state_ids)))
            continue
        # build |overlap| matrix [variant_state, nominal_state]
        o = np.zeros((len(t.state_ids), len(nominal.state_ids)))
        for i, vi in enumerate(t.ci_vectors):
            di = {k: c for k, c in zip(vi.determinant_keys, vi.coefficients)}
            for j, vj in enumerate(nominal.ci_vectors):
                dj = {k: c for k, c in zip(vj.determinant_keys, vj.coefficients)}
                keys = set(di) & set(dj)
                o[i, j] = abs(sum(np.conj(di[k]) * dj[k] for k in keys))
        perm = [int(x) for x in np.argmax(o, axis=1)]  # best nominal state per variant state
        out[variant.variant_id] = perm
    return out


def _describe_variant(prob, variant, teacher, weights, nominal_rdm, align):
    norb = prob.nparent
    active = list(variant.active_orbitals)

    states = []
    for s in range(len(teacher.state_ids)):
        fw = family_weights_for_state(teacher.ci_vectors[s], norb)
        # reduce to active-orbital occupancy, dedup, sort descending
        fams = {}
        for f, w in fw.items():
            occ = _active_occ(f, active)
            fams[occ] = fams.get(occ, 0.0) + w
        top = sorted(fams.items(), key=lambda kv: -kv[1])[:TOP_K]
        states.append({
            "state_id": teacher.state_ids[s].name,
            "spin_S": teacher.state_ids[s].spin_S,
            "energy": float(teacher.energies_eh[s]),
            "spin_square": float(teacher.spin_square[s]),
            "families": [{"occ": list(o), "weight": round(float(w), 6)} for o, w in top],
        })

    sa_energy = float(np.dot(weights, teacher.energies_eh))
    rdm = _sa_rdm(teacher, weights)

    # 1-RDM variation vs nominal (state-averaged)
    delta_gamma_SA = _fro(rdm, nominal_rdm)
    # per-state 1-RDM variation vs nominal (aligned)
    delta_gamma_state = []
    for s in range(len(teacher.state_ids)):
        nom_s = align[s]
        gs = np.real(teacher.gamma1_parent[s, s])
        gn = np.real(
            prob.teachers[prob.nominal_variant.variant_id].gamma1_parent[nom_s, nom_s]
        )
        delta_gamma_state.append(_fro(gs, gn))

    # active-orbital diagonal occupations (state-averaged)
    rdm_diag_active = [round(float(rdm[o, o]), 6) for o in active]

    return {
        "variant_id": variant.variant_id,
        "generation_rule": variant.generation_rule,
        "active_orbitals": active,
        "n_active": len(active),
        "sa_energy": round(sa_energy, 8),
        "states": states,
        "rdm_diag_active": rdm_diag_active,
        "delta_gamma_SA": round(delta_gamma_SA, 8),
        "delta_gamma_state": [round(x, 8) for x in delta_gamma_state],
        "align_perm": align,
    }


def analyze(name, build_fn, n_active_spaces=None, extra_cfg=None):
    cfg = MSRefConfig()
    if n_active_spaces is not None:
        cfg.active_space_ensemble.n_active_spaces = n_active_spaces
    if extra_cfg:
        for k, v in extra_cfg.items():
            setattr(cfg.active_space_ensemble, k, v)

    mf, prob = build_fn(cfg)
    weights = [float(w) for w in prob.state_weights]
    align = _alignment(prob)

    nominal_teacher = prob.teachers[prob.nominal_variant.variant_id]
    nominal_rdm = _sa_rdm(nominal_teacher, weights)

    variants = []
    for variant in prob.variants:
        teacher = prob.teachers[variant.variant_id]
        variants.append(
            _describe_variant(prob, variant, teacher, weights, nominal_rdm, align[variant.variant_id])
        )

    # max pairwise SA 1-RDM variation across the ensemble
    rdm_sa = {
        v.variant_id: _sa_rdm(prob.teachers[v.variant_id], weights)
        for v in prob.variants
    }
    vids = [v.variant_id for v in prob.variants]
    max_pairwise = 0.0
    pair = (None, None)
    for i in range(len(vids)):
        for j in range(i + 1, len(vids)):
            d = _fro(rdm_sa[vids[i]], rdm_sa[vids[j]])
            if d > max_pairwise:
                max_pairwise, pair = d, (vids[i], vids[j])

    # SA energy variation
    sa_energies = [v["sa_energy"] for v in variants]
    delta_SA_vs_nominal = [
        round(v["sa_energy"] - variants[0]["sa_energy"], 8) for v in variants
    ]

    # common-support certification (best-effort), on the extension (add-only)
    # subset: exchange variants swap orbitals out of the active space, which is
    # an "orbital-essentiality" probe rather than an active-space-extension test,
    # and the compact common-support driver is not designed to span them.
    common_support = None
    add_variants = [v for v in prob.variants if v.generation_rule != "exchange"]
    if add_variants:
        from msref.driver import EnsembleProblem

        add_teachers = {v.variant_id: prob.teachers[v.variant_id] for v in add_variants}
        add_problem = EnsembleProblem(
            parent=prob.parent, variants=add_variants, teachers=add_teachers,
            state_weights=prob.state_weights, h1e_parent=prob.h1e_parent,
            eri_parent=prob.eri_parent, cfg=prob.cfg,
        )
        try:
            fam, evals, report = ensemble_driver(add_problem)
            common_support = {
                "converged": True,
                "n_variants_tested": len(add_variants),
                "n_families": len(fam),
                "passed": report.passed,
                "failed_criteria": report.failed_criteria,
                "max_projector_distance": report.max_projector_distance,
                "max_gamma_error": report.max_gamma_error,
                "max_spin_square_error": report.max_spin_square_error,
                "sa_energy_span_eh": report.sa_energy_span_eh,
                "families": [list(f.occ) for f in fam],
                "variant_results": [
                    {
                        "variant_id": e.variant_id,
                        "dP_op": e.dP_op,
                        "gamma_error": e.gamma_error,
                        "spin_error": e.spin_error,
                        "n_determinants": e.n_determinants,
                    }
                    for e in evals
                ],
            }
        except Exception as exc:  # noqa: BLE001
            common_support = {
                "converged": False,
                "n_variants_tested": len(add_variants),
                "error": f"{type(exc).__name__}: {exc}",
            }

    return {
        "name": name,
        "basis": getattr(getattr(mf, "mol", None), "basis", None),
        "geometry": getattr(getattr(mf, "mol", None), "atom", None),
        "e_nuc": float(mf.mol.energy_nuc()),
        "nmo": int(mf.mo_coeff.shape[1]),
        "nominal_variant_id": prob.nominal_variant.variant_id,
        "n_variants": len(prob.variants),
        "state_weights": weights,
        "delta_SA_vs_nominal": delta_SA_vs_nominal,
        "sa_energy_span_eh": round(max(sa_energies) - min(sa_energies), 8),
        "max_pairwise_delta_gamma_SA": round(max_pairwise, 8),
        "max_pairwise_pair": pair,
        "variants": variants,
        "common_support": common_support,
    }


def main():
    from msref.benchmarks.n2 import build_n2_ensemble
    from msref.benchmarks.o2 import build_o2_ensemble
    from msref.benchmarks.c2 import build_c2_ensemble
    from msref.benchmarks.common import build_ensemble_problem

    # O2 with an expanded virtual buffer for a richer active-space scan
    def _o2_rich(cfg):
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

    results = {}
    for name, fn, nspaces in [
        ("n2", lambda c: build_n2_ensemble(r=1.10, cfg=c), 12),
        ("o2", _o2_rich, 11),
        ("c2", lambda c: build_c2_ensemble(r=1.24, cfg=c), 12),
    ]:
        print(f"=== analyzing {name} ===", flush=True)
        try:
            results[name] = analyze(name, fn, n_active_spaces=nspaces)
            r = results[name]
            print(f"  variants={r['n_variants']} SA_span={r['sa_energy_span_eh']:.6f} "
                  f"max_pairwise_dg={r['max_pairwise_delta_gamma_SA']:.6f}")
            cs = r["common_support"]
            print(f"  common_support: {cs.get('converged')} "
                  f"{cs.get('failed_criteria', cs.get('error'))}")
        except Exception:
            print(f"  FAILED:\n{traceback.format_exc()}")

    out = "/Users/vnvdvc/Documents/Agents/MSDFT/MS-SCF/msref_m4_stability_data.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=1, default=_json_default)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
