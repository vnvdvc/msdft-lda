#!/usr/bin/env python
"""Run the M4 active-space ensemble benchmarks and dump a JSON report.

Usage: python scripts/run_m4_ensemble.py [--molecule h2|n2|o2|c2] [--out FILE]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from msref.benchmarks.common import ensemble_ci_analysis
from msref.config import MSRefConfig
from msref.driver import ensemble_driver


def _run(mf, prob, label):
    fam, evals, report = ensemble_driver(prob)
    return {
        "label": label,
        "n_variants": len(prob.variants),
        "n_families": len(fam),
        "passed": report.passed,
        "failed": report.failed_criteria,
        "max_projector_distance": report.max_projector_distance,
        "max_gamma_error": report.max_gamma_error,
        "max_spin_error": report.max_spin_square_error,
        "sa_energy_span_eh": report.sa_energy_span_eh,
        "families": [list(f.occ) for f in fam],
        "variants": ensemble_ci_analysis(prob),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--molecule", default="h2", choices=["h2", "n2", "o2", "c2"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = MSRefConfig()
    cfg.active_space_ensemble.n_active_spaces = 8
    cfg.active_space_ensemble.max_add_orbitals = 2

    results = []
    if args.molecule == "h2":
        from msref.benchmarks.h2 import H2_GEOMETRIES, build_h2_ensemble

        for r in H2_GEOMETRIES:
            mf, prob = build_h2_ensemble(r=r, cfg=cfg)
            results.append(_run(mf, prob, f"h2_r{r}"))
    elif args.molecule == "n2":
        from msref.benchmarks.n2 import build_n2_ensemble

        for r in [0.90, 1.10, 1.50, 2.00, 2.50]:
            mf, prob = build_n2_ensemble(r=r, cfg=cfg)
            results.append(_run(mf, prob, f"n2_r{r}"))
    elif args.molecule == "o2":
        from msref.benchmarks.o2 import build_o2_ensemble

        for r in [1.21, 1.5, 2.0, 2.5]:
            mf, prob = build_o2_ensemble(r=r, cfg=cfg)
            results.append(_run(mf, prob, f"o2_r{r}"))
    elif args.molecule == "c2":
        from msref.benchmarks.c2 import build_c2_ensemble

        for r in [1.15, 1.24, 1.5, 2.0]:
            mf, prob = build_c2_ensemble(r=r, cfg=cfg)
            results.append(_run(mf, prob, f"c2_r{r}"))

    payload = {"config_hash": cfg.config_hash(), "results": results}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {args.out}")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
