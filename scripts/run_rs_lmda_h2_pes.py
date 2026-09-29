#!/usr/bin/env python
"""Run spin-polarized RS-LMDA H2 dissociation curves.

The singlet calculation is a two-state state-averaged optimization (S0/S1)
in a fixed Sz=0 sector.  The triplet calculation is a one-state optimization
in the Sz=1 sector and reports T1.  The active space is varied while the
number of target states remains fixed.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy
import torch
import pyscf.gto

from mlmsdft.dft.density import TargetStateMultistateMatrixDensityCAS
from mlmsdft.dft.hamiltonian import HamiltonianTargetStateLMDA
from mlmsdft.dft.hamiltonian import minimize_subspace_energy
from mlmsdft.dft.spin import SpinType


def molecule(bond_length: float, basis: str, spin: int):
    return pyscf.gto.M(
        atom=f"H 0 0 0; H 0 0 {bond_length}",
        basis=basis,
        charge=0,
        spin=spin,
        verbose=0,
    )


def optimize_case(
        bond_length: float,
        basis: str,
        active_orbitals: int,
        state: str,
        omega: float,
        grid_level: int,
        maxiter: int,
        previous=None,
    ):
    if state in {"S0", "S1"}:
        mol = molecule(bond_length, basis, spin=0)
        nelec = (1, 1)
        target_states = 2
        state_index = 0 if state == "S0" else 1
    elif state == "T1":
        mol = molecule(bond_length, basis, spin=2)
        nelec = (2, 0)
        target_states = 1
        state_index = 0
    else:
        raise ValueError(f"unknown state {state!r}")

    msmd = TargetStateMultistateMatrixDensityCAS.from_guess(
        mol,
        norb=active_orbitals,
        nelec=nelec,
        target_states=target_states,
        spin_symmetry=True,
        spin_type=SpinType.POLARIZED,
        guess="rohf",
        target_parameterization="stiefel_k",
    )
    if previous is not None:
        msmd.orbital_rotation_params.data.copy_(previous.orbital_rotation_params.data)

    hamiltonian = HamiltonianTargetStateLMDA(
        mol,
        spin_type=SpinType.POLARIZED,
        omega=omega,
        grid_level=grid_level,
    )
    started = time.time()
    energies, msmd = minimize_subspace_energy(
        hamiltonian,
        msmd,
        optimizer="torch_lbfgs",
        maxiter=maxiter,
        gtol=1.0e-6,
    )
    elapsed = time.time() - started
    energy = float(energies[state_index].detach().cpu())
    diagnostics = msmd.invariant_diagnostics()
    return energy, msmd, {
        "seconds": elapsed,
        "energies": [float(value.detach().cpu()) for value in energies],
        "target_states": target_states,
        "spin_s2": diagnostics["spin_s2_min"],
        "target_orthonormality_error": diagnostics["target_orthonormality_error"],
        "orbital_orthonormality_error": diagnostics["orbital_orthonormality_error"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--active-orbitals", type=int, required=True)
    parser.add_argument("--basis", default="aug-cc-pvdz")
    parser.add_argument("--omega", type=float, default=0.4)
    parser.add_argument("--grid-level", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=80)
    parser.add_argument("--start", type=float, default=0.5)
    parser.add_argument("--stop", type=float, default=5.0)
    parser.add_argument("--step", type=float, default=0.25)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This PES driver requires a CUDA GPU")
    torch.set_default_dtype(torch.float64)
    torch.cuda.set_device(0)

    bond_lengths = numpy.arange(args.start, args.stop + 0.5 * args.step, args.step)
    rows = []
    metadata = {
        "basis": args.basis,
        "omega": args.omega,
        "grid_level": args.grid_level,
        "active_orbitals": args.active_orbitals,
        "bond_lengths_angstrom": bond_lengths.tolist(),
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
    }
    output_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_dir, exist_ok=True)
    with open(args.output + ".json", "w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)

    fieldnames = [
        "bond_length_angstrom", "active_orbitals", "state", "energy_hartree",
        "seconds", "energies", "target_states", "spin_s2",
        "target_orthonormality_error", "orbital_orthonormality_error",
    ]
    with open(args.output, "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        stream.flush()
        for state in ("S0", "S1", "T1"):
            previous = None
            for bond_length in bond_lengths:
                energy, previous, diagnostics = optimize_case(
                    float(bond_length), args.basis, args.active_orbitals,
                    state, args.omega, args.grid_level, args.maxiter,
                    previous=previous,
                )
                row = {
                    "bond_length_angstrom": float(bond_length),
                    "active_orbitals": args.active_orbitals,
                    "state": state,
                    "energy_hartree": energy,
                    **diagnostics,
                }
                rows.append(row)
                writer.writerow(row)
                stream.flush()
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
