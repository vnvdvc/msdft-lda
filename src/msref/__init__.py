"""msref: robust compact multi-state reference construction from spin-mixed SA-CASSCF.

This package builds a compact, spin-complete, common multi-state selected
reference subspace from a spin-mixed state-averaged CASSCF teacher.  The
primary scientific objects are the selected state manifold (projector) and its
matrix one-particle reduced density matrix (1-RDM), not the individual
truncated roots.

The selection unit is the *spatial-occupation family* (all spin assignments of
a spatial occupancy pattern), which preserves closure under S^2 so that the
selected spin-free Hamiltonian retains spin-pure eigenstates.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
