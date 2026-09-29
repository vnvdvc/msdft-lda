"""Pytest configuration: make the src-layout packages importable without install.

Adds ``src/`` to ``sys.path`` so ``import msref`` (and ``import mlmsdft``) work
when tests are collected directly from a source checkout.  This keeps the msref
test suite independent of the heavy ``mlmsdft`` install requirements.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
