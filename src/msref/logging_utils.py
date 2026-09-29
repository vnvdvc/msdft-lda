"""Runtime provenance and version capture for reproducible msref runs."""

from __future__ import annotations

import datetime
import os
import platform
import subprocess
from typing import Any, Optional

from . import __version__


def _package_version(name: str) -> Optional[str]:
    try:
        mod = __import__(name)
    except Exception:
        return None
    return getattr(mod, "__version__", None)


def _git_commit(path: Optional[str] = None) -> Optional[str]:
    """Return the current git commit hash, or None if unavailable."""
    cwd = path or os.getcwd()
    try:
        out = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return None


def collect_provenance(
    config_hash: Optional[str] = None, repo_path: Optional[str] = None
) -> dict[str, Any]:
    """Collect package/runtime/version provenance as a JSON-compatible dict."""
    prov: dict[str, Any] = {
        "package": "msref",
        "package_version": __version__,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy_version": _package_version("numpy"),
        "scipy_version": _package_version("scipy"),
        "pyscf_version": _package_version("pyscf"),
        "h5py_version": _package_version("h5py"),
        "yaml_version": _package_version("yaml"),
        "created_utc": datetime.datetime.now(datetime.timezone.utc)
        .isoformat(timespec="seconds"),
        "git_commit": _git_commit(repo_path),
    }
    if config_hash is not None:
        prov["config_hash"] = config_hash
    return prov


__all__ = ["collect_provenance"]
