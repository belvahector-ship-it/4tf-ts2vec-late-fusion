"""
scripts/check_env.py

Guard the pinned environment (ADR-001, INV-007): compare every `pkg==ver`
pin in requirements.txt and the Python minor version against what is
actually installed. Exits 1 on any mismatch so a notebook "Run All" stops
before training on a drifted stack.

Usage:  <venv-python> scripts/check_env.py
"""

from __future__ import annotations

import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

EXPECTED_PYTHON = (3, 11)
REQ = Path(__file__).resolve().parents[1] / "requirements.txt"


def pinned() -> dict[str, str]:
    pins = {}
    for line in REQ.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*([A-Za-z0-9_.\-]+)==([^\s#;]+)", line)
        if m:
            pins[m.group(1)] = m.group(2)
    return pins


def main() -> int:
    bad = []
    py = sys.version_info[:2]
    print(f"python   {sys.version.split()[0]:<14} expected {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]}.x")
    if py != EXPECTED_PYTHON:
        bad.append("python")

    for pkg, want in pinned().items():
        try:
            got = version(pkg)
        except PackageNotFoundError:
            got = "MISSING"
        ok = got.split("+")[0] == want  # ignore local label, e.g. 2.3.1+cu121
        print(f"{pkg:<14} {got:<14} expected {want}{'' if ok else '   <-- MISMATCH'}")
        if not ok:
            bad.append(pkg)

    try:
        import torch

        cuda = torch.cuda.is_available()
        print(f"cuda     build={torch.version.cuda} available={cuda} gpus={torch.cuda.device_count()}")
        if cuda:
            print("gpu     ", torch.cuda.get_device_name(0))
    except Exception as e:  # torch import failure is already a mismatch above
        print("torch import failed:", e)
        bad.append("torch-import")

    try:
        from ts2vec import TS2Vec  # noqa: F401  (vendored, via PYTHONPATH)
        print("ts2vec   import OK (vendored)")
    except Exception as e:
        print("ts2vec   import FAILED:", e)
        bad.append("ts2vec")

    if bad:
        print("\nENV CHECK FAILED:", ", ".join(bad))
        return 1
    print("\nENV CHECK PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
