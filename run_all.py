#!/usr/bin/env python3
"""Run the complete thesis experiment from the source-package root."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> None:
    """Delegate to the complete pipeline while preserving the active interpreter."""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    environment.setdefault("MPLCONFIGDIR", "/tmp/acs-matplotlib")
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "src" / "run_all.py")],
        cwd=PROJECT_ROOT,
        env=environment,
        check=True,
    )


if __name__ == "__main__":
    main()

