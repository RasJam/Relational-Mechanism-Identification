#!/usr/bin/env python3
"""Run the complete reproducible benchmark pipeline."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = (
    "generate_dataset.py",
    "validate_dataset.py",
    "build_features.py",
    "evaluate_baselines.py",
    "train_encoder_decoder.py",
    "analyze_neural_slices.py",
    "analyze_results.py",
    "create_visuals.py",
)


# -----------------------------------------------------------------------------
# Execute every pipeline stage with a shared Python environment
# -----------------------------------------------------------------------------

def main():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment.setdefault("MPLCONFIGDIR", "/tmp/acs-matplotlib")
    for script in SCRIPTS:
        print(f"\n=== {script} ===", flush=True)
        subprocess.run(
            [sys.executable, str(ROOT / "src" / script)],
            cwd=ROOT,
            env=environment,
            check=True,
        )
    print("\nActive Constraint Space pipeline completed.")


if __name__ == "__main__":
    main()
