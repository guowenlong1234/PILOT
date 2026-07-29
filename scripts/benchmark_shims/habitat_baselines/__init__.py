"""Minimal Habitat-Baselines package initializer for isolated benchmarks."""

import os
from pathlib import Path


root = os.environ.get("ETPR1_BENCH_HABITAT_BASELINES_ROOT", "").strip()
if not root:
    raise RuntimeError("ETPR1_BENCH_HABITAT_BASELINES_ROOT is required")

package_root = Path(root).resolve()
if not (package_root / "common" / "baseline_registry.py").is_file():
    raise RuntimeError(
        "Invalid ETPR1_BENCH_HABITAT_BASELINES_ROOT: "
        f"{package_root}"
    )

__path__ = [str(package_root)]
__version__ = "0.3.3"
