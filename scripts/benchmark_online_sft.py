#!/usr/bin/env python3
"""Run a short online SFT throughput benchmark without writing checkpoints."""

import json
import os
import runpy
import sys
import time
import types
from pathlib import Path

import torch


def _install_minimal_habitat_baselines_package() -> None:
    root = os.environ.get("ETPR1_BENCH_HABITAT_BASELINES_ROOT", "").strip()
    if not root:
        return

    package_root = Path(root).resolve()
    if not (package_root / "common" / "baseline_registry.py").is_file():
        raise RuntimeError(
            "Invalid ETPR1_BENCH_HABITAT_BASELINES_ROOT: "
            f"{package_root}"
        )

    package = types.ModuleType("habitat_baselines")
    package.__file__ = str(package_root / "__init__.py")
    package.__package__ = "habitat_baselines"
    package.__path__ = [str(package_root)]
    package.__version__ = "0.3.3"
    sys.modules["habitat_baselines"] = package


def main() -> None:
    _install_minimal_habitat_baselines_package()

    from vlnce_baselines.ss_trainer_ETP_R1 import RLTrainer

    original_train_interval = RLTrainer._train_interval

    def timed_train_interval(self, interval, ml_weight, sample_ratio):
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        result = original_train_interval(
            self,
            interval,
            ml_weight,
            sample_ratio,
        )
        torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started
        world_size = int(self.config.GPU_NUMBERS)
        environments_per_rank = int(self.config.NUM_ENVIRONMENTS)
        accumulation_steps = int(self.config.IL.gradient_accumulation_steps)
        updates = int(interval)
        global_trajectories = (
            updates
            * world_size
            * environments_per_rank
            * accumulation_steps
        )
        payload = {
            "rank": int(self.local_rank),
            "world_size": world_size,
            "updates": updates,
            "environments_per_rank": environments_per_rank,
            "gradient_accumulation_steps": accumulation_steps,
            "global_trajectories": global_trajectories,
            "elapsed_seconds": elapsed,
            "seconds_per_update": elapsed / updates,
            "trajectories_per_second": global_trajectories / elapsed,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(self.device)
            / (1024**2),
            "peak_reserved_mib": torch.cuda.max_memory_reserved(self.device)
            / (1024**2),
        }
        print("ETPR1_BENCHMARK_RESULT=" + json.dumps(payload, sort_keys=True))
        return result

    def skip_checkpoint(self, iteration):
        print(
            "ETPR1_BENCHMARK_CHECKPOINT_SKIPPED="
            + json.dumps(
                {"rank": int(self.local_rank), "iteration": int(iteration)},
                sort_keys=True,
            )
        )

    RLTrainer._train_interval = timed_train_interval
    RLTrainer.save_checkpoint = skip_checkpoint

    local_rank = os.environ.get("LOCAL_RANK", "0")
    sys.argv = [
        "run.py",
        "--local_rank",
        local_rank,
        *sys.argv[1:],
    ]
    runpy.run_path(
        str(Path(__file__).resolve().parents[1] / "run.py"),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
