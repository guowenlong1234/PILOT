#!/usr/bin/env python3

"""Reproduce training-server lockups with isolated, gradually riskier loads.

The script intentionally uses only the Python standard library and PyTorch.
Run ``monitor`` in a separate process before starting a stress mode so the
last successful telemetry sample survives in the log if the host locks up.
"""

import argparse
import faulthandler
import json
import math
import os
import queue
import shutil
import subprocess
import threading
import time
from datetime import timedelta
from pathlib import Path


MIB = 1024**2
GIB = 1024**3


def emit(event, **fields):
    payload = {
        "event": event,
        "monotonic_sec": round(time.monotonic(), 3),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        **fields,
    }
    print(json.dumps(payload, default=str, sort_keys=True), flush=True)


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def nonnegative_float(value):
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Isolate a training-server hard lockup across CPU memory, one GPU, "
            "two-GPU NCCL/NVLink, and buffered checkpoint I/O."
        )
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=("monitor", "cpu-memory", "gpu", "distributed"),
    )
    parser.add_argument("--duration-sec", type=positive_int, default=300)
    parser.add_argument("--report-every-sec", type=positive_int, default=5)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--cpu-memory-gib",
        type=positive_int,
        default=8,
        help="Resident memory touched by cpu-memory mode.",
    )
    parser.add_argument(
        "--cpu-chunk-mib",
        type=positive_int,
        default=256,
    )

    parser.add_argument(
        "--matrix-size",
        type=positive_int,
        default=4096,
        help="Square BF16 matrix size for CUDA GEMM.",
    )
    parser.add_argument(
        "--gpu-index",
        type=nonnegative_int,
        default=0,
        help="CUDA device index for gpu mode; distributed mode uses LOCAL_RANK.",
    )
    parser.add_argument(
        "--collective-mib",
        type=positive_int,
        default=256,
        help="BF16 NCCL all-reduce payload per rank.",
    )
    parser.add_argument(
        "--dist-timeout-sec",
        type=positive_int,
        default=90,
    )

    parser.add_argument(
        "--io-dir",
        type=Path,
        help="Enable rank-0 buffered writes in this directory.",
    )
    parser.add_argument(
        "--io-gib",
        type=nonnegative_float,
        default=0.0,
        help="Bytes written per checkpoint cycle; 4.5 matches the failed run.",
    )
    parser.add_argument("--io-chunk-mib", type=positive_int, default=64)
    parser.add_argument(
        "--io-start-delay-sec",
        type=nonnegative_float,
        default=30.0,
    )
    parser.add_argument(
        "--io-every-sec",
        type=positive_int,
        default=120,
    )
    parser.add_argument(
        "--io-fsync",
        action="store_true",
        help="Call fsync after the write. The failed training path did not.",
    )
    parser.add_argument(
        "--keep-io-file",
        action="store_true",
        help="Keep the last generated file after a clean exit.",
    )
    return parser


def validate_args(args):
    if args.io_gib and args.io_dir is None:
        raise ValueError("--io-dir is required when --io-gib is nonzero")
    if args.io_dir is not None and not args.io_gib:
        raise ValueError("--io-gib must be nonzero when --io-dir is set")
    if args.mode not in {"gpu", "distributed"} and args.io_dir is not None:
        raise ValueError(
            "checkpoint I/O is only valid in gpu or distributed mode"
        )
    if args.mode != "gpu" and args.gpu_index != 0:
        raise ValueError(
            "--gpu-index is only valid in gpu mode; distributed mode uses "
            "LOCAL_RANK"
        )


def read_text(path):
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError, OSError) as error:
        return f"<{type(error).__name__}:{error}>"


def run_probe(command, timeout_sec=3):
    try:
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "command": command,
            "error": f"{type(error).__name__}:{error}",
        }
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def memory_snapshot():
    selected = {}
    for line in read_text("/proc/meminfo").splitlines():
        key, _, value = line.partition(":")
        if key in {
            "MemAvailable",
            "MemFree",
            "Cached",
            "Dirty",
            "Writeback",
            "SwapFree",
        }:
            selected[key] = value.strip()
    return selected


def diskstats_snapshot():
    selected = []
    for line in read_text("/proc/diskstats").splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[2] in {"nvme0n1", "nvme1n1"}:
            selected.append(line)
    return selected


def run_monitor(args):
    started = time.monotonic()
    emit(
        "monitor_started",
        duration_sec=args.duration_sec,
        report_every_sec=args.report_every_sec,
    )
    while time.monotonic() - started < args.duration_sec:
        emit(
            "telemetry",
            loadavg=read_text("/proc/loadavg"),
            pressure={
                name: read_text(f"/proc/pressure/{name}")
                for name in ("cpu", "memory", "io")
            },
            memory=memory_snapshot(),
            diskstats=diskstats_snapshot(),
            gpu=run_probe(
                [
                    "nvidia-smi",
                    "--query-gpu=index,temperature.gpu,power.draw,"
                    "utilization.gpu,memory.used,pstate",
                    "--format=csv,noheader,nounits",
                ]
            ),
            nvlink=run_probe(["nvidia-smi", "nvlink", "-e"]),
        )
        time.sleep(args.report_every_sec)
    emit("monitor_finished")


def run_cpu_memory(args):
    import torch

    torch.manual_seed(args.seed)
    chunk_bytes = args.cpu_chunk_mib * MIB
    chunk_count = max(1, (args.cpu_memory_gib * GIB) // chunk_bytes)
    chunks = []
    emit(
        "cpu_memory_allocating",
        chunk_bytes=chunk_bytes,
        chunk_count=chunk_count,
        target_gib=args.cpu_memory_gib,
    )
    for index in range(chunk_count):
        tensor = torch.empty(chunk_bytes, dtype=torch.uint8)
        tensor.fill_((index * 37 + 11) % 251)
        chunks.append(tensor)

    started = time.monotonic()
    last_report = started
    passes = 0
    checksum = 0
    while time.monotonic() - started < args.duration_sec:
        for index, tensor in enumerate(chunks):
            pattern = (passes + index * 37 + 11) % 251
            tensor.fill_(pattern)
            sample = tensor[:: MIB]
            expected = pattern * sample.numel()
            observed = int(sample.sum(dtype=torch.int64))
            if observed != expected:
                raise RuntimeError(
                    f"memory mismatch chunk={index}: {observed} != {expected}"
                )
            checksum += observed
        passes += 1
        now = time.monotonic()
        if now - last_report >= args.report_every_sec:
            emit(
                "cpu_memory_progress",
                passes=passes,
                checksum=checksum,
                memory=memory_snapshot(),
            )
            last_report = now
    emit("cpu_memory_finished", passes=passes, checksum=checksum)


def checkpoint_writer(args, stop_event, errors, rank):
    if rank != 0 or args.io_dir is None:
        return
    try:
        args.io_dir.mkdir(parents=True, exist_ok=True)
        destination = args.io_dir / "hardlock-repro-checkpoint.bin"
        temporary = args.io_dir / (
            f".hardlock-repro-checkpoint.tmp.{os.getpid()}"
        )
        total_bytes = int(args.io_gib * GIB)
        chunk = bytes(min(args.io_chunk_mib * MIB, total_bytes))
        if stop_event.wait(args.io_start_delay_sec):
            return

        cycle = 0
        while not stop_event.is_set():
            cycle += 1
            started = time.monotonic()
            written = 0
            free_bytes = shutil.disk_usage(args.io_dir).free
            required_bytes = total_bytes + GIB
            if free_bytes < required_bytes:
                raise RuntimeError(
                    f"insufficient free space in {args.io_dir}: "
                    f"{free_bytes} < {required_bytes}"
                )
            emit(
                "checkpoint_write_started",
                cycle=cycle,
                path=str(destination),
                total_bytes=total_bytes,
                free_bytes=free_bytes,
                fsync=args.io_fsync,
            )
            with temporary.open("wb", buffering=0) as writer:
                next_progress = GIB
                while written < total_bytes and not stop_event.is_set():
                    count = min(len(chunk), total_bytes - written)
                    actual = writer.write(chunk[:count])
                    if actual is None or actual <= 0:
                        raise RuntimeError(
                            f"checkpoint write made no progress: {actual}"
                        )
                    written += actual
                    if written >= next_progress:
                        emit(
                            "checkpoint_write_progress",
                            cycle=cycle,
                            written_bytes=written,
                            total_bytes=total_bytes,
                        )
                        next_progress += GIB
                if stop_event.is_set():
                    emit(
                        "checkpoint_write_interrupted",
                        cycle=cycle,
                        written_bytes=written,
                        total_bytes=total_bytes,
                    )
                    break
                if args.io_fsync:
                    os.fsync(writer.fileno())
            os.replace(temporary, destination)
            emit(
                "checkpoint_write_finished",
                cycle=cycle,
                written_bytes=written,
                elapsed_sec=round(time.monotonic() - started, 3),
                memory=memory_snapshot(),
            )
            if stop_event.wait(args.io_every_sec):
                break
        if destination.exists() and not args.keep_io_file:
            destination.unlink()
            emit("checkpoint_file_removed", path=str(destination))
    except BaseException as error:  # propagate worker failures to the main loop
        errors.put(error)
        stop_event.set()
    finally:
        if "temporary" in locals() and temporary.exists():
            temporary.unlink()


def start_checkpoint_thread(args, rank):
    stop_event = threading.Event()
    errors = queue.Queue()
    thread = None
    if rank == 0 and args.io_dir is not None:
        thread = threading.Thread(
            target=checkpoint_writer,
            args=(args, stop_event, errors, rank),
            name="checkpoint-writer",
            daemon=True,
        )
        thread.start()
    return stop_event, errors, thread


def stop_checkpoint_thread(stop_event, errors, thread):
    stop_event.set()
    if thread is not None:
        thread.join(timeout=30)
        if thread.is_alive():
            raise RuntimeError("checkpoint writer did not stop within 30 seconds")
    if not errors.empty():
        raise errors.get()


def cuda_loop(args, *, distributed):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")

    dist = None
    rank = 0
    world_size = 1
    local_rank = args.gpu_index
    if distributed:
        import torch.distributed as dist_module

        dist = dist_module
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(seconds=args.dist_timeout_sec),
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        if world_size != 2:
            raise ValueError(
                f"distributed mode requires exactly two ranks, got {world_size}"
            )

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    matrix_a = torch.randn(
        (args.matrix_size, args.matrix_size),
        dtype=torch.bfloat16,
        device=device,
    )
    matrix_b = torch.randn_like(matrix_a)
    collective = None
    if distributed:
        elements = args.collective_mib * MIB // 2
        collective = torch.full(
            (elements,),
            fill_value=rank + 1,
            dtype=torch.bfloat16,
            device=device,
        )
        dist.barrier()

    stop_event, errors, io_thread = start_checkpoint_thread(args, rank)
    started = time.monotonic()
    last_report = started
    iterations = 0
    checksum = 0.0
    emit(
        "cuda_started",
        distributed=distributed,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device_name=torch.cuda.get_device_name(device),
        pci_domain_id=getattr(properties, "pci_domain_id", None),
        pci_bus_id=getattr(properties, "pci_bus_id", None),
        pci_device_id=getattr(properties, "pci_device_id", None),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        matrix_size=args.matrix_size,
        collective_mib=args.collective_mib if distributed else 0,
        nccl_p2p_disable=os.environ.get("NCCL_P2P_DISABLE"),
        io_gib=args.io_gib,
        io_dir=str(args.io_dir) if args.io_dir else None,
    )
    try:
        while time.monotonic() - started < args.duration_sec:
            output = torch.mm(matrix_a, matrix_b)
            if distributed:
                collective.add_(rank + 1)
                dist.all_reduce(collective)
                collective.mul_(1.0 / world_size)
            iterations += 1

            if not errors.empty():
                raise errors.get()
            now = time.monotonic()
            if now - last_report >= args.report_every_sec:
                torch.cuda.synchronize(device)
                checksum = float(output[0, 0])
                collective_sample = (
                    float(collective[0]) if distributed else None
                )
                if not math.isfinite(checksum) or (
                    collective_sample is not None
                    and not math.isfinite(collective_sample)
                ):
                    raise RuntimeError(
                        "CUDA result became non-finite: "
                        f"gemm={checksum}, collective={collective_sample}"
                    )
                emit(
                    "cuda_progress",
                    rank=rank,
                    iterations=iterations,
                    checksum=checksum,
                    collective_sample=collective_sample,
                    allocated_mib=round(
                        torch.cuda.memory_allocated(device) / MIB, 1
                    ),
                    reserved_mib=round(
                        torch.cuda.memory_reserved(device) / MIB, 1
                    ),
                )
                last_report = now
        torch.cuda.synchronize(device)
        if distributed:
            dist.barrier()
        emit(
            "cuda_finished",
            rank=rank,
            iterations=iterations,
            checksum=checksum,
        )
    finally:
        stop_checkpoint_thread(stop_event, errors, io_thread)
        if dist is not None and dist.is_initialized():
            dist.destroy_process_group()


def main(argv=None):
    faulthandler.enable(all_threads=True)
    args = build_parser().parse_args(argv)
    validate_args(args)
    emit("arguments", values=vars(args))

    if args.mode == "monitor":
        run_monitor(args)
    elif args.mode == "cpu-memory":
        run_cpu_memory(args)
    elif args.mode == "gpu":
        cuda_loop(args, distributed=False)
    elif args.mode == "distributed":
        cuda_loop(args, distributed=True)
    else:  # argparse prevents this path
        raise AssertionError(args.mode)


if __name__ == "__main__":
    main()
