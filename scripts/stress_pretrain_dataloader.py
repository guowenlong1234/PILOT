#!/usr/bin/env python3

"""Stress the real pretraining data pipeline without constructing the model."""

import argparse
import faulthandler
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_PACKAGE_ROOT = REPO_ROOT / "pretrain_src" / "pretrain_src"
if str(PRETRAIN_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PRETRAIN_PACKAGE_ROOT))

from data.dataset import R2RTextPathData  # noqa: E402
from data.loader import MetaLoader, move_to_cuda  # noqa: E402
from data.tasks import MlmDataset, SapDataset, mlm_collate, sap_collate  # noqa: E402


def load_configs(config_path, model_config_path):
    with Path(config_path).open("r", encoding="utf-8") as reader:
        config = json.load(reader)
    with Path(model_config_path).open("r", encoding="utf-8") as reader:
        model_config = json.load(reader)
    return config, model_config


def read_process_status(pid):
    values = {}
    status_path = Path(f"/proc/{pid}/status")
    try:
        lines = status_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return values
    for line in lines:
        if line.startswith(("VmRSS:", "VmSize:", "Threads:")):
            key, value = line.split(":", 1)
            values[key] = value.strip()
    return values


def dataloader_worker_pids(meta_loader):
    pids = []
    for iterator in meta_loader.name2iter.values():
        for worker in getattr(iterator, "_workers", ()):
            if worker.pid is not None:
                pids.append(worker.pid)
    return sorted(set(pids))


def build_nav_db(config, model_config):
    data_config = config["train_datasets"]["R2R"]
    return R2RTextPathData(
        data_config["train_traj_files"],
        data_config["img_ft_file"],
        data_config["dep_ft_file"],
        data_config["scanvp_cands_file"],
        data_config["connectivity_dir"],
        image_prob_size=model_config["image_prob_size"],
        image_feat_size=model_config["image_feat_size"],
        raw_image_feat_size=model_config["raw_image_feat_size"],
        rgb_encoder_type=model_config["rgb_encoder_type"],
        depth_feat_size=model_config["depth_feat_size"],
        angle_feat_size=model_config["angle_feat_size"],
        max_txt_len=config["max_txt_len"],
        in_memory=True,
        val_sample_num=None,
    )


def build_meta_loader(
    nav_db,
    tokenizer,
    *,
    batch_size,
    workers,
    pin_memory,
    start_method,
    accum_steps,
    ratio_dict,
):
    task_specs = {
        "mlm": (MlmDataset(nav_db, tokenizer), mlm_collate),
        "sap": (SapDataset(nav_db, tokenizer, end_vp_pos_ratio=0.15), sap_collate),
    }
    loaders = {}
    for task, (dataset, collate_fn) in task_specs.items():
        loader_kwargs = {
            "dataset": dataset,
            "sampler": RandomSampler(dataset),
            "batch_size": batch_size,
            "num_workers": workers,
            "pin_memory": pin_memory,
            "collate_fn": collate_fn,
            "drop_last": False,
        }
        if workers:
            loader_kwargs["multiprocessing_context"] = start_method
        loader = DataLoader(**loader_kwargs)
        loaders[task] = (loader, None, lambda _epoch: None)
    return MetaLoader(
        loaders,
        ratio_dict,
        accum_steps=accum_steps,
        distributed=False,
        device=torch.device("cpu"),
    )


def batch_checksum(task, batch):
    checksum = int(batch["txt_ids"].reshape(-1)[0])
    checksum += int(batch["traj_view_img_fts"].numel())
    checksum += int(batch["traj_view_dep_fts"].numel())
    if task == "sap":
        checksum += int(batch["global_act_labels"].sum())
    return checksum


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Stress the real MLM/SAP DataLoader path. Run only on the evaluation "
            "machine, and do not overlap CUDA modes with formal training."
        )
    )
    parser.add_argument(
        "--config",
        default="pretrain_src/run_pt/mix_pretrain_rae_dino.json",
    )
    parser.add_argument(
        "--model-config",
        default="pretrain_src/run_pt/mix_model_config_rae_dino.json",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--micro-batches",
        type=int,
        default=100000,
        help="Number of DataLoader batches; eight batches equal one formal update.",
    )
    parser.add_argument("--workers", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--start-method",
        choices=("fork", "spawn", "forkserver"),
        default="fork",
    )
    parser.add_argument("--accum-steps", type=int, default=8)
    parser.add_argument("--report-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--cuda-before-workers",
        action="store_true",
        help="Initialize CUDA before DataLoader workers, matching the risky old order.",
    )
    parser.add_argument(
        "--cuda-transfer",
        action="store_true",
        help="Move every batch to CUDA; requires an otherwise idle GPU.",
    )
    return parser


def validate_args(args):
    if args.workers == 0 and args.start_method != "fork":
        raise ValueError("--start-method only applies when --workers is nonzero")
    if args.cuda_transfer and not torch.cuda.is_available():
        raise RuntimeError("--cuda-transfer requested but CUDA is unavailable")
    if args.cuda_before_workers and not torch.cuda.is_available():
        raise RuntimeError("--cuda-before-workers requested but CUDA is unavailable")
    if args.micro_batches <= 0:
        raise ValueError("--micro-batches must be positive")
    if args.report_every <= 0:
        raise ValueError("--report-every must be positive")


def main(argv=None):
    faulthandler.enable(all_threads=True)
    args = build_parser().parse_args(argv)
    validate_args(args)

    os.chdir(REPO_ROOT)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config, model_config = load_configs(args.config, args.model_config)
    data_config = config["train_datasets"]["R2R"]
    tokenizer = AutoTokenizer.from_pretrained("./bert_config/xlm-roberta-base")
    nav_db = build_nav_db(config, model_config)

    device = torch.device("cpu")
    if args.cuda_before_workers:
        device = torch.device("cuda", 0)
        torch.empty(1, device=device)

    started_at = time.monotonic()
    meta_loader = build_meta_loader(
        nav_db,
        tokenizer,
        batch_size=args.batch_size,
        workers=args.workers,
        pin_memory=args.pin_memory,
        start_method=args.start_method,
        accum_steps=args.accum_steps,
        ratio_dict=data_config["mix_ratio"],
    )
    worker_pids = dataloader_worker_pids(meta_loader)
    if args.cuda_transfer:
        device = torch.device("cuda", 0)

    print(
        json.dumps(
            {
                "event": "started",
                "pid": os.getpid(),
                "worker_pids": worker_pids,
                "workers": args.workers,
                "pin_memory": args.pin_memory,
                "start_method": args.start_method,
                "cuda_before_workers": args.cuda_before_workers,
                "cuda_transfer": args.cuda_transfer,
                "main_status": read_process_status(os.getpid()),
                "worker_status": {
                    str(pid): read_process_status(pid) for pid in worker_pids
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )

    checksum = 0
    for index, (task, batch) in enumerate(meta_loader, start=1):
        if args.cuda_transfer:
            batch = move_to_cuda(batch, device)
            torch.cuda.synchronize(device)
        checksum += batch_checksum(task, batch)
        if index % args.report_every == 0 or index == args.micro_batches:
            elapsed = time.monotonic() - started_at
            print(
                json.dumps(
                    {
                        "event": "progress",
                        "micro_batches": index,
                        "formal_steps": index / args.accum_steps,
                        "elapsed_seconds": round(elapsed, 3),
                        "batches_per_second": round(index / elapsed, 3),
                        "checksum": checksum,
                        "main_status": read_process_status(os.getpid()),
                        "worker_status": {
                            str(pid): read_process_status(pid)
                            for pid in worker_pids
                        },
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if index >= args.micro_batches:
            break

    print(
        json.dumps(
            {
                "event": "finished",
                "micro_batches": args.micro_batches,
                "checksum": checksum,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
