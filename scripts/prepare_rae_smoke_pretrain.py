#!/usr/bin/env python3

import argparse
import copy
import json
from pathlib import Path

import torch


SMOKE_OVERRIDES = {
    "train_batch_size": 1,
    "val_batch_size": 1,
    "val_sample_num": 1,
    "n_workers": 0,
    "pin_mem": False,
    "num_train_steps": 1,
    "valid_steps": 1,
    "log_steps": 1,
    "warmup_steps": 0,
    "fp16": False,
}
EXPECTED_IMG_LINEAR_SUFFIXES = {"weight", "bias"}


def snapshot_initial_img_linear(model, destination):
    parameters = {}
    marker = "img_embeddings.img_linear."
    for key, value in model.state_dict().items():
        if marker not in key:
            continue
        suffix = key.split(marker, 1)[1]
        if suffix in EXPECTED_IMG_LINEAR_SUFFIXES:
            parameters[key] = value.detach().cpu().clone()
    actual_suffixes = {
        key.split(marker, 1)[1]
        for key in parameters
    }
    if actual_suffixes != EXPECTED_IMG_LINEAR_SUFFIXES:
        raise ValueError(
            "initial model does not contain the complete 768-dim img_linear; "
            f"found={sorted(actual_suffixes)}"
        )
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(parameters, destination)


def _copy_first_jsonl_record(source, destination):
    source = Path(source)
    with source.open("r", encoding="utf-8") as reader:
        first_line = reader.readline()
    if not first_line:
        raise ValueError(f"empty JSONL source: {source}")
    json.loads(first_line)
    destination.write_text(first_line.rstrip("\n") + "\n", encoding="utf-8")


def prepare_smoke_config(source_config, output_config, seed=20260710):
    source_config = Path(source_config)
    output_config = Path(output_config)
    config = json.loads(source_config.read_text(encoding="utf-8"))
    prepared = copy.deepcopy(config)
    prepared.update(SMOKE_OVERRIDES)
    prepared["seed"] = seed

    output_config.parent.mkdir(parents=True, exist_ok=False)
    dataset = prepared["train_datasets"]["R2R"]
    fixture_specs = (
        ("train_traj_files", "train.jsonl"),
        ("val_unseen_r2r_traj_files", "r2r_val.jsonl"),
        ("val_unseen_rxr_traj_files", "rxr_val.jsonl"),
    )
    for field, filename in fixture_specs:
        sources = dataset[field]
        if not sources:
            raise ValueError(f"pretrain config has no source for {field}")
        fixture_path = output_config.parent / filename
        _copy_first_jsonl_record(sources[0], fixture_path)
        dataset[field] = [str(fixture_path.resolve())]

    output_config.write_text(
        json.dumps(prepared, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return prepared


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--seed", type=int, default=20260710)
    args = parser.parse_args()
    prepare_smoke_config(
        args.source_config,
        args.output_config,
        seed=args.seed,
    )
    print(f"SMOKE_PRETRAIN_CONFIG={args.output_config}")


if __name__ == "__main__":
    main()
