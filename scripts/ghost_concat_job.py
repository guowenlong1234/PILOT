#!/usr/bin/env python3
"""Explicit train/eval entry for batched post-panorama ghost concat.

Uses the existing isolated runtime, resource checks and result manifest logic.
It does not join or restart the previous A/B/C experiment queues.
"""
import argparse
from pathlib import Path
import re

from rgb_only_optimization import ROOT, BASE, common, execute

CONFIG = "run_r2r/iter_train_rae_dino_ghost_concat.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "eval"))
    parser.add_argument("--machine", choices=("server", "eval"), default="server")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--batch", type=int, default=8, help="training batch/environments per rank")
    parser.add_argument("--environments", type=int, default=8, help="evaluation environments")
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--checkpoint", help="required for evaluation")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--output", default="data/logs/ghost_concat_20260908")
    parser.add_argument("--episodes", type=int, default=-1, help="positive counts are smoke tests only")
    parser.add_argument("--port", type=int, default=24851)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.wait_ready = False
    args.ready_timeout = 172800
    if not re.fullmatch(r"\d+(,\d+)*", args.gpus) or len(set(args.gpus.split(","))) != len(args.gpus.split(",")):
        parser.error("--gpus must be distinct numeric indices")
    if min(args.batch, args.environments, args.iters, args.log_every) < 1:
        parser.error("batch, environments, iters and log-every must be positive")
    if Path(args.output).is_absolute() or ".." in Path(args.output).parts:
        parser.error("--output must be project-relative without '..'")
    if args.action == "train" and args.machine != "server":
        parser.error("Training uses the server project environment")
    if args.action == "eval" and (not args.checkpoint or "," in args.gpus):
        parser.error("Evaluation requires --checkpoint and one GPU")
    expected = "/home/gwl/project/etpr1/ETP-R1" if args.machine == "server" else "/home/a6000/gwl/ETP-R1"
    if not args.dry_run and str(ROOT) != expected:
        parser.error("Wrong target project workspace: " + str(ROOT))
    name = "ghost_concat_v1_train" if args.action == "train" else "ghost_concat_v1_eval"
    opts = common(args.gpus, args.batch if args.action == "train" else args.environments)
    opts.update({"MODEL.RAENWM.rgb_fusion_type": "ghost_concat",
                 "MODEL.RAENWM.ghost_concat_hidden_dim": 1536,
                 "MODEL.RAENWM.rgb_fusion_enabled": True,
                 "MODEL.RAENWM.rgb_fusion_trainable": True,
                 "MODEL.RAENWM.rgb_fusion_align_navigation_cls": False,
                 "MODEL.RAENWM.rgb_fusion_alpha": args.alpha,
                 "IL.freeze_navigation_backbone": True})
    if args.action == "train":
        opts.update({"IL.iters": args.iters, "IL.log_every": args.log_every,
                     "IL.batch_size": args.batch, "IL.is_requeue": args.resume,
                     "IL.ckpt_to_load": BASE, "IL.checkpoint_sync_enabled": args.sync,
                     "IL.checkpoint_sync_destination": "a6000@10.10.10.2:/home/a6000/gwl/ETP-R1/" +
                         args.output + "/train/" + name + "/checkpoints/" + name})
        execute(args, name, opts, BASE, "dagger", args.iters, config_file=CONFIG)
    else:
        opts.update({"EVAL.CKPT_PATH_DIR": str(ROOT / args.checkpoint),
                     "EVAL.EPISODE_COUNT": args.episodes, "EVAL.SAVE_RESULTS": True,
                     "EVAL.USE_CKPT_CONFIG": False, "EVAL.fast_eval": False})
        execute(args, name, opts, args.checkpoint, "eval", config_file=CONFIG)


if __name__ == "__main__":
    main()
