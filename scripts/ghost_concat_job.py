#!/usr/bin/env python3
"""Explicit train/eval entry for batched post-panorama ghost concat.

Uses the existing isolated runtime, resource checks and result manifest logic.
It does not join or restart the previous A/B/C experiment queues.
"""
import argparse
from pathlib import Path
import re
import json

from rgb_only_optimization import ROOT, BASE, common, execute

CONFIG = "run_r2r/iter_train_rae_dino_ghost_concat.yaml"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "eval", "watch"))
    parser.add_argument("--machine", choices=("server", "eval"), default="server")
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--batch", type=int, default=8, help="training batch/environments per rank")
    parser.add_argument("--environments", type=int, default=8, help="evaluation environments")
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--checkpoint", help="required for evaluation")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--panorama-context-mode",choices=("auto","front","world_exact_select"),default="world_exact_select")
    parser.add_argument('--observation-source',choices=('cube','direct'))
    parser.add_argument('--visual-precision',choices=('float32','fp16','bf16'))
    parser.add_argument('--dino-batch',type=int)
    parser.add_argument('--nwm-batch',type=int)
    parser.add_argument("--train-policy", action="store_true", help="jointly update navigation and fusion")
    parser.add_argument("--policy-lr", type=float, default=None)
    parser.add_argument("--fusion-lr", type=float, default=1e-5)
    parser.add_argument("--eval-iterations", default="200,400,600,800,1000,1200,1400,1600,1800,2000")
    parser.add_argument("--ready-timeout", type=int, default=172800)
    parser.add_argument("--output", default="data/logs/ghost_concat_20260908")
    parser.add_argument("--episodes", type=int, default=-1, help="positive counts are smoke tests only")
    parser.add_argument("--port", type=int, default=24851)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sync", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.wait_ready = args.action == "watch"
    if args.policy_lr is None:
        args.policy_lr = 2e-6 if args.train_policy else 1e-5
    if not re.fullmatch(r"\d+(,\d+)*", args.gpus) or len(set(args.gpus.split(","))) != len(args.gpus.split(",")):
        parser.error("--gpus must be distinct numeric indices")
    if min(args.batch, args.environments, args.iters, args.log_every) < 1:
        parser.error("batch, environments, iters and log-every must be positive")
    if any(v is not None and v < 1 for v in (args.dino_batch,args.nwm_batch)):
        parser.error('DINO and world-model batch sizes must be positive')
    if Path(args.output).is_absolute() or ".." in Path(args.output).parts:
        parser.error("--output must be project-relative without '..'")
    if args.action == "train" and args.machine != "server":
        parser.error("Training uses the server project environment")
    if args.action == "eval" and (not args.checkpoint or "," in args.gpus):
        parser.error("Evaluation requires --checkpoint and one GPU")
    if args.action == "watch" and (args.machine != "eval" or "," in args.gpus or args.checkpoint):
        parser.error("Watcher uses the evaluation host, one GPU, and derived checkpoint paths")
    expected = "/home/gwl/project/etpr1/ETP-R1" if args.machine == "server" else "/home/a6000/gwl/ETP-R1"
    if not args.dry_run and str(ROOT) != expected:
        parser.error("Wrong target project workspace: " + str(ROOT))
    prefix = "ghost_concat_v1_joint" if args.train_policy else "ghost_concat_v1"
    train_name = prefix + "_train"
    name = train_name if args.action == "train" else prefix + "_eval"
    opts = common(args.gpus, args.batch if args.action == "train" else args.environments)
    opts.update({"MODEL.RAENWM.rgb_fusion_type": "ghost_concat",
                 "MODEL.RAENWM.panorama_context_mode": args.panorama_context_mode,
                 "MODEL.RAENWM.ghost_concat_hidden_dim": 1536,
                 "MODEL.RAENWM.rgb_fusion_enabled": True,
                 "MODEL.RAENWM.rgb_fusion_trainable": True,
                 "MODEL.RAENWM.rgb_fusion_align_navigation_cls": False,
                 "MODEL.RAENWM.rgb_fusion_alpha": args.alpha,
                 "IL.freeze_navigation_backbone": not args.train_policy})
    if args.train_policy:
        opts.update({"IL.lr": args.policy_lr, "IL.rgb_fusion_lr": args.fusion_lr})
    for value,key in [(args.observation_source,'panorama_observation_source'),
                      (args.visual_precision,'panorama_visual_precision'),
                      (args.dino_batch,'panorama_encode_batch_size'),
                      (args.nwm_batch,'panorama_prediction_batch_size')]:
        if value is not None:opts['MODEL.RAENWM.'+key]=value
    if args.action == "train":
        opts.update({"IL.iters": args.iters, "IL.log_every": args.log_every,
                     "IL.batch_size": args.batch, "IL.is_requeue": args.resume,
                     "IL.ckpt_to_load": BASE, "IL.checkpoint_sync_enabled": args.sync,
                     "IL.checkpoint_sync_destination": "a6000@10.10.10.2:/home/a6000/gwl/ETP-R1/" +
                         args.output + "/train/" + name + "/checkpoints/" + name})
        execute(args, name, opts, BASE, "dagger", args.iters, config_file=CONFIG)
    elif args.action == "eval":
        opts.update({"EVAL.CKPT_PATH_DIR": str(ROOT / args.checkpoint),
                     "EVAL.EPISODE_COUNT": args.episodes, "EVAL.SAVE_RESULTS": True,
                     "EVAL.USE_CKPT_CONFIG": False, "EVAL.fast_eval": False})
        execute(args, name, opts, args.checkpoint, "eval", config_file=CONFIG)
    else:
        iterations = [int(value) for value in args.eval_iterations.split(",")]
        if iterations != sorted(set(iterations)) or not iterations or min(iterations) < 1:
            parser.error("--eval-iterations must be unique positive ascending iterations")
        summary = []
        from rgb_only_optimization import save
        if not args.dry_run:
            save(ROOT / args.output / "eval_summary.json", {
                "status": "waiting", "requested_iterations": iterations, "results": summary,
            })
        try:
            for iteration in iterations:
                checkpoint = args.output + "/train/" + train_name + "/checkpoints/" + train_name + f"/ckpt.iter{iteration}.pth"
                eval_name = name + f"_iter{iteration}"
                eval_opts = dict(opts)
                eval_opts.update({"EVAL.CKPT_PATH_DIR": str(ROOT / checkpoint), "EVAL.EPISODE_COUNT": args.episodes,
                                  "EVAL.SAVE_RESULTS": True, "EVAL.USE_CKPT_CONFIG": False, "EVAL.fast_eval": False})
                execute(args, eval_name, eval_opts, checkpoint, "eval", config_file=CONFIG)
                if not args.dry_run:
                    manifest = json.loads((ROOT / args.output / "eval" / eval_name / "manifest.json").read_text())
                    # Require exact original IDs, not merely the expected count.
                    if args.episodes == -1:
                        from summarize_rgb_only_optimization import load_episodes, expected_ids
                        ids = expected_ids(ROOT / "data/datasets/R2R_VLNCE_v1-3_preprocessed_xlmr/val_unseen/val_unseen.json.gz")
                        load_episodes(manifest["validation"]["episode_result"], 1839, ids)
                    summary.append({"iteration": iteration, "checkpoint_sha256": manifest["checkpoint_sha256"],
                                    **manifest["validation"]})
                    from rgb_only_optimization import save
                    save(ROOT / args.output / "eval_summary.json", {
                        "status": "completed" if iteration == iterations[-1] else "running",
                        "requested_iterations": iterations, "results": summary,
                    })
        except Exception as exc:
            if not args.dry_run:
                save(ROOT / args.output / "eval_summary.json", {
                    "status": "failed", "error": str(exc),
                    "requested_iterations": iterations, "results": summary,
                })
            raise


if __name__ == "__main__":
    main()
