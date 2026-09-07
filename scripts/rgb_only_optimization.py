#!/usr/bin/env python3
"""Finite, restartable RGB-only experiments. Run on server or evaluation HOST.

Examples (after resource/preflight checks and Git synchronization):
  python3 scripts/rgb_only_optimization.py train --branch B --gpus 0,1
  python3 scripts/rgb_only_optimization.py train --branch C --gpus 0 --batch 16
  python3 scripts/rgb_only_optimization.py eval --machine eval --cases base,legacy,a0,a025,a05,a1
  python3 scripts/rgb_only_optimization.py eval --machine eval --cases A,B,C

Use nohup setsid for a durable finite worker. Rerunning eval skips only validated
completed manifests; training requires --resume after any interrupted attempt.
No processes are killed and no checkpoints or prior results are deleted.
"""
import argparse
import contextlib
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "run_r2r/iter_train_rae_dino_native_cls_rgb_fusion.yaml"
BASE = "pretrained/active_lookahead/base_iter14200.pth"
BASE_EVAL = "pretrained/active_lookahead/ckpt.iter14200.pth"
BASE_SHA = "1694b175d913404bfef8a53519d6f405db5de7f8b051e8343700125e43f05c61"
GRAD = ("data/logs/raenwm_rgb_fusion/native_cls_sft_lowlevel_bs16_gradfix_ea860aa_20260904/"
        "checkpoints/etpr1_native_cls_rgb_fusion_sft_lowlevel_bs16_gradfix_ea860aa/ckpt.iter5200.pth")
DEFAULT_ROOT = "data/logs/rgb_only_optimization_20260907"
VERSIONS = ("import sys,torch,transformers,habitat,habitat_sim; "
            "print(dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,"
            "transformers=transformers.__version__,habitat=habitat.__version__,habitat_sim=habitat_sim.__version__))")


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def output(command):
    return subprocess.check_output(command, cwd=ROOT, text=True).strip()


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def runtime(machine, arguments, gpus):
    inner = ["bash", "scripts/rgb_only_optimization_runtime.sh", machine, *arguments]
    if machine == "eval":
        return ["docker", "exec", "-w", str(ROOT), "-e", "CUDA_VISIBLE_DEVICES=" + gpus,
                "-e", "GLOG_minloglevel=2", "-e", "MAGNUM_LOG=quiet", "-e", "HABITAT_SIM_LOG=quiet",
                "gwl-etpr1-rae", *inner]
    return ["env", "CUDA_VISIBLE_DEVICES=" + gpus, *inner]


@contextlib.contextmanager
def resources(machine, gpus):
    locks = []
    try:
        for gpu in sorted(set(gpus.split(","))):
            lock_path = "/tmp/etpr1-eval-gpu.lock" if machine == "eval" else f"/tmp/etpr1-rgb-gpu-{gpu}.lock"
            lock = open(lock_path, "a")
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            locks.append(lock)
        if machine == "eval":
            print(output(["docker", "ps", "--format", "{{.Names}}|{{.Status}}"]), flush=True)
            assert output(["docker", "inspect", "-f", "{{.State.Running}}", "gwl-etpr1-rae"]) == "true"
            protected = subprocess.run(["docker", "exec", "gwl-etpnav", "ps", "-eo", "args"],
                                       capture_output=True, text=True)
            # A stopped protected container is harmless; other inspection errors are not.
            if protected.returncode and "is not running" not in protected.stderr:
                raise RuntimeError("Cannot inspect protected gwl-etpnav: " + protected.stderr)
            if any(re.search(r"(?:torchrun|run\.py|train\.py)", line)
                   for line in protected.stdout.splitlines()):
                raise RuntimeError("Protected ETPNav task is running; retry when resources are free")
        gpu_state = output(["nvidia-smi", "--id=" + gpus,
                            "--query-gpu=index,name,memory.used,utilization.gpu", "--format=csv,noheader,nounits"])
        print(gpu_state, flush=True)
        for line in gpu_state.splitlines():
            if int(line.split(",")[-2]) > 1024:
                raise RuntimeError("Selected GPU is occupied; worker exits without launching")
        yield
    finally:
        for lock in locks:
            lock.close()


def execute(args, name, overrides, checkpoint, mode, expected_iteration=None):
    root = ROOT / args.output
    job = root / ("train" if mode == "dagger" else "eval") / name
    manifest_path = job / "manifest.json"
    opts = dict(overrides)
    opts.update({"CHECKPOINT_FOLDER": str(job / "checkpoints") + "/",
                 "TENSORBOARD_DIR": str(job / "tensorboard") + "/",
                 "RESULTS_DIR": str(job / "results") + "/"})
    command = ["run.py", "--exp_name", name, "--run-type", mode, "--exp-config", CONFIG]
    for key, value in opts.items():
        command.extend([key, str(value)])
    if mode == "dagger":
        command = ["-m", "torch.distributed.run", "--nproc_per_node=" + str(len(args.gpus.split(","))),
                   "--master_port=" + str(args.port), *command]
    command = runtime(args.machine, command, args.gpus)
    if args.dry_run:
        print(shlex.join(command))
        return
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        def resume_neutral(cmd):
            cmd = list(cmd)
            if "IL.is_requeue" in cmd:
                cmd[cmd.index("IL.is_requeue") + 1] = "RESUME_FLAG"
            return cmd
        if resume_neutral(old["command"]) != resume_neutral(command):
            raise RuntimeError("Existing manifest has different command; choose a new --output")
        if old.get("status") == "completed":
            validate(job, name, mode, expected_iteration, args)
            print("skip_validated_completed=" + name, flush=True)
            return
        if mode == "dagger" and not args.resume:
            raise RuntimeError("Interrupted training exists; inspect and use --resume")
        save(job / ("manifest_previous_" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S") + ".json"), old)
    checkpoint = ROOT / checkpoint
    waiting_since = time.monotonic()
    while not checkpoint.is_file():
        if mode != "eval" or not args.wait_ready:
            raise FileNotFoundError(checkpoint)
        if time.monotonic() - waiting_since >= args.ready_timeout:
            raise TimeoutError("Timed out waiting for atomically published checkpoint: " + str(checkpoint))
        print(f"waiting_at={now()} checkpoint={checkpoint}", flush=True)
        time.sleep(30)
    digest = sha(checkpoint)
    if (str(checkpoint).endswith(BASE) or str(checkpoint).endswith(BASE_EVAL)) and digest != BASE_SHA:
        raise RuntimeError("Base checkpoint SHA256 mismatch")
    resume_source = None
    if mode == "dagger" and args.resume:
        model_dir = job / "checkpoints" / name
        pairs = [(int(re.search(r"iter(\d+)", p.name)[1]), p)
                 for p in model_dir.glob("ckpt.iter*.pth")
                 if (model_dir / "train_states" / ("train_state." + p.name.split(".", 1)[1])).is_file()]
        if not pairs:
            raise RuntimeError("No complete model/state pair available for resume")
        resume_iter, resume_model = max(pairs)
        resume_state = model_dir / "train_states" / f"train_state.iter{resume_iter}.pth"
        resume_source = dict(iteration=resume_iter, model=str(resume_model), state=str(resume_state),
                             model_sha256=sha(resume_model), state_bytes=resume_state.stat().st_size)
    with resources(args.machine, args.gpus):
        job.mkdir(parents=True, exist_ok=True)
        record = dict(status="running", started_at=now(), machine=socket.gethostname(),
                      cwd=str(ROOT), commit=output(["git", "rev-parse", "HEAD"]), command=command,
                      checkpoint=str(checkpoint), checkpoint_sha256=digest, overrides=opts,
                      resume_source=resume_source,
                      log_file=str(job / "run.log"))
        save(manifest_path, record)
        with (job / "run.log").open("a") as log:
            print(shlex.join(command), flush=True)
            log.write(json.dumps(record) + "\n")
            log.flush()
            versions = subprocess.run(runtime(args.machine, ["-c", VERSIONS], args.gpus),
                                      cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if versions.returncode:
                result = versions
            else:
                result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        record.update(exit_code=result.returncode, finished_at=now(), status="failed")
        if result.returncode == 0:
            try:
                record["validation"] = validate(job, name, mode, expected_iteration, args)
                record["status"] = "completed"
            except Exception as exc:
                record["validation_error"] = str(exc)
        save(manifest_path, record)
        if record["status"] != "completed":
            raise RuntimeError("Experiment failed: " + str(manifest_path))
        print("completed=" + name, flush=True)


def validate(job, name, mode, iteration, args):
    if mode == "dagger":
        model = job / "checkpoints" / name / f"ckpt.iter{iteration}.pth"
        state = model.parent / "train_states" / f"train_state.iter{iteration}.pth"
        if not model.is_file() or not state.is_file() or min(model.stat().st_size, state.stat().st_size) == 0:
            raise RuntimeError("Final checkpoint/training-state pair missing")
        return dict(iteration=iteration, model=str(model), model_bytes=model.stat().st_size,
                    state=str(state), model_sha256=sha(model))
    result_dir = job / "results" / name / "eval_results"
    summaries = list(result_dir.glob("stats_ckpt_*_val_unseen.json"))
    episodes = list(result_dir.glob("stats_ep_ckpt_*_val_unseen_r0_w1.json"))
    if len(summaries) != 1 or len(episodes) != 1:
        raise RuntimeError("Expected exactly one aggregate and one episode result")
    metrics, per_ep = json.loads(summaries[0].read_text()), json.loads(episodes[0].read_text())
    expected = 1839 if args.episodes == -1 else args.episodes
    if len(per_ep) < expected or (args.episodes == -1 and len(per_ep) != expected):
        raise RuntimeError(f"Episode count {len(per_ep)} != expected {expected}")
    for key in ("success", "spl"):
        if not math.isfinite(metrics[key]) or not 0 <= metrics[key] <= 1:
            raise RuntimeError("Invalid metric: " + key)
    return dict(episodes=len(per_ep), metrics=metrics, result=str(summaries[0]),
                episode_result=str(episodes[0]))


def common(gpus, environments):
    count = len(gpus.split(","))
    return {"TASK_CONFIG.SEED": 100, "SIMULATOR_GPU_IDS": str(list(range(count))),
            "TORCH_GPU_IDS": str(list(range(count))), "TORCH_GPU_ID": 0,
            "GPU_NUMBERS": count, "NUM_ENVIRONMENTS": environments,
            "MODEL.ACTIVE_LOOKAHEAD.enabled": False,
            "MODEL.RAENWM.condition_source_pose": "context_last",
            "MODEL.RGB_ENCODER.precision": "ambient",
            "TASK_CONFIG.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING": True}


def train(args):
    branch = args.branch
    name = f"rgbopt_{branch}_alpha1_{'frozen' if branch != 'A' else 'full'}_align{int(branch == 'C')}_context_last"
    opts = common(args.gpus, args.environments or args.batch)
    opts.update({"IL.iters": args.iters, "IL.log_every": args.log_every,
                 "IL.batch_size": args.batch, "IL.gradient_accumulation_steps": 1,
                 "IL.lr": 1e-5, "IL.sample_ratio": 0.75, "IL.decay_interval": 3000,
                 "IL.sample_ratio_iteration_offset": 14200, "IL.sample_ratio_zero_threshold": 0.0,
                 "IL.warmup_iters": 0, "IL.min_lr_ratio": 1.0, "IL.waypoint_aug": True,
                 "IL.amp_init_scale": 16384.0, "IL.use_fused_adamw": True,
                 "IL.cudnn_benchmark": True, "IL.log_cuda_memory": True,
                 "IL.resumable_checkpoints": True, "IL.keep_last_train_states": 3,
                 "IL.keep_train_state_every_n_iters": 2000,
                 "IL.checkpoint_sync_enabled": args.sync,
                 "IL.checkpoint_sync_destination": "a6000@10.10.10.2:/home/a6000/gwl/ETP-R1/" +
                     f"{args.output}/train/{name}/checkpoints/{name}",
                 "IL.load_from_ckpt": True, "IL.is_requeue": args.resume,
                 "IL.ckpt_to_load": BASE, "IL.freeze_navigation_backbone": branch != "A",
                 "MODEL.RAENWM.enabled": branch != "A", "MODEL.RAENWM.rgb_fusion_enabled": branch != "A",
                 "MODEL.RAENWM.context_source": "high_level_nav_latent" if branch == "A" else "low_level_move_rgb_anchor",
                 "MODEL.RAENWM.rgb_fusion_trainable": branch != "A",
                 "MODEL.RAENWM.rgb_fusion_align_navigation_cls": branch == "C",
                 "MODEL.RAENWM.rgb_fusion_alpha": 1.0, "ONLY_LAST_SAVEALL": True})
    # Command identity must remain stable across a resume; execute records the actual command.
    execute(args, name, opts, BASE, "dagger", args.iters)


def evaluate(args):
    if len(args.gpus.split(",")) != 1:
        raise ValueError("Evaluation uses one GPU for comparable episode traversal")
    for case in args.cases.split(","):
        if case not in ("base", "legacy", "a0", "a025", "a05", "a1", "A", "B", "C"):
            raise ValueError("Unknown case " + case)
        for iteration in (map(int, args.iterations.split(",")) if case in "ABC" and len(case) == 1 else [None]):
            source = "query_current" if case == "legacy" else "context_last"
            alpha = {"a0": 0.0, "a025": 0.25, "a05": 0.5}.get(case, 1.0)
            enabled = case not in ("base", "A")
            frozen, aligned = case in ("B", "C"), case == "C"
            name = f"rgbopt_{case}_alpha{alpha:g}_freeze{int(frozen)}_align{int(aligned)}_{source}"
            if iteration is not None:
                name += f"_iter{iteration}"
                train_name = f"rgbopt_{case}_alpha1_{'frozen' if frozen else 'full'}_align{int(aligned)}_context_last"
                checkpoint = f"{args.output}/train/{train_name}/checkpoints/{train_name}/ckpt.iter{iteration}.pth"
            else:
                checkpoint = BASE_EVAL if case == "base" else GRAD
                if case == "base" and not args.dry_run:
                    alias = ROOT / BASE_EVAL
                    if not alias.exists() and not alias.is_symlink():
                        alias.parent.mkdir(parents=True, exist_ok=True)
                        alias.symlink_to("base_iter14200.pth")
            checkpoint = args.checkpoint or checkpoint
            opts = common(args.gpus, args.environments or 8)
            opts.update({"EVAL.CKPT_PATH_DIR": str(ROOT / checkpoint), "EVAL.EPISODE_COUNT": args.episodes,
                         "EVAL.SAVE_RESULTS": True, "EVAL.USE_CKPT_CONFIG": False, "EVAL.fast_eval": False,
                         "IL.freeze_navigation_backbone": frozen, "MODEL.RAENWM.enabled": enabled,
                         "MODEL.RAENWM.context_source": "low_level_move_rgb_anchor" if enabled else "high_level_nav_latent",
                         "MODEL.RAENWM.rgb_fusion_enabled": enabled,
                         "MODEL.RAENWM.rgb_fusion_alpha": alpha,
                         "MODEL.RAENWM.rgb_fusion_align_navigation_cls": aligned,
                         "MODEL.RAENWM.condition_source_pose": source})
            execute(args, name, opts, checkpoint, "eval")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "eval"])
    parser.add_argument("--machine", choices=["server", "eval"], default="server")
    parser.add_argument("--output", default=DEFAULT_ROOT)
    parser.add_argument("--gpus", default=None)
    parser.add_argument("--environments", type=int)
    parser.add_argument("--branch", choices=list("ABC"), default="B")
    parser.add_argument("--batch", type=int, default=8, help="batch per rank; global batch = ranks * batch")
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument("--port", type=int, default=24747)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sync", action="store_true", help="atomically sync new model files to eval machine")
    parser.add_argument("--cases", default="base,legacy,a0,a025,a05,a1")
    parser.add_argument("--iterations", default="1000,2000")
    parser.add_argument("--checkpoint", help="override checkpoint for a single selected eval case")
    parser.add_argument("--episodes", type=int, default=-1, help="-1 full1839; positive is smoke only, not a paired subset")
    parser.add_argument("--wait-ready", action="store_true", help="wait for an atomically synced missing evaluation checkpoint")
    parser.add_argument("--ready-timeout", type=int, default=172800, help="maximum wait per missing checkpoint, seconds")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.gpus = args.gpus or ("0,1" if args.action == "train" else "0")
    if not re.fullmatch(r"\d+(,\d+)*", args.gpus) or len(set(args.gpus.split(","))) != len(args.gpus.split(",")):
        parser.error("--gpus must contain unique numeric GPU indices")
    if Path(args.output).is_absolute() or ".." in Path(args.output).parts:
        parser.error("--output must be a project-relative path")
    if args.action == "train" and args.machine != "server":
        parser.error("Training uses the server runtime")
    if args.checkpoint and ("," in args.cases or (args.cases in ("A", "B", "C") and "," in args.iterations)):
        parser.error("--checkpoint override requires a single evaluation case and iteration")
    if not args.dry_run:
        expected = "/home/gwl/project/etpr1/ETP-R1" if args.machine == "server" else "/home/a6000/gwl/ETP-R1"
        if str(ROOT) != expected:
            parser.error("Wrong machine/workspace: " + str(ROOT))
        print(f"host={socket.gethostname()} user={output(['whoami'])} cwd={ROOT}", flush=True)
    (train if args.action == "train" else evaluate)(args)


if __name__ == "__main__":
    main()
