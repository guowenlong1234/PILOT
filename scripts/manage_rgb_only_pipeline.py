#!/usr/bin/env python3
"""Start/adopt the finite RGB-only pipeline on either project's remote HOST.

Run `python3 scripts/manage_rgb_only_pipeline.py start` on each host after Git
synchronization. Existing verified workers are adopted, never duplicated.
`status` is a one-shot check. Computation failures remain visible in manifests;
the manager never kills training or evaluation processes to reclaim GPUs.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/logs/rgb_only_optimization_20260907"
EVAL_PYTHON = "/home/a6000/gwl/miniconda3/envs/etpr1_rae/bin/python"
OLD_BASE = ("data/logs/rae_dinov2_etpnav_cls_768/r2r_sft_eval_best465000_20260816/"
            "eval_watch_val_unseen/results/rae_dinov2_etpnav_cls_768_eval_best465000_r2r_sft_eval_watch/"
            "eval_results/stats_ep_ckpt_14200_val_unseen_r0_w1.json")
READ_PROCESS = """
import json,pathlib,sys
p=pathlib.Path('/proc')/sys.argv[1]
try:
    fields=(p/'stat').read_text().rsplit(')',1)[1].split()
    args=(p/'cmdline').read_bytes().split(b'\\0')
    print(json.dumps(dict(alive=fields[0] not in ('Z','X'),
                         argv=[a.decode(errors='replace') for a in args if a])))
except FileNotFoundError:
    print(json.dumps(dict(alive=False,argv=[])))
"""


def machine():
    if ROOT == Path("/home/gwl/project/etpr1/ETP-R1"):
        return "server"
    if ROOT == Path("/home/a6000/gwl/ETP-R1"):
        return "eval"
    raise RuntimeError("Run this manager in the approved remote project workspace")


def process(role):
    path = OUTPUT / (role + ".pid")
    if not path.exists():
        return {"alive": False, "argv": []}
    pid = int(path.read_text().strip())
    if pid <= 1:
        raise RuntimeError("Invalid worker PID file: " + str(path))
    command = [sys.executable, "-c", READ_PROCESS, str(pid)]
    if role == "finalizer":
        command = ["docker", "exec", "gwl-etpr1-rae", EVAL_PYTHON,
                   "-c", READ_PROCESS, str(pid)]
    value = json.loads(subprocess.check_output(command, text=True))
    value["pid"] = pid
    if value["alive"]:
        joined = " ".join(value["argv"])
        expected = {"completion_publisher": "publish_rgb_only_completion.py",
                    "finalizer": "finalize_rgb_only_optimization.py"}.get(role, "rgb_only_optimization.py")
        managed = "manage_rgb_only_pipeline.py" in joined and "worker" in value["argv"] and role in value["argv"]
        if expected not in joined and not managed:
            raise RuntimeError("PID belongs to an unexpected process; refusing to launch: " + str(path))
    return value


def train_name(branch):
    return "rgbopt_{}_alpha1_{}_align{}_context_last".format(
        branch, "full" if branch == "A" else "frozen", int(branch == "C"))


def completed(role):
    if role.startswith("train_gpu"):
        branches = "AC" if role == "train_gpu0_queue" else "B"
        paths = [OUTPUT / "train" / train_name(b) / "manifest.json" for b in branches]
    elif role == "completion_publisher":
        paths = [OUTPUT / "audit/train_completion/publisher_status.json"]
    elif role == "finalizer":
        paths = [OUTPUT / "final_summary.json"]
    else:
        from finalize_rgb_only_optimization import cases
        paths = [OUTPUT / "eval" / item["name"] / "manifest.json" for item in cases()]
    if not all(p.is_file() and json.loads(p.read_text()).get("status") == "completed" for p in paths):
        return False
    return role != "finalizer" or json.loads(paths[0].read_text()).get("analysis_status") == "deferred_until_user_request"


def worker(role):
    if role.startswith("train_gpu"):
        gpu = "0" if role == "train_gpu0_queue" else "1"
        for branch in ("AC" if gpu == "0" else "B"):
            cmd = [sys.executable, "scripts/rgb_only_optimization.py", "train", "--branch", branch,
                   "--gpus", gpu, "--batch", "16", "--port", "24747" if gpu == "0" else "24748", "--sync"]
            manifest = OUTPUT / "train" / train_name(branch) / "manifest.json"
            if manifest.exists() and json.loads(manifest.read_text()).get("status") != "completed":
                cmd.append("--resume")
            subprocess.run(cmd, check=True)
    elif role == "eval_queue":
        subprocess.run([sys.executable, "scripts/rgb_only_optimization.py", "eval", "--machine", "eval",
                        "--cases", "base,legacy,a0,a025,a05,a1,A,B,C", "--wait-ready"], check=True)
    elif role == "completion_publisher":
        subprocess.run([sys.executable, "scripts/publish_rgb_only_completion.py", "--wait"], check=True)
    else:
        raise ValueError("Unsupported host worker " + role)


def start(role, previous):
    if role == "finalizer":
        if previous["alive"] and "--collect-only" in previous["argv"]:
            return dict(role=role, action="adopted", pid=previous["pid"])
        # Only this verified CPU result collector is replaced to honor the
        # user's request to defer analysis; GPU workers are never stopped.
        code = """
import json,os,pathlib,signal,subprocess,sys,time
root=pathlib.Path(sys.argv[1]); previous=int(sys.argv[2]); output=root/'data/logs/rgb_only_optimization_20260907'
if previous:
    p=pathlib.Path('/proc')/str(previous)
    command=(p/'cmdline').read_bytes()
    if b'finalize_rgb_only_optimization.py' not in command: raise RuntimeError('Unexpected collector PID')
    os.kill(previous,signal.SIGTERM)
    for _ in range(100):
        if not p.exists() or (p/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z': break
        time.sleep(0.1)
    else: raise RuntimeError('Collector did not exit')
cmd=['bash','scripts/rgb_only_optimization_runtime.sh','eval','scripts/finalize_rgb_only_optimization.py',
     '--wait','--collect-only','--historical-baseline',sys.argv[3]]
with (output/'finalizer.log').open('a') as log:
    child=subprocess.Popen(cmd,cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
(output/'finalizer.pid').write_text(str(child.pid)+'\\n')
print(json.dumps(dict(pid=child.pid)))
"""
        result = subprocess.check_output(["docker", "exec", "gwl-etpr1-rae", EVAL_PYTHON,
                                          "-c", code, str(ROOT), str(previous.get("pid", 0) if previous["alive"] else 0), OLD_BASE], text=True)
        return dict(role=role, action="collect_only_started", **json.loads(result))
    if previous["alive"]:
        return dict(role=role, action="adopted", pid=previous["pid"])
    with (OUTPUT / (role + ".log")).open("a") as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "worker", "--role", role],
                                 cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    (OUTPUT / (role + ".pid")).write_text(str(child.pid) + "\n")
    return dict(role=role, action="started", pid=child.pid)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "status", "worker"))
    parser.add_argument("--role", choices=("train_gpu0_queue", "train_gpu1_queue", "eval_queue", "completion_publisher"))
    args = parser.parse_args()
    host = machine()
    os.chdir(ROOT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    roles = ("train_gpu0_queue", "train_gpu1_queue", "completion_publisher") if host == "server" else ("eval_queue", "finalizer")
    if args.action == "worker":
        if args.role not in roles or args.role == "finalizer":
            parser.error("worker requires a role belonging to this host")
        worker(args.role)
        return
    manager_lock = None
    if args.action == "start":
        manager_lock = (OUTPUT / "pipeline_manager.lock").open("a")
        fcntl.flock(manager_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for role in roles:
        state = process(role)
        if args.action == "status":
            print(json.dumps(dict(role=role, completed=completed(role), **state)))
        elif completed(role):
            print(json.dumps(dict(role=role, action="already_completed")))
        else:
            print(json.dumps(start(role, state)))


if __name__ == "__main__":
    main()
