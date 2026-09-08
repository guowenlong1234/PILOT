#!/usr/bin/env python3
"""Finite background joint SFT / ascending checkpoint evaluation on each host."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = "data/logs/ghost_concat_joint_bs8_20260908"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "resume", "status"), default="status", nargs="?")
    args = parser.parse_args()
    if str(ROOT) == "/home/gwl/project/etpr1/ETP-R1":
        role = "train"
        command = [sys.executable, "scripts/ghost_concat_job.py", "train", "--train-policy",
                   "--gpus", "0,1", "--batch", "4", "--iters", "2000", "--log-every", "200",
                   "--policy-lr", "2e-6", "--fusion-lr", "1e-5", "--sync", "--output", OUTPUT]
        manifest = ROOT / OUTPUT / "train/ghost_concat_v1_joint_train/manifest.json"
    elif str(ROOT) == "/home/a6000/gwl/ETP-R1":
        role = "eval"
        command = [sys.executable, "scripts/ghost_concat_job.py", "watch", "--machine", "eval",
                   "--train-policy", "--gpus", "0", "--environments", "8", "--output", OUTPUT,
                   "--eval-iterations", "200,400,600,800,1000,1200,1400,1600,1800,2000"]
        manifest = ROOT / OUTPUT / "eval_summary.json"
    else:
        parser.error("Use the approved training or evaluation project workspace")
    root = ROOT / OUTPUT
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / (role + "_manager.lock")).open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pid_path = root / (role + ".pid")
    alive = False
    pid = None
    if pid_path.exists():
        pid = int(pid_path.read_text())
        proc = Path("/proc") / str(pid)
        if proc.exists():
            state = (proc / "stat").read_text().rsplit(")", 1)[1].split()[0]
            argv = (proc / "cmdline").read_bytes().decode(errors="replace")
            if state not in ("Z", "X"):
                if "ghost_concat_job.py" not in argv or OUTPUT not in argv:
                    raise RuntimeError("PID belongs to another process; refusing to start")
                alive = True
    saved = json.loads(manifest.read_text()) if manifest.exists() else {}
    if args.action == "status" or alive or saved.get("status") == "completed":
        print(json.dumps(dict(role=role, alive=alive, pid=pid, status=saved.get("status", "not_started"),
                              manifest=str(manifest))))
        return
    if role == "train" and saved:
        if args.action != "resume":
            raise RuntimeError("An interrupted training exists; use resume after inspecting its error")
        command.append("--resume")
    elif role == "train" and args.action == "resume":
        raise RuntimeError("There is no training to resume")
    with (root / (role + "_supervisor.log")).open("a") as log:
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    tmp = pid_path.with_suffix(".tmp")
    tmp.write_text(str(child.pid) + "\n"); tmp.replace(pid_path)
    print(json.dumps(dict(role=role, pid=child.pid, command=command, log=str(root/(role+"_supervisor.log")))))


if __name__ == "__main__":
    main()
