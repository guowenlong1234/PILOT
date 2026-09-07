#!/usr/bin/env python3
"""Publish completed RGB-only training evidence from the training machine.

Runs CPU checkpoint audits in separate project-runtime processes. Transfers
only small JSON evidence to the evaluation machine, never model weights.
--once exits 2 while training remains pending, 0 when complete, 1 on failure.
--wait polls every 30 seconds for at most 48 hours.
"""
import argparse
import datetime
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = Path("/home/gwl/project/etpr1/ETP-R1")
EVAL_ROOT = Path("/home/a6000/gwl/ETP-R1")
DEFAULT_ROOT = "data/logs/rgb_only_optimization_20260907"
BASE = "pretrained/active_lookahead/base_iter14200.pth"
NAMES = {
    "A": "rgbopt_A_alpha1_full_align0_context_last",
    "B": "rgbopt_B_alpha1_frozen_align0_context_last",
    "C": "rgbopt_C_alpha1_frozen_align1_context_last",
}
REMOTE_WRITE = r'''
import hashlib, json, os, pathlib, socket, sys, tempfile
root = pathlib.Path('/home/a6000/gwl/ETP-R1')
destination = pathlib.Path(sys.argv[1])
if os.getuid() != 1000 or not root.is_dir():
    raise RuntimeError('Unexpected evaluation container uid/project')
destination.resolve().relative_to(root.resolve())
payload = sys.stdin.buffer.read(2 * 1024 * 1024 + 1)
if len(payload) > 2 * 1024 * 1024:
    raise RuntimeError('Refusing oversized evidence JSON')
json.loads(payload)
destination.parent.mkdir(parents=True, exist_ok=True)
fd, temporary = tempfile.mkstemp(prefix=destination.name + '.tmp.', dir=destination.parent)
try:
    with os.fdopen(fd, 'wb') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
print(json.dumps(dict(hostname=socket.gethostname(), uid=os.getuid(),
                     path=str(destination), sha256=hashlib.sha256(payload).hexdigest())))
'''


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def time_left(deadline, limit):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Completion publisher reached its wait limit")
    return min(limit, remaining)


def publish(relative_root, name, payload, deadline):
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    destination = EVAL_ROOT / relative_root / "audit/train_completion" / name
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
               "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2",
               "a6000@10.10.10.2", shlex.join([
                   "docker", "exec", "-i", "-w", str(EVAL_ROOT), "gwl-etpr1-rae",
                   "/home/a6000/gwl/miniconda3/envs/etpr1_rae/bin/python",
                   "-c", REMOTE_WRITE, str(destination)])]
    result = subprocess.run(command, input=data, capture_output=True,
                            timeout=time_left(deadline, 60), cwd=ROOT)
    if result.returncode:
        raise RuntimeError("Evidence publish failed: " + result.stderr.decode(errors="replace")[-2000:])
    receipt = json.loads(result.stdout)
    if receipt.get("sha256") != hashlib.sha256(data).hexdigest():
        raise RuntimeError("Published JSON checksum mismatch")
    return receipt


def queue_identity(pid_file):
    if not pid_file.is_file():
        raise RuntimeError("Training is incomplete and queue PID file is missing: " + str(pid_file))
    raw = pid_file.read_text().strip()
    if not raw.isdigit() or int(raw) <= 1:
        raise RuntimeError("Invalid training queue PID: " + repr(raw))
    proc = Path("/proc") / raw
    try:
        # comm may contain spaces/parentheses; the last ')' ends field 2.
        fields = (proc / "stat").read_text().rsplit(")", 1)[1].split()
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except FileNotFoundError:
        raise RuntimeError("Training is incomplete but queue PID disappeared: " + raw)
    recognized = ("rgb_only_optimization.py" in command and "train" in command) or (
        "manage_rgb_only_pipeline.py" in command and "worker" in command and pid_file.stem in command
    )
    if fields[0] in ("Z", "X") or not recognized:
        raise RuntimeError("Training queue PID is dead or no longer a recognized trainer: " + raw)
    return {"pid": int(raw), "start_ticks": fields[19], "command": command}


def verify_completed(root, branch, manifest):
    name = NAMES[branch]
    model_dir = root / "train" / name / "checkpoints" / name
    model = model_dir / "ckpt.iter2000.pth"
    state = model_dir / "train_states/train_state.iter2000.pth"
    validation = manifest.get("validation", {})
    if manifest.get("exit_code") != 0 or validation.get("iteration") != 2000:
        raise RuntimeError(f"{branch}: completed manifest lacks successful iteration-2000 validation")
    if Path(validation.get("model", "")).resolve() != model.resolve():
        raise RuntimeError(f"{branch}: manifest model path does not match this experiment")
    if Path(validation.get("state", "")).resolve() != state.resolve():
        raise RuntimeError(f"{branch}: manifest training-state path does not match this experiment")
    for path in (model, state):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"{branch}: missing or empty completed model/training-state pair: {path}")
    if model.stat().st_size != validation.get("model_bytes") or sha(model) != validation.get("model_sha256"):
        raise RuntimeError(f"{branch}: final model size/hash differs from completed manifest")
    return model_dir


def run_audit(evidence, branch, iteration, model_dir, deadline):
    report_path = evidence / f"{branch}_{iteration}_frozen.json"
    model = model_dir / f"ckpt.iter{iteration}.pth"
    command = ["bash", "scripts/rgb_only_optimization_runtime.sh", "server",
               "scripts/audit_rgb_frozen_checkpoint.py", "--base", BASE,
               "--checkpoint", str(model), "--expected-align", "true" if branch == "C" else "false",
               "--expected-source", "context_last", "--output", str(report_path)]
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    with (evidence / f"{branch}_{iteration}_frozen.log").open("a") as log:
        log.write(now() + " " + shlex.join(command) + "\n")
        log.flush()
        result = subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                                stderr=subprocess.STDOUT, timeout=time_left(deadline, 1800))
    if result.returncode:
        raise RuntimeError(f"{branch}/{iteration}: frozen audit exited {result.returncode}; see {report_path.with_suffix('.log')}")
    report = json.loads(report_path.read_text())
    if report.get("ok") is not True or report.get("iteration") != iteration:
        raise RuntimeError(f"{branch}/{iteration}: frozen audit did not validate expected iteration")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--wait", action="store_true")
    parser.add_argument("--root", default=DEFAULT_ROOT, help="Project-relative experiment root")
    parser.add_argument("--timeout", type=float, default=172800, help="Wait limit in seconds, at most 48 hours")
    args = parser.parse_args()
    if ROOT != TRAIN_ROOT:
        parser.error(f"This publisher must run from {TRAIN_ROOT}; actual ROOT={ROOT}")
    relative_root = Path(args.root)
    if relative_root.is_absolute() or ".." in relative_root.parts:
        parser.error("--root must be a project-relative path without '..'")
    if not 0 < args.timeout <= 172800:
        parser.error("--timeout must be positive and at most 172800 seconds")
    os.chdir(ROOT)
    root = ROOT / relative_root
    evidence = root / "audit/train_completion"
    evidence.mkdir(parents=True, exist_ok=True)
    lock = (evidence / "publisher.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("Another completion publisher holds this experiment lock")
    started = time.monotonic()
    hard_deadline = started + args.timeout
    # Reserve a small part of the limit for publishing a timeout/failure
    # status, so the evaluation finalizer never waits on a stale "waiting".
    deadline = hard_deadline - min(60, args.timeout * 0.1)
    status = dict(status="waiting", started_at=now(), updated_at=now(), hostname=socket.gethostname(),
                  user=getpass.getuser(), pid=os.getpid(), root=str(root), branches={}, receipts={}, errors=[])
    status_path = evidence / "publisher_status.json"
    published, identities = set(), {}
    try:
        while True:
            time_left(deadline, 1)
            for branch, name in NAMES.items():
                if branch in published:
                    continue
                manifest_path = root / "train" / name / "manifest.json"
                manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
                stage = manifest.get("status", "not_started")
                status["branches"][branch] = stage
                if stage == "failed":
                    raise RuntimeError(f"{branch}: training manifest reports failed: {manifest_path}")
                if stage == "completed":
                    model_dir = verify_completed(root, branch, manifest)
                    if branch in ("B", "C"):
                        for iteration in (1000, 2000):
                            report = run_audit(evidence, branch, iteration, model_dir, deadline)
                            filename = f"{branch}_{iteration}_frozen.json"
                            status["receipts"][filename] = publish(relative_root, filename, report, deadline)
                    save(evidence / f"{branch}.json", manifest)
                    status["receipts"][f"{branch}.json"] = publish(relative_root, f"{branch}.json", manifest, deadline)
                    published.add(branch)
                    status["branches"][branch] = "completed_and_published"
            # Check queue liveness after refreshing ALL manifests: a queue can
            # exit normally when its final completion manifest is published.
            rescan = False
            for queue, branches in (("train_gpu0_queue.pid", {"A", "C"}), ("train_gpu1_queue.pid", {"B"})):
                if branches <= published:
                    continue
                try:
                    identity = queue_identity(root / queue)
                except RuntimeError:
                    # Normal completion can race the last manifest read.
                    latest = []
                    for branch in branches - published:
                        path = root / "train" / NAMES[branch] / "manifest.json"
                        latest.append(path.is_file() and json.loads(path.read_text()).get("status") == "completed")
                    if all(latest):
                        rescan = True
                        continue
                    raise
                signature = (identity["pid"], identity["start_ticks"])
                if queue in identities and identities[queue] != signature:
                    raise RuntimeError("Incomplete training queue PID identity changed: " + queue)
                identities[queue] = signature
            if rescan:
                continue
            status.update(updated_at=now(), elapsed_seconds=round(time.monotonic() - started, 2))
            if len(published) == 3:
                status["status"] = "completed"
            save(status_path, status)
            publish(relative_root, "publisher_status.json", status, deadline)
            if status["status"] == "completed":
                print(json.dumps(status, indent=2, sort_keys=True))
                return 0
            if args.once:
                print(json.dumps(status, indent=2, sort_keys=True))
                return 2
            if time.monotonic() - started >= args.timeout:
                status["status"] = "timed_out"
                raise TimeoutError("Timed out waiting for completed training evidence")
            print(json.dumps({"status": "waiting", "updated_at": now(), "branches": status["branches"]}), flush=True)
            time.sleep(min(30, max(0.0, deadline - time.monotonic())))
    except Exception as exc:
        status["status"] = "timed_out" if time.monotonic() >= deadline else "failed"
        status.update(updated_at=now(), finished_at=now())
        status["errors"].append(f"{type(exc).__name__}: {exc}")
        save(status_path, status)
        try:
            publish(relative_root, "publisher_status.json", status, hard_deadline)
        except Exception as publish_error:
            status["errors"].append("Failure-status publication failed: " + str(publish_error))
            save(status_path, status)
        print(json.dumps(status, indent=2, sort_keys=True))
        return 1
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())
