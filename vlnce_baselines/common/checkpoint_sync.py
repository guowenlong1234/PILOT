import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


def _remote_destination(destination):
    destination = str(destination)
    if ":" not in destination:
        return None
    host, directory = destination.split(":", 1)
    if not host or not directory.startswith("/"):
        return None
    if "/" in host or "\\" in host:
        return None
    directory = directory.rstrip("/")
    if not directory or any(character in directory for character in "\r\n"):
        raise ValueError("Remote checkpoint destination must be a non-root path")
    return host, directory


def _same_size(source, destination):
    try:
        return source.stat().st_size == destination.stat().st_size
    except FileNotFoundError:
        return False


def sync_checkpoint(source, destination):
    """Publish one complete checkpoint atomically to local or SSH storage."""
    source = Path(source)
    if not source.is_file() or source.stat().st_size <= 0:
        raise ValueError(f"Checkpoint source is missing or empty: {source}")
    destination = str(destination).strip()
    if not destination:
        raise ValueError("Checkpoint sync destination must not be empty")

    remote = _remote_destination(destination)
    if remote is None:
        destination_dir = Path(destination).expanduser()
        destination_path = destination_dir / source.name
        if _same_size(source, destination_path):
            return destination_path
        incoming_dir = destination_dir / ".incoming"
        incoming_dir.mkdir(parents=True, exist_ok=True)
        incoming = incoming_dir / f"{source.name}.part.{os.getpid()}"
        try:
            shutil.copy2(source, incoming)
            if not _same_size(source, incoming):
                raise OSError(
                    f"Checkpoint sync size mismatch: {source} -> {incoming}"
                )
            destination_dir.mkdir(parents=True, exist_ok=True)
            os.replace(incoming, destination_path)
        finally:
            if incoming.exists():
                incoming.unlink()
        return destination_path

    host, destination_dir = remote
    destination_path = f"{destination_dir}/{source.name}"
    incoming = (
        f"{destination_dir}/.incoming/{source.name}.part.{os.getpid()}"
    )
    subprocess.run(
        [
            "ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            host,
            f"mkdir -p -- {shlex.quote(destination_dir + '/.incoming')}",
        ],
        check=True,
    )
    remote_size = subprocess.run(
        [
            "ssh", "-n", "-o", "BatchMode=yes", host,
            f"stat -c %s -- {shlex.quote(destination_path)}",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if (
        remote_size.returncode == 0
        and remote_size.stdout.strip() == str(source.stat().st_size)
    ):
        return f"{host}:{destination_path}"
    subprocess.run(
        [
            "rsync", "-a", "--partial", "--protect-args",
            str(source), f"{host}:{incoming}",
        ],
        check=True,
        stdin=subprocess.DEVNULL,
    )
    published_size = subprocess.run(
        [
            "ssh", "-n", "-o", "BatchMode=yes", host,
            f"stat -c %s -- {shlex.quote(incoming)}",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if published_size != str(source.stat().st_size):
        raise OSError(
            "Remote checkpoint sync size mismatch: "
            f"local={source.stat().st_size}, remote={published_size}"
        )
    subprocess.run(
        [
            "ssh", "-n", "-o", "BatchMode=yes", host,
            f"mv -f -- {shlex.quote(incoming)} {shlex.quote(destination_path)}",
        ],
        check=True,
    )
    return f"{host}:{destination_path}"


def launch_checkpoint_sync(source, destination, log_path=None):
    """Launch one detached transfer so checkpoint saving never waits on I/O."""
    source = Path(source)
    destination = str(destination).strip()
    if not destination:
        raise ValueError("Checkpoint sync destination must not be empty")
    if log_path is None:
        log_path = source.parent / "checkpoint_sync.log"
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("ab", buffering=0)
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--source",
                str(source),
                "--destination",
                destination,
            ],
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log_handle.close()
    return process.pid


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    args = parser.parse_args(argv)
    try:
        published = sync_checkpoint(args.source, args.destination)
    except Exception as error:
        print(
            f"checkpoint_sync_failed source={args.source} "
            f"destination={args.destination} error={type(error).__name__}: {error}",
            flush=True,
        )
        return 1
    print(
        f"checkpoint_sync_finished source={args.source} published={published}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
