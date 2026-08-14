import hashlib
import re
from pathlib import Path


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path, chunk_size: int = 8 * 1024 * 1024) -> str:
    path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def validate_external_asset(path, expected_sha256: str, label: str) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {path}")
    expected = str(expected_sha256).strip().lower()
    if not _SHA256_RE.fullmatch(expected):
        raise ValueError(f"{label} requires a 64-character SHA256 checksum")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"{label} SHA256 mismatch: expected {expected}, got {actual}: {path}"
        )
    return path
