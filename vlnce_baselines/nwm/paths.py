from pathlib import Path
from typing import Union


PathLike = Union[str, Path]

REPO_ROOT = Path(__file__).resolve().parents[2]

NWM_CONFIG_DIR = REPO_ROOT / "configs" / "nwm"

DEFAULT_NWM_OUTPUT_DIR = REPO_ROOT / "ETPNav_data" / "nwm"


def resolve_repo_path(path: PathLike) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate    #穿进来的是绝对路径，就返回
    return REPO_ROOT / candidate    #传进来的是相对路径就展开成用户目录
