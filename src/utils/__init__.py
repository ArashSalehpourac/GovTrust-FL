"""Small shared utilities."""

from .io_utils import save_json
from .seed import set_seed

__all__ = ["bootstrap_project", "save_json", "set_seed"]


def bootstrap_project(seed: int = 42) -> None:
    """Create standard directories and initialize repeatability controls."""

    from src.config import ensure_project_dirs

    ensure_project_dirs()
    set_seed(seed)
