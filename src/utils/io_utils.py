"""Input/output helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(payload: dict[str, Any], path: str | Path) -> Path:
    """Write a JSON artifact with stable formatting."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output_path
