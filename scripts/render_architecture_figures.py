"""Render manuscript architecture DOT figures with Graphviz."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCH_ROOT = PROJECT_ROOT / "paper_figures" / "architecture"
DOT_DIR = ARCH_ROOT / "dot"
OUTPUT_DIRS = {
    "svg": ARCH_ROOT / "svg",
    "pdf": ARCH_ROOT / "pdf",
    "png": ARCH_ROOT / "png",
}


def main() -> int:
    dot_binary = find_dot_binary()
    if dot_binary is None:
        message = (
            "Graphviz 'dot' executable was not found. Install Graphviz and ensure "
            "'dot' is on PATH, or install it in the standard Graphviz location."
        )
        raise SystemExit(message)

    if not DOT_DIR.exists():
        raise SystemExit(f"DOT source directory not found: {DOT_DIR}")

    dot_files = sorted(DOT_DIR.glob("*.dot"))
    if not dot_files:
        raise SystemExit(f"No .dot files found in: {DOT_DIR}")

    for output_dir in OUTPUT_DIRS.values():
        output_dir.mkdir(parents=True, exist_ok=True)

    for dot_file in dot_files:
        for ext, output_dir in OUTPUT_DIRS.items():
            output_path = output_dir / f"{dot_file.stem}.{ext}"
            render(dot_binary, dot_file, output_path, ext)
            print(output_path)

    return 0


def find_dot_binary() -> str | None:
    """Find Graphviz dot on PATH or in common Windows install locations."""

    path_dot = shutil.which("dot")
    if path_dot:
        return path_dot

    candidates = [
        Path("C:/Program Files/Graphviz/bin/dot.exe"),
        Path("C:/Program Files (x86)/Graphviz/bin/dot.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def render(dot_binary: str, input_path: Path, output_path: Path, output_format: str) -> None:
    command = [
        dot_binary,
        f"-T{output_format}",
        str(input_path),
        "-o",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"Graphviz failed for {input_path.name} as {output_format}: {detail}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
