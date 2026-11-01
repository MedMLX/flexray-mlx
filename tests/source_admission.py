"""Admit local oracle source bytes against existing pinned Git objects, without fetching."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path


def require_git_sources(
    root: Path,
    revision: str,
    files: Mapping[str, str],
    repository: Path | None = None,
) -> None:
    """Compare consumed files to pinned blobs before executing upstream code.

    Archive/raw-file inputs need a separate local repository containing the pin.
    This proves source content only; checkpoint provenance is independent.

    Raises:
        RuntimeError: Git proof is unavailable, a blob is absent, or bytes differ.
    """
    proof = repository if repository is not None else root
    for relative, git_path in files.items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"Missing upstream source file: {path}")
        try:
            result = subprocess.run(
                [
                    "git",
                    "--no-lazy-fetch",
                    "--no-replace-objects",
                    "-C",
                    str(proof),
                    "show",
                    f"{revision}:{git_path}",
                ],
                capture_output=True,
                check=False,
            )
        except FileNotFoundError as error:
            raise RuntimeError("Source admission requires an installed Git executable") from error
        if result.returncode:
            detail = result.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"Cannot prove {path} at {revision}:{git_path}. Supply --source-repository "
                f"with the pinned Git objects (no automatic fetch). Git: {detail}"
            )
        if path.read_bytes() != result.stdout:
            raise RuntimeError(
                f"Upstream source differs from pinned blob {revision}:{git_path}: {path}"
            )
