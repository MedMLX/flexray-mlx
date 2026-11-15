"""Radiograph anatomy segmentation with FleXray in native MLX.

``run`` runs one radiograph end to end. Names load on first access, so
importing the package (or the medmlx declaration) does not import MLX.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flexray_mlx.medmlx import run as run
    from flexray_mlx.mlx_model import FleXrayMlxModel as FleXrayMlxModel

__all__ = ["FleXrayMlxModel", "run"]

__version__ = "0.2.7"

_EXPORTS = {"FleXrayMlxModel": "flexray_mlx.mlx_model", "run": "flexray_mlx.medmlx"}


def __getattr__(name: str) -> object:
    """Load a public name from its module on first access and cache it."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value
