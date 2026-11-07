"""Native MLX implementation of FleXray."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from flexray_mlx.mlx_model import FleXrayMlxModel

__all__ = ["FleXrayMlxModel"]

__version__ = "0.2.7"


def __getattr__(name: str) -> object:
    # MLX loads with the network, so importing the medmlx declaration stays light.
    """Load the native model lazily when its public name is requested."""
    if name == "FleXrayMlxModel":
        from flexray_mlx.mlx_model import FleXrayMlxModel

        return FleXrayMlxModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
