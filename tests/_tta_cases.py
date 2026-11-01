"""Seeded inputs and small asymmetric convolution weights for fxr TTA parity."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from _arrays import FloatArray

WEIGHT_SEEDS = (1729, 1730, 1731, 1732, 1733)
INPUT_SEED = 2718
TTA_CASES = (
    ("zero", 0, 0, 1, "none"),
    ("plain", 0, 1, 1, "none"),
    ("ensemble_plain", 7, 1, 3, "none"),
    ("all_branches", 0, 17, 2, "percentile_minmax"),
    ("other_seed", 7, 5, 1, "minmax"),
    ("standardize", 0, 2, 2, "standardize"),
    ("standardize_gamma", 7, 5, 1, "standardize"),
)


def seeded_input(shape: tuple[int, ...] = (2, 1, 24, 40)) -> FloatArray:
    """Create the reproducible float32 image batch used in TTA comparisons."""
    return np.random.default_rng(INPUT_SEED).uniform(0, 1, shape).astype(np.float32)


def standardize_inputs() -> dict[str, FloatArray]:
    """Build varied input distributions for source normalization comparisons."""
    rng = np.random.default_rng(17)
    ordinary = rng.normal(size=(16, 1, 13, 17)).astype(np.float32)
    return {
        "ordinary": ordinary,
        "offset": (ordinary + np.float32(1e5)),
        "constant": np.full((2, 1, 3, 5), 7, dtype=np.float32),
        "near_constant": np.float32(7) + ordinary * np.float32(1e-5),
        "singleton": np.full((2, 1, 1, 1), 7, dtype=np.float32),
        "model_grid": seeded_input((1, 1, 256, 256)),
        "multichannel": seeded_input((3, 3, 13, 17)),
    }


def seeded_weights(seed: int) -> tuple[FloatArray, FloatArray]:
    """Create reproducible convolution weights and bias for a fixture member."""
    rng = np.random.default_rng(seed)
    return rng.normal(0, 0.4, (1, 1, 3, 3)).astype(np.float32), rng.normal(0, 0.1, (1,)).astype(
        np.float32
    )


@dataclass(frozen=True)
class SeededForward:
    """Small asymmetric convolution; identical 61 output kernels compress losslessly."""

    seed: int

    def __call__(self, images: FloatArray) -> FloatArray:
        """Apply the seeded reference convolution and repeat it across label channels."""
        kernel, bias = seeded_weights(self.seed)
        height, width = images.shape[-2:]
        padded = np.pad(images, ((0, 0), (0, 0), (1, 1), (1, 1)))
        output = np.zeros_like(images)
        for y in range(3):
            for x in range(3):
                output += padded[:, :, y : y + height, x : x + width] * kernel[0, 0, y, x]
        return np.repeat(output + bias[None, :, None, None], 61, axis=1)
