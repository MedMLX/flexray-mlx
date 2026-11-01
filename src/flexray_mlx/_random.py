"""Torch CPU draw order for the pinned FleXray float32 augmentation chain.

MT19937 uses its low 24 bits for float uniforms. Contiguous normal tensors use
16-value Box-Muller blocks; short tensors use cached double-precision pairs.
Transcendental rounding can differ between CPU libraries, as it does in Torch.
Adapted from PyTorch's ATen/core and native/cpu distribution helpers (BSD;
see root THIRD_PARTY_NOTICES.md).
"""

from __future__ import annotations

import math

import numpy as np

from flexray_mlx._types import FloatArray, UIntArray


class TorchRandom:
    """An isolated stream that reproduces Torch CPU draws, without importing Torch.

    Draws follow Torch's order of consumption for a given seed and call sequence;
    the module docstring notes where float rounding can still differ. Values are
    float32.
    """

    def __init__(self, seed: int | None) -> None:
        """Seed from the low 32 bits of ``seed`` (negative values wrap); None is unseeded."""

        self._engine = np.random.RandomState(None if seed is None else seed & 0xFFFFFFFF)
        self._next_normal: float | None = None

    def _words(self, count: int) -> UIntArray:
        """Draw raw 32-bit words from the isolated MT19937 stream."""
        return self._engine.randint(0, 2**32, size=count, dtype=np.uint32)

    def _units(self, count: int) -> FloatArray:
        """Convert raw words to Torch-compatible float32 unit uniforms."""
        return (self._words(count) & np.uint32(0xFFFFFF)).astype(np.float32) * np.float32(2**-24)

    def uniform(self, low: float = 0.0, high: float = 1.0) -> float:
        """Draw one float32 uniform between ``low`` and ``high``, like Torch's ``uniform_``."""
        # Deferred: flexray_mlx.medmlx imports this module (via tta) at load time.
        from flexray_mlx.medmlx import fma32

        lower, upper = np.float32(low), np.float32(high)
        return float(fma32(self._units(1)[0], upper - lower, lower))

    def chance(self, probability: float) -> np.bool_:
        """Consume one uniform draw and return whether it is below ``probability``."""
        return self.uniform() < np.float32(probability)

    def _double_unit(self) -> float:
        """Combine two words into a 53-bit unit uniform."""
        words = self._words(2).astype(np.uint64)
        value = (int(words[0]) << 32) | int(words[1])
        return (value & (2**53 - 1)) * 2**-53

    def _short_normal(self) -> float:
        """One double-precision Box-Muller normal; the pair's sine half is cached."""
        if self._next_normal is not None:
            value, self._next_normal = self._next_normal, None
            return value
        theta = 2.0 * math.pi * self._double_unit()
        radius = math.sqrt(-2.0 * math.log1p(-self._double_unit()))
        self._next_normal = radius * math.sin(theta)
        return radius * math.cos(theta)

    @staticmethod
    def _normal_blocks(uniforms: FloatArray) -> FloatArray:
        """Convert groups of 16 uniforms to Torch-compatible normal draws."""
        blocks = uniforms.reshape(-1, 16)
        radius = np.sqrt(np.float32(-2.0) * np.log(np.float32(1.0) - blocks[:, :8]))
        # The scalar CPU kernel multiplies by double pi before rounding theta.
        theta = (2.0 * math.pi * blocks[:, 8:].astype(np.float64)).astype(np.float32)
        return np.concatenate((radius * np.cos(theta), radius * np.sin(theta)), axis=1).ravel()

    def normal(self, shape: tuple[int, ...]) -> FloatArray:
        """Draw a float32 standard-normal array of ``shape``.

        Fewer than 16 values use cached double-precision Box-Muller pairs; 16 or
        more use float32 blocks of 16, as Torch's CPU kernel does.
        """
        count = math.prod(shape)
        if count < 16:
            return np.asarray([self._short_normal() for _ in range(count)], np.float32).reshape(
                shape
            )
        uniforms = self._units(count)
        complete = count // 16 * 16
        values = np.empty(count, dtype=np.float32)
        values[:complete] = self._normal_blocks(uniforms[:complete])
        if count % 16:
            # Torch redraws the final 16, including the overlapping completed block.
            values[-16:] = self._normal_blocks(self._units(16))
        return values.reshape(shape)
