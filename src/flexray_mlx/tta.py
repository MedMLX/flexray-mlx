"""NumPy port of pinned fxr ``tta_v3`` (FleXray bf6d014, MIT).

Each view is drawn once, normalized again, shared by all members, and unflipped
on the logits. Accumulation is sequential in view/member order, in float32
probability space; ``mean_probabilities`` lets callers that evaluate members
one at a time reduce stored terms in that same order. The plain view is
already normalized and stays untouched.
See root THIRD_PARTY_NOTICES.md for source notices.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import numpy as np

from flexray_mlx._random import TorchRandom
from flexray_mlx._types import BoolArray, FloatArray, IntArray

if TYPE_CHECKING:
    from mlx.core import array

    from flexray_mlx._types import NativeRuntime

Forward = Callable[[FloatArray], FloatArray]


class NativeForward(Protocol):
    """A member that can also run on MLX arrays (``FleXrayMlxModel``)."""

    def __call__(self, images: FloatArray) -> FloatArray:
        """Return NCHW logits for NCHW NumPy images."""
        ...

    def forward_native(self, images: array) -> array:
        """Return NHWC logits for an NCHW float32 MLX batch."""
        ...


@dataclass(frozen=True, slots=True)
class TtaSettings:
    """Validated pass count and isolated Torch CPU seed."""

    samples: int = 1
    seed: int | None = None

    def __post_init__(self) -> None:
        """Reject invalid augmentation counts and Torch-compatible seed values."""
        if (
            isinstance(self.samples, bool)
            or not isinstance(cast(object, self.samples), int)
            or self.samples < 0
        ):
            raise ValueError("tta_samples must be a non-negative integer")
        if self.seed is not None and (
            isinstance(self.seed, bool)
            or not isinstance(cast(object, self.seed), int)
            or not -(2**63) <= self.seed < 2**64
        ):
            raise ValueError("tta_seed must be an integer in [-2**63, 2**64 - 1], or None")

    @property
    def resolved_samples(self) -> int:
        """Passes per member actually run; ``samples`` of 0 resolves to one plain pass."""
        return max(self.samples, 1)


def predict_with_tta(
    forward_fn: Forward | Sequence[Forward],
    images: FloatArray,
    *,
    tta_samples: int = 1,
    seed: int | None = None,
    normalizer: Forward | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Return multilabel ``(logits, probabilities)``, following fxr's total-pass contract.

    The first pass is the plain input; each further pass is an augmented view
    shared by all members. Probabilities are the mean of every member's sigmoid
    over every pass. For a single plain pass of one member the logits are that
    member's raw output; otherwise they are the logit of the mean probability
    clipped to [1e-6, 1 - 1e-6]. Use :func:`tta_probabilities` when only the
    probabilities are needed.

    Args:
        forward_fn: One callable, or a sequence of member callables, mapping
            float32 ``(B, C, H, W)`` images to float32 ``(B, L, H, W)`` logits.
        images: Prepared float32 ``(B, C, H, W)`` batch with 1 or 3 channels,
            already normalized.
        tta_samples: Non-negative integer total passes per member; 0 or 1 means
            one plain pass.
        seed: Integer in ``[-2**63, 2**64 - 1]`` for the isolated Torch-CPU-compatible
            stream; None is unseeded.
        normalizer: Applied to each augmented view (not the plain input) so views
            are normalized as the plain input was.

    Raises:
        ValueError: Invalid ``tta_samples``/``seed``, no forward callable, or
            ``images`` that is not float32 ``(B, C, H, W)`` with 1 or 3 channels.
    """

    # Deferred: flexray_mlx.medmlx imports this module at load time.
    from flexray_mlx.medmlx import multilabel_probabilities

    settings, forwards = _admit(forward_fn, images, tta_samples, seed)
    if settings.resolved_samples == 1 and len(forwards) == 1:
        logits = forwards[0](images)
        return logits, multilabel_probabilities(logits)
    probabilities = tta_probabilities(
        forwards, images, tta_samples=tta_samples, seed=seed, normalizer=normalizer
    )
    clamped = np.clip(probabilities, np.float32(1e-6), np.float32(1 - 1e-6))
    return np.log(clamped) - np.log1p(-clamped), probabilities


def tta_probabilities(
    forward_fn: Forward | Sequence[Forward],
    images: FloatArray,
    *,
    tta_samples: int = 1,
    seed: int | None = None,
    normalizer: Forward | None = None,
) -> FloatArray:
    """Return only the mean probabilities of :func:`predict_with_tta`, bit for bit.

    Terms are summed in float32 in pass-major order (every member on the plain
    input, then every member on each augmented view) and divided once by the
    term count. Members with a ``forward_native`` method (``FleXrayMlxModel``)
    compute each term's unflip and sigmoid on Metal with a kernel whose float32
    bits equal the strict CPU emulation; other callables use the CPU emulation.
    Arguments and errors are those of :func:`predict_with_tta`.
    """

    settings, forwards = _admit(forward_fn, images, tta_samples, seed)
    terms = (
        view_probabilities(forward, view, flipped)
        for view, flipped in prepared_views(images, settings, normalizer)
        for forward in forwards
    )
    return mean_probabilities(terms, settings.resolved_samples * len(forwards))


def _admit(
    forward_fn: Forward | Sequence[Forward], images: FloatArray, samples: int, seed: int | None
) -> tuple[TtaSettings, tuple[Forward, ...]]:
    """Validate TTA settings, member callables and the image batch."""
    settings = TtaSettings(samples, seed)
    forwards = (forward_fn,) if callable(forward_fn) else tuple(forward_fn)
    if not forwards:
        raise ValueError("predict_with_tta requires at least one forward callable")
    if images.ndim != 4 or images.shape[1] not in {1, 3} or images.dtype != np.float32:
        raise ValueError("FleXray TTA expects float32 BxCxHxW images with 1 or 3 channels")
    return settings, forwards


def prepared_views(
    images: FloatArray, settings: TtaSettings, normalizer: Forward | None
) -> Iterator[tuple[FloatArray, BoolArray | None]]:
    """Yield ``(view, flipped)`` per pass: the plain input (``None``), then augmented views.

    Views are drawn in fxr's order from one isolated stream. The draws do not
    depend on network outputs, so callers may materialize the views once and
    reuse them for members evaluated in any order.
    """

    yield images, None
    random = TorchRandom(settings.seed)
    for _ in range(settings.resolved_samples - 1):
        augmented, flipped = augment_batch(images, random)
        if normalizer is not None:
            augmented = normalizer(augmented)
        yield augmented, flipped


def view_probabilities(forward: Forward, view: FloatArray, flipped: BoolArray | None) -> FloatArray:
    """One member's unflipped sigmoid probabilities for one view, as a new float32 array.

    ``flipped`` marks samples mirrored along the width axis (``None`` for the
    plain pass). Native members run ``forward_native`` and the Torch-CPU-exact
    Metal sigmoid, copying only probabilities to the host.
    """

    if hasattr(forward, "forward_native"):
        from medmlx_core import import_mlx

        from flexray_mlx._kernels import torch_cpu_sigmoid

        mx = cast("NativeRuntime", import_mlx())
        native = cast(NativeForward, forward)
        flags = np.zeros(len(view), np.int32) if flipped is None else flipped.astype(np.int32)
        logits = native.forward_native(mx.array(np.asarray(view, dtype=np.float32)))
        probabilities = torch_cpu_sigmoid(logits, mx.array(flags), mx=mx)
        mx.eval(probabilities)
        return np.array(probabilities, dtype=np.float32)
    # Deferred: flexray_mlx.medmlx imports this module at load time.
    from flexray_mlx.medmlx import multilabel_probabilities

    logits = forward(view)
    if flipped is not None:
        logits = np.where(flipped[:, None, None, None], logits[..., ::-1], logits)
    return multilabel_probabilities(logits)


def mean_probabilities(terms: Iterable[FloatArray], count: int) -> FloatArray:
    """Sum float32 terms in the given order, then divide once by ``count``.

    The first term becomes the accumulator and is modified in place.
    """

    iterator = iter(terms)
    total = next(iterator)
    for term in iterator:
        total += term
    return total / np.float32(count)


def augment_batch(images: FloatArray, random: TorchRandom) -> tuple[FloatArray, BoolArray]:
    """Draw the fxr chain independently per sample, in batch order.

    Returns the stacked ``(B, C, H, W)`` views and a ``(B,)`` boolean mask of
    which samples were flipped along the width axis.
    """

    views, flips = zip(*(augment_sample(image, random) for image in images), strict=True)
    return np.stack(views), np.asarray(flips, dtype=np.bool_)


def augment_sample(image: FloatArray, random: TorchRandom) -> tuple[FloatArray, np.bool_]:
    """Horizontal flip, invert, CLAHE/gamma, raw contrast, sharpness, then noise.

    ``image`` is one prepared ``(C, H, W)`` float32 sample. Random draws are
    consumed in fxr's order. The CLAHE, gamma, contrast and sharpness steps clip
    to [0, 1]; invert and noise do not. Fractional gamma is applied before its
    clip, so negative values can produce NaN; clipping does not remove that NaN.
    Returns the view and whether it was flipped (so logits can be unflipped).
    """

    flipped = random.chance(0.5)
    if flipped:
        image = image[..., ::-1]
    if random.chance(0.5):
        image = np.float32(1.0) - image
    choice = random.uniform()
    if choice < 0.1:
        image = clahe(image, clip_limit=random.uniform(1.0, 2.0), grid_size=(8, 8))
    elif choice < 0.35:
        gamma, gain = random.uniform(0.9, 1.1), random.uniform(0.9, 1.1)
        image = np.clip(np.float32(gain) * np.power(image, np.float32(gamma)), 0, 1)
    if random.chance(0.25):
        image = cast(FloatArray, np.clip(image * np.float32(random.uniform(0.7, 1.3)), 0, 1))
    if random.chance(0.5):
        image = adjust_sharpness(image, random.uniform(0.7, 1.3))
    if random.chance(0.25):
        image = cast(FloatArray, image + random.normal(image.shape) * np.float32(0.01))
    return image, flipped


def adjust_sharpness(image: FloatArray, factor: float) -> FloatArray:
    """Sharpen ``(C, H, W)`` ``image`` as ``factor * image + (1 - factor) * blurred``.

    The blur is torchvision's 3x3 ``[1,1,1; 1,5,1; 1,1,1] / 13`` kernel, and the
    one-pixel border keeps the original values. The blend is clipped to [0, 1].
    Images with a side of 2 pixels or fewer are returned unchanged.
    """

    if min(image.shape[-2:]) <= 2:
        return image
    # Deferred: flexray_mlx.medmlx imports this module at load time.
    from flexray_mlx.medmlx import fma32

    height, width = image.shape[-2:]
    blurred = image.copy()
    interior = np.zeros_like(image[:, 1:-1, 1:-1])
    for y in range(3):
        for x in range(3):
            weight = np.float32(5.0 if (y, x) == (1, 1) else 1.0) / np.float32(13.0)
            interior = fma32(image[:, y : y + height - 2, x : x + width - 2], weight, interior)
    blurred[:, 1:-1, 1:-1] = interior
    return cast(
        FloatArray, np.clip(np.float32(factor) * image + np.float32(1.0 - factor) * blurred, 0, 1)
    )


def _interpolation_axis(
    length: int, tile_size: int, tile_count: int
) -> tuple[IntArray, IntArray, FloatArray]:
    """Per pixel: nearest lower/upper tile index and the weight of the upper tile.

    Tile centers sit at ``(k + 0.5) * tile_size``; indices clamp at the edges.
    """

    position = np.arange(length, dtype=np.float32) / np.float32(tile_size) - np.float32(0.5)
    lower = np.floor(position).astype(np.int64)
    return (
        np.clip(lower, 0, tile_count - 1),
        np.clip(lower + 1, 0, tile_count - 1),
        (position - lower.astype(np.float32)),
    )


def clahe(image: FloatArray, *, clip_limit: float, grid_size: tuple[int, int]) -> FloatArray:
    """Contrast-limited adaptive histogram equalization of a ``(C, H, W)`` image.

    Follows fxr: values are clipped to [0, 1] and quantized to 256 bins per
    channel, histograms are built per tile and clipped at
    ``clip_limit * pixels / 256`` with the excess redistributed, and each pixel
    interpolates bilinearly between the four nearest tile centers. ``grid_size``
    is ``(rows, cols)`` of tiles, capped at the image size; tiles are rounded up
    to even sizes and the image is padded to fit (reflect, or edge when the
    padding reaches the image size in either direction). Supply two integer tile
    counts; the function does not check their types. Returns float32 ``(C, H, W)`` in [0, 1].

    Raises:
        ValueError: ``image`` is not a non-empty 3D array, ``clip_limit`` is not
            positive and finite, or ``grid_size`` has the wrong length, a boolean
            value, or a nonpositive value.
    """

    if image.ndim != 3 or min(image.shape) < 1:
        raise ValueError(f"CLAHE expects a non-empty CxHxW image, got {image.shape}")
    if not math.isfinite(clip_limit) or clip_limit <= 0:
        raise ValueError("CLAHE clip_limit must be positive and finite")
    if len(grid_size) != 2 or any(isinstance(n, bool) or n <= 0 for n in grid_size):
        raise ValueError("CLAHE grid_size must contain two positive integers")
    channels, height, width = image.shape
    rows, cols = min(grid_size[0], height), min(grid_size[1], width)
    th, tw = math.ceil(height / rows), math.ceil(width / cols)
    th, tw = th + th % 2, tw + tw % 2
    ph, pw = th * rows - height, tw * cols - width
    work = np.clip(image, 0, 1).astype(np.float32)
    if ph or pw:
        mode = "reflect" if ph < height and pw < width else "edge"
        work = np.pad(work, ((0, 0), (0, ph), (0, pw)), mode=mode)
    tiles = work.reshape(channels, rows, th, cols, tw).transpose(1, 3, 0, 2, 4)
    pixels = th * tw
    bins = (tiles.reshape(-1, pixels) * np.float32(255)).astype(np.int64)
    offsets = np.arange(len(bins), dtype=np.int64)[:, None] * 256
    histograms = np.bincount((bins + offsets).ravel(), minlength=len(bins) * 256).reshape(-1, 256)
    histograms = np.minimum(histograms, max(math.floor(clip_limit * pixels / 256), 1))
    clipped = pixels - histograms.sum(axis=1)
    residual = clipped % 256
    histograms += ((clipped - residual) // 256)[:, None]
    histograms += np.arange(256)[None] < residual[:, None]
    lookup = np.floor(np.cumsum(histograms, axis=1).astype(np.float32) * np.float32(255 / pixels))
    lookup = np.clip(lookup, 0, 255).reshape(rows, cols, channels, 256).transpose(2, 0, 1, 3)
    pixel_bins = (work * np.float32(255)).astype(np.int64)
    y0, y1, wy = _interpolation_axis(work.shape[1], th, rows)
    x0, x1, wx = _interpolation_axis(work.shape[2], tw, cols)
    channel = cast(IntArray, np.arange(channels))[:, None, None]
    v00 = lookup[channel, y0[None, :, None], x0[None, None], pixel_bins]
    v01 = lookup[channel, y0[None, :, None], x1[None, None], pixel_bins]
    v10 = lookup[channel, y1[None, :, None], x0[None, None], pixel_bins]
    v11 = lookup[channel, y1[None, :, None], x1[None, None], pixel_bins]
    wy, wx = wy[None, :, None], wx[None, None]
    top = v00 * (np.float32(1) - wx) + v01 * wx
    bottom = v10 * (np.float32(1) - wx) + v11 * wx
    return cast(
        FloatArray,
        ((top * (np.float32(1) - wy) + bottom * wy) / np.float32(255))[:, :height, :width],
    )
