"""FP32 Metal fusions for the released FleXray decoder's NHWC tensors."""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, cast

from medmlx_core.typing import MetalKernel

from flexray_mlx._types import NativeRuntime

if TYPE_CHECKING:
    from mlx.core import array


_RESIZE_CONCAT = """
// Preserve the original graph's separate multiply/add rounding.
#pragma clang fp contract(off)
uint i = thread_position_in_grid.x;
constexpr int C = CI + CS;
if (i >= N * OH * OW * C) return;
uint c = i % C;
uint ox = (i / C) % OW;
uint oy = (i / (C * OW)) % OH;
uint n = i / (C * OW * OH);
if (c >= CI) {
    out[i] = skip[((n * OH + oy) * OW + ox) * CS + c - CI];
    return;
}
// align_corners=True: match the FP32 source-to-destination endpoint ratio.
float iy = float(oy) * (float(H - 1) / float(OH - 1));
float ix = float(ox) * (float(W - 1) / float(OW - 1));
iy = metal::clamp(iy, 0.0f, float(H - 1));
ix = metal::clamp(ix, 0.0f, float(W - 1));
uint y0 = uint(metal::floor(iy)), y1 = uint(metal::ceil(iy));
uint x0 = uint(metal::floor(ix)), x1 = uint(metal::ceil(ix));
float fy = iy - float(y0), fx = ix - float(x0);
float a = inp[((n * H + y0) * W + x0) * CI + c];
float b = inp[((n * H + y0) * W + x1) * CI + c];
float d = inp[((n * H + y1) * W + x0) * CI + c];
float e = inp[((n * H + y1) * W + x1) * CI + c];
out[i] = ((1.0f - fy) * (1.0f - fx)) * a
       + ((1.0f - fy) * fx) * b
       + (fy * (1.0f - fx)) * d
       + (fy * fx) * e;
"""


@lru_cache(maxsize=1)
def _resize_concat_kernel(mx: NativeRuntime) -> MetalKernel:
    """Compile and cache the Metal kernel for corner-aligned resize and concatenation."""
    return cast(
        MetalKernel,
        mx.fast.metal_kernel(
            name="flexray_bilinear_concat",
            input_names=["inp", "skip"],
            output_names=["out"],
            source=_RESIZE_CONCAT,
            compile_options={"math_mode": "safe"},
        ),
    )


def upsample_concat(x: array, skip: array, *, mx: NativeRuntime) -> array:
    """Join a 2x bilinear NHWC resize with ``skip``, keeping resized channels first.

    Inputs must be FP32, with the same positive batch size and a skip grid
    exactly twice ``x`` in both spatial dimensions. Corner-aligned sampling
    matches the released decoder, including singleton source dimensions.
    One kernel writes the joined tensor without materializing the four bilinear
    sample tensors or the separate resized image. Noncontiguous inputs are
    made contiguous by MLX's custom-kernel boundary.

    Raises ``ValueError`` when the output exceeds signed 32-bit indexing, before
    dispatch. The output is larger than either input, so this also bounds their
    index products.
    """
    if x.ndim != 4 or skip.ndim != 4:
        raise ValueError("FleXray decoder fusion requires NHWC tensors")
    n, h, w, ci = x.shape
    if min(n, h, w, ci, skip.shape[-1]) < 1 or skip.shape[:3] != (n, 2 * h, 2 * w):
        raise ValueError("FleXray MLX decoder grid mismatch")
    if x.dtype != mx.float32 or skip.dtype != mx.float32:
        raise TypeError("FleXray decoder fusion requires float32 inputs")
    oh, ow, cs = skip.shape[1:]
    shape = (n, oh, ow, ci + cs)
    count = n * oh * ow * (ci + cs)
    if count > 2**31 - 1:
        raise ValueError("FleXray decoder fusion exceeds signed 32-bit indexing")
    return _resize_concat_kernel(mx)(
        inputs=[x, skip],
        template=[
            ("N", n), ("H", h), ("W", w), ("CI", ci), ("CS", cs), ("OH", oh), ("OW", ow),
        ],
        grid=(count, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[shape],
        output_dtypes=[mx.float32],
    )[0]
