"""Native MLX execution of the pinned FleXray v1.0.2 U-Net checkpoint.

The released network is fixed to seven encoder stages and six decoder stages.
Upstream FleXray uses PyTorch for preprocessing and test-time augmentation; this
port implements those steps separately and uses MLX for the network forward pass.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import yaml
from medmlx_core import import_mlx

from flexray_mlx._kernels import upsample_concat
from flexray_mlx._types import FloatArray

if TYPE_CHECKING:
    from mlx.core import array

    from flexray_mlx._types import NativeRuntime, NeuralRuntime, SafetensorsRuntime

# Namespaces stay lazy until this native implementation is admitted.
mx = cast("NativeRuntime", import_mlx())
nn = cast("NeuralRuntime", import_module("mlx.nn"))
safetensors = cast("SafetensorsRuntime", import_module("safetensors"))

_FILTERS = (64, 128, 256, 512, 512, 720, 1024)


class FleXrayMlxModel:
    """Released residual U-Net run natively in MLX.

    Activations are channels-last (NHWC) internally, as MLX convolutions expect.
    Convolution weights are stored upstream in PyTorch OIHW order and are
    transposed to OHWI when loaded. The public input and output stay NCHW.
    Tensor names and shapes are validated once at load. Each residual block is
    then a pure function compiled by MLX and evaluated before the next block.

    Attributes:
        weights: Every tensor in ``model.safetensors`` as float32, under its
            original key.
    """

    def __init__(self, bundle: Path) -> None:
        """Load one member bundle after checking it matches the released network.

        Args:
            bundle: Directory holding the released ``config.yml`` (its ``model``
                section) and ``model.safetensors``.

        Raises:
            ValueError: The config describes any other architecture or activation, or
                the checkpoint's tensor names or shapes differ from that network.
            OSError: ``config.yml`` or ``model.safetensors`` cannot be read.
                Malformed files raise the YAML or safetensors library's own errors.
        """
        config = cast(
            dict[str, dict[str, object]],
            yaml.safe_load((bundle / "config.yml").read_text(encoding="utf-8")),
        )["model"]
        expected: dict[str, object] = {
            "_class": "fxr.models.UNet",
            "in_channels": 1,
            "out_channels": 61,
            "filters": list(_FILTERS),
            "convs_per_block": 3,
            "dropout": 0.0,
            "norm": "instance",
            "residual": True,
            "norm_after_activation": True,
            "skip_connections": True,
            "upsample_align_corners": True,
            "residual_projection": "auto",
            "residual_shortcut_norm": True,
        }
        for key, value in expected.items():
            if config.get(key) != value:
                raise ValueError(f"Unsupported FleXray MLX architecture: {key}={config.get(key)!r}")
        if str(config.get("activation", "LeakyReLU")).casefold() != "leakyrelu":
            raise ValueError("Unsupported FleXray MLX activation")
        if config.get("bottleneck_convs") not in (None, 3) or config.get("up_filters") is not None:
            raise ValueError("Unsupported FleXray MLX bottleneck or decoder configuration")

        self.weights: dict[str, array] = {}
        with safetensors.safe_open(bundle / "model.safetensors", framework="np") as reader:
            for key in reader.keys():  # noqa: SIM118 - safe_open is not iterable
                value = np.asarray(reader.get_tensor(key), dtype=np.float32)
                # 4D weights are convolution kernels: OIHW -> OHWI. Norm weights are 1D.
                if key.endswith("weight") and value.ndim == 4:
                    value = value.transpose(0, 2, 3, 1)
                self.weights[key] = mx.array(value)
        # MLX is lazy; evaluate now so loading is complete before the first forward pass.
        mx.eval(*self.weights.values())
        # Check every tensor name and shape once, so the forward graph is pure and
        # never runs a different network. Missing, extra or misshaped tensors all
        # raise ValueError here, before any inference.
        shapes = _expected_shapes()
        if self.weights.keys() != shapes.keys():
            missing = sorted(shapes.keys() - self.weights.keys())
            extra = sorted(self.weights.keys() - shapes.keys())
            raise ValueError(
                f"FleXray MLX tensor names differ: missing {missing[:3]}, unused {extra[:3]}"
            )
        for key, shape in shapes.items():
            if tuple(self.weights[key].shape) != shape:
                raise ValueError(f"FleXray MLX tensor {key} has shape {self.weights[key].shape}")
        self._pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # One compiled pure function per residual block; mx.compile retraces when the
        # batch shape changes. Weights are captured as constants, which is valid
        # because they never change after loading.
        prefixes = [f"down_blocks.{index}" for index in range(len(_FILTERS))]
        prefixes += [f"up_blocks.{index}" for index in range(len(_FILTERS) - 1)]
        self._blocks = {prefix: _compiled_block(self.weights, prefix) for prefix in prefixes}

    def _run_block(self, x: array, prefix: str) -> array:
        """Evaluate one compiled block before building the next one.

        Evaluating block by block lets MLX release each block's convolution
        workspaces and intermediates before the next block allocates its own. The
        arithmetic and operation order are those of the whole-graph evaluation.
        """
        y = self._blocks[prefix](x)
        mx.eval(y)
        return y

    def forward_native(self, images: array) -> array:
        """Run the network on a float32 NCHW MLX batch; return NHWC logits.

        Residual blocks are compiled and evaluated in order; only the final 1x1
        output convolution is left lazy so callers can fuse their own epilogue.

        Args:
            images: ``(B, 1, 256, 256)`` float32 MLX array, prepared as for ``__call__``.

        Returns:
            Unevaluated ``(B, 256, 256, 61)`` float32 logits (channels last).

        Raises:
            ValueError: ``images`` has another shape or dtype.
        """
        if images.ndim != 4 or tuple(images.shape[1:]) != (1, 256, 256):
            raise ValueError(f"FleXray MLX expects Bx1x256x256, got {images.shape}")
        if images.dtype != mx.float32:
            raise ValueError("FleXray MLX expects float32 images")
        x = mx.transpose(images, (0, 2, 3, 1))
        skips: list[array] = []
        for index in range(len(_FILTERS)):
            x = self._run_block(x, f"down_blocks.{index}")
            if index < len(_FILTERS) - 1:
                skips.append(x)
                x = self._pool(x)
        for index, skip in enumerate(reversed(skips)):
            # Preserve corner alignment and resized-before-skip channel order,
            # while avoiding the separate bilinear samples and resize tensor.
            x = self._run_block(upsample_concat(x, skip, mx=mx), f"up_blocks.{index}")
        return _conv(self.weights, x, "output_conv", padding=0)

    def __call__(self, images: FloatArray) -> FloatArray:
        """Run the U-Net on prepared images and return raw logits.

        Args:
            images: Float32 ``(B, 1, 256, 256)`` NCHW radiographs, already scaled
                and normalized as the bundle's preprocessing specifies.

        Returns:
            A new float32 ``(B, 61, 256, 256)`` NCHW array of logits, one channel
            per label in the bundle's label-schema order (channel 0 is background).
            Apply a sigmoid for probabilities.

        Raises:
            ValueError: ``images`` has another shape.
        """
        if images.ndim != 4 or images.shape[1:] != (1, 256, 256):
            raise ValueError(f"FleXray MLX expects Bx1x256x256, got {images.shape}")
        logits = self.forward_native(mx.array(np.asarray(images, dtype=np.float32)))
        mx.eval(logits)
        return cast(FloatArray, np.asarray(logits)).transpose(0, 3, 1, 2).copy()


def _compiled_block(weights: Mapping[str, array], prefix: str) -> Callable[[array], array]:
    """Compile the pure residual block ``prefix`` as a one-array function.

    The closure holds only the weight mapping, not the model, so a released
    model and its compiled graphs are freed by reference counting.
    """

    def block(x: array) -> array:
        return _block(weights, x, prefix)

    return mx.compile(block)


def _conv(weights: Mapping[str, array], x: array, prefix: str, *, padding: int) -> array:
    """Apply a learned convolution and bias using the named prefix."""
    weight = weights[f"{prefix}.weight"]
    return mx.conv2d(x, weight, padding=padding) + weights[f"{prefix}.bias"]


def _norm(weights: Mapping[str, array], x: array, prefix: str) -> array:
    """Apply per-instance spatial normalization with learned scale and bias.

    Two-pass mean and centered variance, then ``(x - mean) * rsqrt(var + eps) *
    gamma + beta`` in that operation order. Under compilation MLX fuses the
    elementwise tail into one kernel; the reductions stay separate.
    """
    gamma = weights[f"{prefix}.weight"]
    beta = weights[f"{prefix}.bias"]
    # Instance norm on NHWC: per sample and channel over H and W. eps=1e-5 is
    # PyTorch's InstanceNorm2d default; the upstream parity test covers it.
    mean = mx.mean(x, axis=(1, 2), keepdims=True)
    variance = mx.mean(mx.square(x - mean), axis=(1, 2), keepdims=True)
    return (x - mean) * mx.rsqrt(variance + 1e-5) * gamma + beta


def _block(weights: Mapping[str, array], x: array, prefix: str) -> array:
    """Three conv, LeakyReLU, instance-norm steps plus a residual shortcut."""
    residual = x
    out = x
    for index in range(3):
        # Checkpoint key layout per step: conv at layers.3i, activation at
        # 3i+1 (no tensors), norm at 3i+2 (norm_after_activation).
        out = _conv(weights, out, f"{prefix}.layers.{index * 3}", padding=1)
        # LeakyReLU with PyTorch's default negative slope of 0.01.
        out = mx.maximum(out, out * 0.01)
        out = _norm(weights, out, f"{prefix}.layers.{index * 3 + 2}")
    # A 1x1 projection shortcut exists only where channel counts differ.
    if f"{prefix}.shortcut.conv.weight" in weights:
        residual = _conv(weights, residual, f"{prefix}.shortcut.conv", padding=0)
        residual = _norm(weights, residual, f"{prefix}.shortcut.instancenorm")
    return out + residual


def _expected_shapes() -> dict[str, tuple[int, ...]]:
    """Every released tensor name with its loaded shape (convolutions in OHWI)."""
    shapes: dict[str, tuple[int, ...]] = {}

    def conv(prefix: str, inputs: int, outputs: int, kernel: int) -> None:
        shapes[f"{prefix}.weight"] = (outputs, kernel, kernel, inputs)
        shapes[f"{prefix}.bias"] = (outputs,)

    def norm(prefix: str, channels: int) -> None:
        shapes[f"{prefix}.weight"] = shapes[f"{prefix}.bias"] = (channels,)

    def block(prefix: str, inputs: int, outputs: int) -> None:
        for index in range(3):
            conv(f"{prefix}.layers.{index * 3}", inputs if index == 0 else outputs, outputs, 3)
            norm(f"{prefix}.layers.{index * 3 + 2}", outputs)
        if inputs != outputs:
            conv(f"{prefix}.shortcut.conv", inputs, outputs, 1)
            norm(f"{prefix}.shortcut.instancenorm", outputs)

    channels = 1
    for index, outputs in enumerate(_FILTERS):
        block(f"down_blocks.{index}", channels, outputs)
        channels = outputs
    for index, outputs in enumerate(reversed(_FILTERS[:-1])):
        block(f"up_blocks.{index}", channels + outputs, outputs)
        channels = outputs
    conv("output_conv", channels, 61, 1)
    return shapes
