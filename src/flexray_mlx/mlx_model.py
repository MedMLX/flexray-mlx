"""Native MLX execution of the pinned FleXray v1.0.2 U-Net checkpoint.

The released network is fixed to seven encoder stages and six decoder stages.
Upstream FleXray uses PyTorch for preprocessing and test-time augmentation; this
port implements those steps separately and uses MLX for the network forward pass.
"""

from __future__ import annotations

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
            ValueError: The config describes any other architecture or activation.
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
        self._consumed: set[str] = set()
        self._pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def _weight(self, key: str) -> array:
        """Record and return a named checkpoint tensor."""
        self._consumed.add(key)
        return self.weights[key]

    def _conv(self, x: array, prefix: str, *, padding: int) -> array:
        """Apply a learned convolution and bias using the named prefix."""
        weight = self._weight(f"{prefix}.weight")
        bias = self._weight(f"{prefix}.bias")
        return mx.conv2d(x, weight, padding=padding) + bias

    def _norm(self, x: array, prefix: str) -> array:
        """Apply per-instance spatial normalization with learned scale and bias."""
        gamma = self._weight(f"{prefix}.weight")
        beta = self._weight(f"{prefix}.bias")
        # Instance norm on NHWC: per sample and channel over H and W. eps=1e-5 is
        # PyTorch's InstanceNorm2d default; the upstream parity test covers it.
        mean = mx.mean(x, axis=(1, 2), keepdims=True)
        variance = mx.mean(mx.square(x - mean), axis=(1, 2), keepdims=True)
        return (x - mean) * mx.rsqrt(variance + 1e-5) * gamma + beta

    def _block(self, x: array, prefix: str) -> array:
        """Three conv, LeakyReLU, instance-norm steps plus a residual shortcut."""
        residual = x
        out = x
        for index in range(3):
            # Checkpoint key layout per step: conv at layers.3i, activation at
            # 3i+1 (no tensors), norm at 3i+2 (norm_after_activation).
            layer = f"{prefix}.layers.{index * 3}"
            out = self._conv(out, layer, padding=1)
            # LeakyReLU with PyTorch's default negative slope of 0.01.
            out = mx.maximum(out, out * 0.01)
            out = self._norm(out, f"{prefix}.layers.{index * 3 + 2}")
        # A 1x1 projection shortcut exists only where channel counts differ.
        if f"{prefix}.shortcut.conv.weight" in self.weights:
            residual = self._conv(residual, f"{prefix}.shortcut.conv", padding=0)
            residual = self._norm(residual, f"{prefix}.shortcut.instancenorm")
        return out + residual

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
            ValueError: ``images`` has another shape, or the bundle holds tensors
                this network does not use. A tensor the network needs but the
                bundle lacks raises ``KeyError``.
        """
        if images.ndim != 4 or images.shape[1:] != (1, 256, 256):
            raise ValueError(f"FleXray MLX expects Bx1x256x256, got {images.shape}")
        x = mx.array(np.asarray(images, dtype=np.float32).transpose(0, 2, 3, 1))
        skips: list[array] = []
        for index in range(len(_FILTERS)):
            x = self._block(x, f"down_blocks.{index}")
            if index < len(_FILTERS) - 1:
                skips.append(x)
                x = self._pool(x)
        for index, skip in enumerate(reversed(skips)):
            # Preserve corner alignment and resized-before-skip channel order,
            # while avoiding the separate bilinear samples and resize tensor.
            x = self._block(upsample_concat(x, skip, mx=mx), f"up_blocks.{index}")
        logits = self._conv(x, "output_conv", padding=0)
        mx.eval(logits)
        # Fail on extra or misnamed tensors instead of silently running a different net.
        if self._consumed != self.weights.keys():
            missing = sorted(self.weights.keys() - self._consumed)
            raise ValueError(f"FleXray MLX did not consume every released tensor: {missing[:3]}")
        return cast(FloatArray, np.asarray(logits)).transpose(0, 3, 1, 2).copy()
