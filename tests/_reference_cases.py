"""Deterministic bundle inputs shared by fixture generation and equivalence tests."""

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, cast

import numpy as np
import safetensors.numpy as safetensors_numpy
import yaml
from _arrays import FloatArray, HostArray


class NumpySafetensors(Protocol):
    def save_file(self, tensor_dict: Mapping[str, HostArray], filename: str) -> None:
        """Write named NumPy tensors to a safetensors checkpoint."""
        ...


save_file = cast(NumpySafetensors, safetensors_numpy).save_file

FIXTURES = Path(__file__).parent / "fixtures" / "darwin-arm64"
UPSTREAM_REVISION = "bf6d01426fd3ae012aff9253ec9c9952c58bc960"


FILTERS = (64, 128, 256, 512, 512, 720, 1024)
MODEL_CONFIG: dict[str, object] = {
    "_class": "fxr.models.UNet",
    "in_channels": 1,
    "out_channels": 61,
    "filters": list(FILTERS),
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


def full_state_dict(seed: int) -> dict[str, FloatArray]:
    """Seed all released tensor shapes; share output kernels for lossless compression."""
    rng = np.random.default_rng(seed)
    weights: dict[str, FloatArray] = {}

    def conv(prefix: str, inputs: int, outputs: int, kernel: int) -> None:
        """Add deterministic convolution weights and bias to the fixture state."""
        values = rng.standard_normal((outputs, inputs, kernel, kernel), dtype=np.float32)
        values *= np.float32(0.2 / np.sqrt(inputs * kernel * kernel))
        weights[f"{prefix}.weight"] = values
        weights[f"{prefix}.bias"] = rng.normal(0, 0.05, outputs).astype(np.float32)

    def norm(prefix: str, channels: int) -> None:
        """Add deterministic normalization scale and bias to the fixture state."""
        weights[f"{prefix}.weight"] = rng.uniform(0.8, 1.2, channels).astype(np.float32)
        weights[f"{prefix}.bias"] = rng.normal(0, 0.05, channels).astype(np.float32)

    def block(prefix: str, inputs: int, outputs: int) -> None:
        """Add a three-layer residual block and any needed projection."""
        for index in range(3):
            conv(f"{prefix}.layers.{index * 3}", inputs if index == 0 else outputs, outputs, 3)
            norm(f"{prefix}.layers.{index * 3 + 2}", outputs)
        if inputs != outputs:
            conv(f"{prefix}.shortcut.conv", inputs, outputs, 1)
            norm(f"{prefix}.shortcut.instancenorm", outputs)

    channels = 1
    for index, outputs in enumerate(FILTERS):
        block(f"down_blocks.{index}", channels, outputs)
        channels = outputs
    for index, outputs in enumerate(reversed(FILTERS[:-1])):
        block(f"up_blocks.{index}", channels + outputs, outputs)
        channels = outputs
    conv("output_conv", channels, 1, 1)
    weights["output_conv.weight"] = np.repeat(weights["output_conv.weight"], 61, axis=0)
    weights["output_conv.bias"] = np.repeat(weights["output_conv.bias"], 61)
    return weights


def seeded_images(seed: int) -> FloatArray:
    """Create the seeded full-size input for the upstream network fixture."""
    return np.random.default_rng(seed).standard_normal((1, 1, 256, 256), dtype=np.float32)


def write_bundle(
    bundle: Path,
    state_dict: Mapping[str, HostArray] | None = None,
    *,
    weight_seed: int = 1729,
    config: dict[str, object] | None = None,
) -> None:
    """Write a test configuration and safetensors bundle."""
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "config.yml").write_text(
        yaml.safe_dump({"model": MODEL_CONFIG if config is None else config}), encoding="utf-8"
    )
    save_file(
        full_state_dict(weight_seed) if state_dict is None else state_dict,
        str(bundle / "model.safetensors"),
    )
