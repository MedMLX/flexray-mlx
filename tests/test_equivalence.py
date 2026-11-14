"""Upstream UNet parity."""

import os
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from _reference_cases import FIXTURES, UPSTREAM_REVISION, full_state_dict, write_bundle
from _runner_cases import seeded_member_logits, write_case, write_release_bundle
from medmlx_core.errors import AssetNotReadyError

pytestmark = pytest.mark.mlx
LOGIT_BOUND = 4.35e-4


def test_full_model_matches_upstream_unet(tmp_path: Path) -> None:
    """Compare seeded full-size Metal logits with the recorded source U-Net."""
    import json

    from flexray_mlx import FleXrayMlxModel

    with np.load(FIXTURES / "flexray_full.npz", allow_pickle=False) as fixture:
        metadata = json.loads(str(fixture["metadata"]))
        assert metadata["flexray_revision"] == UPSTREAM_REVISION
        write_bundle(tmp_path, weight_seed=int(fixture["weight_seed"]))
    default = Path(__file__).resolve().parents[1] / "work/real-reference"
    directory = Path(os.environ.get("MEDMLX_TEST_REFERENCES", str(default)))
    with np.load(directory / "flexray_full.npz", allow_pickle=False) as fixture:
        assert json.loads(str(fixture["real_metadata"]))["flexray_revision"] == UPSTREAM_REVISION
        actual = FleXrayMlxModel(tmp_path)(fixture["images"])
        expected = fixture["output_nhwc"].transpose(0, 3, 1, 2)
        assert actual.shape == (1, 61, 256, 256)
        assert actual.dtype == np.float32
        # Observed Metal/Torch CPU max abs 4.3442845e-4 on these seeded inputs.
        # 4.35e-4 bounds convolution/reduction rounding on this seeded case.
        np.testing.assert_allclose(actual, expected, rtol=0, atol=LOGIT_BOUND)


@pytest.mark.parametrize("problem", ["missing", "shape"])
def test_checkpoint_mismatch_is_rejected_at_load(tmp_path: Path, problem: str) -> None:
    """Reject a missing or misshaped released tensor before any inference."""
    from flexray_mlx import medmlx

    state = full_state_dict(1729)
    if problem == "missing":
        del state["up_blocks.2.layers.5.bias"]
    else:
        state["down_blocks.3.layers.3.weight"] = state["down_blocks.3.layers.3.weight"][:, :-1]
    root = write_release_bundle(tmp_path / "weights", state_dict=state)
    with pytest.raises(AssetNotReadyError, match="do not match the released network"):
        medmlx.segment(
            image_path=write_case(tmp_path / "inputs", "l8"),
            weights_path=root,
            output_dir=tmp_path / "out",
        )
    assert not (tmp_path / "out").exists()


def test_metal_sigmoid_matches_torch_cpu_emulation() -> None:
    """Unflip and sigmoid on Metal give the strict CPU emulation's float32 bits."""
    from medmlx_core import import_mlx

    from flexray_mlx._kernels import torch_cpu_sigmoid
    from flexray_mlx._types import NativeRuntime
    from flexray_mlx.medmlx import multilabel_probabilities

    mx = cast(NativeRuntime, import_mlx())
    logits = np.concatenate(seeded_member_logits(2))
    # Quotients below the smallest normal float32, and the saturation branches.
    logits[1, 0, 0, :6] = np.array([-87.5, -88.7, -103.9, -104.5, 100.5, 1e-45], np.float32)
    flipped = np.array([True, False])
    expected = multilabel_probabilities(
        np.where(flipped[:, None, None, None], logits[..., ::-1], logits)
    )
    actual = torch_cpu_sigmoid(
        mx.array(np.ascontiguousarray(logits.transpose(0, 2, 3, 1))),
        mx.array(flipped.astype(np.int32)),
        mx=mx,
    )
    assert np.array_equal(np.asarray(actual).view(np.uint32), expected.view(np.uint32))
