"""Upstream UNet parity."""

from pathlib import Path

import numpy as np
import pytest
from _reference_cases import FIXTURES, UPSTREAM_REVISION, write_bundle

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
        actual = FleXrayMlxModel(tmp_path)(fixture["images"])
        expected = fixture["output_nhwc"].transpose(0, 3, 1, 2)
        assert actual.shape == (1, 61, 256, 256)
        assert actual.dtype == np.float32
        # Observed Metal/Torch CPU max abs 4.3442845e-4 on these seeded inputs.
        # 4.35e-4 bounds convolution/reduction rounding on this seeded case.
        np.testing.assert_allclose(actual, expected, rtol=0, atol=LOGIT_BOUND)
