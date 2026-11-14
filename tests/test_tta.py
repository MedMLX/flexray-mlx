"""Recorded fxr parity; no Torch, torchvision, fxr or MLX imports."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import cast

import numpy as np
import pytest
from _reference_cases import FIXTURES, UPSTREAM_REVISION
from _runner_cases import (
    MEMBERS,
    RELEASE_PREPROCESSING,
    source_grid,
    write_release_bundle,
)
from _tta_cases import TTA_CASES, WEIGHT_SEEDS, SeededForward, seeded_input, standardize_inputs
from PIL import Image

from flexray_mlx import medmlx
from flexray_mlx._random import TorchRandom
from flexray_mlx._types import HostArray, RealArray
from flexray_mlx.tta import augment_batch, clahe, predict_with_tta


@pytest.fixture(scope="module")
def recorded() -> dict[str, HostArray]:
    """Load the fxr TTA recordings and verify their provenance and weight seeds."""
    arrays: dict[str, HostArray] = {}
    default = Path(__file__).resolve().parents[1] / "work/real-reference"
    directory = Path(os.environ.get("MEDMLX_TEST_REFERENCES", str(default)))
    for name in ("transforms", "predictions", "runner_percentile_minmax", "runner_standardize"):
        with np.load(FIXTURES / f"tta_{name}.npz", allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            assert metadata["flexray_revision"] == UPSTREAM_REVISION
            assert metadata["weight_seeds"] == list(WEIGHT_SEEDS)
            arrays.update({key: data[key] for key in data.files if key != "metadata"})
        with np.load(directory / f"tta_{name}.npz", allow_pickle=False) as data:
            assert json.loads(str(data["real_metadata"]))["flexray_revision"] == UPSTREAM_REVISION
            arrays.update({key: data[key] for key in data.files if key != "real_metadata"})
    return arrays


@pytest.mark.parametrize("seed", (0, 7, -1, 2**63 + 9))
def test_uniform_draws_match_torch_exactly(recorded: dict[str, HostArray], seed: int) -> None:
    """Compare seeded uniform draws with the recorded Torch stream."""
    random = TorchRandom(seed)
    actual = np.array([random.uniform(0.7, 1.3) for _ in range(100)], dtype=np.float32)
    assert np.array_equal(actual, recorded[f"uniform/{seed}"])


@pytest.mark.parametrize("index", range(4))
def test_noise_matches_torch_draw_order(recorded: dict[str, HostArray], index: int) -> None:
    """Compare normal draws and the subsequent random value with Torch."""
    expected = recorded[f"noise/{index}"]
    random = TorchRandom(7)
    actual = random.normal(expected.shape)
    # Observed max 1.1920929e-7: scalar/SIMD Box-Muller transcendental rounding.
    np.testing.assert_allclose(actual, cast(RealArray, expected), rtol=0, atol=1.2e-7)
    assert random.uniform() == recorded[f"noise_next/{index}"].item()


def test_all_augmented_views_and_flip_masks_match_fxr(recorded: dict[str, HostArray]) -> None:
    """Compare seeded augmented views and flip masks with source recordings."""
    assert all(json.loads(str(recorded["view_branch_counts"])).values())
    random = TorchRandom(0)
    for index in range(16):
        actual, flipped = augment_batch(seeded_input(), random)
        assert np.array_equal(flipped, recorded[f"flip/{index}"])
        # Observed max 1.7881393e-7 from gamma/sharpness/noise float32 rounding.
        np.testing.assert_allclose(
            actual, cast(RealArray, recorded[f"view/{index}"]), rtol=0, atol=1.8e-7
        )


@pytest.mark.parametrize("index", range(4))
def test_clahe_matches_fxr_bitwise(recorded: dict[str, HostArray], index: int) -> None:
    """Compare CLAHE outputs with source recordings across image shapes."""
    expected = recorded[f"clahe/{index}"]
    actual = clahe(seeded_input(expected.shape), clip_limit=1.4, grid_size=(8, 8))
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("name", tuple(standardize_inputs()))
def test_standardize_matches_fxr(recorded: dict[str, HostArray], name: str) -> None:
    """Compare standardization with source cases including constant inputs."""
    eps = 1e-3 if name == "near_constant" else 1e-8
    preprocessing = medmlx.ImagePreprocessing(normalization_scheme="standardize", eps=eps)
    with np.errstate(invalid="ignore", divide="ignore"):
        actual = preprocessing.normalize(standardize_inputs()[name])
    assert actual.dtype == np.float32 and actual.flags.c_contiguous
    expected = recorded[f"standardize/{name}"]
    if name == "singleton":
        # Both yield NaN from correction=1; NaN payload bits are platform-dependent.
        assert np.isnan(actual).all() and np.isnan(expected).all()
        return
    assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize(("name", "seed", "samples", "members", "scheme"), TTA_CASES)
def test_seeded_convolution_tta_matches_fxr(
    recorded: dict[str, HostArray], name: str, seed: int, samples: int, members: int, scheme: str
) -> None:
    """Compare seeded TTA logits and probabilities with source recordings."""
    preprocessing = medmlx.ImagePreprocessing(
        normalization_scheme=scheme,
        percentiles=(0.5, 99.5) if scheme == "percentile_minmax" else None,
    )
    with np.errstate(invalid="ignore"):
        logits, probabilities = predict_with_tta(
            tuple(SeededForward(s) for s in WEIGHT_SEEDS[:members]),
            preprocessing.normalize(seeded_input()),
            tta_samples=samples,
            seed=seed,
            normalizer=preprocessing.normalize,
        )
    shape = (2, 61, 24, 40)
    assert logits.shape == probabilities.shape == shape
    assert logits.dtype == probabilities.dtype == np.float32
    expected_probabilities = np.broadcast_to(recorded[f"probabilities/{name}"], shape)
    expected_logits = np.broadcast_to(recorded[f"logits/{name}"], shape)
    # Observed maxima 1.7881393e-7 / 8.3446503e-7: convolution, sigmoid and
    # aggregation rounding; log-odds amplifies probability error.
    np.testing.assert_allclose(
        probabilities, cast(RealArray, expected_probabilities), rtol=0, atol=1.8e-7
    )
    np.testing.assert_allclose(logits, cast(RealArray, expected_logits), rtol=0, atol=8.4e-7)


@pytest.mark.parametrize("scheme", ("percentile_minmax", "standardize"))
@pytest.mark.parametrize("override", (False, True))
def test_runner_tta_and_normalization_restore_the_source_grid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    recorded: dict[str, HostArray],
    scheme: medmlx.NormalizationScheme,
    override: bool,
) -> None:
    """Compare ensemble TTA and exported source-grid masks with recordings."""
    root = write_release_bundle(tmp_path / "weights", members=MEMBERS, state_dict={})
    if not override:
        for member in MEMBERS:
            directory = root / "members" / member
            metadata = {**RELEASE_PREPROCESSING, "normalization": {"scheme": scheme}}
            if scheme == "percentile_minmax":
                cast(dict[str, object], metadata["normalization"])["percentiles"] = [0.5, 99.5]
            (directory / "preprocessing.json").write_text(json.dumps(metadata), encoding="utf-8")
    image = tmp_path / "image.png"
    Image.fromarray((seeded_input((37, 53)) * np.float32(255)).astype(np.uint8)).save(image)
    loaded: list[str] = []

    def load(bundles: tuple[Path, ...]) -> tuple[SeededForward, ...]:
        """Record requested members and return each member's seeded fixture forward."""
        loaded.extend(bundle.name for bundle in bundles)
        return tuple(SeededForward(WEIGHT_SEEDS[MEMBERS.index(b.name)]) for b in bundles)

    monkeypatch.setattr(medmlx, "_load_models", load)
    preprocessing = medmlx.ImagePreprocessing(
        normalization_scheme=scheme,
        percentiles=(0.5, 99.5) if scheme == "percentile_minmax" else None,
    )
    assert np.array_equal(preprocessing.prepare(image), recorded[f"runner/{scheme}/prepared"])
    result = medmlx.run(
        image_path=image,
        weights_path=root,
        output_dir=tmp_path / "out",
        ensemble=True,
        tta_samples=2,
        tta_seed=0,
        normalization_scheme=scheme if override else None,
    )
    assert loaded == list(MEMBERS)
    actual = np.load(result["outputs"]["probabilities"], allow_pickle=False)
    # One resident member at a time must reduce the spooled terms in the same
    # pass-major float32 order, so the probabilities are bit-identical.
    one = medmlx.run(
        image_path=image,
        weights_path=root,
        output_dir=tmp_path / "one",
        ensemble=True,
        tta_samples=2,
        tta_seed=0,
        normalization_scheme=scheme if override else None,
        member_residency="one",
    )
    assert loaded == list(MEMBERS) * 2
    one_resident = np.load(one["outputs"]["probabilities"], allow_pickle=False)
    assert np.array_equal(one_resident.view(np.uint32), actual.view(np.uint32))
    expected = np.broadcast_to(recorded[f"runner/{scheme}/probabilities"], actual.shape)
    # Observed max 1.7881393e-7 on both 256x256 five-member cases: convolution
    # and sequential averaging round differently between Torch and NumPy.
    np.testing.assert_allclose(actual, cast(RealArray, expected), rtol=0, atol=1.8e-7)
    with Image.open(result["outputs"]["mask"]) as mask:
        # Every channel holds the recorded probabilities, so the lungs channel does too.
        lungs = cast(RealArray, recorded[f"runner/{scheme}/probabilities"])
        expected_mask = source_grid((lungs >= 0.5).astype(np.uint8), 37, 53)
        assert np.array_equal(np.asarray(mask), expected_mask)
        assert mask.size == (53, 37)
    record = json.loads(Path(result["outputs"]["prediction"]).read_text(encoding="utf-8"))
    assert record["content_box_tlbr"] == recorded[f"runner/{scheme}/content_box"].tolist()
