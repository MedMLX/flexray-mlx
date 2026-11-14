"""MedMLX runner parity against pinned fxr recordings, plus argument rejection.

Reference scripts import only upstream fxr and Torch; tests read recorded arrays.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import cast

import medmlx_core.weights as core_weights
import numpy as np
import pytest
from _reference_cases import FIXTURES, UPSTREAM_REVISION
from _runner_cases import (
    CASES,
    CONTENT_BOX_SIZES,
    E2E_CASE,
    LABEL_NAMES,
    PERCENTILE_PAIRS,
    PREPROCESSING_VARIANTS,
    RELEASE_PREPROCESSING,
    RESTORE_CASES,
    seeded_member_logits,
    seeded_percentile_batch,
    seeded_probabilities,
    source_grid,
    write_case,
    write_release_bundle,
)
from medmlx_core.errors import AssetNotReadyError, InvalidInputError
from medmlx_core.outputs import validate_outputs
from PIL import Image

from flexray_mlx import medmlx
from flexray_mlx._types import FloatArray, HostArray, RunResult
from flexray_mlx.tta import Forward, predict_with_tta

PROBABILITY_BOUND = 9e-3
# Untyped view of the runner for deserialized invalid argument values.
run_untyped = cast(Callable[..., RunResult], medmlx.segment)


def _recording(name: str) -> dict[str, HostArray]:
    """Load a darwin-arm64 recording and check its pinned fxr revision."""
    with np.load(FIXTURES / f"{name}.npz", allow_pickle=False) as data:
        assert json.loads(str(data["metadata"]))["flexray_revision"] == UPSTREAM_REVISION
        arrays = {key: data[key] for key in data.files}
    if name in {"runner_reference", "runner_preprocessing", "runner_run"}:
        default = Path(__file__).resolve().parents[1] / "work/real-reference"
        directory = Path(os.environ.get("MEDMLX_TEST_REFERENCES", str(default)))
        with np.load(directory / f"{name}.npz", allow_pickle=False) as data:
            assert json.loads(str(data["real_metadata"]))["flexray_revision"] == UPSTREAM_REVISION
            arrays.update({key: data[key] for key in data.files})
    return arrays


@pytest.fixture(scope="module")
def reference() -> dict[str, HostArray]:
    """Recorded display, preparation, normalization and postprocessing cases."""
    return {**_recording("runner_reference"), **_recording("runner_preprocessing")}


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    """A release layout whose flagship has no tensors (admission only)."""
    return write_release_bundle(tmp_path / "flexray", state_dict={})


@pytest.fixture
def stub_network(monkeypatch: pytest.MonkeyPatch) -> FloatArray:
    """Replace only the MLX network call with seeded 61x256x256 probabilities."""
    predicted = seeded_probabilities()

    def predict(bundles: Sequence[Path], prepared: FloatArray, **_options: object) -> FloatArray:
        """Return the seeded probabilities without executing the network."""
        return predicted.copy()

    monkeypatch.setattr(medmlx, "_predict", predict)
    return predicted


def _image(path: str | Path) -> HostArray:
    """Read a saved image as a NumPy array."""
    with Image.open(path) as image:
        return np.asarray(image).copy()


@pytest.mark.mlx
def test_run_matches_upstream_with_seeded_network(tmp_path: Path) -> None:
    """Compare seeded Metal inference and exported outputs with the upstream recording."""
    fixture = _recording("runner_run")
    case, label, threshold = E2E_CASE
    image = write_case(tmp_path / "inputs", case)
    weights = write_release_bundle(tmp_path / "flexray", weight_seed=int(fixture["weight_seed"]))
    output_dir = tmp_path / "out"
    result = medmlx.segment(
        image_path=str(image),
        weights_path=str(weights),
        output_dir=str(output_dir),
        label_name=label,
        threshold=threshold,
    )
    outputs = result["outputs"]
    _, artifacts = validate_outputs(cast("dict[str, object]", result))
    assert set(artifacts) == set(outputs)
    assert artifacts["mask"]["kind"] == "binary_mask"
    assert artifacts["mask"]["coordinate_space"] == "pixel"
    assert artifacts["mask"].get("grid_of") == "image_path"
    assert artifacts["mask"].get("labels") == [{"value": 1, "name": label}]
    assert artifacts["probabilities"]["kind"] == "scalar_field"
    assert artifacts["probabilities"].get("grid_of") is None
    probabilities = np.load(outputs["probabilities"], allow_pickle=False)
    assert probabilities.shape == (61, 256, 256) and probabilities.dtype == np.float32
    channel = cast(FloatArray, fixture["probabilities_channel"])
    expected = np.broadcast_to(channel, (61, 256, 256))
    # This 9e-3 bound holds on Metal against the Mac fxr recording. Convolution,
    # interpolation and InstanceNorm accumulate rounding on this structured
    # input; a tighter measured Metal bound remains to be calibrated.
    np.testing.assert_allclose(probabilities, expected, rtol=0, atol=PROBABILITY_BOUND)
    mask = _image(outputs["mask"])
    assert mask.shape == (120, 90) and mask.dtype == np.uint8
    source_margin = cast(FloatArray, source_grid(np.abs(channel - np.float32(threshold)), 120, 90))
    # Rounding may change threshold decisions only inside the allowed probability error.
    stable = source_margin > PROBABILITY_BOUND
    expected_mask = source_grid((channel >= threshold).astype(np.uint8), 120, 90)
    assert np.array_equal(mask[stable], expected_mask[stable])
    assert np.array_equal(_image(outputs["preview"]), fixture["display"])
    prediction = json.loads(Path(outputs["metadata"]).read_text(encoding="utf-8"))
    assert prediction["content_box_tlbr"] == fixture["content_box"].tolist() == [0, 32, 256, 224]
    assert prediction["selected_channel"] == 59 and prediction["label_names"] == LABEL_NAMES


def test_runner_exports_recorded_upstream_unet_prediction(
    tmp_path: Path, bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replay recorded network logits through preparation, sigmoid and file export."""
    fixture = _recording("runner_run")
    case, label, threshold = E2E_CASE

    def forward(images: FloatArray) -> FloatArray:
        """Check prepared inputs and replay recorded network logits."""
        assert np.array_equal(images, fixture["prepared"])
        logits = cast(FloatArray, fixture["logits_channel"])
        return np.broadcast_to(logits, (1, 61, 256, 256)).copy()

    def load_models(bundles: Sequence[Path]) -> tuple[Forward, ...]:
        """Return the recorded forward callable for the runner."""
        return (forward,)

    monkeypatch.setattr(medmlx, "_load_models", load_models)
    result = medmlx.segment(
        image_path=write_case(tmp_path, case),
        weights_path=bundle,
        output_dir=tmp_path / "out",
        label_name=label,
        threshold=threshold,
    )
    outputs = result["outputs"]
    expected = np.broadcast_to(fixture["probabilities_channel"], (61, 256, 256))
    assert np.array_equal(np.load(outputs["probabilities"]), expected)
    decisions = (cast(FloatArray, fixture["probabilities_channel"]) >= threshold).astype(np.uint8)
    assert np.array_equal(_image(outputs["mask"]), source_grid(decisions, 120, 90))
    assert np.array_equal(_image(outputs["preview"]), fixture["display"])
    prediction = json.loads(Path(outputs["metadata"]).read_text())
    assert prediction["content_box_tlbr"] == fixture["content_box"].tolist()
    assert prediction["tta_samples"] == 1 and prediction["tta_seed"] is None


@pytest.mark.parametrize(("case", "label", "threshold"), RESTORE_CASES)
def test_run_postprocessing_matches_upstream_decisions(
    tmp_path: Path,
    bundle: Path,
    stub_network: FloatArray,
    reference: dict[str, HostArray],
    case: str,
    label: str,
    threshold: float,
) -> None:
    """Compare restored masks and output metadata with upstream cases."""
    output_dir = tmp_path / "out"
    result = medmlx.segment(
        image_path=write_case(tmp_path / "inputs", case),
        weights_path=bundle,
        output_dir=output_dir,
        label_name=label,
        threshold=threshold,
    )
    outputs = result["outputs"]
    height, width = reference[f"restore/{case}/mask"].shape
    decisions = stub_network[LABEL_NAMES.index(label)] >= threshold
    expected_mask = source_grid(decisions.astype(np.uint8), height, width)
    mask = _image(outputs["mask"])
    assert mask.dtype == np.uint8 and np.array_equal(mask, expected_mask)
    assert np.array_equal(_image(outputs["preview"]), reference[f"restore/{case}/display"])
    assert np.array_equal(np.load(outputs["probabilities"]), stub_network)
    prediction = json.loads(Path(outputs["metadata"]).read_text(encoding="utf-8"))
    assert prediction["content_box_tlbr"] == reference[f"restore/{case}/content_box"].tolist()
    assert prediction["selected_channel"] == LABEL_NAMES.index(label)


@pytest.mark.parametrize("case", CASES)
def test_display_and_preprocessing_match_upstream(
    tmp_path: Path, reference: dict[str, HostArray], case: str
) -> None:
    """Compare raster display and preprocessing variants with recorded outputs."""
    path = write_case(tmp_path, case)
    display = np.asarray(medmlx.display_image(path))
    assert display.dtype == np.uint8 and np.array_equal(display, reference[f"display/{case}"])
    for variant, metadata in PREPROCESSING_VARIANTS.items():
        expected = reference[f"prepared/{case}/{variant}"]
        actual = medmlx.ImagePreprocessing.from_metadata(metadata).prepare(path)
        assert actual.dtype == expected.dtype == np.float32, variant
        assert np.array_equal(actual, expected), variant


def test_release_preprocessing_and_content_boxes_match_upstream(
    tmp_path: Path, reference: dict[str, HostArray]
) -> None:
    """Compare release preparation and content boxes with upstream recordings."""
    release = medmlx.ImagePreprocessing.from_metadata(RELEASE_PREPROCESSING)
    prepared = release.prepare(write_case(tmp_path, "l8"))
    assert prepared.shape == (1, 1, 256, 256)
    assert np.array_equal(prepared, reference["prepared/l8/release"])
    for variant, metadata in {"release": RELEASE_PREPROCESSING, **PREPROCESSING_VARIANTS}.items():
        preprocessing = medmlx.ImagePreprocessing.from_metadata(metadata)
        boxes = [preprocessing.content_box(h, w) for h, w in CONTENT_BOX_SIZES]
        assert np.array_equal(np.array(boxes), reference[f"content_box/{variant}"]), variant


@pytest.mark.parametrize("index", range(len(PERCENTILE_PAIRS)))
def test_percentile_normalization_matches_torch_quantile(
    reference: dict[str, HostArray], index: int
) -> None:
    """Compare percentile normalization with recorded Torch quantiles."""
    # Compare each radiograph tile with Torch's percentile interpolation.
    preprocessing = medmlx.ImagePreprocessing(
        normalization_scheme="percentile_minmax", percentiles=PERCENTILE_PAIRS[index]
    )
    actual = preprocessing.normalize(seeded_percentile_batch())
    assert np.array_equal(actual, reference[f"percentile_minmax/{index}"])


@pytest.mark.parametrize(("copies", "step"), [(1, 1), (5, 1), (5, -1)])
def test_member_probabilities_match_torch_sigmoid_and_mean(
    reference: dict[str, HostArray], copies: int, step: int
) -> None:
    """Compare single-member and ensemble probabilities with Torch recordings."""
    # Repetition crosses processing blocks; reversed views exercise strided inputs.
    repeats = (copies, 1, 1, 1)
    logits = [np.tile(value, repeats)[..., ::step] for value in seeded_member_logits(3)]

    def replay(value: FloatArray) -> Forward:
        """Return a member whose forward pass yields recorded logits."""
        return lambda _images: value

    images = seeded_percentile_batch()[:1, :1, :1, :1]
    _, single = predict_with_tta(replay(logits[0]), images)
    _, ensemble = predict_with_tta([replay(value) for value in logits], images)
    assert single.dtype == ensemble.dtype == np.float32
    assert np.array_equal(single, np.tile(reference["probabilities/single"], repeats)[..., ::step])
    assert np.array_equal(
        ensemble, np.tile(reference["probabilities/ensemble3"], repeats)[..., ::step]
    )


@pytest.mark.parametrize(
    "problem",
    [
        "missing_image",
        "threshold_out_of_range",
        "non_bool_overwrite",
        "existing_output",
        "dangling_output",
    ],
)
def test_run_rejects_bad_arguments_before_downloading_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    """Each rejection category fails before the weights are requested."""

    def download(_model_id: str) -> Path:
        """Fail if the runner asks for weights before checking its arguments."""
        pytest.fail("weights were requested before the arguments were checked")

    monkeypatch.setattr(core_weights, "weights_dir", download)
    output = tmp_path / "out"
    changes: dict[str, dict[str, object]] = {
        "missing_image": {"image_path": tmp_path / "absent.png"},
        "threshold_out_of_range": {"threshold": 1.5},
        "non_bool_overwrite": {"overwrite": "yes"},
        "existing_output": {},
        "dangling_output": {},
    }
    if problem == "existing_output":
        output.mkdir()
        (output / medmlx.MASK_NAME).write_bytes(b"previous")
    elif problem == "dangling_output":
        output.mkdir()
        (output / medmlx.MASK_NAME).symlink_to(tmp_path / "absent.png")
    arguments = {"image_path": write_case(tmp_path, "l8"), "output_dir": output, **changes[problem]}
    with pytest.raises(InvalidInputError):
        run_untyped(**arguments)
    if problem == "dangling_output":
        assert (output / medmlx.MASK_NAME).is_symlink()


@pytest.mark.parametrize("damage", ["weights_file", pytest.param("tensors", marks=pytest.mark.mlx)])
def test_unusable_bundles_raise_asset_not_ready(tmp_path: Path, bundle: Path, damage: str) -> None:
    """Reject a missing checkpoint and one without the released tensors."""
    if damage == "weights_file":
        (bundle / "members/flux0375/model.safetensors").unlink()
    # "tensors": the real network runs on an admitted bundle that holds no tensors.
    with pytest.raises(AssetNotReadyError):
        medmlx.segment(
            image_path=write_case(tmp_path, "l8"), weights_path=bundle, output_dir=tmp_path / "out"
        )
