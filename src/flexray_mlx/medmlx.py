"""MedMLX runner for pinned FleXray v1.0.2 radiograph segmentation.

The reference is upstream fxr at bf6d01426fd3ae012aff9253ec9c9952c58bc960
(MIT): UNet, file preprocessing, DICOM decoding, multilabel probabilities and
TTA. NumPy/Pillow implement the inference helpers without Torch at runtime.
The runner exports upstream model-grid probabilities and a selected source-grid
mask, with nearest-neighbor restoration of upstream's thresholded decisions.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp
from typing import TYPE_CHECKING, Literal, SupportsFloat, SupportsIndex, SupportsInt, cast

import numpy as np
from medmlx_core.errors import (
    AssetNotReadyError,
    InvalidInputError,
    MissingDependencyError,
    ModelExecutionError,
)
from medmlx_core.outputs import artifact
from medmlx_core.weights import weights_dir, weights_revision
from PIL import Image, ImageOps, UnidentifiedImageError

from flexray_mlx._types import FloatArray, IntArray, JsonValue, RealArray, RunResult
from flexray_mlx.tta import Forward, TtaSettings

if TYPE_CHECKING:
    from flexray_mlx._types import DicomRuntime, EnsembleManifest, Normalization

MODEL = {
    "id": "flexray",
    "label": "FleXray",
    "task": "segmentation",
    "run": "flexray_mlx.medmlx:run",
    # Measured MLX peak for a 1929x2207 chest radiograph, single model; larger inputs need more.
    "peak_memory_gb": 1.2,
    "summary": (
        "Anatomy segmentation of one 2D radiograph: 61 independent sigmoid "
        "channels, one selected mask restored to the source pixel grid."
    ),
}

MEMBERS = ("flux0375", "flux000", "flux025", "flux050", "flux075")
MODEL_GRID = (256, 256)
CHANNELS = 61
PREDICTION_SCHEMA = "flexray_mlx.prediction.v2"
DISPLAY_NAME = "source_image.png"
MASK_NAME = "source_image_label.png"
PROBABILITIES_NAME = "probabilities.npy"
PREDICTION_NAME = "prediction.json"
_BUNDLE_FILES = (
    "config.yml",
    "label_schema.json",
    "model.safetensors",
    "preprocessing.json",
)
_WEIGHTS_HINT = (
    "Omit weights_path to download MedMLX/flexray-mlx (the flux0375 flagship), or pass a "
    "FleXray snapshot directory holding ensemble.json and members/<name>/{config.yml,"
    "model.safetensors,label_schema.json,preprocessing.json}. The five-member ensemble is "
    "huggingface.co/VictorButoi/flexray at revision 6b63ddf09cfdff99aea494e3738cf243a4e22cdb."
)

DEFAULT_IMAGE_SIZE = (256, 256)
DEFAULT_NORMALIZATION_EPS = 1.0e-8
DEFAULT_NORMALIZATION_PERCENTILES = (0.5, 99.5)
_NORMALIZATION_SCHEMES = frozenset({"none", "minmax", "percentile_minmax", "standardize"})
type NormalizationScheme = Literal["none", "minmax", "percentile_minmax", "standardize"]
type MemberResidency = Literal["all", "one"]
DICOM_SUFFIXES = frozenset({".dcm", ".dicom"})
_DICOM_PREAMBLE_LENGTH = 128
_DICOM_MAGIC = b"DICM"


# --- Released bundle admission ------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AdmittedBundle:
    """Admitted ensemble members and their shared label names.

    Attributes:
        bundles: Member directories under ``<root>/members``: the flagship only, or
            all five in ``MEMBERS`` order.
        labels: The 61 label names; a label's position is its probability channel.
    """

    bundles: tuple[Path, ...]
    labels: tuple[str, ...]


def _missing(path: Path) -> AssetNotReadyError:
    """Construct the standard missing-model-asset error for a path."""
    return AssetNotReadyError(
        f"Missing FleXray asset: {path}", reason=f"missing {path}", hint=_WEIGHTS_HINT
    )


def _read_json(path: Path) -> JsonValue:
    """Read a JSON model asset and classify missing or malformed files."""
    if not path.is_file():
        raise _missing(path)
    try:
        return cast(JsonValue, json.loads(path.read_text(encoding="utf-8")))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssetNotReadyError(
            f"Unreadable FleXray asset {path}: {error}",
            reason=f"unreadable {path}",
            hint=_WEIGHTS_HINT,
        ) from error


def admit_bundle(root: Path, *, ensemble: bool) -> AdmittedBundle:
    """Check the pinned ensemble layout, labels and shared preprocessing.

    Only the members that will run are inspected: the flagship alone, or all five
    when ``ensemble`` is true. Tensor contents are not read here; the network
    loads them and validates tensor use during inference. These checks do not
    establish the identity or origin of the bundle or checkpoint.

    Raises:
        AssetNotReadyError: A file is missing or unreadable, the ensemble layout
            differs from the pinned release, members disagree on labels or
            preprocessing, or the label list is not 61 names starting with
            ``background``.
    """

    if not root.is_dir():
        raise _missing(root)
    manifest = cast("EnsembleManifest", _read_json(root / "ensemble.json"))
    try:
        listed = tuple(Path(str(item["subfolder"])).name for item in manifest["members"])
        flagship = manifest.get("flagship")
    except (AttributeError, KeyError, TypeError) as error:
        raise AssetNotReadyError(
            "FleXray ensemble.json is malformed",
            reason="malformed ensemble.json",
            hint=_WEIGHTS_HINT,
        ) from error
    if flagship != "members/flux0375" or listed != MEMBERS:
        raise AssetNotReadyError(
            "FleXray ensemble layout differs from the pinned release",
            reason="unpinned ensemble layout",
            hint=_WEIGHTS_HINT,
        )
    members = MEMBERS if ensemble else MEMBERS[:1]
    bundles = tuple(root / "members" / name for name in members)
    reference_schema: JsonValue = None
    reference_preprocessing: JsonValue = None
    for bundle in bundles:
        for name in _BUNDLE_FILES:
            if not (bundle / name).is_file():
                raise _missing(bundle / name)
        schema = _read_json(bundle / "label_schema.json")
        preprocessing = _read_json(bundle / "preprocessing.json")
        if reference_schema is None:
            reference_schema, reference_preprocessing = schema, preprocessing
        elif schema != reference_schema or preprocessing != reference_preprocessing:
            raise AssetNotReadyError(
                "FleXray ensemble members disagree on labels or preprocessing",
                reason="inconsistent ensemble members",
                hint=_WEIGHTS_HINT,
            )
    assert reference_schema is not None
    raw_names = reference_schema.get("label_names") if isinstance(reference_schema, dict) else None
    if not isinstance(raw_names, list):
        raise AssetNotReadyError(
            "FleXray released label schema has no label list",
            reason="malformed label_schema.json",
            hint=_WEIGHTS_HINT,
        )
    labels = tuple(str(name) for name in raw_names)
    if len(labels) != CHANNELS or labels[0] != "background":
        raise AssetNotReadyError(
            "FleXray released label schema is invalid",
            reason="invalid label_schema.json",
            hint=_WEIGHTS_HINT,
        )
    return AdmittedBundle(bundles=bundles, labels=labels)


# --- DICOM input (upstream fxr/inference/dicom.py) ------------------------------------


def _pydicom() -> DicomRuntime:
    """Import the DICOM reader or raise an actionable dependency error."""
    try:
        import pydicom
    except ImportError as error:
        raise MissingDependencyError(
            "Reading DICOM radiographs requires pydicom",
            hint="Install pydicom (a flexray-mlx dependency) in this environment.",
        ) from error
    return cast("DicomRuntime", pydicom)


def is_dicom_path(path: str | Path) -> bool:
    """Return whether a path is a DICOM file by suffix or, without a suffix, DICM magic."""

    candidate = Path(path)
    if candidate.suffix.lower() in DICOM_SUFFIXES:
        return True
    if candidate.suffix or not candidate.is_file():
        return False
    try:
        with candidate.open("rb") as handle:
            handle.seek(_DICOM_PREAMBLE_LENGTH)
            return handle.read(len(_DICOM_MAGIC)) == _DICOM_MAGIC
    except OSError:
        return False


def load_dicom_grayscale(path: str | Path) -> FloatArray:
    """Read one single-frame DICOM into a 2D float32 image min-max scaled to [0, 1].

    Pixels are rescaled with ``RescaleSlope``/``RescaleIntercept`` when present
    (a false-valued slope is treated as 1), RGB(A) is reduced to luma, and
    MONOCHROME1 is inverted. No VOI LUT or windowing is applied, so the full
    stored intensity range maps to [0, 1]. Declared multiple frames are rejected;
    without a frame count a 3D array with last dimension 3 or 4 is treated as color.

    Raises:
        ValueError: The file has no pixel data or is not a single-frame 2D image.
    """

    dataset = _pydicom().dcmread(str(path), force=True)
    if "PixelData" not in dataset:
        raise ValueError(f"DICOM file has no pixel data: {path}.")
    if int(getattr(dataset, "NumberOfFrames", 1) or 1) > 1:
        raise ValueError(f"Only single-frame 2D DICOM images are supported: {path}.")
    pixels: RealArray = np.asarray(dataset.pixel_array)
    if pixels.ndim == 3 and pixels.shape[-1] in {3, 4}:
        rgb = pixels[..., :3].astype(np.float64, copy=False)
        pixels = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    if pixels.ndim != 2:
        raise ValueError(
            "Only single-frame 2D DICOM images are supported; got pixel shape "
            f"{pixels.shape} for {path}."
        )
    values = pixels.astype(np.float64, copy=False)
    slope = float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)
    values = values * slope + intercept
    if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        values = values.max() - values
    return scale_to_unit_range(values)


def scale_to_unit_range(values: RealArray) -> FloatArray:
    """Min-max scale one image into [0, 1] float32; a constant image becomes zeros."""

    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.integer):
        low, high = int(values.min()), int(values.max())
        origin = max(low, 0) + min(high, 0)
        values = values - np.asarray(origin, dtype=values.dtype)
    values = np.asarray(values, dtype=np.float64)
    low = float(values.min())
    high = float(values.max())
    values = (values - low) / (high - low) if high > low else np.zeros_like(values)
    return values.astype(np.float32)


# --- Source-grid display ------------------------------------------------------------


def display_image(path: Path) -> Image.Image:
    """Return the source radiograph as an 8-bit grayscale image on its own pixel grid.

    This is the image the exported mask overlays; it is not the model input.
    Integer files of 16 bits or fewer are scaled by their dtype range, wider
    integers and floats outside [0, 1] by min-max. DICOM is min-max scaled on load.

    Raises:
        InvalidInputError: The image is not 2D after grayscale conversion.
    """

    image = _load_image_file(path)
    array = _grayscale_pixels(image) if isinstance(image, Image.Image) else image
    values = _scale_numpy_integer_range(array, source_dtype=array.dtype)
    if values.ndim != 2:
        raise InvalidInputError("FleXray requires a 2D radiograph")
    values = _check_pixel_values(values[None, None])[0, 0]
    # Float files retain their intensity units in fxr; display those via its minmax rule.
    if np.issubdtype(array.dtype, np.floating) and (values.min() < 0 or values.max() > 1):
        values = scale_to_unit_range(values)
    return Image.fromarray(np.rint(values * 255).astype(np.uint8))


def _grayscale_pixels(image: Image.Image) -> RealArray:
    """Convert a Pillow image to a 2D grayscale array as upstream does.

    Applies the EXIF orientation and composites transparency over black. Modes
    ``1``, ``L``, ``I``, ``I;16`` and ``F`` keep their values; other modes go
    through Pillow's ``convert("L")``.
    """
    image = ImageOps.exif_transpose(image)
    if (
        image.mode in {"RGBA", "LA", "PA"}
        or "transparency" in image.info
        or (image.palette is not None and image.palette.mode == "RGBA")
    ):
        canvas = Image.new("RGBA", image.size, (0, 0, 0, 255))
        image = Image.alpha_composite(canvas, image.convert("RGBA"))
    grayscale = image if image.mode in {"1", "L", "I", "I;16", "F"} else image.convert("L")
    return np.array(grayscale, copy=True)


# --- Preprocessing (upstream fxr/inference/preprocessing.py) --------------------------


@dataclass(frozen=True)
class ImagePreprocessing:
    """The released ``preprocessing.json`` contract for file inputs.

    Mirrors upstream ``ImagePreprocessing`` with Torch tensors replaced by
    float32 NumPy arrays shaped ``1x1xHxW``. ``prepare`` scales the file to float
    intensities, zero-pads to a square (if ``pad_to_square``), resizes with
    Pillow bilinear interpolation, then normalizes.

    Attributes:
        image_size: Model grid as ``(height, width)`` in pixels.
        color_mode: Only ``"grayscale"`` is supported.
        pad_to_square: Center-pad the shorter side with zeros before resizing.
        scale: ``"zero_one"`` scales integer files of 16 bits or fewer by their
            dtype range and wider integers by min-max, and passes floats through;
            ``"none"`` casts everything to float32.
        mean: Optional scalar subtracted after ``normalization_scheme``.
        std: Optional positive scalar divisor applied after ``mean``.
        normalization_scheme: ``"none"``, ``"minmax"``, ``"percentile_minmax"``
            or ``"standardize"``, applied per sample.
        percentiles: ``(lower, upper)`` in percent; required for, and only valid
            with, ``"percentile_minmax"``.
        eps: Positive stabilizer for the per-sample schemes: a floor on the value
            range for the min-max schemes, added to the std for ``"standardize"``.
    """

    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE
    color_mode: str = "grayscale"
    pad_to_square: bool = True
    scale: str = "zero_one"
    mean: float | None = None
    std: float | None = None
    normalization_scheme: str = "none"
    percentiles: tuple[float, float] | None = None
    eps: float = DEFAULT_NORMALIZATION_EPS

    def __post_init__(self) -> None:
        """Validate image geometry, scaling, and normalization settings."""
        _validate_image_size(self.image_size)
        if self.color_mode != "grayscale":
            raise ValueError(
                "FleXray inference currently supports only grayscale preprocessing; "
                f"got color_mode={self.color_mode!r}."
            )
        if self.scale not in {"zero_one", "none"}:
            raise ValueError(
                f"preprocessing scale must be 'zero_one' or 'none', got {self.scale!r}."
            )
        _validate_normalization(
            scheme=self.normalization_scheme,
            percentiles=self.percentiles,
            eps=self.eps,
            mean=self.mean,
            std=self.std,
        )

    @classmethod
    def from_metadata(cls, metadata: Mapping[str, object] | None) -> ImagePreprocessing:
        """Build preprocessing from ``preprocessing.json`` metadata."""

        data = dict(metadata or {})
        size_value = cast(
            Sequence[SupportsInt], data.get("image_size", data.get("size", DEFAULT_IMAGE_SIZE))
        )
        image_size = _validate_image_size(size_value)
        color_mode = str(data.get("color_mode", "grayscale")).lower()
        if color_mode != "grayscale":
            raise ValueError(
                "FleXray inference currently supports only grayscale preprocessing; "
                f"got color_mode={color_mode!r}."
            )
        scale = str(data.get("scale", "zero_one")).lower()
        if scale not in {"zero_one", "none"}:
            raise ValueError(f"preprocessing scale must be 'zero_one' or 'none', got {scale!r}.")
        normalization = _normalization_values(data)
        return cls(
            image_size=image_size,
            color_mode=color_mode,
            pad_to_square=bool(data.get("pad_to_square", True)),
            scale=scale,
            mean=normalization["mean"],
            std=normalization["std"],
            normalization_scheme=normalization["scheme"],
            percentiles=normalization["percentiles"],
            eps=normalization["eps"],
        )

    def content_box(self, height: int, width: int) -> tuple[int, int, int, int]:
        """``(top, left, bottom, right)`` of the input footprint on the prepared canvas.

        Coordinates are model-grid pixels with exclusive ``bottom`` and ``right``.
        The box approximates the resized source footprint, not a guarantee that
        outside pixels are zero: bilinear interpolation can mix the boundary and
        normalization can change padding values. ``height`` and ``width`` are
        the source image size in pixels.

        Raises:
            ValueError: ``height`` or ``width`` is not positive.
        """

        if height <= 0 or width <= 0:
            raise ValueError("Input height and width must be positive.")
        side = max(height, width)
        canvas_height = side if self.pad_to_square else height
        canvas_width = side if self.pad_to_square else width
        top = (canvas_height - height) // 2
        left = (canvas_width - width) // 2
        scale_y = self.image_size[0] / canvas_height
        scale_x = self.image_size[1] / canvas_width
        out_top = min(round(top * scale_y), self.image_size[0] - 1)
        out_left = min(round(left * scale_x), self.image_size[1] - 1)
        bottom = max(out_top + 1, round((top + height) * scale_y))
        right = max(out_left + 1, round((left + width) * scale_x))
        return out_top, out_left, bottom, right

    def prepare(self, path: str | Path) -> FloatArray:
        """Load one image or DICOM file as a normalized float32 ``1x1xHxW`` array.

        ``H`` and ``W`` are ``image_size``. Non-finite values that reach the
        preprocessing check become 0 with a ``RuntimeWarning``; a constant image
        also warns. DICOM min-max scaling occurs first and may turn non-finite
        decoded values into a constant zero image before that check.
        """

        image = _load_image_file(path)
        if isinstance(image, Image.Image):
            return self._prepare_pillow(image)
        return self._prepare_array(image)

    def _prepare_pillow(self, image: Image.Image) -> FloatArray:
        """Convert a raster image to grayscale and prepare it for inference."""
        array = _grayscale_pixels(image)
        return self._prepare_grayscale_array(array, source_dtype=array.dtype)

    def _prepare_array(self, array: FloatArray) -> FloatArray:
        # File inputs reach this only as decoded 2D DICOM pixels.
        """Validate decoded DICOM pixels and prepare the grayscale array."""
        np_array = np.asarray(array)
        if np_array.size == 0:
            raise ValueError("Image inputs must be non-empty.")
        if np_array.ndim != 2:
            raise ValueError(f"DICOM image inputs must have shape HxW; got {np_array.shape}.")
        return self._prepare_grayscale_array(np_array, source_dtype=np_array.dtype)

    def _prepare_grayscale_array(
        self, array: RealArray, *, source_dtype: np.dtype[np.generic]
    ) -> FloatArray:
        """Scale, pad, resize, and normalize one grayscale image."""
        if array.size == 0:
            raise ValueError("Image inputs must be non-empty.")
        if self.scale == "zero_one":
            values = _scale_numpy_integer_range(array, source_dtype=source_dtype)
        else:
            values = np.asarray(array, dtype=np.float32)
        values = _check_pixel_values(np.array(values, copy=True)[None, None])[0, 0]
        prepared = Image.fromarray(values)
        if self.pad_to_square:
            prepared = _pad_to_square(prepared)
        height, width = self.image_size
        resized = prepared.resize((width, height), resample=Image.Resampling.BILINEAR)
        return self.normalize(np.array(resized, dtype=np.float32, copy=True)[None, None])

    def normalize(self, batch: FloatArray) -> FloatArray:
        """Apply the configured per-sample or scalar normalization to ``BxCxHxW``.

        Each sample is normalized over all of its channels and pixels. The
        per-sample schemes and the scalar ``mean``/``std`` are mutually exclusive
        (enforced at construction), so at most one of them takes effect.
        """

        normalized = batch
        if self.normalization_scheme == "standardize":
            dims = tuple(range(1, normalized.ndim))
            mean = _standardize_mean(normalized)
            # Torch defaults to correction=1 and accumulates CPU variance in double.
            std = normalized.std(axis=dims, keepdims=True, dtype=np.float64, ddof=1).astype(
                np.float32
            )
            normalized = (normalized - mean) / (std + np.float32(self.eps))
        elif self.normalization_scheme == "minmax":
            batch_size = normalized.shape[0]
            flat = normalized.reshape(batch_size, -1)
            view_shape = (batch_size,) + (1,) * (normalized.ndim - 1)
            lower = flat.min(axis=1).reshape(view_shape)
            upper = flat.max(axis=1).reshape(view_shape)
            normalized = (normalized - lower) / np.maximum(upper - lower, np.float32(self.eps))
        elif self.normalization_scheme == "percentile_minmax":
            normalized = _percentile_minmax(
                normalized,
                percentiles=self.percentiles or DEFAULT_NORMALIZATION_PERCENTILES,
                eps=self.eps,
            )
        if self.mean is not None:
            normalized = normalized - np.float32(self.mean)
        if self.std is not None:
            normalized = normalized / np.float32(self.std)
        return np.ascontiguousarray(normalized, dtype=np.float32)


def _load_image_file(path: str | Path) -> Image.Image | FloatArray:
    """Load DICOM pixels or a detached Pillow raster image."""
    if is_dicom_path(path):
        return load_dicom_grayscale(path)
    with Image.open(path) as opened:
        opened.load()
        return opened.copy()


def _standardize_mean(batch: FloatArray) -> FloatArray:
    """Torch CPU cascade sum with four float32 lanes, then division by sample size.

    Mirrors SumKernel.cpp's four ILP accumulators and four cascade levels. The
    fixed lane order avoids large-offset cancellation differences from NumPy's
    pairwise mean. All samples and chunks are accumulated together in NumPy.
    PyTorch's BSD notice is in root THIRD_PARTY_NOTICES.md.
    """

    flat = np.ascontiguousarray(batch).reshape(batch.shape[0], -1)
    count = flat.shape[1]
    lanes = 4 if count >= 4 else 1
    vectors = count // lanes
    groups = vectors // 4
    values = flat[:, : groups * 4 * lanes].reshape(len(flat), groups, 4, lanes)
    step = 1 << max(4, (max(groups, 1) - 1).bit_length() // 4)
    accumulators: list[FloatArray] = []
    for level in range(4):
        complete = values.shape[1] // step if level < 3 else 0
        tail = values[:, complete * step :]
        accumulator = np.zeros((len(flat), 4, lanes), dtype=np.float32)
        if tail.shape[1]:
            accumulator = cast(FloatArray, np.add.accumulate(tail, axis=1))[:, -1]
        accumulators.append(accumulator)
        values = values[:, : complete * step].reshape(len(flat), complete, step, 4, lanes)
        values = cast(FloatArray, np.add.accumulate(values, axis=2))[:, :, -1]
    partial = accumulators[0]
    for accumulator in accumulators[1:]:
        partial = partial + accumulator
    for index in range(groups * 4, vectors):
        partial[:, 0] += flat[:, index * lanes : (index + 1) * lanes]
    for index in range(1, 4):
        partial[:, 0] += partial[:, index]
    total = np.zeros(len(flat), dtype=np.float32)
    for index in range(vectors * lanes, count):
        total += flat[:, index]
    for index in range(lanes):
        total += partial[:, 0, index]
    return (total / np.float32(count)).reshape((len(flat),) + (1,) * (batch.ndim - 1))


def _validate_image_size(size: Sequence[SupportsInt]) -> tuple[int, int]:
    """Require positive height and width and return them as integers."""
    if len(size) != 2:
        raise ValueError(f"Image size must be an H W pair, got {tuple(size)}.")
    height, width = int(size[0]), int(size[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Image size dimensions must be positive, got {(height, width)}.")
    return height, width


def _scale_numpy_integer_range(
    array: RealArray, *, source_dtype: np.dtype[np.generic]
) -> FloatArray:
    """Scale an array to float32 by its source dtype (the ``zero_one`` scale mode).

    Unsigned integers of at most 16 bits divide by the dtype maximum; signed
    ones map the dtype range onto [0, 1]. Wider integers use data min-max.
    Booleans and floats are cast to float32 unchanged.
    """

    dtype = np.dtype(source_dtype)
    if np.issubdtype(dtype, np.bool_) or not np.issubdtype(dtype, np.integer):
        return np.asarray(array, dtype=np.float32)
    if dtype.itemsize > 2:
        return scale_to_unit_range(array)
    values = np.asarray(array, dtype=np.float32)
    limits = np.iinfo(cast(np.dtype[np.integer], dtype))
    if limits.min >= 0:
        return values / float(limits.max)
    return (values - float(limits.min)) / float(limits.max - limits.min)


def _normalization_values(data: Mapping[str, object]) -> Normalization:
    """Canonicalize ``preprocessing.json`` normalization fields.

    Accepts the current ``normalization: {scheme, percentiles, eps}`` mapping (with
    top-level ``mean``/``std``) and the legacy forms that carry only scalar
    ``mean``/``std``, either top-level or inside ``normalization``.
    """
    normalization = data.get("normalization")
    if isinstance(normalization, Mapping):
        normalization = cast(Mapping[str, object], normalization)
        if "scheme" in normalization:
            unknown = sorted(set(normalization).difference({"scheme", "percentiles", "eps"}))
            if unknown:
                raise ValueError(f"Unsupported preprocessing normalization keys: {unknown}.")
            scheme = normalization.get("scheme")
            percentiles = normalization.get("percentiles")
            eps = normalization.get("eps", DEFAULT_NORMALIZATION_EPS)
            mean = data.get("mean")
            std = data.get("std")
        else:
            scheme = "none"
            percentiles = None
            eps = DEFAULT_NORMALIZATION_EPS
            mean = normalization.get("mean", data.get("mean"))
            std = normalization.get("std", data.get("std"))
    elif normalization is None:
        scheme = "none"
        percentiles = None
        eps = DEFAULT_NORMALIZATION_EPS
        mean = data.get("mean")
        std = data.get("std")
    else:
        raise TypeError("preprocessing normalization must be a mapping.")
    canonical_mean = (
        None if mean is None else float(cast(str | SupportsFloat | SupportsIndex, mean))
    )
    canonical_std = None if std is None else float(cast(str | SupportsFloat | SupportsIndex, std))
    canonical_percentiles = (
        None if percentiles is None else _validate_percentiles(cast(Sequence[float], percentiles))
    )
    canonical_eps = _validate_eps(eps)
    _validate_normalization(
        scheme=scheme,
        percentiles=canonical_percentiles,
        eps=canonical_eps,
        mean=canonical_mean,
        std=canonical_std,
    )
    return {
        "scheme": cast(str, scheme),
        "percentiles": canonical_percentiles,
        "eps": canonical_eps,
        "mean": canonical_mean,
        "std": canonical_std,
    }


def _validate_normalization(
    *,
    scheme: object,
    percentiles: Sequence[float] | None,
    eps: object,
    mean: float | None,
    std: float | None,
) -> None:
    """Reject unsupported, incompatible, or non-finite normalization settings."""
    if not isinstance(scheme, str) or scheme not in _NORMALIZATION_SCHEMES:
        raise ValueError(
            "preprocessing normalization scheme must be one of "
            f"{sorted(_NORMALIZATION_SCHEMES)}; got {scheme!r}."
        )
    _validate_eps(eps)
    if scheme == "percentile_minmax":
        if percentiles is None:
            raise ValueError("percentile_minmax preprocessing requires two percentiles.")
        _validate_percentiles(percentiles)
    elif percentiles is not None:
        raise ValueError("preprocessing percentiles are valid only for percentile_minmax.")
    if scheme != "none" and (mean is not None or std is not None):
        raise ValueError(
            "Per-sample preprocessing normalization cannot be combined with legacy scalar mean/std."
        )
    if mean is not None and not math.isfinite(mean):
        raise ValueError("preprocessing mean must be finite.")
    if std is not None and (not math.isfinite(std) or std <= 0.0):
        raise ValueError("preprocessing std must be a positive finite number.")


def _validate_percentiles(values: object) -> tuple[float, float]:
    """Require two finite increasing percentiles in the range zero to 100."""
    if isinstance(values, str | bytes) or not isinstance(values, Sequence):
        raise TypeError("preprocessing percentiles must be a two-value sequence.")
    values = cast(Sequence[str | SupportsFloat | SupportsIndex], values)
    if len(values) != 2:
        raise ValueError("preprocessing percentiles must contain two values.")
    if any(isinstance(value, bool) for value in values):
        raise TypeError("preprocessing percentiles must contain two numbers.")
    try:
        lower, upper = float(values[0]), float(values[1])
    except (TypeError, ValueError) as exc:
        raise TypeError("preprocessing percentiles must contain two numbers.") from exc
    if not math.isfinite(lower) or not math.isfinite(upper):
        raise ValueError("preprocessing percentiles must be finite.")
    if not 0.0 <= lower < upper <= 100.0:
        raise ValueError("preprocessing percentiles must satisfy 0 <= lower < upper <= 100.")
    return lower, upper


def _validate_eps(value: object) -> float:
    """Require and return a positive finite normalization stabilizer."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError("preprocessing normalization eps must be a number.")
    eps = float(value)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("preprocessing normalization eps must be a positive finite number.")
    return eps


def _percentile_minmax(
    batch: FloatArray, *, percentiles: tuple[float, float], eps: float
) -> FloatArray:
    """Per-sample percentile clipping and scaling with ``torch.quantile`` semantics."""

    batch_size = batch.shape[0]
    flat = batch.reshape(batch_size, -1)
    fractions = np.asarray([percentiles[0] / 100.0, percentiles[1] / 100.0], dtype=np.float32)
    bounds = torch_quantile_linear(flat, fractions)
    view_shape = (batch_size,) + (1,) * (batch.ndim - 1)
    lower = bounds[0].reshape(view_shape)
    upper = bounds[1].reshape(view_shape)
    clipped = np.minimum(np.maximum(batch, lower), upper)
    return (clipped - lower) / np.maximum(upper - lower, np.float32(eps))


def torch_quantile_linear(rows: FloatArray, fractions: FloatArray) -> FloatArray:
    """Reproduce ``torch.quantile(rows, fractions, dim=1)`` for finite float32 rows.

    Torch forms the ranks in float32 and interpolates with ``lerp``, whose CPU
    kernel compiles to fused multiply-adds; this follows that arithmetic so the
    result matches Torch's float32 CPU output on the recorded test inputs.
    ``rows`` is ``(samples, values)`` and ``fractions`` are quantile levels in
    [0, 1]. Returns ``(len(fractions), samples)``.
    """

    ordered = np.sort(rows, axis=1)
    ranks = fractions.astype(np.float32) * np.float32(ordered.shape[1] - 1)
    below = ranks.astype(np.int64)
    weights = ranks - below.astype(np.float32)
    above = np.ceil(ranks).astype(np.int64)
    start = ordered[:, below]
    end = ordered[:, above]
    weight = np.broadcast_to(weights, start.shape)
    small = np.abs(weight) < 0.5
    diff = end - start
    interpolated = np.where(
        small,
        fma32(weight, diff, start),
        fma32(-diff, np.float32(1.0) - weight, end),
    )
    return np.ascontiguousarray(interpolated.T)


def _check_pixel_values(batch: FloatArray) -> FloatArray:
    """Replace non-finite pixels with 0 and warn; also warn on a constant sample."""

    finite = np.isfinite(batch)
    if not bool(finite.all()):
        warnings.warn(
            f"Image contains {int((~finite).sum())} non-finite pixel values; "
            "they were replaced with 0.",
            RuntimeWarning,
            stacklevel=2,
        )
        batch = np.nan_to_num(batch, nan=0.0, posinf=0.0, neginf=0.0)
    flat = batch.reshape(batch.shape[0], -1)
    if bool((flat.max(axis=1) == flat.min(axis=1)).any()):
        warnings.warn(
            "Image has no contrast (every pixel is identical).", RuntimeWarning, stacklevel=2
        )
    return batch


def _pad_to_square(image: Image.Image) -> Image.Image:
    """Center the image on a zero-filled square; an odd remainder goes right/bottom."""

    width, height = image.size
    side = max(width, height)
    if width == height:
        return image
    pad_left = (side - width) // 2
    pad_right = side - width - pad_left
    pad_top = (side - height) // 2
    pad_bottom = side - height - pad_top
    return ImageOps.expand(image, border=(pad_left, pad_top, pad_right, pad_bottom), fill=0)


# --- Probabilities (upstream probabilities.py and tta.py, Torch CPU arithmetic) ------


def fma32(
    a: RealArray | np.floating | float,
    b: RealArray | np.floating | float,
    c: RealArray | np.floating | float,
) -> FloatArray:
    """Correctly rounded float32 ``a * b + c`` (one rounding), elementwise.

    The float64 product of two float32 values is exact. The float64 sum can
    round once more; when it lands exactly on a float32 rounding midpoint, the
    TwoSum error term decides the direction, so the result equals a hardware FMA.
    """

    product = np.asarray(a, dtype=np.float32).astype(np.float64) * np.asarray(
        b, dtype=np.float32
    ).astype(np.float64)
    addend = np.asarray(c, dtype=np.float32).astype(np.float64)
    total = product + addend
    virtual = total - product
    error = (product - (total - virtual)) + (addend - virtual)
    bits = np.asarray(total, dtype=np.float64).view(np.uint64)
    midpoint = (bits & np.uint64(0x1FFFFFFF)) == np.uint64(0x10000000)
    nudged = np.where(
        midpoint & (error != 0.0), np.nextafter(total, np.copysign(np.inf, error)), total
    )
    return np.asarray(nudged).astype(np.float32)


_R_LN2 = np.float32(1.442695040888963407359924681001892137426645954152985934135449406931)
_L2U = np.float32(0.693145751953125)
_L2L = np.float32(1.428606765330187045e-06)
_EXP_COEFFICIENTS = tuple(
    np.float32(value)
    for value in (
        0.000198527617612853646278381,
        0.00139304355252534151077271,
        0.00833336077630519866943359,
        0.0416664853692054748535156,
        0.166666671633720397949219,
        0.5,
    )
)


def sleef_expf(d: FloatArray) -> FloatArray:
    """SLEEF ``expf`` (1.0-ULP, FMA) as used by Torch's vectorized CPU ``sigmoid``."""

    d = np.asarray(d, dtype=np.float32)
    with np.errstate(invalid="ignore", over="ignore"):
        q = np.rint(d * _R_LN2).astype(np.int32)
        qf = q.astype(np.float32)
        s = fma32(qf, -_L2U, d)
        s = fma32(qf, -_L2L, s)
        u = np.full_like(d, _EXP_COEFFICIENTS[0])
        for coefficient in _EXP_COEFFICIENTS[1:]:
            u = fma32(u, s, coefficient)
        u = np.float32(1.0) + fma32(s * s, u, s)
        half = q >> 1
        u = u * np.ldexp(np.float32(1.0), half).astype(np.float32)
        u = u * np.ldexp(np.float32(1.0), q - half).astype(np.float32)
        u = np.where(d < np.float32(-104.0), np.float32(0.0), u)
        u = np.where(np.float32(100.0) < d, np.float32(np.inf), u)
    return u.astype(np.float32)


def multilabel_probabilities(logits: FloatArray) -> FloatArray:
    """``torch.sigmoid`` on float32 CPU logits: ``1 / (exp(0 - x) + 1)``."""

    logits = np.asarray(logits, dtype=np.float32)
    values = logits.reshape(-1)
    probabilities = np.empty(logits.shape, dtype=np.float32)
    output = probabilities.reshape(-1)
    # Bound the FP64 FMA-emulation temporaries while retaining each FP32 rounding.
    block_size = 16384
    for start in range(0, values.size, block_size):
        stop = start + block_size
        exponent = sleef_expf(np.float32(0.0) - values[start:stop])
        output[start:stop] = np.float32(1.0) / (exponent + np.float32(1.0))
    return probabilities


# --- Source-grid mask export --------------------------------------------------------


def restore_mask(
    probabilities: FloatArray,
    *,
    channel: int,
    pad_to_square: bool,
    source_size: tuple[int, int],
    threshold: float,
) -> Image.Image:
    """Threshold on the model grid, then map the decisions back to the source pixels.

    This inverts ``ImagePreprocessing.prepare``'s geometry: source pixel ``(y, x)``
    lies at ``(y + top, x + left)`` on the canvas the source was padded onto, and
    takes the model pixel whose footprint contains that pixel's center under the
    center-aligned resize, ``floor((c + 0.5) * n / canvas)``. Integer arithmetic
    keeps centers that fall exactly on a footprint boundary on the later pixel.
    Stretching the rounded ``content_box`` instead would move edges by up to half
    a model pixel along the padded axis.

    Args:
        probabilities: ``(61, H, W)`` float32 sigmoid probabilities on the model grid.
        channel: Index of the selected label in ``probabilities``.
        pad_to_square: Whether preprocessing centered the source on a square canvas.
        source_size: Source image ``(width, height)`` in pixels (Pillow order).
        threshold: Pixels with probability ``>= threshold`` are foreground.

    Returns:
        A mode ``L`` image of size ``source_size`` holding 0/1 values. Thresholding
        happens before the nearest-neighbor resize so that no new decision is
        created by interpolating probabilities.
    """

    width, height = source_size
    side = max(width, height)
    canvas_width, canvas_height = (side, side) if pad_to_square else (width, height)
    grid_height, grid_width = probabilities.shape[1:]

    def centres(size: int, canvas: int, grid: int) -> IntArray:
        offset = (canvas - size) // 2
        return ((2 * (np.arange(size, dtype=np.int64) + offset) + 1) * grid) // (2 * canvas)

    rows = centres(height, canvas_height, grid_height)
    columns = centres(width, canvas_width, grid_width)
    selected = (probabilities[channel] >= threshold).astype(np.uint8)
    return Image.fromarray(np.ascontiguousarray(selected[rows[:, None], columns]))


# --- Runner ---------------------------------------------------------------------------


def _load_models(bundles: Sequence[Path]) -> tuple[Forward, ...]:
    """Admit MLX once and load each member once, to share every augmented view.

    Each member's tensor names and shapes are checked while it loads, so a
    checkpoint that does not match the released network fails here, before
    inference.
    """
    from medmlx_core.runtime import import_mlx
    from safetensors import SafetensorError
    from yaml import YAMLError

    import_mlx()
    from flexray_mlx.mlx_model import FleXrayMlxModel

    models: list[Forward] = []
    for bundle in bundles:
        try:
            models.append(FleXrayMlxModel(bundle))
        # Unreadable or malformed bundle files only; a config that is not a mapping
        # surfaces as KeyError, TypeError or AttributeError. MLX/Metal RuntimeErrors
        # propagate unchanged instead of being reported as incompatible weights.
        except (
            OSError,
            KeyError,
            TypeError,
            AttributeError,
            ValueError,
            SafetensorError,
            YAMLError,
        ) as error:
            raise AssetNotReadyError(
                f"FleXray weights in {bundle} do not match the released network: {error}",
                reason=f"incompatible {bundle}",
                hint=_WEIGHTS_HINT,
            ) from error
    return tuple(models)


MEMBER_RESIDENCY = ("all", "one")


def _execute[T](call: Callable[[], T]) -> T:
    """Run inference work, reporting ``ValueError``/``RuntimeError`` as ``ModelExecutionError``."""
    try:
        return call()
    except (ValueError, RuntimeError) as error:
        raise ModelExecutionError(f"FleXray inference failed: {error}") from error


def _predict(
    bundles: Sequence[Path],
    prepared: FloatArray,
    *,
    preprocessing: ImagePreprocessing,
    settings: TtaSettings,
    member_residency: str = "all",
) -> FloatArray:
    """Run every member's MLX U-Net on every view; return ``61x256x256``.

    ``"all"`` loads every member first and evaluates pass by pass. ``"one"``
    keeps one member's weights resident at a time: the views are drawn once,
    each member writes its per-view probabilities to a private temporary
    spool and is released, and the spooled terms are then summed in the same
    pass-major order, so both policies return identical float32 bits. The
    spool holds ``passes * members`` arrays of ``B x 61 x 256 x 256`` float32.
    """
    from flexray_mlx.tta import (
        mean_probabilities,
        prepared_views,
        tta_probabilities,
        view_probabilities,
    )

    if member_residency == "all" or len(bundles) == 1:
        models = _load_models(bundles)
        probabilities = _execute(
            lambda: tta_probabilities(
                models,
                prepared,
                tta_samples=settings.samples,
                seed=settings.seed,
                normalizer=preprocessing.normalize,
            )
        )
        return probabilities[0]
    views = _execute(lambda: list(prepared_views(prepared, settings, preprocessing.normalize)))
    with TemporaryDirectory(prefix="flexray-mlx-spool-") as spool:
        term = [
            [Path(spool) / f"{view}-{member}.npy" for member in range(len(bundles))]
            for view in range(len(views))
        ]
        for member, bundle in enumerate(bundles):
            (model,) = _load_models((bundle,))
            for index, (view, flipped) in enumerate(views):
                values = _execute(partial(view_probabilities, model, view, flipped))
                np.save(term[index][member], values)
            # Release this member before the next one loads.
            del model
        probabilities = mean_probabilities(
            (cast(FloatArray, np.load(path, allow_pickle=False)) for row in term for path in row),
            len(views) * len(bundles),
        )
    return probabilities[0]


def _input_errors() -> tuple[type[BaseException], ...]:
    """Exceptions meaning an unreadable input; pydicom's parse error only if installed."""
    errors: tuple[type[BaseException], ...] = (
        EOFError,
        OSError,
        UnidentifiedImageError,
        ValueError,
    )
    try:
        from pydicom.errors import InvalidDicomError
    except ImportError:
        return errors
    return (*errors, InvalidDicomError)


def _require_bool(name: str, value: object) -> bool:
    """Require a boolean runner argument or raise an input error."""
    if not isinstance(value, bool):
        raise InvalidInputError(f"FleXray {name} must be true or false, got {value!r}")
    return value


def _publish(staged: dict[Path, Path]) -> None:
    """Replace staged artifacts together, restoring earlier files on failure."""
    recovery = Path(mkdtemp(prefix=".flexray-mlx-backup-", dir=next(iter(staged)).parent))
    cleanup = True
    try:
        backups: dict[Path, Path] = {}
        for index, path in enumerate(staged):
            if path.exists() or path.is_symlink():
                backup = recovery / str(index)
                os.link(path, backup, follow_symlinks=False)
                backups[path] = backup
        changed: list[Path] = []
        try:
            for path, temporary in staged.items():
                os.replace(temporary, path)
                changed.append(path)
        except BaseException:
            try:
                for path in reversed(changed):
                    backup = backups.get(path)
                    if backup is None:
                        path.unlink(missing_ok=True)
                    else:
                        os.replace(backup, path)
            except OSError as error:
                cleanup = False
                raise OSError(f"Cannot restore outputs; backups retained at {recovery}") from error
            raise
    finally:
        if cleanup:
            shutil.rmtree(recovery)


def run(
    *,
    image_path: str | Path,
    weights_path: str | Path | None = None,
    output_dir: str | Path,
    label_name: str = "lungs",
    ensemble: bool = False,
    threshold: float = 0.5,
    tta_samples: int = 1,
    tta_seed: int | None = None,
    normalization_scheme: NormalizationScheme | None = None,
    overwrite: bool = False,
    member_residency: MemberResidency = "all",
) -> RunResult:
    """Segment one conventional 2D radiograph with the released FleXray network.

    The model natively predicts on a 256x256 grid after the bundle's preprocessing
    (normally zero-padding the image to a square). All 61 independent sigmoid
    channels (background plus 60 structures) are saved on that model grid. The
    selected structure's binary
    mask is thresholded on the model grid, resized with nearest neighbor to the padded
    canvas and unpadded, so it keeps the source image's pixel dimensions.

    Args:
        image_path: One radiograph: a raster image Pillow reads (PNG, TIFF, ...)
            or a single-frame 2D DICOM file. DICOM detection uses ``.dcm`` or
            ``.dicom`` (case-insensitive), or ``DICM`` at byte offset 128 for a
            file without a suffix.
        weights_path: Optional model folder. Omit to download ``MedMLX/flexray-mlx``.
            The flagship snapshot at revision
            ``722d8341570e8b9e1196cf50e57892da84974fcc``, containing
            ``ensemble.json`` and the ``members/flux0375/`` bundle
            (``config.yml``, ``model.safetensors``, ``label_schema.json``,
            ``preprocessing.json``).
        output_dir: Directory for the outputs; created if missing.
        label_name: Structure to threshold into the mask, e.g. ``lungs``,
            ``heart``, ``rib_5`` or ``background`` (any name in the label schema).
        ensemble: Average the probabilities of all five released members
            instead of running only the ``flux0375`` flagship (default False).
            Requires all five member bundles: ``flux0375``, ``flux000``,
            ``flux025``, ``flux050``, ``flux075``. The published MedMLX snapshot
            contains only the flagship; use the full ``VictorButoi/flexray``
            snapshot at ``6b63ddf09cfdff99aea494e3738cf243a4e22cdb`` for an ensemble.
        threshold: Probability cutoff in [0, 1] for the selected structure; pixels
            at or above it are foreground.
        tta_samples: Non-negative integer total passes per member; 0 or 1 means
            one plain pass.
        tta_seed: Integer in ``[-2**63, 2**64 - 1]`` for the isolated stream that
            reproduces Torch CPU augmentation draws; None is unseeded, as in fxr.
        normalization_scheme: Override the bundle's normalization with ``none``,
            ``minmax``, ``percentile_minmax`` or ``standardize``. None keeps the
            bundle contract. Standardize uses sample std and adds bundle epsilon.
        overwrite: Replace existing outputs instead of refusing; they are replaced
            only after every new output is written. Output paths must remain
            separate from each other, the input radiograph and the documented
            model bundle files, including their file aliases.
        member_residency: ``"all"`` (default) loads every ensemble member before
            inference. ``"one"`` keeps one member's weights resident at a time and
            spools its per-view probabilities to a temporary directory
            (``tta_samples * members`` arrays of 61x256x256 float32); the result is
            bit-identical. Without ``ensemble`` both values run the same path.

    Returns:
        ``outputs`` maps ``mask`` to ``source_image_label.png`` (uint8 0/1 on the
        source pixel grid), ``display_source`` to ``source_image.png`` (the 8-bit
        grayscale source), ``probabilities`` to ``probabilities.npy`` (float32
        ``61x256x256`` on the model grid) and ``prediction`` to
        ``prediction.json``. ``artifacts`` describes them, all in ``pixel`` space
        (none has an affine): ``mask`` is a ``binary_mask`` with one label (value 1,
        named ``label_name``) and ``display_source`` is a ``preview``, both on the
        grid of ``image_path``; ``probabilities`` is a ``scalar_field`` on the model
        grid; ``prediction`` is a ``report``. ``details`` holds the selected label and channel,
        threshold, members, TTA settings, normalization scheme, source size as
        ``[width, height]``, the content box as ``[top, left, bottom, right]`` in
        model-grid pixels (exclusive bottom/right) and the mask pixel count.

    Raises:
        InvalidInputError: Missing or unreadable input, a non-2D DICOM, an
            unknown label, an out-of-range threshold, invalid TTA,
            normalization or residency settings, outputs that alias an input, model asset or
            each other, or existing outputs without ``overwrite``.
        AssetNotReadyError: Missing or unreadable assets, an incompatible bundle
            layout or network, or inconsistent member metadata.
        MissingDependencyError: MLX or pydicom is unavailable.
        ModelExecutionError: Inference failed, or the network returned
            non-finite or wrongly shaped probabilities. Raised before any
            output file is written.
    """

    source_path = Path(image_path).expanduser()
    if not source_path.is_file():
        raise InvalidInputError(f"Missing FleXray input radiograph: {source_path}")
    source_path = source_path.resolve()
    if not isinstance(cast(object, label_name), str):
        raise InvalidInputError(f"FleXray label_name must be a string, got {label_name!r}")
    ensemble = _require_bool("ensemble", ensemble)
    if cast(object, member_residency) not in MEMBER_RESIDENCY:
        raise InvalidInputError(f"FleXray member_residency must be one of {MEMBER_RESIDENCY}")
    overwrite = _require_bool("overwrite", overwrite)
    if (
        isinstance(threshold, bool)
        or not isinstance(cast(object, threshold), int | float)
        or not 0.0 <= threshold <= 1.0
    ):
        raise InvalidInputError("FleXray threshold must be between 0 and 1")
    threshold = float(threshold)
    try:
        settings = TtaSettings(tta_samples, tta_seed)
    except ValueError as error:
        raise InvalidInputError(str(error)) from error
    if normalization_scheme is not None and (
        not isinstance(cast(object, normalization_scheme), str)
        or normalization_scheme not in _NORMALIZATION_SCHEMES
    ):
        raise InvalidInputError(
            f"normalization_scheme must be one of {sorted(_NORMALIZATION_SCHEMES)}, or None"
        )
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise InvalidInputError(f"FleXray output_dir is not a directory: {output}")
    names = (DISPLAY_NAME, MASK_NAME, PROBABILITIES_NAME, PREDICTION_NAME)
    destinations = tuple(output / name for name in names)
    # resolve() catches symlink aliases and samefile() catches hard links.
    if any(
        path.resolve() == source_path or (path.exists() and path.samefile(source_path))
        for path in destinations
    ):
        raise InvalidInputError("FleXray outputs must be separate from the input radiograph")
    if any(
        first.resolve() == second.resolve()
        or (first.exists() and second.exists() and first.samefile(second))
        for first, second in combinations(destinations, 2)
    ):
        raise InvalidInputError("FleXray outputs must not refer to the same file")
    existing = [path.as_posix() for path in destinations if path.exists() or path.is_symlink()]
    if existing and not overwrite:
        raise InvalidInputError(
            f"FleXray outputs already exist: {', '.join(existing)}; "
            "pass overwrite=True to replace them"
        )
    try:
        source = display_image(source_path)
    except InvalidInputError:
        # InvalidInputError is a ValueError; do not re-wrap it as unreadable input.
        raise
    except _input_errors() as error:
        raise InvalidInputError(f"Unreadable FleXray radiograph {source_path}: {error}") from error

    # Every argument the bundle does not define is checked; only now may weights download.
    weights_root = (
        weights_dir("flexray") if weights_path is None else Path(weights_path).expanduser()
    )
    model_assets = (
        weights_root / "ensemble.json",
        *(weights_root / "members" / member / name for member in MEMBERS for name in _BUNDLE_FILES),
    )
    if any(
        path.resolve() == asset.resolve()
        or (path.exists() and asset.exists() and path.samefile(asset))
        for path in destinations
        for asset in model_assets
    ):
        raise InvalidInputError("FleXray outputs must be separate from each model asset")
    admitted = admit_bundle(weights_root, ensemble=ensemble)
    labels = admitted.labels
    if label_name not in labels:
        raise InvalidInputError(f"Unknown FleXray anatomy label: {label_name}")
    metadata = _read_json(admitted.bundles[0] / "preprocessing.json")
    try:
        preprocessing = ImagePreprocessing.from_metadata(cast(Mapping[str, object], metadata))
    except (TypeError, ValueError) as error:
        raise AssetNotReadyError(
            f"Unsupported FleXray preprocessing.json: {error}",
            reason="unsupported preprocessing.json",
            hint=_WEIGHTS_HINT,
        ) from error
    if normalization_scheme is not None:
        preprocessing = replace(
            preprocessing,
            normalization_scheme=normalization_scheme,
            mean=None,
            std=None,
            percentiles=(
                preprocessing.percentiles or DEFAULT_NORMALIZATION_PERCENTILES
                if normalization_scheme == "percentile_minmax"
                else None
            ),
        )
    try:
        prepared = preprocessing.prepare(source_path)
    except _input_errors() as error:
        raise InvalidInputError(f"Unreadable FleXray radiograph {source_path}: {error}") from error
    probabilities = _predict(
        admitted.bundles,
        prepared,
        preprocessing=preprocessing,
        settings=settings,
        member_residency=member_residency,
    )
    if probabilities.shape != (CHANNELS, *MODEL_GRID) or not np.isfinite(probabilities).all():
        raise ModelExecutionError("FleXray returned invalid full-channel probabilities")
    channel = labels.index(label_name)
    box = preprocessing.content_box(source.height, source.width)
    mask = restore_mask(
        probabilities,
        channel=channel,
        pad_to_square=preprocessing.pad_to_square,
        source_size=source.size,
        threshold=threshold,
    )

    output.mkdir(parents=True, exist_ok=True)
    display_path = output / DISPLAY_NAME
    mask_path = output / MASK_NAME
    probabilities_path = output / PROBABILITIES_NAME
    prediction_path = output / PREDICTION_NAME
    members = [bundle.name for bundle in admitted.bundles]
    record = {
        "schema_version": PREDICTION_SCHEMA,
        "source_path": source_path.as_posix(),
        "source_size": [source.width, source.height],
        "model_grid": list(MODEL_GRID),
        "content_box_tlbr": list(box),
        "label_names": list(labels),
        "selected_label": label_name,
        "selected_channel": channel,
        "threshold": threshold,
        "tta_samples": settings.resolved_samples,
        "tta_seed": settings.seed,
        "normalization_scheme": preprocessing.normalization_scheme,
        "ensemble": ensemble,
        "members": members,
        "backend": "mlx",
        "probability_mode": "multilabel",
        "overlay_is_source_grid": True,
    }
    # Private staging keeps predictable top-level symlinks away from the writers.
    with TemporaryDirectory(prefix=".flexray-mlx-", dir=output) as staging:
        partial = {name: Path(staging) / name for name in names}
        source.save(partial[DISPLAY_NAME])
        mask.save(partial[MASK_NAME])
        np.save(partial[PROBABILITIES_NAME], probabilities, allow_pickle=False)
        partial[PREDICTION_NAME].write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        _publish({output / name: temporary for name, temporary in partial.items()})
    return {
        "outputs": {
            "mask": mask_path.as_posix(),
            "display_source": display_path.as_posix(),
            "probabilities": probabilities_path.as_posix(),
            "prediction": prediction_path.as_posix(),
        },
        "artifacts": {
            "mask": artifact(
                kind="binary_mask",
                coordinate_space="pixel",
                grid_of="image_path",
                labels=[{"value": 1, "name": label_name}],
            ),
            "display_source": artifact(
                kind="preview", coordinate_space="pixel", grid_of="image_path"
            ),
            "probabilities": artifact(kind="scalar_field", coordinate_space="pixel"),
            "prediction": artifact(kind="report", coordinate_space="not_applicable"),
        },
        "details": {
            "weights_revision": weights_revision(weights_root),
            "selected_label": label_name,
            "selected_channel": channel,
            "threshold": threshold,
            "ensemble": ensemble,
            "members": members,
            "tta_samples": settings.resolved_samples,
            "tta_seed": settings.seed,
            "normalization_scheme": preprocessing.normalization_scheme,
            "source_size": [source.width, source.height],
            "model_grid": list(MODEL_GRID),
            "content_box_tlbr": list(box),
            "mask_pixels": int(np.count_nonzero(np.asarray(mask))),
            "probability_mode": "multilabel",
            "backend": "mlx",
        },
    }
