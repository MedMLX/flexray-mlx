"""Array and namespace contracts for the pinned native and file boundaries."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Literal, NotRequired, Protocol, TypedDict

import numpy as np
from medmlx_core.typing import MlxRuntime
from numpy.typing import NDArray

if TYPE_CHECKING:
    from mlx.core import StreamOrDevice, array

type HostArray = NDArray[np.generic]
type RealArray = NDArray[np.bool_ | np.integer | np.floating]
type FloatArray = NDArray[np.float32]
type DoubleArray = NDArray[np.float64]
type BoolArray = NDArray[np.bool_]
type IntArray = NDArray[np.int64]
type UIntArray = NDArray[np.uint32]
type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None


class NativeRuntime(MlxRuntime, Protocol):
    def square(self, x: array, /, *, stream: StreamOrDevice = None) -> array:
        """Square an MLX array elementwise."""
        ...

    def compile(self, fun: Callable[[array], array]) -> Callable[[array], array]:
        """Compile a pure one-array MLX function."""
        ...


class NativeOperation(Protocol):
    def __call__(self, x: array) -> array:
        """Apply a native operation to an MLX array."""
        ...


class NeuralRuntime(Protocol):
    def MaxPool2d(
        self, kernel_size: int | tuple[int, int], stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] | None = 0,
    ) -> NativeOperation:
        """Construct a two-dimensional MLX max-pooling layer."""
        ...


class TensorReader(Protocol):
    def __enter__(self) -> TensorReader:
        """Open the tensor-reader context."""
        ...
    def __exit__(self, *args: object) -> None:
        """Close the tensor-reader context."""
        ...
    def keys(self) -> list[str]:
        """List the named tensors in the checkpoint."""
        ...
    def get_tensor(self, name: str) -> HostArray:
        """Read one named tensor as a host array."""
        ...


class SafetensorsRuntime(Protocol):
    def safe_open(self, filename: str | Path, framework: Literal["np"]) -> TensorReader:
        """Open named safetensors as NumPy arrays."""
        ...


class DicomDataset(Protocol):
    pixel_array: RealArray

    def __contains__(self, name: str) -> bool:
        """Return whether the dataset contains a named DICOM element."""
        ...


class DicomRuntime(Protocol):
    def dcmread(self, fp: str, *, force: bool = False) -> DicomDataset:
        """Read one DICOM dataset from a file path."""
        ...


class Normalization(TypedDict):
    """Normalization settings parsed from ``preprocessing.json``.

    ``percentiles`` applies only to ``percentile_minmax``; ``mean``/``std`` are the
    legacy scalar form and exclude the per-sample schemes.
    """

    scheme: str
    percentiles: tuple[float, float] | None
    eps: float
    mean: float | None
    std: float | None


class EnsembleMember(TypedDict):
    subfolder: str


class EnsembleManifest(TypedDict):
    members: list[EnsembleMember]
    flagship: str


class RunOutputs(TypedDict):
    """POSIX paths of the files ``run`` writes to its output directory."""

    mask: str
    display_source: str
    probabilities: str
    prediction: str


class RunDetails(TypedDict):
    """Settings and geometry of one ``run``.

    ``source_size`` is ``[width, height]`` of the input; ``model_grid`` is
    ``[height, width]``. ``content_box_tlbr`` is ``[top, left, bottom, right]``
    in model-grid pixels with exclusive bottom/right.
    """

    weights_revision: NotRequired[str | None]
    selected_label: str
    selected_channel: int
    threshold: float
    ensemble: bool
    members: list[str]
    tta_samples: int
    tta_seed: int | None
    normalization_scheme: str
    source_size: list[int]
    model_grid: list[int]
    content_box_tlbr: list[int]
    mask_pixels: int
    probability_mode: str
    backend: str


class RunResult(TypedDict):
    """Return value of ``run``: output file paths and run details."""

    outputs: RunOutputs
    details: RunDetails
