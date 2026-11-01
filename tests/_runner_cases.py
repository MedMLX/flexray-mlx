"""Deterministic radiographs, bundles and arrays shared by runner fixtures and tests."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from _arrays import DoubleArray, FloatArray, HostArray
from _reference_cases import write_bundle
from PIL import Image

RELEASE_PREPROCESSING: dict[str, object] = {
    "color_mode": "grayscale",
    "image_size": [256, 256],
    "mean": None,
    "normalization": {"eps": 1e-08, "percentiles": [0.5, 99.5], "scheme": "percentile_minmax"},
    "pad_to_square": True,
    "probability_mode": "multilabel",
    "scale": "zero_one",
    "std": None,
}
PREPROCESSING_VARIANTS: dict[str, dict[str, object]] = {
    "small_percentile": {
        "image_size": [24, 40],
        "normalization": {"scheme": "percentile_minmax", "percentiles": [2.0, 97.5], "eps": 1e-3},
    },
    "small_minmax_nopad": {
        "image_size": [32, 32],
        "pad_to_square": False,
        "normalization": {"scheme": "minmax"},
    },
    "small_scalar": {"image_size": [20, 28], "scale": "none", "mean": 100.0, "std": 50.0},
}
LABEL_NAMES = [
    "background", "skull", "scapulae", "clavicles", "humeri", "phalanges", "metacarpals",
    "carpals", "radii", "ulnae", "tibiae", "femurs", "patellae", "fibulae", "metatarsals",
    "tarsals", "toes", *(f"rib_{index}" for index in range(1, 13)), "sternum",
    *(f"vertebra_c{index}" for index in range(1, 8)),
    *(f"vertebra_t{index}" for index in range(1, 13)),
    *(f"vertebra_l{index}" for index in range(1, 6)),
    "hips", "sacrum", "kidneys", "liver", "spleen", "lungs", "heart",
]
MEMBERS = ("flux0375", "flux000", "flux025", "flux050", "flux075")
CONTENT_BOX_SIZES = ((37, 53), (53, 37), (1, 100), (100, 1), (256, 256), (3, 2), (1000, 999))
# (input case, label, threshold) exercises source-grid mask export.
RESTORE_CASES = (
    ("l8", "lungs", 0.5),
    ("i16", "heart", 0.3),
    ("dicom_cr", "rib_5", 0.7),
    ("float_tiff", "skull", 0.5),
    ("bit", "liver", 0.0),
    ("rgba", "spleen", 1.0),
    ("exif", "background", 0.5),
    ("dicom_ct", "lungs", 0.5),
)
E2E_CASE = ("l8_e2e", "lungs", 0.5)
PROBABILITY_SEED = 31337
LOGIT_SEED = 4242


def _ramp(height: int, width: int, seed: int) -> DoubleArray:
    """A smooth radiograph-like ramp with seeded texture in [0, 1]."""

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    base = 0.5 + 0.35 * np.sin(yy / max(height, 2) * 3.1) * np.cos(xx / max(width, 2) * 2.3)
    return np.clip(base + rng.normal(0, 0.08, (height, width)), 0.0, 1.0)


def _write_dicom(path: Path, pixels: HostArray, *, modality: str, monochrome1: bool) -> None:
    """Write a synthetic DICOM input for runner preparation tests."""
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian

    meta = FileMetaDataset()
    meta.add_new(0x00020002, "UI", "1.2.840.10008.5.1.4.1.1.1.1")
    meta.add_new(0x00020003, "UI", "1.2.826.0.1.3680043.8.498.17")
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = Dataset()
    dataset.file_meta = meta
    dataset.SOPClassUID = meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.Modality = modality
    dataset.Rows, dataset.Columns = pixels.shape
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME1" if monochrome1 else "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 12
    dataset.HighBit = 11
    dataset.PixelRepresentation = 0
    dataset.RescaleSlope = "0.5"
    dataset.RescaleIntercept = "-20"
    dataset.PixelData = np.ascontiguousarray(pixels, dtype="<u2").tobytes()
    dataset.save_as(path, enforce_file_format=True)


def write_case(directory: Path, name: str) -> Path:
    """Write one named synthetic radiograph and return its path."""

    directory.mkdir(parents=True, exist_ok=True)
    if name == "l8":
        path = directory / "l8.png"
        Image.fromarray(np.rint(_ramp(37, 53, 1) * 255).astype(np.uint8), mode="L").save(path)
    elif name == "l8_e2e":
        path = directory / "l8_e2e.png"
        Image.fromarray(np.rint(_ramp(120, 90, 9) * 255).astype(np.uint8), mode="L").save(path)
    elif name == "exif":
        path = directory / "exif.png"
        image = Image.fromarray(np.rint(_ramp(29, 47, 2) * 255).astype(np.uint8), mode="L")
        exif = Image.Exif()
        exif[0x0112] = 6
        image.save(path, exif=exif)
    elif name == "i16":
        path = directory / "i16.png"
        values = np.rint(_ramp(41, 29, 3) * 60000).astype(np.uint16)
        Image.fromarray(values).save(path)
    elif name == "rgba":
        path = directory / "rgba.png"
        rng = np.random.default_rng(4)
        rgba = rng.integers(0, 256, (23, 31, 4), dtype=np.uint8)
        rgba[..., 3] = np.where(rng.random((23, 31)) < 0.3, 0, rgba[..., 3])
        Image.fromarray(rgba, mode="RGBA").save(path)
    elif name == "palette":
        path = directory / "palette.png"
        rng = np.random.default_rng(5)
        image = Image.fromarray(rng.integers(0, 16, (26, 18), dtype=np.uint8), mode="P")
        image.putpalette([value for index in range(16) for value in (index * 16, 255 - index, 7)])
        image.info["transparency"] = 3
        image.save(path, transparency=3)
    elif name == "bit":
        path = directory / "bit.png"
        Image.fromarray(_ramp(19, 27, 6) > 0.5).save(path)
    elif name == "float_tiff":
        path = directory / "float.tiff"
        Image.fromarray((_ramp(33, 21, 7) * 400 - 50).astype(np.float32), mode="F").save(path)
    elif name == "int32_tiff":
        path = directory / "int32.tiff"
        values = (_ramp(25, 36, 8) * 2_000_000 - 700_000).astype(np.int32)
        Image.fromarray(values, mode="I").save(path)
    elif name == "dicom_cr":
        path = directory / "cr.dcm"
        pixels = np.rint(_ramp(45, 38, 10) * 4095).astype(np.uint16)
        _write_dicom(path, pixels, modality="CR", monochrome1=True)
    elif name == "dicom_noext":
        path = directory / "dx_without_suffix"
        pixels = np.rint(_ramp(31, 44, 11) * 4000).astype(np.uint16)
        _write_dicom(path, pixels, modality="DX", monochrome1=False)
    elif name == "dicom_ct":
        path = directory / "ct.dcm"
        pixels = np.rint(_ramp(16, 16, 12) * 4000).astype(np.uint16)
        _write_dicom(path, pixels, modality="CT", monochrome1=False)
    else:
        raise KeyError(name)
    return path


CASES = (
    "l8", "exif", "i16", "rgba", "palette", "bit", "float_tiff", "int32_tiff", "dicom_cr",
    "dicom_noext", "dicom_ct",
)


def seeded_probabilities(seed: int = PROBABILITY_SEED) -> FloatArray:
    """Smooth per-channel probability maps on the 256 model grid."""

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:256, 0:256].astype(np.float32) / np.float32(255)
    phases = rng.uniform(0, 6.28, (61, 1, 1)).astype(np.float32)
    maps = np.float32(0.5) + np.float32(0.45) * np.sin(yy[None] * 7 + phases) * np.cos(
        xx[None] * 5 - phases
    )
    noise = rng.normal(0, 0.03, maps.shape).astype(np.float32)
    return np.clip(maps + noise, 0, 1).astype(np.float32)


def seeded_member_logits(members: int, seed: int = LOGIT_SEED) -> list[FloatArray]:
    """Small logits covering SLEEF expf's ordinary range and both saturation branches."""

    rng = np.random.default_rng(seed)
    logits = [rng.normal(0, 6, (1, 61, 8, 8)).astype(np.float32) for _ in range(members)]
    sweep = np.linspace(-130, 130, 61 * 8 * 8 // 4, dtype=np.float32)
    logits[0].reshape(-1)[: sweep.size] = sweep
    logits[0].reshape(-1)[-6:] = np.array([0.0, -0.0, 1e-30, -1e-30, 88.7, -103.9], np.float32)
    return logits


PERCENTILE_PAIRS = ((0.5, 99.5), (2.0, 97.5), (10.0, 90.0), (0.0, 100.0), (33.3, 66.6))


def seeded_percentile_batch(seed: int = 2) -> FloatArray:
    """Heavy-tailed float32 samples; with seed 2, fused and unfused percentile
    interpolation differ in the last bit for three bounds of the first two pairs."""

    rng = np.random.default_rng(seed)
    return (rng.standard_cauchy((16, 1, 13, 17)) * 3).astype(np.float32)


def write_release_bundle(
    root: Path,
    *,
    members: tuple[str, ...] = MEMBERS[:1],
    state_dict: Mapping[str, HostArray] | None = None,
    weight_seed: int = 1729,
) -> Path:
    """Write a release-layout snapshot: ensemble.json plus member bundles."""

    root.mkdir(parents=True, exist_ok=True)
    (root / "ensemble.json").write_text(
        json.dumps(
            {
                "aggregation": "mean_probability",
                "flagship": "members/flux0375",
                "members": [{"subfolder": f"members/{name}"} for name in MEMBERS],
            }
        ),
        encoding="utf-8",
    )
    for name in members:
        bundle = root / "members" / name
        write_bundle(bundle, state_dict, weight_seed=weight_seed)
        (bundle / "label_schema.json").write_text(
            json.dumps({"label_names": LABEL_NAMES, "num_labels": 61}), encoding="utf-8"
        )
        (bundle / "preprocessing.json").write_text(
            json.dumps(RELEASE_PREPROCESSING), encoding="utf-8"
        )
    return root
