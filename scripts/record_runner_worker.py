"""Run the pinned FleXray preprocessing and UNet in its Torch environment."""

import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from PIL import __version__ as pillow_version
from safetensors.torch import load_file


def main():
    """Write independent outputs for the supplied real radiograph cases."""
    config = json.loads(Path(sys.argv[1]).read_text())
    source = Path(config["source"]) / "src/fxr"
    for name, directory in (
        ("fxr", source),
        ("fxr.inference", source / "inference"),
        ("fxr.models", source / "models"),
    ):
        module = types.ModuleType(name)
        module.__path__ = [str(directory)]
        sys.modules[name] = module
    prep = importlib.import_module("fxr.inference.preprocessing")
    dicom = importlib.import_module("fxr.inference.dicom")
    unet = importlib.import_module("fxr.models.unet")
    torch.set_num_threads(1)
    if torch.__version__.split("+")[0] != "2.6.0":
        raise RuntimeError(f"Expected Torch 2.6.0, got {torch.__version__}")
    output = Path(config["output"])
    metadata = np.asarray(
        json.dumps(
            {
                "flexray_revision": config["revision"],
                "source": "Wikimedia Commons chest radiograph",
                "torch_version": torch.__version__,
                "numpy_version": np.__version__,
                "pillow_version": pillow_version,
                "reference_backend": "torch-cpu",
            }
        )
    )
    display = {}
    for case, path in config["paths"].items():
        image = prep._load_image_file(path)
        if isinstance(image, Image.Image):
            image = ImageOps.exif_transpose(image)
            if (
                image.mode in {"RGBA", "LA", "PA"}
                or "transparency" in image.info
                or (image.palette is not None and image.palette.mode == "RGBA")
            ):
                image = Image.alpha_composite(
                    Image.new("RGBA", image.size, (0, 0, 0, 255)), image.convert("RGBA")
                )
            grayscale = image if image.mode in {"1", "L", "I", "I;16", "F"} else image.convert("L")
            pixels = np.array(grayscale, copy=True)
        else:
            pixels = image
        values = prep._scale_numpy_integer_range(pixels, source_dtype=pixels.dtype)
        values = prep._check_pixel_values(torch.from_numpy(values)[None, None])[0, 0].numpy()
        if np.issubdtype(pixels.dtype, np.floating) and (values.min() < 0 or values.max() > 1):
            values = dicom.scale_to_unit_range(values)
        display[case] = np.rint(values * 255).astype(np.uint8)
    release = prep.ImagePreprocessing.from_metadata(config["release"])
    reference = {f"display/{case}": image for case, image in display.items() if case != "l8_e2e"}
    reference["real_metadata"] = metadata
    reference["prepared/l8/release"] = release.prepare(config["paths"]["l8"]).numpy()
    percentile_input = torch.from_numpy(np.load(config["percentile_input"], allow_pickle=False))
    for index, pair in enumerate(config["percentile_pairs"]):
        normalizer = prep.ImagePreprocessing(
            normalization_scheme="percentile_minmax", percentiles=tuple(pair)
        )
        reference[f"percentile_minmax/{index}"] = normalizer._normalize_tensor(
            percentile_input
        ).numpy()
    for case in ("l8", "i16", "dicom_cr", "float_tiff", "bit", "rgba", "exif", "dicom_ct"):
        reference[f"restore/{case}/display"] = display[case]
    np.savez_compressed(output / "runner_reference.npz", **reference)
    variants = {"real_metadata": metadata}
    for case, path in config["paths"].items():
        if case == "l8_e2e":
            continue
        for variant, settings in config["variants"].items():
            variants[f"prepared/{case}/{variant}"] = (
                prep.ImagePreprocessing.from_metadata(settings).prepare(path).numpy()
            )
    np.savez_compressed(output / "runner_preprocessing.npz", **variants)
    model_config = dict(config["model"])
    model_config.pop("_class")
    model = unet.UNet(**model_config).cpu().eval()
    model.load_state_dict(load_file(config["weights"], device="cpu"), strict=True)
    prepared = release.prepare(config["paths"]["l8_e2e"])
    with torch.inference_mode():
        logits = model(prepared)
        probabilities = logits.sigmoid()
    channel = probabilities[0, 0].numpy()
    box = release.content_box(120, 90)
    np.savez_compressed(
        output / "runner_run.npz",
        prepared=prepared.numpy(),
        logits_channel=logits[0, 0].numpy(),
        probabilities_channel=channel,
        display=display["l8_e2e"],
        content_box=np.asarray(box),
        real_metadata=metadata,
    )
    np.savez_compressed(
        output / "flexray_full.npz",
        images=prepared.numpy(),
        output_nhwc=logits.permute(0, 2, 3, 1).numpy(),
        real_metadata=metadata,
    )


if __name__ == "__main__":
    main()
