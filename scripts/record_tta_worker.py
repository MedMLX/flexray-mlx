"""Run pinned FleXray TTA and normalization on supplied X-ray inputs."""

import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from PIL import __version__ as pillow_version
from torch.nn import functional as F


def main():
    """Evaluate source augmentations and ensembles in the pinned Torch runtime."""
    config = json.loads(Path(sys.argv[1]).read_text())
    source = Path(config["source"]) / "src/fxr"
    for name, directory in (("fxr", source), ("fxr.inference", source / "inference")):
        module = types.ModuleType(name)
        module.__path__ = [str(directory)]
        sys.modules[name] = module
    tta = importlib.import_module("fxr.inference.tta")
    prep = importlib.import_module("fxr.inference.preprocessing")
    if torch.__version__.split("+")[0] != "2.6.0":
        raise RuntimeError(f"Expected Torch 2.6.0, got {torch.__version__}")
    torch.set_num_threads(1)
    with np.load(config["inputs"], allow_pickle=False) as archive:
        inputs = dict(archive)
    output = Path(config["output"])
    meta = np.asarray(
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
    transforms = {"real_metadata": meta}
    torch.manual_seed(0)
    for index in range(16):
        view, flip = tta._augment_batch(torch.from_numpy(inputs["image"]))
        transforms[f"view/{index}"], transforms[f"flip/{index}"] = view.numpy(), flip.numpy()
    for index in range(4):
        transforms[f"clahe/{index}"] = tta._clahe(
            torch.from_numpy(inputs[f"clahe/{index}"]), clip_limit=1.4, grid_size=(8, 8)
        ).numpy()
    for key, value in inputs.items():
        if key.startswith("standardize/"):
            eps = 1e-3 if key.endswith("near_constant") else 1e-8
            normalizer = prep.ImagePreprocessing(normalization_scheme="standardize", eps=eps)
            transforms[key] = normalizer._normalize_tensor(torch.from_numpy(value)).numpy()
    np.savez_compressed(output / "tta_transforms.npz", **transforms)

    def forward(seed):
        weight = torch.from_numpy(inputs[f"weight/{seed}"])
        bias = torch.from_numpy(inputs[f"bias/{seed}"])
        return lambda image: F.conv2d(image, weight, bias, padding=1).repeat(1, 61, 1, 1)

    predictions = {"real_metadata": meta}
    for name, seed, samples, members, scheme in config["cases"]:
        normalizer = prep.ImagePreprocessing(
            normalization_scheme=scheme,
            percentiles=(0.5, 99.5) if scheme == "percentile_minmax" else None,
        )
        image = normalizer._normalize_tensor(torch.from_numpy(inputs["image"]))
        logits, probabilities = tta._predict_with_tta(
            [forward(s) for s in config["seeds"][:members]],
            image,
            mode="multilabel",
            tta_samples=samples,
            seed=seed,
            normalizer=normalizer._normalize_tensor,
        )
        predictions[f"logits/{name}"] = logits[:, :1].numpy()
        predictions[f"probabilities/{name}"] = probabilities[:, :1].numpy()
    np.savez_compressed(output / "tta_predictions.npz", **predictions)
    for scheme in ("percentile_minmax", "standardize"):
        normalizer = prep.ImagePreprocessing(
            normalization_scheme=scheme,
            percentiles=(0.5, 99.5) if scheme == "percentile_minmax" else None,
        )
        prepared = normalizer.prepare(config["image"])
        _, probabilities = tta._predict_with_tta(
            [forward(s) for s in config["seeds"]],
            prepared,
            mode="multilabel",
            tta_samples=2,
            seed=0,
            normalizer=normalizer._normalize_tensor,
        )
        np.savez_compressed(
            output / f"tta_runner_{scheme}.npz",
            real_metadata=meta,
            **{
                f"runner/{scheme}/prepared": prepared.numpy(),
                f"runner/{scheme}/probabilities": probabilities[0, 0].numpy(),
                f"runner/{scheme}/content_box": np.asarray(normalizer.content_box(37, 53)),
            },
        )


if __name__ == "__main__":
    main()
