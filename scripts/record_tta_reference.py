"""Prepare chest X-ray inputs for the pinned FleXray TTA recorder."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from _reference_cases import UPSTREAM_REVISION
from _tta_cases import TTA_CASES, WEIGHT_SEEDS, seeded_input, seeded_weights, standardize_inputs


def main() -> None:
    """Stage real image batches and regenerate pinned Torch references."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--torch-python", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("work/real-reference"))
    args = parser.parse_args()
    revision = subprocess.run(
        ["git", "-C", str(args.source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"Expected upstream {UPSTREAM_REVISION}, got {revision}")
    subprocess.run(
        ["git", "-C", str(args.source), "diff", "--exit-code", "HEAD", "--", "src/fxr"],
        check=True,
        capture_output=True,
    )
    fixtures = Path(__file__).resolve().parents[1] / "tests/fixtures/darwin-arm64"
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="flexray-tta-") as temporary:
        root = Path(temporary)
        inputs = {"image": seeded_input()}
        inputs.update(
            {f"standardize/{name}": value for name, value in standardize_inputs().items()}
        )
        for i, shape in enumerate(((1, 24, 40), (3, 13, 17), (1, 1, 1), (1, 3, 5))):
            inputs[f"clahe/{i}"] = seeded_input(shape)
        for seed in WEIGHT_SEEDS:
            inputs[f"weight/{seed}"], inputs[f"bias/{seed}"] = seeded_weights(seed)
        np.savez_compressed(root / "inputs.npz", **inputs)
        Image.fromarray((seeded_input((37, 53)) * np.float32(255)).astype(np.uint8)).save(
            root / "image.png"
        )
        config = {
            "source": str(args.source),
            "revision": revision,
            "output": str(args.output),
            "inputs": str(root / "inputs.npz"),
            "image": str(root / "image.png"),
            "cases": TTA_CASES,
            "seeds": WEIGHT_SEEDS,
        }
        (root / "config.json").write_text(json.dumps(config))
        subprocess.run(
            [
                str(args.torch_python),
                str(Path(__file__).with_name("record_tta_worker.py")),
                str(root / "config.json"),
            ],
            check=True,
            env=os.environ.copy(),
        )
    for name in ("transforms", "predictions", "runner_percentile_minmax", "runner_standardize"):
        path = fixtures / f"tta_{name}.npz"
        with np.load(path, allow_pickle=False) as archive:
            old = dict(archive)
        if name == "transforms":
            retained = {
                key: value
                for key, value in old.items()
                if key == "metadata"
                or key == "view_branch_counts"
                or key.startswith(("uniform/", "noise/", "noise_next/"))
            }
        else:
            retained = {"metadata": old["metadata"]}
        np.savez_compressed(path, **retained)


if __name__ == "__main__":
    main()
