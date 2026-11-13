"""Prepare real X-ray inputs and invoke pinned FleXray's Torch CPU recorder."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from _reference_cases import MODEL_CONFIG, UPSTREAM_REVISION, write_bundle
from _runner_cases import (
    CASES,
    PERCENTILE_PAIRS,
    PREPROCESSING_VARIANTS,
    RELEASE_PREPROCESSING,
    seeded_percentile_batch,
    write_case,
)


def main() -> None:
    """Generate all file-format cases and retain image-bearing outputs outside Git."""
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
    with tempfile.TemporaryDirectory(prefix="flexray-real-reference-") as temp:
        root = Path(temp)
        paths = {case: str(write_case(root / "inputs" / case, case)) for case in (*CASES, "l8_e2e")}
        write_bundle(root / "weights", weight_seed=1729)
        np.save(root / "percentiles.npy", seeded_percentile_batch())
        config = {
            "paths": paths,
            "weights": str(root / "weights/model.safetensors"),
            "source": str(args.source),
            "output": str(args.output),
            "revision": revision,
            "model": MODEL_CONFIG,
            "release": RELEASE_PREPROCESSING,
            "variants": PREPROCESSING_VARIANTS,
            "percentile_input": str(root / "percentiles.npy"),
            "percentile_pairs": PERCENTILE_PAIRS,
        }
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config))
        subprocess.run(
            [
                str(args.torch_python),
                str(Path(__file__).with_name("record_runner_worker.py")),
                str(config_path),
            ],
            check=True,
            env=os.environ.copy(),
        )
    for name in ("runner_reference", "runner_preprocessing", "runner_run", "flexray_full"):
        path = fixtures / f"{name}.npz"
        with np.load(path, allow_pickle=False) as archive:
            old = dict(archive)
        if name == "runner_reference":
            keep = {
                k: v
                for k, v in old.items()
                if not k.startswith(("display/", "prepared/", "percentile_minmax/"))
                and not (k.startswith("restore/") and k.endswith("/display"))
            }
        else:
            keep = {k: v for k, v in old.items() if k in {"metadata", "weight_seed", "input_seed"}}
        np.savez_compressed(path, **keep)


if __name__ == "__main__":
    main()
