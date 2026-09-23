#!/usr/bin/env python3
"""Run the publisher-supplied PET-only DOTATATE ONNX ensemble."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torchio as tio


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()

    source = Path(args.source)
    sys.path.insert(0, str(source))
    spec = importlib.util.spec_from_file_location("dotatate_publisher_inference", source / "main.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load DOTATATE publisher inference source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    image = tio.ScalarImage(args.input)
    normalized = tio.RescaleIntensity(out_min_max=(0, 1))(image)
    # Publisher checkpoint convention is (channel, z, y, x).
    model_input = np.transpose(normalized.data.numpy(), axes=[0, 3, 2, 1])
    prediction = module.run_onnx_model(model_input, args.model)
    output = nib.Nifti1Image(prediction.astype(np.uint8), image.affine)
    nib.save(output, args.output)


if __name__ == "__main__":
    main()
