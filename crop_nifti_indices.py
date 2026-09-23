#!/usr/bin/env python3
"""Crop a NIfTI by voxel-index bounds while preserving world coordinates."""

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("bounds", nargs=6, type=int, metavar=("X0", "X1", "Y0", "Y1", "Z0", "Z1"))
    args = parser.parse_args()
    image = nib.load(args.input)
    x0, x1, y0, y1, z0, z1 = args.bounds
    slices = (slice(x0, x1), slice(y0, y1), slice(z0, z1))
    data = np.asanyarray(image.dataobj)[slices]
    if not all(v > 0 for v in data.shape):
        raise SystemExit(f"empty crop {data.shape}")
    affine = image.affine @ nib.affines.from_matvec(np.eye(3), (x0, y0, z0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(data, affine, image.header), args.output)
    print({"input": image.shape, "output": data.shape, "bounds": args.bounds})


if __name__ == "__main__":
    main()
