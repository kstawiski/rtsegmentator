"""Combine native-space left/right HippUnfold subfields into one RT label map."""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    reference = nib.load(args.reference)
    hemispheres = [nib.load(args.left), nib.load(args.right)]
    for name, image in zip(("left", "right"), hemispheres, strict=True):
        if image.shape != reference.shape:
            raise SystemExit(f"{name} shape {image.shape} does not match reference {reference.shape}")
        if not np.allclose(image.affine, reference.affine, atol=1e-3):
            raise SystemExit(f"{name} affine does not match reference")

    left = np.rint(np.asanyarray(hemispheres[0].dataobj)).astype(np.uint8)
    right = np.rint(np.asanyarray(hemispheres[1].dataobj)).astype(np.uint8)
    for name, data in (("left", left), ("right", right)):
        unexpected = set(np.unique(data).tolist()) - set(range(9))
        if unexpected:
            raise SystemExit(f"unexpected {name} HippUnfold labels: {sorted(unexpected)}")
    overlap = np.count_nonzero((left > 0) & (right > 0))
    if overlap:
        raise SystemExit(f"left/right HippUnfold masks overlap in {overlap} voxels")

    combined = left.astype(np.uint16)
    right_mask = right > 0
    combined[right_mask] = right[right_mask].astype(np.uint16) + 8
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy()
    header.set_data_dtype(np.uint16)
    nib.save(nib.Nifti1Image(combined, reference.affine, header), output)
    print({"labels": sorted(set(np.unique(combined).tolist()) - {0}), "overlap": overlap})


if __name__ == "__main__":
    main()
