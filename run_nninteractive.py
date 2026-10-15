"""Run the official nnInteractive checkpoint from portal point/box prompts."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch


def integer_point(value, shape):
    if not isinstance(value, list) or len(value) != 3:
        raise SystemExit("nnInteractive points must be [column,row,slice]")
    point = [int(round(float(item))) for item in value]
    if any(point[axis] < 0 or point[axis] >= shape[axis] for axis in range(3)):
        raise SystemExit(f"nnInteractive point {point} is outside image shape {shape}")
    return point


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--site-packages", required=True)
    args = parser.parse_args()

    # The source tree is authoritative; the small NFS site directory supplies
    # package metadata required by importlib.metadata without duplicating torch.
    sys.path[:0] = [args.site_packages, args.source]
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

    image = nib.load(args.input)
    data = np.asanyarray(image.dataobj)
    if data.ndim != 3 or not np.any(np.isfinite(data) & (data != 0)):
        raise SystemExit(f"nnInteractive requires a non-empty 3-D image, got {data.shape}")
    prompt = json.loads(Path(args.prompt).read_text())
    positive = [integer_point(item, data.shape) for item in prompt.get("positive_points", [])]
    negative = [integer_point(item, data.shape) for item in prompt.get("negative_points", [])]
    box = prompt.get("box")
    interactions = []
    if box:
        if not isinstance(box, list) or len(box) != 2:
            raise SystemExit("nnInteractive box requires two [column,row,slice] corners")
        corners = [integer_point(item, data.shape) for item in box]
        bbox = [[min(corners[0][axis], corners[1][axis]),
                 max(corners[0][axis], corners[1][axis]) + 1] for axis in range(3)]
        if sum(end - start == 1 for start, end in bbox) != 1:
            raise SystemExit("The official nnInteractive checkpoint requires a 2-D box on one slice")
        interactions.append(("box", bbox, True))
    interactions.extend(("point", point, True) for point in positive)
    interactions.extend(("point", point, False) for point in negative)
    if not interactions or not (box or positive):
        raise SystemExit("nnInteractive requires a positive point or a 2-D box")

    session = nnInteractiveInferenceSession(
        device=torch.device("cuda:0"), use_torch_compile=False, verbose=False,
        torch_n_threads=min(8, os.cpu_count() or 1), do_autozoom=True,
    )
    session.initialize_from_trained_model_folder(args.model_dir, use_fold=0)
    # nibabel exposes [column,row,slice], matching the portal prompt contract.
    session.set_image(data[None])
    target = torch.zeros(data.shape, dtype=torch.uint8)
    session.set_target_buffer(target)
    for index, (kind, coordinates, include) in enumerate(interactions):
        predict = index == len(interactions) - 1
        if kind == "box":
            session.add_bbox_interaction(coordinates, include_interaction=include,
                                         run_prediction=predict)
        else:
            session.add_point_interaction(coordinates, include_interaction=include,
                                          run_prediction=predict)
    result = target.numpy().astype(np.uint8, copy=False)
    if not np.any(result):
        raise SystemExit("nnInteractive produced an empty mask")
    header = image.header.copy()
    header.set_data_dtype(np.uint8)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(result, image.affine, header), output)
    print({"voxels": int(result.sum()), "license": session.license,
           "interactions": len(interactions)})


if __name__ == "__main__":
    main()
