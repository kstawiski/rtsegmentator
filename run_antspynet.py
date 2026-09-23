#!/usr/bin/env python3
"""Run selected single-volume ANTsPyNet brain models with an explicit NFS cache."""

from __future__ import annotations

import argparse
from pathlib import Path

import ants
import antspynet
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    args = parser.parse_args()

    args.cache.mkdir(parents=True, exist_ok=True)
    antspynet.set_antsxnet_cache_directory(str(args.cache))
    image = ants.image_read(str(args.input), pixeltype="float")

    if args.task == "brain_extraction":
        result = antspynet.brain_extraction(image, modality="t1", verbose=True)
        if isinstance(result, dict):
            result = result["segmentation_image"]
        segmentation = ants.threshold_image(result, 0.5, 1.0, 1, 0)
    elif args.task == "deep_atropos":
        segmentation = antspynet.deep_atropos(image, verbose=True)["segmentation_image"]
    elif args.task == "hippmapp3r":
        result = antspynet.hippmapp3r_segmentation(image, verbose=True)
        # Upstream returns a binary mask in current ANTsPyNet releases.
        segmentation = ants.threshold_image(result, 0.5, np.inf, 1, 0)
    elif args.task == "hypothalamus":
        segmentation = antspynet.hypothalamus_segmentation(image, verbose=True)["segmentation_image"]
    elif args.task == "dkt":
        result = antspynet.desikan_killiany_tourville_labeling(image, verbose=True)
        segmentation = result["segmentation_image"] if isinstance(result, dict) else result
    elif args.task == "deep_flash":
        segmentation = antspynet.deep_flash(image, verbose=True)["segmentation_image"]
    elif args.task == "cerebellum":
        segmentation = antspynet.cerebellum_morphology(image, verbose=True)["parcellation_segmentation_image"]
    else:
        raise SystemExit(f"Unsupported ANTsPyNet task: {args.task}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ants.image_write(segmentation, str(args.output))


if __name__ == "__main__":
    main()
