#!/usr/bin/env python3
"""Run VIBESegmentator with an explicit offline model root."""

import argparse
import json
import os
from pathlib import Path

from TPTBox.segmentation.VibeSeg.vibeseg import VibeSeg_map, run_vibeseg


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--labels", required=True)
parser.add_argument("--model-root", required=True)
args = parser.parse_args()

model_root = Path(args.model_root)
os.environ["VIBESEG_WEIGHTS_PATH"] = str(model_root / "nnUNet_results")
run_vibeseg(
    args.input,
    args.output,
    override=True,
    ddevice="cuda",
    dataset_id=100,
    model_path=model_root,
    verbose=False,
)
Path(args.labels).write_text(
    json.dumps({str(key): value for key, value in VibeSeg_map.items()}, indent=2, sort_keys=True) + "\n"
)
