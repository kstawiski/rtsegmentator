"""Run SAM-Med2D on every axial slice that contains a WebUI prompt."""

import argparse
from collections import defaultdict
import importlib
import json
import os
import sys
import types
from argparse import Namespace
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--prompt", required=True)
args = parser.parse_args()

model_root = Path(os.environ.get("RTSEG_MODEL_ROOT", "/models"))
repo = model_root / "sources/SAM-Med2D"
# Avoid the upstream package initializer: it eagerly imports the automatic
# mask generator and therefore torchvision NMS, which prompted inference does
# not use and which is unavailable in the worker's current Torch build.
package = types.ModuleType("segment_anything")
package.__path__ = [str(repo / "segment_anything")]
sys.modules["segment_anything"] = package
sam_model_registry = importlib.import_module("segment_anything.build_sam").sam_model_registry
from segment_anything.predictor_sammed import SammedPredictor  # noqa: E402

image = sitk.ReadImage(args.input)
volume = sitk.GetArrayFromImage(image).astype(np.float32)
prompt = json.loads(Path(args.prompt).read_text())
positive = prompt.get("positive_points", [])
negative = prompt.get("negative_points", [])
box = prompt.get("box")
slices = {int(p[2]) for p in positive + negative}
if box:
    if int(box[0][2]) != int(box[1][2]):
        raise SystemExit("SAM-Med2D box corners must be on one slice")
    slices.add(int(box[0][2]))
if not slices:
    raise SystemExit("SAM-Med2D requires at least one point or box")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# The official checkpoint includes the training optimizer alongside model
# weights. PyTorch 2.6+ requires this known upstream class to be allowlisted.
torch.serialization.add_safe_globals([torch.optim.Adam, defaultdict, dict])
_torch_load = torch.load
def trusted_checkpoint_load(*load_args, **load_kwargs):
    load_kwargs["weights_only"] = False
    return _torch_load(*load_args, **load_kwargs)
torch.load = trusted_checkpoint_load
model_args = Namespace(image_size=256, encoder_adapter=True,
    sam_checkpoint=str(model_root / "weights/sam-med2d/sam-med2d_b.pth"))
predictor = SammedPredictor(sam_model_registry["vit_b"](model_args).to(device).eval())
result = np.zeros(volume.shape, dtype=np.uint8)
for z in sorted(slices):
    frame = volume[z]
    finite = frame[np.isfinite(frame)]
    low, high = np.percentile(finite, [1, 99]) if finite.size else (0, 1)
    if high <= low: high = low + 1
    gray = np.clip((frame-low)/(high-low)*255, 0, 255).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    predictor.set_image(rgb)
    points = [(p, 1) for p in positive if int(p[2]) == z] + [(p, 0) for p in negative if int(p[2]) == z]
    coords = np.asarray([[p[0], p[1]] for p, _ in points], dtype=np.float32) if points else None
    labels = np.asarray([label for _, label in points], dtype=np.int32) if points else None
    slice_box = None
    if box and int(box[0][2]) == z:
        slice_box = np.asarray([box[0][0], box[0][1], box[1][0], box[1][1]], dtype=np.float32)
    masks, scores, _ = predictor.predict(point_coords=coords, point_labels=labels,
        box=slice_box, multimask_output=True)
    result[z] = masks[int(np.argmax(scores))].astype(np.uint8)

if not np.any(result):
    raise SystemExit("SAM-Med2D produced an empty mask")
output = sitk.GetImageFromArray(result)
output.CopyInformation(image)
sitk.WriteImage(output, args.output)
