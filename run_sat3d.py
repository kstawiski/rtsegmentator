"""Prompt-driven SAT3D inference on a 128-cubed ROI around WebUI points."""

import argparse
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--prompt", required=True)
args = parser.parse_args()

model_root = Path(os.environ.get("RTSEG_MODEL_ROOT", "/home/konrad/lymph-models"))
repo = model_root / "sources/SAT3D"
sys.path.insert(0, str(repo))
# SAT3D only needs two small timm layer helpers. Supplying them locally avoids
# timm's unrelated eager torchvision/NMS import, incompatible with this worker.
class DropPath(torch.nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__(); self.drop_prob = drop_prob
    def forward(self, value):
        if self.drop_prob == 0.0 or not self.training: return value
        keep = 1 - self.drop_prob
        shape = (value.shape[0],) + (1,) * (value.ndim - 1)
        return value.div(keep) * (keep + torch.rand(shape, device=value.device, dtype=value.dtype)).floor()
timm = types.ModuleType("timm")
timm_layers = types.ModuleType("timm.layers")
timm_layers.DropPath = DropPath
timm_layers.trunc_normal_ = torch.nn.init.trunc_normal_
timm.layers = timm_layers
sys.modules["timm"] = timm
sys.modules["timm.layers"] = timm_layers
from networks import Discriminator  # noqa: E402
sat_package = types.ModuleType("segment_anything_with_swin_conf")
sat_package.__path__ = [str(repo / "segment_anything_with_swin_conf")]
sys.modules["segment_anything_with_swin_conf"] = sat_package
builder_path = repo / "segment_anything_with_swin_conf/build_samswin3D.py"
builder_source = builder_path.read_text()
# Repair two upstream aliases that refer to a nonexistent build_sam3D_swin;
# the implemented constructor in this release is build_sam3D_swin2.
builder_source = builder_source.replace("build_sam3D = build_sam3D_swin\n", "build_sam3D = build_sam3D_swin2\n")
builder_source = builder_source.replace('"default": build_sam3D_swin,', '"default": build_sam3D_swin2,')
builder_source = builder_source.replace('"swin_c": build_sam3D_swin,', '"swin_c": build_sam3D_swin2,')
builder = types.ModuleType("segment_anything_with_swin_conf.build_samswin3D")
builder.__package__ = "segment_anything_with_swin_conf"
sys.modules[builder.__name__] = builder
exec(compile(builder_source, str(builder_path), "exec"), builder.__dict__)
sam_model_registry3D = builder.sam_model_registry3D

image = sitk.ReadImage(args.input)
zyx = sitk.GetArrayFromImage(image).astype(np.float32)
xyz = np.transpose(zyx, (2, 1, 0))
prompt = json.loads(Path(args.prompt).read_text())
points = [(p, 1) for p in prompt.get("positive_points", [])]
points += [(p, 0) for p in prompt.get("negative_points", [])]
if not any(label == 1 for _, label in points):
    raise SystemExit("SAT3D requires at least one positive point")

anchor = np.mean([p for p, label in points if label == 1], axis=0)
size = 128
starts = [int(round(anchor[d])) - size // 2 for d in range(3)]
ends = [s + size for s in starts]
crop = np.zeros((size, size, size), dtype=np.float32)
source_slices = []
target_slices = []
for dim, extent in enumerate(xyz.shape):
    source_start, source_end = max(0, starts[dim]), min(extent, ends[dim])
    target_start = source_start - starts[dim]
    source_slices.append(slice(source_start, source_end))
    target_slices.append(slice(target_start, target_start + source_end-source_start))
crop[tuple(target_slices)] = xyz[tuple(source_slices)]
mask = crop > 0
if np.any(mask):
    crop[mask] = (crop[mask] - crop[mask].mean()) / max(crop[mask].std(), 1e-6)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = sam_model_registry3D["swin_c"](checkpoint=None).to(device).eval()
critic = Discriminator().to(device).eval()
load_kwargs = {"map_location": device, "weights_only": False}
model_state = torch.load(model_root / "weights/sat3d/sam_model_dice_best.pth", **load_kwargs)
critic_state = torch.load(model_root / "weights/sat3d/critic_dice_best.pth", **load_kwargs)
def clean(state):
    state = state["model_state_dict"]
    return {key.removeprefix("module."): value for key, value in state.items()}
model.load_state_dict(clean(model_state), strict=False)
critic.load_state_dict(clean(critic_state), strict=False)

tensor = torch.from_numpy(crop)[None, None].to(device)
coordinates = []
labels = []
for point, label in points:
    local = [float(point[d]) - starts[d] for d in range(3)]
    if all(0 <= local[d] < size for d in range(3)):
        coordinates.append(local); labels.append(label)
if not coordinates:
    raise SystemExit("No supplied SAT3D points fall inside the prompt-centred ROI")
coordinates = torch.tensor(coordinates, dtype=torch.float32, device=device)[None]
labels = torch.tensor(labels, dtype=torch.int64, device=device)[None]

with torch.no_grad():
    embedding = model.image_encoder(tensor)
    previous = torch.zeros((1, 1, size, size, size), device=device)
    low_previous = F.interpolate(previous, size=(32, 32, 32), mode="trilinear", align_corners=False)
    confidence = (torch.sigmoid(critic(torch.sigmoid(previous))) > 0.5).float()
    low_confidence = F.interpolate(confidence, size=(32, 32, 32), mode="trilinear", align_corners=False)
    sparse, dense = model.prompt_encoder(points=[coordinates, labels], boxes=None,
        masks=low_previous, conf=low_confidence)
    low_mask, _ = model.mask_decoder(image_embeddings=embedding,
        image_pe=model.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense, multimask_output=False)
    prediction = (torch.sigmoid(F.interpolate(low_mask, size=(size, size, size),
        mode="trilinear", align_corners=False)) > 0.5).cpu().numpy()[0, 0].astype(np.uint8)

full = np.zeros(xyz.shape, dtype=np.uint8)
full[tuple(source_slices)] = prediction[tuple(target_slices)]
if not np.any(full):
    raise SystemExit("SAT3D produced an empty mask")
output = sitk.GetImageFromArray(np.transpose(full, (2, 1, 0)))
output.CopyInformation(image)
sitk.WriteImage(output, args.output)
