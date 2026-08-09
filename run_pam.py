"""Run PAM's published slice-to-volume propagation with a WebUI seed prompt."""

import argparse
import ast
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from scipy.ndimage import label as scipy_label
import SimpleITK as sitk
import torch
import torch.nn.functional as F


def load_published_functions(notebook: Path) -> None:
    """Load only function definitions from the authoritative PAM tutorial."""
    document = json.loads(notebook.read_text())
    definitions = []
    for cell in document["cells"]:
        if cell.get("cell_type") != "code":
            continue
        try:
            tree = ast.parse("".join(cell.get("source", [])))
        except SyntaxError:
            continue
        definitions.extend(node for node in tree.body if isinstance(node, ast.FunctionDef))
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(notebook), "exec"), globals())


parser = argparse.ArgumentParser()
parser.add_argument("--input", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--prompt", required=True)
parser.add_argument("--modality", required=True, choices=["CT", "MR", "PT"])
args = parser.parse_args()

model_root = Path(os.environ.get("RTSEG_MODEL_ROOT", "/home/konrad/lymph-models"))
repo = model_root / "sources/PAM"
sys.path.insert(0, str(repo))
from model.PAM import PAM  # noqa: E402

PAM_IMG_SIZE = 224
MIN_SLICE_AREA = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = A.Compose([
    A.Resize(width=PAM_IMG_SIZE, height=PAM_IMG_SIZE),
    A.Normalize(mean=[0.5], std=[0.5]),
    ToTensorV2(),
])
load_published_functions(repo / "tutorials/3d-propagation.ipynb")

image = sitk.ReadImage(args.input)
array = sitk.GetArrayFromImage(image).astype(np.float32)
prompt = json.loads(Path(args.prompt).read_text())
seed = np.zeros(array.shape, dtype=np.uint8)
positive = prompt.get("positive_points", [])
box = prompt.get("box")
if box:
    (x1,y1,z1),(x2,y2,z2) = box
    if int(z1) != int(z2): raise SystemExit("PAM box corners must be on one slice")
    xa,xb = sorted((int(x1),int(x2))); ya,yb = sorted((int(y1),int(y2)))
    seed[int(z1),ya:yb+1,xa:xb+1] = 1
for x,y,z in positive:
    x,y,z = int(x),int(y),int(z)
    yy,xx = np.ogrid[:seed.shape[1],:seed.shape[2]]
    seed[z,(xx-x)**2+(yy-y)**2 <= 25] = 1
guiding = np.where(seed.reshape(seed.shape[0],-1).sum(axis=1)>0)[0]
if len(guiding) != 1:
    raise SystemExit("PAM requires positive seed pixels on exactly one axial slice")

model = PAM(conv_dim=2,input_channels=1,n_stages=6,max_channels=512,
            num_classes=2,deep_supervision=True,n_attn_stage=4,from_scratch_ratio=0.0)
model = load_checkpoint(model, str(model_root / "weights/pam/propmask.pth"))
model = model.to(DEVICE).eval()
_, prediction = infer_one_sample(
    args=SimpleNamespace(), modality="CT" if args.modality=="CT" else "MRI",
    box2seg=None, pam=model, img=image, img_array=array, spacing=image.GetSpacing(),
    guiding_mask_array=seed, guiding_z=int(guiding[0]), save_file=args.output,
    target_size=PAM_IMG_SIZE, dynamic_crop_ratio=1.5, neighbor_spacing_z=20,
    max_Z=40, device=DEVICE, printer=print)
if prediction is None or not np.any(prediction):
    raise SystemExit("PAM produced an empty propagated mask")

# The published tutorial only writes voxel spacing. Restore the complete
# physical frame so DICOM contour coordinates remain tied to the source series.
prediction_image = sitk.ReadImage(args.output)
prediction_image.CopyInformation(image)
sitk.WriteImage(prediction_image, args.output)
