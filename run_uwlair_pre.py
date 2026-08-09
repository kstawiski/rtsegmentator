"""Run the published UWLAIR HNTS-MRG pre-RT T2 ensemble on one volume."""

from pathlib import Path
import os
import sys

import SimpleITK as sitk


model_root = Path(os.environ.get("RTSEG_MODEL_ROOT", "/home/konrad/lymph-models"))
source = model_root / "sources/HNTS-MRG24-UWLAIR/inference/Task1_preRT"
weights = model_root / "weights/hntsmrg-uwlair/extracted/preRT_models"
sys.path.insert(0, str(source))

from segmenter import inference  # noqa: E402


input_nifti = Path(sys.argv[1])
output_nifti = Path(sys.argv[2])
work = output_nifti.parent / "uwlair-mha"
input_dir = work / "input"
output_dir = work / "output"
input_dir.mkdir(parents=True, exist_ok=True)
output_dir.mkdir(parents=True, exist_ok=True)
sitk.WriteImage(sitk.ReadImage(str(input_nifti)), str(input_dir / "case.mha"), True)

runner = inference(roi_size=[192, 192, 128], modality="mri")
runner.infer_image(
    ckpt_path_list=[weights / f"model_{index}.pt" for index in range(10)],
    image_location=input_dir,
    mask_location=output_dir,
)

result = output_dir / "output.mha"
if not result.exists():
    raise SystemExit("UWLAIR pre-RT inference did not create output.mha")
sitk.WriteImage(sitk.ReadImage(str(result)), str(output_nifti), True)
