"""Run the official FastSurferVINN three-view segmentation pipeline."""
import argparse
import os
import subprocess
from pathlib import Path

import nibabel as nib

p = argparse.ArgumentParser()
p.add_argument("--input", required=True)
p.add_argument("--output", required=True)
p.add_argument("--source", required=True)
p.add_argument("--runtime", required=True)
p.add_argument("--overlay", required=True)
p.add_argument("--work", required=True)
a = p.parse_args()

source, runtime, overlay, work = map(Path, (a.source, a.runtime, a.overlay, a.work))
subjects = work / "subjects"
env = os.environ.copy()
env["PYTHONPATH"] = os.pathsep.join((str(overlay), str(source), env.get("PYTHONPATH", "")))
command = [
    str(runtime / "bin/python"), str(source / "FastSurferCNN/run_prediction.py"),
    "--t1", a.input, "--sd", str(subjects), "--sid", "case",
    "--device", "cuda", "--viewagg_device", "cuda", "--batch_size", "8",
    "--orientation", "native", "--vox_size", "any", "--threads", "8",
    "--ckpt_ax", str(source / "checkpoints/aparc_vinn_axial_v2.0.0.pkl"),
    "--ckpt_cor", str(source / "checkpoints/aparc_vinn_coronal_v2.0.0.pkl"),
    "--ckpt_sag", str(source / "checkpoints/aparc_vinn_sagittal_v2.0.0.pkl"),
]
subprocess.run(command, env=env, cwd=source, check=True)
prediction = subjects / "case/mri/aparc.DKTatlas+aseg.deep.mgz"
if not prediction.is_file():
    raise SystemExit(f"FastSurfer did not create {prediction}")
# SimpleITK does not ship MGZ ImageIO in the portal environment. Nibabel
# preserves FastSurfer's affine and discrete values while writing portable NIfTI.
image = nib.load(prediction)
nib.save(nib.Nifti1Image(image.get_fdata(dtype="float32"), image.affine), a.output)
